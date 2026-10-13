#include "RhoCPFFTTensorProblem.h"

#include "DDCPFFTForwardSolver.h"
#include "DomainAction.h"
#include "MarlinUtils.h"
#include "MooseError.h"

#include <ATen/ATen.h>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>

registerMooseObject("RhocpFFTApp", RhoCPFFTTensorProblem);

InputParameters
RhoCPFFTTensorProblem::validParams()
{
  InputParameters params = TensorProblem::validParams();
  params.addClassDescription(
      "Marlin TensorProblem with persistent per-rank checkpoint/recovery of committed DDCP and "
      "initialization tensors. Requires the same FFT MPI decomposition on recovery.");
  params.addParam<std::vector<std::string>>(
      "checkpoint_buffers", {"F", "state", "K", "mu"},
      "All tensors needed to resume the next FFT step: committed history plus any initialized "
      "inputs not recomputed upon MOOSE --recover. Do not save only F/state if K/mu are initialized "
      "exclusively during EXEC_INITIAL.");
  params.addParam<std::vector<std::string>>(
      "recovery_seed_from", {"F", "state"},
      "Source tensors copied into trial buffers immediately after recovery.");
  params.addParam<std::vector<std::string>>(
      "recovery_seed_to", {"Fnew", "state_new"},
      "Trial tensors initialized from recovery_seed_from in order.");
  return params;
}

RhoCPFFTTensorProblem::RhoCPFFTTensorProblem(const InputParameters & parameters)
  : TensorProblem(parameters),
    _checkpoint_buffer_names(getParam<std::vector<std::string>>("checkpoint_buffers")),
    _seed_from(getParam<std::vector<std::string>>("recovery_seed_from")),
    _seed_to(getParam<std::vector<std::string>>("recovery_seed_to")),
    _saved_domain(declareRestartableData<std::vector<std::int64_t>>("rhocp_tensor_domain_v1")),
    _saved_names(declareRestartableData<std::vector<std::string>>("rhocp_tensor_names_v1")),
    _saved_shapes(declareRestartableData<std::vector<std::int64_t>>("rhocp_tensor_shapes_v1")),
    _saved_precisions(declareRestartableData<std::vector<unsigned int>>("rhocp_tensor_precisions_v1")),
    _saved_values(declareRestartableData<std::vector<Real>>("rhocp_tensor_values_v1")),
    _saved_timestep(declareRestartableData<int>("rhocp_tensor_timestep_v1", -1))
{
  if (_checkpoint_buffer_names.empty())
    paramError("checkpoint_buffers", "At least F and state must be checkpointed.");
  if (std::find(_checkpoint_buffer_names.begin(), _checkpoint_buffer_names.end(), "F") ==
          _checkpoint_buffer_names.end() ||
      std::find(_checkpoint_buffer_names.begin(), _checkpoint_buffer_names.end(), "state") ==
          _checkpoint_buffer_names.end())
    paramError("checkpoint_buffers", "The committed F and state tensors must be checkpointed.");
  if (_seed_from.size() != _seed_to.size())
    paramError("recovery_seed_to", "Must have the same number of entries as recovery_seed_from.");
  std::vector<std::string> unique = _checkpoint_buffer_names;
  std::sort(unique.begin(), unique.end());
  if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
    paramError("checkpoint_buffers", "Duplicate tensor names are not supported.");
}

void
RhoCPFFTTensorProblem::execute(const ExecFlagType & exec_type)
{
  // Recovery restarts after the normal EXEC_INITIAL tensor-IC stage. The
  // restartable MOOSE data is available by the time the first timestep runs,
  // and all Marlin buffers have already been allocated by TensorProblem::init.
  if (exec_type == EXEC_TIMESTEP_BEGIN && _app.isRecovering() && !_recovered_tensor_data)
  {
    restoreCheckpointTensors();
    _recovered_tensor_data = true;
  }

  // In the current nested FFT architecture the MOOSE FE system can report
  // convergence despite the inner FFT being rejected. Avoid running Marlin
  // postprocessing (notably ComputeDisplacements) on that failed trial.
  // The timestepper still checks the same FFT status and rejects the step.
  if (exec_type == EXEC_TIMESTEP_END &&
      !getSolver<DDCPFFTForwardSolver>().isConverged())
  {
    FEProblem::execute(exec_type);
    return;
  }

  TensorProblem::execute(exec_type);

  // Checkpoint output is called AFTER the problem's TIMESTEP_END execution;
  // hence the restartable vectors contain the newly accepted committed state.
  if (exec_type == EXEC_TIMESTEP_END)
    saveAcceptedTensors();
}

void
RhoCPFFTTensorProblem::saveAcceptedTensors()
{
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(_domain.getMPIComm(), &rank);
  MPI_Comm_size(_domain.getMPIComm(), &ranks);

  std::array<int64_t, 3> begin{}, end{};
  _domain.getLocalBounds(static_cast<unsigned int>(rank), begin, end);
  const auto n = _domain.getGridSize();

  // Version, MPI rank/count, problem dimension, global grid, local bounds.
  _saved_domain = {1, rank, ranks, _domain.getDim(), n[0], n[1], n[2],
                   begin[0], begin[1], begin[2], end[0], end[1], end[2]};
  _saved_names = _checkpoint_buffer_names;
  _saved_shapes.clear();
  _saved_precisions.clear();
  _saved_values.clear();

  for (const auto & name : _checkpoint_buffer_names)
  {
    const auto & t = getBuffer<torch::Tensor>(name);
    if (!t.defined() || t.numel() == 0)
      mooseError("Checkpoint buffer '", name, "' is undefined or empty.");
    if (t.scalar_type() != torch::kFloat64 && t.scalar_type() != torch::kFloat32)
      mooseError("Only real float32/float64 Marlin buffers are supported: '", name, "'.");

    const auto shape = t.sizes().vec();
    if (shape.size() < _domain.getDim())
      mooseError("Checkpoint tensor '", name, "' has fewer dimensions than the FFT domain.");
    for (unsigned int d = 0; d < _domain.getDim(); ++d)
      if (shape[d] != _domain.getLocalGridSize()[d])
        mooseError("Checkpoint tensor '", name, "' has incorrect local spatial dimensions.");

    if (name == "F" &&
        (shape.size() != _domain.getDim() + 2 || shape[shape.size()-2] != 3 || shape.back() != 3))
      mooseError("Checkpointed 'F' must have value dimensions [3,3].");

    _saved_shapes.push_back(static_cast<std::int64_t>(shape.size()));
    _saved_shapes.insert(_saved_shapes.end(), shape.begin(), shape.end());
    _saved_precisions.push_back(t.scalar_type() == torch::kFloat64 ? 64u : 32u);

    // Persist in MOOSE-supported host vectors (never persist Torch handles).
    auto cpu = t.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto * raw = cpu.data_ptr<double>();
    _saved_values.insert(_saved_values.end(), raw, raw + cpu.numel());
  }
  _saved_timestep = static_cast<int>(timeStep());
}

void
RhoCPFFTTensorProblem::validateCheckpointLayout() const
{
  if (_saved_timestep < 0 || _saved_domain.size() != 13 || _saved_domain[0] != 1)
    mooseError("No compatible RhoCP FFT tensor checkpoint found. The old MOOSE-only checkpoint "
               "cannot recover F/state. Rerun from time zero with RhoCPFFTTensorProblem enabled.");

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(_domain.getMPIComm(), &rank);
  MPI_Comm_size(_domain.getMPIComm(), &ranks);
  std::array<int64_t, 3> begin{}, end{};
  _domain.getLocalBounds(static_cast<unsigned int>(rank), begin, end);
  const auto n = _domain.getGridSize();
  const std::vector<std::int64_t> expected = {
      1, rank, ranks, _domain.getDim(), n[0], n[1], n[2],
      begin[0], begin[1], begin[2], end[0], end[1], end[2]};

  if (_saved_domain != expected)
    mooseError("Incompatible tensor checkpoint MPI decomposition/grid on rank ", rank,
               ". Recover with the same rank count, domain and FFT partitioning.");
  if (_saved_names != _checkpoint_buffer_names)
    mooseError("Checkpoint buffer names/order differ from the original run. "
               "Keep Problem/checkpoint_buffers unchanged for --recover.");
  if (_saved_precisions.size() != _saved_names.size())
    mooseError("Corrupt tensor checkpoint: incompatible precision metadata.");
}

void
RhoCPFFTTensorProblem::restoreCheckpointTensors()
{
  validateCheckpointLayout();
  std::size_t shape_cursor = 0;
  std::size_t value_cursor = 0;

  for (std::size_t i = 0; i < _saved_names.size(); ++i)
  {
    if (shape_cursor >= _saved_shapes.size())
      mooseError("Corrupt tensor checkpoint: shape record missing.");
    const auto nd = _saved_shapes[shape_cursor++];
    if (nd < static_cast<int64_t>(_domain.getDim()) || nd > 12 ||
        shape_cursor + static_cast<std::size_t>(nd) > _saved_shapes.size())
      mooseError("Corrupt tensor checkpoint: invalid tensor dimension record.");

    std::vector<int64_t> shape(_saved_shapes.begin() + shape_cursor,
                               _saved_shapes.begin() + shape_cursor + nd);
    shape_cursor += static_cast<std::size_t>(nd);

    int64_t count = 1;
    for (const auto d : shape)
    {
      if (d < 1 || count > std::numeric_limits<int64_t>::max() / d)
        mooseError("Invalid/overflowed checkpoint tensor shape.");
      count *= d;
    }
    for (unsigned int d = 0; d < _domain.getDim(); ++d)
      if (shape[d] != _domain.getLocalGridSize()[d])
        mooseError("Saved tensor '", _saved_names[i], "' has incorrect local domain dimensions.");
    if (_saved_names[i] == "F" &&
        (shape.size() != _domain.getDim()+2 || shape[shape.size()-2] != 3 || shape.back() != 3))
      mooseError("Invalid checkpoint 'F' shape: expected spatial dims followed by [3,3].");
    if (_saved_precisions[i] != 32u && _saved_precisions[i] != 64u)
      mooseError("Unsupported tensor checkpoint precision.");
    if (value_cursor > _saved_values.size() ||
        static_cast<std::size_t>(count) > _saved_values.size() - value_cursor)
      mooseError("Corrupt tensor checkpoint: flattened values truncated.");

    auto & target = getBuffer<torch::Tensor>(_saved_names[i]);
    // The state buffer has an explicit [TensorBuffers]/state value_dimensions
    // allocation even during recovery: enforce exact agreement with the file.
    if (_saved_names[i] == "state" &&
        (!target.defined() || target.sizes().vec() != shape))
      mooseError("Checkpoint DDCP state shape differs from [TensorBuffers]/state allocation. "
                 "Do not change the DDCP state layout between runs.");

    const auto runtime_options = MooseTensor::floatTensorOptions();
    if ((_saved_precisions[i] == 64u && runtime_options.dtype() != torch::kFloat64) ||
        (_saved_precisions[i] == 32u && runtime_options.dtype() != torch::kFloat32))
      mooseError("Checkpoint precision differs from [Domain]/floating_precision.");

    auto host = torch::from_blob(_saved_values.data() + value_cursor,
                                 shape, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    target = host.clone().to(runtime_options);
    value_cursor += static_cast<std::size_t>(count);
  }

  if (shape_cursor != _saved_shapes.size() || value_cursor != _saved_values.size())
    mooseError("Corrupt tensor checkpoint: trailing shape/value data.");

  seedTrialTensors();
  mooseInfo("Recovered RhoCP FFT tensors from accepted timestep ", _saved_timestep,
            " (", _saved_names.size(), " buffers on each MPI rank).");
}

void
RhoCPFFTTensorProblem::seedTrialTensors()
{
  for (std::size_t i = 0; i < _seed_from.size(); ++i)
  {
    const auto & from = getBuffer<torch::Tensor>(_seed_from[i]);
    if (!from.defined())
      mooseError("Recovery seed source buffer '", _seed_from[i], "' is undefined.");
    getBuffer<torch::Tensor>(_seed_to[i]) = from.clone();
  }
}
