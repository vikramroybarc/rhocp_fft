#include "DDCPFFTForwardSolver.h"

#include "DDCPFFTMechanics.h"
#include "TensorProblem.h"
#include "MooseError.h"

registerMooseObject("RhocpFFTApp", DDCPFFTForwardSolver);

InputParameters
DDCPFFTForwardSolver::validParams()
{
  InputParameters params = ForwardEulerSolver::validParams();
  params.addClassDescription(
      "Commit FFT deformation/state only after a successful DDCP mechanics solve; "
      "snapshot and restore tensor buffers for timestep cutback.");
  params.addRequiredParam<TensorComputeName>(
      "mechanics_compute", "Name of DDCPFFTMechanics compute (e.g. 'mech')");
  params.addParam<std::vector<TensorOutputBufferName>>(
      "rollback_buffers",
      {"F", "state", "Fnew", "state_new", "stress", "dstressdstrain", "applied_strain"},
      "Every mutable tensor buffer affecting a retried mechanics increment. "
      "Add other persistent/trial buffers created by custom computes.");
  return params;
}

DDCPFFTForwardSolver::DDCPFFTForwardSolver(const InputParameters & parameters)
  : ForwardEulerSolver(parameters),
    _mechanics(_tensor_problem.getCompute<DDCPFFTMechanics>(
        getParam<TensorComputeName>("mechanics_compute")))
{
  if (!getParam<std::vector<TensorOutputBufferName>>("buffer").empty())
    paramError("buffer", "The DDCP quasi-static FFT cutback adapter does not integrate PDE buffers.");

  if (_substeps != 1)
    paramError("substeps",
               "DDCP FFT cutback currently requires substeps = 1 so failed trials "
               "cannot partially advance tensor histories.");

  for (const auto & name : getParam<std::vector<TensorOutputBufferName>>("rollback_buffers"))
    _rollback_buffers.emplace_back(_tensor_problem.getBuffer<torch::Tensor>(name));
  if (_rollback_buffers.empty())
    paramError("rollback_buffers", "Specify all mutable deformation/state tensors");
}

void
DDCPFFTForwardSolver::snapshotBuffers()
{
  _snapshots.clear();
  _snapshots.reserve(_rollback_buffers.size());
  for (auto & buffer : _rollback_buffers)
  {
    auto & t = buffer.get();
    _snapshots.emplace_back(t.defined() ? t.clone() : torch::Tensor{});
  }
}

void
DDCPFFTForwardSolver::restoreBuffers()
{
  if (_snapshots.empty())
    return;

  for (std::size_t i = 0; i < _rollback_buffers.size(); ++i)
  {
    // Reassign a deep clone rather than performing an in-place copy.  Some
    // tensors (e.g. initialized identity buffers) may be expanded views.
    _rollback_buffers[i].get() =
        _snapshots[i].defined() ? _snapshots[i].clone() : torch::Tensor{};
  }
}

bool
DDCPFFTForwardSolver::isConverged() const
{
  return _mechanics.isConverged();
}

unsigned int
DDCPFFTForwardSolver::nonlinearIterations() const
{
  return _mechanics.nonlinearIterations();
}

void
DDCPFFTForwardSolver::computeBuffer()
{
  _mechanics.resetStatus();
  snapshotBuffers();

  // The parent ForwardEulerSolver normally calls forwardBuffers() even after
  // an unsuccessful nested FFT solve.  Instead, run the compute group once and
  // commit only on success.  No reciprocal-space integration variables are used
  // for the DDCP quasistatic mechanics use case.
  _sub_time = _time;
  _sub_dt = _dt;
  _compute->computeBuffer();

  if (isConverged())
    forwardBuffers();
  else
    restoreBuffers();
}
