#include "DDCPTensorCompute.h"
#include "DDCP4P2Tangent.h"
#include "DDCPOrientationMap.h"
#include "DomainAction.h"
#include "MooseError.h"
#include "TensorProblem.h"

#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <mpi.h>

registerMooseObject("RhocpFFTApp", DDCPTensorCompute);

namespace
{
std::vector<Real>
loadProperties(const FileName & path)
{
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Unable to open DDCP4P2 properties file: " + std::string(path));
  std::vector<Real> values;
  Real value;
  while (f >> value) values.push_back(value);
  if (values.size() < 54)
    throw std::runtime_error("DDCP4P2 requires >=54 numeric properties but " +
                             std::to_string(values.size()) + " were supplied in " + std::string(path));
  return values;
}


DDCP4P2Data
loadData(const FileName & props, const FileName & slips,
         bool climb, bool irradiation, bool deltaH_eV,
         unsigned int slip_law, unsigned int crss_law,
         unsigned int decomposition, Real tol)
{
  DDCP4P2Data data;
  data.props = loadProperties(props);
  data.climb = climb;
  data.irradiation = irradiation;
  data.deltaH_eV = deltaH_eV;
  data.slip_law = slip_law;
  data.crss_law = crss_law;
  data.decomposition = decomposition;
  data.tolerance = tol;
  std::ifstream f(slips);
  if (!f) throw std::runtime_error("Unable to open DDCP4P2 slip-system file: " + std::string(slips));
  std::array<Real,3> m,s;
  // Exact file convention of DDCP4P2: normal (3) then direction (3).
  while (f >> m[0] >> m[1] >> m[2] >> s[0] >> s[1] >> s[2])
  {
    data.normals.push_back(m);
    data.directions.push_back(s);
  }
  if (data.normals.empty()) throw std::runtime_error("No DDCP4P2 slip systems");
  if (decomposition > 1) throw std::runtime_error("decomposition must be 0 (NONE) or 1 (VOLDEV)");
  return data;
}

RankTwoTensor
loadMatrix(const double * values)
{
  RankTwoTensor r;
  for (unsigned int i=0; i<3; ++i)
    for (unsigned int j=0; j<3; ++j)
      r(i,j) = values[3*i+j];
  return r;
}
}

InputParameters
DDCPTensorCompute::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("CPU DDCP4P2 with local implicit-function tangent and full-update FD fallback.");
  params.addRequiredParam<TensorInputBufferName>("F_trial", "Trial deformation gradient buffer");
  params.addRequiredParam<TensorInputBufferName>("F_old", "Committed deformation gradient buffer");
  params.addRequiredParam<TensorInputBufferName>("state_old", "Committed state-variable buffer");
  params.addRequiredParam<TensorOutputBufferName>("state_new", "Trial state-variable buffer");
  params.addParam<TensorOutputBufferName>("tangent_operator", "dstressdstrain", "FFT tangent buffer");
  params.addParam<std::string>("material_model", "DDCP4P2Point",
                               "Name of a DDCP4P2Point subclass registered with DDCPPointFactory. Available built-ins: "
                               "DDCP4P2Point (original) or DDCP4P2PointModified "
                               "(stress-scale demonstrator only).");
  params.addParam<Real>("model_stress_scale", 1.0,
                        "Demonstration-only stress multiplier for DDCP4P2PointModified. "
                        "Does not modify dislocation-density evolution.");
  params.addRequiredParam<FileName>("propsFile", "Original 54-entry DDCP properties file");
  params.addRequiredParam<FileName>("slipSysFile", "Original six-columns-per-system slip input");
  params.addParam<FileName>("orientation_file", "orientation.in",
                            "Global polycrystal file: GRID dimensions, GRAINS with Bunge Euler "
                            "angles (degrees), then VOXELS grain IDs in z-fastest order");
  params.addParam<Real>("temperature", 300., "Temperature in K");
  params.addParam<Real>("g", 1., "Elastic degradation factor");
  params.addParam<Real>("gp", 1., "Plastic degradation factor");
  params.addParam<Real>("fd_step", 1e-6, "Relative perturbation for full-update dP/dF and IFT F derivatives");
  params.addParam<std::string>("tangent_method", "legacy",
      "legacy: use_implicit_tangent chooses IFT/FD (unchanged); "
      "ift: IFT with FD fallback; fd: full-update FD; "
      "option_a: approximate Jaumann-Cauchy pull-back of C_local");
  params.addParam<bool>("use_implicit_tangent", true,
                        "Use local Newton IFT tangent if a single subincrement converged; "
                        "otherwise fall back to the full-update FD tangent");
  params.addParam<Real>("ift_rate_step", 1e-5,
                        "Relative finite difference step for local glide/climb rates");
  params.addParam<bool>("verify_tangent", false,
                        "Independent mixed-direction dP/dF check on the first voxel");
  params.addParam<Real>("verify_tolerance", 1e-3,
                        "Relative error tolerance for the mixed-direction check");
  params.addParam<bool>("verify_tangent_warn_only", false,
                        "Report finite tangent-check failures without terminating; "
                        "diagnostic only, not a validation pass");
  params.addParam<bool>("verify_reference_fd", false,
                        "Also compute full-update FD reference tangent for the first voxel "
                        "and print comparison diagnostics; expensive");
  params.addParam<unsigned int>("verify_max_reports", 10,
                                "Maximum tangent diagnostic reports per MPI rank");
  params.addParam<Real>("tol", 1e-6, "Local Newton tolerance");
  params.addParam<bool>("climbmodel", false, "Enable original climb equations");
  params.addParam<bool>("irad_defects", false, "Enable irradiation loops");
  params.addParam<bool>("deltaH_eV", false, "Activation enthalpy convention");
  params.addParam<unsigned int>("sliplaw", 1, "Original slip-rate law selector");
  params.addParam<unsigned int>("crsslaw", 1, "Original hardening selector");
  params.addParam<unsigned int>("decomposition", 0, "0 NONE; 1 VOLDEV");
  return params;
}

DDCPTensorCompute::DDCPTensorCompute(const InputParameters & params)
  : TensorOperator<>(params),
    _F_trial(getInputBuffer("F_trial")),
    _F_old(getInputBuffer("F_old")),
    _state_old(getInputBuffer("state_old")),
    _state_new(getOutputBuffer("state_new")),
    _tangent(getOutputBuffer("tangent_operator")),
    _data(loadData(getParam<FileName>("propsFile"),
                   getParam<FileName>("slipSysFile"),
                   getParam<bool>("climbmodel"),
                   getParam<bool>("irad_defects"),
                   getParam<bool>("deltaH_eV"),
                   getParam<unsigned int>("sliplaw"),
                   getParam<unsigned int>("crsslaw"),
                   getParam<unsigned int>("decomposition"),
                   getParam<Real>("tol"))),
    _material_model_name(getParam<std::string>("material_model")),
    _material([&]()
    {
      DDCPPointOptions options;
      options.stress_scale = getParam<Real>("model_stress_scale");
      return DDCPPointFactory::create(_material_model_name, _data, options);
    }()),
    _temperature(getParam<Real>("temperature")),
    _g(getParam<Real>("g")),
    _gp(getParam<Real>("gp")),
    _fd_step(getParam<Real>("fd_step")),
    _use_implicit_tangent(getParam<bool>("use_implicit_tangent")),
    _tangent_method(getParam<std::string>("tangent_method")),
    _ift_rate_step(getParam<Real>("ift_rate_step")),
    _verify_tangent(getParam<bool>("verify_tangent")),
    _verify_tolerance(getParam<Real>("verify_tolerance")),
    _verify_tangent_warn_only(getParam<bool>("verify_tangent_warn_only")),
    _verify_reference_fd(getParam<bool>("verify_reference_fd")),
    _verify_max_reports(getParam<unsigned int>("verify_max_reports"))
{
  if (_tangent_method != "legacy" && _tangent_method != "ift" &&
      _tangent_method != "fd" && _tangent_method != "option_a")
    paramError("tangent_method", "Use legacy, ift, fd, or option_a");
  if (_ift_rate_step <= 0. || !std::isfinite(_ift_rate_step))
    paramError("ift_rate_step", "Must be positive and finite");
  if (_fd_step <= 0. || !std::isfinite(_fd_step))
    paramError("fd_step", "Must be positive and finite");
  if (_verify_tolerance <= 0. || !std::isfinite(_verify_tolerance))
    paramError("verify_tolerance", "Must be positive and finite");
  // Load a single global microstructure map on each MPI rank, then select its
  // own non-overlapping spatial subset using Marlin's true real-space bounds.
  // This works with serial, FFT_SLAB and FFT_PENCIL decompositions and avoids
  // relying on local flat indices that change with MPI rank count.
  const auto orientation_file = getParam<FileName>("orientation_file");
  DDCPOrientation::Map orientation_map;
  try
  {
    orientation_map = DDCPOrientation::Map::read(std::string(orientation_file));
  }
  catch (const std::exception & e)
  {
    paramError("orientation_file", e.what());
  }

  if (_dim != 3)
    paramError("orientation_file", "The DDCP4P2 point model requires a 3D voxel grid");

  const auto global_grid = _domain.getGridSize();
  if (orientation_map.grid() != global_grid)
    paramError("orientation_file", "GRID dimensions in orientation.in must match "
               "the global [Domain] nx, ny and nz exactly");

  int mpi_rank = 0;
  MPI_Comm_rank(_domain.getMPIComm(), &mpi_rank);
  std::array<int64_t, 3> first{{0, 0, 0}}, last{{0, 0, 0}};
  _domain.getLocalBounds(static_cast<unsigned int>(mpi_rank), first, last);
  const auto local_grid = _domain.getLocalGridSize();
  for (unsigned int d = 0; d < 3; ++d)
    if (last[d] - first[d] != local_grid[d])
      mooseError("DDCP4P2 orientation: inconsistent Marlin local grid bounds on rank ", mpi_rank);

  _orientations.reserve(static_cast<std::size_t>(local_grid[0] * local_grid[1] * local_grid[2]));
  _voxel_grain_ids.reserve(_orientations.capacity());
  for (int64_t i = first[0]; i < last[0]; ++i)
    for (int64_t j = first[1]; j < last[1]; ++j)
      for (int64_t k = first[2]; k < last[2]; ++k)
      {
        const auto e = orientation_map.eulerDegrees(i, j, k);
        _orientations.push_back({{e[0], e[1], e[2]}});
        _voxel_grain_ids.push_back(orientation_map.grainId(i, j, k));
      }

  if (mpi_rank == 0)
    mooseInfo("DDCP4P2 selected material_model=", _material_model_name,
              "; stateSize=", _material->stateSize(),
              "; polycrystal loaded ", orientation_map.grainCount(),
              " grains and ", orientation_map.voxelCount(),
              " global voxels from ", orientation_file);

}

std::string
DDCPTensorCompute::materialModelSignature() const
{
  std::ostringstream result;
  result << _material_model_name << ";model_stress_scale="
         << std::setprecision(17) << getParam<Real>("model_stress_scale");
  return result.str();
}

void
DDCPTensorCompute::computeBuffer()
{
  // This catches missing tensor checkpoint restoration before LibTorch reshape
  // reports the cryptic "shape [-1,9] is invalid for input of size 4" error.
  const auto expected_F = _domain.getValueShape({3,3});
  const auto expected_state = _domain.getValueShape({static_cast<int64_t>(_material->stateSize())});
  if (!_F_trial.defined() || !_F_old.defined() || !_state_old.defined() ||
      _F_trial.sizes().vec() != expected_F ||
      _F_old.sizes().vec() != expected_F ||
      _state_old.sizes().vec() != expected_state)
    throw std::runtime_error(
        "DDCP model '" + _material_model_name +
        "': inconsistent F/state tensor shape; expected spatial [...,3,3] and "
        "[...,stateSize]. Check TensorBuffers/state value_dimensions and "
        "RhoCPFFTTensorProblem recovery. Timestep cutback cannot repair a shape mismatch.");

  // The scalar material points are CPU-only.
  if (!_F_trial.device().is_cpu() || !_F_old.device().is_cpu() ||
      !_state_old.device().is_cpu())
    mooseError("DDCPTensorCompute requires CPU tensors. Set Marlin's device to CPU.");
  if (_tensor_problem.subDt() <= 0.)
    mooseError("DDCPTensorCompute requires positive substep dt");

  const auto Fnew = _F_trial.to(torch::kFloat64).contiguous().reshape({-1,9});
  const auto Fold = _F_old.to(torch::kFloat64).contiguous().reshape({-1,9});
  const auto state = _state_old.to(torch::kFloat64).contiguous();
  const auto nvox = Fnew.size(0);
  const auto nstate = _material->stateSize();
  if (Fold.size(0) != nvox || state.numel() != nvox*nstate)
    mooseError("DDCP4P2 F/state buffer dimensions do not match grid or stateSize");
  if (static_cast<int64_t>(_orientations.size()) != nvox ||
      static_cast<int64_t>(_voxel_grain_ids.size()) != nvox)
    mooseError("DDCP4P2 orientation count does not match locally-owned FFT voxels; "
               "check [Domain] dimensions and orientation_file");

  const auto old_states = state.reshape({nvox,static_cast<int64_t>(nstate)});
  const auto P = torch::empty_like(Fnew);
  const auto C = torch::empty({nvox,81}, Fnew.options());
  const auto new_states = torch::empty_like(old_states);

  const double * pfnew = Fnew.data_ptr<double>();
  const double * pfold = Fold.data_ptr<double>();
  const double * pstate = old_states.data_ptr<double>();
  double * pp = P.data_ptr<double>();
  double * pc = C.data_ptr<double>();
  double * pnstate = new_states.data_ptr<double>();

  auto & material = *_material;
  for (int64_t v=0; v<nvox; ++v)
  {
    DDCP4P2Input in;
    in.F_old = loadMatrix(pfold+9*v);
    in.F_new = loadMatrix(pfnew+9*v);
    in.state_old.assign(pstate+nstate*v,pstate+nstate*(v+1));
    in.euler_degrees = _orientations[v];
    in.grain_id = _voxel_grain_ids[v];
    in.dt = _tensor_problem.subDt();
    in.temperature = _temperature;
    in.g = _g;
    in.gp = _gp;
    in.request_implicit_tangent =
        (_tangent_method == "ift") ||
        (_tangent_method == "legacy" && _use_implicit_tangent);
    in.request_option_a_tangent = (_tangent_method == "option_a");
    in.ift_F_step = _fd_step;
    in.ift_rate_step = _ift_rate_step;

    DDCP4P2Result result;
    try
    {
      result = material.integrate(in);
    }
    catch (const std::exception & e)
    {
      throw std::runtime_error("DDCP4P2 local voxel " + std::to_string(v) +
                               " grain " + std::to_string(in.grain_id) +
                               " integration failed: " + e.what());
    }
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        pp[9*v+3*i+j] = result.P(i,j);
    if (result.state.size() != nstate)
      throw std::runtime_error("DDCP model '" + _material_model_name +
                               "' returned a state of length " +
                               std::to_string(result.state.size()) +
                               ", expected " + std::to_string(nstate));
    std::copy(result.state.begin(),result.state.end(),pnstate+nstate*v);

    // Select tangent without changing the already-converged constitutive stress.
    // Option A makes an explicit, unverified Jaumann-Cauchy interpretation
    // of the existing local rate stiffness. No constitutive re-integration.
    RankFourTensor A;
    std::string tangent_source;
    if (_tangent_method == "option_a")
    {
      if (!result.option_a_tangent_available)
        throw std::runtime_error("DDCP4P2 Option A tangent was not returned");
      A = result.A_option_a;
      tangent_source = "Option A approximate Jaumann pull-back";
    }
    else if (in.request_implicit_tangent && result.implicit_tangent_available)
    {
      A = result.A_implicit;
      tangent_source = "IFT";
    }
    else
    {
      if (in.request_implicit_tangent && v == 0)
        mooseWarning("DDCP4P2 IFT tangent unavailable; using full-update FD: ",
                     result.ift_status);
      try
      {
        DDCP4P2Input full_fd_input = in;
        full_fd_input.request_implicit_tangent = false;
        full_fd_input.request_option_a_tangent = false;
        A = DDCP4P2Tangent::centralDifferencePK1(full_fd_input, material, _fd_step);
        tangent_source = "full FD";
      }
      catch (const std::exception & e)
      {
        throw std::runtime_error("DDCP4P2 local voxel " + std::to_string(v) +
                                 " grain " + std::to_string(in.grain_id) +
                                 " full-update tangent failed: " + e.what());
      }
    }

    // IMPORTANT: Marlin's ddot42(A,B) uses einsum("...ijkl,...lk->...ij").
    // For trans2(ddot42(K, trans2(dF))), the input transposition cancels
    // ddot42's reversed contracted indices; the output remains transposed.
    // Therefore K(J,i,k,L) = A(i,J,k,L), NOT K(J,i,L,k).
    for (int i=0; i<3; ++i)
      for (int J=0; J<3; ++J)
        for (int k=0; k<3; ++k)
          for (int L=0; L<3; ++L)
            pc[81*v + 27*J + 9*i + 3*k + L] = A(i,J,k,L);

    if (_verify_tangent && v == 0)
    {
      try
      {
        DDCP4P2Input verify_input = in;
        verify_input.request_implicit_tangent = false;
        const Real error = DDCP4P2Tangent::directionalRelativeError(
            verify_input, material, A, 2.0*_fd_step);
        const bool failed = !std::isfinite(error) || error > _verify_tolerance;
        const bool should_report = _verify_reports < _verify_max_reports;
        if (_verify_reference_fd && should_report)
        {
          // Compare to the complete-update FD tangent at exactly the SAME state,
          // with the SAME independent directional perturbation.
          const auto reference_A = DDCP4P2Tangent::centralDifferencePK1(
              verify_input, material, _fd_step);
          const Real reference_error = DDCP4P2Tangent::directionalRelativeError(
              verify_input, material, reference_A, 2.0*_fd_step);
          mooseInfo("DDCP4P2 tangent diagnostic voxel=0; selected=",
                    tangent_source,
                    "; selected_error=", error,
                    "; reference_full_FD_error=", reference_error,
                    "; allowed=", _verify_tolerance,
                    "; dt=", in.dt,
                    "; used_subincrements=", result.used_constitutive_subincrements,
                    "; IFT_status=", result.ift_status);
          ++_verify_reports;
        }
        if (failed)
        {
          // NaN/Inf always stops: it is not merely a tolerance issue.
          if (!_verify_tangent_warn_only || !std::isfinite(error))
            mooseError("DDCP4P2 FFT tangent directional check: relative error = ",
                       error, "; allowed = ", _verify_tolerance,
                       "; source=",
                       tangent_source);
          if (!_verify_reference_fd && should_report)
          {
            mooseWarning("DDCP4P2 FFT tangent diagnostic (NOT VALIDATED): error=",
                         error, "; allowed=", _verify_tolerance,
                         "; source=",
                         tangent_source);
            ++_verify_reports;
          }
        }
      }
      catch (const std::exception & e)
      {
        mooseError("DDCP4P2: independent tangent check failed at voxel 0: ", e.what());
      }
    }
  }
  _u = P.reshape(_F_trial.sizes()).to(_F_trial.options());
  _tangent = C.reshape(_domain.getValueShape({3,3,3,3})).to(_F_trial.options());
  _state_new = new_states.reshape(_state_old.sizes()).to(_state_old.options());
}
