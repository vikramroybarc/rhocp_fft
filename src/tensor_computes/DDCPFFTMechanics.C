#include "DDCPFFTMechanics.h"

#include "DomainAction.h"
#include "MarlinUtils.h"
#include "MooseError.h"

#include <cmath>
#include <exception>
#include <mpi.h>
#include <string>
#include <vector>

registerMooseObject("RhocpFFTApp", DDCPFFTMechanics);

InputParameters
DDCPFFTMechanics::validParams()
{
  InputParameters params = FFTMechanics::validParams();
  params.addClassDescription(
      "FFTMechanics with collective, nonfatal DDCP/FFT failure reporting for timestep cutback.");
  return params;
}

DDCPFFTMechanics::DDCPFFTMechanics(const InputParameters & parameters)
  : FFTMechanics(parameters)
{
}

void
DDCPFFTMechanics::resetStatus()
{
  _converged = true;
  _nonlinear_iterations = 0;
  _failure_reason.clear();
}

void
DDCPFFTMechanics::markFailed(const std::string & message)
{
  _converged = false;
  _failure_reason = message;
  int rank = 0;
  MPI_Comm_rank(_domain.getMPIComm(), &rank);
  if (rank == 0)
    mooseWarning("DDCP FFT solve rejected: ", message);
}

bool
DDCPFFTMechanics::evaluateConstitutiveCollectively()
{
  // DDCP4P2TensorCompute evaluates only locally owned voxels and performs no
  // MPI collectives. Catch local material failures; all ranks then collectively
  // agree whether to continue. IMPORTANT: do not enter a distributed FFT if
  // another rank encountered a local constitutive exception.
  int failed_here = 0;
  std::string local_message;
  try
  {
    _constitutive_model.computeBuffer();
  }
  catch (const std::exception & e)
  {
    failed_here = 1;
    local_message = e.what();
  }
  catch (...)
  {
    failed_here = 1;
    local_message = "non-standard exception in the constitutive compute";
  }

  int failed_anywhere = 0;
  MPI_Allreduce(&failed_here, &failed_anywhere, 1, MPI_INT, MPI_MAX, _domain.getMPIComm());
  if (!failed_anywhere)
    return true;

  int rank = 0;
  MPI_Comm_rank(_domain.getMPIComm(), &rank);
  if (failed_here)
    mooseWarning("DDCP constitutive failure on MPI rank ", rank, ": ", local_message);
  markFailed("at least one MPI rank failed local DDCP integration/tangent evaluation");
  return false;
}

void
DDCPFFTMechanics::computeBuffer()
{
  // This reproduces upstream FFTMechanics::computeBuffer(), except for:
  //   * a collective DDCP-failure check after every constitutive evaluation;
  //   * a nonfatal failure return when nonlinear iterations are exhausted;
  //   * explicit detection of non-finite iteration norms.
  using namespace MooseTensor;
  resetStatus();

  const auto shape = _domain.getShape();
  std::vector<int64_t> r2shape(shape.begin(), shape.end());
  r2shape.push_back(_dim);
  r2shape.push_back(_dim);

  const auto G = [&](const torch::Tensor & A2)
  { return _domain.ifft(ddot42(_Ghat4, _domain.fft(A2))).reshape(-1); };
  const auto K_dF = [&](const torch::Tensor & dFm)
  { return trans2(ddot42(_tK4, trans2(dFm.reshape(r2shape)))); };
  const auto G_K_dF = [&](const torch::Tensor & dFm) { return G(K_dF(dFm)); };
  const auto reduce_sum = [&](const torch::Tensor & t) { return _domain.allreduceSum(t); };
  const auto global_norm = [&](const torch::Tensor & r)
  {
    const auto local = torch::sum(r * r);
    const auto global = reduce_sum(local);
    return std::sqrt(global.cpu().item<double>());
  };

  // Keep the committed _tF unchanged; _u is the trial Fnew buffer.
  _u = _tF.clone();
  if (!evaluateConstitutiveCollectively())
    return;

  auto b = _applied_macroscopic_strain
               ? -G_K_dF(_applied_macroscopic_strain->expand(_r2_shape))
               : -G_K_dF(torch::zeros_like(_tF));

  if (_applied_macroscopic_strain)
    _u = _u + _applied_macroscopic_strain->expand(_r2_shape);

  const double Fn = global_norm(_u);
  if (!std::isfinite(Fn) || Fn <= 0.0)
  {
    markFailed("invalid initial deformation-gradient norm");
    return;
  }

  unsigned int iiter = 0;
  auto dFm = torch::zeros_like(b);
  while (true)
  {
    const auto [dFm_new, linear_iterations, lnorm] = conjugateGradientSolve(
        G_K_dF, b, dFm, _l_tol, _l_max_its,
        [](const torch::Tensor & r) { return r; }, reduce_sum);
    const double bnorm = global_norm(b);
    dFm = dFm_new;

    // The CG implementation must return on every rank; all subsequent checks
    // use globally reduced norms and therefore make the same branch decision.
    if (!std::isfinite(lnorm) || !std::isfinite(bnorm))
    {
      markFailed("non-finite linear CG residual");
      return;
    }
    if (bnorm > 0.0 && linear_iterations >= _l_max_its &&
        lnorm > _l_tol * bnorm)
    {
      markFailed("linear FFT conjugate-gradient solve reached l_max_its without converging");
      return;
    }

    _u = _u + dFm.reshape(r2shape);
    if (!evaluateConstitutiveCollectively())
      return;

    b = -G(_tP);
    const double anorm = global_norm(dFm);
    const double rnorm = anorm / Fn;

    if (!std::isfinite(anorm) || !std::isfinite(rnorm))
    {
      markFailed("non-finite FFT nonlinear update norm");
      return;
    }

    if (_verbose)
      _console << "|R|=" << anorm << "\t|R/R0|=" << rnorm << '\n';

    if ((rnorm < _nl_rel_tol || anorm < _nl_abs_tol) && iiter > 0)
    {
      _nonlinear_iterations = iiter + 1;
      return;
    }

    ++iiter;
    _nonlinear_iterations = iiter;
    if (iiter > _nl_max_its)
    {
      markFailed("FFTMechanics exceeded nl_max_its=" + std::to_string(_nl_max_its));
      return;
    }
  }
}
