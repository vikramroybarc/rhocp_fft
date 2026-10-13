#include "DDCPFFTVolumeAverage.h"
#include "TensorProblem.h"
#include "TensorBufferBase.h"
#include "MooseError.h"

#include <torch/torch.h>
#include <string>

registerMooseObject("RhocpFFTApp", DDCPFFTVolumeAverage);

InputParameters
DDCPFFTVolumeAverage::validParams()
{
  auto params = GeneralPostprocessor::validParams();
  params.addClassDescription(
      "Global FFT voxel average for DDCP4P2 fields, for MOOSE CSV comparison.");
  params.addRequiredParam<std::string>(
      "quantity",
      "P_yy, sigma_yy, sigma_xx, sigma_zz, F_yy, F_xx, F_zz, "
      "engineering_strain_yy, Ep_yy, Ep_eff, E_eff, rho_m_avg, rho_i_avg");
  return params;
}

DDCPFFTVolumeAverage::DDCPFFTVolumeAverage(const InputParameters & parameters)
  : GeneralPostprocessor(parameters),
    _tensor_problem(dynamic_cast<TensorProblem &>(_fe_problem)),
    _quantity(getParam<std::string>("quantity"))
{
}

void
DDCPFFTVolumeAverage::initialize()
{
  _local_sum = 0.0;
  _local_count = 0.0;
  _value = 0.0;
}

void
DDCPFFTVolumeAverage::execute()
{
  using torch::indexing::Ellipsis;

  // Owned cells only, so MPI ghost cells are not counted twice.
  // Actual FFT domain averaging uses these buffers, NOT dummy FE mesh elements.
  auto owned = [this](const std::string & name) {
    auto t = _tensor_problem.getBufferBase(name).ownedView();
    if (!t.defined())
      mooseError("FFT buffer '", name, "' is not initialized.");
    return t;
  };

  torch::Tensor field;

  if (_quantity == "P_yy" || _quantity == "sigma_yy" ||
      _quantity == "sigma_xx" || _quantity == "sigma_zz")
  {
    const auto P = owned("stress"); // first Piola-Kirchhoff stress
    if (P.dim() != 5 || P.size(-2) != 3 || P.size(-1) != 3)
      mooseError("Expected FFT first Piola stress shape [nx,ny,nz,3,3].");

    if (_quantity == "P_yy")
      field = P.index({Ellipsis, 1, 1});
    else
    {
      const auto F = owned("F"); // committed deformation gradient
      if (F.sizes() != P.sizes())
        mooseError("F and P must have identical voxel/tensor shapes.");

      const auto J = torch::det(F);
      if (!(J > 0.0).all().item<bool>())
        mooseError("Cannot convert first Piola stress to Cauchy: det(F) <= 0.");

      const int i = (_quantity == "sigma_xx") ? 0 :
                    (_quantity == "sigma_zz") ? 2 : 1;
      // sigma_ii = [P * F^T]_ii / det(F) = sum_J P_iJ F_iJ / det(F).
      field = (P.index({Ellipsis, i, 0}) * F.index({Ellipsis, i, 0}) +
               P.index({Ellipsis, i, 1}) * F.index({Ellipsis, i, 1}) +
               P.index({Ellipsis, i, 2}) * F.index({Ellipsis, i, 2})) / J;
    }
  }
  else if (_quantity == "F_yy" || _quantity == "F_xx" ||
           _quantity == "F_zz" || _quantity == "engineering_strain_yy")
  {
    const auto F = owned("F");
    if (F.dim() != 5 || F.size(-2) != 3 || F.size(-1) != 3)
      mooseError("Expected FFT deformation gradient shape [nx,ny,nz,3,3].");
    const int i = (_quantity == "F_xx") ? 0 :
                  (_quantity == "F_zz") ? 2 : 1;
    field = F.index({Ellipsis, i, i});
    if (_quantity == "engineering_strain_yy")
      field = field - 1.0;
  }
  else
  {
    const auto state = owned("state"); // DDCP4P2 committed state
    if (state.dim() != 4 || state.size(-1) < 52)
      mooseError("Expected DDCP4P2 state shape [nx,ny,nz,nstate>=52].");

    // ZERO-BASED positions, confirmed in DDCP4P2Point.C:
    // SDV 37..45 : plastic Green-Lagrange tensor Ep (3x3, row-major)
    // SDV 46     : effective total strain
    // SDV 47     : Ep_eff = sqrt(2/3 Ep:Ep); NOT cumulative slip strain
    // SDV 48/49  : mean mobile/immobile dislocation densities
    int a = -1;
    if (_quantity == "Ep_yy") a = 40;       // 36 + 3*1 + 1
    else if (_quantity == "E_eff") a = 45;
    else if (_quantity == "Ep_eff") a = 46;
    else if (_quantity == "rho_m_avg") a = 47;
    else if (_quantity == "rho_i_avg") a = 48;
    else paramError("quantity", "Unknown FFT output quantity: ", _quantity);

    field = state.index({Ellipsis, a});
  }

  // Reduce locally. MPI reductions happen in finalize(), not execute().
  // torch reductions work for CPU as well as GPU tensor storage.
  _local_sum = field.to(torch::kFloat64).sum().cpu().item<double>();
  _local_count = static_cast<Real>(field.numel());
}

void
DDCPFFTVolumeAverage::finalize()
{
  gatherSum(_local_sum);
  gatherSum(_local_count);
  if (_local_count <= 0.0)
    mooseError("No owned FFT voxels available for volume average.");
  _value = _local_sum / _local_count;
}

PostprocessorValue
DDCPFFTVolumeAverage::getValue() const
{
  return _value;
}
