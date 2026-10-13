#include "DDCPAffineLoadTensor.h"
#include "DomainAction.h"
#include "MarlinUtils.h"
#include "MooseError.h"
#include <ATen/TensorIndexing.h>

registerMooseObject("RhocpFFTApp", DDCPAffineLoadTensor);

InputParameters
DDCPAffineLoadTensor::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addParam<TensorInputBufferName>("F", "F", "Committed deformation gradient");
  params.addParam<unsigned int>("i", 1, "Row of imposed F component (zero based)");
  params.addParam<unsigned int>("j", 1, "Column of imposed F component (zero based)");
  params.addParam<Real>("rate", 1e-3, "Rate of macroscopic F_ij component");
  return params;
}

DDCPAffineLoadTensor::DDCPAffineLoadTensor(const InputParameters & params)
  : TensorOperator<>(params),
    _F(getInputBuffer("F")),
    _i(getParam<unsigned int>("i")),
    _j(getParam<unsigned int>("j")),
    _rate(getParam<Real>("rate"))
{
  if (_i >= _dim || _j >= _dim)
    paramError("i", "Requested affine deformation component exceeds dimensionality");
}

void
DDCPAffineLoadTensor::computeBuffer()
{
  auto target = torch::eye(_dim, MooseTensor::floatTensorOptions());
target.index_put_(
    {static_cast<int64_t>(_i), static_cast<int64_t>(_j)},
    (_i == _j ? 1.0 : 0.0) + _rate * _time
);
  _u = target - _domain.average(_F);
}
