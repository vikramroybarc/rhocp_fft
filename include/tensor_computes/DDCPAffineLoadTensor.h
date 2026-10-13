#pragma once
#include "TensorOperator.h"

// Uniform tension/shear macro deformation increment for FFTMechanics.
class DDCPAffineLoadTensor : public TensorOperator<>
{
public:
  static InputParameters validParams();
  explicit DDCPAffineLoadTensor(const InputParameters & params);
  void computeBuffer() override;

private:
  const torch::Tensor & _F;
  unsigned int _i;
  unsigned int _j;
  Real _rate;
};
