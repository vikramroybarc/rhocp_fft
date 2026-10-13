#pragma once

#include "TensorOperator.h"
#include "DDCP4P2Point.h"
#include <array>
#include <vector>

// CPU-only, voxel-by-voxel adapter around the unchanged local DDCP4P2 equations.
// Only trial buffers are modified during the global FFT solve.
class DDCP4P2TensorCompute : public TensorOperator<>
{
public:
  static InputParameters validParams();
  explicit DDCP4P2TensorCompute(const InputParameters & params);
  void computeBuffer() override;
  bool supportsJIT() const override { return false; }

private:
  const torch::Tensor & _F_trial;
  const torch::Tensor & _F_old;
  const torch::Tensor & _state_old;
  torch::Tensor & _state_new;
  torch::Tensor & _tangent;
  DDCP4P2Data _data;
  std::vector<std::array<Real,3>> _orientations;
  std::array<Real,3> _euler_degrees;
  Real _temperature;
  Real _g;
  Real _gp;
  Real _fd_step;
  bool _use_implicit_tangent;
  Real _ift_rate_step;
  bool _verify_tangent;
  Real _verify_tolerance;
  bool _verify_tangent_warn_only;
  bool _verify_reference_fd;
  unsigned int _verify_max_reports;
  unsigned int _verify_reports = 0;
};
