#pragma once

#include "TensorOperator.h"
#include "DDCPPointFactory.h"
#include <memory>
#include <string>
#include <array>
#include <vector>

// CPU-only, voxel-by-voxel adapter around a selectable local constitutive model.
// Only trial buffers are modified during the global FFT solve.
class DDCPTensorCompute : public TensorOperator<>
{
public:
  static InputParameters validParams();
  explicit DDCPTensorCompute(const InputParameters & params);
  void computeBuffer() override;
  bool supportsJIT() const override { return false; }
  const std::string & materialModelName() const { return _material_model_name; }
  std::string materialModelSignature() const;

private:
  const torch::Tensor & _F_trial;
  const torch::Tensor & _F_old;
  const torch::Tensor & _state_old;
  torch::Tensor & _state_new;
  torch::Tensor & _tangent;
  DDCP4P2Data _data;
  const std::string _material_model_name;
  std::unique_ptr<DDCP4P2Point> _material;
  // One orientation and grain ID per local voxel, in contiguous Torch order.
  std::vector<std::array<Real,3>> _orientations;
  std::vector<int> _voxel_grain_ids;
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
