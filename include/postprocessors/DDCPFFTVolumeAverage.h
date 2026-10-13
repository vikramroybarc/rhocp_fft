#pragma once

#include "GeneralPostprocessor.h"

class TensorProblem;

/**
 * Global voxel-volume average of selected DDCP4P2 FFT fields.
 * Uses the FFT TensorProblem buffers, not the dummy FE mesh.
 */
class DDCPFFTVolumeAverage : public GeneralPostprocessor
{
public:
  static InputParameters validParams();
  DDCPFFTVolumeAverage(const InputParameters & parameters);

  void initialize() override;
  void execute() override;
  void finalize() override;
  PostprocessorValue getValue() const override;

private:
  TensorProblem & _tensor_problem;
  const std::string _quantity;
  Real _local_sum = 0.0;
  Real _local_count = 0.0;
  PostprocessorValue _value = 0.0;
};
