#pragma once

#include "TensorProblem.h"
#include <cstdint>
#include <string>
#include <vector>

/**
 * TensorProblem with checkpoint/recovery of selected Marlin Torch buffers.
 *
 * Unlike normal FE solution vectors, Marlin tensors are not automatically
 * registered with MOOSE restartable data. This class serializes accepted
 * tensors at TIMESTEP_END and restores them before the first recovered solve.
 *
 * Checkpoint format v1: per-MPI-rank metadata, names, shapes, precisions and
 * flattened (double-precision host) values. Recovery requires identical
 * domain shape, rank count, decomposition and selected buffer names.
 */
class RhoCPFFTTensorProblem : public TensorProblem
{
public:
  static InputParameters validParams();
  explicit RhoCPFFTTensorProblem(const InputParameters & parameters);

  void execute(const ExecFlagType & exec_type) override;

protected:
  void saveAcceptedTensors();
  void restoreCheckpointTensors();
  void validateCheckpointLayout() const;
  void seedTrialTensors();

  const std::vector<std::string> _checkpoint_buffer_names;
  const std::vector<std::string> _seed_from;
  const std::vector<std::string> _seed_to;

  // All restartable data must be registered during object construction.
  std::vector<std::int64_t> & _saved_domain;
  std::vector<std::string> & _saved_names;
  std::vector<std::int64_t> & _saved_shapes;
  std::vector<unsigned int> & _saved_precisions;
  std::vector<Real> & _saved_values;
  int & _saved_timestep;

  bool _recovered_tensor_data = false; // runtime-only flag
};
