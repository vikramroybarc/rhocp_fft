#pragma once

#include "ForwardEulerSolver.h"
#include <functional>
#include <vector>

class DDCPFFTMechanics;

/**
 * Drives one FFT mechanics increment and only commits forwarded F/state on
 * success. Holds deep copies of tensor buffers for MOOSE timestep rejection.
 */
class DDCPFFTForwardSolver : public ForwardEulerSolver
{
public:
  static InputParameters validParams();
  explicit DDCPFFTForwardSolver(const InputParameters & parameters);

  void computeBuffer() override;
  bool isConverged() const;
  unsigned int nonlinearIterations() const;
  void restoreBuffers();

private:
  void snapshotBuffers();

  DDCPFFTMechanics & _mechanics;
  std::vector<std::reference_wrapper<torch::Tensor>> _rollback_buffers;
  std::vector<torch::Tensor> _snapshots;
};
