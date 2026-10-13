#pragma once

#include "ConstantDT.h"

class TensorProblem;
class DDCPFFTForwardSolver;

/** MOOSE timestep rejection driven by the nested DDCP FFT solve status. */
class DDCPFFTCutbackDT : public ConstantDT
{
public:
  static InputParameters validParams();
  explicit DDCPFFTCutbackDT(const InputParameters & parameters);

  bool converged() const override;
  void rejectStep() override;

private:
  TensorProblem & _tensor_problem;
};
