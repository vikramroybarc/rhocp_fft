#include "DDCPFFTCutbackDT.h"

#include "DDCPFFTForwardSolver.h"
#include "TensorProblem.h"

registerMooseObject("RhocpFFTApp", DDCPFFTCutbackDT);

InputParameters
DDCPFFTCutbackDT::validParams()
{
  InputParameters params = ConstantDT::validParams();
  params.addClassDescription(
      "Constant target dt with MOOSE failure cutback when the nested DDCP FFT "
      "mechanics solver reports nonconvergence.");
  return params;
}

DDCPFFTCutbackDT::DDCPFFTCutbackDT(const InputParameters & parameters)
  : ConstantDT(parameters),
    _tensor_problem(TensorProblem::cast(this, _fe_problem))
{
}

bool
DDCPFFTCutbackDT::converged() const
{
  // MOOSE's FE solve may succeed with an empty nonlinear system while the
  // nested Marlin FFT solve failed during EXEC_TIMESTEP_BEGIN.
  return ConstantDT::converged() &&
         _tensor_problem.getSolver<DDCPFFTForwardSolver>().isConverged();
}

void
DDCPFFTCutbackDT::rejectStep()
{
  ConstantDT::rejectStep(); // Restore FE solutions and other MOOSE state.
  _tensor_problem.getSolver<DDCPFFTForwardSolver>().restoreBuffers(); // Separately restore Marlin Torch tensors.
}
