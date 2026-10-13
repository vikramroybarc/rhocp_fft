#pragma once

#include "FFTMechanics.h"
#include <string>

/**
 * The deGeus FFT mechanics solve with a recoverable nonlinear / DDCP failure.
 * All MPI ranks agree on a material-point failure before entering an FFT collective.
 */
class DDCPFFTMechanics : public FFTMechanics
{
public:
  static InputParameters validParams();
  explicit DDCPFFTMechanics(const InputParameters & parameters);
  void computeBuffer() override;

  void resetStatus();
  bool isConverged() const { return _converged; }
  unsigned int nonlinearIterations() const { return _nonlinear_iterations; }
  const std::string & failureReason() const { return _failure_reason; }

private:
  // Material-model failures are local; reduce success/failure before doing FFTs.
  bool evaluateConstitutiveCollectively();
  void markFailed(const std::string & message);

  bool _converged = true;
  unsigned int _nonlinear_iterations = 0;
  std::string _failure_reason;
};
