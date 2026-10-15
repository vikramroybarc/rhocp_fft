#include "DDCP4P2PointModified.h"

#include <cmath>
#include <stdexcept>

DDCP4P2PointModified::DDCP4P2PointModified(const DDCP4P2Data & data, Real scale)
  : DDCP4P2Point(data), _stress_scale(scale)
{
  if (!std::isfinite(_stress_scale) || _stress_scale <= 0.)
    throw std::invalid_argument(
        "DDCP4P2PointModified: model_stress_scale must be positive and finite");
}

DDCP4P2Result
DDCP4P2PointModified::integrate(const DDCP4P2Input & input)
{
  // Base integration reloads committed input before every evaluation,
  // including the perturbations used by DDCP4P2Tangent finite differences.
  DDCP4P2Result out = DDCP4P2Point::integrate(input);
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      out.P(i,j) *= _stress_scale;
      out.sigma(i,j) *= _stress_scale;
      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int l = 0; l < 3; ++l)
        {
          out.A_implicit(i,j,k,l) *= _stress_scale;
          out.C_local(i,j,k,l) *= _stress_scale;
        }
    }
  return out;
}
