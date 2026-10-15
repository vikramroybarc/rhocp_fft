#pragma once

// Option A for rhocp_fft: an APPROXIMATE finite-strain FFT tangent.
//
// ASSUMPTION: C_alg relates the Jaumann rate of Cauchy stress to d = sym(l):
//       sigma_dot - w*sigma + sigma*w = C_alg : d.
// This has NOT been established for the original DDCP4P2 Eq.(31) modulus;
// selecting this approximation must be explicit, never the default.
//
// Given F and sigma from the converged point update, compute the full
// 81-component A(i,J,k,L) = dP(i,J)/dF(k,L), where P=det(F)*sigma*F^(-T).
// For dF, l = dF*F^(-1), d = sym(l), w = skew(l),
//   d_sigma = C_alg:d + w*sigma - sigma*w,
//   d_tau = J*d_sigma + trace(l)*tau,
//   dP = d_tau*F^(-T) - P*dF^T*F^(-T).
// This includes spin, volumetric and geometric terms. Do NOT enforce minor
// symmetry on A. No additional constitutive integrations are performed.

#include "RankTwoTensor.h"
#include "RankFourTensor.h"
#include <cmath>
#include <stdexcept>

namespace DDCP4P2OptionATangent
{
inline RankFourTensor
fromJaumannCauchy(const RankTwoTensor & F,
                   const RankTwoTensor & sigma,
                   const RankFourTensor & C_alg)
{
  const Real J = F.det();
  if (!std::isfinite(J) || J <= 0.)
    throw std::runtime_error("Option A: nonpositive/nonfinite det(F)");

  const RankTwoTensor Finv = F.inverse();
  const RankTwoTensor FinvT = Finv.transpose();
  const RankTwoTensor tau = J * sigma;
  const RankTwoTensor P = tau * FinvT;
  RankFourTensor A;
  A.zero();

  for (unsigned int k = 0; k < 3; ++k)
    for (unsigned int L = 0; L < 3; ++L)
    {
      RankTwoTensor dF;
      dF.zero();
      dF(k,L) = 1.;
      const RankTwoTensor dl = dF * Finv;
      const RankTwoTensor d = 0.5 * (dl + dl.transpose());
      const RankTwoTensor w = 0.5 * (dl - dl.transpose());

      RankTwoTensor dsigma;
      dsigma.zero();
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          for (unsigned int a = 0; a < 3; ++a)
            for (unsigned int b = 0; b < 3; ++b)
              dsigma(i,j) += C_alg(i,j,a,b) * d(a,b);

      dsigma += w * sigma - sigma * w;
      const Real trace_dl = dl(0,0) + dl(1,1) + dl(2,2);
      const RankTwoTensor dtau = J * dsigma + trace_dl * tau;
      const RankTwoTensor dP = dtau * FinvT - P * dF.transpose() * FinvT;

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int JJ = 0; JJ < 3; ++JJ)
        {
          const Real value = dP(i,JJ);
          if (!std::isfinite(value))
            throw std::runtime_error("Option A: nonfinite transformed dP/dF");
          A(i,JJ,k,L) = value;
        }
    }
  return A;
}
} // namespace DDCP4P2OptionATangent
