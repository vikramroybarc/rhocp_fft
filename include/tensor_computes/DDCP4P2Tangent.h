#pragma once

// The discrete material update is P_{n+1} = P(F_{n+1}; F_n, q_n, dt,...).
// For the full nonlinear DDCP update, differentiate THIS MAP (not Eq.31's
// approximate spatial rate modulus) while holding all committed inputs fixed.
// The current implementation uses a central finite difference; it is a
// numerically consistent algorithmic tangent, NOT an analytic derivative.

#include "DDCP4P2Point.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace DDCP4P2Tangent
{

inline bool
finiteMatrix(const RankTwoTensor & P)
{
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      if (!std::isfinite(P(i,j)))
        return false;
  return true;
}

// A(i,J,k,L) = d P(i,J) / d F(k,L).
// A new full constitutive integration is performed for every perturbation.
// All perturbed calls start from the SAME old state (not a previous trial).
template <typename Material>
RankFourTensor
centralDifferencePK1(const DDCP4P2Input & committed_input,
                     Material & material,
                     Real relative_step = 1.e-6,
                     unsigned int max_halvings = 8)
{
  if (!std::isfinite(relative_step) || relative_step <= 0.)
    throw std::invalid_argument("DDCP4P2: fd_step must be positive and finite");

  RankFourTensor A;
  A.zero();

  for (unsigned int k = 0; k < 3; ++k)
    for (unsigned int L = 0; L < 3; ++L)
    {
      const Real f0 = committed_input.F_new(k,L);
      Real h = relative_step * std::max(Real(1.), std::abs(f0));
      bool succeeded = false;
      std::string last_error = "";

      for (unsigned int attempt = 0; attempt <= max_halvings; ++attempt)
      {
        DDCP4P2Input plus = committed_input;
        DDCP4P2Input minus = committed_input;
        plus.F_new(k,L) += h;
        minus.F_new(k,L) -= h;

        if (!(plus.F_new.det() > 0.) || !(minus.F_new.det() > 0.))
        {
          last_error = "invalid determinant at perturbed F";
          h *= 0.5;
          continue;
        }
        try
        {
          // Material.integrate reinitializes itself from state_old on each call.
          const RankTwoTensor Pplus = material.integrate(plus).P;
          const RankTwoTensor Pminus = material.integrate(minus).P;
          if (!finiteMatrix(Pplus) || !finiteMatrix(Pminus))
            throw std::runtime_error("nonfinite perturbed first Piola stress");
          for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int J = 0; J < 3; ++J)
              A(i,J,k,L) = (Pplus(i,J) - Pminus(i,J)) / (2.*h);
          succeeded = true;
          break;
        }
        catch (const std::exception & e)
        {
          last_error = e.what();
          h *= 0.5;
        }
      }
      if (!succeeded)
        throw std::runtime_error("DDCP4P2 dP/dF failed for (k,L) = (" +
                                 std::to_string(k) + "," + std::to_string(L) +
                                 "): " + last_error);
    }
  return A;
}

// INDEPENDENT DIRECTIONAL CHECK: compare the computed 81-component A with a
// non-coordinate perturbation, again restarting from exactly the same q_n.
// A small error is evidence of numerical consistency at the current state;
// it is not a proof of smoothness across all loading paths.
template <typename Material>
Real
directionalRelativeError(const DDCP4P2Input & base,
                         Material & material,
                         const RankFourTensor & A,
                         Real h = 2.e-6)
{
  RankTwoTensor direction;
  direction.zero();
  // Mixed normal/shear direction, including non-symmetric components.
  const Real v[3][3] = {{0.41, -0.25, 0.17},
                        {0.32, -0.13, 0.29},
                        {-0.21, 0.37, 0.44}};
  for (unsigned int k = 0; k < 3; ++k)
    for (unsigned int L = 0; L < 3; ++L)
      direction(k,L) = v[k][L];

  DDCP4P2Input plus = base;
  DDCP4P2Input minus = base;
  for (unsigned int k = 0; k < 3; ++k)
    for (unsigned int L = 0; L < 3; ++L)
    {
      plus.F_new(k,L) += h * direction(k,L);
      minus.F_new(k,L) -= h * direction(k,L);
    }

  if (!(plus.F_new.det() > 0.) || !(minus.F_new.det() > 0.))
    throw std::runtime_error("directional check: nonpositive det(F)");

  const RankTwoTensor Pplus = material.integrate(plus).P;
  const RankTwoTensor Pminus = material.integrate(minus).P;
  if (!finiteMatrix(Pplus) || !finiteMatrix(Pminus))
    throw std::runtime_error("directional check: nonfinite stress");

  Real diff2 = 0.;
  Real ref2 = 0.;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int J = 0; J < 3; ++J)
    {
      Real predicted = 0.;
      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int L = 0; L < 3; ++L)
          predicted += A(i,J,k,L) * direction(k,L);
      const Real actual = (Pplus(i,J) - Pminus(i,J)) / (2.*h);
      const Real delta = predicted - actual;
      diff2 += delta * delta;
      ref2 += actual * actual;
    }
  // Scaling is in stress-per-unit-F, e.g. MPa. The absolute floor prevents
  // a meaningless large relative error near zero response.
  return std::sqrt(diff2) / std::max(std::sqrt(ref2), Real(1.e-10));
}

// This identity is exact for P = det(F) sigma(F) F^(-T) only if dsigma_dF
// is the FULL discrete derivative (including objective rotation and state
// evolution). The original Eq.31 C_local must not be substituted here.
inline RankFourTensor
fromFullCauchyDerivative(const RankTwoTensor & F,
                        const RankTwoTensor & sigma,
                        const RankFourTensor & dsigma_dF)
{
  const Real J = F.det();
  if (!(J > 0.))
    throw std::runtime_error("DDCP4P2 dP/dF requires positive det(F)");
  const RankTwoTensor G = F.inverse();
  RankFourTensor A;
  A.zero();
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int JJ = 0; JJ < 3; ++JJ)
      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int L = 0; L < 3; ++L)
        {
          Real PoverJ = 0.;
          Real dSigma = 0.;
          Real sigmaG = 0.;
          for (unsigned int a = 0; a < 3; ++a)
          {
            PoverJ += sigma(i,a)*G(JJ,a);
            dSigma += dsigma_dF(i,a,k,L)*G(JJ,a);
            sigmaG += sigma(i,a)*G(L,a);
          }
          A(i,JJ,k,L) = J*(dSigma + PoverJ*G(L,k) - G(JJ,k)*sigmaG);
        }
  return A;
}

} // namespace DDCP4P2Tangent
