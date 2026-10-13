#pragma once

// Hybrid implicit-function-theorem tangent for one converged DDCP subincrement.
// Mathematical identity: A = P_F - P_z (R_z)^{-1} R_F.
// R_z, R_F, P_z, and P_F are CENTRAL-DIFFERENCED partial derivatives of
// NR_residual at FIXED z or F (NOT repeated full nonlinear CP solves).
// Hence this is a semi-analytical / implicit sensitivity tangent, not a
// closed-form analytical differentiation of the DDCP equations.
// With multiple subincrements, history sensitivity must be propagated and
// callers must instead use the complete-update reference tangent.

#include "RankTwoTensor.h"
#include "RankFourTensor.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace DDCP4P2IFT
{
struct LocalEvaluation
{
  std::vector<Real> residual;
  RankTwoTensor P;
};

inline void check(const LocalEvaluation & v, std::size_t n)
{
  if (v.residual.size() != n)
    throw std::runtime_error("IFT evaluator returned incorrect residual dimension");
  for (const auto r : v.residual)
    if (!std::isfinite(r))
      throw std::runtime_error("IFT evaluator returned nonfinite residual");
  for (unsigned int i=0; i<3; ++i)
    for (unsigned int j=0; j<3; ++j)
      if (!std::isfinite(v.P(i,j)))
        throw std::runtime_error("IFT evaluator returned nonfinite stress");
}

// evaluator(z,F) must use the SAME material history and parameters on
// EVERY call, and return local nonlinear R(z,F) and P(z,F).
template <typename Evaluator>
RankFourTensor compute(const std::vector<Real> & converged_z,
                       const RankTwoTensor & converged_F,
                       Evaluator && evaluator,
                       Real F_relative_step = 1.e-6,
                       Real rate_relative_step = 1.e-5)
{
  const auto n = converged_z.size();
  if (n == 0) throw std::runtime_error("IFT requires at least one local unknown");
  if (!(F_relative_step > 0.) || !(rate_relative_step > 0.))
    throw std::runtime_error("IFT perturbation steps must be positive");

  // Residual Jacobian J_ab = partial R_a / partial z_b.
  libMesh::DenseMatrix<Real> J(n, n);
  std::vector<std::array<Real,9>> P_z(n);
  for (std::size_t b=0; b<n; ++b)
  {
    const Real h = rate_relative_step *
        std::max(std::abs(converged_z[b]), Real(1.e-5));
    auto zp = converged_z, zm = converged_z;
    zp[b] += h;
    zm[b] -= h;
    const auto plus = evaluator(zp,converged_F);
    const auto minus = evaluator(zm,converged_F);
    check(plus,n); check(minus,n);
    for (std::size_t a=0; a<n; ++a)
      J(a,b) = (plus.residual[a]-minus.residual[a])/(2*h);
    for (unsigned int i=0; i<3; ++i)
      for (unsigned int j=0; j<3; ++j)
        P_z[b][3*i+j] = (plus.P(i,j)-minus.P(i,j))/(2*h);
  }

  RankFourTensor tangent;
  tangent.zero();
  libMesh::DenseVector<Real> rhs(n), dz(n);

  // F derivative: nine right-hand sides, reusing the converged local J.
  for (unsigned int k=0; k<3; ++k)
    for (unsigned int L=0; L<3; ++L)
    {
      const Real h=F_relative_step *
          std::max(std::abs(converged_F(k,L)), Real(1.));
      auto Fp=converged_F, Fm=converged_F;
      Fp(k,L) += h;
      Fm(k,L) -= h;
      if (!(Fp.det()>0.) || !(Fm.det()>0.))
        throw std::runtime_error("IFT F perturbation causes negative det(F)");
      const auto plus=evaluator(converged_z,Fp);
      const auto minus=evaluator(converged_z,Fm);
      check(plus,n); check(minus,n);
      for (std::size_t a=0; a<n; ++a)
        rhs(a)=-(plus.residual[a]-minus.residual[a])/(2*h);
      // J dz/dF = - R_F; LU is used, not an explicit inverse.
      J.lu_solve(rhs,dz);

      for (unsigned int i=0; i<3; ++i)
        for (unsigned int jj=0; jj<3; ++jj)
        {
          Real component=(plus.P(i,jj)-minus.P(i,jj))/(2*h);
          for (std::size_t b=0; b<n; ++b)
            component += P_z[b][3*i+jj]*dz(b);
          if (!std::isfinite(component))
            throw std::runtime_error("IFT produced nonfinite tangent component");
          tangent(i,jj,k,L) = component;
        }
    }
  return tangent;
}
} // namespace DDCP4P2IFT
