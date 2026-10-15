#pragma once

#include "DDCP4P2Point.h"

/**
 * EXAMPLE ONLY: reports scaled P, sigma, and implicit tangents from the
 * original DDCP4P2 model. Does not recalibrate or change DDCP kinetics.
 * Replace this class with a genuinely different constitutive implementation
 * before drawing physical conclusions from it.
 */
class DDCP4P2PointModified final : public DDCP4P2Point
{
public:
  DDCP4P2PointModified(const DDCP4P2Data & data, Real stress_scale);
  DDCP4P2Result integrate(const DDCP4P2Input & input) override;

private:
  const Real _stress_scale;
};
