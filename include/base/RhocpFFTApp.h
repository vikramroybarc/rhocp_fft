#pragma once

#include "MarlinApp.h"

/**
 * Research application combining MOOSE finite elements and Marlin FFT tensors.
 * Must derive from MarlinApp: DomainAction expects that application type.
 */
class RhocpFFTApp : public MarlinApp
{
public:
  static InputParameters validParams();

  RhocpFFTApp(const InputParameters & parameters);
  ~RhocpFFTApp() override;

  static void registerApps();
  static void registerAll(Factory & f, ActionFactory & af, Syntax & syntax);
};
