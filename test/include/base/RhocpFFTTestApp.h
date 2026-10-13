#pragma once

#include "RhocpFFTApp.h"

/**
 * Test application for RhocpFFTApp.
 *
 * IMPORTANT: inherit from RhocpFFTApp, not MooseApp.
 * Marlin's DomainAction expects a MarlinApp-derived runtime application.
 */
class RhocpFFTTestApp : public RhocpFFTApp
{
public:
  static InputParameters validParams();

  RhocpFFTTestApp(const InputParameters & parameters);
  ~RhocpFFTTestApp() override;

  static void registerApps();
  static void registerAll(Factory & f,
                          ActionFactory & af,
                          Syntax & s,
                          bool use_test_objs = false);
};
