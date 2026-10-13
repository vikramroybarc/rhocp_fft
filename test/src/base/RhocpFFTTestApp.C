//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#include "RhocpFFTTestApp.h"
#include "RhocpFFTApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
RhocpFFTTestApp::validParams()
{
  InputParameters params = RhocpFFTApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

RhocpFFTTestApp::RhocpFFTTestApp(const InputParameters & parameters) : MooseApp(parameters)
{
  RhocpFFTTestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

RhocpFFTTestApp::~RhocpFFTTestApp() {}

void
RhocpFFTTestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  RhocpFFTApp::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"RhocpFFTTestApp"});
    Registry::registerActionsTo(af, {"RhocpFFTTestApp"});
  }
}

void
RhocpFFTTestApp::registerApps()
{
  registerApp(RhocpFFTApp);
  registerApp(RhocpFFTTestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
RhocpFFTTestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  RhocpFFTTestApp::registerAll(f, af, s);
}
extern "C" void
RhocpFFTTestApp__registerApps()
{
  RhocpFFTTestApp::registerApps();
}
