#include "RhocpFFTApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
RhocpFFTApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

RhocpFFTApp::RhocpFFTApp(const InputParameters & parameters) : MooseApp(parameters)
{
  RhocpFFTApp::registerAll(_factory, _action_factory, _syntax);
}

RhocpFFTApp::~RhocpFFTApp() {}

void
RhocpFFTApp::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  ModulesApp::registerAllObjects<RhocpFFTApp>(f, af, syntax);
  Registry::registerObjectsTo(f, {"RhocpFFTApp"});
  Registry::registerActionsTo(af, {"RhocpFFTApp"});

  /* register custom execute flags, action syntax, etc. here */
}

void
RhocpFFTApp::registerApps()
{
  registerApp(RhocpFFTApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
RhocpFFTApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  RhocpFFTApp::registerAll(f, af, s);
}
extern "C" void
RhocpFFTApp__registerApps()
{
  RhocpFFTApp::registerApps();
}
