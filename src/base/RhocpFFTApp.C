#include "RhocpFFTApp.h"
#include "AppFactory.h"
#include "Moose.h"
#include "Registry.h"

InputParameters
RhocpFFTApp::validParams()
{
  // Retain Marlin-specific application parameters (such as compute_device).
  InputParameters params = MarlinApp::validParams();
  return params;
}

RhocpFFTApp::RhocpFFTApp(const InputParameters & parameters) : MarlinApp(parameters)
{
  // MarlinApp's constructor registers the MOOSE/Marlin objects and syntax.
  // Register ONLY this application's own objects here to avoid duplicate syntax.
  Registry::registerObjectsTo(_factory, {"RhocpFFTApp"});
  Registry::registerActionsTo(_action_factory, {"RhocpFFTApp"});
}

RhocpFFTApp::~RhocpFFTApp() {}

void
RhocpFFTApp::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  // Static registration path, used when dynamically loading this application.
  MarlinApp::registerAll(f, af, syntax);
  Registry::registerObjectsTo(f, {"RhocpFFTApp"});
  Registry::registerActionsTo(af, {"RhocpFFTApp"});
}

void
RhocpFFTApp::registerApps()
{
  MarlinApp::registerApps();
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
