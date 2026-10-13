#include "RhocpFFTTestApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"
#include "Registry.h"

InputParameters
RhocpFFTTestApp::validParams()
{
  return RhocpFFTApp::validParams();
}

RhocpFFTTestApp::RhocpFFTTestApp(const InputParameters & parameters)
  : RhocpFFTApp(parameters)
{
  // The RhocpFFTApp constructor already registers Marlin and custom objects.
  // Register *only* test objects here to avoid registering Marlin's syntax twice.
  if (getParam<bool>("allow_test_objects"))
  {
    Registry::registerObjectsTo(_factory, {"RhocpFFTTestApp"});
    Registry::registerActionsTo(_action_factory, {"RhocpFFTTestApp"});
  }
}

RhocpFFTTestApp::~RhocpFFTTestApp() {}

void
RhocpFFTTestApp::registerAll(Factory & f,
                              ActionFactory & af,
                              Syntax & s,
                              bool use_test_objs)
{
  // This is the static/dynamic-loading registration path, not the constructor path.
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
  RhocpFFTApp::registerApps();
  registerApp(RhocpFFTTestApp);
}

/***************************************************************************************************
 ******************************** Dynamic Library Entry Points ***********************************
 **************************************************************************************************/
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
