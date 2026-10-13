// Optional: launch the production application instead of the Stork-generated test app.
// Use only if you do NOT need the test-objects-enabled executable as default.
#include "RhocpFFTApp.h"
#include "MooseMain.h"

int
main(int argc, char * argv[])
{
  return Moose::main<RhocpFFTApp>(argc, argv);
}
