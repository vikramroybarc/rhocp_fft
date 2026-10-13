# RhocpFFTApp: MOOSE + Marlin build changes

These are drop-in replacements for your **stork.sh generated app**. The new app remains
`rhocp_fft`, with C++ class `RhocpFFTApp`. None of your other code is overwritten.

## Placement

- `Makefile` -> `<app root>/Makefile`
- `RhocpFFTApp.C` -> `<app root>/src/base/RhocpFFTApp.C`
- `RhocpFFTApp.h` -> `<app root>/include/base/RhocpFFTApp.h`

The third file was not uploaded, but it **must be updated**: change the parent class
from `MooseApp` to `MarlinApp`. The attached header is compatible with the attached C++.

## Build

```bash
cd /mnt/data/projects/rhocp_fft
make -j6 METHOD=opt MOOSE_DIR=/mnt/data/projects/moose MARLIN_DIR=/mnt/data/projects/marlin
./rhocp_fft-opt --help
```

The Makefile enables `SOLID_MECHANICS := yes`. To use MOOSE phase-field objects,
change `PHASE_FIELD := no` to `PHASE_FIELD := yes` and rebuild.

The Marlin library is included using **app.mk with BUILD_EXEC=no**, then the
Rhocp FFT app using **app.mk with BUILD_EXEC=yes**. This accumulates Marlin's
header directories, dependencies, and compiled library in the MOOSE build system.
No Marlin source copy is needed.

## Add new objects

Use `registerMooseObject("RhocpFFTApp", DDCP4P2TensorCompute);` for custom FFT computes.
Use `registerMooseObject("RhocpFFTApp", DDCPStressUpdate4P2);` for custom MOOSE materials,
provided their respective headers and sources are placed under the app's `include/` and `src/`.

**Do not compile the same DDCP class in both Marlin and RhocpFFTApp**.
When migrating, delete or move the DDCP object source out of the Marlin library build to
avoid duplicate symbols and conflicting factory registration. Marlin's own stock
FFTMechanics, DomainAction, tensor classes and MOOSE framework facilities remain available.

**Marlin sample caveat:** Its `MacroscopicShearTensor` example compute lives under
`marlin/test/src/`, not `marlin/src/`. If you port the test input `degeus_mechanics/mech.i`,
register a local shear-loading tensor compute under `RhocpFFTApp` or otherwise use a
production-registered alternative.

**Checkpoint caveat:** Extending `MarlinApp` does not fix the missing FFT tensor checkpoint
serialization/restore problem discussed earlier. That requires additional implementation.

**Validation:** These files are based on the uploaded stork Makefile and `RhocpFFTApp.C`,
and publicly inspected MOOSE/Marlin sources. They have not been compiled against your
actual MOOSE/Marlin checkout in this environment.
