# RhoCP-FFT (`rhocp_fft`)

**A research application combining MOOSE finite-element capabilities, Marlin FFT-based mechanics, and dislocation-density crystal plasticity (DDCP).**
=======
These are drop-in replacements for **stork.sh generated app**. The new app remains
`rhocp_fft`, with C++ class `RhocpFFTApp`. 

`rhocp_fft` is a standalone MOOSE application created with the MOOSE Stork application generator and extended to link against [Marlin](https://github.com/idaholab/marlin). The goal is to maintain custom DDCP constitutive models, FFT tensor operators, diagnostics, and solver improvements in one project without directly editing the upstream MOOSE or Marlin source trees for every change. Changes to Marlin core internals may still require a Marlin fork or upstream patch.



## 1. Capabilities and scope

| Component | Role | Status / requirement |
| --- | --- | --- |
| MOOSE framework | Application infrastructure, executioners, FE objects, postprocessors, output, tests | Linked through the standard MOOSE build system |
| MOOSE Solid Mechanics | Existing MOOSE constitutive/material infrastructure, including CP-FEM development | Enabled in the supplied Makefile (`SOLID_MECHANICS := yes`) |
| Marlin | FFT mechanics, tensor buffers/operators, spectral solution and XDMF tensor output | Built and linked as an application dependency |
| `RhocpFFTApp` | Registers production `rhocp_fft` objects and inherits Marlin functionality | Base application; derives from `MarlinApp` |
| `RhocpFFTTestApp` | Optional test-only object registration for development and regression tests | Must derive from `RhocpFFTApp` |
| DDCP4P2 (MOOSE) | FE material-point constitutive integration | Add/maintain as custom MOOSE material objects |
| DDCP4P2 (FFT) | Voxel-wise DDCP constitutive updates, state history and consistent/FD tangent | Port custom tensor-compute sources into this application |
| FFT scalar reductions | Volume-averaged stress, strain and dislocation-density output to CSV | Requires the custom `DDCPFFTVolumeAverage` object if used by the input |
| Adaptive failure recovery / restart | Rejected-step rollback and persistent FFT tensor checkpointing | Development target; **not assumed available** simply because MOOSE checkpoint output is enabled |

Both FEM and FFT workflows can share constitutive concepts and material parameters; their **global equilibrium solvers are different**. FEM computes equilibrium using a mesh and assembled/implicit FE residuals, whereas Marlin FFT mechanics works on a regular voxel grid using spectral operations. In a typical voxel/collocation FFT discretization, each voxel maintains one constitutive state, analogous to an FE integration point.

## 2. Software architecture

```text
MooseApp
  └── MarlinApp                     # Marlin tensor infrastructure + FFT syntax
       └── RhocpFFTApp             # Production application and DDCP extensions
            └── RhocpFFTTestApp    # Optional test objects
```

The build dependency graph is:

```text
MOOSE framework + enabled MOOSE modules
              │
              ▼
          libmarlin
              │
              ▼
        librhocp_fft
              │
              ▼
         rhocp_fft-opt
```

**Why the inheritance matters:** Marlin's `DomainAction` uses a runtime cast to `MarlinApp` and then calls its device/precision configuration methods. The running application therefore needs to be derived from `MarlinApp` (directly or indirectly). The default Stork-generated `RhocpFFTTestApp : MooseApp` is incompatible with this requirement.

The production app is declared conceptually as:

```cpp
#include "MarlinApp.h"

class RhocpFFTApp : public MarlinApp
{
public:
  static InputParameters validParams();
  RhocpFFTApp(const InputParameters & parameters);
  ~RhocpFFTApp() override;

  static void registerApps();
  static void registerAll(Factory & f, ActionFactory & af, Syntax & syntax);
};
```

`MarlinApp` registers Marlin objects and syntax in its constructor. `RhocpFFTApp` then registers **only its own** objects/actions in the normal constructor path, avoiding duplicate Marlin syntax registration. Its static `registerAll()` path may call `MarlinApp::registerAll()` for external/dynamic application registration.

## 3. Source layout

The following is the **recommended** project organization. Directories for custom DDCP objects can be created as the source is migrated; they are not all guaranteed to exist in a freshly generated Stork application.

```text
projects/
├── moose/                              # Upstream MOOSE source/build
├── marlin/                             # Upstream Marlin source/build
└── rhocp_fft/
    ├── Makefile
    ├── README.md
    ├── include/
    │   ├── base/RhocpFFTApp.h
    │   ├── materials/                  # MOOSE DDCP material headers (as added)
    │   ├── tensor_computes/            # Marlin/DDCP FFT object headers (as added)
    │   ├── postprocessors/             # FFT reductions (as added)
    │   └── utils/                      # Constitutive utilities (as added)
    ├── src/
    │   ├── main.C
    │   ├── base/RhocpFFTApp.C
    │   ├── materials/
    │   ├── tensor_computes/
    │   ├── postprocessors/
    │   └── utils/
    ├── test/
    │   ├── include/base/RhocpFFTTestApp.h
    │   ├── src/base/RhocpFFTTestApp.C
    │   └── tests/                      # Future regression tests
    └── examples/
        └── ddcp4p2/
            ├── bcc_ddcp4p2_fft_csv.i
            ├── bcc_props_54.in         # Supply actual material data
            └── bcc_slip_sys1.in       # Supply slip-system data
```

`src/main.C`, as generated by Stork, may invoke `Moose::main<RhocpFFTTestApp>(argc, argv)`. The executable name does **not** determine the app type; this template argument does. Keep that entry point if the test app has the corrected inheritance. Alternatively, use `Moose::main<RhocpFFTApp>` to start the production application directly.

## 4. Prerequisites

- Linux development environment configured for MOOSE (compiler, MPI, PETSc/libMesh and other dependencies as supplied by the MOOSE environment).
- A working MOOSE source/build at `MOOSE_DIR`.
- A working Marlin source checkout at `MARLIN_DIR`, built with the **same MOOSE checkout and compiler environment**.
- Marlin's LibTorch dependency set up as required by the specific Marlin version.
- Your custom DDCP sources and compatible input data for the DDCP examples.

Illustrative directory arrangement on the development machine:

```text
/mnt/data/projects/moose
/mnt/data/projects/marlin
/mnt/data/projects/rhocp_fft
```

Confirm that the original Marlin executable works before debugging this application:

```bash
cd /mnt/data/projects/marlin
./marlin-opt --help
```

See [MOOSE documentation](https://mooseframework.inl.gov/) and the [Marlin repository](https://github.com/idaholab/marlin) for upstream environment and dependency instructions.

## 5. Building the application

The updated `Makefile` follows the MOOSE `app.mk` build system. It enables Solid Mechanics, includes the Marlin library build first (`BUILD_EXEC := no`), and then builds `rhocp_fft` (`BUILD_EXEC := yes`). `PHASE_FIELD := no` by default; enable it if the application begins using phase-field module objects.

From the project directory:

```bash
cd /mnt/data/projects/rhocp_fft

make -j6 METHOD=opt \
  MOOSE_DIR=/mnt/data/projects/moose \
  MARLIN_DIR=/mnt/data/projects/marlin
```

The expected optimized executable is:

```text
./rhocp_fft-opt
```

For debugging, build a version containing more useful symbols:

```bash
make -j6 METHOD=dbg \
  MOOSE_DIR=/mnt/data/projects/moose \
  MARLIN_DIR=/mnt/data/projects/marlin
```

The debug executable is normally `./rhocp_fft-dbg`.

### Library checks

```bash
ldd ./rhocp_fft-opt | grep -E 'marlin|moose|torch|not found'
```

This should resolve the intended `libmarlin-opt.so`, `libmoose-opt.so`, LibTorch libraries and enabled MOOSE module libraries. No dependency should be marked `not found`. A successful `ldd` and `--help` do **not** by themselves prove that the Marlin `DomainAction` can be constructed.

## 6. First-run verification

Run these checks in order:

```bash
cd /mnt/data/projects/rhocp_fft

# Check that the executable starts.
./rhocp_fft-opt --help

# Identify the actual application class started by src/main.C.
./rhocp_fft-opt --show-type

# Verify input parsing and object construction before advancing time.
./rhocp_fft-opt \
  -i examples/ddcp4p2/bcc_ddcp4p2_fft_csv.i \
  --check-input

# Execute the DDCP FFT example once the required custom objects are linked.
./rhocp_fft-opt \
  -i examples/ddcp4p2/bcc_ddcp4p2_fft_csv.i
```

With the default Stork `main.C` and the corrected test-app hierarchy, `--show-type` is expected to identify **`RhocpFFTTestApp`**. That is normal. When `main.C` is changed to start the production app, it should identify **`RhocpFFTApp`** instead.

The DDCP example will only parse and run if its custom tensor operators/postprocessors have been compiled into `rhocp_fft` or are available through a correctly linked dependency, and its input data files are present. Passing `--help` is not sufficient.

## 7. Adding and registering new objects

Keep research-specific implementation files inside `rhocp_fft` where possible.

For example, a custom FFT constitutive compute may live at:

```text
include/tensor_computes/DDCP4P2TensorCompute.h
src/tensor_computes/DDCP4P2TensorCompute.C
```

Its implementation should register the class with the **new** application label:

```cpp
registerMooseObject("RhocpFFTApp", DDCP4P2TensorCompute);
```

The same principle applies to MOOSE materials and scalar postprocessors:

```cpp
registerMooseObject("RhocpFFTApp", DDCPStressUpdate4P2);
registerMooseObject("RhocpFFTApp", DDCPFFTVolumeAverage);
```

These examples assume that the corresponding source classes exist and have compatible interfaces. The registration statement alone does not implement or enable the object.

Experimental objects meant only for testing can be registered under `"RhocpFFTTestApp"` and enabled via `--allow-test-objects`.

**Avoid duplicate definitions:** If a modified DDCP class already exists in the Marlin library under the same C++ class name, do not compile that class a second time in `rhocp_fft`. Migrate/remove the old copy or deliberately rename/version the new one, and rebuild the relevant libraries. Duplicate source definitions and duplicate MOOSE object names are different issues; both require attention.

## 8. DDCP FFT example: important input blocks

The current research input `examples/ddcp4p2/bcc_ddcp4p2_fft_csv.i` uses the Marlin tensor workflow, approximately as follows:

```ini
[Domain]
  dim = 3
  nx = 4
  ny = 4
  nz = 4
  mesh_mode = DUMMY
  device_names = cpu
  parallel_mode = NONE
  floating_precision = DOUBLE
[]

[TensorBuffers]
  [state]
    type = PlainTensorBuffer
    value_dimensions = '148'
  []
  [state_new]
    type = PlainTensorBuffer
    value_dimensions = '148'
  []
[]

[TensorComputes]
  [Initialize]
    # Initialize F, K, and mu buffers.
  []
  [Solve]
    [ddcp]
      type = DDCP4P2TensorCompute
      buffer = stress
      tangent_operator = dstressdstrain
      F_trial = Fnew
      F_old = F
      state_old = state
      state_new = state_new
      # Other material and numerical parameters omitted here.
    []
    [root]
      [macro]
        type = DDCPAffineLoadTensor
        buffer = applied_strain
        F = F
        i = 1
        j = 1
        rate = 1e-3
      []
      [mech]
        type = FFTMechanics
        buffer = Fnew
        F = F
        K = K
        mu = mu
        constitutive_model = ddcp
        stress = stress
        tangent_operator = dstressdstrain
        applied_macroscopic_strain = applied_strain
      []
    []
  []
[]

[TensorSolver]
  type = ForwardEulerSolver
  root_compute = root
  forward_buffer = 'F state'
  forward_buffer_new = 'Fnew state_new'
  substeps = 1
[]
```

The `148`-component state is the layout used by the particular DDCP4P2 configuration represented in this research input (without climb/irradiation state extensions). It is **not** a universal state size; update the count and state mappings whenever the constitutive model changes.

In this example, `DDCP4P2TensorCompute` updates local constitutive states and stresses for the trial deformation gradient. `FFTMechanics` performs the global periodic mechanics solve. The `ForwardEulerSolver` forwards accepted `Fnew` and `state_new` into the histories used on subsequent steps. These responsibilities should remain distinct when implementing failure handling or checkpoint/recovery.

### Output

The example requests XDMF/HDF5 tensor output and MOOSE CSV postprocessors. A custom `DDCPFFTVolumeAverage` postprocessor can expose voxel-averaged quantities such as Cauchy stress, first Piola stress, engineering strain, plastic strain and mobile/immobile dislocation densities. Standard FE element-average postprocessors are not automatically replacements for FFT voxel reductions in `mesh_mode = DUMMY`.

Before comparing FFT against CP-FEM results, use consistent stress/strain measures and macroscopic boundary conditions. A prescribed deformation gradient with fixed lateral components is not the same as uniaxial stress with free lateral contraction.

## 9. Debugging

### Segmentation fault when parsing `[Domain]`

A previously observed backtrace contained:

```text
DomainAction::DomainAction(InputParameters const&)
...
Moose::main<RhocpFFTTestApp>(...)
```

The Stork-generated test app inherited directly from `MooseApp`. Marlin's `DomainAction` expects an application derived from `MarlinApp`, so this could produce a null `dynamic_cast` followed by a segmentation fault. The corrected hierarchy is:

```cpp
class RhocpFFTTestApp : public RhocpFFTApp
```

and its constructor must initialize `RhocpFFTApp(parameters)`, not `MooseApp(parameters)`. Check `src/main.C` as well: it controls which app is started. Changing the Makefile alone does not fix this inheritance error.

### GDB backtrace

```bash
cd /mnt/data/projects/rhocp_fft

gdb -q -batch \
  -ex 'set pagination off' \
  -ex run \
  -ex 'thread apply all bt 30' \
  --args ./rhocp_fft-opt \
  -i examples/ddcp4p2/bcc_ddcp4p2_fft_csv.i \
  > rhocp_crash.log 2>&1

tail -n 100 rhocp_crash.log
```

If optimized symbols are insufficient, reproduce the issue using `METHOD=dbg`.

### Typical failure categories

| Symptom | Check |
| --- | --- |
| Linker cannot locate Marlin headers/library | `MARLIN_DIR`, Makefile dependency order, shared build environment |
| `--help` works but `[Domain]` segfaults | Running app class and `RhocpFFTTestApp` inheritance |
| Unknown `DDCP4P2TensorCompute` or `DDCPFFTVolumeAverage` | Source compiled into active app; correct `registerMooseObject` label |
| Missing DDCP property/slip file | Input-relative file paths and availability of the full 54-property data |
| Constitutive Newton iteration fails | DDCP local update, tangent, timestep and buffer commit semantics |
| `--recover` fails with tensor shape mismatch | FFT tensor-buffer restoration; MOOSE FE checkpoints do not automatically serialize custom Marlin tensor history |

## 10. Development roadmap

1. **Establish a clean baseline:** validate app startup, inheritance, loading and a Marlin-only mechanics example.
2. **Migrate DDCP4P2:** add the MOOSE material, FFT tensor compute, material-point update and slip/state utilities under one maintained source tree.
3. **Verify constitutive updates:** compare a single FFT voxel with an equivalent MOOSE material-point calculation over identical deformation/time/temperature histories.
4. **Verify global FFT mechanics:** test elasticity, homogeneous plasticity, multi-voxel fields, grid refinement and MPI decomposition.
5. **Develop diagnostics:** volume averages, voxel-wise outputs, tangent FD/IFT verification and robust conservation/energy checks.
6. **Improve nonlinear robustness:** local Newton error reporting, collective failure detection, timestep cutback and consistent rollback of trial tensors.
7. **Implement durable FFT restart:** serialize/restore current and historical tensor buffers with shape, dtype, ownership/decomposition and constitutive history; verify continuous-versus-recovered solutions.
8. **Extend microstructural studies:** grain orientation maps, grain-boundary descriptors, localization and subsequent coupling to phase-field damage where applicable.

## 11. Reproducibility and source control

Track the exact MOOSE and Marlin commits used for each study, together with compiler environment, input files, parameter files and numerical tolerances. Use separate branches for constitutive modifications and FFT infrastructure updates. Add small automated regression tests before replacing a working numerical algorithm.

Upstream projects:

- [MOOSE](https://github.com/idaholab/moose)
- [Marlin](https://github.com/idaholab/marlin)

`rhocp_fft` is a separate research application; upstream MOOSE and Marlin licenses and attribution continue to apply to their respective code. Specify this application's own license separately when distributing it.