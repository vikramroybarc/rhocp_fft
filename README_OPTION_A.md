# Option A: Jaumann-assumed Cauchy tangent -> finite-strain FFT dP/dF

**Target:** `https://github.com/vikramroybarc/rhocp_fft`, `main` as inspected on 2026-10-15.

This package adds a selectable **approximate** FFT tangent without modifying the DDCP4P2 constitutive stress, slip, hardening, and subincrement equations. The existing `ift` and complete-update `fd` tangent paths remain available. The default `legacy` setting retains existing behavior.

**Important scientific qualification:** the original `C_local = (I_s + C_eff:B)^(-1):C_eff` is not demonstrably a Jaumann Cauchy tangent from the original constitutive update. This package **assumes** that interpretation to define a useful candidate tangent. It is **not** an exact algorithmic derivative of the original multiplicative finite-strain DDCP model. Do not label or use it as 'verified consistent' until comparing against full-update FD on the states of interest.

## Formula implemented

Given converged `F`, Cauchy stress `sigma`, and local modulus `C_local`, assume

    sigma_dot - w*sigma + sigma*w = C_local : d.

For each unit component perturbation `dF(k,L)=1`, form

    G      = inverse(F)
    J      = det(F)
    tau    = J*sigma
    P      = tau*transpose(G)
    l      = dF*G
    d      = 0.5*(l+transpose(l))
    w      = 0.5*(l-transpose(l))
    dsigma = C_local:d + w*sigma - sigma*w
    dtau   = J*dsigma + trace(l)*tau
    dP     = dtau*transpose(G) - P*transpose(dF)*transpose(G)
    A(i,J,k,L) = dP(i,J)

`A` retains all 81 entries, including geometric and objective spin terms. No minor symmetries are assumed. This direct directional formulation avoids risky index algebra in the Jaumann/Truesdell conversion.

Marlin receives `K(J,i,k,L) = A(i,J,k,L)` following the adapter's existing einsum convention; **this permutation is not changed**.

## Changed/added source files

| Path | Change |
| --- | --- |
| `include/tensor_computes/DDCP4P2OptionATangent.h` | **new** full directional pull-back, with finite checks |
| `include/tensor_computes/DDCP4P2Point.h` | input flag, returned tangent tensor and availability flag |
| `src/tensor_computes/DDCP4P2Point.C` | compute Option A after completed update, only if requested |
| `src/tensor_computes/DDCP4P2PointModified.C` | scale Option A with demonstration stress scaling |
| `include/tensor_computes/DDCPTensorCompute.h` | tangent selection member |
| `src/tensor_computes/DDCPTensorCompute.C` | `tangent_method` switch; unchanged default and FD fallback |

## Apply to a local clone

```bash
git clone https://github.com/vikramroybarc/rhocp_fft.git
# Extract DDCP4P2_OptionA.zip somewhere, e.g. /tmp/DDCP4P2_OptionA
python3 /tmp/DDCP4P2_OptionA/apply_option_a.py ./rhocp_fft --check
# To generate COMPLETE modified source files without changing your checkout:
python3 /tmp/DDCP4P2_OptionA/apply_option_a.py ./rhocp_fft \
  --output-dir /tmp/ddcp_option_a_modified
# OR to install into the source checkout, backing up five edited files:
python3 /tmp/DDCP4P2_OptionA/apply_option_a.py ./rhocp_fft
cd rhocp_fft
git diff --check
git diff -- src/tensor_computes/DDCP4P2Point.C \
  src/tensor_computes/DDCPTensorCompute.C
make -j6 METHOD=opt
```

The installer is intentionally strict: it refuses missing/changed source anchors or an already patched checkout. It does not pull anything from the network and does not push commits. It is designed for the repository files inspected on 2026-10-15.

## Select Option A in input file

Within your existing `[TensorComputes]/[Solve]/[ddcp]` block:

```ini
    [ddcp]
      type = DDCPTensorCompute
      # ... retain original F/state buffers, propsFile, slipSysFile, etc.
      tangent_method = option_a
      verify_tangent = false
    []
```

**Actual registered compute name is `DDCPTensorCompute`.** Some older example files still show `DDCP4P2TensorCompute`; use the registered name in your build.

Available values:

- `tangent_method = legacy`: previous behavior governed by `use_implicit_tangent`, default.
- `tangent_method = ift`: attempt local IFT, full-update FD fallback.
- `tangent_method = fd`: complete-update central differences only.
- `tangent_method = option_a`: local Jaumann-assumed pull-back, no IFT/FD fallback.

`use_implicit_tangent` remains for backward compatibility but is ignored when `tangent_method` is not `legacy`.

## Validation against complete constitutive update

For a diagnostic test on the first local voxel, use:

```ini
      tangent_method = option_a
      verify_tangent = true
      verify_reference_fd = true
      verify_tangent_warn_only = true
      verify_tolerance = 1e-3
      verify_max_reports = 10
      fd_step = 1e-6
```

The code reports `selected_error` and `reference_full_FD_error`. **A failing Option A comparison is expected in some finite elastic/plastic states and must not be disregarded.** `verify_tangent_warn_only=true` is for diagnostics, not validation. Change to `false` when conducting strict pass/fail checks. The `option_a` branch does not silently substitute an FD tangent if it fails a verification check.

Separately test an elastic single crystal, plastically flowing single crystal, hardening, strong finite shear/rotation, subincremented steps, and `decomposition = 1`. Compare Newton iteration counts and final stress against `tangent_method = fd`.

## Unit test without MOOSE

From the extracted package root:

```bash
g++ -std=c++17 -O2 -Itests/mock -Iinclude/tensor_computes \
    tests/test_option_a.cpp -o /tmp/test_option_a
/tmp/test_option_a
```

The test compiles the **actual new header** against minimal RankTwo/RankFour mocks and checks hydrostatic finite differences, reference isotropic elasticity, and rigid-rotation frame indifference. This is not a build or run of the full Marlin application, which requires your MOOSE/Marlin environment and material property files.

## Commit/recovery

No state-variable layout or constitutive integration algorithm is changed. `tangent_method` changes the global Newton search direction, not the committed state update. Compare newly generated checkpoints and do not use numerical stability under Option A as proof that its Jacobian is consistent.
