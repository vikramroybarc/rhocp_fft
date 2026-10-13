# Educational Marlin finite-strain FFT example.
# The macroscopic shear compute is a test helper in public Marlin;
# link/register it before expecting this to run in a production binary.
[Domain]
  dim = 2
  nx = 48
  ny = 48
  xmax = ${fparse 2*pi}
  ymax = ${fparse 2*pi}
  mesh_mode = DUMMY
[]

[TensorComputes]
  [Initialize]
    [phase_map]
      type = ParsedCompute
      buffer = phase
      expression = '(0.5+0.5*cos(x))*(0.5+0.5*cos(y))'
      extra_symbols = true
    []
    [bulk_modulus]
      type = ParsedCompute
      buffer = K
      inputs = phase
      expression = 'Km+(Ki-Km)*phase'
      constant_names = 'Km Ki'
      constant_expressions = '1.2 9.6'
    []
    [shear_modulus]
      type = ParsedCompute
      buffer = mu
      inputs = phase
      expression = 'mum+(mui-mum)*phase'
      constant_names = 'mum mui'
      constant_expressions = '0.6 4.8'
    []
    [initial_F]
      type = RankTwoIdentity
      buffer = F
    []
  []
  [Solve]
    [material]
      type = HyperElasticIsotropic
      buffer = stress
      F = Fnew
      K = K
      mu = mu
      tangent_operator = dstressdstrain
    []
    [root]
      [shear_increment]
        type = MacroscopicShearTensor
        buffer = applied_strain
        F = F
      []
      [equilibrium]
        type = FFTMechanics
        buffer = Fnew
        F = F
        K = K
        mu = mu
        constitutive_model = material
        stress = stress
        tangent_operator = dstressdstrain
        applied_macroscopic_strain = applied_strain
        l_tol = 1e-5
        l_max_its = 150
        nl_rel_tol = 1e-4
        nl_abs_tol = 1e-6
        nl_max_its = 80
        verbose = true
      []
    []
  []
  [Postprocess]
    [reconstruct_u]
      type = ComputeDisplacements
      buffer = disp
      F = F
    []
    [scalar_stress_indicator]
      type = ComputeVonMisesStress
      buffer = sV
      stress = stress
    []
  []
[]

[TensorSolver]
  type = ForwardEulerSolver
  root_compute = root
  forward_buffer = F
  forward_buffer_new = Fnew
  substeps = 2
[]

[TensorOutputs]
  [fields]
    type = XDMFTensorOutput
    buffer = 'disp sV F phase'
    output_mode = 'OVERSIZED_NODAL CELL CELL NODE'
    enable_hdf5 = true
    execute_on = 'TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  num_steps = 8
  dt = 0.005
[]
