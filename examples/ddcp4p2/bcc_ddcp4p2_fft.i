# EARLY DEVELOPMENT TEST INPUT. Requires Marlin source additions and
# a complete 54-property bcc_props_54.in file (NOT supplied).
# This input is for a single-orientation, periodic 8^3 CPU trial.
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
    value_dimensions = '148' # 52 + 4*24; no irradiation or climb
  []
  [state_new]
    type = PlainTensorBuffer
    value_dimensions = '148'
  []
[]

[TensorComputes]
  [Initialize]
    [Finit]
      type = RankTwoIdentity
      buffer = F
    []
    # FFTMechanics currently requires K and mu buffers, even if it
    # uses the constitutive-model tangent for its actual solve.
    [K]
      type = ConstantTensor
      buffer = K
      real = 1.0
    []
    [mu]
      type = ConstantTensor
      buffer = mu
      real = 1.0
    []
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
      propsFile = bcc_props_54.in
      slipSysFile = bcc_slip_sys1.in
      euler_degrees = '0 0 0'
      temperature = 300
      g = 1
      gp = 1
      climbmodel = false
      irad_defects = false
      decomposition = 0
      use_implicit_tangent = true
      ift_rate_step = 1e-5
      fd_step = 1e-6
      verify_tangent = true
      verify_tolerance = 1e-3
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
        l_tol = 1e-3
        nl_rel_tol = 1e-4
        nl_abs_tol = 1e-7
        nl_max_its = 30
      []
    []
  []
  [Postprocess]
    [displacements]
      type = ComputeDisplacements
      buffer = disp
      F = F
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



[TensorOutputs]
  [vtk]
    type = XDMFTensorOutput
    buffer = 'disp F stress state'
    output_mode = 'OVERSIZED_NODAL CELL CELL CELL'
    enable_hdf5 = true
    execute_on = 'TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  num_steps = 1000
  dt = 0.005
[]
