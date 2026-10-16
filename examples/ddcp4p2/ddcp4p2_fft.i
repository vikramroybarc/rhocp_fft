# EARLY DEVELOPMENT TEST INPUT. Requires Marlin source additions and
# DDCPFFTVolumeAverage postprocessor compiled in your existing application.
# a complete 54-property bcc_props_54.in file (NOT supplied).
# This input is for a two-grain 4x4x4 periodic CPU trial.
# Checkpoint created by this input is required for --recover: old
# MOOSE-only checkpoints cannot be used to reconstruct DDCP state.
[Domain]
  dim = 3
  nx = 4
  ny = 4
  nz = 4
  mesh_mode = DUMMY
  device_names = cpu
  parallel_mode = FFT_AUTO
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
      type = DDCPTensorCompute
      material_model = DDCP4P2Point
      buffer = stress
      tangent_operator = dstressdstrain
      F_trial = Fnew
      F_old = F
      state_old = state
      state_new = state_new
      propsFile = bcc_props_54.in
      slipSysFile = bcc_slip_sys1.in
      orientation_file = orientation.in
      temperature = 300
      g = 1
      gp = 1
      climbmodel = false
      irad_defects = false
      decomposition = 0
      tangent_method = option_a
      verify_tangent = true
      verify_reference_fd = true
      # verify_tangent_warn_only = true
      # verify_tolerance = 1e-2
      fd_step = 1e-6
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
        type = DDCPFFTMechanics
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

# Persistent Marlin tensor checkpoint bridge. MUST be present in the
# original run from which --recover will resume.
[Problem]
  type = RhoCPFFTTensorProblem
  # constitutive_compute = ddcp
  checkpoint_buffers = 'F state K mu'
  recovery_seed_from = 'F state'
  recovery_seed_to = 'Fnew state_new'
[]

[TensorSolver]
  type = DDCPFFTForwardSolver
  mechanics_compute = mech
  rollback_buffers = 'F state Fnew state_new stress dstressdstrain applied_strain'
  root_compute = root
  forward_buffer = 'F state'
  forward_buffer_new = 'Fnew state_new'
  substeps = 1
[]



[TensorOutputs]
  [vtk]
    type = XDMFTensorOutput
    buffer = 'F stress state'
    output_mode = 'CELL CELL CELL'
    enable_hdf5 = true
    execute_on = 'TIMESTEP_END'
  []
[]

[Postprocessors]
  [sigma_yy]
    type = DDCPFFTVolumeAverage
    quantity = sigma_yy
    execute_on = TIMESTEP_END
  []
  [P_yy]
    type = DDCPFFTVolumeAverage
    quantity = P_yy
    execute_on = TIMESTEP_END
  []
  [eng_strain_yy]
    type = DDCPFFTVolumeAverage
    quantity = engineering_strain_yy
    execute_on = TIMESTEP_END
  []
  [Ep_yy]
    type = DDCPFFTVolumeAverage
    quantity = Ep_yy
    execute_on = TIMESTEP_END
  []
  [Ep_eff]
    type = DDCPFFTVolumeAverage
    quantity = Ep_eff
    execute_on = TIMESTEP_END
  []
  [rho_m_avg]
    type = DDCPFFTVolumeAverage
    quantity = rho_m_avg
    execute_on = TIMESTEP_END
  []
  [rho_i_avg]
    type = DDCPFFTVolumeAverage
    quantity = rho_i_avg
    execute_on = TIMESTEP_END
  []
[]

[Outputs]
  file_base = out_ddtfft
  [checkpoint]
    type = Checkpoint
    num_files = 3
    time_step_interval = 5
    execute_on = TIMESTEP_END
  []
  [csv]
    type = CSV
    execute_on = TIMESTEP_END
  []
[]

[Executioner]
  type = Transient
  end_time = 10.0
  dtmin = 1e-6
  dtmax = 0.1
  abort_on_solve_fail = false
  [TimeStepper]
    type = DDCPFFTCutbackDT
    dt = 0.1
    growth_factor = 1.2
    cutback_factor_at_failure = 0.5
  []
[]
