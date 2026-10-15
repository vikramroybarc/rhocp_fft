#include "DDCP4P2Point.h"
#include "DDCP4P2ImplicitTangent.h"
#include "DDCP4P2OptionATangent.h"
#include "MatrixTools.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
  template <typename... Args>
  [[noreturn]] void ddcpError(const Args & ... args)
  {
    std::ostringstream os;
    (os << ... << args);
    throw std::runtime_error(os.str());
  }
}

unsigned int DDCP4P2Point::stateCount(unsigned int ns, bool irradiation, bool climb)
{
  return 52 + 4*ns + (irradiation ? 2*ns : 0) + (climb ? ns + 3 : 0);
}

DDCP4P2Point::DDCP4P2Point(const DDCP4P2Data & data)
  : _num_slip_sys(static_cast<unsigned int>(data.normals.size())),
    _num_state_vars(stateCount(_num_slip_sys, data.irradiation, data.climb)),
    _num_props(static_cast<unsigned int>(data.props.size())),
    _tol(data.tolerance),
    _climbmodel(data.climb),
    _irad_defects(data.irradiation),
    _deltaH_eV(data.deltaH_eV),
    _sliplaw(data.slip_law),
    _crsslaw(data.crss_law),
    _decomposition(data.decomposition ? Decomposition::voldev : Decomposition::none),
    _data(data)
{
  if (data.props.size() < 54)
    ddcpError("DDCP4P2 needs 54 properties, supplied ", data.props.size());
  if (_num_slip_sys == 0 || data.directions.size() != _num_slip_sys)
    ddcpError("Invalid slip system list");
  _state_var[0].resize(_num_state_vars, 0.);
  _state_var_old[0].resize(_num_state_vars, 0.);
  _properties[0] = data.props;
  _properties_old[0] = data.props;
  _y[0].assign(3, std::vector<Real>(_num_slip_sys, 0.));
  _z[0].assign(3, std::vector<Real>(_num_slip_sys, 0.));
  _y_old[0] = _y[0];
  _z_old[0] = _z[0];
}

DDCP4P2Result DDCP4P2Point::integrate(const DDCP4P2Input & input)
{
  if (input.dt <= 0.)
    ddcpError("Positive dt required");
  _dt = input.dt;
  _temp[0] = input.temperature;
  _g[0] = input.g;
  _gp[0] = input.gp;
  _request_ift = input.request_implicit_tangent;
  _ift_F_step = input.ift_F_step;
  _ift_rate_step = input.ift_rate_step;
  _ift_valid = false;
  _ift_status.clear();
  _ever_subdivided = false;
  _A_implicit.zero();
  _input_euler = input.euler_degrees;
  _grainid = input.grain_id;
  _has_grain_size = input.grain_size > 0.;
  _grain_size = input.grain_size;
  _deformation_gradient_old[0] = input.F_old;
  _deformation_gradient[0] = input.F_new;
  _stress_old[0] = input.stress_old;
  _rotation_increment[0] = RankTwoTensor(RankTwoTensor::initIdentity);
  _t_step = 1;
  if (input.state_old.size() != _num_state_vars)
    ddcpError("Incorrect old state length; expected ", _num_state_vars);
  _state_var_old[0] = input.state_old;
  // A completed material point has a nonzero-norm initial orientation matrix
  // in the first nine state variables. This avoids needing a global step index.
  Real ori_norm = 0.;
  for (int k=0;k<9;++k)
    ori_norm += input.state_old[k]*input.state_old[k];
  if (ori_norm > 0.5)
    _t_step = 2;
  if (_t_step == 2) {
    for (unsigned int a = 0; a < _num_slip_sys; ++a)
      for (unsigned int i=0; i<3; ++i) {
        _y_old[0][i][a] = _data.normals[a][i];
        _z_old[0][i][a] = _data.directions[a][i];
      }
    for (unsigned int a=0;a<_num_slip_sys;++a) {
      normalize_vector(&_y_old[0][0][a],&_y_old[0][1][a],&_y_old[0][2][a]);
      normalize_vector(&_z_old[0][0][a],&_z_old[0][1][a],&_z_old[0][2][a]);
    }
  }
  computeQpStress();
  DDCP4P2Result out;
  out.sigma = _stress[0];
  const Real J = input.F_new.det();
  if (J <= 0.) ddcpError("Non-positive det(F)");
  out.P = J * out.sigma * input.F_new.inverse().transpose();
  out.C_local = _Jacobian_mult[0];
  out.A_option_a.zero();
  if (input.request_option_a_tangent)
  {
    // Treat Eq.(31) C_local as a Jaumann Cauchy tangent by ASSUMPTION.
    // The local stress and modulus have already been fully updated.
    out.A_option_a = DDCP4P2OptionATangent::fromJaumannCauchy(
        input.F_new, out.sigma, out.C_local);
    out.option_a_tangent_available = true;
  }
  out.A_implicit = _A_implicit;
  out.implicit_tangent_available = _ift_valid;
  out.used_constitutive_subincrements = _ever_subdivided;
  out.ift_status = _ift_status;
  out.state = _state_var[0];
  out.psie = _psie_active[0];
  out.psip = _psip_active[0];
  out.euler = _euler_ang[0].a;
  return out;
}

// ============================================================================
// computeQpStress()
// ============================================================================
// Performs one material-point constitutive update. The main sequence is:
//
//   deformation increment
//        -> crystal/intermediate kinematics
//        -> subincremented local Newton solve for slip/climb rates
//        -> updated Fp^{-1} and internal variables
//        -> Cauchy stress
//        -> algorithmic tangent
//        -> phase-field energies and state-variable storage.
// ============================================================================
void DDCP4P2Point::computeQpStress()
{
  Real PI = 4.0*atan(1.0);

  Real psi[3]; // Euler angles
  RankTwoTensor E_el; // Elastic Green stran tensor
  RankTwoTensor E_tot; // Green stran tensor E_el + E_p
  RankTwoTensor F0; // F at beginning of sub increment
  RankTwoTensor F1; // F at end of sub increment
  RankTwoTensor F_el; //  Elastic part of F
  RankTwoTensor F_el_inv; // Inverse of elastic part of F
  RankTwoTensor F_p_inv_0; // Inverse of plastic part of F at beginning of step
  RankTwoTensor F_p_inv; // Inverse of plastic part of F at end of step
  RankTwoTensor E_p; // Plastic strain tensor
  Real E_eff;
  Real E_p_eff;
  Real E_p_eff_cum;

  Real rho_m_avg0, rho_m_avg;
  Real rho_i_avg0, rho_i_avg;

  Real dir_cos[3][3]; // Rotation matrix
  Real dir_cos0[3][3]; // Rotation matrix at the beginning of simulation
  Real C0[3][3][3][3]; // Elastic stiffness tensor
  Real C[3][3][3][3];
  Real C_avg[3][3][3][3];

  RankTwoTensor sig_avg;
  RankTwoTensor sig;
  RankTwoTensor Spk2;

  Real ddpdsig[3][3][3][3];
  Real ddsdde_4th[3][3][3][3];

  // slip system dependent arrays
  std::vector<std::vector<Real>> xs0(3, std::vector<Real>(_num_slip_sys));
  std::vector<std::vector<Real>> xs(3, std::vector<Real>(_num_slip_sys));
  std::vector<std::vector<Real>> xm0(3, std::vector<Real>(_num_slip_sys));
  std::vector<std::vector<Real>> xm(3, std::vector<Real>(_num_slip_sys));

  std::vector<Real> rho_m0(_num_slip_sys); // mobile dislocation density at beginning of step
  std::vector<Real> rho_m(_num_slip_sys); // mobile dislocation density at end of step
  std::vector<Real> rho_i0(_num_slip_sys); // immobile dislocation density at beginning of step
  std::vector<Real> rho_i(_num_slip_sys); // immobile dislocation density at end of step
  std::vector<Real> bstress0(_num_slip_sys); // backstress at beginning of step
  std::vector<Real> bstress(_num_slip_sys); // backstress at end of step
  std::vector<Real> tau(_num_slip_sys); // resolved shear stress
  std::vector<Real> tau_eff(_num_slip_sys); // effective resolved shear stress
  std::vector<Real> s_a(_num_slip_sys); // athermal slip resistance
  std::vector<Real> s_t(_num_slip_sys); // thermal slip resistance
  std::vector<Real> gamma_dot(_num_slip_sys); // shear strain rate
  std::vector<Real> gamma_dot_g(_num_slip_sys); // shear strain rate due to glide
  std::vector<Real> gamma_dot_c (_num_slip_sys); // Shear Strain Rate due to climb
  std::vector<Real> gamma_try(_num_slip_sys); // trial shear strain rate
  std::vector<Real> gamma_try_g(_num_slip_sys); // trial shear strain rate due to glide
  std::vector<Real> gamma_try_c(_num_slip_sys); // trial shear strain rate due to climb

  std::vector<Real> residual(_num_slip_sys); // residual during Newton-Raphson iteration
  std::vector<std::vector<Real>> dTaudGd(_num_slip_sys, std::vector<Real>(_num_slip_sys));

  // Climb Variables
  Real interdensity, vacdensity; // Density of Interstitials and Vacancies
  Real Di, Dv;  // Interstial and Vacancy Diffusivity at the Temp of Calculation
  Real vacdensity_th; // Zero stress thermal equilibrium density of vacancies
  Real lg; // Mean Free Path of Dislocation Climb *NOT USED TBD*
  std::vector<Real> climbstress(_num_slip_sys); 
  std::vector<Real> zi(_num_slip_sys); // Finite Stress Interstitial Capture Efficency
  std::vector<Real> zv(_num_slip_sys); // Finite Stress Vacancy Capture Efficency
  std::vector<Real> Kis(_num_slip_sys); // Insterstial sink (Dislocation) reaction rate coefficient
  std::vector<Real> Kvs(_num_slip_sys); // Vacancy sink (Dislocation) Reaction rate coefficient
  std::vector<Real> vel_climb(_num_slip_sys); // Climb Velocity
  std::vector<Real> vacdensity_0(_num_slip_sys); // Density of Vacancies on each slip system 
  std::vector<std::vector<Real>> dEtadGc(_num_slip_sys, std::vector<Real>(_num_slip_sys)); // d(climbstress)/d(gammadot_climb)   

  // Variables for Irradiation Defects
  std::vector<Real> Nloop0(_num_slip_sys); // Number Density of Irradiation Loops - Initial Value Read from Input File/SDV
  std::vector<Real> dloop0(_num_slip_sys); // Diameter of Irradiation Loops - Initial Value Read from Input File/SDV
  std::vector<Real> Nloop(_num_slip_sys); // Number Density of Irradiation Loops - Final Value to be stored into SDV
  std::vector<Real> dloop(_num_slip_sys); // Diameter of Irradiation Loops - Final Value to be stored into SDV


  // Old Elastic and Plastic Energies

  Real psie_active0, psip_active0;


  // interaction coefficient martix between different slip systems

  std::vector<std::vector<Real>> A(_num_slip_sys, std::vector<Real>(_num_slip_sys));
  std::vector<std::vector<Real>> H(_num_slip_sys, std::vector<Real>(_num_slip_sys));


  // Resize Some Vectors if Climb is considered

  if(_climbmodel){
    residual.resize(2*_num_slip_sys);
  }
  else {
    interdensity = 0;
    vacdensity = 0;
    Di = 0;
    Dv = 0;
    vacdensity_th = 0;
    for(unsigned int k = 0; k < _num_slip_sys; k++){
      gamma_dot_c[k] = 0;
      gamma_try_c[k] = 0;
      climbstress[k] = 0;
      zi[k] = 0;
      zv[k] = 0;
      Kis[k] = 0;
      Kvs[k] = 0;
      vel_climb[k] = 0;
      vacdensity_0[k] = 0;
    }
  }


  // Our model is based on V.R decomposition as opposed to MOOSE's R.U decomposition of the deformation gradient
  // F = V.R = R.U
  // V = R.U.R_T
  // Our model needs everything in the current configuration

/* NOT IN USE TBD
  RankTwoTensor delV;
  delV = _rotation_increment[_qp] * _strain_increment[_qp] * _rotation_increment[_qp].transpose();
*/

  // Rotate stress to current configuration
  _stress[_qp] = _rotation_increment[_qp] * _stress_old[_qp] * _rotation_increment[_qp].transpose();


  // Recover "old" state variables
  std::copy(_state_var_old[_qp].begin(), _state_var_old[_qp].end(), _state_var[_qp].begin());
  std::copy(_properties_old[_qp].begin(), _properties_old[_qp].end(), _properties[_qp].begin());


  // Recover "old" miller indices
  std::copy(_y_old[_qp].begin(), _y_old[_qp].end(), _y[_qp].begin());
  std::copy(_z_old[_qp].begin(), _z_old[_qp].end(), _z[_qp].begin());

  if (_t_step <= 1) {

  // FFT voxel orientation (degrees), supplied by caller.
  psi[0] = _input_euler[0];
  psi[1] = _input_euler[1];
  psi[2] = _input_euler[2];
  _grain_size = _has_grain_size ? _grain_size : 1.0e10;
  }


  // Read in material parameters
  if (_t_step <= 1){
      readPropsFile();
  }
  assignProperties();


  // Initialize Kronecker delta tensor
  Real del[3][3];
  for (unsigned int i = 0; i < 3; i++){
    for (unsigned int j = 0; j < 3; j++){
      del[i][j] = 0;
    }
    del[i][i] = 1;
  }


  // Slip-system normals and directions are loaded once by the adapter.
  // The original first-step normalization and orthogonality checks remain.
  if (_t_step <= 1) {
    for (unsigned int islip = 0; islip < _num_slip_sys; ++islip) {
      for (unsigned int d = 0; d < 3; ++d) {
        _y[0][d][islip] = _data.normals[islip][d];
        _z[0][d][islip] = _data.directions[islip][d];
      }
      normalize_vector(&_y[0][0][islip], &_y[0][1][islip], &_y[0][2][islip]);
      normalize_vector(&_z[0][0][islip], &_z[0][1][islip], &_z[0][2][islip]);
      Real dot = 0.;
      for (unsigned int d = 0; d < 3; ++d)
        dot += _y[0][d][islip] * _z[0][d][islip];
      if (std::abs(dot) > 1e-8)
        throw std::runtime_error("Slip normal and slip direction not perpendicular");
    }
  }


  // Initialize interaction coefficient matrix

  for (unsigned int i = 0; i < _num_slip_sys; i++) {
      for (unsigned int j = 0; j < _num_slip_sys; j++) {
        A[i][j] = p0*Alatent;
        H[i][j] = 1.0;
      }
      A[i][i] = p0*1.0;
      H[i][i] = 1.0;
  }


  if (_t_step <= 1) {

    // Convert Euler angles to radians
    psi[0] = psi[0]*PI/180;
    psi[1] = psi[1]*PI/180;
    psi[2] = psi[2]*PI/180;


    // Initialize F_p_inv_0
    for (unsigned int i = 0; i<3; i++) {
      for(unsigned int j = 0; j<3; j++) {
        F_p_inv_0(i,j) = 0.0;
      }
      F_p_inv_0(i,i) = 1.0;
    }


    // Initialize E_p.
    for (unsigned int i = 0; i < 3; i++) {
      for(unsigned int j = 0; j < 3; j++) {
        E_p(i,j) = 0;
      }
    }


    // Initialize E_eff
    E_eff = 0;


    // Initialize E_p_eff
    E_p_eff = 0;


    // Initialize E_p_eff_cum
    E_p_eff_cum = 0;


    // Initialize Active Work densities
    _psie_active[_qp] = 0;
    _psip_active[_qp] = 0;
    psie_active0 = 0;
    psip_active0 = 0;


    // Initialize average values of defect densities and the corresponding loop sizes
    rho_m_avg0 = rho_m_zero;
    rho_i_avg0 = rho_i_zero;

    for (unsigned int n = 0; n<_num_slip_sys; n++){
      rho_m0[n] = rho_m_zero;
    }

    for (unsigned int n = 0; n<_num_slip_sys; n++){
      rho_i0[n] = rho_i_zero;
    }

    for (unsigned int n = 0; n<_num_slip_sys; n++){
      bstress0[n] = 0.0;
    }

    if(_irad_defects){
    for (unsigned int n = 0; n <_num_slip_sys; n++){
      Nloop0[n] = Nloop_zero/(_num_slip_sys);
      dloop0[n] = dloop_zero;
    }
    }
    else {
    for (unsigned int n = 0; n <_num_slip_sys; n++){
      Nloop0[n] = 0.0;
      dloop0[n] = 0.0;
      beta_loop = 1;
    }
    }
  }
  //  End of initializations.  Read in internal variables

  if(_t_step > 1) {
    // checkpoint("Initializing ISVs, nonzero time step")
    int n = -1;

    // Read in dir_cos0 SDV 1-9
    for (unsigned int i = 0; i<3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        n = n + 1;
        dir_cos0[i][j] = _state_var[_qp][n];
      }
    }


    // Read in dir_cos SDV 10-18
    for (unsigned int i = 0; i<3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        n = n + 1;
        dir_cos[i][j] = _state_var[_qp][n];
      }
    }


    // Read the elastic part of F SDV 19-27
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        n = n + 1;

      }
    }


    // Read inverse of the plastic part of F SDV 28-36
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        n = n + 1;
        F_p_inv_0(i,j) = _state_var[_qp][n];
      }
    }


    // Read E_p SDV 37-45
    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        n = n + 1;
        E_p(i,j) = _state_var[_qp][n];
      }
    }


    // Read E_eff SDV 46
    n = n + 1;
    E_eff = _state_var[_qp][n];


    // Read E_p_eff SDV 47
    n = n + 1;
    E_p_eff = _state_var[_qp][n];


    // Read average values of defect densities and loop sizes SDV 48-49
    n = n + 1;
    rho_m_avg0 = _state_var[_qp][n];

    n = n + 1;
    rho_i_avg0 = _state_var[_qp][n];


    // Read average dislocation glide rates  SDV 50
    n = n + 1;


    // Read older work densities SDV51 & 52
    n = n+1;
    psie_active0  = _state_var[_qp][n];
    n = n+1;
    psip_active0  = _state_var[_qp][n];


    // Read mobile dislocation density values SDV 52 + _num_slip_sys
    for (unsigned int i = 0; i < _num_slip_sys; i++) {
      n = n + 1;
      rho_m0[i] = _state_var[_qp][n];
    }


    // Read immobile dislocation density values SDV 52 + _num_slip_sys*2
    for (unsigned int i = 0; i < _num_slip_sys; i++) {
      n = n + 1;
      rho_i0[i] = _state_var[_qp][n];
    }


    // Read back stress values SDV 52 + _num_slip_sys*3
    for (unsigned int i = 0; i < _num_slip_sys; i++) {
      n = n + 1;
      bstress0[i] = _state_var[_qp][n];
    }


    // Read Dislocation Loop Densities from SDV 52 + _num_slip_sys*4
    if(_irad_defects){
      for (unsigned int i = 0; i < _num_slip_sys; i++) {
        n = n + 1;
        Nloop0[i] = _state_var[_qp][n];
      }


    //  Read Dislocation Loop Densities from SDV 52 + _num_slip_sys*5
    for (unsigned int i = 0; i < _num_slip_sys; i++) {
        n = n + 1;
        dloop0[i] = _state_var[_qp][n];
      }
    }
    else {
      for (unsigned int i = 0; i < _num_slip_sys; i++){
        Nloop0[i] = 0.0;
        dloop0[i] = 0.0;
      }
    }


  }

  if(_t_step <= 1) {
    Real s1 = sin(psi[0]);
    Real c1 = cos(psi[0]);
    Real s2 = sin(psi[1]);
    Real c2 = cos(psi[1]);
    Real s3 = sin(psi[2]);
    Real c3 = cos(psi[2]);

    dir_cos0[0][0] = c1*c3-s1*s3*c2;
    dir_cos0[0][1] = s1*c3+c1*s3*c2;
    dir_cos0[0][2] = s3*s2;
    dir_cos0[1][0] = -c1*s3-s1*c3*c2;
    dir_cos0[1][1] = -s1*s3+c1*c3*c2;
    dir_cos0[1][2] = c3*s2;
    dir_cos0[2][0] = s1*s2;
    dir_cos0[2][1] = -c1*s2;
    dir_cos0[2][2] = c2;

    for (unsigned int i = 0; i < 3; i++){
      for (unsigned int j = 0; j < 3; j++){
        dir_cos[i][j] = dir_cos0[i][j];
      }
    }
  }


  // Initialize ANISOTROPIC 4th rank elastic stiffness tensor
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int l = 0; l < 3; l++) {
          C0[i][j][k][l] = C12*del[i][j]*del[k][l] + C44*(del[i][k]*del[j][l]+del[i][l]*del[k][j]);
        }
      }
    }
  }
  C0[0][0][0][0] = C11;
  C0[1][1][1][1] = C11;
  C0[2][2][2][2] = C11;


  // Initialize arrays for averaging over grains
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      sig_avg(i,j) = 0.0;
    }
  }


  // Begin calculations over the element
  rotate_4th(dir_cos, C0, C);


  // Convert Miller Indices to global coordinates
  for (unsigned int n = 0; n < _num_slip_sys; n++) {
    for (unsigned int i = 0; i < 3; i++) {
      xs0[i][n] = 0.0;
      xm0[i][n] = 0.0;
      for (unsigned int j = 0; j < 3; j++) {
        xs0[i][n] = xs0[i][n] + dir_cos0[i][j]*_z[_qp][j][n];
        xm0[i][n] = xm0[i][n] + dir_cos0[i][j]*_y[_qp][j][n];
      }
    }
  }


  // Initialize number of sub-increments.  Note that the subincrement initially equals the total increment. 
  // This remains the case unless the process starts to diverge.
  unsigned int N_incr, N_incr_total, iNR, N_ctr;
  Real dt_incr;
  RankTwoTensor dfgrd0, dfgrd1, Ftheta, Ftheta_old;
  RankTwoTensor total_eigenstrain, total_eigenstrain_old;
  RankTwoTensor array1, array2;


// ============================================================================
// PRECOMPUTE SLIP / CLIMB STRESS-DERIVATIVE GEOMETRY
//
// During subincrementing:
//     dTaudGd[a][b] = -dt_incr * Ktau[a][b]
//
// where
//     Ktau[a][b] = Pglide[a] : C0 : Pglide[b]
//
// xs0, xm0 and C0 remain unchanged during the local subincrement solve.
// Therefore the expensive tensor contractions are calculated only once.
// ============================================================================
// Each 3x3 tensor is stored as a flat 9-component array.

// ============================================================================
// PRECOMPUTED REFERENCE-CONFIGURATION STRESS-DERIVATIVE GEOMETRY
//
// The reference formulation evaluates the slip/climb stress sensitivities with
// the crystal-frame stiffness C0.  Only geometry and C0 enter Ktau/Keta, so the
// intact contractions can be assembled once per quadrature-point evaluation.
//
//   Pg[a] = s0[a] (x) m0[a]
//   Pc[a] = s0[a] (x) s0[a]
//
//   Ktau[a][b] = Pg[a] : C0 : Pg[b]
//   Keta[a][b] = Pc[a] : C0 : Pc[b]
//
// Phase-field degradation is applied later at the current Newton iterate.

using Tensor9 = std::array<Real, 9>;


// ============================================================================
// GLIDE PROJECTION TENSORS
//
// Pglide[a] = xs0[a] tensor xm0[a]
// ============================================================================

std::vector<Tensor9> Pglide(_num_slip_sys);
std::vector<Tensor9> CPglide(_num_slip_sys);


// Build Pglide once

for (unsigned int islip = 0;
     islip < _num_slip_sys;
     ++islip)
{
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            const unsigned int ij = 3*i + j;

            Pglide[islip][ij] =
                xs0[i][islip]
                * xm0[j][islip];
        }
    }
}


// ============================================================================
// Calculate
//
//        CPglide[b] = C0 : Pglide[b]
//
// IMPORTANT:
// This is calculated only ONCE for each slip system b.
//
// Precomputing CPglide avoids repeating the fourth-order contraction for every (a,b) pair.
// ============================================================================

for (unsigned int ib = 0;
     ib < _num_slip_sys;
     ++ib)
{
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            Real value = 0.0;

            for (unsigned int k = 0; k < 3; ++k)
            {
                for (unsigned int l = 0; l < 3; ++l)
                {
                    value +=
                        C0[i][j][k][l]
                        * Pglide[ib][3*k + l];
                }
            }

            CPglide[ib][3*i + j] =
                value;
        }
    }
}


// ============================================================================
// Base glide derivative matrix
//
//       Ktau[a][b] = Pglide[a] : CPglide[b]
//                  = Pglide[a] : C0 : Pglide[b]
//
// This matrix contains NO dt_incr.
// ============================================================================

std::vector<std::vector<Real>> Ktau(
    _num_slip_sys,
    std::vector<Real>(_num_slip_sys, 0.0));


for (unsigned int ia = 0;
     ia < _num_slip_sys;
     ++ia)
{
    for (unsigned int ib = 0;
         ib < _num_slip_sys;
         ++ib)
    {
        Real value = 0.0;

        for (unsigned int ij = 0;
             ij < 9;
             ++ij)
        {
            value +=
                Pglide[ia][ij]
                * CPglide[ib][ij];
        }

        Ktau[ia][ib] =
            value;
    }
}


// ============================================================================
// CLIMB DERIVATIVE BASE MATRIX
//
// Climb projection tensor:
//
//      Pc[a] = s[a] tensor s[a]
//
// and
//
//      Keta[a][b] = Pc[a] : C0 : Pc[b]
//
// Therefore:
//
//      dEtadGc[a][b] = -dt_incr * Keta[a][b]
//
// Pc, C0 and therefore Keta do not change during subincrementing.
// ============================================================================

std::vector<std::vector<Real>> Keta(
    _num_slip_sys,
    std::vector<Real>(_num_slip_sys, 0.0));


if (_climbmodel)
{

    // Pc = s tensor s

    std::vector<Tensor9> Pclimb(_num_slip_sys);


    // CPc = C0 : Pc

    std::vector<Tensor9> CPclimb(_num_slip_sys);


    // ------------------------------------------------------------------------
    // Construct Pc[a] = s[a] tensor s[a]
    // ------------------------------------------------------------------------

    for (unsigned int islip = 0;
         islip < _num_slip_sys;
         ++islip)
    {
        for (unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                const unsigned int ij =
                    3*i + j;

                Pclimb[islip][ij] =
                    xs0[i][islip]
                    * xs0[j][islip];
            }
        }
    }


    // ------------------------------------------------------------------------
    // Calculate
    //
    //      CPclimb[b] = C0 : Pclimb[b]
    //
    // This expensive fourth-order contraction is performed only Ns times.
    // ------------------------------------------------------------------------

    for (unsigned int ib = 0;
         ib < _num_slip_sys;
         ++ib)
    {
        for (unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                Real value = 0.0;

                for (unsigned int k = 0; k < 3; ++k)
                {
                    for (unsigned int l = 0; l < 3; ++l)
                    {
                        value +=
                            C0[i][j][k][l]
                            * Pclimb[ib][3*k + l];
                    }
                }

                CPclimb[ib][3*i + j] =
                    value;
            }
        }
    }


    // ------------------------------------------------------------------------
    // Calculate
    //
    //      Keta[a][b]
    //          = Pclimb[a] : CPclimb[b]
    //          = Pclimb[a] : C0 : Pclimb[b]
    //
    // Only inexpensive 9-component dot products remain here.
    // ------------------------------------------------------------------------

    for (unsigned int ia = 0;
         ia < _num_slip_sys;
         ++ia)
    {
        for (unsigned int ib = 0;
             ib < _num_slip_sys;
             ++ib)
        {
            Real value = 0.0;

            for (unsigned int ij = 0;
                 ij < 9;
                 ++ij)
            {
                value +=
                    Pclimb[ia][ij]
                    * CPclimb[ib][ij];
            }

            Keta[ia][ib] =
                value;
        }
    }
}

  bool converged, improved;

  // Initialize deformation gradients for beginning and end of subincrement.

  // No eigenstrain in the initial Marlin adapter.
  // Future adapters must explicitly pass thermal/eigen deformation gradients.

  Ftheta.zero();
  Ftheta_old.zero();
  for (int i=0; i<3; ++i)
  {
      Ftheta(i,i) = sqrt(1.0 + 2.0* total_eigenstrain(i,i));
      Ftheta_old(i,i) = sqrt(1.0 + 2.0* total_eigenstrain_old(i,i));
  }

  dfgrd0 = _deformation_gradient_old[_qp] * Ftheta_old.inverse();
  dfgrd1 = _deformation_gradient[_qp] * Ftheta.inverse();
  F0 = dfgrd0;
  F1 = dfgrd1;


  // counters for time step sub-increment
  N_incr = 1;
  N_incr_total = 1;
  Real sse_old, sse_ref;


  // ==================================================================
  // Reusable Newton workspace
  // Allocate ONCE per quadrature-point material evaluation rather than
  // once per Newton iteration.
  // ==================================================================

  const unsigned int totalsys =
      _climbmodel ? 2u * _num_slip_sys : _num_slip_sys;


// Variables for derivative with respect to glide component.
//
// temp1-temp6 are no longer required because these matrices were
// diagonal in the present constitutive formulation. Their inverse
// products can therefore be evaluated by scalar division.

    std::vector<std::vector<Real>> drhomdgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> drhoidgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dbsdgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dsadgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dsa1dgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dsa2dgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dstdgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dNldgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> ddldgb(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));


   // Climb Matrices

    std::vector<std::vector<Real>> drhomdgc(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> drhoidgc(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dvcdgc(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dzidgc(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dzvdgc(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));

    std::vector<std::vector<Real>> dcvth0dgc(
        _num_slip_sys,
        std::vector<Real>(_num_slip_sys, 0.0));


  // Newton Jacobian

  std::vector<std::vector<Real>> array3(
      totalsys,
      std::vector<Real>(totalsys, 0.0));


  // Newton vectors

  std::vector<Real> gradient(totalsys, 0.0);
  std::vector<Real> d_gamma_dot(totalsys, 0.0);


// ============================================================================
// Reusable dense linear-solver workspace
//
// Current Newton Jacobian is block diagonal:
//
//           [ Jg   0 ]
//       J = [        ]
//           [ 0   Jc ]
//
// Solve the two Ns x Ns blocks independently.
// ============================================================================

libMesh::DenseMatrix<Real> Jg(
    _num_slip_sys,
    _num_slip_sys);

libMesh::DenseVector<Real> rhs_g(
    _num_slip_sys);

libMesh::DenseVector<Real> delta_g(
    _num_slip_sys);


libMesh::DenseMatrix<Real> Jc(
    _num_slip_sys,
    _num_slip_sys);

libMesh::DenseVector<Real> rhs_c(
    _num_slip_sys);

libMesh::DenseVector<Real> delta_c(
    _num_slip_sys);


// ============================================================================
// Reusable NR_residual workspace
//
// Allocate ONCE for this quadrature-point material evaluation.
// The same storage is reused by:
//
//   4. every constitutive subincrement.
//
// No constitutive state is stored here.  These are temporary arrays only.
// ============================================================================

NRWorkspace nr_workspace(_num_slip_sys);


  // Begin time step subincrement

  do {
  dt_incr = _dt/N_incr_total;

  F0 = dfgrd0 + (dfgrd1 - dfgrd0)*(N_incr - 1)/N_incr_total;
  F1 = dfgrd0 + (dfgrd1 - dfgrd0)*N_incr/N_incr_total;


  // Initialize mobile dislocation density
  for (unsigned int n = 0; n < _num_slip_sys; n++) {
    rho_m[n] = rho_m0[n];
  }


  // Initialize immobile dislocation density
  for (unsigned int n = 0; n < _num_slip_sys; n++) {
    rho_i[n] = rho_i0[n];
  }


  // Initialize backstress
  for (unsigned int n = 0; n < _num_slip_sys; n++) {
    bstress[n] = bstress0[n];
  }


  // Initialize Number Density and Size of Dislocation Loops
  for (unsigned int n = 0; n < _num_slip_sys; n++) {
    Nloop[n] = Nloop0[n];
    dloop[n] = dloop0[n];
  }


  // Initialize average defect densities and loop sizes
  rho_m_avg = rho_m_avg0;
  rho_i_avg = rho_i_avg0;


  // Multiply F() by F_p_inv() to get F_el()
  F_el = F0 * F_p_inv_0;
  if (F_el.det() <= 0.0)
    throw std::runtime_error("Non-positive elastic Jacobian before constitutive solve");

  F_el_inv = F_el.inverse();


  // Rotate xs0 and xm0 to current coordinates, called xs and xm
  for (unsigned int n = 0; n < _num_slip_sys; n++) {
    for (unsigned int i = 0; i < 3; i++) {
      xs[i][n] = 0.0;
      xm[i][n] = 0.0;
      for (unsigned int j = 0; j < 3; j++) {
        xs[i][n] = xs[i][n] + F_el(i,j)*xs0[j][n];
        xm[i][n] = xm[i][n] + xm0[j][n]*F_el_inv(j,i);
      }
    }
  }


  // Evaluate the phase-field-degraded Cauchy stress from the current elastic state.
  sig = computeCauchyStress(
    F_el,
    C,
    false
  );


  // Calculate resolved shear stress for each slip system
  for (unsigned int k = 0; k < _num_slip_sys; k++) {
    tau[k] = 0.0;
    for (unsigned int j = 0; j < 3; j++) {
      for (unsigned int i = 0; i < 3; i++) {
        tau[k] = tau[k] + xs[i][k]*xm[j][k]*sig(i,j);
      }
    }
  }


  // Calculate effective shear stress for each slip system
  for (unsigned int k = 0; k < _num_slip_sys; k++) {
    tau_eff[k] = tau[k] - bstress[k];
  }


  // Calculate athermal slip resistance for each slip system.
  calc_crss(
    _num_slip_sys,
    rho_m,
    rho_i,
    Nloop,
    dloop,
    s_a,
    A);


  // Take care of degradation due to phase-field variable
    const Real gp_raw1 =
        (_gp[_qp]);
   for(unsigned int ia = 0; ia < _num_slip_sys; ia++) {
     s_a[ia] = s_a[ia]*gp_raw1;
  }


  // Calculate reference shear stress for each slip system.
  for (unsigned int ia = 0; ia < _num_slip_sys; ia++) {
    s_t[ia] = frictional_stress;
  }


  std::vector<Real> d_disl(_num_slip_sys);
  std::vector<Real> d_disl4(_num_slip_sys);
  calc_mfp(
    _num_slip_sys,
    rho_m,
    rho_i,
    Nloop,
    dloop,
    H,
    d_disl);
    d_disl4 = d_disl;


  calc_sliprate(
    _num_slip_sys,
    tau_eff,
    s_a,
    s_t,
    gamma_dot_g,
    gamma_dot);


  // Calculate the resolved climb stresses on each of the slip system
  if(_climbmodel){
    for (unsigned int k = 0; k < _num_slip_sys; k++) {
        climbstress[k] = 0.0;
        for (unsigned int j = 0; j< 3; j++) {
            for (unsigned int i = 0; i < 3 ; i++) {
              climbstress[k] = climbstress[k] + xs[i][k]*xs[j][k]*sig(i,j);
            }
        }
    }


    // Calculate the Di & Dv
    Di = Di0 * exp(-((Em_i*Ecorr)/(B_k * _temp[_qp])));
    Dv = Dv0 * exp(-((Em_v*Ecorr)/(B_k * _temp[_qp])));


    // Calculate the zero stress thermal equilibrium Density of vacancies 
    vacdensity_th = Nv * exp((-G0*Ecorr) / (B_k * _temp[_qp]));


    // Calculate The ClimbRate
    calc_climbrate(
      _num_slip_sys,
      Di,
      Dv,
      vacdensity_th,
      interdensity,
      vacdensity,
      climbstress,
      zi,
      zv,
      Kis,
      Kvs,
      vacdensity_0,
      vel_climb,
      rho_m,
      rho_i,
      gamma_dot_c);
    }
  else {
      Di = 0;
      Dv = 0;
      vacdensity_th = 0;
      interdensity = 0;
      vacdensity = 0;
    for (unsigned int i = 0; i < _num_slip_sys; i++){
      climbstress[i] = 0;
      zi[i] = 0;
      zv[i] = 0;
      Kis[i] = 0;
      Kvs[i] = 0;
      vacdensity_0[i] = 0;
      vel_climb[i] = 1e-25;
      gamma_dot_c[i] = 0;
      for(unsigned int j = 0; j < _num_slip_sys; j++){
        dEtadGc[i][j] = 0;
      }

    }
  }


  // Begin Newton-Raphson iterative loops.
  iNR = 0;
  converged = false;
  bool subdivide_increment = false;

  do {
    iNR = iNR + 1;


    NR_residual (
    _num_slip_sys,
    xs0,
    xm0,
    _temp[_qp],
    dt_incr,
    gamma_dot,
    gamma_dot_c,
    F1,
    F_el,
    F_p_inv,
    F_p_inv_0,
    C,
    rho_m0,
    rho_m,
    rho_i0,
    rho_i,
    bstress0,
    bstress,
    Nloop0,
    Nloop,
    dloop0,
    dloop,
    sig,
    tau,
    climbstress,
    tau_eff,
    s_a,
    s_t,
    A,
    H,
    interdensity,
    vacdensity,
    vacdensity_th,
    Di,
    Dv,
    nr_workspace,
    residual,
    sse);

    sse_ref = sse;

// ============================================================================
// PHASE-FIELD CORRECTION OF LOCAL NEWTON STRESS DERIVATIVES
//
// C0 is retained exactly as in the reference constitutive derivative.
//
// NONE and VOLDEV tension:
//
//   d(tau_a)/d(gammadot_g_b) = -dt * g * Ktau[a][b]
//   d(eta_a)/d(gammadot_c_b) = -dt * g * Keta[a][b]
//
// In VOLDEV compression the inactive volumetric contribution must remain
// undegraded.  The code therefore contracts Pg/Pc with a C0-based effective
// tangent for the current Fe rather than multiplying the whole derivative by g.

    const Real g_local = (_g[_qp]);
    const Real Je_local = F_el.det();

    if (Je_local <= 0.0)
      ddcpError("Non-positive elastic Jacobian in local constitutive Newton: Je = ", Je_local);

    if (_decomposition == Decomposition::none ||
        (_decomposition == Decomposition::voldev && Je_local > 1.0))
    {
      for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
        for (unsigned int ib = 0; ib < _num_slip_sys; ++ib)
          dTaudGd[ia][ib] = -dt_incr * g_local * Ktau[ia][ib];

      if (_climbmodel)
        for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
          for (unsigned int ib = 0; ib < _num_slip_sys; ++ib)
            dEtadGc[ia][ib] = -dt_incr * g_local * Keta[ia][ib];
    }
    else
    {
      const RankTwoTensor Ce_local = F_el.transpose() * F_el;
      const RankTwoTensor Ce_inv_local = Ce_local.inverse();
      const Real Je23_local = std::pow(Je_local, 2.0 / 3.0);

      for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
      {
        for (unsigned int ib = 0; ib < _num_slip_sys; ++ib)
        {
          Real value = 0.0;

          for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
              for (unsigned int k = 0; k < 3; ++k)
                for (unsigned int l = 0; l < 3; ++l)
                {
                  const Real Pg_a = xs0[i][ia] * xm0[j][ia];
                  const Real Pg_b = xs0[k][ib] * xm0[l][ib];
                  const Real C0_Pvol =
                      (1.0 / 3.0) * Je23_local *
                      (C0[i][j][0][0] + C0[i][j][1][1] + C0[i][j][2][2]) *
                      Ce_inv_local(k,l);

// C0-based VOLDEV compression tangent component:
//
//   (C0_eff)_ijkl = g (C0)_ijkl + (1-g) (C0:Pvol)_ijkl
//
//   (C0:Pvol)_ijkl
//       = (1/3) Je^(2/3)
//         [C0_ij00 + C0_ij11 + C0_ij22]
//         (Ce^{-1})_kl,
//
// where Ce = Fe^T Fe.

                  const Real C0_eff =
                      g_local * C0[i][j][k][l] + (1.0 - g_local) * C0_Pvol;

                  value += Pg_a * C0_eff * Pg_b;
                }

          dTaudGd[ia][ib] = -dt_incr * value;
        }
      }

      if (_climbmodel)
      {
        for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
        {
          for (unsigned int ib = 0; ib < _num_slip_sys; ++ib)
          {
            Real value = 0.0;

            for (unsigned int i = 0; i < 3; ++i)
              for (unsigned int j = 0; j < 3; ++j)
                for (unsigned int k = 0; k < 3; ++k)
                  for (unsigned int l = 0; l < 3; ++l)
                  {
                    const Real Pc_a = xs0[i][ia] * xs0[j][ia];
                    const Real Pc_b = xs0[k][ib] * xs0[l][ib];
                    const Real C0_Pvol =
                        (1.0 / 3.0) * Je23_local *
                        (C0[i][j][0][0] + C0[i][j][1][1] + C0[i][j][2][2]) *
                        Ce_inv_local(k,l);
                    const Real C0_eff =
                        g_local * C0[i][j][k][l] + (1.0 - g_local) * C0_Pvol;

                    value += Pc_a * C0_eff * Pc_b;
                  }

            dEtadGc[ia][ib] = -dt_incr * value;
          }
        }
      }
    }


    // Begin calculation of the partial derivatives needed for the Newton-Raphson step
    // Calculate derivative of the mobile dislocation density, rho_m, immobile dislocation density, rho_i, backstress, bstress, w.r.t. gamma-dot-beta
    // Update Effective Mean Free Paths

    calc_mfp(
      _num_slip_sys,
      rho_m,
      rho_i,
      Nloop,
      dloop,
      H,
      d_disl);
      d_disl4 = d_disl;



    // ################## OPTIMIZED VERSION #########################
    // ------------------------------------------------------------------
    // Optimized derivatives with respect to glide slip rate.
    //
    // The original implementation constructed temp1-temp6 as full
    // matrices, explicitly inverted temp1/temp3/temp5, and then performed
    // dense matrix multiplication.
    //
    // In the current formulation these matrices are diagonal.
    // Therefore all three matrix equations reduce to scalar division.
    // ------------------------------------------------------------------
    // ================================================================
    // Mobile dislocation density derivative
    // drhomdgb = inverse(temp1) * temp2
    // ================================================================

    for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
    {
        const Real abs_gamma =
            std::abs(gamma_dot[ia]);

        const Real sign_gamma =
            sgn(gamma_dot[ia]);

        const Real temp2_diag =
            (
                k_M / (b_mag * d_disl[ia])
              - k_I / (b_mag * d_disl4[ia])
              - k_ann * R_c * rho_m[ia] / b_mag
            )
            * sign_gamma
            * dt_incr;

        const Real temp1_diag =
            1.0
            - (
                0.5 * k_M * d_disl[ia] / b_mag
              + k_ann * R_c / b_mag
              + 0.5 * k_I * d_disl4[ia] / b_mag
              )
              * abs_gamma
              * dt_incr;

        drhomdgb[ia][ia] =
            temp2_diag / temp1_diag;
    }


    // ================================================================
    // Immobile dislocation density derivative
    // drhoidgb = inverse(temp3) * temp4
    // ================================================================

    for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
    {
        const Real abs_gamma =
            std::abs(gamma_dot[ia]);

        const Real sign_gamma =
            sgn(gamma_dot[ia]);

        const Real t1 =
            (k_I * d_disl4[ia])
            / (2.0 * b_mag);

        const Real temp3_diag =
            1.0
            + (-t1 + k_D)
            * abs_gamma
            * dt_incr;

        const Real t5 =
            k_I
            / (b_mag * d_disl4[ia]);

        const Real t6 =
            k_D * rho_i[ia];

        const Real temp4_diag =
            (t5 - t6)
            * dt_incr
            * sign_gamma;

        drhoidgb[ia][ia] =
            temp4_diag / temp3_diag;
    }


    // ================================================================
    // Irradiation-loop density derivative
    // dNldgb = inverse(temp5) * temp6
    // ================================================================

    if (_irad_defects)
    {
        for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
        {
            const Real loop_term =
                std::sqrt(
                    Nloop[ia]
                    * dloop[ia]);

            const Real sqrt_rhom =
                std::sqrt(rho_m[ia]);

            const Real abs_gamma =
                std::abs(gamma_dot[ia]);

            const Real sign_gamma =
                sgn(gamma_dot[ia]);

            const Real temp5_diag =
                1.0
                + (
                    sqrt_rhom
                    * abs_gamma
                    * dt_incr
                  )
                  /
                  (
                    4.0
                    * b_mag
                    * loop_term
                  );

            const Real temp6_diag =
                loop_term
                * sqrt_rhom
                * (
                    -dt_incr
                    / (2.0 * b_mag)
                  )
                * sign_gamma;

            dNldgb[ia][ia] =
                temp6_diag / temp5_diag;
        }
    }


    // backstress
// ================================================================
// Backstress derivative
// Keep dbsdgb as full Ns x Ns matrix.
// ================================================================

    const Real krhoGb =
        k_rho * G * b_mag;

    for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
    {
        const Real rho_sum =
            rho_m[ia] + rho_i[ia];

        const Real sqrt_rho =
            std::sqrt(rho_sum);

        const Real inv_sqrt_rho =
            1.0 / sqrt_rho;

        const Real abs_gamma =
            std::abs(gamma_dot[ia]);

        const Real sign_gamma =
            sgn(gamma_dot[ia]);

        const Real sign_tau =
            sgn(tau[ia] - bstress[ia]);

        const Real prefactor =
              k_bs1
            * 0.5
            * krhoGb
            * inv_sqrt_rho
            * dt_incr
            * sign_tau
            * abs_gamma;

        const Real inv_denominator =
            1.0 /
            (1.0 + k_bs2 * abs_gamma * dt_incr);


        // Every element of this row is explicitly overwritten.

        for (unsigned int ib = 0; ib < _num_slip_sys; ++ib)
        {
            dbsdgb[ia][ib] =
                prefactor *
                (
                    drhomdgb[ia][ib]
                    + drhoidgb[ia][ib]
                );
        }

        dbsdgb[ia][ia] +=
              k_bs1
            * krhoGb
            * dt_incr
            * sqrt_rho
            * sign_tau
            * sign_gamma

            - k_bs2
            * dt_incr
            * bstress[ia]
            * sign_gamma;

        for (unsigned int ib = 0; ib < _num_slip_sys; ++ib)
            dbsdgb[ia][ib] *= inv_denominator;
    }


    // ================================================================
    // Derivative of athermal slip resistance wrt glide slip rate
    // All derivative matrices remain Ns x Ns.
    // ================================================================

    const Real gp_raw =
        (_gp[_qp]);


    for (unsigned int ia = 0; ia < _num_slip_sys; ++ia)
    {


        // ------------------------------------------------------------
        // This summation depends only on ia.
        // Do it ONCE per row instead of Ns times.
        // ------------------------------------------------------------

        Real sum1 = 0.0;

        for (unsigned int ib1 = 0;
            ib1 < _num_slip_sys;
            ++ib1)
        {
            sum1 +=
                A[ia][ib1]
                * (rho_m[ib1] + rho_i[ib1]);
        }

        const Real sqrt_sum1 =
            std::sqrt(sum1);

        const Real inv_sqrt_sum1 =
            1.0 / sqrt_sum1;

        const Real s_a1 =
            krhoGb * sqrt_sum1;


        // ------------------------------------------------------------
        // Irradiation-loop term also depends only on ia.
        // ------------------------------------------------------------

        Real s_a2 = 1e-18;
        Real loop_term = 1.0;

        if (_irad_defects)
        {
            loop_term =
                std::sqrt(
                    Nloop[ia] * dloop[ia]);

            s_a2 =
                0.5
                * G
                * b_mag
                * loop_term;
        }


        const Real s_a_res =
            std::sqrt(
                s_a1*s_a1
                + s_a2*s_a2);

        const Real t2 =
            s_a1 / s_a_res;

        const Real t3 =
            s_a2 / s_a_res;


        // ------------------------------------------------------------
        // Only genuinely ib-dependent operations remain here.
        // Matrices stay NxN.
        // ------------------------------------------------------------

        for (unsigned int ib = 0;
            ib < _num_slip_sys;
            ++ib)
        {
            dsa1dgb[ia][ib] =
                0.5
                * krhoGb
                * A[ia][ib]
                * (
                    drhomdgb[ia][ib]
                    + drhoidgb[ia][ib]
                  )
                * inv_sqrt_sum1;


            if (_irad_defects)
            {
                dsa2dgb[ia][ib] =
                    0.5
                    * G
                    * b_mag
                    * (
                        dNldgb[ia][ib]
                        + ddldgb[ia][ib]
                      )
                    / loop_term;
            }
            else
            {
                dsa2dgb[ia][ib] = 0.0;
            }


            dsadgb[ia][ib] =
                (
                    t2*dsa1dgb[ia][ib]
                    + t3*dsa2dgb[ia][ib]
                )
                * gp_raw;
        }
    }

// Climb Model Derivatives
  if(_climbmodel){
    for (unsigned int ia = 0; ia <_num_slip_sys; ia++) {
          drhomdgc[ia][ia] = -k_c*rho_m[ia]*sgn(gamma_dot_c[ia])*dt_incr;
    }


    // drhoidgc 
    for (unsigned int ia = 0; ia < _num_slip_sys; ia++){
      drhoidgc[ia][ia] =  (- k_D*rho_i[ia])*sgn(gamma_dot_c[ia])*dt_incr;
    }


    // Calculate dvcdgc
    // Calculate dependent terms 
    for (unsigned int ia = 0; ia < _num_slip_sys; ia++) {
      for (unsigned int ib = 0; ib < _num_slip_sys; ib++) {
        dzidgc[ia][ib] = (Zsi/G) * dEtadGc[ia][ib];
        dzvdgc[ia][ib] = 0.0;
        Real x1 = (Ecorr * atomvol) / (B_k * _temp[_qp]);
        dcvth0dgc[ia][ib] = vacdensity_0[ia] * dEtadGc[ia][ib] * x1;
      }
    }


    // dvcdgc
    for (unsigned int ia = 0; ia < _num_slip_sys; ia++) {
      for (unsigned int ib = 0; ib< _num_slip_sys; ib++) {
        dvcdgc[ia][ib] = atomvol/b_mag * ( (dzidgc[ia][ib] * Di * interdensity) -
        - (zv[ia]*Dv*(0- dcvth0dgc[ia][ib])) );
      }
    }

  }


    // ================================================================
    // Optimized construction of glide part of Newton Jacobian
    // All derivative arrays remain full Ns x Ns matrices.
    // ================================================================

    const Real thermal_base =
        delF0 / (B_k * _temp[_qp]);

    for (unsigned int ia = 0;
        ia < _num_slip_sys;
        ++ia)
    {
        tau_eff[ia] =
            tau[ia] - bstress[ia];

        const Real abs_tau_eff =
            std::abs(tau_eff[ia]);

        const Real sign_tau_eff =
            sgn(tau_eff[ia]);


        // ============================================================
        // Slip law 2
        // ============================================================

        if (_sliplaw == 2)
        {
            Real x1 =
                (abs_tau_eff - s_a[ia])
                / s_t[ia];


            const Real x1_p =
                power(x1, p);

            const Real qterm =
                power(
                    1.0 - x1_p,
                    q - 1.0);

            const Real pterm =
                power(
                    x1,
                    p - 1.0);

            const Real inv_st =
                1.0 / s_t[ia];


            const Real row_prefactor =
                -gamma_dot_g[ia]
                * p
                * q
                * thermal_base
                * qterm
                * pterm;


            for (unsigned int ib = 0;
                ib < _num_slip_sys;
                ++ib)
            {
                array3[ia][ib] =
                    row_prefactor
                    *
                    (
                        (
                            (
                                dTaudGd[ia][ib]
                                - dbsdgb[ia][ib]
                            )
                            * sign_tau_eff

                            - dsadgb[ia][ib] // - (abs(tau_eff[ia]) - s_a[ia])*dstdgb[ia][ib]/(s_t[ia]*s_t[ia])); Removed as dstdgb = 0.0
                        )
                        * inv_st
                    );
            }
        }


        // ============================================================
        // Slip law 1 
        // ============================================================

        else if (_sliplaw == 1)
        {
            Real x2 =
                abs_tau_eff / s_a[ia];

            if (x2 == 0.0)
                x2 = 1e-18;


            const Real x3 =
                std::pow(x2, p);

            const Real x4 =
                std::pow(x2, p - 1.0);

            const Real x5 =
                q * std::pow(
                    1.0 - x3,
                    q - 1.0);

            const Real x6 =
                -p * x4;


            // No pow(s_a,2)

            const Real inv_sa2 =
                1.0 /
                (s_a[ia] * s_a[ia]);


            const Real row_prefactor =
                gamma_dot_g[ia]
                * thermal_base
                * x5
                * x6
                * inv_sa2;


            for (unsigned int ib = 0;
                ib < _num_slip_sys;
                ++ib)
            {
                const Real x7 =
                      s_a[ia]*
                      (
                          dTaudGd[ia][ib]
                        - dbsdgb[ia][ib]
                      )
                    
                    * sign_tau_eff
                    - tau_eff[ia]
                    * dsadgb[ia][ib];


                array3[ia][ib] =
                    row_prefactor * x7;
            }
        }


        array3[ia][ia] += 1.0;
    }

    if(_climbmodel){

      //Climb Systems

      for(unsigned int ia = _num_slip_sys; ia< (totalsys); ia++){
        int ij = ia - _num_slip_sys;
        for(unsigned int ib= _num_slip_sys; ib< (totalsys); ib++){
          int ik = ib - _num_slip_sys;
          array3[ia][ib] = - (   (drhomdgc[ij][ik]*b_mag*vel_climb[ij])
                              + (rho_m[ij]*b_mag*dvcdgc[ij][ik]) );
        }
        array3[ia][ia] = array3[ia][ia] + 1;
      }
    }


    for(unsigned int j = 0; j < (totalsys); j++) {
      gradient[j] = 0.0;
      for(unsigned int i = 0; i < (totalsys); i++) {
        gradient[j] = gradient[j] + residual[i]*array3[i][j];
      }
      gradient[j] = 2.0*gradient[j];
    }


    // Optimized Version
// ============================================================================
// GLIDE BLOCK
//
//      Jg * delta_g = -R_g
// ============================================================================

for (unsigned int i = 0;
     i < _num_slip_sys;
     ++i)
{
    rhs_g(i) =
        -residual[i];

    for (unsigned int j = 0;
         j < _num_slip_sys;
         ++j)
    {
        Jg(i,j) =
            array3[i][j];
    }
}

Jg.lu_solve(
    rhs_g,
    delta_g);

for (unsigned int i = 0;
     i < _num_slip_sys;
     ++i)
{
    d_gamma_dot[i] =
        delta_g(i);
}


// ============================================================================
// CLIMB BLOCK
//
//      Jc * delta_c = -R_c
// ============================================================================

if (_climbmodel)
{
    for (unsigned int i = 0;
         i < _num_slip_sys;
         ++i)
    {
        const unsigned int ii =
            _num_slip_sys + i;

        rhs_c(i) =
            -residual[ii];

        for (unsigned int j = 0;
             j < _num_slip_sys;
             ++j)
        {
            const unsigned int jj =
                _num_slip_sys + j;

            Jc(i,j) =
                array3[ii][jj];
        }
    }


    Jc.lu_solve(
        rhs_c,
        delta_c);


    for (unsigned int i = 0;
         i < _num_slip_sys;
         ++i)
    {
        d_gamma_dot[
            _num_slip_sys + i
        ] =
            delta_c(i);
    }
}


    // Check to make sure that N-R step leads 'down hill' the sse surface
    Real sum1 = 0.0;
    for(unsigned int i = 0; i < (totalsys); i++) {
      sum1 = sum1 - gradient[i]*d_gamma_dot[i];
    }

    if(sum1 > 0.0) {
      for(unsigned int i = 0; i < (totalsys); i++) {
        d_gamma_dot[i] = -1*d_gamma_dot[i];
      }
    }


    // Multiply step size by two 'cause next loop will divide it by 2
    for(unsigned int k = 0; k < (totalsys); k++) {
      d_gamma_dot[k] = d_gamma_dot[k]*2;
    }


    // Begin line search
    improved = false;

    N_ctr = 0;

    do {
      sse_old = sse;


      // Divide step size by 2

      for (unsigned int k = 0; k < (_num_slip_sys); k++) {
        d_gamma_dot[k] = d_gamma_dot[k]/2.0;
        gamma_try_g[k] = gamma_dot_g[k] + d_gamma_dot[k];
      }

      if(_climbmodel){
        for (unsigned int k = 0; k < (_num_slip_sys); k++){
          int j = k + _num_slip_sys;
          d_gamma_dot[j] = d_gamma_dot[j]/2.0;
          gamma_try_c[k] = gamma_dot_c[k] + d_gamma_dot[j];
        }
      }
      else {
        for (unsigned int k = 0; k < (_num_slip_sys); k++){
          gamma_try_c[k] = 0;
        }
      }


      NR_residual (
        _num_slip_sys,
        xs0,
        xm0,
        _temp[_qp],
        dt_incr,
        gamma_try_g,
        gamma_try_c,
        F1,
        F_el,
        F_p_inv,
        F_p_inv_0,
        C,
        rho_m0,
        rho_m,
        rho_i0,
        rho_i,
        bstress0,
        bstress,
        Nloop0,
        Nloop,
        dloop0,
        dloop,
        sig,
        tau,
        climbstress,
        tau_eff,
        s_a,
        s_t,
        A,
        H,
        interdensity,
        vacdensity,
        vacdensity_th,
        Di,
        Dv,
        nr_workspace,
        residual,
        sse);

      if ((sse_old <= sse_ref) && (sse >= sse_old) && (N_ctr > 0)) {
        improved = true;

        break;
      }

      N_ctr++;

    } while ((improved == false) && (N_ctr < max_loops));


    // add d_gamma_dot to gamma_dot to get new values for this iteration

    for (unsigned int k = 0; k < _num_slip_sys; k++) {
      gamma_dot[k] = gamma_dot[k] + d_gamma_dot[k]*2.0;
      gamma_dot_g[k] = gamma_dot_g[k] + d_gamma_dot[k]*2.0;
    }

    if(_climbmodel){
      for (unsigned int k = 0; k < _num_slip_sys; k++) {
        int j = k + _num_slip_sys;
        gamma_dot_c[k] = gamma_dot_c[k] + d_gamma_dot[j]*2.0;
      }
    }

// Re-evaluate the full constitutive state at the finally accepted line-search
// slip/climb rates before testing convergence.  This keeps Fp^{-1}, stress,
// dislocation state and residual synchronized at the accepted point.

    NR_residual (
      _num_slip_sys,
      xs0,
      xm0,
      _temp[_qp],
      dt_incr,
      gamma_dot_g,
      gamma_dot_c,
      F1,
      F_el,
      F_p_inv,
      F_p_inv_0,
      C,
      rho_m0,
      rho_m,
      rho_i0,
      rho_i,
      bstress0,
      bstress,
      Nloop0,
      Nloop,
      dloop0,
      dloop,
      sig,
      tau,
      climbstress,
      tau_eff,
      s_a,
      s_t,
      A,
      H,
      interdensity,
      vacdensity,
      vacdensity_th,
      Di,
      Dv,
      nr_workspace,
      residual,
      sse);

    if (sse <= _tol)
      converged = true;

    if ((sse > (sse_ref/2.0)) && (converged == false)) {
      _ever_subdivided = true;
      subdivide_increment = true;
      N_incr = 2*N_incr - 1;
      N_incr_total = 2*N_incr_total;
      break;
    }

  } while((iNR < 10000) && (converged == false));

// A failed constitutive trial is never committed.  Subdivision retries the same
// physical interval with a smaller dt while preserving the last converged
// beginning-of-subincrement state.

  if (subdivide_increment)
    continue;

    if (iNR == 10000) {

      std::cout << "\n SSE: \n";
      std::cout << sse;
      std::cout << "\n SSE_OLD ";
      std::cout << sse_old;
      std::cout << "\n SSE_REF: ";
      std::cout << sse_ref;
throw std::runtime_error("Crystal plasticity Newton Raphson did not converge");
      return;
  }


  // if another subincrement remains to be done, then reinitialize F_p_inv_0 and state variables.  F0 and F1 gets reinitialized back at the top of this loop
  if (N_incr < N_incr_total) {
    if (N_incr == (N_incr/2)*2) {
      N_incr = N_incr/2 + 1;
      N_incr_total = N_incr_total/2;
    }
    else {
      N_incr = N_incr + 1;
    }

    for (unsigned int i = 0; i < 3; i++) {
      for (unsigned int j = 0; j < 3; j++) {
        F_p_inv_0(i,j) = F_p_inv(i,j);
      }
    }

    for (unsigned int k = 0; k < _num_slip_sys; k++) {
      rho_m0[k] = rho_m[k];
      rho_i0[k] = rho_i[k];
      bstress0[k] = bstress[k];

// Advance irradiation-loop history together with the other state variables
// after a successful subincrement so subsequent subincrements start from the
// converged loop density and diameter.

      Nloop0[k] = Nloop[k];
      dloop0[k] = dloop[k];
    }
  }
  else if (N_incr == N_incr_total) {
    break;
  }
  } while (N_incr_total < 10000);

  if (N_incr_total > 10000) {
throw std::runtime_error("Crystal plasticity time step subincrement did not converge for element no " + std::to_string(_grainid) + " ");
    return;
  }


  // ==========================================================================
  // IMPLICIT FUNCTION THEOREM (single successful constitutive subincrement)
  // ==========================================================================
  // The map is R(z,F;q_n)=0, P(z,F;q_n) with z=[glide rates, climb rates].
  // We differentiate the ACTUAL NR_residual at the converged z. Each finite
  // difference starts from identical committed subincrement variables.
  // A = P_F - P_z R_z^{-1} R_F. The sensitivity solve is exact algebraically,
  // but R_z,R_F,P_z,P_F are computed numerically in this first implementation.
  // Multiple successful subincrements require history sensitivity propagation;
  // the adapter automatically uses full-update central differences instead.
  if (_request_ift && !_ever_subdivided)
  {
    const unsigned int ns = _num_slip_sys;
    const unsigned int nz = _climbmodel ? 2u * ns : ns;
    std::vector<Real> z(nz);
    for (unsigned int i = 0; i < ns; ++i)
    {
      z[i] = gamma_dot_g[i];
      if (_climbmodel) z[ns + i] = gamma_dot_c[i];
    }

    // Copy the converged iteration's end-state guesses. NR_residual()
    // updates its trial state on every call, so never reuse mutable copies
    // between sensitivity perturbations.
    const auto rm_seed = rho_m;
    const auto ri_seed = rho_i;
    const auto bs_seed = bstress;
    const auto nl_seed = Nloop;
    const auto dl_seed = dloop;
    const RankTwoTensor Fp0_seed = F_p_inv_0;
    const auto rm0_seed = rho_m0;
    const auto ri0_seed = rho_i0;
    const auto bs0_seed = bstress0;
    const auto nl0_seed = Nloop0;
    const auto dl0_seed = dloop0;

    auto evaluation = [&](const std::vector<Real> & zz,
                          const RankTwoTensor & FF) -> DDCP4P2IFT::LocalEvaluation
    {
      auto rm = rm_seed;
      auto ri = ri_seed;
      auto bs = bs_seed;
      auto nl = nl_seed;
      auto dl = dl_seed;
      auto rm0 = rm0_seed;
      auto ri0 = ri0_seed;
      auto bs0 = bs0_seed;
      auto nl0 = nl0_seed;
      auto dl0 = dl0_seed;
      RankTwoTensor Fp0 = Fp0_seed;
      RankTwoTensor Fe, Fpinv, sigma;
      std::vector<Real> gg(ns), gc(ns, 0.), tau_loc(ns), eta_loc(ns),
                        tau_eff_loc(ns), sa_loc(ns), st_loc(ns), R(nz);
      for (unsigned int i = 0; i < ns; ++i)
      {
        gg[i] = zz[i];
        if (_climbmodel) gc[i] = zz[ns+i];
      }
      NRWorkspace work(ns);
      Real score = 0.;
      NR_residual(ns, xs0, xm0, _temp[_qp], dt_incr, gg, gc, FF,
                  Fe, Fpinv, Fp0, C, rm0, rm, ri0, ri, bs0, bs,
                  nl0, nl, dl0, dl, sigma, tau_loc, eta_loc, tau_eff_loc,
                  sa_loc, st_loc, A, H, interdensity, vacdensity,
                  vacdensity_th, Di, Dv, work, R, score);
      if (!(FF.det() > 0.))
        throw std::runtime_error("IFT: det(F) <= 0");
      DDCP4P2IFT::LocalEvaluation output;
      output.residual = std::move(R);
      output.P = FF.det() * sigma * FF.inverse().transpose();
      return output;
    };
    try
    {
      const auto ref = evaluation(z, F1);
      const RankTwoTensor nominalP =
          F1.det() * sig * F1.inverse().transpose();
      Real mismatch = 0., magnitude = 0.;
      for (unsigned int i=0; i<3; ++i)
        for (unsigned int j=0; j<3; ++j)
        {
          const Real d = ref.P(i,j) - nominalP(i,j);
          mismatch += d*d;
          magnitude += nominalP(i,j)*nominalP(i,j);
        }
      const Real relative = std::sqrt(mismatch) /
          std::max(std::sqrt(magnitude), Real(1.));
      if (!std::isfinite(relative) || relative > 1.e-4)
        throw std::runtime_error(
            "IFT fixed-state evaluator does not reproduce converged stress; "
            "local state iteration may not be at a fixed point");

      _A_implicit = DDCP4P2IFT::compute(z, F1, evaluation,
                                        _ift_F_step, _ift_rate_step);
      _ift_valid = true;
    }
    catch (const std::exception & e)
    {
      _ift_valid = false;
      _ift_status = e.what();
    }
  }
  else if (_request_ift)
    _ift_status = "local constitutive subincrementation used";

  // ==========================================================================
  // Reconstruct final converged elastic state before tangent evaluation
  // ==========================================================================

  F_el = F1 * F_p_inv;

  const Real Je_final = F_el.det();

  if (Je_final <= 0.0)
    ddcpError("Non-positive elastic Jacobian Je = ", Je_final);


  // Make sure stress used by the tangent corresponds to final F_el

  sig = computeCauchyStress(
      F_el,
      C,
      false);


  // Calculate the average Cauchy stress

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      sig_avg(i,j) = sig_avg(i,j) + sig(i,j);
    }
  }

  Real F_el_matrix[3][3];
  for (unsigned int i = 0; i < 3; i++){
    for (unsigned int j = 0; j < 3; j++) {
      F_el_matrix[i][j] = F_el(i,j);
    }
  }


  // Write out Euler Angles in Bunge Convention
  aa_dot_bb(F_el_matrix,dir_cos0,dir_cos);
  bunge_angles(dir_cos,psi);


  // Calculate average elasticity tensor
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int l = 0; l < 3; l++) {
          C_avg[i][j][k][l] = C[i][j][k][l];
        }
      }
    }
  }


  // Rotate xs0 and xm0 to current coordinates, called xs and xm
  F_el_inv = F_el.inverse();
  for (unsigned int n = 0; n < _num_slip_sys; n++) {
    for (unsigned int i = 0; i < 3; i++) {
      xs[i][n] = 0.0;
      xm[i][n] = 0.0;
      for (unsigned int j = 0; j < 3; j++) {
        xs[i][n] = xs[i][n] + F_el(i,j)*xs0[j][n];
        xm[i][n] = xm[i][n] + xm0[j][n]*F_el_inv(j,i);
      }
    }
  }


  // ==========================================================================
  // Recompute final resolved shear stresses using the converged elastic state
  // ==========================================================================

  for (unsigned int n = 0;
       n < _num_slip_sys;
       ++n)
  {
    tau[n] = 0.0;

    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        tau[n] +=
            xs[i][n]
            * xm[j][n]
            * sig(i,j);

    tau_eff[n] =
        tau[n] - bstress[n];
  }


  std::vector<Real> d_disl(_num_slip_sys);
  calc_mfp(
    _num_slip_sys,
    rho_m,
    rho_i,
    Nloop,
    dloop,
    H,
    d_disl);


// ============================================================================
// OPTIMIZED FINAL TANGENT CALCULATION
//
// Calculate:
//
//      ddpdsig = d(Dp) / d(sigma)
//
// Optimization:
//   1. Calculate slip-rate sensitivity once per slip system.
//   2. Calculate climb sensitivity once per slip system.
//   3. Precompute glide and climb projection tensors.
//   4. Assemble the fourth-order tensor using only multiplication/addition.
//
// The constitutive formulas are unchanged from the original implementation.
// ============================================================================
// ============================================================================
// 1. Initialize ddpdsig
// ============================================================================

for (unsigned int i = 0; i < 3; ++i)
{
    for (unsigned int j = 0; j < 3; ++j)
    {
        for (unsigned int k = 0; k < 3; ++k)
        {
            for (unsigned int l = 0; l < 3; ++l)
            {
                ddpdsig[i][j][k][l] = 0.0;
            }
        }
    }
}


// ============================================================================
// 2. Storage for system-level tangent quantities
//
// Mglide[n] =
//      xs_i xm_j + xm_i xs_j
//
// Mclimb[n] =
//      xs_i xs_j + xs_i xs_j
//
// The climb form intentionally preserves the CURRENT implementation.
// ============================================================================

// ============================================================================
// FINAL PLASTIC-RATE SENSITIVITY dDp/dsigma
//
// System-level slip/climb sensitivities are evaluated once and then assembled
// through symmetric Schmid/climb projection tensors.  This preserves the P1
// derivative formulas while avoiding repeated power and tensor calculations in
// the i-j-k-l loops.

using TangentTensor9 = std::array<Real, 9>;

std::vector<TangentTensor9> Mglide(_num_slip_sys);
std::vector<TangentTensor9> Mclimb(_num_slip_sys);

std::vector<Real> glide_sensitivity(
    _num_slip_sys,
    0.0);

std::vector<Real> climb_sensitivity(
    _num_slip_sys,
    0.0);


// ============================================================================
// 3. Precompute glide and climb projection tensors
// ============================================================================

for (unsigned int n = 0;
     n < _num_slip_sys;
     ++n)
{
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            const unsigned int ij =
                3 * i + j;


            // ------------------------------------------------------------
            // Glide symmetric projection
            //
            // Original:
            //
            // xs[i][n]*xm[j][n]
            // +
            // xm[i][n]*xs[j][n]
            // ------------------------------------------------------------

            Mglide[n][ij] =
                  xs[i][n] * xm[j][n]
                + xm[i][n] * xs[j][n];


            // ------------------------------------------------------------
            // Climb projection
            //
            // Original:
            //
            // xs[i][n]*xs[j][n]
            // +
            // xs[i][n]*xs[j][n]
            //
            // = 2 * xs[i][n]*xs[j][n]
            // ------------------------------------------------------------

            if (_climbmodel)
            {
                Mclimb[n][ij] =
                    2.0
                    * xs[i][n]
                    * xs[j][n];
            }
        }
    }
}


// ============================================================================
// 4. Common thermal quantity
//
// Previously this was recalculated inside:
//      i-j-k-l-n
//
// It is independent of all tensor indices and slip-system index.
// ============================================================================

const Real thermal_factor =
    delF0
    / (B_k * _temp[_qp]);


// ============================================================================
// 5. GLIDE TANGENT SENSITIVITY
//
// Calculate once for every slip system:
//
//       glide_sensitivity[n]
//
// rather than recalculating it for every i,j,k,l component.
// ============================================================================
// ============================================================================
// SLIP LAW 2
//
// Original sensitivity:
//
// gamma_dot_g[n]
// *
// p*q*(delF0/(Bk*T))
// *
// power(1 - power(x,p), q-1)
// *
// power(x,p-1)
// *
// sgn(tau_eff[n])
// /
// s_t[n]
//
// where
//
// x = (abs(tau_eff[n]) - s_a[n]) / s_t[n]
//
// NOTE:
// No additional clipping has been introduced here.
// This intentionally reproduces the existing ddpdsig formulation.
// ============================================================================

if (_sliplaw == 2)
{
    for (unsigned int n = 0;
         n < _num_slip_sys;
         ++n)
    {
        tau_eff[n] =
            tau[n] - bstress[n];


        const Real x =
            (
                std::abs(tau_eff[n])
                - s_a[n]
            )
            / s_t[n];


        // Calculate expensive powers ONCE per slip system

        const Real x_p =
            power(x, p);


        const Real q_term =
            power(
                1.0 - x_p,
                q - 1.0);


        const Real p_term =
            power(
                x,
                p - 1.0);


        glide_sensitivity[n] =
              gamma_dot_g[n]

            * p
            * q

            * thermal_factor

            * q_term
            * p_term

            * sgn(tau_eff[n])

            / s_t[n];
    }
}


// ============================================================================
// SLIP LAW 1
//
// Original sensitivity:
//
// gamma_dot_g[n]
// *
// (delF0/(Bk*T))
// *
// q
// *
// pow(1 - pow(x2,p), q-1)
// *
// p
// *
// pow(x2,p-1)
// *
// sgn(tau_eff[n])
// /
// s_a[n]
//
// where the CURRENT code uses:
//
//      x2 = abs(tau[n]) / s_a[n]
//
// Note carefully:
// This intentionally preserves tau[n] rather than replacing it by tau_eff[n].
// ============================================================================

else if (_sliplaw == 1)
{
    for (unsigned int n = 0;
         n < _num_slip_sys;
         ++n)
    {
        tau_eff[n] =
            tau[n] - bstress[n];


        Real x2 =
            std::abs(tau_eff[n])
            / s_a[n];


        if (x2 == 0.0)
        {
            x2 = 1e-18;
        }


        // Expensive powers evaluated only ONCE per system

        const Real x3 =
            std::pow(
                x2,
                p);


        const Real x4 =
            std::pow(
                x2,
                p - 1.0);


        const Real x5 =
            std::pow(
                1.0 - x3,
                q - 1.0);


        glide_sensitivity[n] =
              gamma_dot_g[n]

            * thermal_factor

            * q
            * x5

            * p
            * x4

            * sgn(tau_eff[n])

            / s_a[n];
    }
}


// ============================================================================
// 6. CLIMB TANGENT SENSITIVITY
//
// Original:
//
// (rho_m[m]*b_mag)
// *
// (atomvol/b_mag)
// *
// [
//      (Zsi/G)*Di*interdensity
//
//      +
//
//      zv[m]*Dv*vacdensity_0[m]
//      *(Ecorr*atomvol)/(Bk*T)
// ]
//
// Only zv[m], vacdensity_0[m] and rho_m[m] depend on the system.
// Everything else can be evaluated once.
// ============================================================================

if (_climbmodel)
{
    const Real climb_thermal_factor =
        (Ecorr * atomvol)
        / (B_k * _temp[_qp]);


    const Real zi_stress_factor =
        Zsi / G;


    const Real atomvol_over_b =
        atomvol / b_mag;


    // System-independent interstitial part

    const Real interstitial_term =
        zi_stress_factor
        * Di
        * interdensity;


    for (unsigned int m = 0;
         m < _num_slip_sys;
         ++m)
    {
        const Real vacancy_term =
            zv[m]
            * Dv
            * vacdensity_0[m]
            * climb_thermal_factor;


        climb_sensitivity[m] =
              (rho_m[m] * b_mag)

            * atomvol_over_b

            * (
                  interstitial_term
                + vacancy_term
              );
    }
}


// ============================================================================
// 7. Assemble ddpdsig
//
// At this stage all expensive:
//      pow()
//      power()
//      thermal factors
//      sign calculations
//      climb sensitivity calculations
//
// have already been evaluated.
//
// The inner loop now contains only multiplication and addition.
// ============================================================================
const Real tangent_scale =
    _dt / 4.0;
// ============================================================================
// 8. Scale ddpdsig
//
// Preserve the scaling from the original implementation:
//
//      ddpdsig *= dt / 4
//
// The factor 1/4 is retained exactly as in the current code.
// ============================================================================

for (unsigned int i = 0; i < 3; ++i)
{
    for (unsigned int j = 0; j < 3; ++j)
    {
        const unsigned int ij =
            3 * i + j;


        for (unsigned int k = 0; k < 3; ++k)
        {
            for (unsigned int l = 0; l < 3; ++l)
            {
                const unsigned int kl =
                    3 * k + l;


                Real tangent_value = 0.0;


                // ========================================================
                // Glide contribution
                // ========================================================

                for (unsigned int n = 0;
                     n < _num_slip_sys;
                     ++n)
                {
                    tangent_value +=
                          Mglide[n][ij]

                        * glide_sensitivity[n]

                        * Mglide[n][kl];
                }


                // ========================================================
                // Climb contribution
                // ========================================================

                if (_climbmodel)
                {
                    for (unsigned int m = 0;
                         m < _num_slip_sys;
                         ++m)
                    {
                        tangent_value +=
                              Mclimb[m][ij]

                            * climb_sensitivity[m]

                            * Mclimb[m][kl];
                    }
                }


                ddpdsig[i][j][k][l] =
                    tangent_value;
                ddpdsig[i][j][k][l] *=
                    tangent_scale;                    
            }
        }
    }
}






  // ==========================================================================
  // PHASE-FIELD CORRECTED ELASTIC TANGENT
  //
  // NONE or VOLDEV tension:
  //
  //        C_eff = g C
  //
  // VOLDEV compression:
  //
  //        C_eff = g C + (1-g) C:Pvol
  //
  // where
  //
  // Pvol_ijkl =
  //
  //        (1/3) Je^(2/3) delta_ij (Ce^-1)_kl
  //
  // ==========================================================================

// ============================================================================
// PHASE-FIELD-CORRECTED ELASTIC PART OF THE ALGORITHMIC TANGENT
//
// NONE or VOLDEV tension:
//
//   C_eff = g C
//
// VOLDEV compression:
//
//   C_eff = g C + (1-g)(C:Pvol)
//
//   (Pvol)_mnkl = (1/3) Je^(2/3) delta_mn (Ce^{-1})_kl.
//
// This current/global tangent is used in the final stress-strain Jacobian.  It
// is distinct from the C0-based local slip/climb stress derivative above.

  Real C_eff[3][3][3][3];

  const Real g_val =
      (_g[_qp]);

  const Real Je_tan =
      F_el.det();

  if (Je_tan <= 0.0)
    ddcpError(
        "Non-positive elastic Jacobian in tangent: Je = ",
        Je_tan);

  const bool voldev_compression =
      (_decomposition == Decomposition::voldev &&
       Je_tan <= 1.0);

  RankTwoTensor C_e_inv;
  Real Je23 = 0.0;

  if (voldev_compression)
  {

      // Elastic right Cauchy-Green tensor

      const RankTwoTensor C_e =
          F_el.transpose() * F_el;

      C_e_inv =
          C_e.inverse();

      Je23 =
          std::pow(
              Je_tan,
              2.0 / 3.0);
  }

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int l = 0; l < 3; ++l)
        {
            if (!voldev_compression)
            {


                // NONE decomposition
                // or VOLDEV tensile branch

                C_eff[i][j][k][l] =
                    g_val
                    * C_avg[i][j][k][l];
            }
            else
            {


                // ----------------------------------------------------------
                // (C : Pvol)_ijkl
                //
                // = 1/3 Je^(2/3)
                //   [ C_ij00 + C_ij11 + C_ij22 ]
                //   (Ce^-1)_kl
                // ----------------------------------------------------------

                const Real C_Pvol_ijkl =
                    (1.0 / 3.0)
                    * Je23
                    * (
                        C_avg[i][j][0][0]
                        + C_avg[i][j][1][1]
                        + C_avg[i][j][2][2]
                      )
                    * C_e_inv(k,l);

                C_eff[i][j][k][l] =
                      g_val
                    * C_avg[i][j][k][l]

                    + (1.0 - g_val)
                    * C_Pvol_ijkl;
            }
        }


  // End calculations over the integration point
  // Calculate Green strain

  E_tot = F1.transpose() * F1;
  E_tot(0,0) = E_tot(0,0) - 1.0;
  E_tot(1,1) = E_tot(1,1) - 1.0;
  E_tot(2,2) = E_tot(2,2) - 1.0;
  E_tot = 0.5 * E_tot;


  // Begin calculation of the Jacobian (the tangent stiffness matrix)
  // Store sig_avg() into sig()
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      sig(i,j) = sig_avg(i,j);
    }
  }


  // Output stress, strain, etc. for post processing
  Real sum1;
  sum1 = 0.0;
  for(unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      sum1 = sum1 + sig(i,j) * sig(i,j);
    }
  }


  // Calculate the inverse of F_el
  F_el_inv = F_el.inverse();


  // Multiply the 4th rank elastic stiffness tensor by the derivative of the plastic part of the rate of deformation tensor wrt sig_dot
  Real array6[3][3][3][3], array4[6][6], array4_inv[6][6];
  aaaa_dot_dot_bbbb(C_eff,ddpdsig,array6);

  std::vector<std::vector<Real>> array4_matrix(6, std::vector<Real>(6));
  std::vector<std::vector<Real>> array4_inv_matrix(6, std::vector<Real>(6));


  // Add 4th rank identity tensor to array6()
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int l = 0; l < 3; l++) {
          array6[i][j][k][l] = array6[i][j][k][l] + 0.5*(del[i][k]*del[j][l] + del[i][l]*del[j][k]);
        }
      }
    }
  }

  forth_to_Voigt(array6,array4);

  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = 0; j < 6; j++) {
      array4_matrix[i][j] = array4[i][j];
    }
  }


  // Method 2: MatrixTools::inverse
  MatrixTools::inverse(array4_matrix,array4_inv_matrix);

  for (unsigned int i = 0; i < 6; i++) {
    for (unsigned int j = 0; j < 6; j++) {
      array4_inv[i][j] = array4_inv_matrix[i][j];
    }
  }

  Voigt_to_forth(array4_inv,array6);
  //  Multiply Array6 by C, the elastic stiffness matrix to finally get the Jacobian, but as a 4th rank tensor.

  aaaa_dot_dot_bbbb (array6,C_eff,ddsdde_4th);

  for (unsigned int i = 0; i < 3; i++){
    for (unsigned int j = 0; j < 3; j++) {
      for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int l = 0; l < 3; l++){
          _Jacobian_mult[_qp](i,j,k,l) = ddsdde_4th[i][j][k][l];
        }
      }
    }
  }


 // F_el already corresponds to the final converged state.
  // Calling with converged=true also updates the elastic energy.

  _stress[_qp] =
      computeCauchyStress(
          F_el,
          C,
          true);

  _euler_ang[_qp](0) = psi[0];
  _euler_ang[_qp](1) = psi[1];
  _euler_ang[_qp](2) = psi[2];

  if (_t_step> 0){

  // checkpoint("Storing ISVs")

  int n = -1;

  // Store dir_cos0 SDV 1-9

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      n = n + 1;
      _state_var[_qp][n] = dir_cos0[i][j];
    }
  }


  // Store dir_cos SDV 10-18

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      n = n + 1;
      _state_var[_qp][n] = dir_cos[i][j];
    }
  }


  // Store the elastic part of F SDV 19-27

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      n = n + 1;
      _state_var[_qp][n] = F_el(i,j);
    }
  }


  // Store inverse of the plastic part of F SDV 28-36

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      n = n + 1;
      _state_var[_qp][n] = F_p_inv(i,j);
    }
  }


  // Plastic strain calculations

  RankTwoTensor F_p;
  F_p = F_p_inv.inverse();

  E_p = F_p.transpose() * F_p;
  E_p(0,0) = E_p(0,0) - 1.0;
  E_p(1,1) = E_p(1,1) - 1.0;
  E_p(2,2) = E_p(2,2) - 1.0;
  E_p = 0.5 * E_p;


  // Store E_p SDV 37-45

  for (unsigned int i = 0; i<3; i++) {
    for (unsigned int j = 0; j<3; j++) {
      n = n + 1;
      _state_var[_qp][n] = E_p(i,j);
    }
  }


  // Effective strain calculations

  Real s1 = 0.0; Real s2 = 0.0;

  for (unsigned int i = 0; i<3; i++) {
    for (unsigned int j = 0; j<3; j++) {
      s1 = s1 + E_tot(i,j)*E_tot(i,j);
      s2 = s2 + E_p(i,j)*E_p(i,j);
    }
  }


  // Effective total strain

  E_eff = sqrt((2./3.) * s1);


  // Effective plastic strain

  E_p_eff = sqrt((2./3.) * s2);


  // Store E_eff SDV 46

  n = n + 1;
  _state_var[_qp][n] = E_eff;


  // Store E_p_eff SDV 47

  n = n + 1;
  _state_var[_qp][n] = E_p_eff;


  // Store average values of defect densities and loop sizes SDV 48-49

  rho_m_avg = 0.0;
  rho_i_avg = 0.0;
  Real gamma_dot_avg = 0.0;
  Real gamma_dot_climb_avg = 0.0;
  Real climbfrac1 = 0, climbfrac2=0;
  for (unsigned int i = 0; i < _num_slip_sys; i++) {
    rho_m_avg = rho_m_avg + rho_m[i];
    rho_i_avg = rho_i_avg + rho_i[i];
    gamma_dot_avg = gamma_dot_avg + abs(gamma_dot[i]);
    gamma_dot_climb_avg = gamma_dot_climb_avg + abs(gamma_dot_c[i]);
  }
  rho_m_avg = rho_m_avg/_num_slip_sys;
  rho_i_avg = rho_i_avg/_num_slip_sys;
  climbfrac1 = (gamma_dot_climb_avg)/ (gamma_dot_avg + gamma_dot_climb_avg);

  gamma_dot_avg = gamma_dot_avg/_num_slip_sys;
  gamma_dot_climb_avg = gamma_dot_climb_avg/_num_slip_sys;


  n = n + 1;
  _state_var[_qp][n] = rho_m_avg;

  n = n + 1;
  _state_var[_qp][n] = rho_i_avg;


  // Store average dislocation glide rates SDV 50

  n = n + 1;
  _state_var[_qp][n] = gamma_dot_avg;


   // Store Elastic Strain Energy
  n = n+1;
 _state_var[_qp][n] = (_psie_active[_qp]);

 // Store Plastic Strain Energy
 //  0.1 = Taylor Quinney Factor
  n = n+1;
  _psip_active[_qp] = psip_active0;
  for (unsigned int i = 0; i < _num_slip_sys ; i++) {
    _psip_active[_qp] =  _psip_active[_qp] + 0.1* s_a[i] * abs(gamma_dot_g[i]) * _dt;
  }  
  _state_var[_qp][n] = (_psip_active[_qp]);



  // Store mobile dislocation density values SDV 52 + _num_slip_sys
  for (unsigned int i = 0; i < _num_slip_sys; i++) {
    n = n + 1;
    _state_var[_qp][n] = rho_m[i];
  }


  // Store immobile dislocation density values SDV 52 + _num_slip_sys*2
  for (unsigned int i = 0; i < _num_slip_sys; i++) {
    n = n + 1;
    _state_var[_qp][n] = rho_i[i];
  }


  // Store backstress values SDV 52 + _num_slip_sys*3
  for (unsigned int i = 0; i < _num_slip_sys; i++) {
    n = n + 1;
    _state_var[_qp][n] = bstress[i];
  }

  if(_irad_defects) {

   // Store Nloop values SDV 52 + _num_slip_sys*4
    for (unsigned int i = 0; i < _num_slip_sys; i++) {
      n = n + 1;
      _state_var[_qp][n] = Nloop[i];
    }


    // Store dloop values SDV 52 + _num_slip_sys*5
    for (unsigned int i = 0; i < _num_slip_sys; i++) {
      n = n + 1;
      _state_var[_qp][n] = dloop[i];
    }
  }


  // Store Dislocation Glide Rate SDV52 + _num_slip_sys*6
  for(unsigned int i = 0; i < _num_slip_sys; i++) {
    n = n+1;
    _state_var[_qp][n] = gamma_dot[i];
  }

  if(_climbmodel){

    // Store Dislocation Climb Rates SDV52 + _num_slip_sys*7
    for(unsigned int i = 0; i < _num_slip_sys; i++) {
      n = n+1;
      _state_var[_qp][n] = gamma_dot_c[i];
    }


    // Store average climb rates SDV52 + _num_slip_sys*7 + 1
    n = n+1;
    _state_var[_qp][n] = gamma_dot_climb_avg;


    // Store Effect of Climb rates SDV52 + _num_slip_sys*7 + 3
    n = n+1;
    _state_var[_qp][n] = climbfrac1;


  // Climbfrac calculations by method 2
  Real Lp_g[3][3], Lp_c[3][3];
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      Lp_g[i][j] = 0.0;
      Lp_c[i][j] = 0.0;
      for (unsigned int k = 0; k < _num_slip_sys; k++) {
        Lp_g[i][j] = Lp_g[i][j] +   xs0[i][k]*xm0[j][k]* gamma_dot[k];
        Lp_c[i][j] = Lp_c[i][j] +   xs0[i][k]*xs0[j][k]* gamma_dot_c[k];
      }
    }
  }

  Real Ep_g = 0, Ep_c = 0, sg= 0, sc = 0;
  for (unsigned int i = 0; i<3; i++) {
    for (unsigned int j = 0; j<3; j++) {
      sg = sg + Lp_g[i][j]*Lp_g[i][j];
      sc = sc + Lp_c[i][j]*Lp_c[i][j];
    }
  }
  Ep_g = sqrt((2./3.) * sg);
  Ep_c = sqrt((2./3.) * sc);
  climbfrac2 = Ep_c/(Ep_c + Ep_g);

  n = n+1;
  _state_var[_qp][n] = climbfrac2;

  }

}


  //Store Dislocation 
  // elasticity tensor

  for (unsigned int i = 0; i < 3; ++i){
    for (unsigned int j = 0; j < 3; ++j){
      for (unsigned int k = 0; k < 3; ++k){
	    for (unsigned int l = 0; l < 3; ++l){

	      // Double check

	      _Cel_cp[_qp](i,j,k,l) = C[i][j][k][l];
	    }
      }
    }
  }

}


// NR_residual()

// ============================================================================
// NR_residual()
// ============================================================================
// Evaluates the local constitutive residual at the CURRENT trial slip/climb
// rates. State quantities generated here are therefore n+1 trial quantities.
//
// Glide residual on system a:
//
//      Rg_a = gammadot_g,a - gammadot_g,a^(constitutive)
//
// Climb residual, when enabled:
//
//      Rc_a = gammadot_c,a - gammadot_c,a^(constitutive)
//
// The same routine is used by the main Newton iteration and the line search,
// guaranteeing that both evaluate the same constitutive mapping.
// ============================================================================
void DDCP4P2Point::NR_residual (
  unsigned int num_slip_sys,
  const std::vector<std::vector<Real>> &xs0,
  const std::vector<std::vector<Real>> &xm0,
  Real temp,
  Real dt,
  const std::vector<Real> & gdt,
  const std::vector<Real> & gdc,
  RankTwoTensor F1,
  RankTwoTensor &F_el,
  RankTwoTensor &F_p_inv,
  RankTwoTensor &F_p_inv_0,
  Real C[3][3][3][3],
  std::vector<Real> &rho_m0,
  std::vector<Real> &rho_m,
  std::vector<Real> &rho_i0,
  std::vector<Real> &rho_i,
  std::vector<Real> &bstress0,
  std::vector<Real> &bstress,
  std::vector<Real> &Nloop0,
  std::vector<Real> &Nloop,
  std::vector<Real> &dloop0,
  std::vector<Real> &dloop,
  RankTwoTensor &sig,
  std::vector<Real> &tau,
  std::vector<Real> &climbstress,
  std::vector<Real> &tau_eff,
  std::vector<Real> &s_a,
  std::vector<Real> &s_t,
  const std::vector<std::vector<Real>> & A,
  const std::vector<std::vector<Real>> & H,
  Real interdensity,
  Real vacdensity,
  Real vacdensity_th,
  Real Di,
  Real Dv,
  NRWorkspace &workspace,
  std::vector<Real> &residual,
  Real &sse){


Real xL_p_inter[3][3];

    /* ============================================================================
    * OLDER VERSION
    * ============================================================================
    *
    * These vectors were allocated every time NR_residual() was called.
    *
    * std::vector<Real> shr_g(num_slip_sys);
    * std::vector<Real> shr(num_slip_sys);
    * std::vector<Real> shr_c(num_slip_sys);
    *
    * std::vector<std::vector<Real>> xs(
    *     3,
    *     std::vector<Real>(num_slip_sys));
    *
    * std::vector<std::vector<Real>> xm(
    *     3,
    *     std::vector<Real>(num_slip_sys));
    *
    * ========================================================================== */


    /* ============================================================================
    * OPTIMIZED VERSION
    *
    * Reuse storage allocated once in computeQpStress().
    * No memory allocation occurs here.
    * ========================================================================== */
    auto &shr_g = workspace.shr_g;
    auto &shr   = workspace.shr;
    auto &shr_c = workspace.shr_c;
    auto &xs = workspace.xs;
    auto &xm = workspace.xm;


  // Calculate the plastic part of the velocity gradient in the intermediate configuration - Contribution of Glide & Climb Slip
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      xL_p_inter[i][j] = 0.0;
      for (unsigned int k = 0; k < num_slip_sys; k++) {
        xL_p_inter[i][j] = xL_p_inter[i][j] +
        xs0[i][k]*xm0[j][k]* gdt[k] +
        xs0[i][k]*xs0[j][k]* gdc[k];
      }
    }
  }


  // Calculate the Strain Rate from the velocity gradient
  Real xL_p_inter_s[3][3];
  Real edot{0.e0};
  for (unsigned int i = 0; i <3; i++){
    for (unsigned int j = 0; j <3 ; j++){
      xL_p_inter_s[i][j] = 0.5*(xL_p_inter[i][j] + xL_p_inter[j][i]);
    }
  }

  for (unsigned int i = 0; i <3; i++){
    for (unsigned int j = 0; j <3 ; j++){
      edot = edot + xL_p_inter_s[i][j]*xL_p_inter_s[i][j];
    }
  }
  edot = sqrt(edot);
  edot = sqrt(0.667e0)*edot;
  if (abs(edot) < 1e-18){
        edot = 1e-18;
  }


  // Begin calculation process of F_p_n+1 = exp(xL_p_inter*dt).F_p_n
  Real array1[3][3], array2[3][3];
  for(unsigned int i = 0; i < 3; i++) {
    for(unsigned int j = 0; j < 3; j++) {
      array1[i][j] = xL_p_inter[i][j]*dt;
    }
  }


  // Caculate Omega
  Real sum1 = 0;
  for(unsigned int i = 0; i < 3; i++) {
    for(unsigned int j = 0; j < 3; j++) {
      sum1 = sum1 + array1[i][j]*array1[i][j];
    }
  }
  Real omega = sqrt(0.5 * sum1);


  // Continue calculating intermediate stuff needed for F_p_n+1

  aa_dot_bb(array1,array1,array2);


 // ## Optimized Version

if (omega > 0.0)
{
    const Real omega2 =
        omega * omega;

    const Real coeff1 =
        std::sin(omega)
        / omega;

    const Real coeff2 =
        (1.0 - std::cos(omega))
        / omega2;


    for (unsigned int i = 0;
         i < 3;
         ++i)
    {
        for (unsigned int j = 0;
             j < 3;
             ++j)
        {
            array1[i][j] =
                  coeff1
                * array1[i][j]

                +

                  coeff2
                * array2[i][j];
        }
    }
}


// Add identity

for (unsigned int i = 0;
     i < 3;
     ++i)
{
    array1[i][i] += 1.0;
}

  // Finally multiply arrays to get F_p_inv at end of time step.
  RankTwoTensor array1_tensor, array2_tensor;

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      array1_tensor(i,j) = array1[i][j];
    }
  }

  array2_tensor = array1_tensor.inverse();
  F_p_inv = F_p_inv_0 * array2_tensor;


  // Multiply F() by F_p_inv() to get F_el()
  F_el = F1 * F_p_inv;
  if (F_el.det() <= 0.0)
    throw std::runtime_error("Non-positive elastic Jacobian in NR_residual");
  RankTwoTensor F_el_inv = F_el.inverse();


  // Rotate director vectors from intermediate configuration to the current configuration.
  for(unsigned int n = 0; n < num_slip_sys; n++) {
    for(unsigned int i = 0; i < 3; i++) {
      xs[i][n] = 0.0;
      xm[i][n] = 0.0;
      for(unsigned int j = 0; j < 3; j++) {
        xs[i][n] = xs[i][n] + F_el(i,j)*xs0[j][n];
        xm[i][n] = xm[i][n] + xm0[j][n]*F_el_inv(j,i);
      }
    }
  }


  // Evaluate the phase-field-degraded Cauchy stress for the current Newton trial state.
  sig = computeCauchyStress(F_el, C,

                                        false);


  // Calculate resolved shear stress for each slip system.
  for(unsigned int k = 0; k < num_slip_sys; k++) {
    tau[k] = 0.0e0;
    for(unsigned int i = 0; i < 3; i++) {
      for(unsigned int j = 0; j < 3; j++) {
        tau[k] = tau[k] + xs[i][k]*xm[j][k]*sig(i,j);
      }
    }
  }


  // Calculate resolved climb stress for each slip system
  if(_climbmodel){
    for (unsigned int k = 0; k < num_slip_sys; k++) {
      climbstress[k] = 0.0e0;
      for (unsigned int i = 0; i< 3; i++) {
        for(unsigned int j = 0; j < 3; j++) {
          climbstress[k] = climbstress[k] + xs[i][k]*xs[j][k]*sig(i,j);
        }
      }
    }
  }


  /* ============================================================================
  * OLDER VERSION
  * ============================================================================
  *
  * std::vector<Real> d_disl(num_slip_sys);
  * std::vector<Real> d_disl4(num_slip_sys);
  *
  * std::vector<Real> drhomdt(num_slip_sys);
  * std::vector<Real> drhoidt(num_slip_sys);
  * std::vector<Real> dbstressdt(num_slip_sys);
  * std::vector<Real> dNloopdt(num_slip_sys);
  * std::vector<Real> ddloopdt(num_slip_sys);
  *
  * ========================================================================== */


  /* ============================================================================
  * OPTIMIZED VERSION
  * ========================================================================== */  

  auto &d_disl     = workspace.d_disl;
  auto &drhomdt    = workspace.drhomdt;
  auto &drhoidt    = workspace.drhoidt;
  auto &dbstressdt = workspace.dbstressdt;
  auto &dNloopdt   = workspace.dNloopdt;
  auto &ddloopdt   = workspace.ddloopdt;


  // Mean free path is recalculated, but its MEMORY is reused.

// ============================================================================
// CURRENT/NEW-STATE MEAN FREE PATH
//
// The constitutive evolution is intended to be implicit in the n+1 state.
// Hence the rate equations use the current rho_m, rho_i, Nloop and dloop
// iterate.  The *_0 quantities remain the beginning-of-subincrement history
// values entering the backward-Euler update.

  calc_mfp(
      _num_slip_sys,
      rho_m,
      rho_i,
      Nloop,
      dloop,
      H,
      d_disl);


  // Current formulation uses the same mean free path for both quantities.

  const auto &d_disl4 = d_disl;
  if(!_climbmodel) k_c = 0;
  if (_climbmodel)
  {
      k_D2 =
          k_D2_0
          * std::pow(
              edot0 / edot,
              1.0 / n0);

      k_D = k_D2;
  }
  else
  {
      k_D2 = 0.0;
  }


  // Effective Mean Free Path for Dislocation Trap

  for(unsigned int ia = 0; ia < num_slip_sys; ia++) {


// Mobile dislocation-density evolution evaluated at the current state:
//
//   rho_m^{n+1} = rho_m^n + dt * rhodot_m(rho^{n+1}, gammadot)
//
// The annihilation and climb terms therefore use rho_m at the current iterate.

    drhomdt[ia] = ((k_M/b_mag/d_disl[ia]
                 - k_ann*R_c*rho_m[ia]/b_mag
                 - k_I/b_mag/d_disl4[ia])*abs(gdt[ia])
                 - k_c * abs(gdc[ia]) * rho_m[ia]);


// Immobile dislocation-density evolution is likewise evaluated from the
// current n+1 iterate, consistent with the Newton derivative formulation.

    drhoidt[ia] = (k_I/b_mag/d_disl4[ia]
                  - k_D*rho_i[ia])*abs(gdt[ia])
                  - k_D2 * rho_i[ia]* abs(gdc[ia]);


    if (_irad_defects) {

// Irradiation-loop number-density evolution uses the current loop size,
// current mobile density, and current slip/climb rates.

    dNloopdt[ia] = k_c*rho_m[ia]*abs(gdc[ia])/(std::acos(-1.0)*dloop[ia])
      - (1.2/(2*b_mag))*sqrt(Nloop[ia]*dloop[ia])*sqrt(rho_m[ia])*abs(gdt[ia]);

    Real zi1 = zi0 + (Zsi/G)*climbstress[ia];
    ddloopdt[ia] = (2.0/std::acos(-1.0))*(Ai0_by_i0)*(zi1* Di* rho_m[ia]* interdensity)/(Nloop[ia]*dloop[ia]);
    }

    else {
       dNloopdt[ia] = 0;
       ddloopdt[ia] = 0;
    }
  }

  for(unsigned int ia = 0; ia < num_slip_sys; ia++) {
    rho_m[ia] = rho_m0[ia] + drhomdt[ia]*dt;
    if(rho_m[ia] < 1.0e2) {
      rho_m[ia] = 1.0e2;
    }
    rho_i[ia] = rho_i0[ia] + drhoidt[ia]*dt;
    if(rho_i[ia] < 1.0e2) {
      rho_i[ia] = 1.0e2;
    }

    Nloop[ia] = Nloop0[ia] + dNloopdt[ia]*dt;
    if(Nloop[ia] < 1e-6) {
      Nloop[ia] = 1e-6;
    }

    dloop[ia] = dloop0[ia] + ddloopdt[ia]*dt;
    if (dloop[ia] < 1e-6) {
      dloop[ia] = 1e-6;
    }
  }


  // Update the Mean Free Paths
  // Calculate the back stress for each slip system.
  for(unsigned int ia = 0; ia < num_slip_sys; ia++) {

// Backstress evolution is evaluated from the current backstress and current
// dislocation densities so the residual and its intended new-state derivative
// correspond to the same constitutive discretization.

    dbstressdt[ia] = (k_bs1*k_rho*G*b_mag*sqrt(rho_m[ia] + rho_i[ia])*sgn(tau[ia] - bstress[ia]) - k_bs2*bstress[ia])*abs(gdt[ia]);

    bstress[ia] = bstress0[ia] + dbstressdt[ia]*dt;
  }


  // Calculate the athermal slip resistance for each slip system.

  calc_crss(
    _num_slip_sys,
    rho_m,
    rho_i,
    Nloop,
    dloop,
    s_a,
    A);


  // Take care of degradation
    const Real gp_raw1 =
        (_gp[_qp]);
   for(unsigned int ia = 0; ia < num_slip_sys; ia++) {
     s_a[ia] = s_a[ia]*gp_raw1;
  }


  // Calculate thermal slip resistance for each slip system.
  for(unsigned int ia = 0; ia < num_slip_sys; ia++) {
    s_t[ia] = frictional_stress;
  }


  // Calculate new slip rate for each slip system.
  for(unsigned int k = 0; k < num_slip_sys; k++) {
    tau_eff[k] = tau[k] - bstress[k];
  }

    calc_sliprate(
      _num_slip_sys,
      tau_eff,
      s_a,
      s_t,
      shr_g,
      shr);


    auto &zi           = workspace.zi;
    auto &zv           = workspace.zv;
    auto &Kis          = workspace.Kis;
    auto &Kvs          = workspace.Kvs;
    auto &vel_climb    = workspace.vel_climb;
    auto &vacdensity_0 = workspace.vacdensity_0;


  // (Vik) Calculate interstial and vacancy capture efficiencies and finite stress thermal equilibrium density of vacancies

  if(_climbmodel){

    calc_climbrate(
      _num_slip_sys,
      Di,
      Dv,
      vacdensity_th,
      interdensity,
      vacdensity,
      climbstress,
      zi,
      zv,
      Kis,
      Kvs,
      vacdensity_0,
      vel_climb,
      rho_m,
      rho_i,
      shr_c);


  }


  // (Vik) Calculate residual values

  for(unsigned int k = 0; k < num_slip_sys; k++) {
    residual[k] = gdt[k] - shr_g[k];
    if (residual[k] >= 1.0e100) {
      residual[k] = 1.0e100;
    }
    if (residual[k] <= -1.0e100) {
      residual[k] = -1.0e100;
    }

  }

  if(_climbmodel){
    for(unsigned int k = 0; k < num_slip_sys; k++){
      int kj = num_slip_sys + k;
      residual[kj] = gdc[k] -shr_c[k];
      if (residual[kj] >= 1.0e100) {
        residual[kj] = 1.0e100;
      }
      if (residual[kj] <= -1.0e100) {
        residual[kj] = -1.0e100;
      }

    }
  }


  // (Calculate root mean square error that is to be minimized to zero

  sse = 0.0e0;
  Real gamma_dot_max = abs(gdt[0]);
  if(_climbmodel){
    for(unsigned int k = 0; k < num_slip_sys; k++) {
      int j = num_slip_sys + k;
      sse = sse +
      abs(gdt[k])*(residual[k]*residual[k]) +
      abs(gdc[k])*(residual[j]*residual[j]);

      if(abs(gdt[k]) > gamma_dot_max) {
        gamma_dot_max = abs(gdt[k]);
      }

      if(abs(gdc[k]) > gamma_dot_max){
        gamma_dot_max = abs(gdc[k]);
      }

    }
    if (gamma_dot_max > 1.e-18) {
      int x = 2*num_slip_sys;
      sse = sqrt(sse/gamma_dot_max);
      sse = sse/x;
    }
  }
  else {
    for(unsigned int k = 0; k < num_slip_sys; k++) {
      sse = sse +
      abs(gdt[k])*(residual[k]*residual[k]);

      if(abs(gdt[k]) > gamma_dot_max) {
        gamma_dot_max = abs(gdt[k]);
      }
    }
    if (gamma_dot_max > 1.e-18) {
      int x = num_slip_sys;
      sse = sqrt(sse/gamma_dot_max);
      sse = sse/x;
    }

  }


}

void DDCP4P2Point::readPropsFile()
{
  _properties[0] = _data.props;
}

void DDCP4P2Point::assignProperties(){

  if (_num_props < 54)
    ddcpError("DDCPStressUpdate4P2 requires at least 54 material properties; num_props = ", _num_props);


  // Elastic constants

  C11P = _properties[_qp][0];
  C11_perK = _properties[_qp][1];
  C12P = _properties[_qp][2];
  C12_perK = _properties[_qp][3];
  C44P = _properties[_qp][4];
  C44_perK = _properties[_qp][5];
  GP = _properties[_qp][6];
  G_perK = _properties[_qp][7];

  C11 = C11P - C11_perK*_temp[_qp];
  C12 = C12P - C12_perK*_temp[_qp];
  C44 = C44P - C44_perK*_temp[_qp];
  G = GP - G_perK*_temp[_qp];


  b_mag = _properties[_qp][8];


  gammadot0g = _properties[_qp][9];
  enthalpy_const = _properties[_qp][10];
  p = _properties[_qp][11];
  q = _properties[_qp][12];


  tau0 = _properties[_qp][13];
  hp_coeff = _properties[_qp][14];

  if (!_has_grain_size){
    grain_size = _properties[_qp][15];
  }
  else {

    // if the GrainAreaSize object exists, then grain_size is taken directly from the mesh

    grain_size = _grain_size;
  }
  frictional_stress = _properties[_qp][16];
  p0 = _properties[_qp][17];
  k_rho = _properties[_qp][18];
  k_I = _properties[_qp][19];

  Alatent = _properties[_qp][20];


  rho_m_zero = _properties[_qp][21];
  rho_i_zero = _properties[_qp][22];
  d_disl_zero = _properties[_qp][23];
  k_M = _properties[_qp][24];
  R_c = _properties[_qp][25];
  k_ann = _properties[_qp][26];
  k_D = _properties[_qp][27];
  k_bs1 = _properties[_qp][28];
  k_bs2 = _properties[_qp][29];
  Di0 = _properties[_qp][30];
  Dv0 = _properties[_qp][31];
  Em_i = _properties[_qp][32];
  Em_v = _properties[_qp][33];
  G0 = _properties[_qp][34];
  zi0 = _properties[_qp][35];
  zv0 = _properties[_qp][36];
  Zsi = _properties[_qp][37];
  K0 = _properties[_qp][38];
  Kiv = _properties[_qp][39];
  atomvol = _properties[_qp][40];
  lc = _properties[_qp][41];
  Nv = _properties[_qp][42];
  Ecorr = _properties[_qp][43];
  k_D2_0 = _properties[_qp][44];
  edot0 = _properties[_qp][45];
  n0 = _properties[_qp][46];
  k_c = _properties[_qp][47];
  Nloop_zero = _properties[_qp][48];
  dloop_zero = _properties[_qp][49];
  beta_loop = _properties[_qp][50];
  Ai0_by_i0 = _properties[_qp][51];
  mfpSC =  _properties[_qp][52];
  alpha_loop = _properties[_qp][53];


  B_k = 1.3806503e-23;
  freq = 1e13;


  // adjusted to change units of product with stress to SI units

  act_vol = (pow(b_mag,3))*1.0e-3;


  if(_deltaH_eV){
    delF0 = enthalpy_const*1.6e-19;

  }
  else {

    // delF0 scaled to SI units

    delF0 = (enthalpy_const*G*1.0e6)*(pow(b_mag,3))*1.0e-9;
  }


}


// normalize to unit vector

void DDCP4P2Point::normalize_vector(Real *x, Real *y, Real *z) {
  Real length = sqrt(*x * *x + *y * *y + *z * *z);

  *x = *x/length;
  *y = *y/length;
  *z = *z/length;
}


// Rotate 4th order stiffness tensor

void DDCP4P2Point::rotate_4th(Real a[3][3], Real b[3][3][3][3], Real (&c)[3][3][3][3]) {
  Real d[3][3][3][3];

  for(unsigned int m = 0; m < 3; m++) {
    for(unsigned int n = 0; n < 3; n++) {
      for(unsigned int k = 0; k < 3; k++) {
        for(unsigned int l = 0; l < 3; l++) {
          d[m][n][k][l] = a[k][0]*(a[l][0]*b[m][n][0][0] + a[l][1]*b[m][n][0][1] + a[l][2]*b[m][n][0][2]) +
          a[k][1]*(a[l][0]*b[m][n][1][0] +a[l][1]*b[m][n][1][1] + a[l][2]*b[m][n][1][2]) +
          a[k][2]*(a[l][0]*b[m][n][2][0] +a[l][1]*b[m][n][2][1] + a[l][2]*b[m][n][2][2]);
        }
      }
    }
  }

  for(unsigned int i = 0; i < 3; i++) {
    for(unsigned int j = 0; j < 3; j++) {
      for(unsigned int k = 0; k < 3; k++) {
        for(unsigned int l = 0; l < 3; l++) {
          c[i][j][k][l] = a[i][0]*(a[j][0]*d[0][0][k][l] + a[j][1]*d[0][1][k][l] + a[j][2]*d[0][2][k][l]) +
          a[i][1]*(a[j][0]*d[1][0][k][l] + a[j][1]*d[1][1][k][l] + a[j][2]*d[1][2][k][l]) +
          a[i][2]*(a[j][0]*d[2][0][k][l] + a[j][1]*d[2][1][k][l] + a[j][2]*d[2][2][k][l]);
        }
      }
    }
  }
}


// 4th order tensor to Voigt notation

void DDCP4P2Point::forth_to_Voigt(Real a[3][3][3][3], Real (&b)[6][6]) {
  for(unsigned int i = 1; i <= 3; i++) {
    for(unsigned int j = i; j <= 3; j++) {
      unsigned int ia = i;
      if(i != j) {
        ia = 9 - i - j;
      }
      for(unsigned int k = 1; k <= 3; k++) {
        for(unsigned int l = k; l <= 3; l++) {
          unsigned int ib = k;
          if(k != l) {
            ib = 9 - k - l;
          }
          b[ia-1][ib-1] = a[i-1][j-1][k-1][l-1];
        }
      }
    }
  }
}


// Voigt tensor to 4th order tensor

void DDCP4P2Point::Voigt_to_forth(Real b[6][6], Real (&a)[3][3][3][3]) {
  for(unsigned int i = 1; i <= 3; i++) {
    for(unsigned int j = 1; j <= 3; j++) {
      unsigned int ia = i;
      if(i != j) {
        ia = 9 - i - j;
      }
      for(unsigned int k = 1; k <= 3; k++) {
        for(unsigned int l = 1; l <= 3; l++) {
          unsigned int ib = k;
          if(k != l) {
            ib = 9 - k - l;
          }
          a[i-1][j-1][k-1][l-1] = b[ia-1][ib-1];
          if(ia > 3) {
            a[i-1][j-1][k-1][l-1] = a[i-1][j-1][k-1][l-1]/2;
          }
          if(ib > 3) {
            a[i-1][j-1][k-1][l-1] = a[i-1][j-1][k-1][l-1]/2;
          }
        }
      }
    }
  }
}

void DDCP4P2Point::aaaa_dot_dot_bbbb(Real a[3][3][3][3], Real b[3][3][3][3], Real (&product)[3][3][3][3]) {

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int l = 0; l < 3; l++) {
          product[i][j][k][l] = 0.0;
          for (unsigned int m1 = 0; m1 < 3; m1++) {
            for (unsigned int m2 = 0; m2 < 3; m2++) {
              product[i][j][k][l] = product[i][j][k][l] + a[i][j][m1][m2]*b[m1][m2][k][l];
            }
          }
        }
      }
    }
  }
}

void DDCP4P2Point::aaaa_dot_dot_bb(Real a[3][3][3][3], Real b[3][3], Real (&product)[3][3]) {

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      product[i][j] = 0.0;
      for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int l = 0; l < 3; l++) {
          product[i][j] = product[i][j] + a[i][j][k][l]*b[k][l];
        }
      }
    }
  }
}


// Another Version of aaaa_dot_dot_bb which takes/gives input/output in form of RankTwoTensor

void DDCP4P2Point::aaaa_dot_dot_bb2(Real a[3][3][3][3], RankTwoTensor bT, RankTwoTensor (&productT)) {


  // Convert Rank Two Tensor into Matrix

    Real b[3][3], product[3][3];
    for (unsigned int i = 0; i < 3; i++) {
     for (unsigned int j = 0; j < 3; j++) {
       b[i][j] = bT(i,j);
    }
    }


  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      product[i][j] = 0.0;
      for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int l = 0; l < 3; l++) {
          product[i][j] = product[i][j] + a[i][j][k][l]*b[k][l];
        }
      }
    }
  }


  // Convert Product into RankTwoTensor

    for (unsigned int i = 0; i < 3; i++) {
     for (unsigned int j = 0; j < 3; j++) {
      productT(i,j) = product[i][j];
    }
    }

}

void DDCP4P2Point::aa_dot_bb(Real a[3][3], Real b[3][3], Real (&product)[3][3]) {

  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      product[i][j] = 0.0;
      for (unsigned int k = 0; k < 3; k++) {
        product[i][j] = product[i][j] + a[i][k]*b[k][j];
      }
    }
  }
}

Real DDCP4P2Point::aa_dot_dot_bb(Real a[3][3], Real b[3][3]) {

  Real product = 0.0e0;
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      product = product + a[i][j]*b[i][j];
    }
  }
  return product;
}


// calculate Euler angles in Bunge notation

void DDCP4P2Point::bunge_angles(Real (&array1)[3][3], Real (&psi0)[3]) {

  Real PI = 4.0*atan(1.0);
  Real phi_1, Phi, phi_2, sth;


  // From VPSC

  if (array1[2][2] >= 1.0e0) array1[2][2] = 1.0e0;
  if (array1[2][2] <= -1.0e0) array1[2][2] = -1.0e0;

  Phi = acos(array1[2][2]);

  if (abs(array1[2][2]) >= 0.99999) {
    phi_2 = 0.0;
    phi_1 = atan2(array1[0][1],array1[0][0]);
  }
  else {
    sth = sin(Phi);
    phi_2 = atan2(array1[0][2]/sth,array1[1][2]/sth);
    phi_1 = atan2(array1[2][0]/sth,-array1[2][1]/sth);
  }

  psi0[0] = 180.0e0*phi_1/PI;
  psi0[1] = 180.0e0*Phi/PI;
  psi0[2] = 180.0e0*phi_2/PI;
}


// return power (with sign)

Real DDCP4P2Point::power(Real x, Real y) {

  Real ans;

  if (x == 0.0e0) {
    if (y > 0.0e0) {
      ans = 0.0e0;
    }
    else if (y < 0.0e0) {
      ans = 1.0e300;
    }
    else {
      ans = 1.0e0;
    }
  }
  else {
    ans = y*log10(abs(x));
    if (ans > 300) {
      ans = 1.0e300;
    }
    else {
      ans = pow(10.0e0,ans);
    }
    if (x < 0.0e0) {
      ans = -ans;
    }
  }

  return ans;
}


// return signum of a real number

Real DDCP4P2Point::sgn(Real x) {

  Real ans;
  ans = 1.0;

  if (x < 0.0e0) {
    ans = -1.0;
  }

  return ans;
}


// return maximum of two scalars

Real DDCP4P2Point::max_val(Real a, Real b)
{
  if (a>b)
  {
    return a;
  }
  else
  {
    return b;
  }
}


      // Optimized Version for Explicit A

// ============================================================================
// calc_crss()
// ============================================================================
// Calculates athermal slip resistance from dislocation hardening,
// irradiation-loop hardening, solute-cluster hardening, lattice resistance,
// and the Hall-Petch term.
//
// Dislocation contribution on system a:
//
//      tau_rho,a = k_rho G b sqrt( Sum_b A_ab rho_b )
//
// with rho_b = rho_m,b + rho_i,b.
// ============================================================================
void DDCP4P2Point::calc_crss(
    unsigned int num_slip_sys,
    const std::vector<Real> & rho_m,
    const std::vector<Real> & rho_i,
    const std::vector<Real> & Nloop,
    const std::vector<Real> & dloop,
    std::vector<Real> & s_a,
    const std::vector<std::vector<Real>> & A)
{
    const Real krhoGb =
        k_rho * G * b_mag;

    const Real hp_term =
        hp_coeff / std::sqrt(grain_size);


    // ========================================================================
    // Total dislocation density
    // ========================================================================

    Real rho_total = 0.0;

    for (unsigned int k = 0;
         k < num_slip_sys;
         ++k)
    {
        rho_total +=
            rho_m[k]
            + rho_i[k];
    }


    // ========================================================================
    // Irradiation contributions are identical for every slip system
    // ========================================================================

    Real s_a2 = 0.0;
    Real s_a3 = 0.0;

    if (_irad_defects)
    {
        Real loop_sum = 0.0;

        for (unsigned int k = 0;
             k < num_slip_sys;
             ++k)
        {
            loop_sum +=
                Nloop[k]
                * dloop[k];
        }

        s_a2 =
            alpha_loop
            * G
            * b_mag
            * std::sqrt(loop_sum);

        s_a3 =
            alpha_loop
            * G
            * b_mag
            / mfpSC;
    }


    // ========================================================================
    // CRSS
    //
    // A has:
    //
    //      diagonal     = p0
    //      off-diagonal = p0*Alatent
    //
    // Therefore:
    //
    // sum_j A[i][j]*rho[j]
    //
    // = p0 * [ Alatent*rho_total
    //          + (1-Alatent)*rho_i ]
    // ========================================================================

    for (unsigned int ia = 0;
         ia < num_slip_sys;
         ++ia)
    {
        const Real rho_ia =
            rho_m[ia]
            + rho_i[ia];

        const Real sum1 =
            p0
            * (
                Alatent
                * rho_total

                +

                (1.0 - Alatent)
                * rho_ia
              );


        const Real s_a1 =
            krhoGb
            * std::sqrt(sum1);


        const Real s_a_res =
            std::sqrt(
                s_a1*s_a1
                + s_a2*s_a2
                + s_a3*s_a3);


        s_a[ia] =
            tau0
            + hp_term
            + s_a_res;
    }
}

      // ============================================================================
// calc_sliprate()
// ============================================================================
// Evaluates the constitutive glide rate from effective resolved shear stress,
// athermal resistance, thermal resistance, temperature, and activation energy.
// ============================================================================
void DDCP4P2Point::calc_sliprate(
        unsigned int num_slip_sys,
        std::vector<Real> &tau_eff,
        std::vector<Real> &s_a,
        std::vector<Real> &s_t,
        std::vector<Real> &gamma_dot_g,
        std::vector<Real> &gamma_dot){

        if(_sliplaw == 2) {

        for(unsigned int k = 0; k < _num_slip_sys; k++) {
          Real refgdg = gammadot0g;
          Real x1 = (abs(tau_eff[k]) - s_a[k])/s_t[k];


          if(x1 < 1.0) {
            gamma_dot_g[k] = refgdg*exp((-delF0/B_k/_temp[_qp])*power(1 - power(x1, p), q))*sgn(tau_eff[k]);
          }
           else {
            gamma_dot_g[k] = 1e-18;
          }
          gamma_dot[k] = gamma_dot_g[k];

        }
        }
        else if (_sliplaw == 1) {

          for(unsigned int k = 0; k < _num_slip_sys; k++) {
            Real refgdg = gammadot0g;
            Real x1 = delF0/(B_k*_temp[_qp]);
            Real x2 = abs(tau_eff[k])/s_a[k];
            Real x3 = pow(x2, p);
            Real x4 = pow(1 -x3, q);
            gamma_dot_g[k] = refgdg*exp(-x1*x4)*sgn(tau_eff[k]);
            gamma_dot[k] = gamma_dot_g[k];
          }
        }

        else {
          ddcpError("Invalid Parameter : _sliplaw");
        }

        }


      // ============================================================================
// calc_climbrate()
// ============================================================================
// Calculates stress-dependent point-defect capture, vacancy equilibrium,
// climb velocity, and the resulting climb shear-rate-like contribution.
// ============================================================================
void DDCP4P2Point::calc_climbrate(
        unsigned int num_slip_sys,
        Real Di,
        Real Dv,
        Real vacdensity_th,
        Real &interdensity,
        Real &vacdensity,
        std::vector<Real> &climbstress,
        std::vector<Real> &zi,
        std::vector<Real> &zv,
        std::vector<Real> &Kis,
        std::vector<Real> &Kvs,
        std::vector<Real> &vacdensity_0,
        std::vector<Real> &vel_climb,
        std::vector<Real> &rho_m,
        std::vector<Real> &rho_i,
        std::vector<Real> &gamma_dot_c){


        Real temp1{0}, temp2{0}, temp3{0};
        Real dummy1, dummy2;
        Real lg;


        for (unsigned int i = 0; i < _num_slip_sys ; i++) {
            zi[i] = zi0 + (Zsi/G) * climbstress[i];
            zv[i] = zv0;
            Kis[i] = abs(zi[i]) * Di;
            Kvs[i] = abs(zv[i]) * Dv;
            vacdensity_0[i] = vacdensity_th * exp((climbstress[i] * atomvol * Ecorr)/(B_k*_temp[_qp]));
            }


        for (unsigned int k = 0; k < _num_slip_sys ; k++) {
          temp1 = temp1 + zi[k]*Di*(rho_m[k] + rho_i[k]);
          temp2 = temp2 + zv[k]*Dv*(rho_m[k] + rho_i[k]);
          temp3 = temp3 + zv[k]*Dv*vacdensity_0[k]*(rho_m[k] + rho_i[k]);
        }

        if (K0 == 0) {
        interdensity = 0;
        vacdensity = vacdensity_th;
        }
        else {
        interdensity = (K0 / temp1);
        vacdensity = (K0 + temp3)/temp2;
        }


        for (unsigned int k = 0; k < _num_slip_sys ; k++) {
          dummy1 = zi[k] * Di * interdensity;
          dummy2 = zv[k] * Dv * (vacdensity - vacdensity_0[k] );
          vel_climb[k] = (atomvol/b_mag)* (dummy1 - dummy2);

          if(vel_climb[k] < 1e-25){
            vel_climb[k] = 1e-25;
        }
          gamma_dot_c[k] = rho_m[k] * b_mag  *vel_climb[k];

          if (gamma_dot_c[k] <= 1e-18){
            gamma_dot_c[k] = 1.0e-18; }
        }
     }


     // Optimized Version
// Taylor Hardening when All Coefficients are 1

// ============================================================================
// calc_mfp()
// ============================================================================
// Calculates the dislocation mean free path. In the active P2 formulation the
// total mobile + immobile density enters the forest contribution, with optional
// irradiation-loop and solute-cluster contributions added to the inverse MFP.
// ============================================================================
void DDCP4P2Point::calc_mfp(
    unsigned int num_slip_sys,
    const std::vector<Real> & rho_m,
    const std::vector<Real> & rho_i,
    const std::vector<Real> & Nloop,
    const std::vector<Real> & dloop,
    const std::vector<std::vector<Real>> & H,
    std::vector<Real> & d_disl)
{
    Real density_sum = 0.0;

    for (unsigned int k = 0;
         k < num_slip_sys;
         ++k)
    {
        density_sum +=
            rho_m[k] + rho_i[k];
    }


    Real inv_mfp =
        std::sqrt(density_sum);


    if (_irad_defects)
    {
        Real loop_sum = 0.0;

        for (unsigned int k = 0;
             k < num_slip_sys;
             ++k)
        {
            loop_sum +=
                Nloop[k] * dloop[k];
        }

        inv_mfp +=
            beta_loop * std::sqrt(loop_sum)
            + beta_loop / mfpSC;
    }


    const Real mfp =
        1.0 / inv_mfp;


    std::fill(
        d_disl.begin(),
        d_disl.end(),
        mfp);
}


      // ============================================================================
// computeCauchyStress()
// ============================================================================
// Dispatches to the selected phase-field stress split:
//
//   NONE   : sigma = g(d) sigma_intact
//   VOLDEV : active tensile/isochoric part degraded; compressive volumetric
//            contribution retained according to the implemented split.
// ============================================================================
RankTwoTensor DDCP4P2Point::computeCauchyStress(
        const RankTwoTensor  F_el,
        Real C[3][3][3][3],
        const bool converged
      )
      {
        RankTwoTensor cauchyStress;

        if(_decomposition == Decomposition::none)
          cauchyStress = computeCauchyStressNoDecomposition(F_el, C, converged);

        else if(_decomposition == Decomposition::voldev)
          cauchyStress = computeCauchyStressVolDevDecomposition(F_el, C, converged);

        else
          ddcpError("Unsupported decomposition type");

        return cauchyStress;
      }


      RankTwoTensor DDCP4P2Point::computeCauchyStressNoDecomposition(
        const RankTwoTensor F_el,
        Real C[3][3][3][3],
        const bool converged
      )
      {

      RankTwoTensor Spk2, sig, cauchyStress_intact;
      RankTwoTensor cauchyStress;
      RankTwoTensor I2(RankTwoTensor::initIdentity);
      RankTwoTensor E_el;


      // Calculate elastic Green strain

      E_el = 0.5*(F_el.transpose() * F_el - I2);


    aaaa_dot_dot_bb2(C,E_el,Spk2);


    // Convert from PK2 stress to Cauchy stress

    Real det = F_el.det();
    sig =  F_el * Spk2 * F_el.transpose();
    for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      sig(i,j) = sig(i,j)/det;
    }
    }

    cauchyStress_intact = sig;


    cauchyStress = (_g[_qp]) * cauchyStress_intact;


    // IF called after the convergence then provide the strain energy density

    if(converged){
       _psie_active[_qp] = 0.5*Spk2.doubleContraction(E_el);
    }


    return cauchyStress;

  }


  // Ref : Tu, X., Ray, A., & Ghosh, S., Engineering Fracture Mechanics 2020, 106970

  RankTwoTensor DDCP4P2Point::computeCauchyStressVolDevDecomposition(
    const RankTwoTensor F_el,
    Real C[3][3][3][3],
    const bool converged
  ){

RankTwoTensor Spk2a, Spk2i, sig_active,sig_inactive, cauchyStress, cauchyStress_intact;
    RankTwoTensor F_el_iso, F_el_vol;
    RankTwoTensor E_el_vol, E_el_iso, E_el_active, E_el_inactive;
    RankTwoTensor I2(RankTwoTensor::initIdentity);
    RankTwoTensor E_el;


    // Calculate elastic Green strain

    E_el = 0.5*(F_el.transpose() * F_el - I2);


    // Divide the F_el into Volumetric and Isoclinic Components

    Real det = F_el.det();
    F_el_iso = pow(det, -(1.0/3.0))*F_el;
    F_el_vol = pow(det, (1.0/3.0) )*I2;


    // Determine the Volumetric and Isochoric Components of Green Strain Tensor

    E_el_vol = 0.5*(F_el_vol.transpose() * F_el_vol - I2);
    E_el_iso = 0.5*(F_el_iso.transpose() * F_el_iso - I2);


    if(det - 1.0 > 0){
      E_el_active = (F_el_vol.transpose() * E_el_iso * F_el_vol + E_el_vol);
      E_el_inactive = 0;
    }
    else {
      E_el_active = (F_el_vol.transpose() * E_el_iso * F_el_vol);
      E_el_inactive = E_el_vol;
    }


    // Multiply the anisotropic stiffness tensor 
    // by the Green strain to get the 2nd Piola Kirkhhoff stress

    aaaa_dot_dot_bb2(C,E_el_active,Spk2a);


    // Convert from PK2 stress to Cauchy stress    

    sig_active =  F_el * Spk2a * F_el.transpose();
    for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      sig_active(i,j) = sig_active(i,j)/det;
    }
    }


    // Multiply the anisotropic stiffness tensor 
    // by the Green strain to get the 2nd Piola Kirkhhoff stress

    aaaa_dot_dot_bb2(C,E_el_inactive,Spk2i);


    // Convert from PK2 stress to Cauchy stress

    sig_inactive =  F_el * Spk2i * F_el.transpose();
    for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
      sig_inactive(i,j) = sig_inactive(i,j)/det;
    }
    }

    cauchyStress_intact = sig_active + sig_inactive;

    cauchyStress = (_g[_qp]) * sig_active + sig_inactive;

    if(converged){
      const Real psie_inactive = 0.5*Spk2i.doubleContraction(E_el_inactive);
      _psie_active[_qp] = 0.5*Spk2a.doubleContraction(E_el_active);
    }

    return cauchyStress;


  }
