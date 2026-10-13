#pragma once
// CPU material point integration preserving the original DDCP4P2 constitutive algorithms.
// Port of DDCPStressUpdate4P2.h: no FE quadrature-point or UserObject dependencies.
#include "RankTwoTensor.h"
#include "RankFourTensor.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include <vector>
#include <array>
#include <string>
#include <stdexcept>

struct DDCP4P2Data
{
  std::vector<Real> props;
  std::vector<std::array<Real, 3>> normals;
  std::vector<std::array<Real, 3>> directions;
  bool climb = false;
  bool irradiation = false;
  bool deltaH_eV = false;
  unsigned int slip_law = 1;
  unsigned int crss_law = 1;
  unsigned int decomposition = 0; // 0 NONE; 1 VOLDEV
  Real tolerance = 1e-6;
};

struct DDCP4P2Input
{
  RankTwoTensor F_old;
  RankTwoTensor F_new;
  RankTwoTensor stress_old;
  std::vector<Real> state_old;
  std::array<Real,3> euler_degrees{{0.,0.,0.}};
  int grain_id = 0; // Global grain ID from orientation.in; diagnostic only
  Real dt = 0.;
  Real temperature = 300.;
  Real g = 1.;
  Real gp = 1.;
  Real grain_size = 0.; // zero means use grain size from props file
  bool request_implicit_tangent = false;
  Real ift_F_step = 1e-6;
  Real ift_rate_step = 1e-5;
};

struct DDCP4P2Result
{
  RankTwoTensor sigma;
  RankTwoTensor P;
  RankFourTensor C_local;
  RankFourTensor A_implicit;
  bool implicit_tangent_available = false;
  bool used_constitutive_subincrements = false;
  std::string ift_status;
  std::vector<Real> state;
  Real psie = 0.;
  Real psip = 0.;
  std::array<Real,3> euler{{0.,0.,0.}};
};

class DDCP4P2Point
{
public:
  enum class Decomposition {none, voldev};
  explicit DDCP4P2Point(const DDCP4P2Data & data);
  DDCP4P2Result integrate(const DDCP4P2Input & input);
  static unsigned int stateCount(unsigned int ns, bool irradiation, bool climb);

private:
  unsigned int _qp = 0;
  int _t_step = 1;
  int _grainid = 0;
  unsigned int _num_slip_sys = 0;
  unsigned int _num_state_vars = 0;
  unsigned int _num_props = 0;
  Real _tol = 1e-6;
  Real _dt = 0.;
  Real _grain_size = 0.;
  bool _has_grain_size = false;
  bool _climbmodel = false;
  bool _irad_defects = false;
  bool _deltaH_eV = false;
  unsigned int _sliplaw = 1;
  unsigned int _crsslaw = 1;
  Decomposition _decomposition = Decomposition::none;
  const DDCP4P2Data & _data;
  bool _request_ift = false;
  bool _ift_valid = false;
  bool _ever_subdivided = false;
  std::string _ift_status;
  Real _ift_F_step = 1e-6;
  Real _ift_rate_step = 1e-5;
  RankFourTensor _A_implicit;
  std::array<Real,3> _input_euler{{0.,0.,0.}};
  std::vector<Real> _temp{300.};
  std::vector<Real> _g{1.};
  std::vector<Real> _gp{1.};
  std::vector<Real> _psie_active{0.};
  std::vector<Real> _psip_active{0.};
  std::vector<RankTwoTensor> _deformation_gradient{1};
  std::vector<RankTwoTensor> _deformation_gradient_old{1};
  std::vector<RankTwoTensor> _rotation_increment{1};
  std::vector<RankTwoTensor> _stress_old{1};
  std::vector<RankTwoTensor> _stress{1};
  std::vector<RankFourTensor> _Jacobian_mult{1};
  std::vector<RankFourTensor> _Cel_cp{1};
  struct EulerStore { std::array<Real,3> a; Real &operator()(int i){return a.at(i);} };
  std::vector<EulerStore> _euler_ang{1};
  std::vector<std::vector<Real>> _state_var{1};
  std::vector<std::vector<Real>> _state_var_old{1};
  std::vector<std::vector<Real>> _properties{1};
  std::vector<std::vector<Real>> _properties_old{1};
  std::vector<std::vector<std::vector<Real>>> _y{1};
  std::vector<std::vector<std::vector<Real>>> _y_old{1};
  std::vector<std::vector<std::vector<Real>>> _z{1};
  std::vector<std::vector<std::vector<Real>>> _z_old{1};
  std::vector<std::string> _eigenstrain_names;
  std::vector<const std::vector<RankTwoTensor>*> _eigenstrains;
  std::vector<const std::vector<RankTwoTensor>*> _eigenstrains_old;

  void computeQpStress();
  Real max_val(Real a, Real b);
  // parameters/variables used for calculations
  static const int max_loops = 20;
  void readPropsFile();
  void assignProperties();
  void normalize_vector(Real*, Real*, Real*);
  void rotate_4th(Real a[3][3], Real b[3][3][3][3], Real (&c)[3][3][3][3]);
  void forth_to_Voigt(Real a[3][3][3][3], Real (&b)[6][6]);
  void Voigt_to_forth(Real b[6][6], Real (&a)[3][3][3][3]);
  void aaaa_dot_dot_bbbb(Real a[3][3][3][3], Real b[3][3][3][3], Real (&product)[3][3][3][3]);
  void aaaa_dot_dot_bb(Real a[3][3][3][3], Real b[3][3], Real (&product)[3][3]);
  void aaaa_dot_dot_bb2(Real a[3][3][3][3], RankTwoTensor bT, RankTwoTensor (&productT));
  void aa_dot_bb(Real a[3][3], Real b[3][3], Real (&product)[3][3]);
  Real aa_dot_dot_bb(Real a[3][3], Real b[3][3]);
  void bunge_angles(Real (&array1)[3][3], Real (&psi0)[3]);
  Real power(Real x, Real y);
  Real sgn(Real x);

  // ============================================================================
// Workspace reused by NR_residual()
//
// These arrays are temporary calculation variables.  Their sizes do not
// change during computeQpStress(), so they are allocated once and reused
// during Newton iterations and line-search residual evaluations.
// ============================================================================
struct NRWorkspace
{
  explicit NRWorkspace(unsigned int n)
    : shr_g(n, 0.0),
      shr(n, 0.0),
      shr_c(n, 0.0),

      xs(3, std::vector<Real>(n, 0.0)),
      xm(3, std::vector<Real>(n, 0.0)),

      d_disl(n, 0.0),

      drhomdt(n, 0.0),
      drhoidt(n, 0.0),
      dbstressdt(n, 0.0),
      dNloopdt(n, 0.0),
      ddloopdt(n, 0.0),

      zi(n, 0.0),
      zv(n, 0.0),
      Kis(n, 0.0),
      Kvs(n, 0.0),
      vel_climb(n, 0.0),
      vacdensity_0(n, 0.0)
  {
  }

  // Slip-rate workspace
  std::vector<Real> shr_g;
  std::vector<Real> shr;
  std::vector<Real> shr_c;

  // Rotated slip directions/normals
  std::vector<std::vector<Real>> xs;
  std::vector<std::vector<Real>> xm;

  // Mean-free-path workspace
  std::vector<Real> d_disl;

  // State evolution rates
  std::vector<Real> drhomdt;
  std::vector<Real> drhoidt;
  std::vector<Real> dbstressdt;
  std::vector<Real> dNloopdt;
  std::vector<Real> ddloopdt;

  // Climb workspace
  std::vector<Real> zi;
  std::vector<Real> zv;
  std::vector<Real> Kis;
  std::vector<Real> Kvs;
  std::vector<Real> vel_climb;
  std::vector<Real> vacdensity_0;
};

  void NR_residual (
    unsigned int num_slip_sys, 
    const std::vector<std::vector<Real>> &xs0, 
    const std::vector<std::vector<Real>> &xm0, 
    Real temp, 
    Real dt, 
    const std::vector<Real> & gamma_dot, 
    const std::vector<Real> & gammadot_climb,
    RankTwoTensor F1, 
    RankTwoTensor &F_el, 
    RankTwoTensor &F_p_inv, 
    RankTwoTensor &F_p_inv0, 
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
    NRWorkspace & workspace,    // Reusable temporary storage
    std::vector<Real> &residual, 
    Real &sse);

  // Function for Calculation of CRSS
void calc_crss(
          unsigned int num_slip_sys,
          const std::vector<Real> & rho_m,
          const std::vector<Real> & rho_i,
          const std::vector<Real> & Nloop,
          const std::vector<Real> & dloop,
          std::vector<Real> & s_a,
          const std::vector<std::vector<Real>> & A);    


  // Function for Calculation of Derivative of CRSS
  void calc_dsadgb(
    unsigned int num_slip_sys,
    std::vector<Real> &rho_m,
    std::vector<Real> &rho_i,
    std::vector<Real> &Nloop,
    std::vector<Real> &dloop,
    std::vector<Real> &s_a,
    std::vector<std::vector<Real>> &A,
    std::vector<std::vector<Real>> &drhomdgb,
    std::vector<std::vector<Real>> &drhoidgb,
    std::vector<std::vector<Real>> &dNldgb,
    std::vector<std::vector<Real>> &ddldgb,
    std::vector<std::vector<Real>> &dsadgb);
    
  // Function for Calculation of Slip Rate
  void calc_sliprate(
    unsigned int num_slip_sys,
    std::vector<Real> &tau_eff,
    std::vector<Real> &s_a,
    std::vector<Real> &s_t,
    std::vector<Real> &gamma_dot_g,
    std::vector<Real> &gamma_dot);

  // Function for Calculation of Climb Rate
  void calc_climbrate(
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
    std::vector<Real> &gammadot_c);


  // Function for Calculation of Mean Free Path
void calc_mfp(
        unsigned int num_slip_sys,
        const std::vector<Real> & rho_m,
        const std::vector<Real> & rho_i,
        const std::vector<Real> & Nloop,
        const std::vector<Real> & dloop,
        const std::vector<std::vector<Real>> & H,
        std::vector<Real> & d_disl);    

    
    
  // Function for Calculation of Cauchy Stress
  RankTwoTensor computeCauchyStress(
    const RankTwoTensor F_el,
    Real C[3][3][3][3], 
    const bool converged
  );

  // Function for Calculation of Cauchy Stress No Decomposition
  RankTwoTensor computeCauchyStressNoDecomposition(
    const RankTwoTensor F_el,
    Real C[3][3][3][3], 
    const bool converged
  );


  // Function for Calculation of Cauchy VolDev Spectral Decomposition
  RankTwoTensor computeCauchyStressVolDevDecomposition(
    const RankTwoTensor F_el,
    Real C[3][3][3][3], 
    const bool converged
  );


  Real tolerance;

  Real act_vol;
  Real delF0;

  // Constants
  Real
  B_k,              // Boltzman Constant
  freq;             // Debyde Freq

  // Variables Inside Props File
  Real                  // Line no in Props File 
  C11P,                 //1 Elastic Constant
  C11_perK,             //2 Temp Variation of C11
  C12P,                 //3 Elastic Constant
  C12_perK,             //4 Temperature Variation of C12
  C44P,                 //5 Elastic Constant
  C44_perK,             //6 Temperature Variation of C44
  GP,                   //7 Shear Modulus
  G_perK,               //8 Shear Modulus Temp variation
  b_mag,                //9 Burgers Vector Magnitude
  gammadot0g,           //10 reference strain rate
  enthalpy_const,       //11 Multiplication Const for Enthalpy Term
  p,                    //12 Inner Exponent of Slip Law
  q,                    //13 Outer Exponent of Slip Law
  tau0,                 //14 Athermal Slip Resistance Constant
  hp_coeff,             //15 Hall Petch Coefficient
  grain_size,           //16 Grain Size
  frictional_stress,    //17 Lattice Frictional Resistance
  p0,                   //18 Dislocation Barrier Strength
  k_rho,                //19 Taylor Hardening Parameter
  k_I,                  //20 Mean Free Path Constant
  Alatent,              //21 Latent Hardening Coefficient
  rho_m_zero,           //22 Initial Mobile Dislocation Density
  rho_i_zero,           //23 Initial Immobile Dislocation Density
  d_disl_zero,          //24 Initial Dislocation Line Length
  k_M,                  //25 Dislocation Line Generation Constant
  R_c,                  //26 Critical Capture Radius for Dislocation Annhilation
  k_ann,                //27 Dislocation Evlution annhilation constant
  k_D,                  //28 Immobile Dislocation Evolution Dynamic Recovery Constant
  k_bs1,                //29 Backstress Evolution Constant 1
  k_bs2,                //30 Backstress Evolution Constant 2
  Di0,                  //31 Interstitial Diffusion Constant
  Dv0,                  //32 Vaccany Diffusion Constant
  Em_i,                 //33 Migration Enthalpy of Interstitials
  Em_v,                 //34 Migration Enthalpy of Vacancies
  G0,                   //35 Gibbs Free Energy of Vacancy Formation
  zi0,                  //36 Zero Stress Interstitial Capture Efficency
  zv0,                  //37 Zero Stress Vacacny Capture Efficency
  Zsi,                  //38 ClimbStress Propornality Constant of Interstitial Capture Efficency
  K0,                   //39 Point Defect Generation Rates
  Kiv,                  //40 Point Defect Recombination Rates
  atomvol,              //41 Atomic Volume
  lc,                   //42 Average Climb Distance
  Nv,                   //43 No of Atoms Per Unit Volume in the Alloy
  Ecorr,                //44 Correction Factor to Make Energy in J  
  k_D2_0,               //45 Immobile Dislocation Dynamic Recovery Coefficient Due to Climb
  edot0,                //46 Reference Strain Rate for Dynamic Recovery
  n0,                   //47 Dynamic Recovery strain rate exponent
  k_c,                  //48 Mean Free Path of Precipitates
  Nloop_zero,           //49 Initial Density of Dislocation Loops
  dloop_zero,           //50 Initial Diameter of Dislocation Loops
  beta_loop,            //51 Trapping Coefficient of Dislocation Loops
  Ai0_by_i0,            //52 Increase in Area of Dislocation Loop by Addition of a single interstial
  mfpSC,                //53 Hardening by Solute Clusters  
  alpha_loop;           //54 Alpha Loop used for Calculation of CRSS 



  Real k_D2;
  Real C11, C12, C44, G;






  

  Real sse;
};
