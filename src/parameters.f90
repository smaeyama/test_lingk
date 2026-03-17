MODULE parameters
!-------------------------------------------------------------------------------
!
!    Set parameters
!
!-------------------------------------------------------------------------------
  implicit none

!- constants for fortran -!
  integer, parameter :: DP = selected_real_kind(14)
  real(kind=DP), parameter :: pi = 3.141592653589793_DP
  real(kind=DP), parameter :: eps = 0.0000000001_DP
  complex(kind=DP), parameter :: ci = (0._DP, 1._DP)

!- constants for numerical calculation -!
  integer      , parameter :: litime = 1000     ! Iteration limit of time
  real(kind=DP), parameter :: elt_limit = 120._DP  ! Elapese time limit [sec]
  real(kind=DP), parameter :: time_limit = 10._DP  ! Simulation time limit
  integer      , parameter :: nz = 24*5            ! Grid number in zz direction
  integer      , parameter :: nv = 32              ! Grid number in vl direction
  integer      , parameter :: nm = 31              ! Grid number in mu direction
  integer      , parameter :: ns = 1               ! Number of species
  integer      , parameter :: nzb = 2              ! Number of species
  integer      , parameter :: nvb = 2              ! Number of species
  real(kind=DP), parameter :: lz = 5._DP * pi      ! Box size -lz <= zz < lz
  real(kind=DP), parameter :: lv = 4._DP           ! Box size -lv <= vl < lv
  real(kind=DP), parameter :: lm = 8._DP           ! Box size 0 <= mu <= lm
  real(kind=DP), parameter :: dt_out = 0.1_DP      ! Time step for output
  real(kind=DP)            :: dt = 0.01_DP         ! Time-step size
  logical      , parameter :: flag_dtc = .true.
  integer      , parameter :: flag_runs = 1

  real(kind=DP) :: kx, ky, eps_r, q_0, s_hat, lambda, beta
  real(kind=DP), dimension(0:ns-1) :: R0_Ln, R0_Lt, nu, Anum, Znum, fcs, sgn, tau

 CONTAINS

SUBROUTINE set_param
  namelist/physp/ kx, ky, eps_r, q_0, s_hat, lambda, beta,  &
                  R0_Ln, R0_Lt, nu, Anum, Znum, fcs, sgn, tau
     kx = 0.d0
     ky = 0.2d0
  eps_r = 0.18d0
    q_0 = 1.4d0
  s_hat = 0.8d0
 lambda = 0.d0
   beta = 0.d0
  R0_Ln(:) = 2.2d0
  R0_Lt(:) = 6.9d0
     nu(:) = 0.d0
   Anum(:) = 1.d0
   Znum(:) = 1.d0
    fcs(:) = 1.d0
    sgn(:) = 1.d0
    tau(:) = 1.d0
  open(10,file="param.namelist")
    read(10,nml=physp)
  close(10)
END SUBROUTINE set_param

END MODULE parameters
