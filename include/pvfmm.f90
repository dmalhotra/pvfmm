!> @file pvfmm.f90
!> @author Dhairya Malhotra, dhairya.malhotra@gmail.com
!> @date 8-5-2018
!> @brief This file contains the declarations for the Fortran interface to PVFMM.

! List of PVFMM kernel functions
enum, bind(c)
  enumerator :: PVFMMLaplacePotential    = 0
  enumerator :: PVFMMLaplaceGradient     = 1
  enumerator :: PVFMMStokesPressure      = 2
  enumerator :: PVFMMStokesVelocity      = 3
  enumerator :: PVFMMStokesVelocityGrad  = 4
  enumerator :: PVFMMBiotSavartPotential = 5
end enum

interface

  ! Create single-precision particle FMM context
  subroutine PVFMMCreateContextF(fmm_ctx, box_size, points_per_leaf, multipole_order, kernel, comm)&
      bind(C, name="pvfmmcreatecontextf_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(out) :: fmm_ctx ! FMM context
    real*4, intent(in) :: box_size ! domain size for periodic boundary conditions
    integer*4, intent(in) :: points_per_leaf ! tuning parameter
    integer*4, intent(in) :: multipole_order ! accuracy (even integer)
    integer*4, intent(in) :: kernel ! kernel function
    integer, intent(in) :: comm ! MPI communicator
  end subroutine

  ! Evaluate potential in single-precision
  subroutine PVFMMEvalF(Xs, Vs, Ns, Xt, Vt, Nt, fmm_ctx, setup) bind(C, name="pvfmmevalf_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm_ctx ! FMM context
    integer *8, intent(in) :: Nt ! number of targets
    integer *8, intent(in) :: Ns ! number of sources
    real*4, intent(in) :: Xs(*) ! source position
    real*4, intent(in) :: Vs(*) ! source density
    real*4, intent(in) :: Xt(*) ! target position
    real*4, intent(out) :: Vt(*) ! target value
    integer *4, intent(in) :: setup ! if Xt or Xs changed, then setup=1 else setup=0
  end subroutine

  ! Destroy single-precision particle FMM context
  subroutine PVFMMDestroyContextF(fmm_ctx) bind(C, name="pvfmmdestroycontextf_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm_ctx ! FMM context
  end subroutine


  ! Create double-precision particle FMM context
  subroutine PVFMMCreateContextD(fmm_ctx, box_size, points_per_leaf, multipole_order, kernel, comm)&
      bind(C, name="pvfmmcreatecontextd_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(out) :: fmm_ctx ! FMM context
    real*8, intent(in) :: box_size ! domain size for periodic boundary conditions
    integer*4, intent(in) :: points_per_leaf ! tuning parameter
    integer*4, intent(in) :: multipole_order ! accuracy (even integer)
    integer*4, intent(in) :: kernel ! kernel function
    integer, intent(in) :: comm ! MPI communicator
  end subroutine

  ! Evaluate potential in double-precision
  subroutine PVFMMEvalD(Xs, Vs, Ns, Xt, Vt, Nt, fmm_ctx, setup) bind(C, name="pvfmmevald_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm_ctx ! FMM context
    integer *8, intent(in) :: Nt ! number of targets
    integer *8, intent(in) :: Ns ! number of sources
    real*8, intent(in) :: Xs(*) ! source position
    real*8, intent(in) :: Vs(*) ! source density
    real*8, intent(in) :: Xt(*) ! target position
    real*8, intent(out) :: Vt(*) ! target value
    integer *4, intent(in) :: setup ! if Xt or Xs changed, then setup=1 else setup=0
  end subroutine

  ! Destroy double-precision particle FMM context
  subroutine PVFMMDestroyContextD(fmm_ctx) bind(C, name="pvfmmdestroycontextd_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm_ctx ! FMM context
  end subroutine

end interface
