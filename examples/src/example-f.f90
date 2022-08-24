program main
  use iso_c_binding
  implicit none
  include 'mpif.h'
  include 'pvfmm.f90'
  integer ierror
  integer :: comm
  real*8 :: box_size
  integer*4 :: points_per_leaf, multipole_order, kernel
  type (c_ptr) :: fmm_ctx
  integer*8 :: Ns, Nt

  call MPI_Init(ierror)

  ! Create FMM context
  box_size = -1.0 ! for periodic boundary conditions
  points_per_leaf = 1000 ! tuning parameter
  multipole_order = 10 ! accuracy
  kernel = PVFMMBiotSavartPotential ! kernel function
  comm = MPI_COMM_WORLD
  call PVFMMCreateContextD(fmm_ctx, box_size, points_per_leaf, multipole_order, kernel, comm)

  ! Evaluate FMM
  Ns = 20000
  Nt = 20000
  call test(fmm_ctx, Ns, Nt)

  ! Destroy FMM context
  call PVFMMDestroyContextD(fmm_ctx)

  call MPI_Finalize(ierror)
end

subroutine BiotSavart(Xs, Vs, Ns, Xt, Vt, Nt)
  implicit none
  integer*8 Ns, Nt, s, t
  real*8 :: Xs(Ns * 3), Vs(Ns * 3), Xt(Nt * 3), Vt(Nt * 3)
  real*8 :: oofp, X(3), rinv, rinv3

  oofp = 1/(16*atan(1.0))
  !$omp parallel do private(s,X,rinv,rinv3)
  do t = 0, Nt-1
    Vt(t*3+1) = 0
    Vt(t*3+2) = 0
    Vt(t*3+3) = 0
    do s = 0, Ns-1
      X(1) = Xt(t*3+1) - Xs(s*3+1)
      X(2) = Xt(t*3+2) - Xs(s*3+2)
      X(3) = Xt(t*3+3) - Xs(s*3+3)

      rinv = X(1)*X(1) + X(2)*X(2) + X(3)*X(3)
      if (rinv .gt. 0) then
        rinv = 1/sqrt(rinv)
      endif
      rinv3 = rinv*rinv*rinv

      Vt(t*3+1) = Vt(t*3+1) + (Vs(s*3+2)*X(3) - Vs(s*3+3)*X(2)) * rinv3 * oofp;
      Vt(t*3+2) = Vt(t*3+2) + (Vs(s*3+3)*X(1) - Vs(s*3+1)*X(3)) * rinv3 * oofp;
      Vt(t*3+3) = Vt(t*3+3) + (Vs(s*3+1)*X(2) - Vs(s*3+2)*X(1)) * rinv3 * oofp;
    enddo
  enddo
endsubroutine

subroutine test(fmm_ctx, Ns, Nt)
  use iso_c_binding
  implicit none
  include 'pvfmm.f90'
  integer*8 :: i, Ns, Nt
  real*8 :: Xs(Ns * 3), Vs(Ns * 3), Xt(Nt * 3), Vt(Nt * 3), Vt_ref(Nt * 3)
  double precision :: omp_get_wtime, tt
  type (c_ptr) :: fmm_ctx
  integer*4 :: setup
  real*4 :: rand

  call srand(0)
  do i=1, Ns*3
    Xs(i) = rand(0)
    Vs(i) = rand(0)
  enddo
  do i=1, Nt*3
    Xt(i) = rand(0)
  enddo

  setup = 1
  tt = -omp_get_wtime()
  call PVFMMEvalD(Xs, Vs, Ns, Xt, Vt, Nt, fmm_ctx, setup)
  tt = tt + omp_get_wtime()
  print*, "FMM evaluation time (with setup) : ", tt

  setup = 0
  tt = -omp_get_wtime()
  call PVFMMEvalD(Xs, Vs, Ns, Xt, Vt, Nt, fmm_ctx, setup)
  tt = tt + omp_get_wtime()
  print*, "FMM evaluation time (without setup) : ", tt

  tt = -omp_get_wtime()
  call BiotSavart(Xs, Vs, Ns, Xt, Vt_ref, Nt)
  tt = tt + omp_get_wtime()
  print*, "Direct evaluation time : ", tt

  print*, "Maximum relative error : ", maxval(abs(Vt_ref - Vt)) / maxval(abs(Vt_ref));
endsubroutine

