program main
  use iso_c_binding
  implicit none
  include 'mpif.h'
  include 'pvfmm.f90'
  integer ierror
  integer :: comm
  integer*4 :: kdim0, kdim1, cheb_deg, multipole_order, kernel
  type (c_ptr) :: fmm_ctx

  call MPI_Init(ierror)

  ! Create FMM context
  kdim0 = 3
  kdim1 = 3;
  cheb_deg = 14 ! Chebyshev degree
  multipole_order = 10 ! accuracy
  kernel = PVFMMStokesVelocity ! kernel function
  comm = MPI_COMM_WORLD
  call PVFMMCreateVolumeFMMD(fmm_ctx, multipole_order, cheb_deg, kernel, comm)

  call test1(fmm_ctx, kdim0, kdim1, cheb_deg, comm)
  call test2(fmm_ctx, kdim0, kdim1, cheb_deg, comm)

  ! Destroy FMM context
  call PVFMMDestroyVolumeFMMD(fmm_ctx)

  call MPI_Finalize(ierror)
end

subroutine fn_input(coord, n, val)
  implicit none
  integer*8 :: n, i, dof
  real*8, intent(in) :: coord(n*3)
  real*8, intent(out) :: val(n*3)
  real*8 :: L, r_2

  dof = 3
  L = 125
  do i = 0, n-1
    r_2=(coord(i*3+1)-0.5)**2 + (coord(i*3+2)-0.5)**2 + (coord(i*3+3)-0.5)**2
    val(i*dof+1)=                                                0+2*L*exp(-L*r_2)*(coord(i*3+1)-0.5)
    val(i*dof+2)= 4*L*L*(coord(i*3+3)-0.5)*(5-2*L*r_2)*exp(-L*r_2)+2*L*exp(-L*r_2)*(coord(i*3+2)-0.5)
    val(i*dof+3)=-4*L*L*(coord(i*3+2)-0.5)*(5-2*L*r_2)*exp(-L*r_2)+2*L*exp(-L*r_2)*(coord(i*3+3)-0.5)
  enddo
endsubroutine

subroutine fn_poten(coord, n, val)
  implicit none
  integer*8 :: n, i, dof
  real*8, intent(in) :: coord(n*3)
  real*8, intent(out) :: val(n*3)
  real*8 :: L, r_2

  dof = 3
  L = 125
  do i = 0, n-1
    r_2=(coord(i*3+1)-0.5)**2 + (coord(i*3+2)-0.5)**2 + (coord(i*3+3)-0.5)**2
    val(i*dof+1)= 0;
    val(i*dof+2)= 2*L*(coord(i*3+3)-0.5)*exp(-L*r_2)
    val(i*dof+3)=-2*L*(coord(i*3+2)-0.5)*exp(-L*r_2)
  enddo
endsubroutine

subroutine GetChebNodes(cheb_coord, Nleaf, cheb_deg, depth, leaf_coord)
  integer*8 Nleaf
  integer*4 cheb_deg, depth
  real*8 cheb_coord(Nleaf*(cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1)*3)
  real*8 leaf_coord(Nleaf*3)
  real*8 M_PI, leaf_length
  integer*4 j0, j1, j2, Ncheb
  integer*8 leaf_idx, node_idx

  M_PI=4*datan(1.D0)
  leaf_length = 1./(2**depth)
  Ncheb = (cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1)

  do leaf_idx = 0, Nleaf-1
    do j2 = 0, cheb_deg
      do j1 = 0, cheb_deg
        do j0 = 0, cheb_deg
          node_idx = leaf_idx * Ncheb + (j2 * (cheb_deg+1) + j1) * (cheb_deg+1) + j0
          cheb_coord(node_idx*3+1) = leaf_coord(leaf_idx*3+1) + (1-cos(M_PI*(j0*2+1)/(cheb_deg*2+2))) * leaf_length * 0.5;
          cheb_coord(node_idx*3+2) = leaf_coord(leaf_idx*3+2) + (1-cos(M_PI*(j1*2+1)/(cheb_deg*2+2))) * leaf_length * 0.5;
          cheb_coord(node_idx*3+3) = leaf_coord(leaf_idx*3+3) + (1-cos(M_PI*(j2*2+1)/(cheb_deg*2+2))) * leaf_length * 0.5;
        enddo
      enddo
    enddo
  enddo
endsubroutine

!> Build volume tree using function pointer
subroutine test1(fmm, kdim0, kdim1, cheb_deg, comm)
  use iso_c_binding
  implicit none
  include 'mpif.h'
  include 'pvfmm.f90'
  type (c_ptr) :: fmm, tree
  integer*4 kdim0, kdim1, cheb_deg
  integer :: comm, ierr, mpi_rank

  integer*8 Nt, i
  real*8, allocatable :: trg_coord(:), trg_value(:), trg_value_ref(:)
  real*8 tol, max_err, max_val, max_err_glb, max_val_glb

  interface
    subroutine fn_input(coord, n, val)
      implicit none
      real*8, intent(in) :: coord(n*3)
      real*8, intent(out) :: val(n*3)
      integer*8 :: n
    end subroutine fn_input
  end interface

  call MPI_Comm_rank(comm, mpi_rank, ierr)
  call srand(mpi_rank)

  tol=1e-6
  Nt = 100
  allocate (trg_coord(Nt*3))
  allocate (trg_value(Nt*kdim1))
  allocate (trg_value_ref(Nt*kdim1))
  do i = 1, Nt*3
    trg_coord(i) = rand()
  enddo
  call fn_poten(trg_coord, Nt, trg_value_ref)

  ! Build volume tree
  call PVFMMCreateVolumeTreeD(tree, cheb_deg, kdim0, fn_input, trg_coord, Nt, comm, tol, 100, 0, 0);

  ! Evaluate FMM
  call PVFMMEvaluateVolumeFMMD(trg_value, tree, fmm, Nt);

  ! Print error
  max_err = 0
  max_val = 0;
  do i = 1, Nt*kdim1
    max_err = max(max_err, abs(trg_value(i)-trg_value_ref(i)))
    max_val = max(max_val, abs(trg_value_ref(i)))
  enddo
  call MPI_Reduce(max_err, max_err_glb, 1, MPI_DOUBLE, MPI_SUM, 0, comm, ierr)
  call MPI_Reduce(max_val, max_val_glb, 1, MPI_DOUBLE, MPI_SUM, 0, comm, ierr)
  if (mpi_rank .eq. 0) then
    print*, "Maximum relative error = ", max_err_glb/max_val_glb
  end if

  ! Free resources
  call PVFMMDestroyVolumeTreeD(tree);
  deallocate (trg_coord)
  deallocate (trg_value)
  deallocate (trg_value_ref)
end subroutine


!> Build volume tree from Chebyshev coefficients
subroutine test2(fmm, kdim0, kdim1, cheb_deg, comm)
  use iso_c_binding
  implicit none
  include 'mpif.h'
  include 'pvfmm.f90'
  type (c_ptr) :: fmm, tree
  integer*4 kdim0, kdim1, cheb_deg
  integer :: comm, ierr, mpi_rank, mpi_size

  integer*4 Ncheb, Ncoef, depth
  integer*8 i, Nleaf, Nleaf_loc, leaf_idx, leaf_idx_glb, Ntrg
  real*8, allocatable :: leaf_coord(:), cheb_coord(:), dens_value(:), dens_coeff(:), trg_coord(:), trg_poten(:)
  real*8, allocatable :: potn_coeff(:), potn_value(:), potn_value_ref(:)
  real*8 leaf_length, max_err, max_val, max_err_glb, max_val_glb

  depth = 3
  Ncheb = (cheb_deg+1)*(cheb_deg+1)*(cheb_deg+1)
  Ncoef = (cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6

  call MPI_Comm_rank(comm, mpi_rank, ierr)
  call MPI_Comm_size(comm, mpi_size, ierr)

  !Build uniform tree
  Nleaf = (2**(3*depth))
  Nleaf_loc = Nleaf*(mpi_rank+1)/mpi_size - Nleaf*mpi_rank/mpi_size
  leaf_length = 1./(2**depth)

  allocate (leaf_coord(Nleaf_loc*3))
  allocate (cheb_coord(Nleaf_loc*Ncheb*3))
  allocate (dens_value(Nleaf_loc*Ncheb*kdim0))
  allocate (dens_coeff(Nleaf_loc*Ncoef*kdim0))
  do leaf_idx = 0, Nleaf_loc-1
    leaf_idx_glb = Nleaf*mpi_rank/mpi_size + leaf_idx
    leaf_coord(leaf_idx*3+1) = mod((leaf_idx_glb/(2**(depth*0))), (2**depth)) * leaf_length;
    leaf_coord(leaf_idx*3+2) = mod((leaf_idx_glb/(2**(depth*1))), (2**depth)) * leaf_length;
    leaf_coord(leaf_idx*3+3) = mod((leaf_idx_glb/(2**(depth*2))), (2**depth)) * leaf_length;
  enddo

  Ntrg = 0
  call GetChebNodes(cheb_coord, Nleaf_loc, cheb_deg, depth, leaf_coord)
  call fn_input(cheb_coord, Nleaf_loc*Ncheb, dens_value)
  call PVFMMNodes2CoeffD(dens_coeff, Nleaf_loc, cheb_deg, kdim0, dens_value)
  call PVFMMCreateVolumeTreeFromCoeffD(tree, Nleaf_loc, cheb_deg, kdim0, leaf_coord, dens_coeff, trg_coord, Ntrg, comm, 0)

  deallocate (leaf_coord)
  deallocate (cheb_coord)
  deallocate (dens_value)
  deallocate (dens_coeff)


  ! Evaluate FMM
  call PVFMMEvaluateVolumeFMMD(trg_poten, tree, fmm, Ntrg);

  ! Get potential at Chebyshev nodes
  call PVFMMGetLeafCountD(Nleaf, tree)
  allocate (potn_coeff(Nleaf*Ncoef*kdim1))
  allocate (potn_value(Nleaf*Ncheb*kdim1))
  call PVFMMGetPotentialCoeffD(potn_coeff, tree)
  call PVFMMCoeff2NodesD(potn_value, Nleaf, cheb_deg, kdim1, potn_coeff)

  ! Get reference solution at Chebyshev nodes
  allocate (leaf_coord(Nleaf*3))
  allocate (cheb_coord(Nleaf*Ncheb*3))
  allocate (potn_value_ref(Nleaf*Ncheb*kdim1))
  call PVFMMGetLeafCoordD(leaf_coord, tree)
  call GetChebNodes(cheb_coord, Nleaf, cheb_deg, depth, leaf_coord)
  call fn_poten(cheb_coord, Nleaf*Ncheb, potn_value_ref)

  ! Print error
  max_err = 0
  max_val = 0;
  do i = 1, Nleaf*Ncheb*kdim1
    max_err = max(max_err, abs(potn_value(i)-potn_value_ref(i)))
    max_val = max(max_val, abs(potn_value_ref(i)))
  enddo
  call MPI_Reduce(max_err, max_err_glb, 1, MPI_DOUBLE, MPI_SUM, 0, comm, ierr)
  call MPI_Reduce(max_val, max_val_glb, 1, MPI_DOUBLE, MPI_SUM, 0, comm, ierr)
  if (mpi_rank .eq. 0) then
    print*, "Maximum relative error = ", max_err_glb/max_val_glb
  end if

  ! Free resources
  call PVFMMDestroyVolumeTreeD(tree);
  deallocate (potn_coeff);
  deallocate (potn_value);
  deallocate (leaf_coord);
  deallocate (cheb_coord);
  deallocate (potn_value_ref);
endsubroutine
