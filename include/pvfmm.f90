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

interface ! Volume FMM

  !> Build FMM translation operators.
  !!
  !! @param[out] fmm the volume FMM context pointer.
  !!
  !! @param[in] m the multipole order (positive, even integer).
  !!
  !! @param[in] q the degree of the Chebyshev polynomials.
  !!
  !! @param[in] kernel the kernel function.
  !!
  !! @param[in] comm the MPI communicator.
  subroutine PVFMMCreateVolumeFMMD(fmm, m, q, kernel,  comm)&
      bind(C, name='pvfmmcreatevolumefmmd_')
    use iso_c_binding
    implicit none
    type(c_ptr), intent(out) :: fmm ! FMM context
    integer(c_int32_t), intent(in) :: m ! accuracy (even integer)
    integer(c_int32_t), intent(in) :: q ! Chebyshev degree
    integer(c_int32_t), intent(in) :: kernel ! kernel function
    integer(c_int), intent(in) :: comm ! MPI communicator
  end subroutine

  subroutine PVFMMCreateVolumeFMMF(fmm, m, q, kernel,  comm)&
      bind(C, name='pvfmmcreatevolumefmmf_')
    use iso_c_binding
    implicit none
    type(c_ptr), intent(out) :: fmm ! FMM context
    integer(c_int32_t), intent(in) :: m ! accuracy (even integer)
    integer(c_int32_t), intent(in) :: q ! Chebyshev degree
    integer(c_int32_t), intent(in) :: kernel ! kernel function
    integer(c_int), intent(in) :: comm ! MPI communicator
  end subroutine



  !> Destroy the volume FMM context.
  !!
  !! @param[in,out] fmm a pointer to pointer to the FMM. The pointer value is
  !! set to NULL.
  subroutine PVFMMDestroyVolumeFMMD(fmm)&
      bind(C, name="pvfmmdestroyvolumefmmd_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm ! FMM context
  end subroutine

  subroutine PVFMMDestroyVolumeFMMF(fmm)&
      bind(C, name="pvfmmdestroyvolumefmmf_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm ! FMM context
  end subroutine



  !> Construct a piecewise Chebyshev volume discretization in [0,1]^3.  It
  !! first constructs a tree with the given leaf node coordinates and then adds
  !! the Chebyshev coefficient to each leaf node.
  !!
  !! @param[out] tree the pointer to the constructed tree. It must be destroyed using
  !! PVFMMDestroyVolumeTreeD to free the resources.
  !!
  !! @param[in] Nleaf the number of leaf nodes.
  !!
  !! @param[in] cheb_deg the degree of the Chebyshev polynomials. The number of
  !! coefficients in each leaf node is
  !! data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6.
  !!
  !! @param[in] leaf_coord A vector of points [x1 y1 z1 ...  xn yn zn] where each
  !! point corresponds to a leaf node in the tree.
  !!
  !! @param[in] fn_coeff the vector of Chebyshev coefficients of size
  !! Nleaf*data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6, where
  !! Nleaf=leaf_coord.size()/3 is the number of leaf nodes.
  !!
  !! @param[in] trg_coord the target coordinate vector with values: [x1 y1 z1 ...
  !! xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
  !!
  !! @param[in] comm MPI communicator.
  !!
  !! @param[in] periodic whether to use periodic boundary conditions.
  subroutine PVFMMCreateVolumeTreeFromCoeffD(tree, n_nodes, cheb_deg,&
      data_dim, node_coord, fn_coeff, trg_coord, n_trg, comm, periodic)&
      bind(C, name="pvfmmcreatevolumetreefromcoeffd_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(out) :: tree ! tree context
    integer(c_int64_t), intent(in) :: n_nodes
    integer(c_int32_t), intent(in) :: cheb_deg
    integer(c_int32_t), intent(in) :: data_dim
    real(c_double), intent(in) :: node_coord(*)
    real(c_double), intent(in) :: fn_coeff(*)
    real(c_double), intent(in) :: trg_coord(*)
    integer(c_int64_t), intent(in) :: n_trg
    integer(c_int), intent(in) :: comm ! MPI communicator
    integer(c_int32_t), intent(in) :: periodic
  end subroutine

  subroutine PVFMMCreateVolumeTreeFromCoeffF(tree, n_nodes, cheb_deg,&
      data_dim, node_coord, fn_coeff, trg_coord, n_trg, comm, periodic)&
      bind(C, name="pvfmmcreatevolumetreefromcoefff_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(out) :: tree ! tree context
    integer(c_int64_t), intent(in) :: n_nodes
    integer(c_int32_t), intent(in) :: cheb_deg
    integer(c_int32_t), intent(in) :: data_dim
    real(c_float), intent(in) :: node_coord(*)
    real(c_float), intent(in) :: fn_coeff(*)
    real(c_float), intent(in) :: trg_coord(*)
    integer(c_int64_t), intent(in) :: n_trg
    integer(c_int), intent(in) :: comm ! MPI communicator
    integer(c_int32_t), intent(in) :: periodic
  end subroutine



  !> brief Destroy the volume FMM tree.
  !!
  !! @param[in,out] tree a pointer to pointer to the tree. The pointer value is
  !! set to NULL.
  subroutine PVFMMDestroyVolumeTreeD(tree)&
      bind(C, name="pvfmmdestroyvolumetreed_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: tree
  end subroutine

  subroutine PVFMMDestroyVolumeTreeF(tree)&
      bind(C, name="pvfmmdestroyvolumetreef_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: tree
  end subroutine



  !> Run volume FMM and evaluate the result at the target points.
  !!
  !! @param[out] trg_value the computed potential at the target points (in
  !! array-of-structure order).
  !!
  !! @param[in] tree the pointer to the Chebyshev tree.
  !!
  !! @param[in] fmm the volume FMM context pointer.
  !!
  !! @param[in] loc_size the local size of the output vector (used to partition
  !! it among the MPI ranks).
  subroutine PVFMMEvaluateVolumeFMMD(trg_val, tree, fmm, loc_size)&
      bind(C, name="pvfmmevaluatevolumefmmd_")
    use iso_c_binding
    implicit none
    real(c_double), intent(out) :: trg_val(*)
    type(c_ptr), intent(inout) :: tree ! tree context
    type(c_ptr), intent(in) :: fmm ! FMM context
    integer(c_int64_t), intent(in) :: loc_size
  end subroutine

  subroutine PVFMMEvaluateVolumeFMMF(trg_val, tree, fmm, loc_size)&
      bind(C, name="pvfmmevaluatevolumefmmf_")
    use iso_c_binding
    implicit none
    real(c_float), intent(out) :: trg_val(*)
    type(c_ptr), intent(inout) :: tree ! tree context
    type(c_ptr), intent(in) :: fmm ! FMM context
    integer(c_int64_t), intent(in) :: loc_size
  end subroutine



  !> Get the number leaf nodes.
  !!
  !! @param[out] Nleaf the number of leaf nodes.
  !!
  !! @param[in] tree the pointer to the Chebyshev tree.
  subroutine PVFMMGetLeafCountD(Nleaf, tree)&
      bind(C, name="pvfmmgetleafcountd_")
    use iso_c_binding
    implicit none
    integer(c_int64_t), intent(out) :: Nleaf
    type(c_ptr), intent(in) :: tree ! tree context
  end subroutine

  subroutine PVFMMGetLeafCountF(Nleaf, tree)&
      bind(C, name="pvfmmgetleafcountf_")
    use iso_c_binding
    implicit none
    integer(c_int64_t), intent(out) :: Nleaf
    type(c_ptr), intent(in) :: tree ! tree context
  end subroutine



  !> Get the leaf node coordinates.
  !!
  !! @param[in] leaf_coord A vector of points [x1 y1 z1 ...  xn yn zn] where each
  !! point corresponds to a leaf node in the tree.
  !!
  !! @param[in] tree the pointer to the Chebyshev tree.
  subroutine PVFMMGetLeafCoordD(node_coord, tree)&
      bind(C, name="pvfmmgetleafcoordd_")
    use iso_c_binding
    implicit none
    real(c_double), intent(out) :: node_coord(*)
    type(c_ptr), intent(in) :: tree ! tree context
  end subroutine

  subroutine PVFMMGetLeafCoordF(node_coord, tree)&
      bind(C, name="pvfmmgetleafcoordf_")
    use iso_c_binding
    implicit none
    real(c_float), intent(out) :: node_coord(*)
    type(c_ptr), intent(in) :: tree ! tree context
  end subroutine



  !> Get the Chebyshev coefficients for the potential.
  !!
  !! @param[out] coeff the array of Chebyshev coefficients for the potential of
  !! size Nleaf*data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6, where Nleaf is
  !! the number of leaf nodes.
  !!
  !! @param[in] tree the pointer to the Chebyshev tree.
  subroutine PVFMMGetPotentialCoeffD(coeff, tree)&
      bind(C, name="pvfmmgetpotentialcoeffd_")
    use iso_c_binding
    implicit none
    real(c_double), intent(out) :: coeff(*)
    type(c_ptr), intent(in) :: tree ! tree context
  end subroutine

  subroutine PVFMMGetPotentialCoeffF(coeff, tree)&
      bind(C, name="pvfmmgetpotentialcoefff_")
    use iso_c_binding
    implicit none
    real(c_float), intent(out) :: coeff(*)
    type(c_ptr), intent(in) :: tree ! tree context
  end subroutine



  !> Evaluate Chebyshev coefficients at tensor product Chebyshev nodes of
  !! first kind.
  !!
  !! @param[out] node_val node_val the function values at tensor product Chebyshev nodes.
  !!
  !! @param[in] Nleaf the number of leaf nodes.
  !!
  !! @param[in] ChebDeg the degree of Chebyshev polynomials.
  !!
  !! @param[in] dof the number of scalar values at each node point.
  !!
  !! @param[in] coeff the array of Chebyshev coefficients.
  subroutine PVFMMCoeff2NodesD(node_val, Nleaf, ChebDeg, dof, coeff)&
      bind(C, name="pvfmmcoeff2nodesd_")
    use iso_c_binding
    implicit none
    real(c_double), intent(out) :: node_val(*)
    integer(c_int64_t), intent(in) :: Nleaf
    integer(c_int32_t), intent(in) :: ChebDeg
    integer(c_int32_t), intent(in) :: dof
    real(c_double), intent(in) :: coeff(*)
  end subroutine

  subroutine PVFMMCoeff2NodesF(node_val, Nleaf, ChebDeg, dof, coeff)&
      bind(C, name="pvfmmcoeff2nodesf_")
    use iso_c_binding
    implicit none
    real(c_float), intent(out) :: node_val(*)
    integer(c_int64_t), intent(in) :: Nleaf
    integer(c_int32_t), intent(in) :: ChebDeg
    integer(c_int32_t), intent(in) :: dof
    real(c_float), intent(in) :: coeff(*)
  end subroutine



  !> Convert function values on tensor product Chebyshev nodes (first
  !! kind nodes) to coefficients.
  !!
  !! @param[out] coeff the vector of Chebyshev coefficients.
  !!
  !! @param[in] Nleaf the number of leaf nodes.
  !!
  !! @param[in] ChebDeg the degree of Chebyshev polynomials.
  !!
  !! @param[in] dof the number of scalar values at each node point.
  !!
  !! @param[in] node_val the function values at tensor product Chebyshev nodes.
  subroutine PVFMMNodes2CoeffD(coeff, Nleaf, ChebDeg, dof, node_val)&
      bind(C, name="pvfmmnodes2coeffd_")
    use iso_c_binding
    implicit none
    real(c_double), intent(out) :: coeff(*)
    integer(c_int64_t), intent(in) :: Nleaf
    integer(c_int32_t), intent(in) :: ChebDeg
    integer(c_int32_t), intent(in) :: dof
    real(c_double), intent(in) :: node_val(*)
  end subroutine

  subroutine PVFMMNodes2CoeffF(coeff, Nleaf, ChebDeg, dof, node_val)&
      bind(C, name="pvfmmnodes2coefff_")
    use iso_c_binding
    implicit none
    real(c_float), intent(out) :: coeff(*)
    integer(c_int64_t), intent(in) :: Nleaf
    integer(c_int32_t), intent(in) :: ChebDeg
    integer(c_int32_t), intent(in) :: dof
    real(c_float), intent(in) :: node_val(*)
  end subroutine

end interface



interface ! Particle FMM

  ! Create single-precision particle FMM context
  subroutine PVFMMCreateContextF(fmm_ctx, box_size, points_per_leaf, multipole_order, kernel, comm)&
      bind(C, name="pvfmmcreatecontextf_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(out) :: fmm_ctx ! FMM context
    real(c_float), intent(in) :: box_size ! domain size for periodic boundary conditions
    integer(c_int32_t), intent(in) :: points_per_leaf ! tuning parameter
    integer(c_int32_t), intent(in) :: multipole_order ! accuracy (even integer)
    integer(c_int32_t), intent(in) :: kernel ! kernel function
    integer(c_int), intent(in) :: comm ! MPI communicator
  end subroutine

  ! Evaluate potential in single-precision
  subroutine PVFMMEvalF(Xs, Vs, Ns, Xt, Vt, Nt, fmm_ctx, setup) bind(C, name="pvfmmevalf_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm_ctx ! FMM context
    integer(c_int64_t), intent(in) :: Nt ! number of targets
    integer(c_int64_t), intent(in) :: Ns ! number of sources
    real(c_float), intent(in) :: Xs(*) ! source position
    real(c_float), intent(in) :: Vs(*) ! source density
    real(c_float), intent(in) :: Xt(*) ! target position
    real(c_float), intent(out) :: Vt(*) ! target value
    integer(c_int32_t), intent(in) :: setup ! if Xt or Xs changed, then setup=1 else setup=0
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
    real(c_double), intent(in) :: box_size ! domain size for periodic boundary conditions
    integer(c_int32_t), intent(in) :: points_per_leaf ! tuning parameter
    integer(c_int32_t), intent(in) :: multipole_order ! accuracy (even integer)
    integer(c_int32_t), intent(in) :: kernel ! kernel function
    integer(c_int), intent(in) :: comm ! MPI communicator
  end subroutine

  ! Evaluate potential in double-precision
  subroutine PVFMMEvalD(Xs, Vs, Ns, Xt, Vt, Nt, fmm_ctx, setup) bind(C, name="pvfmmevald_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm_ctx ! FMM context
    integer(c_int64_t), intent(in) :: Nt ! number of targets
    integer(c_int64_t), intent(in) :: Ns ! number of sources
    real(c_double), intent(in) :: Xs(*) ! source position
    real(c_double), intent(in) :: Vs(*) ! source density
    real(c_double), intent(in) :: Xt(*) ! target position
    real(c_double), intent(out) :: Vt(*) ! target value
    integer(c_int32_t), intent(in) :: setup ! if Xt or Xs changed, then setup=1 else setup=0
  end subroutine

  ! Destroy double-precision particle FMM context
  subroutine PVFMMDestroyContextD(fmm_ctx) bind(C, name="pvfmmdestroycontextd_")
    use iso_c_binding
    implicit none
    type(c_ptr), intent(inout) :: fmm_ctx ! FMM context
  end subroutine

end interface
