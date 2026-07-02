# Fortran interface

The Fortran interface is provided as an include file,
[`include/pvfmm.f90`](https://github.com/dmalhotra/pvfmm/blob/develop/include/pvfmm.f90),
containing `bind(C)` interface blocks for dedicated Fortran entry points in
`libpvfmm` (lower-case symbols with a trailing underscore, e.g.
`pvfmmcreatecontextd_`). Include it in your program and link with `-lpvfmm`:

```fortran
program main
  use iso_c_binding
  implicit none
  include 'mpif.h'
  include 'pvfmm.f90'
  ! ...
end program
```

All functions come in double-precision (`D` suffix, `real(c_double)`) and
single-precision (`F` suffix, `real(c_float)`) variants; only the `D` variants
are listed below. Opaque handles are `type(c_ptr)`. Communicators are passed
as plain `integer` Fortran MPI handles (e.g. `MPI_COMM_WORLD` from `mpif.h`).
See {doc}`../tutorials/c-fortran` for complete programs.

The kernel constants (`PVFMMLaplacePotential`, `PVFMMLaplaceGradient`,
`PVFMMStokesPressure`, `PVFMMStokesVelocity`, `PVFMMStokesVelocityGrad`,
`PVFMMBiotSavartPotential`) mirror the C `PVFMMKernel` enum, and the boundary
constants (`PVFMMBoundaryFreeSpace`, `PVFMMBoundaryPXYZ`, `PVFMMBoundaryPX`,
`PVFMMBoundaryPXY`, alias `PVFMMBoundaryPeriodic`) mirror `PVFMMBoundaryType`
— see {doc}`../concepts/boundary-conditions`.

## Particle FMM

```fortran
subroutine PVFMMCreateContextD(fmm_ctx, box_size, points_per_leaf, &
                                 multipole_order, kernel, bndry, comm)
  type(c_ptr),        intent(out) :: fmm_ctx         ! FMM context
  real(c_double),     intent(in)  :: box_size        ! domain length; period along periodic directions
  integer(c_int32_t), intent(in)  :: points_per_leaf ! max points per leaf node
  integer(c_int32_t), intent(in)  :: multipole_order ! accuracy (positive, even)
  integer(c_int32_t), intent(in)  :: kernel          ! PVFMMKernel value
  integer(c_int32_t), intent(in)  :: bndry           ! PVFMMBoundary* constant
  integer(c_int),     intent(in)  :: comm            ! MPI communicator
```

```fortran
subroutine PVFMMEvalD(Xs, Vs, Ns, Xt, Vt, Nt, fmm_ctx, setup)
  real(c_double),     intent(in)    :: Xs(*)   ! source positions [x1 y1 z1 ...]
  real(c_double),     intent(in)    :: Vs(*)   ! single-layer source densities
  integer(c_int64_t), intent(in)    :: Ns      ! number of sources
  real(c_double),     intent(in)    :: Xt(*)   ! target positions
  real(c_double),     intent(out)   :: Vt(*)   ! target values
  integer(c_int64_t), intent(in)    :: Nt      ! number of targets
  type(c_ptr),        intent(inout) :: fmm_ctx ! FMM context
  integer(c_int32_t), intent(in)    :: setup   ! 1 if Xs or Xt changed, else 0
```

```{note}
The Fortran `PVFMMEval` differs from the C `PVFMMEvalD`: the argument order is
sources → targets → context → setup, and there is no double-layer density
argument (single-layer sources only).
```

```fortran
subroutine PVFMMDestroyContextD(fmm_ctx)
  type(c_ptr), intent(inout) :: fmm_ctx  ! set to NULL on return
```

## Volume FMM

The volume subroutines mirror the {doc}`C interface <c>` one-to-one (same
argument meanings, with the created handle returned through the first
argument):

```fortran
! Build/load FMM translation operators.
subroutine PVFMMCreateVolumeFMMD(fmm, m, q, kernel, comm)

! Adaptive Chebyshev tree from a function callback.
subroutine PVFMMCreateVolumeTreeD(tree, cheb_deg, data_dim, fn_ptr, &
                                  trg_coord, n_trg, comm, tol, max_pts, &
                                  periodic, init_depth)

! Chebyshev tree from given leaf nodes and coefficients.
subroutine PVFMMCreateVolumeTreeFromCoeffD(tree, n_nodes, cheb_deg, data_dim, &
                                           node_coord, fn_coeff, trg_coord, &
                                           n_trg, comm, periodic)

! Run the FMM; trg_val receives the potential at the target points.
subroutine PVFMMEvaluateVolumeFMMD(trg_val, tree, fmm, loc_size)

! Tree data access.
subroutine PVFMMGetLeafCountD(Nleaf, tree)
subroutine PVFMMGetLeafCoordD(node_coord, tree)
subroutine PVFMMGetPotentialCoeffD(coeff, tree)

! Chebyshev basis conversions.
subroutine PVFMMCoeff2NodesD(node_val, Nleaf, ChebDeg, dof, coeff)
subroutine PVFMMNodes2CoeffD(coeff, Nleaf, ChebDeg, dof, node_val)

! Cleanup (handles are set to NULL).
subroutine PVFMMDestroyVolumeTreeD(tree)
subroutine PVFMMDestroyVolumeFMMD(fmm)
```

The source-density callback passed to `PVFMMCreateVolumeTreeD` has the
interface

```fortran
subroutine fn_ptr(coord, n, val) bind(C)
  real(c_double),     intent(in)  :: coord(n*3)  ! evaluation points [x1 y1 z1 ...]
  integer(c_int64_t)              :: n           ! number of points
  real(c_double),     intent(out) :: val(*)      ! n*data_dim output values
end subroutine
```

```{note}
Unlike the C callback, the Fortran callback receives no user context pointer,
and `periodic` is passed as `integer(c_int32_t)` — one of the
`PVFMMBoundary*` constants (0/1 keep the old free-space/periodic meaning).
```
