# Boundary conditions

All computations take place on the unit cube $[0,1]^3$ (particle coordinates
must lie inside it; for the particle FMM with periodicity, `box_size` rescales
the period). The boundary condition is selected at tree-construction time.

The C++ interface takes a `pvfmm::BoundaryType` (defined in
`include/mpi_tree.hpp`):

| Value | Meaning |
|---|---|
| `FreeSpace` | free-space (decaying) boundary conditions |
| `PX` | periodic in $x$; free in $y, z$ |
| `PXY` | periodic in $x, y$; free in $z$ |
| `PXYZ` | periodic in all three directions |
| `Periodic` | alias for `PXYZ` |

Periodic sums are evaluated by accumulating the multipole expansion of the
periodic images into the root node's local expansion
(`FMM_Pts::PeriodicBC`); no Ewald-style parameter tuning is needed.

```{note}
For kernels without scale-invariant decay the periodic sum is defined up to
the usual gauge/mean constraints; for the Laplace kernel with periodic
boundary conditions the source must have zero mean over the box (as in
`fmm_cheb -test 1`, whose source integrates to zero).
```

The C-level interfaces mirror the full set of boundary types through
`enum PVFMMBoundaryType` (`PVFMMBoundaryFreeSpace`, `PVFMMBoundaryPXYZ`,
`PVFMMBoundaryPX`, `PVFMMBoundaryPXY`; values 0/1 coincide with the boolean
`periodic` flag accepted by earlier versions). The volume-tree constructors
take it directly, and the particle-FMM constructor `PVFMMCreateContext*`
takes it together with `box_size` (the domain length, which is the period
along the periodic directions; `box_size <= 0` is allowed only with free
space, where the bounding box is determined from the points). The Fortran
interface uses the `PVFMMBoundary*` constants, and the Python/Julia bindings
expose an `FMMBoundaryType` with the same values ({doc}`../api/python`,
{doc}`../api/julia`).
