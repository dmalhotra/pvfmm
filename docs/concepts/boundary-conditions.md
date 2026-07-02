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

```{important}
The partially periodic variants `PX` and `PXY` require the library to be
compiled with `-DPVFMM_EXTENDED_BC`. The CMake build enables this by default
(option `PVFMM_EXTENDED_BC`); the autotools build does not define it. Builds
without the flag support `FreeSpace` and fully periodic `PXYZ` only.
```

```{note}
For kernels without scale-invariant decay the periodic sum is defined up to
the usual gauge/mean constraints; for the Laplace kernel with periodic
boundary conditions the source must have zero mean over the box (as in
`fmm_cheb -test 1`, whose source integrates to zero).
```

The C-level interfaces (C, Fortran, Python, Julia) expose only a boolean
`periodic` flag for the volume FMM (equivalent to `PXYZ`) and, for the
particle FMM, a `box_size` argument where `0` means free space and a positive
value means fully periodic with that period.
