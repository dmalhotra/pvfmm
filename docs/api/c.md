# C interface

The C interface is declared in
[`include/pvfmm.h`](https://github.com/dmalhotra/pvfmm/blob/develop/include/pvfmm.h)
and compiled into `libpvfmm`. Link with `-lpvfmm` (see
{doc}`../install` for the recommended compiler/linker flags).

```{note}
Every function below has a single-precision twin with suffix `F` in place of
`D` and `float` in place of `double` in its signature (e.g. `PVFMMEvalD` /
`PVFMMEvalF`). Only the `D` variants are documented here.
```

```{note}
The C interface builds with or without MPI. When `libpvfmm` was built without
MPI, `MPI_Comm` degenerates to `int` and all communicator arguments are
ignored (single-process operation). MPI initialization is performed on demand
if the application has not done it already.
```

For a complete usage example see {doc}`../tutorials/c-fortran`.

## Kernel selection

```{doxygenenum} PVFMMKernel
```

The (source density, target value) dimensions per point for each kernel are
listed in {doc}`../concepts/kernels`. The Helmholtz and Stokes stress kernels
are not available through the C interface.

## Boundary conditions

See {doc}`../concepts/boundary-conditions` for semantics.

```{doxygenenum} PVFMMBoundaryType
```

## Volume FMM

Typical call sequence: `PVFMMCreateVolumeFMMD` (build/load translation
operators) and `PVFMMCreateVolumeTreeD` (discretize the source density), then
`PVFMMEvaluateVolumeFMMD`, and finally destroy both objects. The expensive
operator construction is cached on disk — see
{doc}`../concepts/precomputed-data`.

```{doxygenfunction} PVFMMCreateVolumeFMMD
```

```{doxygenfunction} PVFMMCreateVolumeTreeD
```

```{doxygenfunction} PVFMMCreateVolumeTreeFromCoeffD
```

```{doxygenfunction} PVFMMEvaluateVolumeFMMD
```

```{doxygenfunction} PVFMMDestroyVolumeFMMD
```

```{doxygenfunction} PVFMMDestroyVolumeTreeD
```

### Tree data access

```{doxygenfunction} PVFMMGetLeafCountD
```

```{doxygenfunction} PVFMMGetLeafCoordD
```

```{doxygenfunction} PVFMMGetPotentialCoeffD
```

### Chebyshev utilities

```{doxygenfunction} PVFMMCoeff2NodesD
```

```{doxygenfunction} PVFMMNodes2CoeffD
```

## Particle FMM

Typical call sequence: `PVFMMCreateContextD` once, then `PVFMMEvalD` as
many times as needed (pass `setup=1` whenever particle positions changed,
`setup=0` if only the densities changed), then `PVFMMDestroyContextD`.

```{doxygenfunction} PVFMMCreateContextD
```

```{doxygenfunction} PVFMMEvalD
```

```{doxygenfunction} PVFMMDestroyContextD
```

C and C++ callers pass their own `MPI_Comm` (e.g. `MPI_COMM_WORLD`). Language
bindings that do not link MPI can instead obtain the world communicator from
the library as a Fortran integer handle and pass it to the Fortran entry
points:

```{doxygenfunction} PVFMMGetCommWorld
```
