# Julia interface

The `PVFMM` Julia package (in
[`julia/`](https://github.com/dmalhotra/pvfmm/tree/develop/julia)) wraps the
{doc}`C interface <c>` using `Libdl`; its API mirrors the
{doc}`Python bindings <python>`.

## Installation and setup

Build the PVFMM library first. The package locates `libpvfmm` from:

1. `ENV["PVFMM"]` — either the direct library path or a directory containing
   `libpvfmm.{so,dylib,dll}`;
2. the system library search path (`Libdl.find_library`).

```julia
ENV["PVFMM"] = "/path/to/dir-containing-libpvfmm"
using PVFMM
```

Precision is selected with the type parameter/keyword (`Float64` default,
`Float32`). Array arguments are `Vector{T}` in `[x1 y1 z1 x2 y2 z2 ...]`
(array-of-structures) layout. Communicators may be passed as an `Integer`
(Fortran-style handle), a `Ptr{Cvoid}`, or any object with a `val` field of
one of those types (which covers `MPI.Comm` from MPI.jl). Contexts and trees
are freed by GC finalizers.

Exported names: `FMMKernel`, `FMMVolumeContext`, `FMMParticleContext`,
`FMMVolumeTree`, `nodes_to_coeff`, `from_function`, `from_coefficients`,
`evaluate`, `leaf_count`, `get_leaf_coordinates`, `get_coefficients`,
`get_values`.

## FMMKernel

```julia
@enum FMMKernel begin
    LaplacePotential    = 0
    LaplaceGradient     = 1
    StokesPressure      = 2
    StokesVelocity      = 3
    StokesVelocityGrad  = 4
    BiotSavartPotential = 5
end
```

Source/target dimensions per kernel are listed in {doc}`../concepts/kernels`.

## Particle FMM

```julia
FMMParticleContext(box_size, max_points, multipole_order, kernel, comm=nothing; T=Float64)
```

Creates a particle-FMM context. `box_size` is the period length for periodic
boundary conditions (`0` for free space); `multipole_order` must be positive
and even. When `comm === nothing` the context is created on `MPI_COMM_WORLD`
via the C `PVFMMCreateContext*World` entry points (Julia is the only binding
exposing these).

```julia
evaluate(ctx::FMMParticleContext{T}, src_pos, sl_den, dl_den, trg_pos; setup=true)
```

Evaluates the potential at the target points. `sl_den` (same length as
`src_pos`) and `dl_den` (twice the length of `src_pos`) may each be
`nothing`. Pass `setup=true` whenever source or target *positions* changed.

```{caution}
As in the Python binding, the returned vector has length `length(trg_pos)`
(three values per target); for kernels whose target dimension is not 3, only
the first `n_trg * trg_dim` entries are meaningful, and `StokesVelocityGrad`
must not be used through this method.
```

## Volume FMM

```julia
FMMVolumeContext(multipole_order, chebyshev_degree, kernel, comm; T=Float64)
```

Builds (or loads from cache — see {doc}`../concepts/precomputed-data`) the
volume-FMM translation operators.

```julia
from_function(FMMVolumeTree{T}, cheb_deg, data_dim, fn_ptr::Ptr{Cvoid},
              fn_ctx::Ptr{Cvoid}, trg_coord, comm, tol, max_pts,
              periodic::Bool, init_depth)
```

Builds an adaptively refined Chebyshev tree from a source-density callback.
`fn_ptr` is a C function pointer with signature
`void fn(const T* coord, long n, T* out, const void* ctx)` (e.g. from
`@cfunction`); `fn_ctx` is passed through as `ctx`.

```julia
from_coefficients(FMMVolumeTree{T}, cheb_deg, data_dim, leaf_coord, fn_coeff,
                  trg_coord, comm, periodic::Bool)
```

Builds the tree from leaf-node coordinates and Chebyshev coefficients;
`trg_coord` may be `nothing`.

```julia
evaluate(tree::FMMVolumeTree{T}, fmm::FMMVolumeContext{T}, loc_size)
```

Runs the volume FMM; returns the potential at the target points
(`n_trg * trg_dim` values).

```julia
leaf_count(tree)             # number of leaf nodes
get_leaf_coordinates(tree)   # leaf corners, 3 per leaf
get_coefficients(tree)       # Chebyshev coefficients of the potential
get_values(tree)             # potential on tensor-product Chebyshev nodes
nodes_to_coeff(N_leaf, cheb_deg, dof, node_val)  # node values -> coefficients
```

`get_coefficients`/`get_values` require a prior `evaluate` call.

## Tests as examples

`julia/test/reference_comparison.jl` validates the particle FMM against
direct O(N²) sums for the Laplace potential/gradient and Stokes
velocity/pressure kernels — a good starting point for usage patterns.
