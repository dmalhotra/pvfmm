# Introduction

PVFMM (Parallel Volume Fast Multipole Method) is a library for evaluating
potentials from particle and volume sources in three dimensions — the
computational core of integral-equation solvers for certain elliptic partial
differential equations. It supports Stokes, Poisson (Laplace), Biot–Savart,
and Helmholtz problems on the unit cube, with free-space or periodic boundary
conditions.

## Two evaluation modes

**Particle FMM** computes N-body sums

$$u(x_i) = \sum_{j} K(x_i, y_j)\, f_j,$$

for $N$ target points $x_i$, source points $y_j$, densities $f_j$, and one of
the built-in {doc}`kernel functions <concepts/kernels>` $K$ — in $O(N)$ work
instead of $O(N^2)$.

**Volume FMM** evaluates volume potentials

$$u(x) = \int_{[0,1]^3} K(x, y)\, f(y)\, dy,$$

where the source density $f$ is represented by piecewise Chebyshev
polynomials on an adaptive octree. This is the basis of volume
integral-equation solvers with smooth or piecewise-smooth right-hand sides.

## Method

PVFMM implements the kernel-independent FMM (KIFMM) of Ying, Biros &
Zorin: multipole and local expansions are replaced by equivalent-density
representations on cubic surfaces, so only kernel evaluations are needed —
no kernel-specific expansion analysis. Far-field translation operators are
precomputed with symmetry compression and applied through FFT-accelerated
Hadamard products (V-list); near interactions are evaluated directly with
SIMD-vectorized kernels. For the volume FMM, singular and near-singular
quadratures for the near field are precomputed as well (see
{doc}`concepts/precomputed-data`).

## Parallelism

- **Distributed memory (MPI):** Morton-ordered distributed octrees with 2:1
  balance refinement, space-filling-curve partitioning, and Local Essential
  Tree (LET) exchange.
- **Shared memory (OpenMP):** all evaluation phases are multithreaded.
- **SIMD:** kernels are written with explicit vector types (via the bundled
  SCTL library) and compiled with `-march=native` by default.
- **GPU (optional):** CUDA offload of selected interaction phases
  (`--with-cuda` / `PVFMM_ENABLE_CUDA`).

Both `float` and `double` precision are supported throughout (C/Fortran via
`F`/`D` function variants, C++/Python/Julia via type parameters).

## Interfaces

The native interface is header-only C++ ({doc}`api/cpp`). A compiled C
wrapper ({doc}`api/c`) underlies the {doc}`Fortran <api/fortran>`,
{doc}`Python <api/python>`, and {doc}`Julia <api/julia>` bindings.

## References

The method and implementation are described in:

- D. Malhotra and G. Biros, *PVFMM: A parallel kernel independent FMM for
  particle and volume potentials*, Communications in Computational Physics,
  18 (2015), pp. 808–830.
- D. Malhotra and G. Biros, *Algorithm 967: A distributed-memory fast
  multipole method for volume potentials*, ACM Transactions on Mathematical
  Software, 43 (2016).
- L. Ying, G. Biros, and D. Zorin, *A kernel-independent adaptive fast
  multipole algorithm in two and three dimensions*, Journal of Computational
  Physics, 196 (2004), pp. 591–626.

Please cite the first paper (and the second for the volume FMM) when using
PVFMM in published work.
