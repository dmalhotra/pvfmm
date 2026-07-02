# C++ interface

The high-level C++ interface is declared in
[`include/pvfmm.hpp`](https://github.com/dmalhotra/pvfmm/blob/develop/include/pvfmm.hpp).
Everything lives in namespace `pvfmm` and is templated on the floating-point
type `Real` (`float` or `double`). The library core is header-only; programs
still link `-lpvfmm` and its numerical dependencies — use the flags exported
in `MakeVariables` (see {doc}`../install`).

Communicators are passed as `sctl::Comm` (from the bundled SCTL library), not
raw `MPI_Comm`. Construct one from an MPI communicator or use the predefined
world/self communicators:

```cpp
sctl::Comm comm(MPI_COMM_WORLD);   // wrap an existing MPI communicator
sctl::Comm comm = sctl::Comm::World();
sctl::Comm comm = sctl::Comm::Self();
```

Usage walk-throughs: {doc}`../tutorials/particle-fmm-cpp` and
{doc}`../tutorials/volume-fmm-cpp`.

## Kernel functions

A kernel is selected by taking a reference to a `pvfmm::Kernel<Real>`
descriptor from one of the kernel-family structs (see
{doc}`../concepts/kernels` for dimensions and formulas):

```cpp
const pvfmm::Kernel<double>& kernel_fn = pvfmm::LaplaceKernel<double>::potential();
```

| Family | Members |
|---|---|
| `pvfmm::LaplaceKernel<Real>` | `potential()`, `gradient()` |
| `pvfmm::StokesKernel<Real>` | `velocity()`, `pressure()`, `stress()`, `vel_grad()` |
| `pvfmm::BiotSavartKernel<Real>` | `potential()` |
| `pvfmm::HelmholtzKernel<Real>` | `potential()` |

## Boundary conditions

Semantics are described in {doc}`../concepts/boundary-conditions`.

```{doxygenenum} pvfmm::BoundaryType
```

## Volume FMM

### Types

```{doxygentypedef} pvfmm::ChebFMM_Node
```

```{doxygentypedef} pvfmm::ChebFMM
```

```{doxygentypedef} pvfmm::ChebFMM_Tree
```

```{doxygentypedef} pvfmm::ChebFn
```

### Functions

```{doxygenfunction} pvfmm::ChebFMM_CreateTree(int, int, ChebFn<Real>, const std::vector<Real>&, const sctl::Comm&, Real, int, BoundaryType, int)
```

```{doxygenfunction} pvfmm::ChebFMM_CreateTree(int, const std::vector<Real>&, const std::vector<Real>&, const std::vector<Real>&, const sctl::Comm&, BoundaryType)
```

```{doxygenfunction} pvfmm::ChebFMM_Evaluate
```

```{doxygenfunction} pvfmm::ChebFMM_GetPotentialCoeff
```

```{doxygenfunction} pvfmm::ChebFMM_GetLeafCoord
```

```{doxygenfunction} pvfmm::ChebFMM_Coeff2Nodes
```

```{doxygenfunction} pvfmm::ChebFMM_Nodes2Coeff
```

### Operator class

The `ChebFMM<Real>` (i.e. `pvfmm::FMM_Cheb`) object manages the volume-FMM
translation operators; construct it once per (kernel, multipole order,
Chebyshev degree) combination and pass it to `FMM_Tree::SetupFMM`.

```{doxygenclass} pvfmm::FMM_Cheb
:members: Initialize
```

## Particle FMM

### Types

```{doxygentypedef} pvfmm::PtFMM
```

```{doxygentypedef} pvfmm::PtFMM_Tree
```

```{doxygentypedef} pvfmm::PtFMM_Node
```

```{doxygentypedef} pvfmm::PtFMM_Data
```

### Functions

```{doxygenfunction} pvfmm::PtFMM_CreateTree(const std::vector<Real>&, const std::vector<Real>&, const std::vector<Real>&, const std::vector<Real>&, const std::vector<Real>&, const sctl::Comm&, int, BoundaryType, int)
```

```{doxygenfunction} pvfmm::PtFMM_CreateTree(const std::vector<Real>&, const std::vector<Real>&, const std::vector<Real>&, const sctl::Comm&, int, BoundaryType, int)
```

```{doxygenfunction} pvfmm::PtFMM_Evaluate
```

### Operator class

```{doxygenclass} pvfmm::FMM_Pts
:members: Initialize
```

## Tree evaluation

The tree returned by the `*_CreateTree` functions is a
`FMM_Tree<FMM_Mat_t>`. In the typical workflow only three members are used
directly — `SetupFMM` (once per tree/operator pairing, or after the tree
changed), the `*_Evaluate` free functions (which call `RunFMM` internally),
and `ClearFMMData` (before re-evaluating with new source densities):

```{doxygenclass} pvfmm::FMM_Tree
:members: SetupFMM, RunFMM, ClearFMMData
```
