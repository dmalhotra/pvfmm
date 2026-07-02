# Test drivers and advanced usage

Beyond the minimal examples, `examples/src/` contains two full-featured test
drivers used for validation, convergence studies, and benchmarking, plus an
example of driving PVFMM through SCTL. All print
per-phase timing/FLOP profiles at the end ({doc}`../concepts/performance`)
and report errors against analytic solutions or direct summation.

## `fmm_pts` — particle FMM driver

```bash
mpirun -n <p> ./examples/bin/fmm_pts -N 1000000 -ker 3 -m 10 -omp <t>
```

| Option | Default | Meaning |
|---|---|---|
| `-N` | (required) | number of source/target points |
| `-M` | 350 | maximum points per octant |
| `-b` | 1 | bounding-box length (0 < b ≤ 1) |
| `-m` | 10 | multipole order (positive, even) |
| `-d` | 15 | maximum tree depth |
| `-sp` | 0 | single precision (0/1) |
| `-dist` | 0 | point distribution: 0 uniform, 1 sphere, 2 ellipse |
| `-ker` | 1 | kernel: 1 Laplace potential, 2 Laplace gradient, 3 Stokes velocity, 4 Helmholtz |
| `-omp` | 1 | OpenMP threads |

Unlike the high-level examples, `fmm_pts` uses the lower-level
`FMM_Tree`/`FMM_Pts` interface directly (`InitFMM_Tree`, `SetupFMM`,
`RunFMM`, …) and prints tree statistics (per-depth leaf histograms) — a good
reference for driving the FMM below the convenience wrappers.

## `fmm_cheb` — volume FMM driver

```bash
mpirun -n <p> ./examples/bin/fmm_cheb -N 8 -test 2 -q 14 -m 10 -tol 1e-6 -adap 1 -omp <t>
```

| Option | Default | Meaning |
|---|---|---|
| `-N` | (required) | number of point sources (tree construction) |
| `-M` | 1 | maximum points per octant |
| `-m` | 10 | multipole order (positive, even) |
| `-q` | 14 | Chebyshev degree |
| `-d` | 15 | maximum tree depth |
| `-tol` | 1e-5 | adaptive refinement tolerance |
| `-adap` | 0 | adaptive refinement (0/1) |
| `-unif` | 0 | uniform point distribution (0/1) |
| `-sp` | 0 | single precision (0/1) |
| `-test` | 1 | test problem (below) |
| `-omp` | 1 | OpenMP threads |

Test problems (analytic input/solution pairs):

1. Laplace, smooth Gaussian, **periodic** boundary
2. Laplace, discontinuous sphere, free space
3. Stokes, smooth Gaussian, free space
4. Biot–Savart, smooth Gaussian, free space
5. Helmholtz, smooth Gaussian, free space

Each run reports absolute/relative L2 and maximum errors for the potential
(and gradient where applicable). This driver has the broadest coverage —
it is the only example exercising the Helmholtz volume FMM and periodic
boundary conditions, and it demonstrates load-balanced adaptive refinement
with the low-level `FMM_Cheb` interface.

The convergence/scaling sweeps in `scripts/` (`conv.sh`, `sscal.sh`,
`wscal.sh`, …) drive these two binaries over parameter grids and collect
results under `result/`.

## `example-sctl` — PVFMM as a far-field backend for SCTL

When PVFMM is compiled with `-DSCTL_HAVE_PVFMM` (the default in this build),
the bundled SCTL library's `sctl::ParticleFMM` can use PVFMM for its
far-field evaluation while composing arbitrary named source/target groups
and kernel pairings:

```cpp
#include "sctl.hpp"
using namespace sctl;

Stokes3D_FSxU kernel_m2l;   // far-field translation kernel
Stokes3D_FxU  kernel_sl;    // single-layer
Stokes3D_DxU  kernel_dl;    // double-layer

ParticleFMM<Real,3> fmm(comm);
fmm.SetAccuracy(10);
fmm.SetKernels(kernel_m2l, kernel_m2l, kernel_sl);
fmm.AddTrg("Potential", kernel_m2l, kernel_sl);
fmm.AddSrc("SingleLayer", kernel_sl, kernel_sl);
fmm.AddSrc("DoubleLayer", kernel_dl, kernel_dl);
fmm.SetKernelS2T("SingleLayer", "Potential", kernel_sl);
fmm.SetKernelS2T("DoubleLayer", "Potential", kernel_dl);

fmm.SetTrgCoord("Potential", trg_coord);
fmm.SetSrcCoord("SingleLayer", sl_coord);
fmm.SetSrcCoord("DoubleLayer", dl_coord, dl_norml);
fmm.SetSrcDensity("SingleLayer", sl_den);
fmm.SetSrcDensity("DoubleLayer", dl_den);

Vector<Real> U;
fmm.Eval(U, "Potential");        // FMM evaluation
fmm.EvalDirect(U, "Potential");  // direct evaluation, for verification
```

See `examples/src/example-sctl.cpp` for the complete program (Stokes single +
double layer).
