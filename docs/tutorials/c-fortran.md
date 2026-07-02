# Using the C and Fortran interfaces

Four example programs demonstrate the {doc}`C <../api/c>` and
{doc}`Fortran <../api/fortran>` interfaces (build them with
`make all-examples`, or individually as `make examples/bin/example-c` etc.):

| Program | Interface | What it does |
|---|---|---|
| `example-c.c` | C | particle FMM, Biot–Savart kernel |
| `example2-c.c` | C | volume FMM, Stokes velocity kernel |
| `example-f.f90` | Fortran | particle FMM, Biot–Savart kernel |
| `example2-f.f90` | Fortran | volume FMM, Stokes velocity kernel |

## Particle FMM in C (`example-c.c`)

The whole workflow is three calls — create a context, evaluate, destroy:

```c
#include <pvfmm.h>

MPI_Init(&argc, &argv);

// box_size <= 0 selects free-space boundary conditions; a positive value
// makes the domain periodic with that period.
double box_size = -1;
int points_per_box = 1000;   // max points per leaf (tuning parameter)
int multipole_order = 10;    // accuracy (positive, even)
void* ctx = PVFMMCreateContextD(box_size, points_per_box, multipole_order,
                                PVFMMBiotSavartPotential, MPI_COMM_WORLD);

// src_X: 3*Ns coords, src_V: 3*Ns densities, trg_X: 3*Nt coords,
// trg_V: 3*Nt outputs; NULL = no double-layer sources; setup=1 because
// the particle positions are new.
PVFMMEvalD(src_X, src_V, NULL, Ns, trg_X, trg_V, Nt, ctx, 1);

// Densities changed but positions did not: skip the setup work.
PVFMMEvalD(src_X, src_V, NULL, Ns, trg_X, trg_V, Nt, ctx, 0);

PVFMMDestroyContextD(&ctx);
MPI_Finalize();
```

The example times both evaluations against a direct OpenMP $O(N^2)$ loop and
prints the maximum relative error.

## Volume FMM in C (`example2-c.c`)

This example solves a Stokes problem (velocity kernel, 3 source / 3 target
values per point) and exercises the whole volume C API. Operators are built
once:

```c
void* fmm = PVFMMCreateVolumeFMMD(mult_order, cheb_deg, PVFMMStokesVelocity, comm);
```

The source density can be supplied in two ways.

**(a) From a function callback** — PVFMM refines the tree adaptively (to
tolerance `1e-6`, at most `100` targets per leaf):

```c
void fn_input(const double* coord, long n, double* out, const void* ctx);

void* tree = PVFMMCreateVolumeTreeD(cheb_deg, kdim0, fn_input, NULL /*ctx*/,
                                    trg_coord, Nt, comm,
                                    1e-6 /*tol*/, 100 /*max_pts*/,
                                    false /*periodic*/, 0 /*init_depth*/);
PVFMMEvaluateVolumeFMMD(trg_value, tree, fmm, Nt);
```

**(b) From Chebyshev coefficients on chosen leaf nodes** — the example builds
a uniform depth-3 tree by hand: it evaluates the density at the tensor-product
Chebyshev nodes of each leaf, converts values to coefficients, and constructs
the tree from them:

```c
PVFMMNodes2CoeffD(dens_coeff, Nleaf_loc, cheb_deg, kdim0, dens_value);
tree = PVFMMCreateVolumeTreeFromCoeffD(Nleaf_loc, cheb_deg, kdim0, leaf_coord,
                                       dens_coeff, NULL, 0, comm, false);
PVFMMEvaluateVolumeFMMD(NULL, tree, fmm, 0);   // no point targets
```

The result is then read back in coefficient form and evaluated on the
Chebyshev nodes:

```c
long Nleaf = PVFMMGetLeafCountD(tree);
PVFMMGetPotentialCoeffD(potn_coeff, tree);
PVFMMCoeff2NodesD(potn_value, Nleaf, cheb_deg, kdim1, potn_coeff);
PVFMMGetLeafCoordD(leaf_coord, tree);          // to locate the nodes
```

Cleanup: `PVFMMDestroyVolumeTreeD(&tree); PVFMMDestroyVolumeFMMD(&fmm);`.

## Fortran (`example-f.f90`, `example2-f.f90`)

The Fortran programs mirror the C ones. The interface file is included
directly in the program (together with MPI):

```fortran
program main
  use iso_c_binding
  implicit none
  include 'mpif.h'
  include 'pvfmm.f90'

  type(c_ptr) :: fmm_ctx
  call MPI_Init(ierror)

  call PVFMMCreateContextD(fmm_ctx, box_size, points_per_leaf, &
                           multipole_order, PVFMMBiotSavartPotential, &
                           MPI_COMM_WORLD)

  call PVFMMEvalD(Xs, Vs, Ns, Xt, Vt, Nt, fmm_ctx, setup)

  call PVFMMDestroyContextD(fmm_ctx)
  call MPI_Finalize(ierror)
end
```

Note the {doc}`differences from the C interface <../api/fortran>`: argument
order in `PVFMMEval` (sources, targets, context, setup), single-layer
densities only, and handles returned through the first argument.
