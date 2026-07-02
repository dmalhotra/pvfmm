# Particle FMM in C++

This tutorial walks through `examples/src/example1.cpp`, which evaluates the
gradient of the Laplace potential of `N` random particle sources at `N`
random targets, then verifies the result against a direct $O(N^2)$ sum.

Build and run it with:

```bash
make examples/bin/example1
mpirun -n 2 ./examples/bin/example1 -N 100000 -m 10 -omp 4
```

## 1. Select a kernel

Every kernel is available as a `pvfmm::Kernel<Real>` descriptor
({doc}`../concepts/kernels`); `ker_dim[0]`/`ker_dim[1]` give the source/target
dimensions per point:

```cpp
const pvfmm::Kernel<double>& kernel_fn = pvfmm::LaplaceKernel<double>::gradient();
```

## 2. Set up sources and targets

Coordinates live in $[0,1]^3$ in array-of-structures layout. The example
draws uniform random points (`point_distrib` is an example helper) and random
densities; it uses single-layer sources only (the double-layer arrays are
created empty but could carry dipole sources — see
{doc}`../api/cpp`):

```cpp
vec trg_coord = point_distrib<double>(RandUnif, N, comm);
vec  sl_coord = point_distrib<double>(RandUnif, N, comm);
vec  dl_coord = point_distrib<double>(RandUnif, 0, comm);   // no double-layer sources

vec sl_den(n_sl * kernel_fn.ker_dim[0]);                    // 1 value/pt for Laplace
vec dl_den(n_dl * (kernel_fn.ker_dim[0] + PVFMM_COORD_DIM)); // density + normal
for (auto& a : sl_den) a = drand48();
```

## 3. Build the tree and load the operators

`PtFMM_CreateTree` partitions the points across MPI ranks and builds the
distributed octree (`max_pts` bounds the number of source points per leaf).
`PtFMM::Initialize` loads/precomputes the translation operators for the
kernel at the requested multipole order, and `SetupFMM` binds them to the
tree:

```cpp
size_t max_pts = 600;
auto* tree = PtFMM_CreateTree(sl_coord, sl_den, dl_coord, dl_den,
                              trg_coord, comm, max_pts, pvfmm::FreeSpace);

pvfmm::PtFMM<double> matrices;
matrices.Initialize(mult_order, comm, &kernel_fn);

tree->SetupFMM(&matrices);
```

## 4. Evaluate

```cpp
vec trg_value;
PtFMM_Evaluate(tree, trg_value, n_trg);
```

`trg_value` receives `n_trg * ker_dim[1]` values (3 per target for the
Laplace gradient), partitioned across ranks according to `n_trg`.

## 5. Re-evaluate with new densities

When only the densities change (same particle positions), skip the tree and
setup work — clear the FMM data and pass the new densities directly:

```cpp
tree->ClearFMMData();
// ... modify sl_den / dl_den ...
PtFMM_Evaluate(tree, trg_value, n_trg, &sl_den, &dl_den);
```

## 6. Verify and clean up

The example samples a subset of targets, computes the direct sum with the
same kernel (`kernel_fn.ker_poten`) gathered over all ranks, and prints the
maximum absolute/relative error, which should reflect the multipole order
(~1e-6 for `m=10`). Finally:

```cpp
delete tree;
```

The same workflow from the other language interfaces:
{doc}`c-fortran`, {doc}`python-julia`.
