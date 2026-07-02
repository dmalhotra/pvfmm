# Python interface

The `pvfmm` Python package (in
[`python/`](https://github.com/dmalhotra/pvfmm/tree/develop/python)) wraps the
{doc}`C interface <c>` using `ctypes`, `numpy`, and `mpi4py`.

## Installation and setup

Build and install the PVFMM library first, then:

```bash
cd python && pip install .
```

If `libpvfmm.so` is not on the system library search path, point the `PVFMM`
environment variable at the directory containing it (e.g. the CMake `build/`
directory or the install prefix's `lib/pvfmm`). Run MPI programs through
mpi4py so that `MPI_Init`/`MPI_Finalize` are handled correctly:

```bash
export PVFMM=/path/to/dir-containing-libpvfmm
mpirun -n 2 python -m mpi4py your_program.py
```

Precision is selected per object with the `dtype` argument (`numpy.float64`,
the default, or `numpy.float32`); all array arguments must be 1-D contiguous
NumPy arrays of that dtype in `[x1 y1 z1 x2 y2 z2 ...]` (array-of-structures)
layout.

```{note}
The classes below have no docstrings yet; this page is the reference.
Signatures correspond to `python/src/pvfmm/wrapper.py`.
```

## FMMKernel

```python
class FMMKernel(Enum):
    LaplacePotential    = 0
    LaplaceGradient     = 1
    StokesPressure      = 2
    StokesVelocity      = 3
    StokesVelocityGrad  = 4
    BiotSavartPotential = 5
```

Mirrors the C `PVFMMKernel` enum. Source/target dimensions per kernel are
listed in {doc}`../concepts/kernels`.

## FMMBoundaryType

```python
class FMMBoundaryType(Enum):
    FreeSpace = 0
    PXYZ      = 1
    PX        = 2
    PXY       = 3
    Periodic  = 1   # alias for PXYZ
```

Mirrors the C `PVFMMBoundaryType` enum
({doc}`../concepts/boundary-conditions`); functions taking a `periodic`
argument also still accept a plain bool (`False`/`True` = free space / fully
periodic).

## Particle FMM

```python
class FMMParticleContext(box_size, max_points, multipole_order, kernel, comm,
                         dtype=np.float64, boundary=None)
```

Creates a particle-FMM context (`PVFMMCreateContext*`). `box_size` is the
domain length and the period along the periodic directions (`<= 0` allowed
only for free space); `max_points` is the maximum number of points per leaf
node; `multipole_order` must be positive and even; `comm` is an `mpi4py`
communicator. Passing `boundary=FMMBoundaryType.PX` (etc.) selects the
boundary conditions explicitly; with `boundary=None` the sign of `box_size`
decides (`> 0` fully periodic, otherwise free space). The underlying context
is freed when the object is garbage-collected.

```python
FMMParticleContext.evaluate(src_pos, sl_den, dl_den, trg_pos, setup=True) -> np.ndarray
```

Evaluates the potential at the target points (`PVFMMEval*`). `sl_den`
(single-layer density, same length as `src_pos`) and `dl_den` (double-layer
density + normals, twice the length of `src_pos`) may each be `None`. Pass
`setup=True` whenever source or target *positions* changed since the last
call; `setup=False` re-uses the tree when only densities changed.

```{caution}
The returned array has length `len(trg_pos)`, i.e. three values per target,
which matches kernels with three output components (Stokes velocity,
Biot–Savart, Laplace gradient). For kernels with a different target dimension
(see {doc}`../concepts/kernels`) only the first `n_trg * trg_dim` entries are
meaningful — and `StokesVelocityGrad` (9 components) does not fit and must not
be used through this method.
```

## Volume FMM

```python
class FMMVolumeContext(multipole_order, chebyshev_degree, kernel, comm, dtype=np.float64)
```

Builds (or loads from cache — see {doc}`../concepts/precomputed-data`) the
volume-FMM translation operators (`PVFMMCreateVolumeFMM*`).

```python
FMMVolumeTree.from_function(cheb_deg, data_dim, fn, context, trg_coord, comm,
                            tol, max_pts, periodic, init_depth) -> FMMVolumeTree
```

Builds an adaptively refined Chebyshev tree from a source-density callback
(`PVFMMCreateVolumeTree*`). `fn` must be a `ctypes` function pointer with C
signature `void fn(const double* coord, long n, double* out, const void* ctx)`
(use `pvfmm.ffi.double_volume_callback` / `float_volume_callback`, e.g. via
`numba.cfunc` as in `python/examples/example2.py`); `context` is passed
through to the callback as `ctx`.

```python
FMMVolumeTree.from_coefficients(cheb_deg, data_dim, leaf_coord, fn_coeff,
                                trg_coord, comm, periodic) -> FMMVolumeTree
```

Builds the tree from given leaf-node coordinates and Chebyshev coefficients
(`PVFMMCreateVolumeTreeFromCoeff*`); `fn_coeff` must have length
`N_leaf * data_dim * (cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6`. `trg_coord` may
be `None`. In both constructors `periodic` accepts a bool or an
`FMMBoundaryType`.

```python
FMMVolumeTree.evaluate(fmm: FMMVolumeContext, loc_size: int) -> np.ndarray
```

Runs the volume FMM and returns the potential at the target points, an array
of length `n_trg * trg_dim` for the context's kernel.

```python
FMMVolumeTree.leaf_count() -> int                 # number of leaf nodes
FMMVolumeTree.get_leaf_coordinates() -> np.ndarray  # leaf corners, 3 per leaf
FMMVolumeTree.get_coefficients() -> np.ndarray    # Chebyshev coeffs of the potential
FMMVolumeTree.get_values() -> np.ndarray          # potential on Chebyshev nodes
```

`get_coefficients`/`get_values` require that `evaluate` has been called
(they raise `ValueError` otherwise).

```python
pvfmm.nodes_to_coeff(N_leaf, cheb_deg, dof, node_val) -> np.ndarray
```

Converts function values on tensor-product Chebyshev nodes (first kind) to
Chebyshev coefficients (`PVFMMNodes2Coeff*`) — the usual way to prepare
`fn_coeff` for `from_coefficients`.

## Examples

- `python/examples/example1.py` — particle FMM (Biot–Savart), timed against a
  direct O(N²) sum compiled with numba.
- `python/examples/example2.py` — volume FMM (Stokes velocity), building the
  tree both from a `numba.cfunc` callback and from Chebyshev coefficients.
