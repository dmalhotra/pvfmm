# Using the Python and Julia bindings

Both bindings wrap the {doc}`C interface <../api/c>` and require the shared
library `libpvfmm.so` — build it first and point the `PVFMM` environment
variable at the directory containing it (see {doc}`../api/python` and
{doc}`../api/julia` for installation details).

## Python: particle FMM

Based on `python/examples/example1.py` (run with
`mpirun -n 2 python -m mpi4py example1.py`):

```python
import numpy as np
from mpi4py import MPI
import pvfmm

N = 20000
src_pos = np.random.rand(3 * N)          # coordinates in [0,1]^3, AoS layout
src_den = np.random.rand(3 * N) - 0.5    # Biot-Savart: 3 values per source
trg_pos = np.random.rand(3 * N)

fmm = pvfmm.FMMParticleContext(
    box_size=-1,            # <= 0: free space
    max_points=1000,        # max points per leaf
    multipole_order=10,     # accuracy (even)
    kernel=pvfmm.FMMKernel.BiotSavartPotential,
    comm=MPI.COMM_WORLD,
)

trg_val = fmm.evaluate(src_pos, src_den, None, trg_pos)             # with setup
trg_val = fmm.evaluate(src_pos, src_den, None, trg_pos, setup=False)  # densities only
```

The example verifies the result against a direct sum compiled with numba.

## Python: volume FMM

Based on `python/examples/example2.py` (Stokes velocity). The source density
callback must be a C function pointer; `numba.cfunc` produces one without
writing C:

```python
import numba
import pvfmm

@numba.cfunc(numba.types.void(
    numba.types.CPointer(numba.types.double),   # coord (3n)
    numba.types.int64,                          # n
    numba.types.CPointer(numba.types.double),   # out (n * data_dim)
    numba.types.voidptr), nopython=True)        # ctx
def fn_input_C(coord_, n, out_, ctx):
    coord = numba.carray(coord_, n * 3)
    out = numba.carray(out_, n * 3)
    # ... evaluate the density ...

fmm = pvfmm.FMMVolumeContext(multipole_order, cheb_deg,
                             pvfmm.FMMKernel.StokesVelocity, comm)

tree = pvfmm.FMMVolumeTree.from_function(
    cheb_deg, 3, fn_input_C.ctypes, None, trg_coord, comm,
    tol=1e-6, max_pts=100, periodic=False, init_depth=0)

trg_val = tree.evaluate(fmm, loc_size=len(trg_coord) // 3)
```

The coefficient path mirrors the C example: prepare density values on the
tensor-product Chebyshev nodes of chosen leaves, convert with
`pvfmm.nodes_to_coeff`, build with `FMMVolumeTree.from_coefficients`, and
read the result back with `tree.get_coefficients()` / `tree.get_values()`.

## Julia: particle FMM

Based on `julia/test/reference_comparison.jl`:

```julia
ENV["PVFMM"] = "/path/to/dir-containing-libpvfmm"
using PVFMM

N = 10000
src_pos = rand(3N); src_den = rand(3N) .- 0.5; trg_pos = rand(3N)

# comm omitted -> MPI_COMM_WORLD (via the C *World constructors)
ctx = FMMParticleContext(-1.0, 1000, 10, PVFMM.StokesVelocity)

trg_val = evaluate(ctx, src_pos, src_den, nothing, trg_pos)
trg_val = evaluate(ctx, src_pos, src_den, nothing, trg_pos; setup=false)
```

An explicit communicator (e.g. `MPI.COMM_WORLD` from MPI.jl) can be passed as
the fifth argument. The volume-FMM path (`FMMVolumeContext`,
`from_function` / `from_coefficients`, `evaluate`, `get_values`) follows the
Python API one-to-one — see {doc}`../api/julia`.
