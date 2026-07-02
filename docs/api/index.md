# API reference

PVFMM exposes the same functionality through five language surfaces. The C++
interface is the native one; the C interface wraps it (compiled into
`libpvfmm`), the Fortran interface binds to the C symbols, and the Python and
Julia bindings load `libpvfmm` dynamically.

| Surface | Entry point | Notes |
|---|---|---|
| [C++](cpp.md) | `#include <pvfmm.hpp>` | Header-only templates; full feature set |
| [C](c.md) | `#include <pvfmm.h>`, link `-lpvfmm` | Works with or without MPI |
| [Fortran](fortran.md) | `include 'pvfmm.f90'`, link `-lpvfmm` | `bind(C)` interfaces to the C API |
| [Python](python.md) | `import pvfmm` (package in `python/`) | ctypes + mpi4py, loads `libpvfmm.so` |
| [Julia](julia.md) | `using PVFMM` (package in `julia/`) | Libdl, loads `libpvfmm` |

Feature-parity notes:

- The Helmholtz kernel and the Stokes *stress* kernel are available only from
  C++ (they are not in the C `PVFMMKernel` enum, and hence absent from
  Fortran/Python/Julia).
- Per-axis periodic boundary conditions (`PX`, `PXY`) are available from every
  interface: C++ takes a `pvfmm::BoundaryType`, the C-level interfaces take a
  `PVFMMBoundaryType` (whose values 0/1 remain compatible with the older
  boolean `periodic` flag), and Python/Julia expose `FMMBoundaryType` — see
  {doc}`../concepts/boundary-conditions`.
- Every interface takes an explicit communicator. The C particle-context
  creator initializes MPI on demand, and all communicator arguments are
  ignored when the library is built without MPI.
- Every C/Fortran function comes in a double-precision variant (suffix `D`)
  and a single-precision variant (suffix `F`); C++/Python/Julia select
  precision through the template/dtype/type parameter instead.

```{toctree}
:maxdepth: 1

cpp
c
fortran
python
julia
```
