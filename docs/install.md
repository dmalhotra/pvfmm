# Installation

## Requirements

- a C++ compiler with C++17 support and OpenMP (GCC, Clang, or Intel)
- an MPI implementation providing `mpicxx` (the library itself is built with
  MPI; the C wrapper can also be built without it)
- BLAS and LAPACK
- FFTW3 (double precision; single/long-double precision used when available)
- the SCTL submodule (fetched with `git submodule`)
- optional: CUDA toolkit for the GPU path, Doxygen for the API docs,
  a Fortran compiler for the Fortran examples

On Debian/Ubuntu the following packages are sufficient (this is what CI
uses):

```bash
sudo apt-get install -y libomp-dev openmpi-bin openmpi-common libopenmpi-dev \
                        libopenblas-dev libfftw3-dev
```

On macOS with Homebrew:

```bash
brew install gcc open-mpi openblas fftw autoconf automake libtool
```

Start by cloning the repository including the SCTL submodule:

```bash
git clone --recurse-submodules https://github.com/dmalhotra/pvfmm.git
cd pvfmm
# or, in an existing clone:
git submodule update --init
```

## Building with autotools (primary)

```bash
./autogen.sh          # only needed when building from a git checkout
./configure
make -j
make install          # optional; default prefix /usr/local
```

Useful `configure` options (see `./configure --help` for the full list):

| Option | Purpose |
|---|---|
| `--prefix=PREFIX` | installation prefix (default `/usr/local`) |
| `MPICXX=<compiler>` | MPI C++ compiler wrapper to use |
| `CXXFLAGS=...` | override optimization/ISA flags |
| `F77=<compiler>` | Fortran compiler (for the Fortran interface checks) |
| `--with-fftw=DIR` | FFTW installation directory |
| `--with-fftw-include=DIR`, `--with-fftw-lib=LIB` | FFTW paths individually |
| `--with-blas=<lib>`, `--with-lapack=<lib>` | BLAS/LAPACK libraries |
| `--with-openmp-flag=FLAG` | OpenMP flag if not auto-detected |
| `--with-cuda=PATH` | enable the CUDA path (plus `NVCCFLAGS=...`) |
| `--with-precomp-dir=DIR` | default directory for {doc}`precomputed data <concepts/precomputed-data>` |

The build is compiled with `-O3 -march=native` by default, so the kernels use
the best SIMD instruction set of the build host; override `CXXFLAGS` to
target a different ISA.

`make install` installs the library under `PREFIX/lib/pvfmm`, headers under
`PREFIX/include/pvfmm`, and a `MakeVariables` fragment plus data files under
`PREFIX/share/pvfmm`; it prints a reminder to set:

```bash
export PVFMM_DIR=PREFIX/share/pvfmm
```

Other targets: `make all-examples` (build the {doc}`example programs
<tutorials/particle-fmm-cpp>`), `make doxygen-doc` (Doxygen API reference).

### Linking your own code (Makefile projects)

`configure` generates a `MakeVariables` file (installed to
`$PVFMM_DIR/MakeVariables`). Include it in your Makefile and use the exported
flags — `examples/Makefile` is a ready-made template:

```make
include $(PVFMM_DIR)/MakeVariables

my_program: my_program.cpp
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) $^ $(LDLIBS_PVFMM) -o $@
```

`CXXFLAGS_PVFMM` carries the include paths and feature macros the library
was configured with; `LDLIBS_PVFMM` carries `-lpvfmm` and all numerical
dependencies.

## Building with CMake

```bash
cmake -B build [options]
cmake --build build -j
cmake --install build
```

Targets `pvfmm` (shared) and `pvfmmStatic` are built, along with the
examples. Options:

| Option | Default | Purpose |
|---|---|---|
| `-DPVFMM_ENABLE_CUDA=ON` | `OFF` | build the CUDA/GPU path |

MKL is detected and used for BLAS/LAPACK/FFTW when present; otherwise
standard BLAS, LAPACK, and FFTW are located. The install places a package
config in `PREFIX/share/pvfmm`, so downstream projects can use
`find_package(pvfmm)` (point `pvfmm_DIR` at that directory if the prefix is
non-standard). It defines `PVFMM_INCLUDE_DIR`, `PVFMM_LIB_DIR`,
`PVFMM_SHARED_LIB` / `PVFMM_STATIC_LIB` (library file names), and
`PVFMM_DEP_LIB` / `PVFMM_DEP_INCLUDE_DIR` for the numerical dependencies:

```cmake
find_package(pvfmm REQUIRED)
target_include_directories(my_target PRIVATE ${PVFMM_INCLUDE_DIR} ${PVFMM_DEP_INCLUDE_DIR})
target_link_directories(my_target PRIVATE ${PVFMM_LIB_DIR})
target_link_libraries(my_target PRIVATE ${PVFMM_SHARED_LIB} ${PVFMM_DEP_LIB})
```

The Python and Julia bindings load the *shared* library `libpvfmm.so` (see
{doc}`api/python` and {doc}`api/julia`); both builds produce it.

## Platform notes

Configuration lines that have been used on particular systems (from
`INSTALL`):

```bash
# Cray (e.g. Titan/ORNL)
module load fftw && module swap PrgEnv-pgi PrgEnv-intel
./configure MPICXX="CC" F77="ftn"

# CUDA build
./configure --with-fftw="$FFTW_DIR" LDFLAGS="-L/usr/lib64/nvidia/" \
            --with-cuda="$CUDA_DIR" NVCCFLAGS="-arch=compute_35 -code=sm_35"

# macOS (Homebrew gcc + openblas)
./configure --with-openmp-flag="fopenmp" --with-fftw=$(brew --prefix fftw) \
            --with-blas="-L$(brew --prefix openblas)/lib -lopenblas" \
            --with-lapack="-L$(brew --prefix openblas)/lib -lopenblas"
```

If shared libraries are not found at run time, check `LD_LIBRARY_PATH`
(Linux) or `DYLD_LIBRARY_PATH` (macOS).

## Building this documentation

```bash
pip install -r docs/requirements.txt   # Sphinx toolchain
# doxygen must be on PATH (apt install doxygen / module load doxygen)
sphinx-build -b html docs docs/_build/html
```
