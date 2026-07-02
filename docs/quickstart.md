# Getting started

This walk-through goes from a fresh clone to a first FMM evaluation using the
autotools build (see {doc}`install` for details, options, and the CMake
route).

## 1. Clone and build

```bash
git clone --recurse-submodules https://github.com/dmalhotra/pvfmm.git
cd pvfmm
./autogen.sh
./configure
make -j
```

This produces `lib/libpvfmm.la` (and the shared library used by the Python/
Julia bindings). To (optionally) install it system-wide, run `make install`
and set `PVFMM_DIR` as instructed.

## 2. Build the examples

```bash
make all-examples -j          # everything, or a single one:
make examples/bin/example1    # particle N-body demo
make examples/bin/example2    # volume potential demo
```

## 3. Run a particle FMM

`example1` evaluates the Laplace gradient of random particle sources and
checks the result against a direct $O(N^2)$ sum:

```bash
mpirun -n 2 ./examples/bin/example1 -N 100000 -m 10 -omp 4
```

Options: `-N` number of source/target points (required), `-m` multipole order
(default 10), `-omp` OpenMP threads per rank. Look for the printed
`Maximum Absolute Error` and `Maximum Relative Error` — the error decreases
exponentially with the multipole order, roughly two digits for every increase
of `m` by 2 (measured for this example: ~1e-5 at `-m 6`, ~3e-7 at `-m 8`,
~6e-9 at `-m 10`).

## 4. Run a volume FMM

`example2` solves a Poisson problem with a Gaussian right-hand side and
compares against the analytic solution:

```bash
mpirun -n 2 ./examples/bin/example2 -N 100000 -m 10 -q 14 -tol 1e-6 -omp 4
```

Options: `-N` target points (required), `-m` multipole order, `-q` Chebyshev
degree (default 14), `-tol` adaptive-refinement tolerance, `-omp` threads.
The printed `Maximum Error` compares against the analytic solution.

```{important}
The **first** volume-FMM run for a given (kernel, `m`, `q`, precision)
combination precomputes quadrature/translation operators, which can take a
long time (minutes to hours depending on the kernel and orders). The result
is cached in a `Precomp_*.data` file and later runs start instantly. Set the
`PVFMM_DIR` environment variable to a persistent directory to keep the cache
in one place — see {doc}`concepts/precomputed-data`.
```

## 5. Where to go next

- {doc}`tutorials/particle-fmm-cpp` and {doc}`tutorials/volume-fmm-cpp` —
  what the two examples do, line by line.
- {doc}`tutorials/c-fortran` and {doc}`tutorials/python-julia` — the same
  functionality from C, Fortran, Python, and Julia.
- {doc}`tutorials/advanced-drivers` — the `fmm_pts` / `fmm_cheb` test drivers
  with selectable kernels, adaptivity, and convergence checks.
- {doc}`concepts/performance` — thread pinning, accuracy/cost trade-offs,
  and profiling.
- Use `examples/Makefile` as the template for compiling your own programs
  against PVFMM ({doc}`install`).
