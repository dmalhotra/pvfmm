# Performance and tuning

## Accuracy/cost knobs

- **Multipole order `m`** (positive, even): controls the accuracy of the
  far-field approximation. Typical values are 4–12; the test drivers default
  to `m=10`. Cost grows roughly with $m^3$ per interaction, so use the
  smallest `m` that meets the accuracy target.
- **Chebyshev degree `q`** (volume FMM): polynomial order per leaf node; the
  volume driver defaults to `q=14`. Higher `q` means fewer, larger leaves for
  smooth data.
- **Refinement tolerance `tol`** (volume FMM): leaves are refined until the
  tails of the Chebyshev expansion fall below this tolerance.
- **`max_pts` / points-per-leaf** (particle FMM): maximum sources per leaf
  node; controls the near/far work balance. The examples use ~100–1000;
  larger values shift work into the direct (near-field) phase.
- **Precision**: single precision (`F` variants / `float`) roughly halves
  memory and bandwidth when ~7 digits suffice.

The kernel evaluation uses explicit SIMD (via SCTL) and the build defaults to
`-O3 -march=native`, so binaries are tuned to the build host's instruction
set — rebuild when moving to a different microarchitecture (see
{doc}`../install`).

## Parallel execution

PVFMM is hybrid MPI + OpenMP. A well-performing configuration binds one MPI
rank per socket (or NUMA domain) with one thread per core, e.g. for two
8-core sockets per node:

```bash
export OMP_PROC_BIND=spread OMP_PLACES=cores
OMP_NUM_THREADS=8 mpirun -n 2 --map-by ppr:1:socket:PE=8 --bind-to core \
  ./examples/bin/fmm_cheb -N 20000 -q 8 -m 8 -test 2 -tol 1e-6 -adap 1 -omp 8
```

Verify the binding (e.g. with `--report-bindings`) — oversubscribed or
unbound threads are the most common cause of poor performance.

## Profiling

The library contains built-in profiling instrumentation (`sctl::Profile`).
Two compile-time macros control it (both can be added to `CXXFLAGS_PVFMM`):

- `SCTL_PROFILE=<level>` — instrumentation depth; defaults to `10` in
  `include/pvfmm_common.hpp`, set `0` to compile it out;
- `SCTL_VERBOSE` — prints progress and diagnostics during the run.

The example drivers end with a profile report (per-phase minimum/average/
maximum times across ranks and FLOP counts) — the primary tool for comparing
configurations. Remember that the first run of a (kernel, m, q) combination
includes operator precomputation; benchmark with a warm
{doc}`precomputed-data <precomputed-data>` cache.
