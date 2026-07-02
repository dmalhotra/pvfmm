# Precomputed operator files

For each combination of (kernel, multipole order `m`, Chebyshev degree `q`,
precision), PVFMM precomputes translation operators (with symmetry
compression). This happens automatically the first time such a combination is
used and the result is written to a data file that is loaded on subsequent
runs:

```
Precomp_<kernel-name>[_q<q>]_m<m>[_f].data
```

- `<kernel-name>` — e.g. `laplace`, `laplace_grad`, `stokes_vel`,
  `biot_savart`, `helmholtz`;
- `_q<q>` — present for volume-FMM operators (Chebyshev degree);
- `_f` — single-precision variant (no suffix = double).

## Search path

The directory for these files is resolved in this order:

1. the compile-time default set with `./configure --with-precomp-dir=DIR`;
2. the `PVFMM_DIR` environment variable;
3. the current working directory.

## First-run cost

Particle-FMM operator construction takes seconds to minutes. Volume-FMM
operators (which include singular near-field quadratures) can take
**minutes to hours** and produce files from a few MB up to ~10 GB
(e.g. Helmholtz with `q=14`). Plan accordingly:

- set `PVFMM_DIR` to a persistent, shared location so every run and every
  user of a machine reuses the same cache;
- do not delete the `Precomp_*.data` files casually — they are expensive to
  regenerate;
- the cache is portable across runs but tied to precision and to the
  (kernel, m, q) triple, so parameter sweeps over `m`/`q` populate several
  files.

On load, rank 0 of the communicator reads the file and broadcasts its
contents to the other ranks.
