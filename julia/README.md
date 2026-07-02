# Julia bindings to PVFMM

This directory provides Julia bindings to the PVFMM C API (`include/pvfmm.h`) with a high-level surface aligned to the Python bindings.

## Public API

- `FMMKernel`
- `FMMVolumeContext`
- `FMMParticleContext`
- `FMMVolumeTree`
- `nodes_to_coeff`
- `from_function`
- `from_coefficients`
- `evaluate`
- `leaf_count`
- `get_leaf_coordinates`
- `get_coefficients`
- `get_values`

## Library discovery

The bindings try to load `libpvfmm` from:

1. `ENV["PVFMM"]` (either direct library path or directory containing `libpvfmm.*`)
2. system library search path (`Libdl.find_library(["pvfmm"])`)

## Usage

```julia
using PVFMM
using Random

# Path to directory containing libpvfmm.dylib / libpvfmm.so
ENV["PVFMM"] = "build"

Random.seed!(1)
n_src = 20
n_trg = 10

# Coordinates are stored in array-of-structures layout:
# [x1,y1,z1, x2,y2,z2, ...]
src_pos = rand(3 * n_src)
trg_pos = rand(3 * n_trg)

# For LaplacePotential, only the first component of each source density triplet is used.
sl_den = zeros(3 * n_src)
sl_den[1:n_src] .= randn(n_src)

ctx = FMMParticleContext(0.0, 50, 8, LaplacePotential)
trg_val = evaluate(ctx, src_pos, sl_den, nothing, trg_pos; setup=true)

println("Computed ", length(trg_val), " output values")
println(trg_val[1:min(end, 5)])
```
