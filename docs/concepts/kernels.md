# Kernel functions

PVFMM ships kernels for four elliptic PDE families. A kernel is identified by
a `pvfmm::Kernel<Real>` descriptor in C++ (obtained from the kernel-family
structs below) or by a `PVFMMKernel` enum value in the C-level interfaces
(C, Fortran, Python, Julia).

Densities and values are per-point quantities with the dimensions listed
below; arrays are laid out in array-of-structures order
(`[u1 v1 w1 u2 v2 w2 ...]`).

| Kernel | C++ accessor | C enum | src dim | trg dim |
|---|---|---|---|---|
| Laplace potential | `LaplaceKernel<Real>::potential()` | `PVFMMLaplacePotential` | 1 | 1 |
| Laplace gradient | `LaplaceKernel<Real>::gradient()` | `PVFMMLaplaceGradient` | 1 | 3 |
| Stokes velocity | `StokesKernel<Real>::velocity()` | `PVFMMStokesVelocity` | 3 | 3 |
| Stokes pressure | `StokesKernel<Real>::pressure()` | `PVFMMStokesPressure` | 3 | 1 |
| Stokes stress | `StokesKernel<Real>::stress()` | — (C++ only) | 3 | 9 |
| Stokes velocity gradient | `StokesKernel<Real>::vel_grad()` | `PVFMMStokesVelocityGrad` | 3 | 9 |
| Biot–Savart | `BiotSavartKernel<Real>::potential()` | `PVFMMBiotSavartPotential` | 3 | 3 |
| Helmholtz | `HelmholtzKernel<Real>::potential()` | — (C++ only) | 2 | 2 |

## Formulas

With target point $x$, source points $y_j$, and source densities $f_j$
(scalar or vector according to the table), the particle sums are:

**Laplace** — Green's function of the Poisson equation $-\Delta u = f$:

$$u(x) = \frac{1}{4\pi} \sum_j \frac{f_j}{|x - y_j|},$$

and `gradient()` evaluates $\nabla u(x)$.

**Stokes velocity** — the Stokeslet (free-space Green's function of the
Stokes equations with unit viscosity):

$$u(x) = \frac{1}{8\pi} \sum_j \left( \frac{f_j}{r_j} + \frac{(f_j \cdot r_j)\, r_j}{r_j^3} \right), \qquad r_j = x - y_j,\; r_j = |r_j|,$$

with the associated pressure

$$p(x) = \frac{1}{4\pi} \sum_j \frac{f_j \cdot r_j}{r_j^3},$$

and `stress()` / `vel_grad()` returning the 3×3 stress and velocity-gradient
tensors per target (9 values, row-major).

**Biot–Savart** — velocity induced by vortex sources $\omega_j$:

$$u(x) = \frac{1}{4\pi} \sum_j \frac{\omega_j \times r_j}{r_j^3}.$$

**Helmholtz** — Green's function of the Helmholtz equation
$-\Delta u - \mu^2 u = f$:

$$u(x) = \frac{1}{4\pi} \sum_j \frac{e^{i \mu |x-y_j|}}{|x - y_j|} f_j,$$

where values are complex numbers stored as interleaved (real, imaginary)
pairs — hence dimensions (2, 2).

```{note}
The Helmholtz wavenumber is currently fixed at $\mu = 20\pi$ in
`include/kernel.txx` (10 wavelengths across the unit cube); change it there
and rebuild to use a different wavenumber.
```

## Double-layer sources

The Laplace and Stokes kernels also provide double-layer (dipole) variants
used when double-layer sources are supplied (e.g. the `dl_*` arguments of
`PtFMM_CreateTree` / `PVFMMEval`). Double-layer densities carry the source
normal alongside the density: 2×3 = 6 values per point for Stokes
(density + normal), and 3 + 1 = 4 values per point for Laplace.

## Custom kernels (C++)

New kernels can be assembled with `pvfmm::BuildKernel` from SIMD micro-kernels
(see `GenericKernel` and the definitions in `include/kernel.txx`); the
`Kernel<Real>` descriptor bundles the evaluation functions used for each FMM
translation type together with scale-invariance metadata used by the
precomputation. Using a custom kernel only requires that it be smooth away
from the origin and (for the fast setup path) homogeneous under scaling.
