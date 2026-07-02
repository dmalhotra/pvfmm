# Volume FMM in C++

This tutorial walks through `examples/src/example2.cpp`, the minimal volume
potential example: it solves the free-space Poisson problem
$-\Delta u = f$ for a Gaussian source

$$f(x) = (4 a^2 r^2 + 6a)\, e^{a r^2}, \qquad u(x) = -e^{a r^2}, \qquad
r = |x - c|,\; a = -160,$$

centered at $c = (0.5, 0.5, 0.5)$, and compares against the analytic
solution.

Build and run it with:

```bash
make examples/bin/example2
mpirun -n 2 ./examples/bin/example2 -N 100000 -m 10 -q 14 -tol 1e-6 -omp 4
```

## 1. Define the source density

The input is a callback that evaluates $f$ at a batch of points (this is the
`ChebFn<Real>` signature — coordinates in, `ker_dim[0]` values per point
out):

```cpp
void fn_input(const double* coord, int n, double* out) {
  double a = -160;
  for (int i = 0; i < n; i++) {
    const double* c = &coord[i*3];
    double r_2 = (c[0]-0.5)*(c[0]-0.5) + (c[1]-0.5)*(c[1]-0.5) + (c[2]-0.5)*(c[2]-0.5);
    out[i] = (2*a*r_2 + 3) * 2*a * exp(a*r_2);
  }
}
```

## 2. Build the Chebyshev tree

`ChebFMM_CreateTree` builds an octree whose leaves carry degree-`cheb_deg`
Chebyshev approximations of `fn_input`, refined adaptively until the
truncation error falls below `tol` (and no leaf holds more than `max_pts`
target points):

```cpp
const pvfmm::Kernel<double>& kernel_fn = pvfmm::LaplaceKernel<double>::potential();

std::vector<double> trg_coord = point_distrib<double>(RandUnif, N, comm);
auto* tree = ChebFMM_CreateTree(cheb_deg, kernel_fn.ker_dim[0], fn_input,
                                trg_coord, comm, tol, max_pts, pvfmm::FreeSpace);
```

(An overload builds the tree from precomputed Chebyshev coefficients on given
leaf nodes instead of a callback — see {doc}`../api/cpp` and the C version in
{doc}`c-fortran`.)

## 3. Load operators, set up, evaluate

```cpp
pvfmm::ChebFMM<double> matrices;
matrices.Initialize(mult_order, cheb_deg, comm, &kernel_fn);

tree->SetupFMM(&matrices);

std::vector<double> trg_value;
size_t n_trg = trg_coord.size() / PVFMM_COORD_DIM;
pvfmm::ChebFMM_Evaluate(trg_value, tree, n_trg);
```

```{important}
The first `Initialize` for a (kernel, `m`, `q`, precision) combination
precomputes the volume quadrature operators, which can take a long time; the
result is cached on disk ({doc}`../concepts/precomputed-data`).
```

To evaluate again (e.g. in an iterative solver where the density was
updated), call `tree->ClearFMMData()` and `ChebFMM_Evaluate` again.

## 4. Beyond target-point values

Besides values at target points, the potential is available as Chebyshev
coefficients per leaf — useful for building solvers that stay in the
polynomial representation:

```cpp
std::vector<double> coeff;
pvfmm::ChebFMM_GetPotentialCoeff(coeff, tree);   // Chebyshev coefficients
pvfmm::ChebFMM_GetLeafCoord(leaf_coord, tree);   // leaf corners
pvfmm::ChebFMM_Coeff2Nodes(node_val, cheb_deg, dof, coeff);  // eval on nodes
```

The example finishes by evaluating the analytic solution at the targets and
printing the maximum error, then `delete tree`.

For richer volume problems (Stokes, Biot–Savart, Helmholtz, periodic
boundaries, adaptive vs uniform refinement), see the `fmm_cheb` driver in
{doc}`advanced-drivers`.
