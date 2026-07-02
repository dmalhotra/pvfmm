# PVFMM

PVFMM (Parallel Volume Fast Multipole Method) is a library for solving
certain types of elliptic partial differential equations: it supports Stokes,
Poisson, Biot–Savart, and Helmholtz problems on the unit cube, with
free-space or periodic boundary conditions, with constant or mildly varying
coefficients. The method is based on a volume-potential integral-equation
formulation accelerated by the kernel-independent fast multipole method
(KIFMM), and the same machinery provides fast $O(N)$ particle N-body sums.

PVFMM is written in C++17 (MPI + OpenMP, optional CUDA) with interfaces for
C, Fortran, Python, and Julia. It scales from laptops to
distributed-memory supercomputers.

- Source code: [github.com/dmalhotra/pvfmm](https://github.com/dmalhotra/pvfmm)
- Doxygen class reference: [pvfmm.org](http://pvfmm.org)
- License: LGPLv3 (see `COPYING` in the source distribution)

```{toctree}
:caption: Getting started
:maxdepth: 1

overview
install
quickstart
```

```{toctree}
:caption: Tutorials
:maxdepth: 1

tutorials/particle-fmm-cpp
tutorials/volume-fmm-cpp
tutorials/c-fortran
tutorials/python-julia
tutorials/advanced-drivers
```

```{toctree}
:caption: Concepts
:maxdepth: 1

concepts/kernels
concepts/boundary-conditions
concepts/precomputed-data
concepts/performance
```

```{toctree}
:caption: API reference
:maxdepth: 2

api/index
```

## Citing PVFMM

- D. Malhotra and G. Biros, *PVFMM: A parallel kernel independent FMM for
  particle and volume potentials*, Communications in Computational Physics,
  18 (2015), pp. 808–830.
- D. Malhotra and G. Biros, *Algorithm 967: A distributed-memory fast
  multipole method for volume potentials*, ACM Transactions on Mathematical
  Software, 43 (2016).

## Acknowledgments

This software has been developed as part of work supported by the US
National Institutes of Health (10042242), the US Department of Energy
(DE-SC0010518, DE-SC0009286), the US National Science Foundation
(CCF-1337393), and the US Air Force Office of Scientific Research
(FA9550-12-10484). The authors also thank ORNL/OLCF and TACC for providing
access to computing resources used in the development, testing, and
benchmarking of this software.
