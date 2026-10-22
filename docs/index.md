% Copyright (C) 2010-2026 The Simons Foundation, Inc. and the PVFMM authors (see NOTICE)
% Author: Dhairya Malhotra
% SPDX-License-Identifier: LGPL-3.0-only

# PVFMM

PVFMM (Parallel Volume Fast Multipole Method) is a library for solving
certain types of elliptic partial differential equations: it supports Stokes,
Poisson, Biot–Savart, and Helmholtz problems on the unit cube, with
free-space or periodic boundary conditions. The method is based on a
volume-potential integral-equation
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

If you use PVFMM, or code derived from it, please cite the software and the
papers below:

- D. Malhotra, G. Biros, C. D. Yu, B. Ward, X. Gao, R. Blackwell, W. Yan,
  *PVFMM: Parallel Volume Fast Multipole Method*,
  <https://github.com/dmalhotra/pvfmm>,
  [doi:10.5281/zenodo.23511939](https://doi.org/10.5281/zenodo.23511939).
- D. Malhotra and G. Biros, *PVFMM: A parallel kernel independent FMM for
  particle and volume potentials*, Communications in Computational Physics,
  18 (2015), pp. 808–830,
  [doi:10.4208/cicp.020215.150515sw](https://doi.org/10.4208/cicp.020215.150515sw).
- D. Malhotra and G. Biros, *Algorithm 967: A distributed-memory fast
  multipole method for volume potentials*, ACM Transactions on Mathematical
  Software, 43 (2016), pp. 1–27,
  [doi:10.1145/2898349](https://doi.org/10.1145/2898349).

<details>
<summary>BibTeX</summary>

```bibtex
@software{pvfmm,
  author    = {Malhotra, Dhairya and Biros, George and Yu, Chenhan D. and Ward, Brian and Gao, Xuanzhao and Blackwell, Robert and Yan, Wen},
  title     = {{PVFMM}: Parallel Volume Fast Multipole Method},
  url       = {https://github.com/dmalhotra/pvfmm},
  doi       = {10.5281/zenodo.23511939},
  version   = {2.0.0},
  year      = {2026}
}

@article{malhotra2015pvfmm,
  author  = {Malhotra, Dhairya and Biros, George},
  title   = {{PVFMM}: A Parallel Kernel Independent {FMM} for Particle and Volume Potentials},
  journal = {Communications in Computational Physics},
  volume  = {18},
  number  = {3},
  pages   = {808--830},
  year    = {2015},
  doi     = {10.4208/cicp.020215.150515sw}
}

@article{malhotra2016algorithm967,
  author  = {Malhotra, Dhairya and Biros, George},
  title   = {Algorithm 967: A Distributed-Memory Fast Multipole Method for Volume Potentials},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {43},
  number  = {2},
  pages   = {1--27},
  year    = {2016},
  doi     = {10.1145/2898349}
}
```

</details>

## Acknowledgments

This software has been developed as part of work supported by the US
National Institutes of Health (10042242), the US Department of Energy
(DE-SC0010518, DE-SC0009286), the US National Science Foundation
(CCF-1337393), and the US Air Force Office of Scientific Research
(FA9550-12-10484). The authors also thank ORNL/OLCF and TACC for providing
access to computing resources used in the development, testing, and
benchmarking of this software.
