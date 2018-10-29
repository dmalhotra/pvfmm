# PVFMM [![Build Status](https://travis-ci.org/dmalhotra/pvfmm.svg?branch=develop)](https://travis-ci.org/dmalhotra/pvfmm)

### new_BC branch
This new_BC branch is forked by Wen Yan from the original pvfmm code to support for 1D, 2D and 3D periodicity in 3D space.

It is implemented in a quick and dirty way. To choose periodicity, define a global variable 
extern PeriodicType pvfmm::periodicType
in one .h or .cpp file and set it to one of the following enum values:
NONE,PX,PXY,PXYZ

to use this package, refer to the repo https://github.com/wenyan4work/PeriodicFMM

update: modified to remove fortran compiler dependence and make it work with clang-openmp/llvm. Specify BLAS/LAPACK directly through CXXFLAGS and LDFLAGS.


My thanks go to Dhairya Malhotra for providing patches and instructions to his code.
Below is the original pvfmm readme file.

### What is PVFMM?

   PVFMM is a library for solving certain types of elliptic partial
   differential equations. 
    
   * We support Stokes, Poisson, and Helmholtz problems on the unit
     cube, with free-space or periodic boundary conditions, with
     constant or mildly varying coefficients. Our method is based on
     volume potential integral equation formulation accelerated by the
     Kernel Independent Fast Multipole Method. 


### How to get PVFMM

   For the latest stable release of PVFMM visit [pvfmm.org](http://pvfmm.org)

### License

   PVFMM is distributed under the LGPLv3 licence. See COPYING in
   the top-level directory of the distribution. 

### Installing PVFMM

   To install PVFMM, follow the steps in the INSTALL file, which is
   located in the top directory of the source distribution. 


### Using PVFMM

   The file examples/Makefile can be used as a template makefile for any
   project using the library. In general the MakeVariables file should
   be included in any makefile and CXXFLAGS_PVFMM and LDFLAGS_PVFMM should
   be used to compile the code.

   Two very simple examples illustrating usage of the library are available:
      For particle N-body  : examples/src/example1.cpp
      For volume potentials: examples/src/example2.cpp 

   To compile these examples:
      make examples/bin/example1
      make examples/bin/example2

   * The volume potentials example will take a long time, the first time
     it is used, since it has to precompute quadrature rules. This data
     is saved to a file and used for subsequent runs. See INSTALL for
     the configure option '--with-precomp-dir=DIR' to set the default
     path for precomputed data.


### Acknowledgment

   This software has been developed as part of the work supported by,
   * US National Institutes of Health/10042242
   * US Department of Energy/DE-SC0010518
   * US Department of Energy/DE-SC0009286
   * US National Science Foundation/CCF-1337393
   * US Air Force Office for Scientific Research /FA9550-12-10484

   The authors would also like to thank ORNL/OLCF and TACC for providing
   access to computing resources for the development, testing and
   benchmarking of this software.

