AC_PREREQ([2.61])
AC_INIT([PVFMM],[m4_esyscmd_s([cat version.txt])],[contact@pvfmm.org])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall no-define])

AC_CONFIG_SRCDIR([src/pvfmm-wrapper.cpp])


# Set language to C++
AC_LANG([C++])


# Checks for programs.
AX_PROG_CXX_MPI(,[CXX=$MPICXX],[])
AX_MPI([CXX=$MPICXX; CC=$MPICC; CFLAGS=$CFLAGS],
       [AC_MSG_ERROR([Cannot find mpicxx. Make sure that MPI is installed and location of compiler is in PATH.])])

# MPI C and Fortran compiler wrappers, exported through MakeVariables
# (CC_PVFMM/FC_PVFMM) for the examples and downstream makefiles. Prefer
# siblings of MPICXX so a full-path MPICXX=... invocation selects the
# matching wrappers; fall back to a PATH search. Both are optional (only
# the C/Fortran examples need them).
AC_ARG_VAR(MPICC,[MPI C compiler command])
AC_ARG_VAR(MPIFC,[MPI Fortran compiler command])
mpi_bindir=`AS_DIRNAME(["$MPICXX"])`
if test -z "$MPICC" && test -x "$mpi_bindir/mpicc"; then MPICC="$mpi_bindir/mpicc"; fi
if test -z "$MPIFC"; then
  for pvfmm_mpifc in mpifort mpif90; do
    if test -x "$mpi_bindir/$pvfmm_mpifc"; then MPIFC="$mpi_bindir/$pvfmm_mpifc"; break; fi
  done
fi
AC_CHECK_PROGS(MPICC, [mpicc], [])
AC_CHECK_PROGS(MPIFC, [mpifort mpif90], [])
AC_SUBST(MPICC)
AC_SUBST(MPIFC)

AX_COMPILER_VENDOR
AC_PROG_AWK
AC_PROG_F77
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_LN_S
AC_PROG_MAKE_SET

# Check for CUDA
AX_CHECK_CUDA

# automake 1.12 seems to require AM_PROG_AR, but automake 1.11 doesn't
# recognize it
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])

if test "x${ac_compile_nvcc}" = xyes; then
  CC_save=$CC
  CC=$NVCC
fi

LT_INIT

if test "x${ac_compile_nvcc}" = xyes; then
  CC=$CC_save
fi

# Doxygen (optional; enables `make doxygen-doc`, which drives ./Doxyfile)
AC_CHECK_PROG([DOXYGEN], [doxygen], [doxygen], [no])
AM_CONDITIONAL([HAVE_DOXYGEN], [test "x$DOXYGEN" != xno])

AX_CXX_DEF_TEMPLATE_ARG
CHECK_QUAD_PRECISION

# Check for math libs
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([imf],[cos])

# Check for libstdc++, libdl
AC_CHECK_LIB(stdc++,main)
AC_CHECK_LIB(dl,dlopen)

# Check for OpenMP
AC_LANG_WERROR([on])
if $USE_OPENMP; then
  AC_ARG_WITH(openmp_flag,
	      [AS_HELP_STRING([--with-openmp-flag="FLAGS"],
			      [FLAGS to enable OpenMP])],
	      [OPENMP_FLAG="-$withval"],
	      [OPENMP_FLAG=""])

  # Check if OpenMP flag is already specified.
  CHECK_OPENMP($OPENMP_FLAG)

  # Check for all common OpenMP flags.
  if test "$cv_openmp" = no; then
    case "$ax_cv_cxx_compiler_vendor" in
      gnu)
      CHECK_OPENMP(-fopenmp)
      ;;
      clang)
      CHECK_OPENMP(-fopenmp)
      ;;
      intel)
      CHECK_OPENMP(-openmp)
      CHECK_OPENMP(-qopenmp)
      ;;
      pgi)
      CHECK_OPENMP(-mp)
      ;;
      msdosmsvc)
      ## FIXME is this the right flag for MSVC?
      CHECK_OPENMP(-openmp)
      ;;
      ## Add other compilers supporting OpenMP here
    esac
  fi

  # Error: OpenMP not available.
  if test "$cv_openmp" = no; then
    AC_MSG_ERROR([Don't know how to enable OpenMP.
		  Please specify the OpenMP flag using: --with-openmp-flag=<flag>.]);
  fi
fi
AC_LANG_WERROR([off])

# Check for FFTW
AC_CHECK_FFTW
AC_CHECK_FFTWF
AC_CHECK_FFTWL


CFLAGS_save=$CFLAGS
CXXFLAGS_save=$CXXFLAGS
CFLAGS=""
CXXFLAGS=""
AX_BLAS(acx_blas_ok=yes, AC_MSG_ERROR([Cannot find BLAS library.
                    Please specify the location of the library using: --with-blas=LIB]))
AX_LAPACK(acx_lapack_ok=yes, AC_MSG_ERROR([Cannot find LAPACK library.
                    Please specify the location of the library using: --with-lapack=LIB]))
CFLAGS=$CFLAGS_save
CXXFLAGS=$CXXFLAGS_save
AC_SUBST(LAPACK_LIBS)
AC_SUBST(BLAS_LIBS)
AC_SUBST(FLIBS)


AX_CHECK_SVML
AX_CHECK_LIBMVEC

#######################################################################
# Assemble the SCTL feature macros from what configure actually found.
# These are substituted into MakeVariables (@SCTL_DEFS@) so the SCTL
# layer is told exactly which optional dependencies are available,
# instead of hard-coding the list.
#######################################################################

# Optimization / ISA / language-standard flags (the -O3 overrides autoconf's -O2).
# -mno-avx512fp16 works around pre-2.38 binutils that can't assemble the
# AVX-512-FP16 ops -march=native emits on new Intel CPUs; skipped on macOS where
# clang neither emits them nor recognizes the flag. Set as a shell conditional
# here (not in MakeVariables.in) because automake parses that included fragment.
PVFMM_OPT_FLAGS="-std=c++17 -O3 -march=native"
case "`uname -s`" in
  Darwin) ;;
  *) PVFMM_OPT_FLAGS="$PVFMM_OPT_FLAGS -mno-avx512fp16" ;;
esac
AC_SUBST(PVFMM_OPT_FLAGS)

# Assemble the -D flags. MPI is mandatory for pvfmm (AX_MPI errors out
# above if mpicxx is missing), so SCTL_HAVE_MPI is always on; the rest
# track detection results.
SCTL_DEFS="-DSCTL_HAVE_MPI"
if test "x$cv_quad_prec"        = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_QUAD_T=$cv_quad_type"; fi
if test "x$acx_blas_ok"         = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_BLAS";    fi
if test "x$acx_lapack_ok"       = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_LAPACK";  fi
if test "x$acx_fftw_ok"         = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_FFTW";    fi
if test "x$acx_fftwf_ok"        = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_FFTWF";   fi
if test "x$acx_fftwl_ok"        = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_FFTWL";   fi
LIBMVEC_LIB=""
# SVML and libmvec are mutually exclusive in SCTL; prefer SVML when present.
if test "x$cv_have_svml"        = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_SVML";
elif test "x$cv_have_libmvec"   = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_LIBMVEC"; LIBMVEC_LIB="-lmvec -lm"; fi
AC_SUBST(LIBMVEC_LIB)
AC_SUBST(SCTL_DEFS)

# Path for precomputed data files.
AC_ARG_WITH(precomp-dir,
            [AS_HELP_STRING([--with-precomp-dir=DIR], [set directory for precomputed data files to DIR])],
            [PRECOMP_DIR="$withval"; AX_NORMALIZE_PATH([PRECOMP_DIR], ["/"]) ])
# pvfmm-specific -D flags (no longer via the generated pvfmm_config.h)
PVFMM_DEFS=""
if test "x$PRECOMP_DIR" != x;         then PVFMM_DEFS="$PVFMM_DEFS -DPVFMM_PRECOMP_DATA_PATH='\"$PRECOMP_DIR\"'"; fi
if test "x${ac_compile_nvcc}" = xyes; then PVFMM_DEFS="$PVFMM_DEFS -DPVFMM_HAVE_CUDA"; fi
AC_SUBST(PVFMM_DEFS)

AC_CONFIG_FILES([Makefile
                 MakeVariables])
AC_OUTPUT

echo "
------------------------------------------------------------------------------
$PACKAGE-$VERSION library configuration:
------------------------------------------------------------------------------

External libs:
  - BLAS ........... : ${acx_blas_ok}
  - LAPACK ......... : ${acx_lapack_ok}
  - FFTW (double) .. : ${acx_fftw_ok}
  - FFTW (float) ... : ${acx_fftwf_ok}
  - FFTW (ldouble) . : ${acx_fftwl_ok}
  - SVML ........... : ${cv_have_svml}
  - CUDA ........... : ${ac_compile_nvcc}

SCTL flags:
  - SCTL_DEFS ...... : ${SCTL_DEFS}

Compiler/linker flags/libs/defs:
  - MPICXX ......... : ${MPICXX}
  - CXXFLAGS ....... : ${CXXFLAGS}
  - LDFLAGS ........ : ${LDFLAGS}
  - LIBS ........... : ${LIBS}
    - BLAS ......... : ${BLAS_LIBS}
    - LAPACK ....... : ${LAPACK_LIBS}
    - FFTW ......... : ${FFTW_LIB} ${FFTWF_LIB}

"

