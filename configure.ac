AC_PREREQ([2.61])
AC_INIT([PVFMM],[1.0.0],[contact@pvfmm.org])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall no-define])

AC_CONFIG_SRCDIR([src/pvfmm-wrapper.cpp])
AM_CONFIG_HEADER([config.h])               # prep config.h from config.h.in
AX_PREFIX_CONFIG_H([include/pvfmm_config.h]) # prep include/pvfmm_config.h from it..


# Set language to C++
AC_LANG([C++])


# Checks for programs.
AX_PROG_CXX_MPI(,[CXX=$MPICXX],[])
AX_MPI([CXX=$MPICXX; CC=$MPICC; CFLAGS=$CFLAGS],
       [AC_MSG_ERROR([Cannot find mpicxx. Make sure that MPI is installed and location of compiler is in PATH.])])
AX_COMPILER_VENDOR
AC_PROG_AWK
AC_PROG_F77
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_LN_S
AC_PROG_MAKE_SET
#AC_PROG_RANLIB
##AM_PROG_AR

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
AC_PROG_LIBTOOL

if test "x${ac_compile_nvcc}" = xyes; then
  CC=$CC_save
fi

# Checks for documentation build tools
AC_CHECK_PROG([doxygen_ok], [doxygen], [yes], [no])
AC_CHECK_PROG([pdflatex_ok], [pdflatex], [yes], [no])
AC_CHECK_PROG([latex_ok], [latex], [yes], [no])
AC_CHECK_PROG([dvips_ok], [dvips], [yes], [no])
AC_CHECK_PROG([gs_ok], [gs], [yes], [no])
if test "x$enable_html_doc" != xno; then
  test "x$doxygen_ok" != xyes && enable_html_doc=no
  test "x$latex_ok" != xyes && enable_html_doc=no
  test "x$dvips_ok" != xyes && enable_html_doc=no
  test "x$gs_ok" != xyes && enable_html_doc=no
  AC_PATH_PROG([PERL], [perl], [])
fi
AM_CONDITIONAL([LATEX_OK], [test "x$latex_ok" = xyes])
AM_CONDITIONAL([PDFLATEX_OK], [test "x$pdflatex_ok" = xyes])
AM_CONDITIONAL([HTML_DOC], [test "x$enable_html_doc" = xyes])
AM_CONDITIONAL([DOXYGEN_OK], [test "x$doxygen_ok" = xyes])

DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(OFF)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_INIT_DOXYGEN($PACKAGE_NAME, Doxyfile, doc/doxygen)


AX_CXX_DEF_TEMPLATE_ARG
CHECK_QUAD_PRECISION

# Check for math libs
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([imf],[cos])

# Check for X11
AC_PATH_X
if test "$no_x" = yes; then
  AC_MSG_WARN([Cannot find X Window System include files and libraries.
               Please specify their location using --x-includes=dir and --x-libraries=dir])
else
  if test "$x_includes" != ""; then
    X_INCLUDES="-I$x_includes"
  fi
  if test "$x_libraries" != ""; then
    X_LIBS="-L$x_libraries"
  fi
  X_LIBS="$X_LIBS -lX11"
  AC_SUBST(X_INCLUDES)
  AC_SUBST(X_LIBS)
fi

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

# Check for Intel Xeon Phi
#CHECK_INTEL_OFFLOAD
DISABLE_INTEL_OFFLOAD(-no-offload)
DISABLE_INTEL_OFFLOAD(-qno-offload)

# Check for FFTW
AC_CHECK_FFTW
AC_CHECK_FFTWF
AC_CHECK_FFTWL


#Check for BLAS LAPACK
#AM_CONDITIONAL(BLAS_EQ_MKL, test "$BLAS" = "MKL")
#AM_CONDITIONAL(BLAS_EQ_GSL, test "$BLAS" = "GSL")
#if (test -r $PETSC_DIR/$PETSC_ARCH/conf/petscvariables) ; then
#  PETSC_BLASLAPACK_LIB=`grep "BLASLAPACK_LIB" $PETSC_DIR/$PETSC_ARCH/conf/petscvariables | sed -e 's/.*BLASLAPACK_LIB = //'`
#  if test "x$PETSC_BLASLAPACK_LIB" != x ; then
#    if test "x$with_blas" != "x" -a "$with_blas" != "$PETSC_BLASLAPACK_LIB" ; then
#      AC_MSG_WARN([BLAS flags specified by --with-blas appear to disagree with PETSc BLAS/LAPACK flags; using PETSc settings])
#    fi
#    if test "x$with_lapack" != "x" -a "$with_lapack" != "$PETSC_BLASLAPACK_LIB" ; then
#      AC_MSG_WARN([LAPACK flags specified by --with-lapack appear to disagree with PETSc BLAS/LAPACK flags; using PETSc settings])
#    fi
#    echo "using PETSc BLAS/LAPACK library flags: $PETSC_BLASLAPACK_LIB"
#    with_blas="$PETSC_BLASLAPACK_LIB"
#    with_lapack=""
#  fi
#  PETSC_BLASLAPACK_INCLUDE=`grep "BLASLAPACK_INCLUDE" $PETSC_DIR/$PETSC_ARCH/conf/petscvariables | sed -e 's/.*BLASLAPACK_INCLUDE = //'`
#  if test "x$PETSC_BLASLAPACK_INCLUDE" != x ; then
#    CPPFLAGS="${PETSC_BLASLAPACK_INCLUDE} $CPPFLAGS"
#  fi
#fi
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


# Checks for header files.
AC_CHECK_HEADERS([stdint.h cstdlib cstddef])


# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_HEADER_STAT
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_UINT16_T
AC_TYPE_UINT8_T
AC_TYPE_SIZE_T


# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([floor memset pow sqrt strtol strtoul])
AX_CHECK_SVML

#######################################################################
# Assemble the SCTL feature macros from what configure actually found.
# These are substituted into MakeVariables (@SCTL_DEFS@) so the SCTL
# layer is told exactly which optional dependencies are available,
# instead of hard-coding the list.
#######################################################################

# Optional: threaded FFTW plans (off by default; pvfmm drives one FFT
# per OpenMP thread, so threaded FFTW would only oversubscribe). Enable
# with --enable-fftw-threads, which also links the threads library.
FFTW_THREADS_LIB=""
acx_fftw_threads_ok=no
AC_ARG_ENABLE([fftw-threads],
              [AS_HELP_STRING([--enable-fftw-threads],
                              [use threaded FFTW plans in SCTL (default: no)])],
              [enable_fftw_threads=$enableval],
              [enable_fftw_threads=no])
if test "x$enable_fftw_threads" = xyes -a "x$acx_fftw_ok" = xyes; then
  save_LIBS="$LIBS"
  LIBS="$FFTW_LIB $LIBS"
  AC_MSG_CHECKING([for fftw_init_threads])
  AC_TRY_LINK_FUNC(fftw_init_threads,
                   [acx_fftw_threads_ok=yes],
                   [acx_fftw_threads_ok=no])
  AC_MSG_RESULT($acx_fftw_threads_ok)
  if test "x$acx_fftw_threads_ok" = xno; then
    AC_CHECK_LIB([fftw3_threads],fftw_init_threads,
                 [acx_fftw_threads_ok=yes; FFTW_THREADS_LIB="-lfftw3_threads"],
                 [AC_CHECK_LIB([fftw3_omp],fftw_init_threads,
                               [acx_fftw_threads_ok=yes; FFTW_THREADS_LIB="-lfftw3_omp"],
                               [acx_fftw_threads_ok=no],
                               [$FFTW_LIB])],
                 [$FFTW_LIB])
  fi
  LIBS="$save_LIBS"
  if test "x$acx_fftw_threads_ok" = xno; then
    AC_MSG_WARN([--enable-fftw-threads requested but no threaded FFTW found; disabling.])
  fi
fi
AC_SUBST(FFTW_THREADS_LIB)

# Profiling verbosity level for sctl::Profile (overridable at make time via
# `make SCTL_PROFILE=<n>`; 0 disables profiling overhead).
AC_ARG_WITH([sctl-profile],
            [AS_HELP_STRING([--with-sctl-profile=N],
                            [default SCTL_PROFILE level (default: 5000)])],
            [SCTL_PROFILE_DEFAULT="$withval"],
            [SCTL_PROFILE_DEFAULT=5000])
AC_SUBST(SCTL_PROFILE_DEFAULT)

# Assemble the -D flags. MPI is mandatory for pvfmm (AX_MPI errors out
# above if mpicxx is missing), so SCTL_HAVE_MPI is always on; the rest
# track detection results.
SCTL_DEFS="-DSCTL_HAVE_MPI"
if test "x$acx_blas_ok"         = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_BLAS";    fi
if test "x$acx_lapack_ok"       = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_LAPACK";  fi
if test "x$acx_fftw_ok"         = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_FFTW";    fi
if test "x$acx_fftwf_ok"        = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_FFTWF";   fi
if test "x$acx_fftwl_ok"        = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_FFTWL";   fi
if test "x$acx_fftw_threads_ok" = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_FFTW_THREADS"; fi
if test "x$cv_have_svml"        = xyes; then SCTL_DEFS="$SCTL_DEFS -DSCTL_HAVE_SVML";    fi
AC_SUBST(SCTL_DEFS)

# Path for precomputed data files.
AC_ARG_WITH(precomp-dir,
            [AS_HELP_STRING([--with-precomp-dir=DIR], [set directory for precomputed data files to DIR])],
            [PRECOMP_DIR="$withval"; AX_NORMALIZE_PATH([PRECOMP_DIR], ["/"]) ])
AC_DEFINE_UNQUOTED(PRECOMP_DATA_PATH,"$PRECOMP_DIR",[Path for precomputed data files.])

#AC_CONFIG_FILES([Makefile])
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
  - FFTW threads ... : ${acx_fftw_threads_ok}
  - SVML ........... : ${cv_have_svml}
  - CUDA ........... : ${ac_compile_nvcc}

SCTL flags:
  - SCTL_DEFS ...... : ${SCTL_DEFS}
  - SCTL_PROFILE ... : ${SCTL_PROFILE_DEFAULT}

Compiler/linker flags/libs/defs:
  - MPICXX ......... : ${MPICXX}
  - CXXFLAGS ....... : ${CXXFLAGS}
  - LDFLAGS ........ : ${LDFLAGS}
  - LIBS ........... : ${LIBS}
    - BLAS ......... : ${BLAS_LIBS}
    - LAPACK ....... : ${LAPACK_LIBS}
    - FFTW ......... : ${FFTW_LIB} ${FFTWF_LIB}
    - X11 .......... : ${X_LIBS}

Documentation tools:
  - doxygen ........ : ${doxygen_ok}
  - latex .......... : ${latex_ok}
  - dvips .......... : ${dvips_ok}
  - ghostscript .... : ${gs_ok}

"

