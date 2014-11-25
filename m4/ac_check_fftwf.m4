
AC_DEFUN([AC_CHECK_FFTWF],[\
    save_CXXFLAGS="$CXXFLAGS";
    save_LIBS="$LIBS"

    ###########
    ## Read command line parameters for FFTW
    ###########

    AC_ARG_WITH(fftw,
                [AS_HELP_STRING([--with-fftw=DIR],
                                [set FFTW installation directory to DIR])],
                [FFTW_DIR="$withval"; FFTW_INCLUDE="-I$FFTW_DIR/include"; FFTWF_LIB="-L$FFTW_DIR/lib"])

  if test "x$FFTW_DIR" != xno; then

    AC_ARG_WITH(fftw_include,
                [AS_HELP_STRING([--with-fftw-include=DIR],
                                [set fftw3.h directory path to DIR])],
                [FFTW_INCLUDE="-I$withval"])

    AC_ARG_WITH(fftw_lib,
                [AS_HELP_STRING([--with-fftw-lib=LIB],
                                [set FFTW library to LIB])],
                [FFTWF_LIB="$withval"])

    ###########
    ## Check for fftw3.h
    ###########

    CXXFLAGS="$FFTW_INCLUDE $CXXFLAGS"
    AC_MSG_CHECKING([for fftw3.h])
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include<fftw3.h>]],[[;]])],\
                      [cv_fftw3_h=yes],\
                      [cv_fftw3_h=no])
    AC_MSG_RESULT($cv_fftw3_h)

    if test "$cv_fftw3_h" = no; then
        AC_MSG_ERROR([Cannot find fftw3.h (with CXXFLAGS=$CXXFLAGS)
                  Please specify the location of fftw3.h using: --with-fftw-include=DIR 
                  or specify the FFTW installation directory using --with-fftw=DIR])
    fi

    ###########
    ## Check for library
    ###########

    LIBS="$FFTWF_LIB $LIBS"
    AC_MSG_CHECKING([for fftwf_plan_dft_1d])
    AC_TRY_LINK_FUNC(fftwf_plan_dft_1d,\
                     [cv_lfftw3f=yes;],\
                     [cv_lfftw3f=no;])
    AC_MSG_RESULT($cv_lfftw3f)

    if test "x$cv_lfftw3f" = xno; then
        AC_CHECK_LIB([fftw3f],fftwf_plan_dft_1d,\
                     [cv_lfftw3f=yes; FFTWF_LIB="$FFTWF_LIB -lfftw3f"],\
                     [cv_lfftw3f=no])
    fi

    if test "x$cv_lfftw3f" = xno; then
        AC_CHECK_LIB([fftw3f],fftwf_plan_dft_1d,\
                     [cv_lfftw3f=yes; FFTWF_LIB="$FFTWF_LIB -lfftw3f -lm"],\
                     [cv_lfftw3f=no],\
                     [-lm])
    fi

    if test "x$cv_lfftw3f" = xno; then
        AC_CHECK_LIB([mkl_intel_lp64],fftwf_plan_dft_1d,\
                     [cv_lfftw3f=yes; FFTWF_LIB="$FFTWF_LIB -lmkl_intel_thread -lmkl_core -liomp5 -lpthread"],\
                     [cv_lfftw3f=no],\
                     [-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread])
    fi

    if test "$cv_lfftw3f" = yes; then
        AC_SUBST(FFTWF_LIB)
        AC_SUBST(FFTW_INCLUDE)
        AC_DEFINE(HAVE_FFTWF,1,[Define if we have FFTW])
        acx_fftwf_ok=yes
        #AC_SUBST(acx_fftwf_ok)
    else
        acx_fftwf_ok=no
        #AC_SUBST(acx_fftwf_ok)
        AC_MSG_WARN([Cannot find single precision FFTW library (with LIBS=$LIBS)
                    Please specify the location of the library using: --with-fftw-lib=LIB 
                    or specify the FFTW installation directory using --with-fftw=DIR])
    fi

  else

    FFTW_INCLUDE="";
    FFTWF_LIB="";

  fi

    LIBS="$save_LIBS"
    CXXFLAGS="$save_CXXFLAGS"
])

