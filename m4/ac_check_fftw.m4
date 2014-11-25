
AC_DEFUN([AC_CHECK_FFTW],[\
    save_CXXFLAGS="$CXXFLAGS";
    save_LIBS="$LIBS"

    ###########
    ## Read command line parameters for FFTW
    ###########

    AC_ARG_WITH(fftw,
                [AS_HELP_STRING([--with-fftw=DIR],
                                [set FFTW installation directory to DIR])],
                [FFTW_DIR="$withval"; FFTW_INCLUDE="-I$FFTW_DIR/include"; FFTW_LIB="-L$FFTW_DIR/lib"])

  if test "x$FFTW_DIR" != xno; then

    AC_ARG_WITH(fftw_include,
                [AS_HELP_STRING([--with-fftw-include=DIR],
                                [set fftw3.h directory path to DIR])],
                [FFTW_INCLUDE="-I$withval"])

    AC_ARG_WITH(fftw_lib,
                [AS_HELP_STRING([--with-fftw-lib=LIB],
                                [set FFTW library to LIB])],
                [FFTW_LIB="$withval"])

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

    LIBS="$FFTW_LIB $LIBS"
    AC_MSG_CHECKING([for fftw_plan_dft_1d])
    AC_TRY_LINK_FUNC(fftw_plan_dft_1d,\
                     [cv_lfftw3=yes;],\
                     [cv_lfftw3=no;])
    AC_MSG_RESULT($cv_lfftw3)

    if test "x$cv_lfftw3" = xno; then
        AC_CHECK_LIB([fftw3],fftw_plan_dft_1d,\
                     [cv_lfftw3=yes; FFTW_LIB="$FFTW_LIB -lfftw3"],\
                     [cv_lfftw3=no])
    fi

    if test "x$cv_lfftw3" = xno; then
        AC_CHECK_LIB([fftw3],fftw_plan_dft_1d,\
                     [cv_lfftw3=yes; FFTW_LIB="$FFTW_LIB -lfftw3 -lm"],\
                     [cv_lfftw3=no],\
                     [-lm])
    fi

    if test "x$cv_lfftw3" = xno; then
        AC_CHECK_LIB([mkl_intel_lp64],fftw_plan_dft_1d,\
                     [cv_lfftw3=yes; FFTW_LIB="$FFTW_LIB -lmkl_intel_thread -lmkl_core -liomp5 -lpthread"],\
                     [cv_lfftw3=no],\
                     [-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread])
    fi

    if test "$cv_lfftw3" = yes; then
        AC_SUBST(FFTW_LIB)
        AC_SUBST(FFTW_INCLUDE)
        AC_DEFINE(HAVE_FFTW,1,[Define if we have FFTW])
        acx_fftw_ok=yes
        #AC_SUBST(acx_fftw_ok)
    else
        acx_fftw_ok=no
        #AC_SUBST(acx_fftw_ok)
        AC_MSG_ERROR([Cannot find FFTW library (with LIBS=$LIBS)
                    Please specify the location of the library using: --with-fftw-lib=LIB 
                    or specify the FFTW installation directory using --with-fftw=DIR])
    fi

  else

    AC_MSG_RESULT(disabling FFTW library)
    FFTW_INCLUDE="";
    FFTW_LIB="";

  fi

    LIBS="$save_LIBS"
    CXXFLAGS="$save_CXXFLAGS"
])

