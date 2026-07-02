
AC_DEFUN([AC_CHECK_FFTWL],[\
    save_CXXFLAGS="$CXXFLAGS";
    save_LIBS="$LIBS"

    ###########
    ## Read command line parameters for FFTW
    ###########

    AC_ARG_WITH(fftw,
                [AS_HELP_STRING([--with-fftw=DIR],
                                [set FFTW installation directory to DIR])],
                [FFTW_DIR="$withval"; FFTW_INCLUDE="-I$FFTW_DIR/include"; FFTWL_LIB="-L$FFTW_DIR/lib"])

  if test "x$FFTW_DIR" != xno; then

    AC_ARG_WITH(fftw_include,
                [AS_HELP_STRING([--with-fftw-include=DIR],
                                [set fftw3.h directory path to DIR])],
                [FFTW_INCLUDE="-I$withval"])

    AC_ARG_WITH(fftw_lib,
                [AS_HELP_STRING([--with-fftw-lib=LIB],
                                [set FFTW library to LIB])],
                [FFTWL_LIB="$withval"])

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

    LIBS="$FFTWL_LIB $LIBS"
    AC_MSG_CHECKING([for fftwl_plan_dft_1d])
    AC_TRY_LINK_FUNC(fftwl_plan_dft_1d,\
                     [cv_lfftw3l=yes;],\
                     [cv_lfftw3l=no;])
    AC_MSG_RESULT($cv_lfftw3l)

    if test "x$cv_lfftw3l" = xno; then
        AC_CHECK_LIB([fftw3l],fftwl_plan_dft_1d,\
                     [cv_lfftw3l=yes; FFTWL_LIB="$FFTWL_LIB -lfftw3l"],\
                     [cv_lfftw3l=no])
    fi

    if test "x$cv_lfftw3l" = xno; then
        AC_CHECK_LIB([fftw3l],fftwl_plan_dft_1d,\
                     [cv_lfftw3l=yes; FFTWL_LIB="$FFTWL_LIB -lfftw3l -lm"],\
                     [cv_lfftw3l=no],\
                     [-lm])
    fi

    if test "$cv_lfftw3l" = yes; then
        AC_SUBST(FFTWL_LIB)
        AC_SUBST(FFTW_INCLUDE)
        acx_fftwl_ok=yes
    else
        acx_fftwl_ok=no
        FFTWL_LIB="";
        AC_MSG_WARN([Cannot find long double FFTW library (with LIBS=$LIBS)
                    Please specify the location of the library using: --with-fftw-lib=LIB
                    or specify the FFTW installation directory using --with-fftw=DIR])
    fi

  else

    FFTW_INCLUDE="";
    FFTWL_LIB="";

  fi

    LIBS="$save_LIBS"
    CXXFLAGS="$save_CXXFLAGS"
])
