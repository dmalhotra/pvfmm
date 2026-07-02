
AC_DEFUN([CHECK_QUAD_PRECISION],
    ## Check for quadruple precision support. If found define 
    ## HAVE_QUAD_PRECISION and add compiler flag to CFLAGS and 
    ## CXXFLAGS.

    [AC_MSG_CHECKING([for quadruple precision support])

    XCFLAGS="$CFLAGS"
    XCXXFLAGS="$CXXFLAGS"
    XLIBS="$LIBS"

    cv_quad_prec=no
    if test "$cv_quad_prec" = no; then
      cv_quad_type=_Quad
      CFLAGS="$XCFLAGS -Qoption,cpp,--extended_float_type"
      CXXFLAGS="$XCXXFLAGS -Qoption,cpp,--extended_float_type"
      AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <stdio.h>]], [[printf("%ld",sizeof($cv_quad_type));]])],cv_quad_prec=yes, [])
    fi

    if test "$cv_quad_prec" = no; then
      cv_quad_type=__float128
      CFLAGS="$XCFLAGS"
      CXXFLAGS="$XCXXFLAGS"
      AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <stdio.h>]], [[printf("%ld",sizeof($cv_quad_type));]])],cv_quad_prec=yes, [])
    fi

    if test "$cv_quad_prec" = yes; then
        AC_MSG_RESULT($cv_quad_type)
    else
        AC_MSG_RESULT($cv_quad_prec)
        CFLAGS="$XCFLAGS"
        CXXFLAGS="$XCXXFLAGS"
        LIBS="$XLIBS"
    fi
])

