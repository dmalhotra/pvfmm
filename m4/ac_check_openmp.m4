
AC_DEFUN([CHECK_OPENMP],
    ## Check for support of OpenMP with a given compiler flag. If
    ## found define HAVE_OPENMP and add the compile flag to CFLAGS
    ## and CXXFLAGS.

    [AC_MSG_CHECKING([for support of OpenMP (with $1)])

    XCFLAGS="$CFLAGS"
    XCXXFLAGS="$CXXFLAGS"
    CFLAGS="$CFLAGS $1"
    CXXFLAGS="$CXXFLAGS $1"

    AC_LINK_IFELSE([AC_LANG_PROGRAM([[
        #include <omp.h>
        #include <stdio.h>
        ]], [[
            #pragma omp parallel
            {
              int i=omp_get_num_threads();
              printf("Hello, world %i.\n",i);
            }
        ]])],cv_openmp=yes, cv_openmp=no)

    AC_MSG_RESULT($cv_openmp)

    if test "$cv_openmp" = yes; then
        AC_DEFINE(HAVE_OPENMP,1,[Define if compiler supports OpenMP])
    else
        CFLAGS="$XCFLAGS"
        CXXFLAGS="$XCXXFLAGS"
    fi
])

