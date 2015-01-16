
AC_DEFUN([AX_CHECK_SVML],
    ## Check for Intel Short Vector Math Library support. If found define 
    ## HAVE_INTEL_SVML.

    [AC_MSG_CHECKING([for Intel SVML])

    cv_have_svml=no
    #AC_LINK_IFELSE([AC_LANG_PROGRAM([[]], [[_mm256_sin_ps(0);]])],[cv_have_svml=yes], [])
    AC_TRY_LINK_FUNC(_mm256_sin_ps, [cv_have_svml=yes], [])

    if test "$cv_have_svml" = yes; then
        AC_MSG_RESULT($cv_have_svml)
        AC_DEFINE(HAVE_INTEL_SVML,1,[Define if SVL library is available])
    fi
])

