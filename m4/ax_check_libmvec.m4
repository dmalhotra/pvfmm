
AC_DEFUN([AX_CHECK_LIBMVEC],
    ## Check for GNU libmvec (vectorized libm, glibc >= 2.22 on x86-64).
    ## Sets shell var cv_have_libmvec=yes/no by link-testing a libmvec symbol
    ## against "-lmvec -lm".

    [AC_MSG_CHECKING([for libmvec])

    cv_have_libmvec=no
    ax_libmvec_save_LIBS="$LIBS"
    LIBS="-lmvec -lm $LIBS"
    AC_TRY_LINK_FUNC(_ZGVbN2v_sin, [cv_have_libmvec=yes], [])
    LIBS="$ax_libmvec_save_LIBS"

    AC_MSG_RESULT($cv_have_libmvec)
])
