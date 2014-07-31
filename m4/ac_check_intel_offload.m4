
AC_DEFUN([CHECK_INTEL_OFFLOAD], [
    ## Check for support of offload pragma and -no-offload flag. If
    ## found define HAVE_INTEL_OFFLOAD_PRAGMA, HAVE_INTEL_NOFFLOAD_FLAG and
    ## HAVE_INTEL_OFFLOAD.

    XCFLAGS="$CFLAGS"
    XCXXFLAGS="$CXXFLAGS"
    AC_LANG_WERROR([on])
    intel_offload_ok=no

    # check for offload pragma
    AC_MSG_CHECKING([for support of offload pragma])
    AC_LINK_IFELSE([AC_LANG_PROGRAM([[
        ]], [[
            #pragma offload target(mic:0)
        ]])],intel_offload_pragma_ok=yes, intel_offload_pragma_ok=no)
    AC_MSG_RESULT($intel_offload_pragma_ok)

    # check for -no-offload flag
    AC_MSG_CHECKING([for support of -no-offload flag])
    CFLAGS="$CFLAGS -no-offload"
    CXXFLAGS="$CXXFLAGS -no-offload"
    AC_LANG_WERROR([on])
    AC_LINK_IFELSE([AC_LANG_PROGRAM([[
        ]], [[
            #ifdef __INTEL_OFFLOAD
            #pragma offload target(mic:0)
            #endif
        ]])],intel_noffload_flag_ok=yes, intel_noffload_flag_ok=no)
    AC_MSG_RESULT($intel_noffload_flag_ok)

    # Substitute original values.
    AC_LANG_WERROR([off])
    CFLAGS="$XCFLAGS"
    CXXFLAGS="$XCXXFLAGS"
    ARFLAGS="$AR_FLAGS"

    if test x"$intel_offload_pragma_ok" = xyes; then
        AC_DEFINE(HAVE_INTEL_OFFLOAD_PRAGMA,1,[Define if you have INTEL_OFFLOAD_PRAGMA.])
    fi
    if test x"$intel_noffload_flag_ok" = xyes; then
        AC_DEFINE(HAVE_INTEL_NOFFLOAD_FLAG,1,[Define if you have INTEL_OFFLOAD.])
        if test x"$intel_offload_pragma_ok" = xyes; then
            AC_DEFINE(HAVE_INTEL_OFFLOAD,1,[Define if you have INTEL_OFFLOAD.])
            intel_offload_ok=yes

            AR="xiar"
            ARFLAGS="cru -qoffload-build"
        else
            CFLAGS="$CFLAGS -no-offload"
            CXXFLAGS="$CXXFLAGS -no-offload"
        fi
    fi
    AC_SUBST(intel_offload_pragma_ok)
    AC_SUBST(intel_noffload_flag_ok)
    AC_SUBST(intel_offload_ok)
    AC_SUBST(ARFLAGS)
    AC_SUBST(AR)
])

