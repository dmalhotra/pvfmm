# AC_LANG_WERROR([VALUE])
# -----------------------
# How to treat warnings from the current language's preprocessor, compiler,
# and linker:
# 1. No arguments: treat warnings as fatal errors.
# 2. One argument: on -- treat, off -- do not treat warnings as fatal errors.
AC_DEFUN([AC_LANG_WERROR],
[m4_if(
    $#, 1, [m4_if(
        [$1], [on], [ac_[]_AC_LANG_ABBREV[]_werror_flag=yes],
        [$1], [off], [ac_[]_AC_LANG_ABBREV[]_werror_flag=],
        [m4_fatal([$0: wrong argument: `$1'])])],
    $#, 0, [ac_[]_AC_LANG_ABBREV[]_werror_flag=yes],
    [m4_fatal([$0: incorrect number of arguments: $#])])
])# AC_LANG_WERROR 
