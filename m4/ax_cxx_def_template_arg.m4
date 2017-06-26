# SYNOPSIS
#
#   AX_CXX_DEF_TEMPLATE_ARG
#
# DESCRIPTION
#
#  Check if the compiler supports default template arguments for function
#  templates and add necessary flags to CXXFLAGS.
#  

m4_define([_AX_CXX_DEF_TEMPLATE_ARG_testbody], [[
  template <int T=0>
  int test(){ return T;};
  int test_default(){return test();};
]])

AC_DEFUN([AX_CXX_DEF_TEMPLATE_ARG], [dnl

  AC_LANG_PUSH([C++])dnl
  AC_LANG_WERROR([on])
  ac_success=no
  AC_CACHE_CHECK(whether $CXX supports default template arguments by default,
  ax_cv_cxx_compile_template,
  [AC_COMPILE_IFELSE([AC_LANG_SOURCE([_AX_CXX_DEF_TEMPLATE_ARG_testbody])],
    [ax_cv_cxx_compile_template=yes],
    [ax_cv_cxx_compile_template=no])])
  if test x$ax_cv_cxx_compile_template = xyes; then
    ac_success=yes
  fi

  if test x$ac_success = xno; then
    for switch in -std=gnu++11 -std=gnu++0x -std=c++11 -std=c++0x; do
      cachevar=AS_TR_SH([ax_cv_cxx_compile_template_$switch])
      AC_CACHE_CHECK(whether $CXX default template arguments with $switch,
                     $cachevar,
        [ac_save_CXXFLAGS="$CXXFLAGS"
         CXXFLAGS="$CXXFLAGS $switch"
         AC_COMPILE_IFELSE([AC_LANG_SOURCE([_AX_CXX_DEF_TEMPLATE_ARG_testbody])],
          [eval $cachevar=yes],
          [eval $cachevar=no])
         CXXFLAGS="$ac_save_CXXFLAGS"])
      if eval test x\$$cachevar = xyes; then
        CXXFLAGS="$CXXFLAGS $switch"
        ac_success=yes
        break
      fi
    done
  fi

  AC_LANG_WERROR([on])
  AC_LANG_POP([C++])
  if test x$ac_success = xno; then
    AC_MSG_ERROR([*** Compiler does not support default template arguments in function templates.
Please use a different compiler or specify the necessary CXXFLAGS.])
  fi
])
