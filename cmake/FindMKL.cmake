################################################################################
#
# \file      cmake/FindMKL.cmake
# \author    J. Bakosi
# \copyright 2012-2015, Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
# \brief     Find the Math Kernel Library from Intel
# \date      Thu 26 Jan 2017 02:05:50 PM MST
#
################################################################################

# Find the Math Kernel Library from Intel
#
#  MKL_FOUND - System has MKL
#  MKL_INCLUDE_DIRS - MKL include files directories
#  MKL_LIBRARIES - The MKL libraries
#  MKL_INTERFACE_LIBRARY - MKL interface library
#  MKL_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  MKL_CORE_LIBRARY - MKL core library
#
#  The environment variables MKLROOT and INTEL are used to find the library.
#  Everything else is ignored. If MKL is found "-DMKL_ILP64" is added to
#  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS.
#
#  Example usage:
#
#  find_package(MKL)
#  if(MKL_FOUND)
#    target_link_libraries(TARGET ${MKL_LIBRARIES})
#  endif()

# If already in cache, be silent
if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_SDL_LIBRARY)
  set (MKL_FIND_QUIETLY TRUE)
endif()

find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS $ENV{MKLROOT}/include)
find_path(MKL_FFTW_INCLUDE_DIR NAMES fftw3_mkl.h HINTS
  $ENV{MKLROOT}/include/fftw
  $ENV{MKLROOT}/include)

find_library(MKL_SDL_LIBRARY
             NAMES mkl_rt
             PATHS $ENV{MKLROOT}/lib
                   $ENV{MKLROOT}/lib/intel64
                   $ENV{INTEL}/mkl/lib/intel64
             NO_DEFAULT_PATH)


set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR} ${MKL_FFTW_INCLUDE_DIR})
set(MKL_LIBRARIES ${MKL_SDL_LIBRARY})

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS MKL_SDL_LIBRARY)

# MARK_AS_ADVANCED(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_INTERFACE_LIBRARY MKL_SEQUENTIAL_LAYER_LIBRARY MKL_CORE_LIBRARY)

# add_library(mkl_rt INTERFACE)
# target_link_libraries(mkl_rt INTERFACE ${MKL_LIBRARIES})
# target_include_directories(mkl_rt INTERFACE ${MKL_INCLUDE_DIRS})
