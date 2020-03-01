cmake_minimum_required(VERSION 3.10)
project(
    PVFMM
  VERSION 1.1
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# compiler flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using intel
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qno-offload")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

set(MPI_CXX_SKIP_MPICXX
    true
    CACHE BOOL
          "If true, the MPI-2 C++ bindings are disabled using definitions.")
# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# library
find_package(MKL)
if(NOT MKL_FOUND)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(FFTW REQUIRED)
endif()

find_file(SYS_STAT sys/stat.h OPTIONAL)

# generate pvfmm_config.h
if(MKL_FOUND)
    option(PVFMM_FFTW3_MKL "PVFMM_FFTW3_MKL" ON)
    option(PVFMM_HAVE_FFTW "PVFMM_HAVE_FFTW" ON)
    option(PVFMM_HAVE_FFTWF "PVFMM_HAVE_FFTWF" ON)
endif()

if(FFTW_DOUBLE_LIB_FOUND)
    option(PVFMM_HAVE_FFTW "PVFMM_HAVE_FFTW" ON)
endif()

if(FFTW_FLOAT_LIB_FOUND)
    option(PVFMM_HAVE_FFTWF "PVFMM_HAVE_FFTWF" ON)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    option(PVFMM_HAVE_INTEL_SVML "PVFMM_HAVE_INTEL_SVML" ON)
endif()

if(SYS_STAT)
    option(PVFMM_HAVE_SYS_STAT_H "PVFMM_HAVE_SYS_STAT_H" ON)
endif()

configure_file(pvfmm_config.h.in pvfmm_config.h @ONLY)

enable_testing()

# add_subdirectory(include)
# add_subdirectory(src)
# add_subdirectory(examples)
# add_subdirectory(Test)

# part 1, core library
file(GLOB pvfmm_SRC "src/*.cpp")

# shared lib
add_library( pvfmm SHARED ${pvfmm_SRC} )
set_target_properties(pvfmm PROPERTIES OUTPUT_NAME pvfmm)
target_include_directories(
  pvfmm
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${MKL_INCLUDE_DIRS}
         ${MPI_FFTW_INCLUDE_DIR}
         ${CMAKE_BINARY_DIR}
         )
target_compile_options(pvfmm PUBLIC ${OpenMP_CXX_FLAGS})

# static lib
add_library( pvfmmStatic STATIC ${pvfmm_SRC} )
set_target_properties(pvfmmStatic PROPERTIES OUTPUT_NAME pvfmm)
target_include_directories(
  pvfmmStatic
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${MKL_INCLUDE_DIRS}
         ${MPI_FFTW_INCLUDE_DIR}
         ${CMAKE_BINARY_DIR}
         )
target_compile_options(pvfmmStatic PUBLIC ${OpenMP_CXX_FLAGS})

# install core library and headers
include(GNUInstallDirs)
install(
  TARGETS pvfmm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(
  TARGETS pvfmmStatic
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pvfmm/)
install(FILES ${CMAKE_BINARY_DIR}/pvfmm_config.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pvfmm/)
