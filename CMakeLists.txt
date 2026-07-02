cmake_minimum_required(VERSION 3.10)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" PVFMM_VERSION)
string(STRIP "${PVFMM_VERSION}" PVFMM_VERSION)
project(
  PVFMM
  VERSION ${PVFMM_VERSION}
  LANGUAGES CXX C)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
include(GNUInstallDirs)

# Optimization / ISA (C++17 comes from CMAKE_CXX_STANDARD above). -mno-avx512fp16
# works around pre-2.38 binutils that can't assemble the AVX-512-FP16 ops
# -march=native emits on new Intel CPUs (GCC/non-Apple Clang only).
add_compile_options(-O3 -march=native)
if(NOT APPLE AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  add_compile_options(-mno-avx512fp16)
endif()

# SCTL always-on flags (FFTW/FFTWF and SVML/libmvec are added conditionally below)
add_compile_options(-DSCTL_HAVE_PVFMM -DSCTL_QUAD_T=__float128 -DSCTL_HAVE_MPI -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK -DSCTL_HAVE_FFTWL)

set(MPI_CXX_SKIP_MPICXX
    true
    CACHE BOOL
          "If true, the MPI-2 C++ bindings are disabled using definitions.")

# required compiler features
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# library
find_package(MKL)
if(MKL_FOUND)
  set(DEP_INC ${MKL_FFTW_INCLUDE_DIR} ${MKL_INCLUDE_DIR})
  set(DEP_LIB ${MKL_SDL_LIBRARY})
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(FFTW REQUIRED)
  set(DEP_INC ${FFTW_INCLUDE_DIRS})
  set(DEP_LIB ${FFTW_FLOAT_LIB}
              ${FFTW_DOUBLE_LIB}
              ${BLAS_LIBRARIES}
              ${LAPACK_LIBRARIES})
endif()

# Periodic boundary conditions (pvfmm-specific; no SCTL equivalent)
option(PVFMM_EXTENDED_BC "Enable extended per-axis periodic boundary conditions" ON)
if(PVFMM_EXTENDED_BC)
  list(APPEND PVFMM_DEFINES "PVFMM_EXTENDED_BC")
endif()

# FFTW availability -> SCTL feature macros
if(MKL_FOUND OR FFTW_DOUBLE_LIB_FOUND)
  list(APPEND PVFMM_DEFINES "SCTL_HAVE_FFTW")
endif()
if(MKL_FOUND OR FFTW_FLOAT_LIB_FOUND)
  list(APPEND PVFMM_DEFINES "SCTL_HAVE_FFTWF")
endif()

# Vectorized libm: Intel SVML, otherwise GNU libmvec (mutually exclusive in SCTL)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  list(APPEND PVFMM_DEFINES "SCTL_HAVE_SVML")
else()
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LIBRARIES "-lmvec" "-lm")
  check_cxx_source_compiles([=[
extern "C" double _ZGVbN2v_sin(double);
int main(){ return (int)_ZGVbN2v_sin(0.0); }
]=] PVFMM_LIBMVEC_FOUND)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(PVFMM_LIBMVEC_FOUND)
    list(APPEND PVFMM_DEFINES "SCTL_HAVE_LIBMVEC")
    list(APPEND DEP_LIB "-lmvec" "-lm")
  endif()
endif()

option(PVFMM_ENABLE_CUDA "Build the CUDA/GPU path" OFF)
if (PVFMM_ENABLE_CUDA)
  enable_language(CUDA)
  file(GLOB pvfmm_SRC_CUDA "src/*.cu")
  set(DEP_LIB_CUDA "${CUDA_CUBLAS_LIBRARIES}")
  list(APPEND PVFMM_DEFINES "PVFMM_HAVE_CUDA")
endif()

# part 1, core library
file(GLOB pvfmm_SRC "src/*.cpp")

# shared lib
add_library(pvfmm SHARED ${pvfmm_SRC} ${pvfmm_SRC_CUDA})
set_target_properties(pvfmm PROPERTIES OUTPUT_NAME pvfmm)
target_include_directories(
  pvfmm
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SCTL/include/>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${CMAKE_CURRENT_BINARY_DIR} ${DEP_INC})
target_link_libraries(pvfmm PUBLIC ${DEP_LIB} ${DEP_LIB_CUDA} OpenMP::OpenMP_CXX MPI::MPI_CXX)
target_compile_definitions(pvfmm PUBLIC "${PVFMM_DEFINES}")

# static lib
add_library(pvfmmStatic STATIC ${pvfmm_SRC}  ${pvfmm_SRC_CUDA})
set_target_properties(pvfmmStatic PROPERTIES OUTPUT_NAME pvfmm)
target_include_directories(
  pvfmmStatic
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SCTL/include/>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
         ${CMAKE_CURRENT_BINARY_DIR} ${DEP_INC})
target_link_libraries(pvfmmStatic PUBLIC ${DEP_LIB} ${DEP_LIB_CUDA} OpenMP::OpenMP_CXX MPI::MPI_CXX)
target_compile_definitions(pvfmmStatic PUBLIC "${PVFMM_DEFINES}")

# Config.cmake files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/pvfmmConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

get_target_property(PVFMM_SHARED_NAME pvfmm OUTPUT_NAME)
get_target_property(PVFMM_STATIC_NAME pvfmmStatic OUTPUT_NAME)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pvfmmConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/pvfmmConfig.cmake INSTALL_DESTINATION
  ${CMAKE_INSTALL_PREFIX}/share/pvfmm)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pvfmmConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/pvfmmConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pvfmm)

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
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/SCTL/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pvfmm/)

add_subdirectory(examples)
