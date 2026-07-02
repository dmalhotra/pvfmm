/**
 * \file pvfmm_common.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-10-2010
 * \brief This file contains common definitions.
 */

#ifndef _PVFMM_COMMON_HPP_
#define _PVFMM_COMMON_HPP_

// Directory for precomputed-data files (overridable via -D from the build,
// e.g. autotools --with-precomp-dir). Empty => current dir / $PVFMM_DIR.
#ifndef PVFMM_PRECOMP_DATA_PATH
#define PVFMM_PRECOMP_DATA_PATH ""
#endif
// Feature toggles (PVFMM_HAVE_CUDA, PVFMM_HAVE_PAPI, PVFMM_EXTENDED_BC) are
// passed as -D by the build when enabled; undefined => off.

//Define NULL
#ifndef NULL
#define NULL 0
#endif

//Disable assert checks.
#ifndef PVFMM_NDEBUG
#define PVFMM_NDEBUG
#endif

//Enable profiling (sctl::Profile level threshold; override with -DSCTL_PROFILE=<n>, 0 to disable)
#ifndef SCTL_PROFILE
#define SCTL_PROFILE 10
#endif

//Verbose (sctl::Profile stdout output + pvfmm diagnostics; enable with -DSCTL_VERBOSE)
//#define SCTL_VERBOSE

#define PVFMM_MAX_DEPTH 30

#define PVFMM_BC_LEVELS 45

#define PVFMM_RAD0 1.05 //Radius of upward equivalent (downward check) surface.
#define PVFMM_RAD1 2.95 //Radius of downward equivalent (upward check) surface.

#define PVFMM_COORD_DIM 3
#define PVFMM_COLLEAGUE_COUNT 27 // 3^COORD_DIM

#define PVFMM_MEM_ALIGN 64
#define PVFMM_DEVICE_BUFFER_SIZE 1024LL //in MB
#define PVFMM_V_BLK_CACHE 25 //in KB
#define PVFMM_GLOBAL_MEM_BUFF 1024LL*0LL //in MB

#ifndef PVFMM_DEVICE_SYNC
#define PVFMM_DEVICE_SYNC 0 // No device synchronization by default.
#endif

#define PVFMM_ALLTOALLV_FIX // Use custom alltoallv implementation

#define PVFMM_UNUSED(x) (void)(x) // to ignore unused variable warning.

#include <cstring>
#ifndef PVFMM_NDEBUG
#include <cassert>
#include <iostream>
#define PVFMM_ASSERT_WITH_MSG(cond, msg) do \
{ if (!(cond)) { std::cerr<<"Error: "<<msg<<'\n'; assert(cond); } \
} while(0)
#else
#define PVFMM_ASSERT_WITH_MSG(cond, msg)
#endif

#include <sctl/stacktrace.h>
#include <cstdint>

namespace pvfmm{
namespace mem{
inline uintptr_t align_ptr(uintptr_t ptr){
  static constexpr uintptr_t     ALIGN_MINUS_ONE=PVFMM_MEM_ALIGN-1;
  static constexpr uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
  return ((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
}
}//end namespace
}//end namespace

// Keep sctl::Morton<3>'s depth range in lock-step with pvfmm's octree depth
// (must be set before sctl.hpp so sctl::MortonCode picks the right code width).
#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH PVFMM_MAX_DEPTH
#endif

#include <sctl.hpp>

#endif //_PVFMM_COMMON_HPP_
