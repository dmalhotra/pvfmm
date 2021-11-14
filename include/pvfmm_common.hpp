/**
 * \file pvfmm_common.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-10-2010
 * \brief This file contains common definitions.
 */

#include "pvfmm_config.h"

#ifndef _PVFMM_COMMON_HPP_
#define _PVFMM_COMMON_HPP_

//Define NULL
#ifndef NULL
#define NULL 0
#endif

//Disable assert checks.
#ifndef PVFMM_NDEBUG
#define PVFMM_NDEBUG
#endif

//Enable profiling
#define PVFMM_PROFILE 5

//Verbose
//#define PVFMM_VERBOSE

#define PVFMM_MAX_DEPTH 30

#define PVFMM_BC_LEVELS 30

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

#ifndef PVFMM_NDEBUG
#include <cassert>
#include <iostream>
#define PVFMM_ASSERT_WITH_MSG(cond, msg) do \
{ if (!(cond)) { std::cerr<<"Error: "<<msg<<'\n'; assert(cond); } \
} while(0)
#else
#define PVFMM_ASSERT_WITH_MSG(cond, msg)
#endif

#include <stacktrace.h>

#include <sctl.hpp>

#endif //_PVFMM_COMMON_HPP_
