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
//#define NDEBUG

//Enable profiling
#define __PROFILE__ 5

//Verbose
#define __VERBOSE__

#define MAX_DEPTH 15

#define BC_LEVELS 15

#define RAD0 1.05 //Radius of upward equivalent (downward check) surface.
#define RAD1 2.95 //Radius of downward equivalent (upward check) surface.

#define COORD_DIM 3
#define COLLEAGUE_COUNT 27 // 3^COORD_DIM

#define MEM_ALIGN 64
#define DEVICE_BUFFER_SIZE 1024LL //in MB
#define V_BLK_CACHE 25 //in KB
#define GLOBAL_MEM_BUFF 1024LL*10LL //in MB

#ifndef __DEVICE_SYNC__
#define __DEVICE_SYNC__ 0 // No device synchronization by default.
#endif

#define UNUSED(x) (void)(x) // to ignore unused variable warning.

#ifndef NDEBUG
#include <cassert>
#include <iostream>
#define ASSERT_WITH_MSG(cond, msg) do \
{ if (!(cond)) { std::cerr<<"Error: "<<msg<<'\n'; assert(cond); } \
} while(0)
#else
#define ASSERT_WITH_MSG(cond, msg)
#endif

#include <stacktrace.h>

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
template <class T>
inline T const_pi(){return 3.1415926535897932384626433832795028841;}

template <class T>
inline T const_e (){return 2.7182818284590452353602874713526624977;}
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <quad_utils.hpp>

#endif //_PVFMM_COMMON_HPP_
