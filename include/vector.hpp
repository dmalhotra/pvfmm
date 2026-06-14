/**
 * \file vector.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief Compatibility header: pvfmm uses sctl::Vector<T> directly.
 *
 * This header now just pulls in <sctl.hpp>; call sites name sctl::Vector.
 */

#include <vector>
#include <cstdlib>
#include <stdint.h>

#include <pvfmm_common.hpp>
#include <sctl.hpp>

#ifndef _PVFMM_VECTOR_HPP_
#define _PVFMM_VECTOR_HPP_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_VECTOR_HPP_
