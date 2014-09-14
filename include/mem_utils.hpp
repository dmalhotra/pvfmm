/**
 * \file mat_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 11-5-2013
 * \brief This file contains memory management utilities.
 */

#include <cstdlib>

#include <pvfmm_common.hpp>

#ifndef _PVFMM_MEM_UTILS_
#define _PVFMM_MEM_UTILS_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{
namespace mem{

  // Aligned memory allocation.
  // Alignment must be power of 2 (1,2,4,8,16...)
  template <class T>
  T* aligned_malloc(size_t size_=1, size_t alignment=MEM_ALIGN);

  // Aligned memory free.
  template <class T>
  void aligned_free(T* p_);

  void * memcopy ( void * destination, const void * source, size_t num );

}//end namespace
}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <mem_utils.txx>

#endif //_PVFMM_MEM_UTILS_
