/**
 * \file mat_utils.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 11-5-2013
 * \brief This file contains implementation of mem_utils.hpp.
 */

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <stdint.h>

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{
namespace mem{

  // For fundamental data types.
  template <class T>
  T* aligned_malloc_f(size_t size_, size_t alignment){
    assert(alignment <= 0x8000);
    size_t size=size_*sizeof(T);
    uintptr_t r = (uintptr_t)malloc(size + --alignment + 2);
    //if (!r) return NULL;
    ASSERT_WITH_MSG(r!=0, "malloc failed.");
    uintptr_t o = (uintptr_t)(r + 2 + alignment) & ~(uintptr_t)alignment;
    ((uint16_t*)o)[-1] = (uint16_t)(o-r);
    return (T*)o;
    //return (T*)fftw_malloc(size);
  }
  template <class T>
  void aligned_free_f(T* p_){
    void* p=(void*)p_;
    if (!p) return;
    free((void*)((uintptr_t)p-((uint16_t*)p)[-1]));
    //fftw_free(p);
  }

  template <class T>
  T* aligned_malloc(size_t size_, size_t alignment){
    //void* p=aligned_malloc_f<T>(size_,alignment);
    //return new(p) T[size_];
    T* A=new T[size_];
    assert(A!=NULL);
    return A;
  }
  template <>
  inline double* aligned_malloc<double>(size_t size_, size_t alignment){
    return aligned_malloc_f<double>(size_,alignment);
  }
  template <>
  inline float* aligned_malloc<float>(size_t size_, size_t alignment){
    return aligned_malloc_f<float>(size_,alignment);
  }
  template <>
  inline char* aligned_malloc<char>(size_t size_, size_t alignment){
    return aligned_malloc_f<char>(size_,alignment);
  }

  template <class T>
  void aligned_free(T* p_){
    delete[] p_;
  }
  template <>
  inline void aligned_free<double>(double* p_){
    aligned_free_f<double>(p_);
  }
  template <>
  inline void aligned_free<float>(float* p_){
    aligned_free_f<float>(p_);
  }
  template <>
  inline void aligned_free<char>(char* p_){
    aligned_free_f<char>(p_);
  }

  inline void * memcopy ( void * destination, const void * source, size_t num ){
    if(destination==source || num==0) return destination;
    return memcpy ( destination, source, num );
  }

}//end namespace
}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif
