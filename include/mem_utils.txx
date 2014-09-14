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

#include <mem_mgr.hpp>

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{
namespace mem{

  template <class T>
  class TypeId{
    public:
      static uintptr_t value(){
        return (uintptr_t)&value;
      }
  };

  struct PtrData{
    size_t n_elem;
    size_t type_id;
    size_t base_size;
    uintptr_t base;
  };

  // For fundamental data types.
  template <class T>
  T* aligned_malloc_f(size_t n_elem, size_t alignment){
    if(!n_elem) return NULL;
    size_t base_size=n_elem*sizeof(T) + --alignment+sizeof(PtrData);
    uintptr_t base_ptr = (uintptr_t)glbMemMgr.malloc(base_size);
    { // Debugging
      #ifndef NDEBUG
      for(size_t i=0;i<base_size;i++){
        ((char*)base_ptr)[i]=MemoryManager::init_mem_val;
      }
      #endif
    }
    ASSERT_WITH_MSG(base_ptr!=0, "malloc failed.");

    uintptr_t A = (uintptr_t)(base_ptr + alignment+sizeof(PtrData)) & ~(uintptr_t)alignment;
    PtrData& p_data=((PtrData*)A)[-1];
    p_data.n_elem=n_elem;
    p_data.type_id=TypeId<T>::value();
    p_data.base_size=base_size;
    p_data.base=base_ptr;

    return (T*)A;
  }
  template <class T>
  void aligned_free_f(T* A){
    void* p=(void*)A;
    if (!p) return;

    PtrData& p_data=((PtrData*)p)[-1];
    { // Debugging
      #ifndef NDEBUG
      for(char* ptr=(char*)p_data.base;ptr<((char*)A)-sizeof(PtrData);ptr++){
        assert(*ptr==MemoryManager::init_mem_val);
      }
      for(char* ptr=(char*)(A+p_data.n_elem);ptr<((char*)p_data.base)+p_data.base_size;ptr++){
        assert(*ptr==MemoryManager::init_mem_val);
      }
      #endif
    }

    glbMemMgr.free((char*)p_data.base);
  }

  template <class T>
  T* aligned_malloc(size_t n_elem, size_t alignment){
    if(!n_elem) return NULL;
    #ifdef NDEBUG
    T* A=new T[n_elem];
    assert(A!=NULL);
    #else
    T* A=aligned_malloc_f<T>(n_elem,alignment);
    PtrData& p_data=((PtrData*)A)[-1];
    assert(p_data.n_elem==n_elem);
    assert(p_data.type_id==TypeId<T>::value());
    for(size_t i=0;i<n_elem;i++){
      T* Ai=new(A+i) T();
      assert(Ai==(A+i));
    }
    #endif
    return A;
  }
  template <>
  inline double* aligned_malloc<double>(size_t n_elem, size_t alignment){
    return aligned_malloc_f<double>(n_elem,alignment);
  }
  template <>
  inline float* aligned_malloc<float>(size_t n_elem, size_t alignment){
    return aligned_malloc_f<float>(n_elem,alignment);
  }
  template <>
  inline char* aligned_malloc<char>(size_t n_elem, size_t alignment){
    return aligned_malloc_f<char>(n_elem,alignment);
  }

  template <class T>
  void aligned_free(T* A){
    #ifdef NDEBUG
    delete[] A;
    #else
    void* p=(void*)A;
    if (!p) return;

    PtrData& p_data=((PtrData*)p)[-1];
    assert(p_data.type_id==TypeId<T>::value());
    size_t n_elem=p_data.n_elem;
    for(size_t i=0;i<n_elem;i++){
      (A+i)->~T();
    }

    aligned_free_f<T>(A);
    #endif
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

