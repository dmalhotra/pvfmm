/**
 * \file vector.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains definition of the class Vector.
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

template <class T>
class Vector{

  template <class Y>
  friend std::ostream& operator<<(std::ostream& output, const Vector<Y>& V);

  public:

  struct
  Device{

    Device& operator=(Vector& V){
      dim=V.Dim();
      dev_ptr=(uintptr_t)&V[0];
      return *this;
    }

    inline T& operator[](size_t j) const{
      return ((T*)dev_ptr)[j];
    }

    size_t dim;
    uintptr_t dev_ptr;
  };

  Vector();

  Vector(size_t dim_, sctl::Iterator<T> data_=sctl::NullIterator<T>(), bool own_data_=true);

#if defined(SCTL_MEMDEBUG)
  // Legacy compatibility: accept raw T* and wrap into Iterator<T>.
  Vector(size_t dim_, T* data_, bool own_data_=true)
    : Vector(dim_, (data_? sctl::Ptr2Itr<T>(data_, (sctl::Long)dim_) : sctl::NullIterator<T>()), own_data_) {}
#endif

  Vector(const Vector& V);

  // Move-construct: transfer ownership of data_ptr / dev to the new Vector,
  // leaving the source empty. The implicit move would shallow-copy data_ptr,
  // causing a double-free when both Vectors are destroyed. This is harmless
  // under pvfmm's old MemoryManager (free silently tolerated double-free) but
  // sctl::MemoryManager (used after Phase 3) asserts on it.
  Vector(const std::vector<T>& V);

  ~Vector();

  void Swap(Vector<T>& v1);

  void ReInit(size_t dim_, sctl::Iterator<T> data_=sctl::NullIterator<T>(), bool own_data_=true);

#if defined(SCTL_MEMDEBUG)
  void ReInit(size_t dim_, T* data_, bool own_data_=true) {
    ReInit(dim_, (data_? sctl::Ptr2Itr<T>(data_, (sctl::Long)dim_) : sctl::NullIterator<T>()), own_data_);
  }
#endif

  Device& AllocDevice(bool copy);

  void Device2Host();

  void FreeDevice(bool copy);

  void Write(const char* fname);

  size_t Dim() const;

  size_t Capacity() const;

  void Resize(size_t dim_);

  void SetZero();

  T* Begin();

  const T* Begin() const;

  Vector& operator=(const Vector& V);

  Vector& operator=(const std::vector<T>& V);

  T& operator[](size_t j);

  const T& operator[](size_t j) const;

  private:

  size_t dim;
  size_t capacity;
  sctl::Iterator<T> data_ptr;
  bool own_data;
  Device dev;
};

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <vector.txx>

#endif //_PVFMM_VECTOR_HPP_
