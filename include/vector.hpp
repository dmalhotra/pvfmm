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

  Vector(size_t dim_, T* data_=NULL, bool own_data_=true);

  Vector(const Vector& V);

  Vector(const std::vector<T>& V);

  ~Vector();

  void Swap(Vector<T>& v1);

  void ReInit(size_t dim_, T* data_=NULL, bool own_data_=true);

  Device& AllocDevice(bool copy);

  void Device2Host();

  void FreeDevice(bool copy);

  void Write(const char* fname);

  size_t Dim() const;

  size_t Capacity() const;

  void Resize(size_t dim_);

  void SetZero();

  Vector& operator=(const Vector& V);

  Vector& operator=(const std::vector<T>& V);

  T& operator[](size_t j) const;

  private:

  size_t dim;
  size_t capacity;
  T* data_ptr;
  bool own_data;
  Device dev;
};

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <vector.txx>

#endif //_PVFMM_VECTOR_HPP_
