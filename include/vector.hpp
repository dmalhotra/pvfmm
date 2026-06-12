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

// Thin adapter over sctl::Vector<T> preserving pvfmm's historical API:
// size_t dimensions, Resize(), and raw-T* overloads under SCTL_MEMDEBUG.
// Copy/move/assignment semantics are sctl's (moves adopt the source's
// storage, including non-owning views — sites that want a deep copy say
// so explicitly via ReInit(n, ptr, true)).
template <class T>
class Vector : public sctl::Vector<T> {

  typedef sctl::Vector<T> Base;

  public:

  Vector() : Base() {}

  Vector(size_t dim_, sctl::Iterator<T> data_=sctl::NullIterator<T>(), bool own_data_=true) : Base((sctl::Long)dim_, data_, own_data_) {}

#if defined(SCTL_MEMDEBUG)
  // Legacy compatibility: accept raw T* and wrap into Iterator<T>.
  Vector(size_t dim_, T* data_, bool own_data_=true)
    : Vector(dim_, (data_? sctl::Ptr2Itr<T>(data_, (sctl::Long)dim_) : sctl::NullIterator<T>()), own_data_) {}
#endif

  Vector(const std::vector<T>& V) : Base(V) {}

  // Adopt/copy a base-class value (e.g. the result of an sctl::Vector
  // returning expression).
  Vector(Base&& V) noexcept : Base(std::move(V)) {}
  Vector(const Base& V) : Base(V) {}

  Vector& operator=(const std::vector<T>& V){ Base::operator=(V); return *this; }

  void Swap(Vector<T>& v1){ Base::Swap(v1); }

  void ReInit(size_t dim_, sctl::Iterator<T> data_=sctl::NullIterator<T>(), bool own_data_=true){ Base::ReInit((sctl::Long)dim_, data_, own_data_); }

#if defined(SCTL_MEMDEBUG)
  void ReInit(size_t dim_, T* data_, bool own_data_=true) {
    ReInit(dim_, (data_? sctl::Ptr2Itr<T>(data_, (sctl::Long)dim_) : sctl::NullIterator<T>()), own_data_);
  }
#endif

  size_t Dim() const { return (size_t)Base::Dim(); }

  // Historical Resize kept contents when the existing capacity sufficed.
  // Base::ReInit has the same fast path in release builds; under
  // SCTL_MEMDEBUG it deliberately destroys and rebuilds (stricter checking),
  // so contents are not preserved there.
  void Resize(size_t dim_){ if(Dim()!=dim_) Base::ReInit((sctl::Long)dim_); }

};

// Null-safe raw-pointer view: NULL for empty vectors and for the
// dim>0/null-storage placeholder state (ReInit(n,NULL,false)) that
// FMM_Pts::CollectNodeData uses to request buffer space. Use only at
// terminal consumption points (MPI, memcpy, device copies); carry
// iterators (v.begin()) everywhere else.
template <class T>
T* VecBegin(sctl::Vector<T>& v){ sctl::Iterator<T> it=v.begin(); return (v.Dim()>0 && it!=sctl::NullIterator<T>() ? &it[0] : (T*)NULL); }
template <class T>
const T* VecBegin(const sctl::Vector<T>& v){ sctl::ConstIterator<T> it=v.begin(); return (v.Dim()>0 && it!=sctl::NullIterator<T>() ? &it[0] : (const T*)NULL); }

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_VECTOR_HPP_
