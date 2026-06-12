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
// size_t dimensions, raw-pointer Begin(), Resize(), and copy-only semantics.
// No move operations: `vec = Vector<T>(n, ptr, false)` must deep-copy
// (MPI_Node::Unpack relies on it); an implicit move would adopt the alias
// and leave vec dangling once the underlying buffer is freed.
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

  Vector(const Vector& V) : Base(V) {}

  Vector(const std::vector<T>& V) : Base(V) {}

  // Adopt/copy a base-class value (e.g. the result of an sctl::Vector
  // returning expression). Same-type rvalues still prefer the copy ctor,
  // so pvfmm::Vector itself remains copy-only.
  Vector(Base&& V) noexcept : Base(std::move(V)) {}
  Vector(const Base& V) : Base(V) {}

  ~Vector() = default; // user-declared: suppresses implicit moves (see class comment)

  Vector& operator=(const Vector& V){ Base::operator=((const Base&)V); return *this; }

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

  // Raw-pointer view of the data; NULL for empty vectors and for the
  // dim>0/null-storage placeholder state (ReInit(n,NULL,false)) that
  // FMM_Pts::CollectNodeData uses to request buffer space.
  T* Begin(){ sctl::Iterator<T> it=Base::begin(); return (Dim()>0 && it!=sctl::NullIterator<T>() ? &it[0] : (T*)NULL); }

  const T* Begin() const{ sctl::ConstIterator<T> it=Base::begin(); return (Dim()>0 && it!=sctl::NullIterator<T>() ? &it[0] : (const T*)NULL); }
};

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_VECTOR_HPP_
