/**
 * \file matrix.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains definition of the class Matrix.
 */

#include <stdint.h>
#include <cstdlib>

#include <pvfmm_common.hpp>
#include <vector.hpp>

#ifndef _PVFMM_MATRIX_HPP_
#define _PVFMM_MATRIX_HPP_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{

// Thin adapter over sctl::Matrix<T> preserving pvfmm's historical API:
// size_t dimensions, raw-pointer Begin() and operator[], Resize(), the
// pvfmm::Permutation-based RowPerm/ColPerm, CUBLASGEMM, and copy-only
// semantics (see Vector<T> for the no-moves rationale).
template <class T>
class Matrix : public sctl::Matrix<T> {

  typedef sctl::Matrix<T> Base;

  public:

  Matrix() : Base() {}

  Matrix(size_t dim1, size_t dim2, sctl::Iterator<T> data_=sctl::NullIterator<T>(), bool own_data_=true) : Base((sctl::Long)dim1, (sctl::Long)dim2, data_, own_data_) {}

#if defined(SCTL_MEMDEBUG)
  Matrix(size_t dim1, size_t dim2, T* data_, bool own_data_=true)
    : Matrix(dim1, dim2, (data_? sctl::Ptr2Itr<T>(data_, (sctl::Long)dim1*(sctl::Long)dim2) : sctl::NullIterator<T>()), own_data_) {}
#endif

  Matrix(const Matrix<T>& M) : Base(M) {}

  // Adopt/copy a base-class value (results of Transpose(), pinv(),
  // operator* etc., which return sctl::Matrix). Same-type rvalues still
  // prefer the copy ctor, so pvfmm::Matrix itself remains copy-only.
  Matrix(Base&& M) noexcept : Base(std::move(M)) {}
  Matrix(const Base& M) : Base(M) {}

  ~Matrix() = default; // user-declared: suppresses implicit moves

  Matrix<T>& operator=(const Matrix<T>& M){ Base::operator=((const Base&)M); return *this; }

  void Swap(Matrix<T>& M){ Base::Swap(M); }

  void ReInit(size_t dim1, size_t dim2, sctl::Iterator<T> data_=sctl::NullIterator<T>(), bool own_data_=true){ Base::ReInit((sctl::Long)dim1, (sctl::Long)dim2, data_, own_data_); }

#if defined(SCTL_MEMDEBUG)
  void ReInit(size_t dim1, size_t dim2, T* data_, bool own_data_=true) {
    ReInit(dim1, dim2, (data_? sctl::Ptr2Itr<T>(data_, (sctl::Long)dim1*(sctl::Long)dim2) : sctl::NullIterator<T>()), own_data_);
  }
#endif

  size_t Dim(size_t i) const { return (size_t)Base::Dim((sctl::Long)i); }

  // See Vector<T>::Resize for content-preservation semantics.
  void Resize(size_t i, size_t j){ if(Dim(0)!=i || Dim(1)!=j) Base::ReInit((sctl::Long)i, (sctl::Long)j); }

  T* Begin(){ sctl::Iterator<T> it=Base::begin(); return (Dim(0)*Dim(1)>0 && it!=sctl::NullIterator<T>() ? &it[0] : (T*)NULL); }

  const T* Begin() const{ sctl::ConstIterator<T> it=Base::begin(); return (Dim(0)*Dim(1)>0 && it!=sctl::NullIterator<T>() ? &it[0] : (const T*)NULL); }

  T* operator[](size_t i){ return &Base::operator[]((sctl::Long)i)[0]; }

  const T* operator[](size_t i) const{ return &Base::operator[]((sctl::Long)i)[0]; }

  // cublasgemm wrapper
  static void CUBLASGEMM(Matrix<T>& M_r, const Matrix<T>& A, const Matrix<T>& B, T beta=0.0);

};


// pvfmm::Permutation is sctl::Permutation. perm holds sctl::Long entries —
// same width as the historical PVFMM_PERM_INT_T (size_t), so the packed
// precomp data and cache files are byte-compatible.
#define PVFMM_PERM_INT_T sctl::Long
template <class T>
using Permutation = sctl::Permutation<T>;

/**
 * Transpose the in_dim1 x in_dim2 row-major matrix at `in` into `out`
 * (which receives in_dim2 x in_dim1). If in==out, the transpose is done in
 * place, staging the input through per-thread scratch storage; otherwise
 * the two ranges must not overlap.
 */
template <class T>
void MatrixTranspose(size_t in_dim1, size_t in_dim2, sctl::ConstIterator<T> in, sctl::Iterator<T> out);

#if defined(SCTL_MEMDEBUG)
// Legacy compatibility: accept raw pointers and wrap into iterators.
template <class T>
void MatrixTranspose(size_t in_dim1, size_t in_dim2, const T* in, T* out){
  const sctl::Long n=(sctl::Long)in_dim1*(sctl::Long)in_dim2;
  MatrixTranspose<T>(in_dim1, in_dim2,
      (in ? sctl::Ptr2ConstItr<T>(in, n) : sctl::ConstIterator<T>(sctl::NullIterator<T>())),
      (out? sctl::Ptr2Itr<T>(out, n) : sctl::NullIterator<T>()));
}
#endif

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <matrix.txx>

#endif //_PVFMM_MATRIX_HPP_
