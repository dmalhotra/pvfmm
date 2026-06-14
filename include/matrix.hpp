/**
 * \file matrix.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief pvfmm::Matrix is an alias for sctl::Matrix<T>.
 *
 * Call sites use sctl::Matrix directly (M[i] yields an iterator; use
 * &M[i][0] or MatBegin(M) where a raw pointer is needed). The pvfmm-specific
 * helpers kept as free functions are Resize(), MatBegin(), MatrixTranspose()
 * and the CUDA CUBLASGEMM wrapper.
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

template <class T> using Matrix = sctl::Matrix<T>;

// Resize-if-needed: matches the historical pvfmm::Matrix::Resize (a no-op when
// the dimensions are unchanged; otherwise reallocates, NOT preserving data).
template <class T> inline void Resize(sctl::Matrix<T>& M, size_t i, size_t j){ if((size_t)M.Dim(0)!=i || (size_t)M.Dim(1)!=j) M.ReInit((sctl::Long)i, (sctl::Long)j); }

// Null-safe raw-pointer view of a matrix (see VecBegin).
template <class T>
T* MatBegin(sctl::Matrix<T>& M){ sctl::Iterator<T> it=M.begin(); return (M.Dim(0)*M.Dim(1)>0 && it!=sctl::NullIterator<T>() ? &it[0] : (T*)NULL); }
template <class T>
const T* MatBegin(const sctl::Matrix<T>& M){ sctl::ConstIterator<T> it=M.begin(); return (M.Dim(0)*M.Dim(1)>0 && it!=sctl::NullIterator<T>() ? &it[0] : (const T*)NULL); }

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

#if defined(PVFMM_HAVE_CUDA)
// cublasgemm wrapper (device GEMM).
template <class T>
void CUBLASGEMM(Matrix<T>& M_r, const Matrix<T>& A, const Matrix<T>& B, T beta=0.0);
#endif

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <matrix.txx>

#endif //_PVFMM_MATRIX_HPP_
