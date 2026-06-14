/**
 * \file matrix.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief Compatibility header: pvfmm uses sctl::Matrix<T> directly.
 *
 * Call sites name sctl::Matrix (M[i] yields an iterator; use &M[i][0] where a
 * raw pointer is needed, M.ReInit(i,j) to resize). The pvfmm-specific helpers
 * kept here are MatrixTranspose() and the CUDA CUBLASGEMM wrapper.
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

// Call sites use sctl::Permutation directly. Its perm holds sctl::Long
// entries (PVFMM_PERM_INT_T) — same width as the historical size_t, so the
// packed precomp data and cache files are byte-compatible.
#define PVFMM_PERM_INT_T sctl::Long

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
void CUBLASGEMM(sctl::Matrix<T>& M_r, const sctl::Matrix<T>& A, const sctl::Matrix<T>& B, T beta=0.0);
#endif

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <matrix.txx>

#endif //_PVFMM_MATRIX_HPP_
