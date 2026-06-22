/**
 * \file mat_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief pvfmm-specific Matrix helpers: MatrixTranspose and the device GEMM
 * wrappers (CUBLASGEMM / mat::cublasgemm).
 */

#include <cassert>

#include <pvfmm_common.hpp>

#ifndef _PVFMM_MAT_UTILS_
#define _PVFMM_MAT_UTILS_

namespace pvfmm{
namespace mat{

  template <class T>
  void cublasgemm(char TransA, char TransB,  int M,  int N,  int K,  T alpha,  const T *A,  int lda,  const T *B,  int ldb,  T beta, T *C,  int ldc);

  // CUDA specializations are defined in device_wrapper.txx, where CUDA_Lock is visible.
  #if defined(PVFMM_HAVE_CUDA)
  template <> void cublasgemm<float >(char TransA, char TransB, int M, int N, int K, float  alpha, const float * A, int lda, const float * B, int ldb, float  beta, float * C, int ldc);
  template <> void cublasgemm<double>(char TransA, char TransB, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc);
  #endif

}//end namespace mat

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
void MatrixTranspose(size_t in_dim1, size_t in_dim2, sctl::ConstIterator<T> in, sctl::Iterator<T> out){
  constexpr size_t B1=128, B2=32; // cache-blocking tile sizes
  const size_t d0=in_dim1;
  const size_t d1=in_dim2;
  if(d0*d1==0) return;

  auto transpose_=[d0,d1](sctl::ConstIterator<T> in_, sctl::Iterator<T> out_){
    const size_t blk0=((d0+B1-1)/B1);
    const size_t blk1=((d1+B1-1)/B1);
    for(size_t k=0;k<blk0*blk1;k++){
      size_t i=(k%blk0)*B1;
      size_t j=(k/blk0)*B1;
      size_t d0_=i+B1; if(d0_>=d0) d0_=d0;
      size_t d1_=j+B1; if(d1_>=d1) d1_=d1;
      for(size_t ii=i;ii<d0_;ii+=B2)
      for(size_t jj=j;jj<d1_;jj+=B2){
        size_t d0__=ii+B2; if(d0__>=d0) d0__=d0;
        size_t d1__=jj+B2; if(d1__>=d1) d1__=d1;
        for(size_t iii=ii;iii<d0__;iii++)
        for(size_t jjj=jj;jjj<d1__;jjj++){
          out_[jjj*d0+iii]=in_[iii*d1+jjj];
        }
      }
    }
  };

  if(in==(sctl::ConstIterator<T>)out){ // in-place: stage the input in scratch
    sctl::ScratchBuf<T> buff((sctl::Long)(d0*d1));
    sctl::Iterator<T> tmp=buff.begin();
    sctl::omp_par::copy(in, in+(sctl::Long)(d0*d1), tmp);
    transpose_((sctl::ConstIterator<T>)tmp, out);
  }else{
    transpose_(in, out);
  }
}

#if defined(PVFMM_HAVE_CUDA)
// cublasgemm wrapper (device GEMM).
template <class T>
void CUBLASGEMM(sctl::Matrix<T>& M_r, const sctl::Matrix<T>& A, const sctl::Matrix<T>& B, T beta=0.0){
  if(A.Dim(0)*A.Dim(1)==0 || B.Dim(0)*B.Dim(1)==0) return;
  assert(A.Dim(1)==B.Dim(0));
  assert(M_r.Dim(0)==A.Dim(0));
  assert(M_r.Dim(1)==B.Dim(1));
  sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, 2*(((long long)A.Dim(0))*A.Dim(1))*B.Dim(1));
  mat::cublasgemm<T>('N', 'N', B.Dim(1), A.Dim(0), A.Dim(1),
      (T)1.0, &B[0][0], B.Dim(1), &A[0][0], A.Dim(1), beta, &M_r[0][0], M_r.Dim(1));
}
#endif

}//end namespace

#endif //_PVFMM_MAT_UTILS_
