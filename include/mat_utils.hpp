/**
 * \file mat_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains BLAS and LAPACK wrapper functions.
 */

#include <pvfmm_common.hpp>

#if defined(PVFMM_HAVE_CUDA)
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#ifndef _PVFMM_MAT_UTILS_
#define _PVFMM_MAT_UTILS_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{
namespace mat{

  template <class T>
  void gemm(char TransA, char TransB,  int M,  int N,  int K,  T alpha,  T *A,  int lda,  T *B,  int ldb,  T beta, T *C,  int ldc) {
    sctl::mat::gemm(TransA, TransB, M, N, K, alpha, sctl::Ptr2Itr<T>(A,M*K), lda, sctl::Ptr2Itr<T>(B,K*N), ldb, beta, sctl::Ptr2Itr<T>(C,M*N), ldc);
  }

  template <class T>
  void cublasgemm(char TransA, char TransB,  int M,  int N,  int K,  T alpha,  T *A,  int lda,  T *B,  int ldb,  T beta, T *C,  int ldc);

  #if defined(PVFMM_HAVE_CUDA)
  template <> inline void cublasgemm<float>(char TransA, char TransB, int M, int N, int K, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc) {
    cublasOperation_t cublasTransA, cublasTransB;
    cublasHandle_t *handle = CUDA_Lock::acquire_handle();
    if (TransA == 'T' || TransA == 't')
      cublasTransA = CUBLAS_OP_T;
    else if (TransA == 'N' || TransA == 'n')
      cublasTransA = CUBLAS_OP_N;
    if (TransB == 'T' || TransB == 't')
      cublasTransB = CUBLAS_OP_T;
    else if (TransB == 'N' || TransB == 'n')
      cublasTransB = CUBLAS_OP_N;
    cublasStatus_t status = cublasSgemm(*handle, cublasTransA, cublasTransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    PVFMM_UNUSED(status);
  }

  template <> inline void cublasgemm<double>(char TransA, char TransB, int M, int N, int K, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc) {
    cublasOperation_t cublasTransA, cublasTransB;
    cublasHandle_t *handle = CUDA_Lock::acquire_handle();
    if (TransA == 'T' || TransA == 't')
      cublasTransA = CUBLAS_OP_T;
    else if (TransA == 'N' || TransA == 'n')
      cublasTransA = CUBLAS_OP_N;
    if (TransB == 'T' || TransB == 't')
      cublasTransB = CUBLAS_OP_T;
    else if (TransB == 'N' || TransB == 'n')
      cublasTransB = CUBLAS_OP_N;
    cublasStatus_t status = cublasDgemm(*handle, cublasTransA, cublasTransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    PVFMM_UNUSED(status);
  }
  #endif

  template <class T>
  void svd(char *JOBU, char *JOBVT, int *M, int *N, T *A, int *LDA, T *S, T *U, int *LDU, T *VT, int *LDVT, T *WORK, int *LWORK, int *INFO) {
    int m = *M;
    int n = *N;
    int k = (m < n ? m : n);
    int wssize = *LWORK;
    sctl::mat::svd(JOBU, JOBVT, M, N, sctl::Ptr2Itr<T>(A,n*m), LDA, sctl::Ptr2Itr<T>(S,k*k), sctl::Ptr2Itr<T>(U,n*k), LDU, sctl::Ptr2Itr<T>(VT,k*m), LDVT, sctl::Ptr2Itr<T>(WORK,wssize), LWORK, INFO);
  }

  /**
   * \brief Computes the pseudo inverse of matrix M(n1xn2) (in row major form)
   * and returns the output M_(n2xn1).
   */
  template <class T>
  void pinv(T* M, int n1, int n2, T eps, T* M_) {
    sctl::mat::pinv(sctl::Ptr2Itr<T>(M, n1*n2), n1, n2, eps, sctl::Ptr2Itr<T>(M_, n2*n1));
  }

}//end namespace
}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_MAT_UTILS_
