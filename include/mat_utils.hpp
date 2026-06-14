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
  void cublasgemm(char TransA, char TransB,  int M,  int N,  int K,  T alpha,  const T *A,  int lda,  const T *B,  int ldb,  T beta, T *C,  int ldc);

  #if defined(PVFMM_HAVE_CUDA)
  template <> inline void cublasgemm<float>(char TransA, char TransB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
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

  template <> inline void cublasgemm<double>(char TransA, char TransB, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc) {
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

}//end namespace
}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_MAT_UTILS_
