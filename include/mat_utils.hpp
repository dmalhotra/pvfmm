/**
 * \file mat_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains BLAS and LAPACK wrapper functions.
 */

#ifndef _PVFMM_MAT_UTILS_
#define _PVFMM_MAT_UTILS_

#include <cstdlib>

namespace pvfmm{
namespace mat{

  void gemm(char TransA, char TransB,  int M,  int N,  int K,  float alpha,  float *A,  int lda,  float *B,  int ldb,  float beta, float *C,  int ldc);

  void gemm(char TransA, char TransB,  int M,  int N,  int K,  double alpha,  double *A,  int lda,  double *B,  int ldb,  double beta, double *C,  int ldc);

  // cublasXgemm wrapper
  void cublasXgemm(char TransA, char TransB,  int M,  int N,  int K,  float alpha,  float *A,  int lda,  float *B,  int ldb,  float beta, float *C,  int ldc);

  void cublasXgemm(char TransA, char TransB,  int M,  int N,  int K,  double alpha,  double *A,  int lda,  double *B,  int ldb,  double beta, double *C,  int ldc);

  void svd(char *JOBU, char *JOBVT, int *M, int *N, float *A, int *LDA,
      float *S, float *U, int *LDU, float *VT, int *LDVT, float *WORK, int *LWORK,
      int *INFO);

  void svd(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
      double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK,
      int *INFO);

  /**
   * \brief Computes the pseudo inverse of matrix M(n1xn2) (in row major form)
   * and returns the output M_(n2xn1).
   */
  template <class T>
  void pinv(T* M, int n1, int n2, T eps, T* M_);

}//end namespace
}//end namespace

#include <mat_utils.txx>

#endif //_PVFMM_MAT_UTILS_
