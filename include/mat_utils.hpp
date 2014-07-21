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

  template <class T>
  void gemm(char TransA, char TransB,  int M,  int N,  int K,  T alpha,  T *A,  int lda,  T *B,  int ldb,  T beta, T *C,  int ldc);

  template <class T>
  void svd(char *JOBU, char *JOBVT, int *M, int *N, T *A, int *LDA,
      T *S, T *U, int *LDU, T *VT, int *LDVT, T *WORK, int *LWORK,
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
