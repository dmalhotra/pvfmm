/**
 * \file mat_utils.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains BLAS and LAPACK wrapper functions.
 */

#include <cassert>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <math.h>
#include <blas.h>
#include <lapack.h>
#include <fft_wrapper.hpp>

#include <device_wrapper.hpp>
#if defined(PVFMM_HAVE_CUDA)
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

namespace pvfmm{
namespace mat{

  inline void gemm(char TransA, char TransB,  int M,  int N,  int K,  float alpha,  float *A,  int lda,  float *B,  int ldb,  float beta, float *C,  int ldc){
      sgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }

  inline void gemm(char TransA, char TransB,  int M,  int N,  int K,  double alpha,  double *A,  int lda,  double *B,  int ldb,  double beta, double *C,  int ldc){
      dgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }

// cublasDgemm wrapper
  inline void cublasXgemm(char TransA, char TransB,  int M,  int N,  int K,  double alpha,  double *A,  int lda,  double *B,  int ldb,  double beta, double *C,  int ldc){
	cublasOperation_t cublasTransA, cublasTransB;
	cublasStatus_t status;
	cublasHandle_t *handle;
	handle = DeviceWrapper::CUDA_Lock::acquire_handle();
	/* Need exeception handling if (handle) */
	if (TransA == 'T' || TransA == 't') cublasTransA = CUBLAS_OP_T;
	else if (TransA == 'N' || TransA == 'n') cublasTransA = CUBLAS_OP_T;
	if (TransB == 'T' || TransB == 't') cublasTransB = CUBLAS_OP_T;
	else if (TransB == 'N' || TransB == 'n') cublasTransB = CUBLAS_OP_T;
    status = cublasDgemm(*handle, cublasTransA, cublasTransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

// cublasDgemm wrapper
  inline void cublasXgemm(char TransA, char TransB,  int M,  int N,  int K,  float alpha,  float *A,  int lda,  float *B,  int ldb,  float beta, float *C,  int ldc){
	cublasOperation_t cublasTransA, cublasTransB;
	cublasStatus_t status;
	cublasHandle_t *handle;
	handle = DeviceWrapper::CUDA_Lock::acquire_handle();
	/* Need exeception handling if (handle) */
	if (TransA == 'T' || TransA == 't') cublasTransA = CUBLAS_OP_T;
	else if (TransA == 'N' || TransA == 'n') cublasTransA = CUBLAS_OP_T;
	if (TransB == 'T' || TransB == 't') cublasTransB = CUBLAS_OP_T;
	else if (TransB == 'N' || TransB == 'n') cublasTransB = CUBLAS_OP_T;
    status = cublasSgemm(*handle, cublasTransA, cublasTransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  inline void svd(char *JOBU, char *JOBVT, int *M, int *N, float *A, int *LDA,
      float *S, float *U, int *LDU, float *VT, int *LDVT, float *WORK, int *LWORK,
      int *INFO){
    sgesvd_(JOBU,JOBVT,M,N,A,LDA,S,U,LDU,VT,LDVT,WORK,LWORK,INFO);
  }

  inline void svd(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
      double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK,
      int *INFO){
    dgesvd_(JOBU,JOBVT,M,N,A,LDA,S,U,LDU,VT,LDVT,WORK,LWORK,INFO);
  }

  /**
   * \brief Computes the pseudo inverse of matrix M(n1xn2) (in row major form)
   * and returns the output M_(n2xn1).
   */
  template <class T>
  void pinv(T* M, int n1, int n2, T eps, T* M_){
    int m = n2;
    int n = n1;
    int k = (m<n?m:n);

    std::vector<T> tU(m*k);
    std::vector<T> tS(k);
    std::vector<T> tVT(k*n);

    //SVD
    int INFO=0;
    char JOBU  = 'S';
    char JOBVT = 'S';

    //int wssize = max(3*min(m,n)+max(m,n), 5*min(m,n));
    int wssize = 3*(m<n?m:n)+(m>n?m:n);
    int wssize1 = 5*(m<n?m:n);
    wssize = (wssize>wssize1?wssize:wssize1);

    T* wsbuf = new T[wssize];

    svd(&JOBU, &JOBVT, &m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k,
        wsbuf, &wssize, &INFO);
    if(INFO!=0)
      std::cout<<INFO<<'\n';
    assert(INFO==0);
    delete [] wsbuf;

    T eps_=tS[0]*eps;
    for(int i=0;i<k;i++)
      if(tS[i]<eps_)
        tS[i]=0;
      else
        tS[i]=1.0/tS[i];


    for(int i=0;i<m;i++){
      for(int j=0;j<k;j++){
        tU[i+j*m]*=tS[j];
      }
    }

    gemm('T','T',n,m,k,1.0,&tVT[0],k,&tU[0],m,0.0,M_,n);
  }

}//end namespace
}//end namespace

