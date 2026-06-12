/**
 * \file matrix.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief Implementation of the pvfmm-specific parts of the Matrix adapter
 * (CUBLASGEMM, MatrixTranspose). Everything else, including Permutation and
 * RowPerm/ColPerm, comes from sctl.
 */

#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>

#include <device_wrapper.hpp> // CUDA_Lock, used by mat_utils.hpp's cublasgemm declaration
#include <mat_utils.hpp>

namespace pvfmm{

// Cache-blocking tile sizes for MatrixTranspose.
#define PVFMM_B1 128
#define PVFMM_B2 32

#if defined(PVFMM_HAVE_CUDA)
template <class T>
void Matrix<T>::CUBLASGEMM(Matrix<T>& M_r, const Matrix<T>& A, const Matrix<T>& B, T beta){
  if(A.Dim(0)*A.Dim(1)==0 || B.Dim(0)*B.Dim(1)==0) return;
  assert(A.Dim(1)==B.Dim(0));
  assert(M_r.Dim(0)==A.Dim(0));
  assert(M_r.Dim(1)==B.Dim(1));
  sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, 2*(((long long)A.Dim(0))*A.Dim(1))*B.Dim(1));
  // cublasgemm takes non-const T*; the inputs are consumed by the device
  // call, so casting away const here is terminal.
  mat::cublasgemm<T>('N', 'N', B.Dim(1), A.Dim(0), A.Dim(1),
      (T)1.0, (T*)B.Begin(), B.Dim(1), (T*)A.Begin(), A.Dim(1), beta, M_r.Begin(), M_r.Dim(1));
}
#endif


template <class T>
void MatrixTranspose(size_t in_dim1, size_t in_dim2, sctl::ConstIterator<T> in, sctl::Iterator<T> out){
  const size_t d0=in_dim1;
  const size_t d1=in_dim2;
  if(d0*d1==0) return;

  auto transpose_=[d0,d1](sctl::ConstIterator<T> in_, sctl::Iterator<T> out_){
    const size_t blk0=((d0+PVFMM_B1-1)/PVFMM_B1);
    const size_t blk1=((d1+PVFMM_B1-1)/PVFMM_B1);
    for(size_t k=0;k<blk0*blk1;k++){
      size_t i=(k%blk0)*PVFMM_B1;
      size_t j=(k/blk0)*PVFMM_B1;
      size_t d0_=i+PVFMM_B1; if(d0_>=d0) d0_=d0;
      size_t d1_=j+PVFMM_B1; if(d1_>=d1) d1_=d1;
      for(size_t ii=i;ii<d0_;ii+=PVFMM_B2)
      for(size_t jj=j;jj<d1_;jj+=PVFMM_B2){
        size_t d0__=ii+PVFMM_B2; if(d0__>=d0) d0__=d0;
        size_t d1__=jj+PVFMM_B2; if(d1__>=d1) d1__=d1;
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

}//end namespace
