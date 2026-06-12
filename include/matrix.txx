/**
 * \file matrix.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief Implementation of the pvfmm-specific parts of the Matrix adapter
 * (CUBLASGEMM, Permutation-based Row/ColPerm) and the Permutation class.
 * Everything else is inherited from sctl::Matrix.
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
void Matrix<T>::RowPerm(const Permutation<T>& P){
  Matrix<T>& M=*this;
  if(P.Dim()==0) return;
  assert(M.Dim(0)==P.Dim());
  size_t d0=M.Dim(0);
  size_t d1=M.Dim(1);

  #pragma omp parallel for
  for(size_t i=0;i<d0;i++){
    T* M_=M[i];
    const T s=P.scal[i];
    for(size_t j=0;j<d1;j++) M_[j]*=s;
  }

  Permutation<T> P_=P;
  for(size_t i=0;i<d0;i++)
  while(P_.perm[i]!=i){
    size_t a=P_.perm[i];
    size_t b=i;
    T* M_a=M[a];
    T* M_b=M[b];
    std::swap<PVFMM_PERM_INT_T>(P_.perm[a],P_.perm[b]);
    for(size_t j=0;j<d1;j++)
      std::swap<T>(M_a[j],M_b[j]);
  }
}

template <class T>
void Matrix<T>::ColPerm(const Permutation<T>& P){
  Matrix<T>& M=*this;
  if(P.Dim()==0) return;
  assert(M.Dim(1)==P.Dim());
  size_t d0=M.Dim(0);
  size_t d1=M.Dim(1);

  int omp_p=omp_get_max_threads();
  Matrix<T> M_buff(omp_p,d1);

  const size_t* perm_=&(P.perm[0]);
  const T* scal_=&(P.scal[0]);
  #pragma omp parallel for
  for(size_t i=0;i<d0;i++){
    int pid=omp_get_thread_num();
    T* buff=&M_buff[pid][0];
    T* M_=M[i];
    for(size_t j=0;j<d1;j++)
      buff[j]=M_[j];
    for(size_t j=0;j<d1;j++){
      M_[j]=buff[perm_[j]]*scal_[j];
    }
  }
}

#define PVFMM_B1 128
#define PVFMM_B2 32

template <class T>
std::ostream& operator<<(std::ostream& output, const Permutation<T>& P){
  output<<std::setprecision(4)<<std::setiosflags(std::ios::left);
  size_t size=P.perm.Dim();
  for(size_t i=0;i<size;i++) output<<std::setw(10)<<P.perm[i]<<' ';
  output<<";\n";
  for(size_t i=0;i<size;i++) output<<std::setw(10)<<P.scal[i]<<' ';
  output<<";\n";
  return output;
}

template <class T>
Permutation<T>::Permutation(size_t size){
  perm.Resize(size);
  scal.Resize(size);
  for(size_t i=0;i<size;i++){
    perm[i]=i;
    scal[i]=1.0;
  }
}

template <class T>
Permutation<T> Permutation<T>::RandPerm(size_t size){
  Permutation<T> P(size);
  for(size_t i=0;i<size;i++){
    P.perm[i]=rand()%size;
    for(size_t j=0;j<i;j++)
      if(P.perm[i]==P.perm[j]){ i--; break; }
    P.scal[i]=((T)rand())/RAND_MAX;
  }
  return P;
}

template <class T>
Matrix<T> Permutation<T>::GetMatrix() const{
  size_t size=perm.Dim();
  Matrix<T> M_r(size,size,NULL);
  for(size_t i=0;i<size;i++)
    for(size_t j=0;j<size;j++)
      M_r[i][j]=(perm[j]==i?scal[j]:0.0);
  return M_r;
}

template <class T>
size_t Permutation<T>::Dim() const{
  return perm.Dim();
}

template <class T>
Permutation<T> Permutation<T>::Transpose(){
  size_t size=perm.Dim();
  Permutation<T> P_r(size);

  Vector<PVFMM_PERM_INT_T>& perm_r=P_r.perm;
  Vector<T>& scal_r=P_r.scal;
  for(size_t i=0;i<size;i++){
    perm_r[perm[i]]=i;
    scal_r[perm[i]]=scal[i];
  }
  return P_r;
}

template <class T>
Permutation<T>& Permutation<T>::operator*=(const Permutation<T>& P){
  size_t size=perm.Dim();
  assert(P.Dim()==size);
  sctl::ScratchBuf<PVFMM_PERM_INT_T> old_perm((sctl::Long)size);
  sctl::ScratchBuf<T> old_scal((sctl::Long)size);
  for(size_t i=0;i<size;i++){ old_perm[i]=perm[i]; old_scal[i]=scal[i]; }
  for(size_t i=0;i<size;i++){
    perm[i]=old_perm[P.perm[i]];
    scal[i]=old_scal[P.perm[i]]*P.scal[i];
  }
  return *this;
}

template <class T>
Permutation<T> Permutation<T>::operator*(const Permutation<T>& P){
  size_t size=perm.Dim();
  assert(P.Dim()==size);

  Permutation<T> P_r(size);
  Vector<PVFMM_PERM_INT_T>& perm_r=P_r.perm;
  Vector<T>& scal_r=P_r.scal;
  for(size_t i=0;i<size;i++){
    perm_r[i]=perm[P.perm[i]];
    scal_r[i]=scal[P.perm[i]]*P.scal[i];
  }
  return P_r;
}

template <class T>
Matrix<T> Permutation<T>::operator*(const Matrix<T>& M){
  if(Dim()==0) return M;
  assert(M.Dim(0)==Dim());
  size_t d0=M.Dim(0);
  size_t d1=M.Dim(1);

  Matrix<T> M_r(d0,d1,NULL);
  for(size_t i=0;i<d0;i++){
    const T s=scal[i];
    const T* M_=M[i];
    T* M_r_=M_r[perm[i]];
    for(size_t j=0;j<d1;j++)
      M_r_[j]=M_[j]*s;
  }
  return M_r;
}

template <class T>
Matrix<T> operator*(const Matrix<T>& M, const Permutation<T>& P){
  if(P.Dim()==0) return M;
  assert(M.Dim(1)==P.Dim());
  size_t d0=M.Dim(0);
  size_t d1=M.Dim(1);

  Matrix<T> M_r(d0,d1,NULL);
  for(size_t i=0;i<d0;i++){
    const PVFMM_PERM_INT_T* perm_=&(P.perm[0]);
    const T* scal_=&(P.scal[0]);
    const T* M_=M[i];
    T* M_r_=M_r[i];
    for(size_t j=0;j<d1;j++)
      M_r_[j]=M_[perm_[j]]*scal_[j];
  }
  return M_r;
}

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
