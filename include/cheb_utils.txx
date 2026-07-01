/**
 * \file cheb_utils.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains chebyshev related functions.
 */

#include <omp.h>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <utility>

#include <mat_utils.hpp>

#include <pvfmm_common.hpp>
#include <kernel.hpp>

namespace pvfmm{

/**
 * \brief Returns the values of all chebyshev polynomials up to degree d,
 * evaluated at points in the input vector. Output format:
 * { T0[in[0]], ..., T0[in[n-1]], T1[in[0]], ..., T(d-1)[in[n-1]] }
 */
template <class T>
inline void cheb_poly(int d, const T* in, int n, T* out){
  if(d==0){
    for(int i=0;i<n;i++)
      out[i]=(sctl::fabs<T>(in[i])<=1?1.0:0);
  }else if(d==1){
    for(int i=0;i<n;i++){
      out[i]=(sctl::fabs<T>(in[i])<=1?1.0:0);
      out[i+n]=(sctl::fabs<T>(in[i])<=1?in[i]:0);
    }
  }else{
    for(int j=0;j<n;j++){
      T x=(sctl::fabs<T>(in[j])<=1?in[j]:0);
      T y0=(sctl::fabs<T>(in[j])<=1?1.0:0);
      out[j]=y0;
      out[j+n]=x;

      T y1=x;
      T* y2=&out[2*n+j];
      for(int i=2;i<=d;i++){
        *y2=2*x*y1-y0;
        y0=y1;
        y1=*y2;
        y2=&y2[n];
      }
    }
  }
}

/**
 * \brief Returns the sum of the absolute value of coeffecients of the highest
 * order polynomial as an estimate of error.
 */
template <class T>
T cheb_err(T* cheb_coeff, int deg, int dof){
  T err=0;
  int indx=0;

  for(int l=0;l<dof;l++)
  for(int i=0;i<=deg;i++)
  for(int j=0;i+j<=deg;j++)
  for(int k=0;i+j+k<=deg;k++){
    if(i+j+k==deg) err+=sctl::fabs<T>(cheb_coeff[indx]);
    indx++;
  }
  return err;
}


template<typename U1, typename U2>
struct SameType{
  bool operator()(){return false;}
};
template<typename U>
struct SameType<U, U>{
  bool operator()(){return true;}
};

/**
 * \brief Computes Chebyshev approximation from function values at cheb node points.
 */
template <class T, class Y>
T cheb_approx(const T* fn_v, int cheb_deg, int dof, T* out){
  int d=cheb_deg+1;

  // Precompute
  sctl::Matrix<Y>* Mp=NULL;
  static std::vector<sctl::Matrix<Y> > precomp;
  #pragma omp critical(PVFMM_CHEB_APPROX)
  {
    if(precomp.size()<=(size_t)d){
      precomp .resize(d+1);
    }
    if(precomp [d].Dim(0)==0 && precomp [d].Dim(1)==0){
      std::vector<Y> x(d);
      for(int i=0;i<d;i++)
        x[i]=-sctl::cos<Y>((i+(T)0.5)*sctl::const_pi<T>()/d);

      std::vector<Y> p(d*d);
      cheb_poly(cheb_deg,&x[0],d,&p[0]);
      for(int i=d;i<d*d;i++)
        p[i]=p[i]*2;
      for(int i=0;i<d*d;i++)
        p[i]=p[i]/d;
      sctl::Matrix<Y> Mp1(d,d, sctl::Ptr2Itr<Y>(&p[0], (d)*(d)),false);
      sctl::Matrix<Y> Mp1_=Mp1.Transpose();
      precomp[d]=Mp1_;
    }
    Mp=&precomp[d];
  }

  // Create work buffers (per-thread scratch via sctl::ScratchBuf).
  size_t buff_size=dof*d*d*d;
  sctl::ScratchBuf<Y> buff_scratch(2*buff_size);
  sctl::Iterator<Y> buff1=buff_scratch.begin()+buff_size*0;
  sctl::Iterator<Y> buff2=buff_scratch.begin()+buff_size*1;

  sctl::Vector<Y> fn_v_in;
  if(SameType<T,Y>()()){ // Initialize fn_v_in
    fn_v_in.ReInit(d*d*d*dof,sctl::Ptr2Itr<Y>((Y*)fn_v,d*d*d*dof),false);
  }else{
    fn_v_in.ReInit(d*d*d*dof,buff1,false);
    for(sctl::Long i=0;i<fn_v_in.Dim();i++) fn_v_in[i]=fn_v[i];
  }

  { // Apply Mp along x-dimension
    sctl::Matrix<Y> Mi(dof*d*d,d, sctl::Ptr2Itr<Y>(&fn_v_in[0], (dof*d*d)*(d)),false);
    sctl::Matrix<Y> Mo(dof*d*d,d, buff2,false);
    Mo=Mi*(*Mp);

    MatrixTranspose<Y>(Mo.Dim(0),Mo.Dim(1),buff2,buff1);
  }
  { // Apply Mp along y-dimension
    sctl::Matrix<Y> Mi(d*dof*d,d, buff1,false);
    sctl::Matrix<Y> Mo(d*dof*d,d, buff2,false);
    Mo=Mi*(*Mp);

    MatrixTranspose<Y>(Mo.Dim(0),Mo.Dim(1),buff2,buff1);
  }
  { // Apply Mp along z-dimension
    sctl::Matrix<Y> Mi(d*d*dof,d, buff1,false);
    sctl::Matrix<Y> Mo(d*d*dof,d, buff2,false);
    Mo=Mi*(*Mp);

    MatrixTranspose<Y>(Mo.Dim(0),Mo.Dim(1),buff2,buff1);
  }

  { // Rearrange and write to out
    int indx=0;
    for(int l=0;l<dof;l++){
      for(int i=0;i<d;i++){
        for(int j=0;i+j<d;j++){
          Y* buff_ptr=&buff1[(j+i*d)*d*dof+l];
          for(int k=0;i+j+k<d;k++){
            out[indx]=buff_ptr[k*dof];
            indx++;
          }
        }
      }
    }
  }

  // buff is freed automatically by ScratchBuf destructor at scope exit.

  return cheb_err(out,cheb_deg,dof);
}

/**
 * \brief Returns the values of all legendre polynomials up to degree d,
 * evaluated at points in the input vector. Output format:
 * { P0[in[0]], ..., P0[in[n-1]], P1[in[0]], ..., P(d-1)[in[n-1]] }
 */
template <class T>
inline void legn_poly(int d, T* in, int n, T* out){
  if(d==0){
    for(int i=0;i<n;i++)
      out[i]=(sctl::fabs<T>(in[i])<=1?1.0:0);
  }else if(d==1){
    for(int i=0;i<n;i++){
      out[i]=(sctl::fabs<T>(in[i])<=1?1.0:0);
      out[i+n]=(sctl::fabs<T>(in[i])<=1?in[i]:0);
    }
  }else{
    for(int j=0;j<n;j++){
      T x=(sctl::fabs<T>(in[j])<=1?in[j]:0);
      T y0=(sctl::fabs<T>(in[j])<=1?1.0:0);
      out[j]=y0;
      out[j+n]=x;

      T y1=x;
      T* y2=&out[2*n+j];
      for(int i=2;i<=d;i++){
        *y2=( (2*i-1)*x*y1-(i-1)*y0 )/i;
        y0=y1;
        y1=*y2;
        y2=&y2[n];
      }
    }
  }
}

/**
 * \brief Computes Legendre-Gauss-Lobatto nodes and weights.
 */
template <class T>
void gll_quadrature(int deg, T* x_, T* w){//*
  T eps=sctl::machine_eps<T>()*64;
  int d=deg+1;
  assert(d>1);
  int N=deg;

  sctl::Vector<T> x(d,x_,false);
  for(int i=0;i<d;i++)
    x[i]=-sctl::cos<T>((sctl::const_pi<T>()*i)/N);
  sctl::Matrix<T> P(d,d); P.SetZero();

  T err=1;
  sctl::Vector<T> xold(d);
  while(err>eps){
    xold=x;
    for(int i=0;i<d;i++){
      P[i][0]=1;
      P[i][1]=x[i];
    }
    for(int k=2;k<=N;k++)
      for(int i=0;i<d;i++)
        P[i][k]=( (2*k-1)*x[i]*P[i][k-1]-(k-1)*P[i][k-2] )/k;
    err=0;
    for(int i=0;i<d;i++){
      T dx=-( x[i]*P[i][N]-P[i][N-1] )/( d*P[i][N] );
      err=(err<sctl::fabs<T>(dx)?sctl::fabs<T>(dx):err);
      x[i]=xold[i]+dx;
    }
  }
  for(int i=0;i<d;i++)
    w[i]=2.0/(N*d*P[i][N]*P[i][N]);
}

/**
 * \brief Computes Chebyshev approximation from function values at GLL points.
 */
template <class T, class Y>
T gll2cheb(T* fn_v, int deg, int dof, T* out){//*
  //T eps=sctl::machine_eps<T>()*64;

  int d=deg+1;
  static std::vector<sctl::Matrix<Y> > precomp;
  static std::vector<sctl::Matrix<Y> > precomp_;
  sctl::Matrix<Y>* Mp ;
  sctl::Matrix<Y>* Mp_;
  #pragma omp critical(PVFMM_GLL_TO_CHEB)
  {
    if(precomp.size()<=(size_t)d){
      precomp .resize(d+1);
      precomp_.resize(d+1);

      std::vector<Y> x(d); //Cheb nodes.
      for(int i=0;i<d;i++)
        x[i]=-sctl::cos<Y>((i+(T)0.5)*sctl::const_pi<Y>()/d);

      sctl::Vector<T> w(d);
      sctl::Vector<T> x_legn(d); // GLL nodes.
      gll_quadrature(deg, &x_legn[0], &w[0]);

      sctl::Matrix<T> P(d,d); //GLL node 2 GLL coeff.
      legn_poly(deg,&x_legn[0],d,&P[0][0]);
      for(int i=0;i<d;i++)
        for(int j=0;j<d;j++)
          P[i][j]*=w[j]*0.5*(i<deg?(2*i+1):(i));

      sctl::Matrix<T> M_gll2cheb(d,d); //GLL coeff 2 cheb node.
      legn_poly(deg,&x[0],d,&M_gll2cheb[0][0]);

      sctl::Matrix<T> M_g2c; //GLL node to cheb node.
      M_g2c=M_gll2cheb.Transpose()*P;

      std::vector<Y> p(d*d);
      cheb_poly(deg,&x[0],d,&p[0]);
      for(int i=0;i<d*d;i++)
        p[i]=p[i]*2.0/d;
      sctl::Matrix<Y> Mp1(d,d, sctl::Ptr2Itr<Y>(&p[0], (d)*(d)),false);
      Mp1=Mp1*M_g2c;

      sctl::Matrix<Y> Mp1_=Mp1.Transpose();
      precomp [d]=Mp1 ;
      precomp_[d]=Mp1_;
    }
    Mp =&precomp [d];
    Mp_=&precomp_[d];
  }

  std::vector<Y> fn_v0(d*d*d*dof);
  std::vector<Y> fn_v1(d*d*d);
  std::vector<Y> fn_v2(d*d*d);
  std::vector<Y> fn_v3(d*d*d);

  for(size_t i=0;i<(size_t)(d*d*d*dof);i++)
    fn_v0[i]=fn_v[i];

  int indx=0;
  for(int l=0;l<dof;l++){
    {
      sctl::Matrix<Y> M0(d*d,d, sctl::Ptr2Itr<Y>(&fn_v0[d*d*d*l], (d*d)*(d)),false);
      sctl::Matrix<Y> M1(d*d,d, sctl::Ptr2Itr<Y>(&fn_v1[0], (d*d)*(d)),false);
      M1=M0*(*Mp_);
    }
    {
      sctl::Matrix<Y> M0(d,d*d, sctl::Ptr2Itr<Y>(&fn_v1[0], (d)*(d*d)),false);
      sctl::Matrix<Y> M1(d,d*d, sctl::Ptr2Itr<Y>(&fn_v2[0], (d)*(d*d)),false);
      M1=(*Mp)*M0;
    }
    for(int i=0;i<d;i++){
      sctl::Matrix<Y> M0(d,d, sctl::Ptr2Itr<Y>(&fn_v2[d*d*i], (d)*(d)),false);
      sctl::Matrix<Y> M1(d,d, sctl::Ptr2Itr<Y>(&fn_v3[d*d*i], (d)*(d)),false);
      M1=(*Mp)*M0;
    }

    for(int i=0;i<d;i++)
      for(int j=0;j<d;j++){
        fn_v3[i*d+j*d*d]/=2.0;
        fn_v3[i+j*d*d]/=2.0;
        fn_v3[i+j*d]/=2.0;
      }
    Y sum=0;
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      sum+=sctl::fabs<T>(fn_v3[k+(j+i*d)*d]);
    }
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      out[indx]=fn_v3[k+(j+i*d)*d];
      //if(sctl::fabs<T>(out[indx])<eps*sum) out[indx]=0;
      indx++;
    }
  }

  return cheb_err(out,deg,dof);
}

/**
 * \brief Computes Chebyshev approximation from the input function pointer.
 */
template <class T>
T cheb_approx(T (*fn)(T,T,T), int cheb_deg, T* coord, T s, std::vector<T>& out){
  int d=cheb_deg+1;
  std::vector<T> x(d);
  for(int i=0;i<d;i++)
    x[i]=sctl::cos<T>((i+(T)0.5)*sctl::const_pi<T>()/d);

  std::vector<T> p;
  cheb_poly(cheb_deg,&x[0],d,&p[0]);

  std::vector<T> x1(d);
  std::vector<T> x2(d);
  std::vector<T> x3(d);
  for(int i=0;i<d;i++){
    x1[i]=(x[i]+1.0)/2.0*s+coord[0];
    x2[i]=(x[i]+1.0)/2.0*s+coord[1];
    x3[i]=(x[i]+1.0)/2.0*s+coord[2];
  }

  std::vector<T> fn_v(d*d*d);
  T* fn_p=&fn_v[0];
  for(int i=0;i<d;i++){
    for(int j=0;j<d;j++){
      for(int k=0;k<d;k++){
        *fn_p=fn(x3[k],x2[j],x1[i]);
        fn_p++;
      }
    }
  }

  out.resize((d*(d+1)*(d+2))/6);
  return cheb_approx(&fn_v[0], cheb_deg, 1, &out[0]);
}

/**
 * \brief Evaluates polynomial values from input coefficients at points on
 * a regular grid defined by in_x, in_y, in_z the values in the input vector.
 */
template <class T>
void cheb_eval(const sctl::Vector<T>& coeff_, int cheb_deg, const std::vector<T>& in_x, const std::vector<T>& in_y, const std::vector<T>& in_z, sctl::Vector<T>& out){
  size_t d=(size_t)cheb_deg+1;
  size_t n_coeff=(d*(d+1)*(d+2))/6;
  size_t dof=coeff_.Dim()/n_coeff;
  assert((size_t)coeff_.Dim()==dof*n_coeff);

  // Resize out
  size_t n1=in_x.size();
  size_t n2=in_y.size();
  size_t n3=in_z.size();
  if((size_t)out.Dim()!=(size_t)(n1*n2*n3*dof)) out.ReInit(n1*n2*n3*dof);
  if(n1==0 || n2==0 || n3==0) return;

  // Precomputation
  std::vector<T> p1(n1*d);
  std::vector<T> p2(n2*d);
  std::vector<T> p3(n3*d);
  cheb_poly(cheb_deg,&in_x[0],n1,&p1[0]);
  cheb_poly(cheb_deg,&in_y[0],n2,&p2[0]);
  cheb_poly(cheb_deg,&in_z[0],n3,&p3[0]);
  sctl::Matrix<T> Mp1(d,n1, sctl::Ptr2Itr<T>(&p1[0], (d)*(n1)),false);
  sctl::Matrix<T> Mp2(d,n2, sctl::Ptr2Itr<T>(&p2[0], (d)*(n2)),false);
  sctl::Matrix<T> Mp3(d,n3, sctl::Ptr2Itr<T>(&p3[0], (d)*(n3)),false);

  // Create work buffers (per-thread scratch). v1 and v2 are half-and-half
  // views into the single ScratchBuf.
  size_t buff_size=std::max(d,n1)*std::max(d,n2)*std::max(d,n3)*dof;
  sctl::ScratchBuf<T> buff_scratch(2*buff_size);
  sctl::Iterator<T> v1 = buff_scratch.begin() + buff_size*0;
  sctl::Iterator<T> v2 = buff_scratch.begin() + buff_size*1;

  { // Rearrange coefficients into a tensor.
    std::memset(&v1[0], 0, d*d*d*dof*sizeof(T));
    size_t indx=0;
    for(size_t l=0;l<dof;l++){
      for(size_t i=0;i<d;i++){
        for(size_t j=0;i+j<d;j++){
          T* coeff_ptr=&v1[(j+(i+l*d)*d)*d];
          for(size_t k=0;i+j+k<d;k++){
            coeff_ptr[k]=coeff_[indx];
            indx++;
          }
        }
      }
    }
  }

  { // Apply Mp1
    sctl::Matrix<T> Mi  ( d* d*dof, d, v1,false);
    sctl::Matrix<T> Mo  ( d* d*dof,n1, v2,false);
    sctl::Matrix<T>::GEMM(Mo, Mi, Mp1);

    MatrixTranspose<T>(Mo.Dim(0),Mo.Dim(1),v2,v1);
  }
  { // Apply Mp2
    sctl::Matrix<T> Mi  (n1* d*dof, d, v1,false);
    sctl::Matrix<T> Mo  (n1* d*dof,n2, v2,false);
    sctl::Matrix<T>::GEMM(Mo, Mi, Mp2);

    MatrixTranspose<T>(Mo.Dim(0),Mo.Dim(1),v2,v1);
  }
  { // Apply Mp3
    sctl::Matrix<T> Mi  (n2*n1*dof, d, v1,false);
    sctl::Matrix<T> Mo  (n2*n1*dof,n3, v2,false);
    sctl::Matrix<T>::GEMM(Mo, Mi, Mp3);

    MatrixTranspose<T>(Mo.Dim(0),Mo.Dim(1),v2,v1);
  }

  { // Copy to out
    sctl::Matrix<T> Mo  ( n3*n2*n1,dof, v1,false);
    MatrixTranspose<T>(Mo.Dim(0),Mo.Dim(1),v1,out.begin());
  }

  // buff_scratch freed automatically at scope exit.
}

/**
 * \brief Evaluates polynomial values from input coefficients at points
 * in the coord vector.
 */
template <class T>
inline void cheb_eval(sctl::Vector<T>& coeff_, int cheb_deg, std::vector<T>& coord, sctl::Vector<T>& out){
  if(!coord.size()) return;
  int dim=3;
  int d=cheb_deg+1;
  int n=coord.size()/dim;
  int dof=coeff_.Dim()/((d*(d+1)*(d+2))/6);
  assert((size_t)coeff_.Dim()==(size_t)(d*(d+1)*(d+2)*dof)/6);

  std::vector<T> coeff(d*d*d*dof);
  {// Rearrange data
    int indx=0;
    for(int l=0;l<dof;l++)
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      coeff[(k+(j+(i+l*d)*d)*d)]=coeff_[indx];
      indx++;
    }
  }

  sctl::Matrix<T> coord_(n,dim, sctl::Ptr2Itr<T>(&coord[0], (n)*(dim)));
  coord_=coord_.Transpose();

  sctl::Matrix<T> px(d,n);
  sctl::Matrix<T> py(d,n);
  sctl::Matrix<T> pz(d,n);
  cheb_poly(cheb_deg,&(coord_[0][0]),n,&(px[0][0]));
  cheb_poly(cheb_deg,&(coord_[1][0]),n,&(py[0][0]));
  cheb_poly(cheb_deg,&(coord_[2][0]),n,&(pz[0][0]));

  sctl::Matrix<T> M_coeff0(d*d*dof, d, sctl::Ptr2Itr<T>(&coeff[0], (d*d*dof)*(d)), false);
  sctl::Matrix<T> M0 = (M_coeff0 * px).Transpose(); // {n, dof*d*d}

  py = py.Transpose();
  pz = pz.Transpose();
  if((size_t)out.Dim()!=(size_t)(n*dof)) out.ReInit(n*dof);
  for(int i=0; i<n; i++)
    for(int j=0; j<dof; j++){
      sctl::Matrix<T> M0_  (d, d, sctl::Ptr2Itr<T>(&(M0[i][  j*d*d]), (d)*(d)), false);
      sctl::Matrix<T> py_  (d, 1, sctl::Ptr2Itr<T>(&(py[i][      0]), (d)*(1)), false);
      sctl::Matrix<T> pz_  (1, d, sctl::Ptr2Itr<T>(&(pz[i][      0]), (1)*(d)), false);

      sctl::Matrix<T> M_out(1, 1, sctl::Ptr2Itr<T>(&(  out[i*dof+j]), (1)*(1)), false);
      M_out += pz_ * M0_ * py_;
    }
}

/**
 * \brief Returns the values of all Chebyshev basis functions of degree up to d
 * evaluated at the point coord.
 */
template <class T>
inline void cheb_eval(int cheb_deg, T* coord, T* coeff0,T* buff){
  int d=cheb_deg+1;
  std::vector<T> coeff(d*d*d);

  T* p=&buff[0];
  T* p_=&buff[3*d];
  cheb_poly(cheb_deg,&coord[0],3,&p[0]);

  for(int i=0;i<d;i++){
    p_[i]=p[i*3];
    p_[i+d]=p[i*3+1];
    p_[i+2*d]=p[i*3+2];
  }
  T* coeff_=&buff[2*3*d];

  sctl::Matrix<T> v_p0    (1, d, sctl::Ptr2Itr<T>(&    p_[0], (1)*(d)),false);
  sctl::Matrix<T> v_p1    (d, 1, sctl::Ptr2Itr<T>(&    p_[d], (d)*(1)),false);
  sctl::Matrix<T> M_coeff_(d, d, sctl::Ptr2Itr<T>(&coeff_[0], (d)*(d)),false);
  M_coeff_ = v_p1 * v_p0; // */
  //mat::gemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,d,d,1,1.0,&p_[d],1,&p_[0],d,0.0,&coeff_[0],d);

  sctl::Matrix<T> v_p2    (d,   1, sctl::Ptr2Itr<T>(&    p_[2*d], (d)*(1)),false);
  sctl::Matrix<T> v_coeff_(1, d*d, sctl::Ptr2Itr<T>(&coeff_[  0], (1)*(d*d)),false);
  sctl::Matrix<T> M_coeff (d, d*d, sctl::Ptr2Itr<T>(&coeff [  0], (d)*(d*d)),false);
  M_coeff = v_p2 * v_coeff_; // */
  //mat::gemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,d,d*d,1,1.0,&p_[2*d],1,&coeff_[0],d*d,0.0,&coeff[0],d*d);

  {// Rearrange data
    int indx=0;
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      coeff0[indx]=coeff[(k+(j+i*d)*d)];
      indx++;
    }
  }
}

/**
 * \brief Computes a least squares solution for Chebyshev approximation over a
 * cube from point samples.
 * \param[in] deg Maximum degree of the polynomial.
 * \param[in] coord Coordinates of points (x,y,z interleaved).
 * \param[in] node_coord Coordinates of the octant.
 * \param[in] node_size Length of the side of the octant.
 * \param[out] cheb_coeff Output coefficients.
 */
template <class T>
void points2cheb(int deg, T* coord, T* val, int n, int dim, T* node_coord, T node_size, sctl::Vector<T>& cheb_coeff){
  if(n==0) return;
  int deg_=((int)(sctl::pow<T>(n*6,1.0/3.0)+0.5))/2;
  deg_=(deg_>deg?deg:deg_);
  deg_=(deg_>0?deg_:1);
  int deg3=((deg_+1)*(deg_+2)*(deg_+3))/6;
  if((size_t)cheb_coeff.Dim()!=(size_t)(dim*((deg+1)*(deg+2)*(deg+3))/6)) cheb_coeff.ReInit(dim*((deg+1)*(deg+2)*(deg+3))/6);
  cheb_coeff.SetZero();

  //Map coordinates to unit cube
  std::vector<T> coord_(n*3);
  for(int i=0;i<n;i++){
    coord_[i*3  ]=(coord[i*3  ]-node_coord[0])*2/node_size-1;
    coord_[i*3+1]=(coord[i*3+1]-node_coord[1])*2/node_size-1;
    coord_[i*3+2]=(coord[i*3+2]-node_coord[2])*2/node_size-1;
  }

  //Compute the matrix M
  sctl::Matrix<T> M(n,deg3);
  std::vector<T> buff((deg_+1)*(deg_+1+3*2));
  for(int i=0;i<n;i++)
    cheb_eval(deg_,&coord_[i*3],&(M[i][0]),&buff[0]);

  //Compute the pinv and get the cheb_coeff.
  sctl::Matrix<T> M_val(n,dim, sctl::Ptr2Itr<T>(&val[0], (n)*(dim)));
  T eps=sctl::machine_eps<T>()*64;
  sctl::Matrix<T> cheb_coeff_=(M.pinv(eps)*M_val).Transpose();

  //Set the output
  int indx=0;
  int indx1=0;
  for(int l=0;l<dim;l++)
  for(int i=0;i    <=deg;i++)
  for(int j=0;i+j  <=deg;j++)
  for(int k=0;i+j+k<=deg;k++){
    if(i+j+k<=deg_){
      cheb_coeff[indx]=cheb_coeff_[0][indx1];
      indx1++;
    }else{
      cheb_coeff[indx]=0;
    }
    indx++;
  }
}

template <class T>
std::pair<const sctl::Vector<T>&, const sctl::Vector<T>&> quad_rule(int n){
  static constexpr int QUAD_RULE_MAX_ORDER = 10000;
  assert(n < QUAD_RULE_MAX_ORDER);

  static std::vector<sctl::Vector<T> > x_lst(QUAD_RULE_MAX_ORDER);
  static std::vector<sctl::Vector<T> > w_lst(QUAD_RULE_MAX_ORDER);
  static std::atomic<bool> ready[QUAD_RULE_MAX_ORDER]; // static storage => zero-initialized
  static std::mutex mtx;

  if(!ready[n].load(std::memory_order_acquire)){ // atomic double-checked locking
    std::lock_guard<std::mutex> lock(mtx);
    if(!ready[n].load(std::memory_order_relaxed)){
      sctl::Vector<T>& x_=x_lst[n];
      sctl::Vector<T>& w_=w_lst[n];
      sctl::LegQuadRule<T>::ComputeNdsWts(&x_, &w_, n); // Gauss-Legendre nodes/weights on [0,1]
      ready[n].store(true, std::memory_order_release);
    }
  }
  return {x_lst[n], w_lst[n]};
}

template <class T>
std::vector<T> integ_pyramid(int m, T* s, T r, int nx, const Kernel<T>& kernel, int* perm){//*
  int ny=nx;
  int nz=nx;

  T eps=sctl::machine_eps<T>()*64;
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];

  std::vector<T> qp_x(nx);
  std::vector<T> qp_y(ny);
  std::vector<T> qp_z(nz);

  // Nodes/weights depend only on nx/ny/nz; fetch the cached rules once.
  const auto [nds_x, wts_x] = quad_rule<T>(nx);
  const auto [nds_y, wts_y] = quad_rule<T>(ny);
  const auto [nds_z, wts_z] = quad_rule<T>(nz);
  std::vector<T> p_x(nx*m);
  std::vector<T> p_y(ny*m);
  std::vector<T> p_z(nz*m);

  std::vector<T> x_;
  { //  Build stack along X-axis.
    x_.push_back(s[0]);
    x_.push_back(sctl::fabs<T>(1-s[0])+s[0]);
    x_.push_back(sctl::fabs<T>(1-s[1])+s[0]);
    x_.push_back(sctl::fabs<T>(1+s[1])+s[0]);
    x_.push_back(sctl::fabs<T>(1-s[2])+s[0]);
    x_.push_back(sctl::fabs<T>(1+s[2])+s[0]);
    std::sort(x_.begin(),x_.end());
    for(size_t i=0;i<x_.size();i++){
      if(x_[i]<-1.0) x_[i]=-1.0;
      if(x_[i]> 1.0) x_[i]= 1.0;
    }

    std::vector<T> x_new;
    T x_jump=sctl::fabs<T>(1-s[0]);
    if(sctl::fabs<T>(1-s[1])>eps) x_jump=std::min(x_jump,(T)sctl::fabs<T>(1-s[1]));
    if(sctl::fabs<T>(1+s[1])>eps) x_jump=std::min(x_jump,(T)sctl::fabs<T>(1+s[1]));
    if(sctl::fabs<T>(1-s[2])>eps) x_jump=std::min(x_jump,(T)sctl::fabs<T>(1-s[2]));
    if(sctl::fabs<T>(1+s[2])>eps) x_jump=std::min(x_jump,(T)sctl::fabs<T>(1+s[2]));
    for(size_t k=0; k<x_.size()-1; k++){
      T x0=x_[k];
      T x1=x_[k+1];

      T A0=0;
      T A1=0;
      { // A0
        T y0=s[1]-(x0-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
        T y1=s[1]+(x0-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
        T z0=s[2]-(x0-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
        T z1=s[2]+(x0-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;
        A0=(y1-y0)*(z1-z0);
      }
      { // A1
        T y0=s[1]-(x1-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
        T y1=s[1]+(x1-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
        T z0=s[2]-(x1-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
        T z1=s[2]+(x1-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;
        A1=(y1-y0)*(z1-z0);
      }
      T V=(T)0.5*(A0+A1)*(x1-x0);
      if(V<eps) continue;

      if(!x_new.size()) x_new.push_back(x0);
      x_jump=std::max(x_jump,x0-s[0]);
      while(s[0]+x_jump*1.5<x1){
        x_new.push_back(s[0]+x_jump);
        x_jump*=2;
      }
      if(x_new.back()+eps<x1) x_new.push_back(x1);
    }
    assert(x_new.size()<30);
    x_.swap(x_new);
  }

  // Per-thread scratch (was previously backed by a function-scoped static
  // MemoryManager of 16·sizeof(T) MB per template instantiation; sctl's
  // ScratchPool serves the same role thread-locally and grows on demand).
  sctl::ScratchBuf<T> k_out(   ny*nz*k_dim);
  sctl::ScratchBuf<T> I0   (   ny*m *k_dim);
  sctl::ScratchBuf<T> I1   (   m *m *k_dim);
  sctl::ScratchBuf<T> I2   (m *m *m *k_dim);
  std::memset(&I2[0], 0, m*m*m*k_dim*sizeof(T));
  if(x_.size()>1)
  for(size_t k=0; k<x_.size()-1; k++){
    T x0=x_[k];
    T x1=x_[k+1];

    { // Set qp_x
      for(int i=0; i<nx; i++)
        qp_x[i]=x0+(x1-x0)*nds_x[i];
    }
    cheb_poly(m-1,&qp_x[0],nx,&p_x[0]);

    for(int i=0; i<nx; i++){
      T y0=s[1]-(qp_x[i]-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
      T y1=s[1]+(qp_x[i]-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
      T z0=s[2]-(qp_x[i]-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
      T z1=s[2]+(qp_x[i]-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;

      { // Set qp_y
        for(int j=0; j<ny; j++)
          qp_y[j]=y0+(y1-y0)*nds_y[j];
      }
      { // Set qp_z
        for(int j=0; j<nz; j++)
          qp_z[j]=z0+(z1-z0)*nds_z[j];
      }
      cheb_poly(m-1,&qp_y[0],ny,&p_y[0]);
      cheb_poly(m-1,&qp_z[0],nz,&p_z[0]);
      { // k_out =  kernel x qw
        T src[3]={0,0,0};
        std::vector<T> trg(ny*nz*3);
        for(int i0=0; i0<ny; i0++){
          size_t indx0=i0*nz*3;
          for(int i1=0; i1<nz; i1++){
            size_t indx1=indx0+i1*3;
            trg[indx1+perm[0]]=(s[0]-qp_x[i ])*r*(T)0.5*perm[1];
            trg[indx1+perm[2]]=(s[1]-qp_y[i0])*r*(T)0.5*perm[3];
            trg[indx1+perm[4]]=(s[2]-qp_z[i1])*r*(T)0.5*perm[5];
          }
        }
        {
          sctl::Matrix<T> k_val(ny*nz*kernel.ker_dim[0],kernel.ker_dim[1]);
          kernel.BuildMatrix(&src[0],1,&trg[0],ny*nz,&k_val[0][0]);
          MatrixTranspose<T>(ny*nz*kernel.ker_dim[0],kernel.ker_dim[1],(sctl::ConstIterator<T>)k_val[0],k_out.begin());
        }
        for(int kk=0; kk<k_dim; kk++){
          for(int i0=0; i0<ny; i0++){
            size_t indx=(kk*ny+i0)*nz;
            for(int i1=0; i1<nz; i1++){
              k_out[indx+i1] *= wts_y[i0]*wts_z[i1];
            }
          }
        }
      }

      std::memset(&I0[0], 0, ny*m*k_dim*sizeof(T));
      for(int kk=0; kk<k_dim; kk++){
        for(int i0=0; i0<ny; i0++){
          size_t indx0=(kk*ny+i0)*nz;
          size_t indx1=(kk*ny+i0)* m;
          for(int i2=0; i2<m; i2++){
            for(int i1=0; i1<nz; i1++){
              I0[indx1+i2] += k_out[indx0+i1]*p_z[i2*nz+i1];
            }
          }
        }
      }

      std::memset(&I1[0], 0, m*m*k_dim*sizeof(T));
      for(int kk=0; kk<k_dim; kk++){
        for(int i2=0; i2<ny; i2++){
          size_t indx0=(kk*ny+i2)*m;
          for(int i0=0; i0<m; i0++){
            size_t indx1=(kk* m+i0)*m;
            T py=p_y[i0*ny+i2];
            for(int i1=0; i0+i1<m; i1++){
              I1[indx1+i1] += I0[indx0+i1]*py;
            }
          }
        }
      }

      T v=(x1-x0)*(y1-y0)*(z1-z0);
      for(int kk=0; kk<k_dim; kk++){
        for(int i0=0; i0<m; i0++){
          T px=p_x[i+i0*nx]*wts_x[i]*v;
          for(int i1=0; i0+i1<m; i1++){
            size_t indx0= (kk*m+i1)*m;
            size_t indx1=((kk*m+i0)*m+i1)*m;
            for(int i2=0; i0+i1+i2<m; i2++){
              I2[indx1+i2] += I1[indx0+i2]*px;
            }
          }
        }
      }
    }
  }
  for(int i=0;i<m*m*m*k_dim;i++)
    I2[i]=I2[i]*r*r*r/8;

  if(x_.size()>1)
  sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, ( 2*ny*nz*m*k_dim
                     +ny*m*(m+1)*k_dim
                     +2*m*(m+1)*k_dim
                     +m*(m+1)*(m+2)/3*k_dim)*nx*(x_.size()-1));

  std::vector<T> I2_(&I2[0], &I2[0]+I2.Dim());  // I2.Dim() == m*m*m*k_dim
  // k_out, I0, I1, I2 freed automatically by ScratchBuf destructors in reverse
  // declaration order at scope exit.
  return I2_;
}

template <class T>
std::vector<T> integ(int m, T* s, T r, int n, const Kernel<T>& kernel){//*
  //Compute integrals over pyramids in all directions.
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];
  T s_[3];
  s_[0]=s[0]*2/r-1;
  s_[1]=s[1]*2/r-1;
  s_[2]=s[2]*2/r-1;

  T s1[3];
  int perm[6];
  std::vector<T> U_[6];

  s1[0]= s_[0];s1[1]=s_[1];s1[2]=s_[2];
  perm[0]= 0; perm[2]= 1; perm[4]= 2;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[0]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]=-s_[0];s1[1]=s_[1];s1[2]=s_[2];
  perm[0]= 0; perm[2]= 1; perm[4]= 2;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[1]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]= s_[1];s1[1]=s_[0];s1[2]=s_[2];
  perm[0]= 1; perm[2]= 0; perm[4]= 2;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[2]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]=-s_[1];s1[1]=s_[0];s1[2]=s_[2];
  perm[0]= 1; perm[2]= 0; perm[4]= 2;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[3]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]= s_[2];s1[1]=s_[0];s1[2]=s_[1];
  perm[0]= 2; perm[2]= 0; perm[4]= 1;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[4]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]=-s_[2];s1[1]=s_[0];s1[2]=s_[1];
  perm[0]= 2; perm[2]= 0; perm[4]= 1;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[5]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  std::vector<T> U; U.assign(m*m*m*k_dim,0);
  for(int kk=0; kk<k_dim; kk++){
    for(int i=0;i<m;i++){
      for(int j=0;j<m;j++){
        for(int k=0;k<m;k++){
          U[kk*m*m*m + k*m*m + j*m + i]+=U_[0][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + k*m*m + j*m + i]+=U_[1][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1:1);
        }
      }
    }
  }

  for(int kk=0; kk<k_dim; kk++){
    for(int i=0; i<m; i++){
      for(int j=0; j<m; j++){
        for(int k=0; k<m; k++){
          U[kk*m*m*m + k*m*m + i*m + j]+=U_[2][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + k*m*m + i*m + j]+=U_[3][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1:1);
        }
      }
    }
  }

  for(int kk=0; kk<k_dim; kk++){
    for(int i=0; i<m; i++){
      for(int j=0; j<m; j++){
        for(int k=0; k<m; k++){
          U[kk*m*m*m + i*m*m + k*m + j]+=U_[4][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + i*m*m + k*m + j]+=U_[5][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1:1);
        }
      }
    }
  }

  return U;
}

/**
 * \brief
 * \param[in] r Length of the side of cubic region.
 */
template <class T>
std::vector<T> cheb_integ(int m, T* s_, T r_, const Kernel<T>& kernel){
  T eps=sctl::machine_eps<T>();

  T r=r_;
  T s[3]={s_[0],s_[1],s_[2]};

  int n=m+2;
  T err=1.0;
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];
  std::vector<T> U=integ<T>(m+1,s,r,n,kernel);
  std::vector<T> U_;
  while(err>eps*n){
    n=(int)round(n*1.3);
    if(n>300){
      std::cout<<"Cheb_Integ::Failed to converge.[";
      std::cout<<((double)err )<<",";
      std::cout<<((double)s[0])<<",";
      std::cout<<((double)s[1])<<",";
      std::cout<<((double)s[2])<<"]\n";
      break;
    }
    U_=integ<T>(m+1,s,r,n,kernel);
    err=0;
    for(int i=0;i<(m+1)*(m+1)*(m+1)*k_dim;i++)
      if(sctl::fabs<T>(U[i]-U_[i])>err)
        err=sctl::fabs<T>(U[i]-U_[i]);
    U=U_;
  }

  std::vector<T> U0(((m+1)*(m+2)*(m+3)*k_dim)/6);
  {// Rearrange data
    int indx=0;
    const int* ker_dim=kernel.ker_dim;
    for(int l0=0;l0<ker_dim[0];l0++)
    for(int l1=0;l1<ker_dim[1];l1++)
    for(int i=0;i<=m;i++)
    for(int j=0;i+j<=m;j++)
    for(int k=0;i+j+k<=m;k++){
      U0[indx]=U[(k+(j+(i+(l0*ker_dim[1]+l1)*(m+1))*(m+1))*(m+1))];
      indx++;
    }
  }

  return U0;
}

template <class T>
std::vector<T> cheb_nodes(int deg, int dim){
  int d=deg+1;
  std::vector<T> x(d);
  for(int i=0;i<d;i++)
    x[i]=-sctl::cos<T>((i+(T)0.5)*sctl::const_pi<T>()/d)*(T)0.5+(T)0.5;
  if(dim==1) return x;

  int n1=sctl::pow<int>(d,dim);
  std::vector<T> y(n1*dim);
  for(int i=0;i<dim;i++){
    int n2=sctl::pow<int>(d,i);
    for(int j=0;j<n1;j++){
      y[j*dim+i]=x[(j/n2)%d];
    }
  }
  return y;
}


template <class T>
void cheb_diff(const sctl::Vector<T>& A, int deg, int diff_dim, sctl::Vector<T>& B){
  size_t d=deg+1;

  // Precompute
  static sctl::Matrix<T> M;
  #pragma omp critical(PVFMM_CHEB_DIFF1)
  if((size_t)M.Dim(0)!=(size_t)d){
    M.ReInit(d,d);
    for(size_t i=0;i<d;i++){
      for(size_t j=0;j<d;j++) M[j][i]=0;
      for(size_t j=1-(i%2);j<i;j=j+2){
        M[j][i]=2*i*2;
      }
      if(i%2==1) M[0][i]-=i*2;
    }
  }

  // Create work buffers (per-thread scratch).
  size_t buff_size=A.Dim();
  sctl::ScratchBuf<T> buff_scratch(2*buff_size);
  T* buff=&buff_scratch.begin()[0];
  T* buff1=buff+buff_size*0;
  T* buff2=buff+buff_size*1;

  size_t n1=sctl::pow<unsigned int>(d,diff_dim);
  size_t n2=A.Dim()/(n1*d);

  for(size_t k=0;k<n2;k++){ // Rearrange A to make diff_dim the last array dimension
    sctl::Matrix<T> Mi(d,       n1, sctl::Ptr2Itr<T>((T*)&A[d*n1*k], (d)*(n1)),false);
    sctl::Matrix<T> Mo(d,A.Dim()/d, sctl::Ptr2Itr<T>(&buff1[  n1*k], (d)*(A.Dim()/d)),false);
    for(size_t i=0;i< d;i++)
    for(size_t j=0;j<n1;j++){
      Mo[i][j]=Mi[i][j];
    }
  }

  { // Apply M
    sctl::Matrix<T> Mi(d,A.Dim()/d, sctl::Ptr2Itr<T>(&buff1[0], (d)*(A.Dim()/d)),false);
    sctl::Matrix<T> Mo(d,A.Dim()/d, sctl::Ptr2Itr<T>(&buff2[0], (d)*(A.Dim()/d)),false);
    sctl::Matrix<T>::GEMM(Mo, M, Mi);
  }

  for(size_t k=0;k<n2;k++){ // Rearrange and write output to B
    sctl::Matrix<T> Mi(d,A.Dim()/d, sctl::Ptr2Itr<T>(&buff2[  n1*k], (d)*(A.Dim()/d)),false);
    sctl::Matrix<T> Mo(d,       n1, sctl::Ptr2Itr<T>(&B[d*n1*k], (d)*(n1)),false);
    for(size_t i=0;i< d;i++)
    for(size_t j=0;j<n1;j++){
      Mo[i][j]=Mi[i][j];
    }
  }

  // buff is freed automatically at scope exit.
}

template <class T>
void cheb_grad(const sctl::Vector<T>& A, int deg, sctl::Vector<T>& B){
  size_t dim=3;
  size_t d=(size_t)deg+1;
  size_t n_coeff =(d*(d+1)*(d+2))/6;
  size_t n_coeff_=sctl::pow<unsigned int>(d,dim);
  size_t dof=A.Dim()/n_coeff;

  // Create work buffers (per-thread scratch).
  sctl::ScratchBuf<T> buff_scratch(2*n_coeff_*dof);
  T* buff=&buff_scratch.begin()[0];
  sctl::Vector<T> A_(n_coeff_*dof,sctl::Ptr2Itr<T>(buff+n_coeff_*dof*0,n_coeff_*dof),false); A_.SetZero();
  sctl::Vector<T> B_(n_coeff_*dof,sctl::Ptr2Itr<T>(buff+n_coeff_*dof*1,n_coeff_*dof),false); B_.SetZero();

  {// Rearrange data
    size_t indx=0;
    for(size_t l=0;l<dof;l++){
      for(size_t i=0;i<d;i++){
        for(size_t j=0;i+j<d;j++){
          T* A_ptr=&A_[(j+(i+l*d)*d)*d];
          for(size_t k=0;i+j+k<d;k++){
            A_ptr[k]=A[indx];
            indx++;
          }
        }
      }
    }
  }

  if((size_t)B.Dim()!=(size_t)(A.Dim()*dim)) B.ReInit(A.Dim()*dim);
  for(size_t q=0;q<dim;q++){
    // Compute derivative in direction q
    cheb_diff(A_,deg,q,B_);

    for(size_t l=0;l<dof;l++){// Rearrange data
      size_t indx=(q+l*dim)*n_coeff;
      for(size_t i=0;i<d;i++){
        for(size_t j=0;i+j<d;j++){
          T* B_ptr=&B_[(j+(i+l*d)*d)*d];
          for(size_t k=0;i+j+k<d;k++){
            B[indx]=B_ptr[k];
            indx++;
          }
        }
      }
    }
  }

  // buff is freed automatically at scope exit.
}

template <class T>
void cheb_div(T* A_, int deg, T* B_){
  int dim=3;
  int d=deg+1;
  int n1 =sctl::pow<unsigned int>(d,dim);
  sctl::Vector<T> A(n1*dim); A.SetZero();
  sctl::Vector<T> B(n1    ); B.SetZero();

  {// Rearrange data
    int indx=0;
    for(int l=0;l<dim;l++)
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      A[k+(j+(i+l*d)*d)*d]=A_[indx];
      indx++;
    }
  }
  sctl::Matrix<T> MB(n1,1, sctl::Ptr2Itr<T>(&B[0], (n1)*(1)),false);
  sctl::Matrix<T> MC(n1,1);
  for(int i=0;i<3;i++){
    {
      sctl::Vector<T> A_vec(n1,&A[n1*i],false);
      sctl::Vector<T> B_vec(n1,MC[0],false);
      cheb_diff(A_vec,deg,i,B_vec);
    }
    MB+=MC;
  }
  {// Rearrange data
    int indx=0;
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      B_[indx]=B[k+(j+i*d)*d];
      indx++;
    }
  }
}

template <class T>
void cheb_curl(T* A_, int deg, T* B_){
  int dim=3;
  int d=deg+1;
  int n1 =sctl::pow<unsigned int>(d,dim);
  sctl::Vector<T> A(n1*dim); A.SetZero();
  sctl::Vector<T> B(n1*dim); B.SetZero();

  {// Rearrange data
    int indx=0;
    for(int l=0;l<dim;l++)
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      A[k+(j+(i+l*d)*d)*d]=A_[indx];
      indx++;
    }
  }
  sctl::Matrix<T> MC1(n1,1);
  sctl::Matrix<T> MC2(n1,1);
  for(int i=0;i<3;i++){
    sctl::Matrix<T> MB(n1,1, sctl::Ptr2Itr<T>(&B[n1*i], (n1)*(1)),false);
    int j1=(i+1)%3;
    int j2=(i+2)%3;
    {
      sctl::Vector<T> A1(n1,&A[n1*j1],false);
      sctl::Vector<T> A2(n1,&A[n1*j2],false);
      sctl::Vector<T> B1(n1,MC1[0],false);
      sctl::Vector<T> B2(n1,MC2[0],false);
      cheb_diff(A1,deg,j2,B1);
      cheb_diff(A2,deg,j1,B2);
    }
    MB=MC2;
    MB-=MC1;
  }
  {// Rearrange data
    int indx=0;
    for(int l=0;l<dim;l++)
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      B_[indx]=B[k+(j+(i+l*d)*d)*d];
      indx++;
    }
  }
}

//TODO: Fix number of cheb_coeff to (d+1)*(d+2)*(d+3)/6 for the following functions.

template <class T>
void cheb_laplacian(T* A, int deg, T* B){
  int dim=3;
  int d=deg+1;
  int n1=sctl::pow<unsigned int>(d,dim);

  sctl::ScratchBuf<T> C1_buf(n1);
  sctl::ScratchBuf<T> C2_buf(n1);
  T* C1 = &C1_buf.begin()[0];
  T* C2 = &C2_buf.begin()[0];

  sctl::Matrix<T> M_(1,n1, sctl::Ptr2Itr<T>(C2, (1)*(n1)),false);
  for(int i=0;i<3;i++){
    sctl::Matrix<T> M (1,n1, sctl::Ptr2Itr<T>(&B[n1*i], (1)*(n1)),false);
    for(int j=0;j<n1;j++) M[0][j]=0;
    for(int j=0;j<3;j++){
      cheb_diff(&A[n1*i],deg,3,j,C1);
      cheb_diff( C1     ,deg,3,j,C2);
      M+=M_;
    }
  }
  // C1, C2 freed automatically at scope exit.
}

/*
 * \brief Computes image of the chebyshev interpolation along the specified axis.
 */
template <class T>
void cheb_img(T* A, T* B, int deg, int dir, bool neg_){
  int d=deg+1;
  int n1=sctl::pow<unsigned int>(d,3-dir);
  int n2=sctl::pow<unsigned int>(d,  dir);
  int indx;
  T sgn,neg;
  neg=(T)(neg_?-1.0:1.0);
  for(int i=0;i<n1;i++){
    indx=i%d;
    sgn=(T)(indx%2?-neg:neg);
    for(int j=0;j<n2;j++){
      B[i*n2+j]=sgn*A[i*n2+j];
    }
  }
}

}//end namespace
