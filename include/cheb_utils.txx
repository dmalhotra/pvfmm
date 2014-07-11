/**
 * \file cheb_utils.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains chebyshev related functions.
 */

#include <assert.h>
#include <algorithm>
#include <matrix.hpp>
#include <mem_mgr.hpp>
#include <legendre_rule.hpp>

namespace pvfmm{

/**
 * \brief Returns the values of all chebyshev polynomials up to degree d,
 * evaluated at points in the input vector. Output format:
 * { T0[in[0]], ..., T0[in[n-1]], T1[in[0]], ..., T(d-1)[in[n-1]] }
 */
template <class T>
inline void cheb_poly(int d, T* in, int n, T* out){
  if(d==0){
    for(int i=0;i<n;i++)
      out[i]=(fabs(in[i])<=1?1.0:0);
  }else if(d==1){
    for(int i=0;i<n;i++){
      out[i]=(fabs(in[i])<=1?1.0:0);
      out[i+n]=(fabs(in[i])<=1?in[i]:0);
    }
  }else{
    for(int j=0;j<n;j++){
      T x=(fabs(in[j])<=1?in[j]:0);
      T y0=(fabs(in[j])<=1?1.0:0);
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
    if(i+j+k==deg) err+=fabs(cheb_coeff[indx]);
    indx++;
  }
  return err;
}

/**
 * \brief Computes Chebyshev approximation from function values at cheb node points.
 */
template <class T, class Y>
T cheb_approx(T* fn_v, int cheb_deg, int dof, T* out){
  //double eps=(typeid(T)==typeid(float)?1e-7:1e-12);

  int d=cheb_deg+1;
  static std::vector<Matrix<Y> > precomp;
  static std::vector<Matrix<Y> > precomp_;
  Matrix<Y>* Mp ;
  Matrix<Y>* Mp_;
  #pragma omp critical (CHEB_APPROX)
  {
    if(precomp.size()<=(size_t)d){
      precomp .resize(d+1);
      precomp_.resize(d+1);
    }
    if(precomp [d].Dim(0)==0 && precomp [d].Dim(1)==0){
      std::vector<Y> x(d);
      for(int i=0;i<d;i++)
        x[i]=-cos((i+0.5)*M_PI/d);

      std::vector<Y> p(d*d);
      cheb_poly(d-1,&x[0],d,&p[0]);
      for(int i=0;i<d*d;i++)
        p[i]=p[i]*(2.0/d);
      Matrix<Y> Mp1(d,d,&p[0],false);
      Matrix<Y> Mp1_=Mp1.Transpose();
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
      Matrix<Y> M0(d*d,d,&fn_v0[d*d*d*l],false);
      Matrix<Y> M1(d*d,d,&fn_v1[0],false);
      M1=M0*(*Mp_);
    }
    {
      Matrix<Y> M0(d,d*d,&fn_v1[0],false);
      Matrix<Y> M1(d,d*d,&fn_v2[0],false);
      M1=(*Mp)*M0;
    }
    for(int i=0;i<d;i++){
      Matrix<Y> M0(d,d,&fn_v2[d*d*i],false);
      Matrix<Y> M1(d,d,&fn_v3[d*d*i],false);
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
      sum+=fabs(fn_v3[k+(j+i*d)*d]);
    }
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      out[indx]=fn_v3[k+(j+i*d)*d];
      //if(fabs(out[indx])<eps*sum) out[indx]=0;
      indx++;
    }
  }

  return cheb_err(out,d-1,dof);
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
      out[i]=(fabs(in[i])<=1?1.0:0);
  }else if(d==1){
    for(int i=0;i<n;i++){
      out[i]=(fabs(in[i])<=1?1.0:0);
      out[i+n]=(fabs(in[i])<=1?in[i]:0);
    }
  }else{
    for(int j=0;j<n;j++){
      T x=(fabs(in[j])<=1?in[j]:0);
      T y0=(fabs(in[j])<=1?1.0:0);
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
  double eps=(typeid(T)==typeid(float)?1e-7:1e-12);
  int d=deg+1;
  assert(d>1);
  int N=d-1;

  Vector<T> x(d,x_,false);
  for(int i=0;i<d;i++)
    x[i]=-cos((M_PI*i)/N);
  Matrix<T> P(d,d); P.SetZero();

  T err=1;
  Vector<T> xold(d);
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
      err=(err<fabs(dx)?fabs(dx):err);
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
  //double eps=(typeid(T)==typeid(float)?1e-7:1e-12);

  int d=deg+1;
  static std::vector<Matrix<Y> > precomp;
  static std::vector<Matrix<Y> > precomp_;
  Matrix<Y>* Mp ;
  Matrix<Y>* Mp_;
  #pragma omp critical (GLL_TO_CHEB)
  {
    if(precomp.size()<=(size_t)d){
      precomp .resize(d+1);
      precomp_.resize(d+1);

      std::vector<Y> x(d); //Cheb nodes.
      for(int i=0;i<d;i++)
        x[i]=-cos((i+0.5)*M_PI/d);

      Vector<T> w(d);
      Vector<T> x_legn(d); // GLL nodes.
      gll_quadrature(d-1, &x_legn[0], &w[0]);

      Matrix<T> P(d,d); //GLL node 2 GLL coeff.
      legn_poly(d-1,&x_legn[0],d,&P[0][0]);
      for(int i=0;i<d;i++)
        for(int j=0;j<d;j++)
          P[i][j]*=w[j]*0.5*(i<d-1?(2*i+1):(i));

      Matrix<T> M_gll2cheb(d,d); //GLL coeff 2 cheb node.
      legn_poly(d-1,&x[0],d,&M_gll2cheb[0][0]);

      Matrix<T> M_g2c; //GLL node to cheb node.
      M_g2c=M_gll2cheb.Transpose()*P;

      std::vector<Y> p(d*d);
      cheb_poly(d-1,&x[0],d,&p[0]);
      for(int i=0;i<d*d;i++)
        p[i]=p[i]*(2.0/d);
      Matrix<Y> Mp1(d,d,&p[0],false);
      Mp1=Mp1*M_g2c;

      Matrix<Y> Mp1_=Mp1.Transpose();
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
      Matrix<Y> M0(d*d,d,&fn_v0[d*d*d*l],false);
      Matrix<Y> M1(d*d,d,&fn_v1[0],false);
      M1=M0*(*Mp_);
    }
    {
      Matrix<Y> M0(d,d*d,&fn_v1[0],false);
      Matrix<Y> M1(d,d*d,&fn_v2[0],false);
      M1=(*Mp)*M0;
    }
    for(int i=0;i<d;i++){
      Matrix<Y> M0(d,d,&fn_v2[d*d*i],false);
      Matrix<Y> M1(d,d,&fn_v3[d*d*i],false);
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
      sum+=fabs(fn_v3[k+(j+i*d)*d]);
    }
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      out[indx]=fn_v3[k+(j+i*d)*d];
      //if(fabs(out[indx])<eps*sum) out[indx]=0;
      indx++;
    }
  }

  return cheb_err(out,d-1,dof);
}

/**
 * \brief Computes Chebyshev approximation from the input function pointer.
 */
template <class T>
T cheb_approx(T (*fn)(T,T,T), int cheb_deg, T* coord, T s, std::vector<T>& out){
  int d=cheb_deg+1;
  std::vector<T> x(d);
  for(int i=0;i<d;i++)
    x[i]=cos((i+0.5)*M_PI/d);

  std::vector<T> p;
  cheb_poly(d-1,&x[0],d,&p[0]);

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
  return cheb_approx(&fn_v[0], d-1, 1, &out[0]);
}

/**
 * \brief Evaluates polynomial values from input coefficients at points on
 * a regular grid defined by in_x, in_y, in_z the values in the input vector.
 */
template <class T>
void cheb_eval(Vector<T>& coeff_, int cheb_deg, std::vector<T>& in_x, std::vector<T>& in_y, std::vector<T>& in_z, Vector<T>& out){
  int d=cheb_deg+1;
  int dof=coeff_.Dim()/((d*(d+1)*(d+2))/6);
  assert(coeff_.Dim()==(size_t)(d*(d+1)*(d+2)*dof)/6);

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

  int n1=in_x.size();
  int n2=in_y.size();
  int n3=in_z.size();
  out.Resize(n1*n2*n3*dof);
  if(n1==0 || n2==0 || n3==0) return;

  std::vector<T> p1(n1*d);
  std::vector<T> p2(n2*d);
  std::vector<T> p3(n3*d);
  cheb_poly(d-1,&in_x[0],n1,&p1[0]);
  cheb_poly(d-1,&in_y[0],n2,&p2[0]);
  cheb_poly(d-1,&in_z[0],n3,&p3[0]);

  std::vector<T> fn_v1(n1*d *d );
  std::vector<T> fn_v2(n1*d *n3);
  std::vector<T> fn_v3(n1*n2*n3);
  Matrix<T> Mp1(d,n1,&p1[0],false);
  Matrix<T> Mp2(d,n2,&p2[0],false);
  Matrix<T> Mp3(d,n3,&p3[0],false);
  Matrix<T> Mp2_=Mp2.Transpose();
  Matrix<T> Mp3_=Mp3.Transpose();
  for(int k=0;k<dof;k++){
    {
      Matrix<T> M0(d*d,d,&coeff[k*d*d*d],false);
      Matrix<T> M1(d*d,n1,&fn_v1[0],false);
      M1=M0*Mp1;
    }
    {
      Matrix<T> M0(d,d*n1,&fn_v1[0],false);
      Matrix<T> M1(n3,d*n1,&fn_v2[0],false);
      M1=Mp3_*M0;
    }
    {
      int dn1=d*n1;
      int n2n1=n2*n1;
      for(int i=0;i<n3;i++){
        Matrix<T> M0(d,n1,&fn_v2[i*dn1],false);
        Matrix<T> M1(n2,n1,&fn_v3[i*n2n1],false);
        M1=Mp2_*M0;
      }
    }
    mem::memcopy(&out[n1*n2*n3*k],&fn_v3[0],n1*n2*n3*sizeof(T));
  }
}

/**
 * \brief Evaluates polynomial values from input coefficients at points
 * in the coord vector.
 */
template <class T>
inline void cheb_eval(Vector<T>& coeff_, int cheb_deg, std::vector<T>& coord, Vector<T>& out){
  int dim=3;
  int d=cheb_deg+1;
  int n=coord.size()/dim;
  int dof=coeff_.Dim()/((d*(d+1)*(d+2))/6);
  assert(coeff_.Dim()==(size_t)(d*(d+1)*(d+2)*dof)/6);

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

  Matrix<T> coord_(n,dim,&coord[0]);
  coord_=coord_.Transpose();

  Matrix<T> px(d,n);
  Matrix<T> py(d,n);
  Matrix<T> pz(d,n);
  cheb_poly(d-1,&(coord_[0][0]),n,&(px[0][0]));
  cheb_poly(d-1,&(coord_[1][0]),n,&(py[0][0]));
  cheb_poly(d-1,&(coord_[2][0]),n,&(pz[0][0]));

  Matrix<T> M_coeff0(d*d*dof, d, &coeff[0], false);
  Matrix<T> M0 = (M_coeff0 * px).Transpose(); // {n, dof*d*d}

  py = py.Transpose();
  pz = pz.Transpose();
  out.Resize(n*dof);
  for(int i=0; i<n; i++)
    for(int j=0; j<dof; j++){
      Matrix<T> M0_  (d, d, &(M0[i][  j*d*d]), false);
      Matrix<T> py_  (d, 1, &(py[i][      0]), false);
      Matrix<T> pz_  (1, d, &(pz[i][      0]), false);

      Matrix<T> M_out(1, 1, &(  out[i*dof+j]), false);
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
  cheb_poly(d-1,&coord[0],3,&p[0]);

  for(int i=0;i<d;i++){
    p_[i]=p[i*3];
    p_[i+d]=p[i*3+1];
    p_[i+2*d]=p[i*3+2];
  }
  T* coeff_=&buff[2*3*d];

  Matrix<T> v_p0    (1, d, &    p_[0],false);
  Matrix<T> v_p1    (d, 1, &    p_[d],false);
  Matrix<T> M_coeff_(d, d, &coeff_[0],false);
  M_coeff_ = v_p1 * v_p0; // */
  //mat::gemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,d,d,1,1.0,&p_[d],1,&p_[0],d,0.0,&coeff_[0],d);

  Matrix<T> v_p2    (d,   1, &    p_[2*d],false);
  Matrix<T> v_coeff_(1, d*d, &coeff_[  0],false);
  Matrix<T> M_coeff (d, d*d, &coeff [  0],false);
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
void points2cheb(int deg, T* coord, T* val, int n, int dim, T* node_coord, T node_size, Vector<T>& cheb_coeff){
  if(n==0) return;
  int deg_=((int)(pow((T)n*6,1.0/3.0)+0.5))/2;
  deg_=(deg_>deg?deg:deg_);
  deg_=(deg_>0?deg_:1);
  int deg3=((deg_+1)*(deg_+2)*(deg_+3))/6;
  cheb_coeff.Resize(dim*((deg+1)*(deg+2)*(deg+3))/6);
  cheb_coeff.SetZero();

  //Map coordinates to unit cube
  std::vector<T> coord_(n*3);
  for(int i=0;i<n;i++){
    coord_[i*3  ]=(coord[i*3  ]-node_coord[0])*2.0/node_size-1.0;
    coord_[i*3+1]=(coord[i*3+1]-node_coord[1])*2.0/node_size-1.0;
    coord_[i*3+2]=(coord[i*3+2]-node_coord[2])*2.0/node_size-1.0;
  }

  //Compute the matrix M
  Matrix<T> M(n,deg3);
  std::vector<T> buff((deg_+1)*(deg_+1+3*2));
  for(int i=0;i<n;i++)
    cheb_eval(deg_,&coord_[i*3],&(M[i][0]),&buff[0]);

  //Compute the pinv and get the cheb_coeff.
  Matrix<T> M_val(n,dim,&val[0]);
  T eps=(typeid(T)==typeid(float)?1e-5:1e-12);
  Matrix<T> cheb_coeff_=(M.pinv(eps)*M_val).Transpose();

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
void quad_rule(int n, T* x, T* w){//*
  static std::vector<Vector<double> > x_lst(10000);
  static std::vector<Vector<double> > w_lst(10000);
  assert(n<10000);

  bool done=false;
  #pragma omp critical (QUAD_RULE)
  if(x_lst[n].Dim()>0){
    Vector<double>& x_=x_lst[n];
    Vector<double>& w_=w_lst[n];
    for(int i=0;i<n;i++){
      x[i]=x_[i];
      w[i]=w_[i];
    }
    done=true;
  }
  if(done) return;

  Vector<double> x_(n);
  Vector<double> w_(n);
  T alpha=0.0;
  T beta=0.0;
  T a=-1.0;
  T b= 1.0;
  int kind = 1;
  cgqf ( n, kind, (double)alpha, (double)beta, (double)a, (double)b, &x_[0], &w_[0] );
  #pragma omp critical (QUAD_RULE)
  { // Set x_lst, w_lst
    x_lst[n]=x_;
    w_lst[n]=w_;
  }
  quad_rule(n, x, w);

  //Trapezoidal quadrature nodes and weights
/*  for(int i=0;i<n;i++){
    x[i]=(2.0*i+1.0)/(1.0*n)-1.0;
    w[i]=2.0/n;
  }// */
  //Gauss-Chebyshev quadrature nodes and weights
/*  for(int i=0;i<n;i++){
    x[i]=cos((2.0*i+1.0)/(2.0*n)*M_PI);
    w[i]=sqrt(1.0-x[i]*x[i])*M_PI/n;
  }// */
  //Gauss-Legendre quadrature nodes and weights
/*  T x_[10]={-0.97390652851717,  -0.86506336668898,  -0.67940956829902,  -0.43339539412925,  -0.14887433898163,
          0.14887433898163,   0.43339539412925,   0.67940956829902,   0.86506336668898,   0.97390652851717};
  T w_[10]={0.06667134430869,   0.14945134915058,   0.21908636251598,   0.26926671931000,   0.29552422471475,
          0.29552422471475,   0.26926671931000,   0.21908636251598,   0.14945134915058,   0.06667134430869};
  for(int i=0;i<10;i++){
    x[i]=x_[i];
    w[i]=w_[i];
  }// */
}

template <class Y>
std::vector<double> integ_pyramid(int m, double* s, double r, int nx, typename Kernel<Y>::Ker_t kernel, int* ker_dim, int* perm){//*
  static mem::MemoryManager mem_mgr(16*1024*1024*sizeof(double));
  int ny=nx;
  int nz=nx;

  double eps=1e-14;
  int k_dim=ker_dim[0]*ker_dim[1];

  std::vector<double> qp_x(nx), qw_x(nx);
  std::vector<double> qp_y(ny), qw_y(ny);
  std::vector<double> qp_z(nz), qw_z(nz);
  std::vector<double> p_x(nx*m);
  std::vector<double> p_y(ny*m);
  std::vector<double> p_z(nz*m);

  std::vector<double> x_;
  { //  Build stack along X-axis.
    x_.push_back(s[0]);
    x_.push_back(fabs(1.0-s[0])+s[0]);
    x_.push_back(fabs(1.0-s[1])+s[0]);
    x_.push_back(fabs(1.0+s[1])+s[0]);
    x_.push_back(fabs(1.0-s[2])+s[0]);
    x_.push_back(fabs(1.0+s[2])+s[0]);
    std::sort(x_.begin(),x_.end());
    for(int i=0;i<x_.size();i++){
      if(x_[i]<-1.0) x_[i]=-1.0;
      if(x_[i]> 1.0) x_[i]= 1.0;
    }

    std::vector<double> x_new;
    double x_jump=fabs(1.0-s[0]);
    if(fabs(1.0-s[1])>eps) x_jump=std::min(x_jump,fabs(1.0-s[1]));
    if(fabs(1.0+s[1])>eps) x_jump=std::min(x_jump,fabs(1.0+s[1]));
    if(fabs(1.0-s[2])>eps) x_jump=std::min(x_jump,fabs(1.0-s[2]));
    if(fabs(1.0+s[2])>eps) x_jump=std::min(x_jump,fabs(1.0+s[2]));
    for(int k=0; k<x_.size()-1; k++){
      double x0=x_[k];
      double x1=x_[k+1];

      double A0=0;
      double A1=0;
      { // A0
        double y0=s[1]-(x0-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
        double y1=s[1]+(x0-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
        double z0=s[2]-(x0-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
        double z1=s[2]+(x0-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;
        A0=(y1-y0)*(z1-z0);
      }
      { // A1
        double y0=s[1]-(x1-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
        double y1=s[1]+(x1-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
        double z0=s[2]-(x1-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
        double z1=s[2]+(x1-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;
        A1=(y1-y0)*(z1-z0);
      }
      double V=0.5*(A0+A1)*(x1-x0);
      if(V<eps) continue;

      if(!x_new.size()) x_new.push_back(x0);
      x_jump=std::max(x_jump,x0-s[0]);
      while(s[0]+x_jump*1.5<x1){
        x_new.push_back(s[0]+x_jump);
        x_jump*=2.0;
      }
      if(x_new.back()+eps<x1) x_new.push_back(x1);
    }
    assert(x_new.size()<30);
    x_.swap(x_new);
  }

  Vector<double> k_out(   ny*nz*k_dim,(double*)mem_mgr.malloc(   ny*nz*k_dim*sizeof(double)),false); //Output of kernel evaluation.
  Vector<double> I0   (   ny*m *k_dim,(double*)mem_mgr.malloc(   ny*m *k_dim*sizeof(double)),false);
  Vector<double> I1   (   m *m *k_dim,(double*)mem_mgr.malloc(   m *m *k_dim*sizeof(double)),false);
  Vector<double> I2   (m *m *m *k_dim,(double*)mem_mgr.malloc(m *m *m *k_dim*sizeof(double)),false); I2.SetZero();
  if(x_.size()>1)
  for(int k=0; k<x_.size()-1; k++){
    double x0=x_[k];
    double x1=x_[k+1];

    { // Set qp_x
      std::vector<double> qp(nx);
      std::vector<double> qw(nx);
      quad_rule(nx,&qp[0],&qw[0]);
      for(int i=0; i<nx; i++)
        qp_x[i]=(x1-x0)*qp[i]/2.0+(x1+x0)/2.0;
      qw_x=qw;
    }
    cheb_poly(m-1,&qp_x[0],nx,&p_x[0]);

    for(int i=0; i<nx; i++){
      double y0=s[1]-(qp_x[i]-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
      double y1=s[1]+(qp_x[i]-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
      double z0=s[2]-(qp_x[i]-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
      double z1=s[2]+(qp_x[i]-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;

      { // Set qp_y
        std::vector<double> qp(ny);
        std::vector<double> qw(ny);
        quad_rule(ny,&qp[0],&qw[0]);
        for(int j=0; j<ny; j++)
          qp_y[j]=(y1-y0)*qp[j]/2.0+(y1+y0)/2.0;
        qw_y=qw;
      }
      { // Set qp_z
        std::vector<double> qp(nz);
        std::vector<double> qw(nz);
        quad_rule(nz,&qp[0],&qw[0]);
        for(int j=0; j<nz; j++)
          qp_z[j]=(z1-z0)*qp[j]/2.0+(z1+z0)/2.0;
        qw_z=qw;
      }
      cheb_poly(m-1,&qp_y[0],ny,&p_y[0]);
      cheb_poly(m-1,&qp_z[0],nz,&p_z[0]);
      { // k_out =  kernel x qw
        Y src[3]={0,0,0};
        std::vector<Y> trg(ny*nz*3);
        for(int i0=0; i0<ny; i0++){
          size_t indx0=i0*nz*3;
          for(int i1=0; i1<nz; i1++){
            size_t indx1=indx0+i1*3;
            trg[indx1+perm[0]]=(s[0]-qp_x[i ])*r*0.5*perm[1];
            trg[indx1+perm[2]]=(s[1]-qp_y[i0])*r*0.5*perm[3];
            trg[indx1+perm[4]]=(s[2]-qp_z[i1])*r*0.5*perm[5];
          }
        }
        {
          Matrix<double> k_val(ny*nz*ker_dim[0],ker_dim[1]);
          Kernel<double>::Eval(&src[0],1,&trg[0],ny*nz,&k_val[0][0],kernel,ker_dim);
          Matrix<double> k_val_t(ker_dim[1],ny*nz*ker_dim[0],&k_out[0], false);
          k_val_t=k_val.Transpose();
        }
        for(int kk=0; kk<k_dim; kk++){
          for(int i0=0; i0<ny; i0++){
            size_t indx=(kk*ny+i0)*nz;
            for(int i1=0; i1<nz; i1++){
              k_out[indx+i1] *= qw_y[i0]*qw_z[i1];
            }
          }
        }
      }

      I0.SetZero();
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

      I1.SetZero();
      for(int kk=0; kk<k_dim; kk++){
        for(int i2=0; i2<ny; i2++){
          size_t indx0=(kk*ny+i2)*m;
          for(int i0=0; i0<m; i0++){
            size_t indx1=(kk* m+i0)*m;
            double py=p_y[i0*ny+i2];
            for(int i1=0; i0+i1<m; i1++){
              I1[indx1+i1] += I0[indx0+i1]*py;
            }
          }
        }
      }

      double v=(x1-x0)*(y1-y0)*(z1-z0);
      for(int kk=0; kk<k_dim; kk++){
        for(int i0=0; i0<m; i0++){
          double px=p_x[i+i0*nx]*qw_x[i]*v;
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
    I2[i]=I2[i]*r*r*r/64.0;

  if(x_.size()>1)
  Profile::Add_FLOP(( 2*ny*nz*m*k_dim
                     +ny*m*(m+1)*k_dim
                     +2*m*(m+1)*k_dim
                     +m*(m+1)*(m+2)/3*k_dim)*nx*(x_.size()-1));

  std::vector<double> I2_(&I2[0], &I2[0]+I2.Dim());
  mem_mgr.free(&k_out[0]);
  mem_mgr.free(&I0   [0]);
  mem_mgr.free(&I1   [0]);
  mem_mgr.free(&I2   [0]);
  return I2_;
}

template <class Y>
std::vector<double> integ(int m, double* s, double r, int n, typename Kernel<Y>::Ker_t kernel, int* ker_dim){//*
  //Compute integrals over pyramids in all directions.
  int k_dim=ker_dim[0]*ker_dim[1];
  double s_[3];
  s_[0]=s[0]*2.0/r-1.0;
  s_[1]=s[1]*2.0/r-1.0;
  s_[2]=s[2]*2.0/r-1.0;

  double s1[3];
  int perm[6];
  std::vector<double> U_[6];

  s1[0]= s_[0];s1[1]=s_[1];s1[2]=s_[2];
  perm[0]= 0; perm[2]= 1; perm[4]= 2;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[0]=integ_pyramid<Y>(m,s1,r,n,kernel,ker_dim,perm);

  s1[0]=-s_[0];s1[1]=s_[1];s1[2]=s_[2];
  perm[0]= 0; perm[2]= 1; perm[4]= 2;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[1]=integ_pyramid<Y>(m,s1,r,n,kernel,ker_dim,perm);

  s1[0]= s_[1];s1[1]=s_[0];s1[2]=s_[2];
  perm[0]= 1; perm[2]= 0; perm[4]= 2;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[2]=integ_pyramid<Y>(m,s1,r,n,kernel,ker_dim,perm);

  s1[0]=-s_[1];s1[1]=s_[0];s1[2]=s_[2];
  perm[0]= 1; perm[2]= 0; perm[4]= 2;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[3]=integ_pyramid<Y>(m,s1,r,n,kernel,ker_dim,perm);

  s1[0]= s_[2];s1[1]=s_[0];s1[2]=s_[1];
  perm[0]= 2; perm[2]= 0; perm[4]= 1;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[4]=integ_pyramid<Y>(m,s1,r,n,kernel,ker_dim,perm);

  s1[0]=-s_[2];s1[1]=s_[0];s1[2]=s_[1];
  perm[0]= 2; perm[2]= 0; perm[4]= 1;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[5]=integ_pyramid<Y>(m,s1,r,n,kernel,ker_dim,perm);

  std::vector<double> U; U.assign(m*m*m*k_dim,0);
  for(int kk=0; kk<k_dim; kk++){
    for(int i=0;i<m;i++){
      for(int j=0;j<m;j++){
        for(int k=0;k<m;k++){
          U[kk*m*m*m + k*m*m + j*m + i]+=U_[0][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + k*m*m + j*m + i]+=U_[1][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
        }
      }
    }
  }

  for(int kk=0; kk<k_dim; kk++){
    for(int i=0; i<m; i++){
      for(int j=0; j<m; j++){
        for(int k=0; k<m; k++){
          U[kk*m*m*m + k*m*m + i*m + j]+=U_[2][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + k*m*m + i*m + j]+=U_[3][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
        }
      }
    }
  }

  for(int kk=0; kk<k_dim; kk++){
    for(int i=0; i<m; i++){
      for(int j=0; j<m; j++){
        for(int k=0; k<m; k++){
          U[kk*m*m*m + i*m*m + k*m + j]+=U_[4][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + i*m*m + k*m + j]+=U_[5][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
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
template <class Y>
std::vector<Y> cheb_integ(int m, Y* s_, Y r_, typename Kernel<Y>::Ker_t kernel, int* ker_dim){
  double eps=(typeid(Y)==typeid(float)?1e-7:1e-14);

  double r=r_;
  double s[3]={s_[0],s_[1],s_[2]};

  int n=m+1;
  double err=1.0;
  int k_dim=ker_dim[0]*ker_dim[1];
  std::vector<double> U=integ<Y>(m+1,s,r,n,kernel,ker_dim);
  std::vector<double> U_;
  while(err>eps){
    n=(int)round(n*1.3);
    if(n>300){
      std::cout<<"Cheb_Integ::Failed to converge.["<<err<<","<<s[0]<<","<<s[1]<<","<<s[2]<<"]\n";
      break;
    }
    U_=integ<Y>(m+1,s,r,n,kernel,ker_dim);
    err=0;
    for(int i=0;i<(m+1)*(m+1)*(m+1)*k_dim;i++)
      if(fabs(U[i]-U_[i])>err)
        err=fabs(U[i]-U_[i]);
    U=U_;
  }

  std::vector<Y> U0(((m+1)*(m+2)*(m+3)*k_dim)/6);
  {// Rearrange data
    int indx=0;
    for(int l0=0;l0<ker_dim[0];l0++)
    for(int l1=0;l1<ker_dim[1];l1++)
    for(int i=0;i<=m;i++)
    for(int j=0;i+j<=m;j++)
    for(int k=0;i+j+k<=m;k++){
      U0[indx]=U[(k+(j+(i+(l1*ker_dim[0]+l0)*(m+1))*(m+1))*(m+1))];
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
    x[i]=-cos((i+0.5)*M_PI/d)*0.5+0.5;
  if(dim==1) return x;

  int n1=(int)(pow((T)d,dim)+0.5);
  std::vector<T> y(n1*dim);
  for(int i=0;i<dim;i++){
    int n2=(int)(pow((T)d,i)+0.5);
    for(int j=0;j<n1;j++){
      y[j*dim+i]=x[(j/n2)%d];
    }
  }
  return y;
}

template <class T>
void cheb_diff(T* A, int deg, T* B){
  int d=deg+1;
  static Matrix<T> M;
  #pragma omp critical (CHEB_DIFF)
  if(M.Dim(0)!=d){
    M.Resize(d,d);
    for(int i=0;i<d;i++){
      for(int j=0;j<d;j++) M[j][i]=0;
      for(int j=1-(i%2);j<i-1;j=j+2){
        M[j][i]=2*i*2;
      }
      if(i%2==1) M[0][i]-=i*2;
    }
  }
  Matrix<T> MA(d,1,A,false);
  Matrix<T> MB(d,1,B,false);
  MB=M*MA;
}


template <class T>
void cheb_diff(T* A, int deg, int dim, int curr_dim, T* B){
  int d=deg+1;
  static Matrix<T> M;
  #pragma omp critical (CHEB_DIFF1)
  if(M.Dim(0)!=(size_t)d){
    M.Resize(d,d);
    for(int i=0;i<d;i++){
      for(int j=0;j<d;j++) M[j][i]=0;
      for(int j=1-(i%2);j<i;j=j+2){
        M[j][i]=2*i*2;
      }
      if(i%2==1) M[0][i]-=i*2;
    }
  }
  int n1=(int)(pow((T)d,curr_dim)+0.5);
  int n2=(int)(pow((T)d,dim-curr_dim-1)+0.5);
  for(int i=0;i<n2;i++){
    Matrix<T> MA(d,n1,&A[i*n1*d],false);
    Matrix<T> MB(d,n1,&B[i*n1*d],false);
    MB=M*MA;
  }
}

template <class T>
void cheb_grad(T* A, int deg, T* B){
  int dim=3;
  int d=deg+1;
  int n1 =(d*(d+1)*(d+2))/6;
  int n1_=(int)(pow((T)d,dim)+0.5);
  Vector<T> A_(n1_); A_.SetZero();
  Vector<T> B_(n1_); B_.SetZero();

  {// Rearrange data
    int indx=0;
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      A_[k+(j+i*d)*d]=A[indx];
      indx++;
    }
  }

  for(int l=0;l<dim;l++){
    cheb_diff(&A_[0],d-1,dim,l,&B_[0]);
    {// Rearrange data
      int indx=l*n1;
      for(int i=0;i<d;i++)
      for(int j=0;i+j<d;j++)
      for(int k=0;i+j+k<d;k++){
        B[indx]=B_[k+(j+i*d)*d];
        indx++;
      }
    }
  }
}

template <class T>
void cheb_div(T* A_, int deg, T* B_){
  int dim=3;
  int d=deg+1;
  int n1 =(int)(pow((T)d,dim)+0.5);
  Vector<T> A(n1*dim); A.SetZero();
  Vector<T> B(n1    ); B.SetZero();

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
  Matrix<T> MB(n1,1,&B[0],false);
  Matrix<T> MC(n1,1);
  for(int i=0;i<3;i++){
    cheb_diff(&A[n1*i],d-1,3,i,MC[0]);
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
  int n1 =(int)(pow((T)d,dim)+0.5);
  Vector<T> A(n1*dim); A.SetZero();
  Vector<T> B(n1*dim); B.SetZero();

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
  Matrix<T> MC1(n1,1);
  Matrix<T> MC2(n1,1);
  for(int i=0;i<3;i++){
    Matrix<T> MB(n1,1,&B[n1*i],false);
    int j1=(i+1)%3;
    int j2=(i+2)%3;
    cheb_diff(&A[n1*j1],d-1,3,j2,MC1[0]);
    cheb_diff(&A[n1*j2],d-1,3,j1,MC2[0]);
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
  int n1=(int)(pow((T)d,dim)+0.5);

  T* C1=new T[n1];
  T* C2=new T[n1];

  Matrix<T> M_(1,n1,C2,false);
  for(int i=0;i<3;i++){
    Matrix<T> M (1,n1,&B[n1*i],false);
    for(int j=0;j<n1;j++) M[0][j]=0;
    for(int j=0;j<3;j++){
      cheb_diff(&A[n1*i],d-1,3,j,C1);
      cheb_diff( C1     ,d-1,3,j,C2);
      M+=M_;
    }
  }

  delete[] C1;
  delete[] C2;
}

/*
 * \brief Computes image of the chebyshev interpolation along the specified axis.
 */
template <class T>
void cheb_img(T* A, T* B, int deg, int dir, bool neg_){
  int d=deg+1;
  int n1=(int)(pow((T)d,3-dir)+0.5);
  int n2=(int)(pow((T)d,  dir)+0.5);
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
