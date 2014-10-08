/**
 * \file kernel.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-20-2011
 * \brief This file contains the implementation of the struct Kernel and also the
 * implementation of various kernels for FMM.
 */

#include <cmath>
#include <cstdlib>
#include <vector>

#include <mem_mgr.hpp>
#include <profile.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <precomp_mat.hpp>

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#if defined(__MIC__)
#include <immintrin.h>
#endif

namespace pvfmm{

/**
 * \brief Constructor.
 */
template <class T>
Kernel<T>::Kernel(Ker_t poten, Ker_t dbl_poten, const char* name, int dim_, std::pair<int,int> k_dim,
                  size_t dev_poten, size_t dev_dbl_poten){
  dim=dim_;
  ker_dim[0]=k_dim.first;
  ker_dim[1]=k_dim.second;
  ker_poten=poten;
  dbl_layer_poten=dbl_poten;
  ker_name=std::string(name);

  dev_ker_poten=dev_poten;
  dev_dbl_layer_poten=dev_dbl_poten;

  k_s2m=NULL;
  k_s2l=NULL;
  k_s2t=NULL;
  k_m2m=NULL;
  k_m2l=NULL;
  k_m2t=NULL;
  k_l2l=NULL;
  k_l2t=NULL;

  homogen=false;
  init=false;
}

/**
 * \brief Initialize the kernel.
 */
template <class T>
void Kernel<T>::Initialize(bool verbose) const{
  if(init) return;
  init=true;

  T eps=1.0;
  while(eps+(T)1.0>1.0) eps*=0.5;

  { // Determine scal
    homogen=true;
    Matrix<T> M_scal(ker_dim[0],ker_dim[1]);
    size_t N=1024;
    T eps_=N*eps;

    T src_coord[3]={0,0,0};
    std::vector<T> trg_coord1(N*COORD_DIM);
    std::vector<T> trg_coord2(N*COORD_DIM);
    for(size_t i=0;i<N;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=sqrt(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*COORD_DIM+0]=x;
      trg_coord1[i*COORD_DIM+1]=y;
      trg_coord1[i*COORD_DIM+2]=z;
    }
    for(size_t i=0;i<N*COORD_DIM;i++){
      trg_coord2[i]=trg_coord1[i]*0.5;
    }

    T max_val=0;
    Matrix<T> M1(N,ker_dim[0]*ker_dim[1]);
    Matrix<T> M2(N,ker_dim[0]*ker_dim[1]);
    for(size_t i=0;i<N;i++){
      BuildMatrix(&src_coord [          0], 1,
                  &trg_coord1[i*COORD_DIM], 1, &(M1[i][0]));
      BuildMatrix(&src_coord [          0], 1,
                  &trg_coord2[i*COORD_DIM], 1, &(M2[i][0]));
      for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
        max_val=std::max<T>(max_val,M1[i][j]);
        max_val=std::max<T>(max_val,M2[i][j]);
      }
    }
    for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
      T dot11=0, dot12=0, dot22=0;
      for(size_t j=0;j<N;j++){
        dot11+=M1[j][i]*M1[j][i];
        dot12+=M1[j][i]*M2[j][i];
        dot22+=M2[j][i]*M2[j][i];
      }
      if(dot11>max_val*max_val*eps_ &&
         dot22>max_val*max_val*eps_ ){
        T s=dot12/dot11;
        M_scal[0][i]=log(s)/log(2.0);
        T err=sqrt(0.5*(dot22/dot11)/(s*s)-0.5);
        if(err>eps_){
          homogen=false;
          M_scal[0][i]=0.0;
        }
        assert(M_scal[0][i]>=0.0); // Kernel function must decay
      }else M_scal[0][i]=-1;
    }

    src_scal.Resize(ker_dim[0]); src_scal.SetZero();
    trg_scal.Resize(ker_dim[1]); trg_scal.SetZero();
    if(homogen){
      Matrix<T> b(ker_dim[0]*ker_dim[1]+1,1); b.SetZero();
      mem::memcopy(&b[0][0],&M_scal[0][0],ker_dim[0]*ker_dim[1]*sizeof(T));

      Matrix<T> M(ker_dim[0]*ker_dim[1]+1,ker_dim[0]+ker_dim[1]); M.SetZero();
      M[ker_dim[0]*ker_dim[1]][0]=1;
      for(size_t i0=0;i0<ker_dim[0];i0++)
      for(size_t i1=0;i1<ker_dim[1];i1++){
        size_t j=i0*ker_dim[1]+i1;
        if(b[j][0]>=0){
          M[j][ 0+        i0]=1;
          M[j][i1+ker_dim[0]]=1;
        }
      }
      Matrix<T> x=M.pinv()*b;

      for(size_t i=0;i<ker_dim[0];i++){
        src_scal[i]=x[i][0];
      }
      for(size_t i=0;i<ker_dim[1];i++){
        trg_scal[i]=x[ker_dim[0]+i][0];
      }

      for(size_t i0=0;i0<ker_dim[0];i0++)
      for(size_t i1=0;i1<ker_dim[1];i1++){
        if(M_scal[i0][i1]>=0){
          if(fabs(src_scal[i0]+trg_scal[i1]-M_scal[i0][i1])>eps_){
            homogen=false;
          }
        }
      }
    }

    if(!homogen){
      src_scal.SetZero();
      trg_scal.SetZero();
      //std::cout<<ker_name<<" not-scale-invariant\n";
    }
  }
  { // Determine symmetry
    perm_vec.Resize(Perm_Count);

    size_t N=1024;
    T eps_=N*eps;
    T src_coord[3]={0,0,0};
    std::vector<T> trg_coord1(N*COORD_DIM);
    std::vector<T> trg_coord2(N*COORD_DIM);
    for(size_t i=0;i<N;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=sqrt(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*COORD_DIM+0]=x;
      trg_coord1[i*COORD_DIM+1]=y;
      trg_coord1[i*COORD_DIM+2]=z;
    }

    for(size_t p_type=0;p_type<C_Perm;p_type++){ // For each symmetry transform

      switch(p_type){ // Set trg_coord2
        case ReflecX:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]=-trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+2];
          }
          break;
        case ReflecY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+1]=-trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+2];
          }
          break;
        case ReflecZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]=-trg_coord1[i*COORD_DIM+2];
          }
          break;
        case SwapXY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+2];
          }
          break;
        case SwapXZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+2];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+0];
          }
          break;
        default:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+2];
          }
      }

      Matrix<long long> M11, M22;
      {
        Matrix<T> M1(N,ker_dim[0]*ker_dim[1]); M1.SetZero();
        Matrix<T> M2(N,ker_dim[0]*ker_dim[1]); M2.SetZero();
        for(size_t i=0;i<N;i++){
          BuildMatrix(&src_coord [          0], 1,
                      &trg_coord1[i*COORD_DIM], 1, &(M1[i][0]));
          BuildMatrix(&src_coord [          0], 1,
                      &trg_coord2[i*COORD_DIM], 1, &(M2[i][0]));
        }

        Matrix<T> dot11(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot11.SetZero();
        Matrix<T> dot12(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot12.SetZero();
        Matrix<T> dot22(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot22.SetZero();
        std::vector<T> norm1(ker_dim[0]*ker_dim[1]);
        std::vector<T> norm2(ker_dim[0]*ker_dim[1]);
        {
          for(size_t k=0;k<N;k++)
          for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++)
          for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
            dot11[i][j]+=M1[k][i]*M1[k][j];
            dot12[i][j]+=M1[k][i]*M2[k][j];
            dot22[i][j]+=M2[k][i]*M2[k][j];
          }
          for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
            norm1[i]=sqrt(dot11[i][i]);
            norm2[i]=sqrt(dot22[i][i]);
          }
          for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++)
          for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
            dot11[i][j]/=(norm1[i]*norm1[j]);
            dot12[i][j]/=(norm1[i]*norm2[j]);
            dot22[i][j]/=(norm2[i]*norm2[j]);
          }
        }

        long long flag=1;
        M11.Resize(ker_dim[0],ker_dim[1]); M11.SetZero();
        M22.Resize(ker_dim[0],ker_dim[1]); M22.SetZero();
        for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
          if(norm1[i]>eps_ && M11[0][i]==0){
            for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
              if(fabs(norm1[i]-norm1[j])<eps_ && fabs(fabs(dot11[i][j])-1.0)<eps_){
                M11[0][j]=(dot11[i][j]>0?flag:-flag);
              }
              if(fabs(norm1[i]-norm2[j])<eps_ && fabs(fabs(dot12[i][j])-1.0)<eps_){
                M22[0][j]=(dot12[i][j]>0?flag:-flag);
              }
            }
            flag++;
          }
        }
      }

      Matrix<long long> P1, P2;
      { // P1
        Matrix<long long>& P=P1;
        Matrix<long long>  M1=M11;
        Matrix<long long>  M2=M22;
        for(size_t i=0;i<M1.Dim(0);i++){
          for(size_t j=0;j<M1.Dim(1);j++){
            if(M1[i][j]<0) M1[i][j]=-M1[i][j];
            if(M2[i][j]<0) M2[i][j]=-M2[i][j];
          }
          std::sort(&M1[i][0],&M1[i][M1.Dim(1)]);
          std::sort(&M2[i][0],&M2[i][M2.Dim(1)]);
        }
        P.Resize(M1.Dim(0),M1.Dim(0));
        for(size_t i=0;i<M1.Dim(0);i++)
        for(size_t j=0;j<M1.Dim(0);j++){
          P[i][j]=1;
          for(size_t k=0;k<M1.Dim(1);k++)
          if(M1[i][k]!=M2[j][k]){
            P[i][j]=0;
            break;
          }
        }
      }
      { // P2
        Matrix<long long>& P=P2;
        Matrix<long long>  M1=M11.Transpose();
        Matrix<long long>  M2=M22.Transpose();
        for(size_t i=0;i<M1.Dim(0);i++){
          for(size_t j=0;j<M1.Dim(1);j++){
            if(M1[i][j]<0) M1[i][j]=-M1[i][j];
            if(M2[i][j]<0) M2[i][j]=-M2[i][j];
          }
          std::sort(&M1[i][0],&M1[i][M1.Dim(1)]);
          std::sort(&M2[i][0],&M2[i][M2.Dim(1)]);
        }
        P.Resize(M1.Dim(0),M1.Dim(0));
        for(size_t i=0;i<M1.Dim(0);i++)
        for(size_t j=0;j<M1.Dim(0);j++){
          P[i][j]=1;
          for(size_t k=0;k<M1.Dim(1);k++)
          if(M1[i][k]!=M2[j][k]){
            P[i][j]=0;
            break;
          }
        }
      }

      std::vector<Permutation<long long> > P1vec, P2vec;
      int dbg_cnt=0;
      { // P1vec
        Matrix<long long>& Pmat=P1;
        std::vector<Permutation<long long> >& Pvec=P1vec;

        Permutation<long long> P(Pmat.Dim(0));
        Vector<PERM_INT_T>& perm=P.perm;
        perm.SetZero();

        // First permutation
        for(size_t i=0;i<P.Dim();i++)
        for(size_t j=0;j<P.Dim();j++){
          if(Pmat[i][j]){
            perm[i]=j;
            break;
          }
        }

        Vector<PERM_INT_T> perm_tmp;
        while(true){ // Next permutation
          perm_tmp=perm;
          std::sort(&perm_tmp[0],&perm_tmp[0]+perm_tmp.Dim());
          for(size_t i=0;i<perm_tmp.Dim();i++){
            if(perm_tmp[i]!=i) break;
            if(i==perm_tmp.Dim()-1){
              Pvec.push_back(P);
            }
          }

          bool last=false;
          for(size_t i=0;i<P.Dim();i++){
            PERM_INT_T tmp=perm[i];
            for(size_t j=perm[i]+1;j<P.Dim();j++){
              if(Pmat[i][j]){
                perm[i]=j;
                break;
              }
            }
            if(perm[i]>tmp) break;
            for(size_t j=0;j<P.Dim();j++){
              if(Pmat[i][j]){
                perm[i]=j;
                break;
              }
            }
            if(i==P.Dim()-1) last=true;
          }
          if(last) break;
        }
      }
      { // P2vec
        Matrix<long long>& Pmat=P2;
        std::vector<Permutation<long long> >& Pvec=P2vec;

        Permutation<long long> P(Pmat.Dim(0));
        Vector<PERM_INT_T>& perm=P.perm;
        perm.SetZero();

        // First permutation
        for(size_t i=0;i<P.Dim();i++)
        for(size_t j=0;j<P.Dim();j++){
          if(Pmat[i][j]){
            perm[i]=j;
            break;
          }
        }

        Vector<PERM_INT_T> perm_tmp;
        while(true){ // Next permutation
          perm_tmp=perm;
          std::sort(&perm_tmp[0],&perm_tmp[0]+perm_tmp.Dim());
          for(size_t i=0;i<perm_tmp.Dim();i++){
            if(perm_tmp[i]!=i) break;
            if(i==perm_tmp.Dim()-1){
              Pvec.push_back(P);
            }
          }

          bool last=false;
          for(size_t i=0;i<P.Dim();i++){
            PERM_INT_T tmp=perm[i];
            for(size_t j=perm[i]+1;j<P.Dim();j++){
              if(Pmat[i][j]){
                perm[i]=j;
                break;
              }
            }
            if(perm[i]>tmp) break;
            for(size_t j=0;j<P.Dim();j++){
              if(Pmat[i][j]){
                perm[i]=j;
                break;
              }
            }
            if(i==P.Dim()-1) last=true;
          }
          if(last) break;
        }
      }

      { // Find pairs which acutally work (neglect scaling)
        std::vector<Permutation<long long> > P1vec_, P2vec_;
        Matrix<long long>  M1=M11;
        Matrix<long long>  M2=M22;
        for(size_t i=0;i<M1.Dim(0);i++){
          for(size_t j=0;j<M1.Dim(1);j++){
            if(M1[i][j]<0) M1[i][j]=-M1[i][j];
            if(M2[i][j]<0) M2[i][j]=-M2[i][j];
          }
        }

        Matrix<long long> M;
        for(size_t i=0;i<P1vec.size();i++)
        for(size_t j=0;j<P2vec.size();j++){
          M=P1vec[i]*M2*P2vec[j];
          for(size_t k=0;k<M.Dim(0)*M.Dim(1);k++){
            if(M[0][k]!=M1[0][k]) break;
            if(k==M.Dim(0)*M.Dim(1)-1){
              P1vec_.push_back(P1vec[i]);
              P2vec_.push_back(P2vec[j]);
            }
          }
        }

        P1vec=P1vec_;
        P2vec=P2vec_;
      }

      Permutation<T> P1_, P2_;
      { // Find pairs which acutally work
        for(size_t k=0;k<P1vec.size();k++){
          Permutation<long long> P1=P1vec[k];
          Permutation<long long> P2=P2vec[k];
          Matrix<long long>  M1=   M11   ;
          Matrix<long long>  M2=P1*M22*P2;

          Matrix<T> M(M1.Dim(0)*M1.Dim(1)+1,M1.Dim(0)+M1.Dim(1));
          M.SetZero(); M[M1.Dim(0)*M1.Dim(1)][0]=1.0;
          for(size_t i=0;i<M1.Dim(0);i++)
          for(size_t j=0;j<M1.Dim(1);j++){
            size_t k=i*M1.Dim(1)+j;
            M[k][          i]= M1[i][j];
            M[k][M1.Dim(0)+j]=-M2[i][j];
          }
          M=M.pinv();
          { // Construct new permutation
            Permutation<long long> P1_(M1.Dim(0));
            Permutation<long long> P2_(M1.Dim(1));
            for(size_t i=0;i<M1.Dim(0);i++){
              P1_.scal[i]=(M[i][M1.Dim(0)*M1.Dim(1)]>0?1:-1);
            }
            for(size_t i=0;i<M1.Dim(1);i++){
              P2_.scal[i]=(M[M1.Dim(0)+i][M1.Dim(0)*M1.Dim(1)]>0?1:-1);
            }
            P1=P1_*P1 ;
            P2=P2 *P2_;
          }

          bool done=true;
          Matrix<long long> Merr=P1*M22*P2-M11;
          for(size_t i=0;i<Merr.Dim(0)*Merr.Dim(1);i++){
            if(Merr[0][i]){
              done=false;
              break;
            }
          }
          if(done){
            P1_=Permutation<T>(P1.Dim());
            P2_=Permutation<T>(P2.Dim());
            for(size_t i=0;i<P1.Dim();i++){
              P1_.perm[i]=P1.perm[i];
              P1_.scal[i]=P1.scal[i];
            }
            for(size_t i=0;i<P2.Dim();i++){
              P2_.perm[i]=P2.perm[i];
              P2_.scal[i]=P2.scal[i];
            }
            break;
          }
        }
      }

      //std::cout<<P1_<<'\n';
      //std::cout<<P2_<<'\n';
      perm_vec[p_type       ]=P1_.Transpose();
      perm_vec[p_type+C_Perm]=P2_;
    }

    for(size_t i=0;i<2*C_Perm;i++){
      if(perm_vec[i].Dim()==0){
        perm_vec.Resize(0);
        std::cout<<"no-symmetry for: "<<ker_name<<'\n';
        break;
      }
    }
  }

  if(verbose){ // Display kernel information
    std::cout<<"\n";
    std::cout<<"Kernel Name    : "<<ker_name<<'\n';
    std::cout<<"Precision      : "<<(double)eps<<'\n';
    std::cout<<"Symmetry       : "<<(perm_vec.Dim()>0?"yes":"no")<<'\n';
    std::cout<<"Scale Invariant: "<<(homogen?"yes":"no")<<'\n';
    if(homogen){
      std::cout<<"Scaling Matrix :\n";
      Matrix<T> Src(ker_dim[0],1);
      Matrix<T> Trg(1,ker_dim[1]);
      for(size_t i=0;i<ker_dim[0];i++) Src[i][0]=pow(2.0,src_scal[i]);
      for(size_t i=0;i<ker_dim[1];i++) Trg[0][i]=pow(2.0,trg_scal[i]);
      std::cout<<Src*Trg;
    }

    std::cout<<"Error          : ";
    for(T rad=1.0; rad>1.0e-2; rad*=0.5){ // Accuracy of multipole expansion
      int m=8; // multipole order

      std::vector<T> equiv_surf;
      std::vector<T> check_surf;
      for(int i0=0;i0<m;i0++){
        for(int i1=0;i1<m;i1++){
          for(int i2=0;i2<m;i2++){
            if(i0==  0 || i1==  0 || i2==  0 ||
               i0==m-1 || i1==m-1 || i2==m-1){

              // Range: [-1/3,1/3]^3
              T x=((T)2*i0-(m-1))/(m-1)/3;
              T y=((T)2*i1-(m-1))/(m-1)/3;
              T z=((T)2*i2-(m-1))/(m-1)/3;

              equiv_surf.push_back(x*RAD0*rad);
              equiv_surf.push_back(y*RAD0*rad);
              equiv_surf.push_back(z*RAD0*rad);

              check_surf.push_back(x*RAD1*rad);
              check_surf.push_back(y*RAD1*rad);
              check_surf.push_back(z*RAD1*rad);
            }
          }
        }
      }
      size_t n_equiv=equiv_surf.size()/COORD_DIM;
      size_t n_check=equiv_surf.size()/COORD_DIM;

      size_t n_src=m;
      size_t n_trg=m;
      std::vector<T> src_coord;
      std::vector<T> trg_coord;
      for(size_t i=0;i<n_src*COORD_DIM;i++){
        src_coord.push_back((2*drand48()-1)/3*rad);
      }
      for(size_t i=0;i<n_trg;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=sqrt(x*x+y*y+z*z);
        }while(r==0.0);
        trg_coord.push_back(x/r*sqrt((T)COORD_DIM)*rad);
        trg_coord.push_back(y/r*sqrt((T)COORD_DIM)*rad);
        trg_coord.push_back(z/r*sqrt((T)COORD_DIM)*rad);
      }

      Matrix<T> M_s2c(n_src*ker_dim[0],n_check*ker_dim[1]);
      BuildMatrix( &src_coord[0], n_src,
                  &check_surf[0], n_check, &(M_s2c[0][0]));

      Matrix<T> M_e2c(n_equiv*ker_dim[0],n_check*ker_dim[1]);
      BuildMatrix(&equiv_surf[0], n_equiv,
                  &check_surf[0], n_check, &(M_e2c[0][0]));
      Matrix<T> M_c2e=M_e2c.pinv();

      Matrix<T> M_e2t(n_equiv*ker_dim[0],n_trg*ker_dim[1]);
      BuildMatrix(&equiv_surf[0], n_equiv,
                   &trg_coord[0], n_trg  , &(M_e2t[0][0]));

      Matrix<T> M_s2t(n_src*ker_dim[0],n_trg*ker_dim[1]);
      BuildMatrix( &src_coord[0], n_src,
                   &trg_coord[0], n_trg  , &(M_s2t[0][0]));

      Matrix<T> M=M_s2c*M_c2e*M_e2t-M_s2t;
      T max_error=0, max_value=0;
      for(size_t i=0;i<M.Dim(0);i++)
      for(size_t j=0;j<M.Dim(1);j++){
        max_error=std::max<T>(max_error,fabs(M    [i][j]));
        max_value=std::max<T>(max_value,fabs(M_s2t[i][j]));
      }

      std::cout<<(double)(max_error/max_value)<<' ';
      if(homogen) break;
    }
    std::cout<<"\n";
    std::cout<<"\n";
  }

  { // Initialize auxiliary FMM kernels
    if(!k_s2m) k_s2m=this;
    if(!k_s2l) k_s2l=this;
    if(!k_s2t) k_s2t=this;
    if(!k_m2m) k_m2m=this;
    if(!k_m2l) k_m2l=this;
    if(!k_m2t) k_m2t=this;
    if(!k_l2l) k_l2l=this;
    if(!k_l2t) k_l2t=this;

    assert(k_s2t->ker_dim[0]==ker_dim[0]);
    assert(k_s2m->ker_dim[0]==k_s2l->ker_dim[0]);
    assert(k_s2m->ker_dim[0]==k_s2t->ker_dim[0]);
    assert(k_m2m->ker_dim[0]==k_m2l->ker_dim[0]);
    assert(k_m2m->ker_dim[0]==k_m2t->ker_dim[0]);
    assert(k_l2l->ker_dim[0]==k_l2t->ker_dim[0]);

    assert(k_s2t->ker_dim[1]==ker_dim[1]);
    assert(k_s2m->ker_dim[1]==k_m2m->ker_dim[1]);
    assert(k_s2l->ker_dim[1]==k_l2l->ker_dim[1]);
    assert(k_m2l->ker_dim[1]==k_l2l->ker_dim[1]);
    assert(k_s2t->ker_dim[1]==k_m2t->ker_dim[1]);
    assert(k_s2t->ker_dim[1]==k_l2t->ker_dim[1]);

    k_s2m->Initialize(verbose);
    k_s2l->Initialize(verbose);
    k_s2t->Initialize(verbose);
    k_m2m->Initialize(verbose);
    k_m2l->Initialize(verbose);
    k_m2t->Initialize(verbose);
    k_l2l->Initialize(verbose);
    k_l2t->Initialize(verbose);
  }
}

/**
 * \brief Compute the transformation matrix (on the source strength vector)
 * to get potential at target coordinates due to sources at the given
 * coordinates.
 * \param[in] r_src Coordinates of source points.
 * \param[in] src_cnt Number of source points.
 * \param[in] r_trg Coordinates of target points.
 * \param[in] trg_cnt Number of target points.
 * \param[out] k_out Output array with potential values.
 */
template <class T>
void Kernel<T>::BuildMatrix(T* r_src, int src_cnt,
                 T* r_trg, int trg_cnt, T* k_out) const{
  int dim=3; //Only supporting 3D
  memset(k_out, 0, src_cnt*ker_dim[0]*trg_cnt*ker_dim[1]*sizeof(T));
  for(int i=0;i<src_cnt;i++) //TODO Optimize this.
    for(int j=0;j<ker_dim[0];j++){
      std::vector<T> v_src(ker_dim[0],0);
      v_src[j]=1.0;
      ker_poten(&r_src[i*dim], 1, &v_src[0], 1, r_trg, trg_cnt,
                &k_out[(i*ker_dim[0]+j)*trg_cnt*ker_dim[1]], NULL);
    }
}

////////////////////////////////////////////////////////////////////////////////
////////                   LAPLACE KERNEL                               ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Poisson's equation. Kernel tensor
 * dimension = 1x1.
 */
template <class T>
void laplace_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(12*dof));
#endif

  const T OOFP = 1.0/(4.0*const_pi<T>());
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p=0;
      for(int s=0;s<src_cnt;s++){
        T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
        T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
        T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
        T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
        if (invR!=0) invR = 1.0/sqrt(invR);
        p += v_src[s*dof+i]*invR;
      }
      k_out[t*dof+i] += p*OOFP;
    }
  }
}

template <class T>
void laplace_poten_(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
//void laplace_poten(T* r_src_, int src_cnt, T* v_src_, int dof, T* r_trg_, int trg_cnt, T* k_out_){
//  int dim=3; //Only supporting 3D
//  T* r_src=mem::aligned_malloc<T>(src_cnt*dim);
//  T* r_trg=mem::aligned_malloc<T>(trg_cnt*dim);
//  T* v_src=mem::aligned_malloc<T>(src_cnt    );
//  T* k_out=mem::aligned_malloc<T>(trg_cnt    );
//  mem::memcopy(r_src,r_src_,src_cnt*dim*sizeof(T));
//  mem::memcopy(r_trg,r_trg_,trg_cnt*dim*sizeof(T));
//  mem::memcopy(v_src,v_src_,src_cnt    *sizeof(T));
//  mem::memcopy(k_out,k_out_,trg_cnt    *sizeof(T));

  #define EVAL_BLKSZ 32
  #define MAX_DOF 100
  //Compute source to target interactions.
  const T OOFP = 1.0/(4.0*const_pi<T>());

  if(dof==1){
    for (int t_=0; t_<trg_cnt; t_+=EVAL_BLKSZ)
    for (int s_=0; s_<src_cnt; s_+=EVAL_BLKSZ){
      int src_blk=s_+EVAL_BLKSZ; src_blk=(src_blk>src_cnt?src_cnt:src_blk);
      int trg_blk=t_+EVAL_BLKSZ; trg_blk=(trg_blk>trg_cnt?trg_cnt:trg_blk);
      for(int t=t_;t<trg_blk;t++){
        T p=0;
        for(int s=s_;s<src_blk;s++){
          T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
          T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
          T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
          T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
          if (invR!=0) invR = 1.0/sqrt(invR);
          p += v_src[s]*invR;
        }
        k_out[t] += p*OOFP;
      }
    }
  }else if(dof==2){
    T p[MAX_DOF];
    for (int t_=0; t_<trg_cnt; t_+=EVAL_BLKSZ)
    for (int s_=0; s_<src_cnt; s_+=EVAL_BLKSZ){
      int src_blk=s_+EVAL_BLKSZ; src_blk=(src_blk>src_cnt?src_cnt:src_blk);
      int trg_blk=t_+EVAL_BLKSZ; trg_blk=(trg_blk>trg_cnt?trg_cnt:trg_blk);
      for(int t=t_;t<trg_blk;t++){
        p[0]=0; p[1]=0;
        for(int s=s_;s<src_blk;s++){
          T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
          T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
          T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
          T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
          if (invR!=0) invR = 1.0/sqrt(invR);
          p[0] += v_src[s*dof+0]*invR;
          p[1] += v_src[s*dof+1]*invR;
        }
        k_out[t*dof+0] += p[0]*OOFP;
        k_out[t*dof+1] += p[1]*OOFP;
      }
    }
  }else if(dof==3){
    T p[MAX_DOF];
    for (int t_=0; t_<trg_cnt; t_+=EVAL_BLKSZ)
    for (int s_=0; s_<src_cnt; s_+=EVAL_BLKSZ){
      int src_blk=s_+EVAL_BLKSZ; src_blk=(src_blk>src_cnt?src_cnt:src_blk);
      int trg_blk=t_+EVAL_BLKSZ; trg_blk=(trg_blk>trg_cnt?trg_cnt:trg_blk);
      for(int t=t_;t<trg_blk;t++){
        p[0]=0; p[1]=0; p[2]=0;
        for(int s=s_;s<src_blk;s++){
          T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
          T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
          T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
          T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
          if (invR!=0) invR = 1.0/sqrt(invR);
          p[0] += v_src[s*dof+0]*invR;
          p[1] += v_src[s*dof+1]*invR;
          p[2] += v_src[s*dof+2]*invR;
        }
        k_out[t*dof+0] += p[0]*OOFP;
        k_out[t*dof+1] += p[1]*OOFP;
        k_out[t*dof+2] += p[2]*OOFP;
      }
    }
  }else{
    T p[MAX_DOF];
    for (int t_=0; t_<trg_cnt; t_+=EVAL_BLKSZ)
    for (int s_=0; s_<src_cnt; s_+=EVAL_BLKSZ){
      int src_blk=s_+EVAL_BLKSZ; src_blk=(src_blk>src_cnt?src_cnt:src_blk);
      int trg_blk=t_+EVAL_BLKSZ; trg_blk=(trg_blk>trg_cnt?trg_cnt:trg_blk);
      for(int t=t_;t<trg_blk;t++){
        for(int i=0;i<dof;i++) p[i]=0;
        for(int s=s_;s<src_blk;s++){
          T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
          T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
          T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
          T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
          if (invR!=0) invR = 1.0/sqrt(invR);
          for(int i=0;i<dof;i++)
            p[i] += v_src[s*dof+i]*invR;
        }
        for(int i=0;i<dof;i++)
          k_out[t*dof+i] += p[i]*OOFP;
      }
    }
  }
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(10+2*dof));
#endif
  #undef MAX_DOF
  #undef EVAL_BLKSZ

//  for (int t=0; t<trg_cnt; t++)
//    k_out_[t] += k_out[t];
//  mem::aligned_free(r_src);
//  mem::aligned_free(r_trg);
//  mem::aligned_free(v_src);
//  mem::aligned_free(k_out);
}

// Laplace double layer potential.
template <class T>
void laplace_dbl_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(19*dof));
#endif

  const T OOFP = -1.0/(4.0*const_pi<T>());
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p=0;
      for(int s=0;s<src_cnt;s++){
        T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
        T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
        T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
        T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
        if (invR!=0) invR = 1.0/sqrt(invR);
        p = v_src[(s*dof+i)*4+3]*invR*invR*invR;
        k_out[t*dof+i] += p*OOFP*( dX_reg*v_src[(s*dof+i)*4+0] +
                                   dY_reg*v_src[(s*dof+i)*4+1] +
                                   dZ_reg*v_src[(s*dof+i)*4+2] );
      }
    }
  }
}

// Laplace grdient kernel.
template <class T>
void laplace_grad(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(10+12*dof));
#endif

  const T OOFP = -1.0/(4.0*const_pi<T>());
  if(dof==1){
    for(int t=0;t<trg_cnt;t++){
      T p=0;
      for(int s=0;s<src_cnt;s++){
        T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
        T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
        T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
        T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
        if (invR!=0) invR = 1.0/sqrt(invR);
        p = v_src[s]*invR*invR*invR;
        k_out[(t)*3+0] += p*OOFP*dX_reg;
        k_out[(t)*3+1] += p*OOFP*dY_reg;
        k_out[(t)*3+2] += p*OOFP*dZ_reg;
      }
    }
  }else if(dof==2){
    for(int t=0;t<trg_cnt;t++){
      T p=0;
      for(int s=0;s<src_cnt;s++){
        T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
        T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
        T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
        T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
        if (invR!=0) invR = 1.0/sqrt(invR);

        p = v_src[s*dof+0]*invR*invR*invR;
        k_out[(t*dof+0)*3+0] += p*OOFP*dX_reg;
        k_out[(t*dof+0)*3+1] += p*OOFP*dY_reg;
        k_out[(t*dof+0)*3+2] += p*OOFP*dZ_reg;

        p = v_src[s*dof+1]*invR*invR*invR;
        k_out[(t*dof+1)*3+0] += p*OOFP*dX_reg;
        k_out[(t*dof+1)*3+1] += p*OOFP*dY_reg;
        k_out[(t*dof+1)*3+2] += p*OOFP*dZ_reg;
      }
    }
  }else if(dof==3){
    for(int t=0;t<trg_cnt;t++){
      T p=0;
      for(int s=0;s<src_cnt;s++){
        T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
        T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
        T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
        T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
        if (invR!=0) invR = 1.0/sqrt(invR);

        p = v_src[s*dof+0]*invR*invR*invR;
        k_out[(t*dof+0)*3+0] += p*OOFP*dX_reg;
        k_out[(t*dof+0)*3+1] += p*OOFP*dY_reg;
        k_out[(t*dof+0)*3+2] += p*OOFP*dZ_reg;

        p = v_src[s*dof+1]*invR*invR*invR;
        k_out[(t*dof+1)*3+0] += p*OOFP*dX_reg;
        k_out[(t*dof+1)*3+1] += p*OOFP*dY_reg;
        k_out[(t*dof+1)*3+2] += p*OOFP*dZ_reg;

        p = v_src[s*dof+2]*invR*invR*invR;
        k_out[(t*dof+2)*3+0] += p*OOFP*dX_reg;
        k_out[(t*dof+2)*3+1] += p*OOFP*dY_reg;
        k_out[(t*dof+2)*3+2] += p*OOFP*dZ_reg;
      }
    }
  }else{
    for(int t=0;t<trg_cnt;t++){
      for(int i=0;i<dof;i++){
        T p=0;
        for(int s=0;s<src_cnt;s++){
          T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
          T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
          T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
          T invR = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
          if (invR!=0) invR = 1.0/sqrt(invR);
          p = v_src[s*dof+i]*invR*invR*invR;
          k_out[(t*dof+i)*3+0] += p*OOFP*dX_reg;
          k_out[(t*dof+i)*3+1] += p*OOFP*dY_reg;
          k_out[(t*dof+i)*3+2] += p*OOFP*dZ_reg;
        }
      }
    }
  }
}

#ifndef __MIC__
#ifdef USE_SSE
namespace
{
#define IDEAL_ALIGNMENT 16
#define SIMD_LEN (int)(IDEAL_ALIGNMENT / sizeof(double))
#define DECL_SIMD_ALIGNED  __declspec(align(IDEAL_ALIGNMENT))
  void laplaceSSE(
      const int ns,
      const int nt,
      const double *sx,
      const double *sy,
      const double *sz,
      const double *tx,
      const double *ty,
      const double *tz,
      const double *srcDen,
      double *trgVal)
  {
    if ( size_t(sx)%IDEAL_ALIGNMENT || size_t(sy)%IDEAL_ALIGNMENT || size_t(sz)%IDEAL_ALIGNMENT )
      abort();

    double OOFP = 1.0/(4.0*const_pi<double>());
    __m128d temp;

    double aux_arr[SIMD_LEN+1];
    double *tempval;
    // if aux_arr is misaligned
    if (size_t(aux_arr)%IDEAL_ALIGNMENT) tempval = aux_arr + 1;
    else tempval = aux_arr;
    if (size_t(tempval)%IDEAL_ALIGNMENT) abort();

    /*! One over four pi */
    __m128d oofp = _mm_set1_pd (OOFP);
    __m128d half = _mm_set1_pd (0.5);
    __m128d opf = _mm_set1_pd (1.5);
    __m128d zero = _mm_setzero_pd ();

    // loop over sources
    int i = 0;
    for (; i < nt; i++) {
      temp = _mm_setzero_pd();

      __m128d txi = _mm_load1_pd (&tx[i]);
      __m128d tyi = _mm_load1_pd (&ty[i]);
      __m128d tzi = _mm_load1_pd (&tz[i]);
      int j = 0;
      // Load and calculate in groups of SIMD_LEN
      for (; j + SIMD_LEN <= ns; j+=SIMD_LEN) {
        __m128d sxj = _mm_load_pd (&sx[j]);
        __m128d syj = _mm_load_pd (&sy[j]);
        __m128d szj = _mm_load_pd (&sz[j]);
        __m128d sden = _mm_set_pd (srcDen[j+1],   srcDen[j]);

        __m128d dX, dY, dZ;
        __m128d dR2;
        __m128d S;

        dX = _mm_sub_pd(txi , sxj);
        dY = _mm_sub_pd(tyi , syj);
        dZ = _mm_sub_pd(tzi , szj);

        sxj = _mm_mul_pd(dX, dX);
        syj = _mm_mul_pd(dY, dY);
        szj = _mm_mul_pd(dZ, dZ);

        dR2 = _mm_add_pd(sxj, syj);
        dR2 = _mm_add_pd(szj, dR2);
        __m128d reqzero = _mm_cmpeq_pd (dR2, zero);

        __m128d xhalf = _mm_mul_pd (half, dR2);
        __m128 dR2_s  =  _mm_cvtpd_ps(dR2);
        __m128 S_s    = _mm_rsqrt_ps(dR2_s);
        __m128d S_d   = _mm_cvtps_pd(S_s);
        // To handle the condition when src and trg coincide
        S_d = _mm_andnot_pd (reqzero, S_d);

        S = _mm_mul_pd (S_d, S_d);
        S = _mm_mul_pd (S, xhalf);
        S = _mm_sub_pd (opf, S);
        S = _mm_mul_pd (S, S_d);

        sden = _mm_mul_pd (sden, S);
        temp = _mm_add_pd (sden, temp);
      }
      temp = _mm_mul_pd (temp, oofp);

      _mm_store_pd(tempval, temp);
      for (int k = 0; k < SIMD_LEN; k++) {
        trgVal[i]   += tempval[k];
      }

      for (; j < ns; j++) {
        double x = tx[i] - sx[j];
        double y = ty[i] - sy[j];
        double z = tz[i] - sz[j];
        double r2 = x*x + y*y + z*z;
        double r = sqrt(r2);
        double invdr;
        if (r == 0)
          invdr = 0;
        else
          invdr = 1/r;
        double den = srcDen[j];
        trgVal[i] += den*invdr*OOFP;
      }
    }

    return;
  }

  void laplaceDblSSE(
      const int ns,
      const int nt,
      const double *sx,
      const double *sy,
      const double *sz,
      const double *tx,
      const double *ty,
      const double *tz,
      const double *srcDen,
      double *trgVal)
  {
    if ( size_t(sx)%IDEAL_ALIGNMENT || size_t(sy)%IDEAL_ALIGNMENT || size_t(sz)%IDEAL_ALIGNMENT )
      abort();

    double OOFP = 1.0/(4.0*const_pi<double>());
    __m128d temp;

    double aux_arr[SIMD_LEN+1];
    double *tempval;
    // if aux_arr is misaligned
    if (size_t(aux_arr)%IDEAL_ALIGNMENT) tempval = aux_arr + 1;
    else tempval = aux_arr;
    if (size_t(tempval)%IDEAL_ALIGNMENT) abort();

    /*! One over four pi */
    __m128d oofp = _mm_set1_pd (OOFP);
    __m128d half = _mm_set1_pd (0.5);
    __m128d opf = _mm_set1_pd (1.5);
    __m128d zero = _mm_setzero_pd ();

    // loop over sources
    int i = 0;
    for (; i < nt; i++) {
      temp = _mm_setzero_pd();

      __m128d txi = _mm_load1_pd (&tx[i]);
      __m128d tyi = _mm_load1_pd (&ty[i]);
      __m128d tzi = _mm_load1_pd (&tz[i]);
      int j = 0;
      // Load and calculate in groups of SIMD_LEN
      for (; j + SIMD_LEN <= ns; j+=SIMD_LEN) {
        __m128d sxj = _mm_load_pd (&sx[j]);
        __m128d syj = _mm_load_pd (&sy[j]);
        __m128d szj = _mm_load_pd (&sz[j]);

        __m128d snormx = _mm_set_pd (srcDen[(j+1)*4+0],   srcDen[j*4+0]);
        __m128d snormy = _mm_set_pd (srcDen[(j+1)*4+1],   srcDen[j*4+1]);
        __m128d snormz = _mm_set_pd (srcDen[(j+1)*4+2],   srcDen[j*4+2]);
        __m128d sden   = _mm_set_pd (srcDen[(j+1)*4+3],   srcDen[j*4+3]);

        __m128d dX, dY, dZ;
        __m128d dR2;
        __m128d S;
        __m128d S2;
        __m128d S3;

        dX = _mm_sub_pd(txi , sxj);
        dY = _mm_sub_pd(tyi , syj);
        dZ = _mm_sub_pd(tzi , szj);

        sxj = _mm_mul_pd(dX, dX);
        syj = _mm_mul_pd(dY, dY);
        szj = _mm_mul_pd(dZ, dZ);

        dR2 = _mm_add_pd(sxj, syj);
        dR2 = _mm_add_pd(szj, dR2);
        __m128d reqzero = _mm_cmpeq_pd (dR2, zero);

        __m128d xhalf = _mm_mul_pd (half, dR2);
        __m128 dR2_s  =  _mm_cvtpd_ps(dR2);
        __m128 S_s    = _mm_rsqrt_ps(dR2_s);
        __m128d S_d   = _mm_cvtps_pd(S_s);
        // To handle the condition when src and trg coincide
        S_d = _mm_andnot_pd (reqzero, S_d);

        S = _mm_mul_pd (S_d, S_d);
        S = _mm_mul_pd (S, xhalf);
        S = _mm_sub_pd (opf, S);
        S = _mm_mul_pd (S, S_d);
        S2 = _mm_mul_pd (S, S);
        S3 = _mm_mul_pd (S2, S);

        __m128d S3_sden=_mm_mul_pd(S3, sden);

        __m128d dot_sum = _mm_add_pd(_mm_mul_pd(snormx,dX),_mm_mul_pd(snormy,dY));
        dot_sum = _mm_add_pd(dot_sum,_mm_mul_pd(snormz,dZ));
        temp = _mm_add_pd(_mm_mul_pd(S3_sden,dot_sum),temp);
      }
      temp = _mm_mul_pd (temp, oofp);
      _mm_store_pd(tempval, temp);

      for (int k = 0; k < SIMD_LEN; k++) {
        trgVal[i] += tempval[k];
      }

      for (; j < ns; j++) {
        double x = tx[i] - sx[j];
        double y = ty[i] - sy[j];
        double z = tz[i] - sz[j];
        double r2 = x*x + y*y + z*z;
        double r = sqrt(r2);
        double invdr;
        if (r == 0)
          invdr = 0;
        else
          invdr = 1/r;
        double invdr2=invdr*invdr;
        double invdr3=invdr2*invdr;

        double dot_sum = x*srcDen[j*4+0] + y*srcDen[j*4+1] + z*srcDen[j*4+2];
        trgVal[i] += OOFP*invdr3*x*srcDen[j*4+3]*dot_sum;
      }
    }

    return;
  }

  void laplaceGradSSE(
      const int ns,
      const int nt,
      const double *sx,
      const double *sy,
      const double *sz,
      const double *tx,
      const double *ty,
      const double *tz,
      const double *srcDen,
      double *trgVal)
  {
    if ( size_t(sx)%IDEAL_ALIGNMENT || size_t(sy)%IDEAL_ALIGNMENT || size_t(sz)%IDEAL_ALIGNMENT )
      abort();

    double OOFP = 1.0/(4.0*const_pi<double>());
    __m128d tempx; __m128d tempy; __m128d tempz;

    double aux_arr[3*SIMD_LEN+1];
    double *tempvalx, *tempvaly, *tempvalz;
    // if aux_arr is misaligned
    if (size_t(aux_arr)%IDEAL_ALIGNMENT) tempvalx = aux_arr + 1;
    else tempvalx = aux_arr;
    if (size_t(tempvalx)%IDEAL_ALIGNMENT) abort();

    tempvaly=tempvalx+SIMD_LEN;
    tempvalz=tempvaly+SIMD_LEN;

    /*! One over four pi */
    __m128d oofp = _mm_set1_pd (OOFP);
    __m128d half = _mm_set1_pd (0.5);
    __m128d opf = _mm_set1_pd (1.5);
    __m128d zero = _mm_setzero_pd ();

    // loop over sources
    int i = 0;
    for (; i < nt; i++) {
      tempx = _mm_setzero_pd();
      tempy = _mm_setzero_pd();
      tempz = _mm_setzero_pd();

      __m128d txi = _mm_load1_pd (&tx[i]);
      __m128d tyi = _mm_load1_pd (&ty[i]);
      __m128d tzi = _mm_load1_pd (&tz[i]);
      int j = 0;
      // Load and calculate in groups of SIMD_LEN
      for (; j + SIMD_LEN <= ns; j+=SIMD_LEN) {
        __m128d sxj = _mm_load_pd (&sx[j]);
        __m128d syj = _mm_load_pd (&sy[j]);
        __m128d szj = _mm_load_pd (&sz[j]);
        __m128d sden = _mm_set_pd (srcDen[j+1],   srcDen[j]);

        __m128d dX, dY, dZ;
        __m128d dR2;
        __m128d S;
        __m128d S2;
        __m128d S3;

        dX = _mm_sub_pd(txi , sxj);
        dY = _mm_sub_pd(tyi , syj);
        dZ = _mm_sub_pd(tzi , szj);

        sxj = _mm_mul_pd(dX, dX);
        syj = _mm_mul_pd(dY, dY);
        szj = _mm_mul_pd(dZ, dZ);

        dR2 = _mm_add_pd(sxj, syj);
        dR2 = _mm_add_pd(szj, dR2);
        __m128d reqzero = _mm_cmpeq_pd (dR2, zero);

        __m128d xhalf = _mm_mul_pd (half, dR2);
        __m128 dR2_s  =  _mm_cvtpd_ps(dR2);
        __m128 S_s    = _mm_rsqrt_ps(dR2_s);
        __m128d S_d   = _mm_cvtps_pd(S_s);
        // To handle the condition when src and trg coincide
        S_d = _mm_andnot_pd (reqzero, S_d);

        S = _mm_mul_pd (S_d, S_d);
        S = _mm_mul_pd (S, xhalf);
        S = _mm_sub_pd (opf, S);
        S = _mm_mul_pd (S, S_d);
        S2 = _mm_mul_pd (S, S);
        S3 = _mm_mul_pd (S2, S);

        __m128d S3_sden=_mm_mul_pd(S3, sden);
        tempx = _mm_add_pd(_mm_mul_pd(S3_sden,dX),tempx);
        tempy = _mm_add_pd(_mm_mul_pd(S3_sden,dY),tempy);
        tempz = _mm_add_pd(_mm_mul_pd(S3_sden,dZ),tempz);

      }
      tempx = _mm_mul_pd (tempx, oofp);
      tempy = _mm_mul_pd (tempy, oofp);
      tempz = _mm_mul_pd (tempz, oofp);

      _mm_store_pd(tempvalx, tempx);
      _mm_store_pd(tempvaly, tempy);
      _mm_store_pd(tempvalz, tempz);

      for (int k = 0; k < SIMD_LEN; k++) {
        trgVal[i*3  ] += tempvalx[k];
        trgVal[i*3+1] += tempvaly[k];
        trgVal[i*3+2] += tempvalz[k];
      }

      for (; j < ns; j++) {
        double x = tx[i] - sx[j];
        double y = ty[i] - sy[j];
        double z = tz[i] - sz[j];
        double r2 = x*x + y*y + z*z;
        double r = sqrt(r2);
        double invdr;
        if (r == 0)
          invdr = 0;
        else
          invdr = 1/r;
        double invdr2=invdr*invdr;
        double invdr3=invdr2*invdr;

        trgVal[i*3  ] += OOFP*invdr3*x*srcDen[j];
        trgVal[i*3+1] += OOFP*invdr3*y*srcDen[j];
        trgVal[i*3+2] += OOFP*invdr3*z*srcDen[j];
      }
    }

    return;
  }
#undef SIMD_LEN

#define X(s,k) (s)[(k)*COORD_DIM]
#define Y(s,k) (s)[(k)*COORD_DIM+1]
#define Z(s,k) (s)[(k)*COORD_DIM+2]
  void laplaceSSEShuffle(const int ns, const int nt, float  const src[], float  const trg[], float  const den[], float  pot[], mem::MemoryManager* mem_mgr=NULL)
  {
    // TODO
  }

  void laplaceSSEShuffle(const int ns, const int nt, double const src[], double const trg[], double const den[], double pot[], mem::MemoryManager* mem_mgr=NULL)
  {
    double* buff=NULL;
    buff=mem::aligned_new<double>((ns+1+nt)*3,mem_mgr);

    double* buff_=buff;
    pvfmm::Vector<double> xs(ns+1,buff_,false); buff_+=ns+1;
    pvfmm::Vector<double> ys(ns+1,buff_,false); buff_+=ns+1;
    pvfmm::Vector<double> zs(ns+1,buff_,false); buff_+=ns+1;

    pvfmm::Vector<double> xt(nt  ,buff_,false); buff_+=nt  ;
    pvfmm::Vector<double> yt(nt  ,buff_,false); buff_+=nt  ;
    pvfmm::Vector<double> zt(nt  ,buff_,false); buff_+=nt  ;

    //std::vector<double> xs(ns+1);
    //std::vector<double> ys(ns+1);
    //std::vector<double> zs(ns+1);

    //std::vector<double> xt(nt  );
    //std::vector<double> yt(nt  );
    //std::vector<double> zt(nt  );

    int x_shift = size_t(&xs[0]) % IDEAL_ALIGNMENT ? 1:0;
    int y_shift = size_t(&ys[0]) % IDEAL_ALIGNMENT ? 1:0;
    int z_shift = size_t(&zs[0]) % IDEAL_ALIGNMENT ? 1:0;

    //1. reshuffle memory
    for (int k =0;k<ns;k++){
      xs[k+x_shift]=X(src,k);
      ys[k+y_shift]=Y(src,k);
      zs[k+z_shift]=Z(src,k);
    }
    for (int k=0;k<nt;k++){
      xt[k]=X(trg,k);
      yt[k]=Y(trg,k);
      zt[k]=Z(trg,k);
    }

    //2. perform caclulation
    laplaceSSE(ns,nt,&xs[x_shift],&ys[y_shift],&zs[z_shift],&xt[0],&yt[0],&zt[0],den,pot);

    mem::aligned_delete<double>(buff,mem_mgr);
    return;
  }

  void laplaceDblSSEShuffle(const int ns, const int nt, float  const src[], float  const trg[], float  const den[], float  pot[], mem::MemoryManager* mem_mgr=NULL)
  {
    // TODO
  }

  void laplaceDblSSEShuffle(const int ns, const int nt, double const src[], double const trg[], double const den[], double pot[], mem::MemoryManager* mem_mgr=NULL)
  {
    std::vector<double> xs(ns+1);   std::vector<double> xt(nt);
    std::vector<double> ys(ns+1);   std::vector<double> yt(nt);
    std::vector<double> zs(ns+1);   std::vector<double> zt(nt);

    int x_shift = size_t(&xs[0]) % IDEAL_ALIGNMENT ? 1:0;
    int y_shift = size_t(&ys[0]) % IDEAL_ALIGNMENT ? 1:0;
    int z_shift = size_t(&zs[0]) % IDEAL_ALIGNMENT ? 1:0;

    //1. reshuffle memory
    for (int k =0;k<ns;k++){
      xs[k+x_shift]=X(src,k);
      ys[k+y_shift]=Y(src,k);
      zs[k+z_shift]=Z(src,k);
    }
    for (int k=0;k<nt;k++){
      xt[k]=X(trg,k);
      yt[k]=Y(trg,k);
      zt[k]=Z(trg,k);
    }

    //2. perform caclulation
    laplaceDblSSE(ns,nt,&xs[x_shift],&ys[y_shift],&zs[z_shift],&xt[0],&yt[0],&zt[0],den,pot);
    return;
  }

  void laplaceGradSSEShuffle(const int ns, const int nt, float  const src[], float  const trg[], float  const den[], float  pot[], mem::MemoryManager* mem_mgr=NULL)
  {
    // TODO
  }

  void laplaceGradSSEShuffle(const int ns, const int nt, double const src[], double const trg[], double const den[], double pot[], mem::MemoryManager* mem_mgr=NULL)
  {
    int tid=omp_get_thread_num();
    static std::vector<std::vector<double> > xs_(100);   static std::vector<std::vector<double> > xt_(100);
    static std::vector<std::vector<double> > ys_(100);   static std::vector<std::vector<double> > yt_(100);
    static std::vector<std::vector<double> > zs_(100);   static std::vector<std::vector<double> > zt_(100);

    std::vector<double>& xs=xs_[tid];   std::vector<double>& xt=xt_[tid];
    std::vector<double>& ys=ys_[tid];   std::vector<double>& yt=yt_[tid];
    std::vector<double>& zs=zs_[tid];   std::vector<double>& zt=zt_[tid];
    xs.resize(ns+1); xt.resize(nt);
    ys.resize(ns+1); yt.resize(nt);
    zs.resize(ns+1); zt.resize(nt);

    int x_shift = size_t(&xs[0]) % IDEAL_ALIGNMENT ? 1:0;
    int y_shift = size_t(&ys[0]) % IDEAL_ALIGNMENT ? 1:0;
    int z_shift = size_t(&zs[0]) % IDEAL_ALIGNMENT ? 1:0;

    //1. reshuffle memory
    for (int k =0;k<ns;k++){
      xs[k+x_shift]=X(src,k);
      ys[k+y_shift]=Y(src,k);
      zs[k+z_shift]=Z(src,k);
    }
    for (int k=0;k<nt;k++){
      xt[k]=X(trg,k);
      yt[k]=Y(trg,k);
      zt[k]=Z(trg,k);
    }

    //2. perform caclulation
    laplaceGradSSE(ns,nt,&xs[x_shift],&ys[y_shift],&zs[z_shift],&xt[0],&yt[0],&zt[0],den,pot);
    return;
  }
#undef X
#undef Y
#undef Z

#undef IDEAL_ALIGNMENT
#undef DECL_SIMD_ALIGNED
}

template <>
void laplace_poten<double>(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(12*dof));

  if(dof==1){
    laplaceSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src, k_out, mem_mgr);
    return;
  }
}

template <>
void laplace_dbl_poten<double>(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(19*dof));

  if(dof==1){
    laplaceDblSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src, k_out, mem_mgr);
    return;
  }
}

template <>
void laplace_grad<double>(double* r_src, int src_cnt, double* v_src, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(10+12*dof));

  if(dof==1){
    laplaceGradSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src, k_out, mem_mgr);
    return;
  }
}
#endif
#endif


////////////////////////////////////////////////////////////////////////////////
////////                   STOKES KERNEL                             ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Stokes's equation. Kernel tensor
 * dimension = 3x3.
 */
template <class T>
void stokes_vel(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(28*dof));
#endif

  const T mu=1.0;
  const T OOEPMU = 1.0/(8.0*const_pi<T>()*mu);
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[3]={0,0,0};
      for(int s=0;s<src_cnt;s++){
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R!=0){
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T v_src[3]={v_src_[(s*dof+i)*3  ],
                      v_src_[(s*dof+i)*3+1],
                      v_src_[(s*dof+i)*3+2]};
          T inner_prod=(v_src[0]*dR[0] +
                        v_src[1]*dR[1] +
                        v_src[2]*dR[2])* invR2;
          p[0] += (v_src[0] + dR[0]*inner_prod)*invR;
          p[1] += (v_src[1] + dR[1]*inner_prod)*invR;
          p[2] += (v_src[2] + dR[2]*inner_prod)*invR;
        }
      }
      k_out[(t*dof+i)*3+0] += p[0]*OOEPMU;
      k_out[(t*dof+i)*3+1] += p[1]*OOEPMU;
      k_out[(t*dof+i)*3+2] += p[2]*OOEPMU;
    }
  }
}

template <class T>
void stokes_sym_dip(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(47*dof));
#endif

  const T mu=1.0;
  const T OOEPMU = -1.0/(8.0*const_pi<T>()*mu);
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[3]={0,0,0};
      for(int s=0;s<src_cnt;s++){
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R!=0){
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;

          T* f=&v_src[(s*dof+i)*6+0];
          T* n=&v_src[(s*dof+i)*6+3];

          T r_dot_n=(n[0]*dR[0]+n[1]*dR[1]+n[2]*dR[2]);
          T r_dot_f=(f[0]*dR[0]+f[1]*dR[1]+f[2]*dR[2]);
          T n_dot_f=(f[0]* n[0]+f[1]* n[1]+f[2]* n[2]);

          p[0] += dR[0]*(n_dot_f - 3*r_dot_n*r_dot_f*invR2)*invR3;
          p[1] += dR[1]*(n_dot_f - 3*r_dot_n*r_dot_f*invR2)*invR3;
          p[2] += dR[2]*(n_dot_f - 3*r_dot_n*r_dot_f*invR2)*invR3;
        }
      }
      k_out[(t*dof+i)*3+0] += p[0]*OOEPMU;
      k_out[(t*dof+i)*3+1] += p[1]*OOEPMU;
      k_out[(t*dof+i)*3+2] += p[2]*OOEPMU;
    }
  }
}

template <class T>
void stokes_press(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(17*dof));
#endif

  const T OOFP = 1.0/(4.0*const_pi<T>());
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p=0;
      for(int s=0;s<src_cnt;s++){
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R!=0){
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;
          T v_src[3]={v_src_[(s*dof+i)*3  ],
                      v_src_[(s*dof+i)*3+1],
                      v_src_[(s*dof+i)*3+2]};
          T inner_prod=(v_src[0]*dR[0] +
                        v_src[1]*dR[1] +
                        v_src[2]*dR[2])* invR3;
          p += inner_prod;
        }
      }
      k_out[t*dof+i] += p*OOFP;
    }
  }
}

template <class T>
void stokes_stress(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(45*dof));
#endif

  const T TOFP = -3.0/(4.0*const_pi<T>());
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[9]={0,0,0,
              0,0,0,
              0,0,0};
      for(int s=0;s<src_cnt;s++){
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R!=0){
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;
          T invR5=invR3*invR2;
          T v_src[3]={v_src_[(s*dof+i)*3  ],
                      v_src_[(s*dof+i)*3+1],
                      v_src_[(s*dof+i)*3+2]};
          T inner_prod=(v_src[0]*dR[0] +
                        v_src[1]*dR[1] +
                        v_src[2]*dR[2])* invR5;
          p[0] += inner_prod*dR[0]*dR[0]; p[1] += inner_prod*dR[1]*dR[0]; p[2] += inner_prod*dR[2]*dR[0];
          p[3] += inner_prod*dR[0]*dR[1]; p[4] += inner_prod*dR[1]*dR[1]; p[5] += inner_prod*dR[2]*dR[1];
          p[6] += inner_prod*dR[0]*dR[2]; p[7] += inner_prod*dR[1]*dR[2]; p[8] += inner_prod*dR[2]*dR[2];
        }
      }
      k_out[(t*dof+i)*9+0] += p[0]*TOFP;
      k_out[(t*dof+i)*9+1] += p[1]*TOFP;
      k_out[(t*dof+i)*9+2] += p[2]*TOFP;
      k_out[(t*dof+i)*9+3] += p[3]*TOFP;
      k_out[(t*dof+i)*9+4] += p[4]*TOFP;
      k_out[(t*dof+i)*9+5] += p[5]*TOFP;
      k_out[(t*dof+i)*9+6] += p[6]*TOFP;
      k_out[(t*dof+i)*9+7] += p[7]*TOFP;
      k_out[(t*dof+i)*9+8] += p[8]*TOFP;
    }
  }
}

template <class T>
void stokes_grad(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(89*dof));
#endif

  const T mu=1.0;
  const T OOEPMU = 1.0/(8.0*const_pi<T>()*mu);
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[9]={0,0,0,
              0,0,0,
              0,0,0};
      for(int s=0;s<src_cnt;s++){
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R!=0){
          T invR2=1.0/R;
          T invR=sqrt(invR2);
          T invR3=invR2*invR;
          T v_src[3]={v_src_[(s*dof+i)*3  ],
                      v_src_[(s*dof+i)*3+1],
                      v_src_[(s*dof+i)*3+2]};
          T inner_prod=(v_src[0]*dR[0] +
                        v_src[1]*dR[1] +
                        v_src[2]*dR[2]);

          p[0] += (                              inner_prod*(1-3*dR[0]*dR[0]*invR2))*invR3; //6
          p[1] += (dR[1]*v_src[0]-v_src[1]*dR[0]+inner_prod*( -3*dR[1]*dR[0]*invR2))*invR3; //9
          p[2] += (dR[2]*v_src[0]-v_src[2]*dR[0]+inner_prod*( -3*dR[2]*dR[0]*invR2))*invR3;

          p[3] += (dR[0]*v_src[1]-v_src[0]*dR[1]+inner_prod*( -3*dR[0]*dR[1]*invR2))*invR3;
          p[4] += (                              inner_prod*(1-3*dR[1]*dR[1]*invR2))*invR3;
          p[5] += (dR[2]*v_src[1]-v_src[2]*dR[1]+inner_prod*( -3*dR[2]*dR[1]*invR2))*invR3;

          p[6] += (dR[0]*v_src[2]-v_src[0]*dR[2]+inner_prod*( -3*dR[0]*dR[2]*invR2))*invR3;
          p[7] += (dR[1]*v_src[2]-v_src[1]*dR[2]+inner_prod*( -3*dR[1]*dR[2]*invR2))*invR3;
          p[8] += (                              inner_prod*(1-3*dR[2]*dR[2]*invR2))*invR3;

        }
      }
      k_out[(t*dof+i)*9+0] += p[0]*OOEPMU;
      k_out[(t*dof+i)*9+1] += p[1]*OOEPMU;
      k_out[(t*dof+i)*9+2] += p[2]*OOEPMU;
      k_out[(t*dof+i)*9+3] += p[3]*OOEPMU;
      k_out[(t*dof+i)*9+4] += p[4]*OOEPMU;
      k_out[(t*dof+i)*9+5] += p[5]*OOEPMU;
      k_out[(t*dof+i)*9+6] += p[6]*OOEPMU;
      k_out[(t*dof+i)*9+7] += p[7]*OOEPMU;
      k_out[(t*dof+i)*9+8] += p[8]*OOEPMU;
    }
  }
}

#ifndef __MIC__
#ifdef USE_SSE
namespace
{
#define IDEAL_ALIGNMENT 16
#define SIMD_LEN (int)(IDEAL_ALIGNMENT / sizeof(double))
#define DECL_SIMD_ALIGNED  __declspec(align(IDEAL_ALIGNMENT))

  void stokesDirectVecSSE(
      const int ns,
      const int nt,
      const double *sx,
      const double *sy,
      const double *sz,
      const double *tx,
      const double *ty,
      const double *tz,
      const double *srcDen,
      double *trgVal,
      const double cof )
  {
    if ( size_t(sx)%IDEAL_ALIGNMENT || size_t(sy)%IDEAL_ALIGNMENT || size_t(sz)%IDEAL_ALIGNMENT )
      abort();
    double mu = cof;

    double OOEP = 1.0/(8.0*const_pi<double>());
    __m128d tempx;
    __m128d tempy;
    __m128d tempz;
    double oomeu = 1/mu;

    double aux_arr[3*SIMD_LEN+1];
    double *tempvalx;
    double *tempvaly;
    double *tempvalz;
    if (size_t(aux_arr)%IDEAL_ALIGNMENT)  // if aux_arr is misaligned
    {
      tempvalx = aux_arr + 1;
      if (size_t(tempvalx)%IDEAL_ALIGNMENT)
        abort();
    }
    else
      tempvalx = aux_arr;
    tempvaly=tempvalx+SIMD_LEN;
    tempvalz=tempvaly+SIMD_LEN;


    /*! One over eight pi */
    __m128d ooep = _mm_set1_pd (OOEP);
    __m128d half = _mm_set1_pd (0.5);
    __m128d opf = _mm_set1_pd (1.5);
    __m128d zero = _mm_setzero_pd ();
    __m128d oomu = _mm_set1_pd (1/mu);

    // loop over sources
    int i = 0;
    for (; i < nt; i++) {
      tempx = _mm_setzero_pd();
      tempy = _mm_setzero_pd();
      tempz = _mm_setzero_pd();

      __m128d txi = _mm_load1_pd (&tx[i]);
      __m128d tyi = _mm_load1_pd (&ty[i]);
      __m128d tzi = _mm_load1_pd (&tz[i]);
      int j = 0;
      // Load and calculate in groups of SIMD_LEN
      for (; j + SIMD_LEN <= ns; j+=SIMD_LEN) {
        __m128d sxj = _mm_load_pd (&sx[j]);
        __m128d syj = _mm_load_pd (&sy[j]);
        __m128d szj = _mm_load_pd (&sz[j]);
        __m128d sdenx = _mm_set_pd (srcDen[(j+1)*3],   srcDen[j*3]);
        __m128d sdeny = _mm_set_pd (srcDen[(j+1)*3+1], srcDen[j*3+1]);
        __m128d sdenz = _mm_set_pd (srcDen[(j+1)*3+2], srcDen[j*3+2]);

        __m128d dX, dY, dZ;
        __m128d dR2;
        __m128d S;

        dX = _mm_sub_pd(txi , sxj);
        dY = _mm_sub_pd(tyi , syj);
        dZ = _mm_sub_pd(tzi , szj);

        sxj = _mm_mul_pd(dX, dX);
        syj = _mm_mul_pd(dY, dY);
        szj = _mm_mul_pd(dZ, dZ);

        dR2 = _mm_add_pd(sxj, syj);
        dR2 = _mm_add_pd(szj, dR2);
        __m128d temp = _mm_cmpeq_pd (dR2, zero);

        __m128d xhalf = _mm_mul_pd (half, dR2);
        __m128 dR2_s  =  _mm_cvtpd_ps(dR2);
        __m128 S_s    = _mm_rsqrt_ps(dR2_s);
        __m128d S_d   = _mm_cvtps_pd(S_s);
        // To handle the condition when src and trg coincide
        S_d = _mm_andnot_pd (temp, S_d);

        S = _mm_mul_pd (S_d, S_d);
        S = _mm_mul_pd (S, xhalf);
        S = _mm_sub_pd (opf, S);
        S = _mm_mul_pd (S, S_d);

        __m128d dotx = _mm_mul_pd (dX, sdenx);
        __m128d doty = _mm_mul_pd (dY, sdeny);
        __m128d dotz = _mm_mul_pd (dZ, sdenz);

        __m128d dot_sum = _mm_add_pd (dotx, doty);
        dot_sum = _mm_add_pd (dot_sum, dotz);

        dot_sum = _mm_mul_pd (dot_sum, S);
        dot_sum = _mm_mul_pd (dot_sum, S);
        dotx = _mm_mul_pd (dot_sum, dX);
        doty = _mm_mul_pd (dot_sum, dY);
        dotz = _mm_mul_pd (dot_sum, dZ);

        sdenx = _mm_add_pd (sdenx, dotx);
        sdeny = _mm_add_pd (sdeny, doty);
        sdenz = _mm_add_pd (sdenz, dotz);

        sdenx = _mm_mul_pd (sdenx, S);
        sdeny = _mm_mul_pd (sdeny, S);
        sdenz = _mm_mul_pd (sdenz, S);

        tempx = _mm_add_pd (sdenx, tempx);
        tempy = _mm_add_pd (sdeny, tempy);
        tempz = _mm_add_pd (sdenz, tempz);

      }
      tempx = _mm_mul_pd (tempx, ooep);
      tempy = _mm_mul_pd (tempy, ooep);
      tempz = _mm_mul_pd (tempz, ooep);

      tempx = _mm_mul_pd (tempx, oomu);
      tempy = _mm_mul_pd (tempy, oomu);
      tempz = _mm_mul_pd (tempz, oomu);

      _mm_store_pd(tempvalx, tempx);
      _mm_store_pd(tempvaly, tempy);
      _mm_store_pd(tempvalz, tempz);
      for (int k = 0; k < SIMD_LEN; k++) {
        trgVal[i*3]   += tempvalx[k];
        trgVal[i*3+1] += tempvaly[k];
        trgVal[i*3+2] += tempvalz[k];
      }

      for (; j < ns; j++) {
        double x = tx[i] - sx[j];
        double y = ty[i] - sy[j];
        double z = tz[i] - sz[j];
        double r2 = x*x + y*y + z*z;
        double r = sqrt(r2);
        double invdr;
        if (r == 0)
          invdr = 0;
        else
          invdr = 1/r;
        double dot = (x*srcDen[j*3] + y*srcDen[j*3+1] + z*srcDen[j*3+2]) * invdr * invdr;
        double denx = srcDen[j*3] + dot*x;
        double deny = srcDen[j*3+1] + dot*y;
        double denz = srcDen[j*3+2] + dot*z;

        trgVal[i*3] += denx*invdr*OOEP*oomeu;
        trgVal[i*3+1] += deny*invdr*OOEP*oomeu;
        trgVal[i*3+2] += denz*invdr*OOEP*oomeu;
      }
    }

    return;
  }

  void stokesPressureSSE(
      const int ns,
      const int nt,
      const double *sx,
      const double *sy,
      const double *sz,
      const double *tx,
      const double *ty,
      const double *tz,
      const double *srcDen,
      double *trgVal)
  {
    if ( size_t(sx)%IDEAL_ALIGNMENT || size_t(sy)%IDEAL_ALIGNMENT || size_t(sz)%IDEAL_ALIGNMENT )
      abort();

    double OOFP = 1.0/(4.0*const_pi<double>());
    __m128d temp_press;

    double aux_arr[SIMD_LEN+1];
    double *tempval_press;
    if (size_t(aux_arr)%IDEAL_ALIGNMENT)  // if aux_arr is misaligned
    {
      tempval_press = aux_arr + 1;
      if (size_t(tempval_press)%IDEAL_ALIGNMENT)
        abort();
    }
    else
      tempval_press = aux_arr;


    /*! One over eight pi */
    __m128d oofp = _mm_set1_pd (OOFP);
    __m128d half = _mm_set1_pd (0.5);
    __m128d opf = _mm_set1_pd (1.5);
    __m128d zero = _mm_setzero_pd ();

    // loop over sources
    int i = 0;
    for (; i < nt; i++) {
      temp_press = _mm_setzero_pd();

      __m128d txi = _mm_load1_pd (&tx[i]);
      __m128d tyi = _mm_load1_pd (&ty[i]);
      __m128d tzi = _mm_load1_pd (&tz[i]);
      int j = 0;
      // Load and calculate in groups of SIMD_LEN
      for (; j + SIMD_LEN <= ns; j+=SIMD_LEN) {
        __m128d sxj = _mm_load_pd (&sx[j]);
        __m128d syj = _mm_load_pd (&sy[j]);
        __m128d szj = _mm_load_pd (&sz[j]);
        __m128d sdenx = _mm_set_pd (srcDen[(j+1)*3],   srcDen[j*3]);
        __m128d sdeny = _mm_set_pd (srcDen[(j+1)*3+1], srcDen[j*3+1]);
        __m128d sdenz = _mm_set_pd (srcDen[(j+1)*3+2], srcDen[j*3+2]);

        __m128d dX, dY, dZ;
        __m128d dR2;
        __m128d S;

        dX = _mm_sub_pd(txi , sxj);
        dY = _mm_sub_pd(tyi , syj);
        dZ = _mm_sub_pd(tzi , szj);

        sxj = _mm_mul_pd(dX, dX);
        syj = _mm_mul_pd(dY, dY);
        szj = _mm_mul_pd(dZ, dZ);

        dR2 = _mm_add_pd(sxj, syj);
        dR2 = _mm_add_pd(szj, dR2);
        __m128d temp = _mm_cmpeq_pd (dR2, zero);

        __m128d xhalf = _mm_mul_pd (half, dR2);
        __m128 dR2_s  =  _mm_cvtpd_ps(dR2);
        __m128 S_s    = _mm_rsqrt_ps(dR2_s);
        __m128d S_d   = _mm_cvtps_pd(S_s);
        // To handle the condition when src and trg coincide
        S_d = _mm_andnot_pd (temp, S_d);

        S = _mm_mul_pd (S_d, S_d);
        S = _mm_mul_pd (S, xhalf);
        S = _mm_sub_pd (opf, S);
        S = _mm_mul_pd (S, S_d);

        __m128d dotx = _mm_mul_pd (dX, sdenx);
        __m128d doty = _mm_mul_pd (dY, sdeny);
        __m128d dotz = _mm_mul_pd (dZ, sdenz);

        __m128d dot_sum = _mm_add_pd (dotx, doty);
        dot_sum = _mm_add_pd (dot_sum, dotz);

        dot_sum = _mm_mul_pd (dot_sum, S);
        dot_sum = _mm_mul_pd (dot_sum, S);
        dot_sum = _mm_mul_pd (dot_sum, S);

        temp_press = _mm_add_pd (dot_sum, temp_press);

      }
      temp_press = _mm_mul_pd (temp_press, oofp);

      _mm_store_pd(tempval_press, temp_press);
      for (int k = 0; k < SIMD_LEN; k++) {
        trgVal[i]   += tempval_press[k];
      }

      for (; j < ns; j++) {
        double x = tx[i] - sx[j];
        double y = ty[i] - sy[j];
        double z = tz[i] - sz[j];
        double r2 = x*x + y*y + z*z;
        double r = sqrt(r2);
        double invdr;
        if (r == 0)
          invdr = 0;
        else
          invdr = 1/r;
        double dot = (x*srcDen[j*3] + y*srcDen[j*3+1] + z*srcDen[j*3+2]) * invdr * invdr * invdr;

        trgVal[i] += dot*OOFP;
      }
    }

    return;
  }

  void stokesStressSSE(
      const int ns,
      const int nt,
      const double *sx,
      const double *sy,
      const double *sz,
      const double *tx,
      const double *ty,
      const double *tz,
      const double *srcDen,
      double *trgVal)
  {
    if ( size_t(sx)%IDEAL_ALIGNMENT || size_t(sy)%IDEAL_ALIGNMENT || size_t(sz)%IDEAL_ALIGNMENT )
      abort();

    double TOFP = -3.0/(4.0*const_pi<double>());
    __m128d tempxx; __m128d tempxy; __m128d tempxz;
    __m128d tempyx; __m128d tempyy; __m128d tempyz;
    __m128d tempzx; __m128d tempzy; __m128d tempzz;

    double aux_arr[9*SIMD_LEN+1];
    double *tempvalxx, *tempvalxy, *tempvalxz;
    double *tempvalyx, *tempvalyy, *tempvalyz;
    double *tempvalzx, *tempvalzy, *tempvalzz;
    if (size_t(aux_arr)%IDEAL_ALIGNMENT)  // if aux_arr is misaligned
    {
      tempvalxx = aux_arr + 1;
      if (size_t(tempvalxx)%IDEAL_ALIGNMENT)
        abort();
    }
    else
      tempvalxx = aux_arr;
    tempvalxy=tempvalxx+SIMD_LEN;
    tempvalxz=tempvalxy+SIMD_LEN;

    tempvalyx=tempvalxz+SIMD_LEN;
    tempvalyy=tempvalyx+SIMD_LEN;
    tempvalyz=tempvalyy+SIMD_LEN;

    tempvalzx=tempvalyz+SIMD_LEN;
    tempvalzy=tempvalzx+SIMD_LEN;
    tempvalzz=tempvalzy+SIMD_LEN;

    /*! One over eight pi */
    __m128d tofp = _mm_set1_pd (TOFP);
    __m128d half = _mm_set1_pd (0.5);
    __m128d opf = _mm_set1_pd (1.5);
    __m128d zero = _mm_setzero_pd ();

    // loop over sources
    int i = 0;
    for (; i < nt; i++) {
      tempxx = _mm_setzero_pd(); tempxy = _mm_setzero_pd(); tempxz = _mm_setzero_pd();
      tempyx = _mm_setzero_pd(); tempyy = _mm_setzero_pd(); tempyz = _mm_setzero_pd();
      tempzx = _mm_setzero_pd(); tempzy = _mm_setzero_pd(); tempzz = _mm_setzero_pd();

      __m128d txi = _mm_load1_pd (&tx[i]);
      __m128d tyi = _mm_load1_pd (&ty[i]);
      __m128d tzi = _mm_load1_pd (&tz[i]);
      int j = 0;
      // Load and calculate in groups of SIMD_LEN
      for (; j + SIMD_LEN <= ns; j+=SIMD_LEN) {
        __m128d sxj = _mm_load_pd (&sx[j]);
        __m128d syj = _mm_load_pd (&sy[j]);
        __m128d szj = _mm_load_pd (&sz[j]);
        __m128d sdenx = _mm_set_pd (srcDen[(j+1)*3],   srcDen[j*3]);
        __m128d sdeny = _mm_set_pd (srcDen[(j+1)*3+1], srcDen[j*3+1]);
        __m128d sdenz = _mm_set_pd (srcDen[(j+1)*3+2], srcDen[j*3+2]);

        __m128d dX, dY, dZ;
        __m128d dR2;
        __m128d S;
        __m128d S2;

        dX = _mm_sub_pd(txi , sxj);
        dY = _mm_sub_pd(tyi , syj);
        dZ = _mm_sub_pd(tzi , szj);

        sxj = _mm_mul_pd(dX, dX);
        syj = _mm_mul_pd(dY, dY);
        szj = _mm_mul_pd(dZ, dZ);

        dR2 = _mm_add_pd(sxj, syj);
        dR2 = _mm_add_pd(szj, dR2);
        __m128d temp = _mm_cmpeq_pd (dR2, zero);

        __m128d xhalf = _mm_mul_pd (half, dR2);
        __m128 dR2_s  =  _mm_cvtpd_ps(dR2);
        __m128 S_s    = _mm_rsqrt_ps(dR2_s);
        __m128d S_d   = _mm_cvtps_pd(S_s);
        // To handle the condition when src and trg coincide
        S_d = _mm_andnot_pd (temp, S_d);

        S = _mm_mul_pd (S_d, S_d);
        S = _mm_mul_pd (S, xhalf);
        S = _mm_sub_pd (opf, S);
        S = _mm_mul_pd (S, S_d);
        S2 = _mm_mul_pd (S, S);

        __m128d dotx = _mm_mul_pd (dX, sdenx);
        __m128d doty = _mm_mul_pd (dY, sdeny);
        __m128d dotz = _mm_mul_pd (dZ, sdenz);

        __m128d dot_sum = _mm_add_pd (dotx, doty);
        dot_sum = _mm_add_pd (dot_sum, dotz);

        dot_sum = _mm_mul_pd (dot_sum, S);
        dot_sum = _mm_mul_pd (dot_sum, S2);
        dot_sum = _mm_mul_pd (dot_sum, S2);

        dotx = _mm_mul_pd (dot_sum, dX);
        doty = _mm_mul_pd (dot_sum, dY);
        dotz = _mm_mul_pd (dot_sum, dZ);

        tempxx = _mm_add_pd (_mm_mul_pd(dotx,dX), tempxx);
        tempxy = _mm_add_pd (_mm_mul_pd(dotx,dY), tempxy);
        tempxz = _mm_add_pd (_mm_mul_pd(dotx,dZ), tempxz);

        tempyx = _mm_add_pd (_mm_mul_pd(doty,dX), tempyx);
        tempyy = _mm_add_pd (_mm_mul_pd(doty,dY), tempyy);
        tempyz = _mm_add_pd (_mm_mul_pd(doty,dZ), tempyz);

        tempzx = _mm_add_pd (_mm_mul_pd(dotz,dX), tempzx);
        tempzy = _mm_add_pd (_mm_mul_pd(dotz,dY), tempzy);
        tempzz = _mm_add_pd (_mm_mul_pd(dotz,dZ), tempzz);

      }
      tempxx = _mm_mul_pd (tempxx, tofp);
      tempxy = _mm_mul_pd (tempxy, tofp);
      tempxz = _mm_mul_pd (tempxz, tofp);

      tempyx = _mm_mul_pd (tempyx, tofp);
      tempyy = _mm_mul_pd (tempyy, tofp);
      tempyz = _mm_mul_pd (tempyz, tofp);

      tempzx = _mm_mul_pd (tempzx, tofp);
      tempzy = _mm_mul_pd (tempzy, tofp);
      tempzz = _mm_mul_pd (tempzz, tofp);

      _mm_store_pd(tempvalxx, tempxx); _mm_store_pd(tempvalxy, tempxy); _mm_store_pd(tempvalxz, tempxz);
      _mm_store_pd(tempvalyx, tempyx); _mm_store_pd(tempvalyy, tempyy); _mm_store_pd(tempvalyz, tempyz);
      _mm_store_pd(tempvalzx, tempzx); _mm_store_pd(tempvalzy, tempzy); _mm_store_pd(tempvalzz, tempzz);

      for (int k = 0; k < SIMD_LEN; k++) {
        trgVal[i*9  ] += tempvalxx[k];
        trgVal[i*9+1] += tempvalxy[k];
        trgVal[i*9+2] += tempvalxz[k];
        trgVal[i*9+3] += tempvalyx[k];
        trgVal[i*9+4] += tempvalyy[k];
        trgVal[i*9+5] += tempvalyz[k];
        trgVal[i*9+6] += tempvalzx[k];
        trgVal[i*9+7] += tempvalzy[k];
        trgVal[i*9+8] += tempvalzz[k];
      }

      for (; j < ns; j++) {
        double x = tx[i] - sx[j];
        double y = ty[i] - sy[j];
        double z = tz[i] - sz[j];
        double r2 = x*x + y*y + z*z;
        double r = sqrt(r2);
        double invdr;
        if (r == 0)
          invdr = 0;
        else
          invdr = 1/r;
        double invdr2=invdr*invdr;
        double dot = (x*srcDen[j*3] + y*srcDen[j*3+1] + z*srcDen[j*3+2]) * invdr2 * invdr2 * invdr;
        double denx = dot*x;
        double deny = dot*y;
        double denz = dot*z;

        trgVal[i*9  ] += denx*x*TOFP;
        trgVal[i*9+1] += denx*y*TOFP;
        trgVal[i*9+2] += denx*z*TOFP;
        trgVal[i*9+3] += deny*x*TOFP;
        trgVal[i*9+4] += deny*y*TOFP;
        trgVal[i*9+5] += deny*z*TOFP;
        trgVal[i*9+6] += denz*x*TOFP;
        trgVal[i*9+7] += denz*y*TOFP;
        trgVal[i*9+8] += denz*z*TOFP;
      }
    }

    return;
  }

  void stokesGradSSE(
      const int ns,
      const int nt,
      const double *sx,
      const double *sy,
      const double *sz,
      const double *tx,
      const double *ty,
      const double *tz,
      const double *srcDen,
      double *trgVal,
      const double cof )
  {
    if ( size_t(sx)%IDEAL_ALIGNMENT || size_t(sy)%IDEAL_ALIGNMENT || size_t(sz)%IDEAL_ALIGNMENT )
      abort();
    double mu = cof;

    double OOEP = 1.0/(8.0*const_pi<double>());
    __m128d tempxx; __m128d tempxy; __m128d tempxz;
    __m128d tempyx; __m128d tempyy; __m128d tempyz;
    __m128d tempzx; __m128d tempzy; __m128d tempzz;
    double oomeu = 1/mu;

    double aux_arr[9*SIMD_LEN+1];
    double *tempvalxx, *tempvalxy, *tempvalxz;
    double *tempvalyx, *tempvalyy, *tempvalyz;
    double *tempvalzx, *tempvalzy, *tempvalzz;
    if (size_t(aux_arr)%IDEAL_ALIGNMENT)  // if aux_arr is misaligned
    {
      tempvalxx = aux_arr + 1;
      if (size_t(tempvalxx)%IDEAL_ALIGNMENT)
        abort();
    }
    else
      tempvalxx = aux_arr;
    tempvalxy=tempvalxx+SIMD_LEN;
    tempvalxz=tempvalxy+SIMD_LEN;

    tempvalyx=tempvalxz+SIMD_LEN;
    tempvalyy=tempvalyx+SIMD_LEN;
    tempvalyz=tempvalyy+SIMD_LEN;

    tempvalzx=tempvalyz+SIMD_LEN;
    tempvalzy=tempvalzx+SIMD_LEN;
    tempvalzz=tempvalzy+SIMD_LEN;

    /*! One over eight pi */
    __m128d ooep = _mm_set1_pd (OOEP);
    __m128d half = _mm_set1_pd (0.5);
    __m128d opf = _mm_set1_pd (1.5);
    __m128d three = _mm_set1_pd (3.0);
    __m128d zero = _mm_setzero_pd ();
    __m128d oomu = _mm_set1_pd (1/mu);
    __m128d ooepmu = _mm_mul_pd(ooep,oomu);

    // loop over sources
    int i = 0;
    for (; i < nt; i++) {
      tempxx = _mm_setzero_pd(); tempxy = _mm_setzero_pd(); tempxz = _mm_setzero_pd();
      tempyx = _mm_setzero_pd(); tempyy = _mm_setzero_pd(); tempyz = _mm_setzero_pd();
      tempzx = _mm_setzero_pd(); tempzy = _mm_setzero_pd(); tempzz = _mm_setzero_pd();

      __m128d txi = _mm_load1_pd (&tx[i]);
      __m128d tyi = _mm_load1_pd (&ty[i]);
      __m128d tzi = _mm_load1_pd (&tz[i]);
      int j = 0;
      // Load and calculate in groups of SIMD_LEN
      for (; j + SIMD_LEN <= ns; j+=SIMD_LEN) {
        __m128d sxj = _mm_load_pd (&sx[j]);
        __m128d syj = _mm_load_pd (&sy[j]);
        __m128d szj = _mm_load_pd (&sz[j]);
        __m128d sdenx = _mm_set_pd (srcDen[(j+1)*3],   srcDen[j*3]);
        __m128d sdeny = _mm_set_pd (srcDen[(j+1)*3+1], srcDen[j*3+1]);
        __m128d sdenz = _mm_set_pd (srcDen[(j+1)*3+2], srcDen[j*3+2]);

        __m128d dX, dY, dZ;
        __m128d dR2;
        __m128d S;
        __m128d S2;
        __m128d S3;

        dX = _mm_sub_pd(txi , sxj);
        dY = _mm_sub_pd(tyi , syj);
        dZ = _mm_sub_pd(tzi , szj);

        sxj = _mm_mul_pd(dX, dX);
        syj = _mm_mul_pd(dY, dY);
        szj = _mm_mul_pd(dZ, dZ);

        dR2 = _mm_add_pd(sxj, syj);
        dR2 = _mm_add_pd(szj, dR2);
        __m128d temp = _mm_cmpeq_pd (dR2, zero);

        __m128d xhalf = _mm_mul_pd (half, dR2);
        __m128 dR2_s  =  _mm_cvtpd_ps(dR2);
        __m128 S_s    = _mm_rsqrt_ps(dR2_s);
        __m128d S_d   = _mm_cvtps_pd(S_s);
        // To handle the condition when src and trg coincide
        S_d = _mm_andnot_pd (temp, S_d);

        S = _mm_mul_pd (S_d, S_d);
        S = _mm_mul_pd (S, xhalf);
        S = _mm_sub_pd (opf, S);
        S = _mm_mul_pd (S, S_d);
        S2 = _mm_mul_pd (S, S);
        S3 = _mm_mul_pd (S2, S);

        __m128d dotx = _mm_mul_pd (dX, sdenx);
        __m128d doty = _mm_mul_pd (dY, sdeny);
        __m128d dotz = _mm_mul_pd (dZ, sdenz);

        __m128d dot_sum = _mm_add_pd (dotx, doty);
        dot_sum = _mm_add_pd (dot_sum, dotz);

        dot_sum = _mm_mul_pd (dot_sum, S2);

        tempxx = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dX, sdenx), _mm_mul_pd(sdenx, dX)), _mm_mul_pd(dot_sum, _mm_sub_pd(dR2 , _mm_mul_pd(three, _mm_mul_pd(dX, dX)))))),tempxx);
        tempxy = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dY, sdenx), _mm_mul_pd(sdeny, dX)), _mm_mul_pd(dot_sum, _mm_sub_pd(zero, _mm_mul_pd(three, _mm_mul_pd(dY, dX)))))),tempxy);
        tempxz = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dZ, sdenx), _mm_mul_pd(sdenz, dX)), _mm_mul_pd(dot_sum, _mm_sub_pd(zero, _mm_mul_pd(three, _mm_mul_pd(dZ, dX)))))),tempxz);

        tempyx = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dX, sdeny), _mm_mul_pd(sdenx, dY)), _mm_mul_pd(dot_sum, _mm_sub_pd(zero, _mm_mul_pd(three, _mm_mul_pd(dX, dY)))))),tempyx);
        tempyy = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dY, sdeny), _mm_mul_pd(sdeny, dY)), _mm_mul_pd(dot_sum, _mm_sub_pd(dR2 , _mm_mul_pd(three, _mm_mul_pd(dY, dY)))))),tempyy);
        tempyz = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dZ, sdeny), _mm_mul_pd(sdenz, dY)), _mm_mul_pd(dot_sum, _mm_sub_pd(zero, _mm_mul_pd(three, _mm_mul_pd(dZ, dY)))))),tempyz);

        tempzx = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dX, sdenz), _mm_mul_pd(sdenx, dZ)), _mm_mul_pd(dot_sum, _mm_sub_pd(zero, _mm_mul_pd(three, _mm_mul_pd(dX, dZ)))))),tempzx);
        tempzy = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dY, sdenz), _mm_mul_pd(sdeny, dZ)), _mm_mul_pd(dot_sum, _mm_sub_pd(zero, _mm_mul_pd(three, _mm_mul_pd(dY, dZ)))))),tempzy);
        tempzz = _mm_add_pd(_mm_mul_pd(S3,_mm_add_pd(_mm_sub_pd(_mm_mul_pd(dZ, sdenz), _mm_mul_pd(sdenz, dZ)), _mm_mul_pd(dot_sum, _mm_sub_pd(dR2 , _mm_mul_pd(three, _mm_mul_pd(dZ, dZ)))))),tempzz);

      }
      tempxx = _mm_mul_pd (tempxx, ooepmu);
      tempxy = _mm_mul_pd (tempxy, ooepmu);
      tempxz = _mm_mul_pd (tempxz, ooepmu);

      tempyx = _mm_mul_pd (tempyx, ooepmu);
      tempyy = _mm_mul_pd (tempyy, ooepmu);
      tempyz = _mm_mul_pd (tempyz, ooepmu);

      tempzx = _mm_mul_pd (tempzx, ooepmu);
      tempzy = _mm_mul_pd (tempzy, ooepmu);
      tempzz = _mm_mul_pd (tempzz, ooepmu);

      _mm_store_pd(tempvalxx, tempxx); _mm_store_pd(tempvalxy, tempxy); _mm_store_pd(tempvalxz, tempxz);
      _mm_store_pd(tempvalyx, tempyx); _mm_store_pd(tempvalyy, tempyy); _mm_store_pd(tempvalyz, tempyz);
      _mm_store_pd(tempvalzx, tempzx); _mm_store_pd(tempvalzy, tempzy); _mm_store_pd(tempvalzz, tempzz);

      for (int k = 0; k < SIMD_LEN; k++) {
        trgVal[i*9  ] += tempvalxx[k];
        trgVal[i*9+1] += tempvalxy[k];
        trgVal[i*9+2] += tempvalxz[k];
        trgVal[i*9+3] += tempvalyx[k];
        trgVal[i*9+4] += tempvalyy[k];
        trgVal[i*9+5] += tempvalyz[k];
        trgVal[i*9+6] += tempvalzx[k];
        trgVal[i*9+7] += tempvalzy[k];
        trgVal[i*9+8] += tempvalzz[k];
      }

      for (; j < ns; j++) {
        double x = tx[i] - sx[j];
        double y = ty[i] - sy[j];
        double z = tz[i] - sz[j];
        double r2 = x*x + y*y + z*z;
        double r = sqrt(r2);
        double invdr;
        if (r == 0)
          invdr = 0;
        else
          invdr = 1/r;
        double invdr2=invdr*invdr;
        double invdr3=invdr2*invdr;
        double dot = (x*srcDen[j*3] + y*srcDen[j*3+1] + z*srcDen[j*3+2]);

        trgVal[i*9  ] += OOEP*oomeu*invdr3*( x*srcDen[j*3  ] - srcDen[j*3  ]*x + dot*(1-3*x*x*invdr2) );
        trgVal[i*9+1] += OOEP*oomeu*invdr3*( y*srcDen[j*3  ] - srcDen[j*3+1]*x + dot*(0-3*y*x*invdr2) );
        trgVal[i*9+2] += OOEP*oomeu*invdr3*( z*srcDen[j*3  ] - srcDen[j*3+2]*x + dot*(0-3*z*x*invdr2) );

        trgVal[i*9+3] += OOEP*oomeu*invdr3*( x*srcDen[j*3+1] - srcDen[j*3  ]*y + dot*(0-3*x*y*invdr2) );
        trgVal[i*9+4] += OOEP*oomeu*invdr3*( y*srcDen[j*3+1] - srcDen[j*3+1]*y + dot*(1-3*y*y*invdr2) );
        trgVal[i*9+5] += OOEP*oomeu*invdr3*( z*srcDen[j*3+1] - srcDen[j*3+2]*y + dot*(0-3*z*y*invdr2) );

        trgVal[i*9+6] += OOEP*oomeu*invdr3*( x*srcDen[j*3+2] - srcDen[j*3  ]*z + dot*(0-3*x*z*invdr2) );
        trgVal[i*9+7] += OOEP*oomeu*invdr3*( y*srcDen[j*3+2] - srcDen[j*3+1]*z + dot*(0-3*y*z*invdr2) );
        trgVal[i*9+8] += OOEP*oomeu*invdr3*( z*srcDen[j*3+2] - srcDen[j*3+2]*z + dot*(1-3*z*z*invdr2) );
      }
    }

    return;
  }
#undef SIMD_LEN

#define X(s,k) (s)[(k)*COORD_DIM]
#define Y(s,k) (s)[(k)*COORD_DIM+1]
#define Z(s,k) (s)[(k)*COORD_DIM+2]
  void stokesDirectSSEShuffle(const int ns, const int nt, double const src[], double const trg[], double const den[], double pot[], const double kernel_coef, mem::MemoryManager* mem_mgr=NULL)
  {

    std::vector<double> xs(ns+1);   std::vector<double> xt(nt);
    std::vector<double> ys(ns+1);   std::vector<double> yt(nt);
    std::vector<double> zs(ns+1);   std::vector<double> zt(nt);

    int x_shift = size_t(&xs[0]) % IDEAL_ALIGNMENT ? 1:0;
    int y_shift = size_t(&ys[0]) % IDEAL_ALIGNMENT ? 1:0;
    int z_shift = size_t(&zs[0]) % IDEAL_ALIGNMENT ? 1:0;

    //1. reshuffle memory
    for (int k =0;k<ns;k++){
      xs[k+x_shift]=X(src,k);
      ys[k+y_shift]=Y(src,k);
      zs[k+z_shift]=Z(src,k);
    }
    for (int k=0;k<nt;k++){
      xt[k]=X(trg,k);
      yt[k]=Y(trg,k);
      zt[k]=Z(trg,k);
    }

    //2. perform caclulation
    stokesDirectVecSSE(ns,nt,&xs[x_shift],&ys[y_shift],&zs[z_shift],&xt[0],&yt[0],&zt[0],den,pot,kernel_coef);
    return;
  }

  void stokesPressureSSEShuffle(const int ns, const int nt, double const src[], double const trg[], double const den[], double pot[], mem::MemoryManager* mem_mgr=NULL)
  {
    std::vector<double> xs(ns+1);   std::vector<double> xt(nt);
    std::vector<double> ys(ns+1);   std::vector<double> yt(nt);
    std::vector<double> zs(ns+1);   std::vector<double> zt(nt);

    int x_shift = size_t(&xs[0]) % IDEAL_ALIGNMENT ? 1:0;
    int y_shift = size_t(&ys[0]) % IDEAL_ALIGNMENT ? 1:0;
    int z_shift = size_t(&zs[0]) % IDEAL_ALIGNMENT ? 1:0;

    //1. reshuffle memory
    for (int k =0;k<ns;k++){
      xs[k+x_shift]=X(src,k);
      ys[k+y_shift]=Y(src,k);
      zs[k+z_shift]=Z(src,k);
    }
    for (int k=0;k<nt;k++){
      xt[k]=X(trg,k);
      yt[k]=Y(trg,k);
      zt[k]=Z(trg,k);
    }

    //2. perform caclulation
    stokesPressureSSE(ns,nt,&xs[x_shift],&ys[y_shift],&zs[z_shift],&xt[0],&yt[0],&zt[0],den,pot);
    return;
  }

  void stokesStressSSEShuffle(const int ns, const int nt, double const src[], double const trg[], double const den[], double pot[], mem::MemoryManager* mem_mgr=NULL)
  {
    std::vector<double> xs(ns+1);   std::vector<double> xt(nt);
    std::vector<double> ys(ns+1);   std::vector<double> yt(nt);
    std::vector<double> zs(ns+1);   std::vector<double> zt(nt);

    int x_shift = size_t(&xs[0]) % IDEAL_ALIGNMENT ? 1:0;
    int y_shift = size_t(&ys[0]) % IDEAL_ALIGNMENT ? 1:0;
    int z_shift = size_t(&zs[0]) % IDEAL_ALIGNMENT ? 1:0;

    //1. reshuffle memory
    for (int k =0;k<ns;k++){
      xs[k+x_shift]=X(src,k);
      ys[k+y_shift]=Y(src,k);
      zs[k+z_shift]=Z(src,k);
    }
    for (int k=0;k<nt;k++){
      xt[k]=X(trg,k);
      yt[k]=Y(trg,k);
      zt[k]=Z(trg,k);
    }

    //2. perform caclulation
    stokesStressSSE(ns,nt,&xs[x_shift],&ys[y_shift],&zs[z_shift],&xt[0],&yt[0],&zt[0],den,pot);
    return;
  }

  void stokesGradSSEShuffle(const int ns, const int nt, double const src[], double const trg[], double const den[], double pot[], const double kernel_coef, mem::MemoryManager* mem_mgr=NULL)
  {
    std::vector<double> xs(ns+1);   std::vector<double> xt(nt);
    std::vector<double> ys(ns+1);   std::vector<double> yt(nt);
    std::vector<double> zs(ns+1);   std::vector<double> zt(nt);

    int x_shift = size_t(&xs[0]) % IDEAL_ALIGNMENT ? 1:0;
    int y_shift = size_t(&ys[0]) % IDEAL_ALIGNMENT ? 1:0;
    int z_shift = size_t(&zs[0]) % IDEAL_ALIGNMENT ? 1:0;

    //1. reshuffle memory
    for (int k =0;k<ns;k++){
      xs[k+x_shift]=X(src,k);
      ys[k+y_shift]=Y(src,k);
      zs[k+z_shift]=Z(src,k);
    }
    for (int k=0;k<nt;k++){
      xt[k]=X(trg,k);
      yt[k]=Y(trg,k);
      zt[k]=Z(trg,k);
    }

    //2. perform caclulation
    stokesGradSSE(ns,nt,&xs[x_shift],&ys[y_shift],&zs[z_shift],&xt[0],&yt[0],&zt[0],den,pot,kernel_coef);
    return;
  }
#undef X
#undef Y
#undef Z

#undef IDEAL_ALIGNMENT
#undef DECL_SIMD_ALIGNED
}

template <>
void stokes_vel<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(28*dof));

  const double mu=1.0;
  stokesDirectSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mu, mem_mgr);
}

template <>
void stokes_press<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(17*dof));

  stokesPressureSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mem_mgr);
  return;
}

template <>
void stokes_stress<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(45*dof));

  stokesStressSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mem_mgr);
}

template <>
void stokes_grad<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(89*dof));

  const double mu=1.0;
  stokesGradSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mu, mem_mgr);
}
#endif
#endif


////////////////////////////////////////////////////////////////////////////////
////////                  BIOT-SAVART KERNEL                            ////////
////////////////////////////////////////////////////////////////////////////////

template <class T>
void biot_savart(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(26*dof));
#endif

  const T OOFP = -1.0/(4.0*const_pi<T>());
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[3]={0,0,0};
      for(int s=0;s<src_cnt;s++){
        T dR[3]={r_trg[3*t  ]-r_src[3*s  ],
                 r_trg[3*t+1]-r_src[3*s+1],
                 r_trg[3*t+2]-r_src[3*s+2]};
        T R2 = (dR[0]*dR[0]+dR[1]*dR[1]+dR[2]*dR[2]);
        if (R2!=0){
          T invR2=1.0/R2;
          T invR=sqrt(invR2);
          T invR3=invR*invR2;

          T v_src[3]={v_src_[(s*dof+i)*3  ],
                      v_src_[(s*dof+i)*3+1],
                      v_src_[(s*dof+i)*3+2]};

          p[0] -= (v_src[1]*dR[2]-v_src[2]*dR[1])*invR3;
          p[1] -= (v_src[2]*dR[0]-v_src[0]*dR[2])*invR3;
          p[2] -= (v_src[0]*dR[1]-v_src[1]*dR[0])*invR3;
        }
      }
      k_out[(t*dof+i)*3+0] += p[0]*OOFP;
      k_out[(t*dof+i)*3+1] += p[1]*OOFP;
      k_out[(t*dof+i)*3+2] += p[2]*OOFP;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
////////                   HELMHOLTZ KERNEL                             ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Helmholtz's equation. Kernel tensor
 * dimension = 2x2.
 */
template <class T>
void helmholtz_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(24*dof));
#endif

  const T mu = (20.0*const_pi<T>());
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[2]={0,0};
      for(int s=0;s<src_cnt;s++){
        T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
        T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
        T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
        T R = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
        if (R!=0){
          R = sqrt(R);
          T invR=1.0/R;
          T G[2]={cos(mu*R)*invR, sin(mu*R)*invR};
          p[0] += v_src[(s*dof+i)*2+0]*G[0] - v_src[(s*dof+i)*2+1]*G[1];
          p[1] += v_src[(s*dof+i)*2+0]*G[1] + v_src[(s*dof+i)*2+1]*G[0];
        }
      }
      k_out[(t*dof+i)*2+0] += p[0];
      k_out[(t*dof+i)*2+1] += p[1];
    }
  }
}

template <class T>
void helmholtz_grad(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
  //TODO Implement this.
}

}//end namespace
