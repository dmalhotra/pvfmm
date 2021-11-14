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
#include <sctl.hpp>

#include <mem_mgr.hpp>
#include <profile.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <precomp_mat.hpp>
#include <cheb_utils.hpp>

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
  surf_dim=PVFMM_COORD_DIM+ker_dim[0]; // including the normal-vector for double-layer kernels
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
  vol_poten=NULL;

  scale_invar=true;
  src_scal.Resize(ker_dim[0]); src_scal.SetZero();
  trg_scal.Resize(ker_dim[1]); trg_scal.SetZero();
  perm_vec.Resize(Perm_Count);
  for(size_t p_type=0;p_type<C_Perm;p_type++){
    perm_vec[p_type       ]=Permutation<T>(ker_dim[0]);
    perm_vec[p_type+C_Perm]=Permutation<T>(ker_dim[1]);
  }
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

  T scal=1.0;
  if(ker_dim[0]*ker_dim[1]>0){ // Determine scaling
    Matrix<T> M_scal(ker_dim[0],ker_dim[1]);
    size_t N=1024;
    T eps_=N*eps;

    T src_coord[3]={0,0,0};
    std::vector<T> trg_coord1(N*PVFMM_COORD_DIM);
    Matrix<T> M1(N,ker_dim[0]*ker_dim[1]);
    while(true){
      T abs_sum=0;
      for(size_t i=0;i<N/2;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=sctl::sqrt<T>(x*x+y*y+z*z);
        }while(r<0.25);
        trg_coord1[i*PVFMM_COORD_DIM+0]=x*scal;
        trg_coord1[i*PVFMM_COORD_DIM+1]=y*scal;
        trg_coord1[i*PVFMM_COORD_DIM+2]=z*scal;
      }
      for(size_t i=N/2;i<N;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=sctl::sqrt<T>(x*x+y*y+z*z);
        }while(r<0.25);
        trg_coord1[i*PVFMM_COORD_DIM+0]=x*1.0/scal;
        trg_coord1[i*PVFMM_COORD_DIM+1]=y*1.0/scal;
        trg_coord1[i*PVFMM_COORD_DIM+2]=z*1.0/scal;
      }
      for(size_t i=0;i<N;i++){
        BuildMatrix(&src_coord [          0], 1,
                    &trg_coord1[i*PVFMM_COORD_DIM], 1, &(M1[i][0]));
        for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
          abs_sum+=sctl::fabs<T>(M1[i][j]);
        }
      }
      if(abs_sum>sctl::sqrt<T>(eps) || scal<eps) break;
      scal=scal*0.5;
    }

    std::vector<T> trg_coord2(N*PVFMM_COORD_DIM);
    Matrix<T> M2(N,ker_dim[0]*ker_dim[1]);
    for(size_t i=0;i<N*PVFMM_COORD_DIM;i++){
      trg_coord2[i]=trg_coord1[i]*0.5;
    }
    for(size_t i=0;i<N;i++){
      BuildMatrix(&src_coord [          0], 1,
                  &trg_coord2[i*PVFMM_COORD_DIM], 1, &(M2[i][0]));
    }

    T max_val=0;
    for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
      T dot11=0, dot22=0;
      for(size_t j=0;j<N;j++){
        dot11+=M1[j][i]*M1[j][i];
        dot22+=M2[j][i]*M2[j][i];
      }
      max_val=std::max<T>(max_val,dot11);
      max_val=std::max<T>(max_val,dot22);
    }
    if(scale_invar)
    for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
      T dot11=0, dot12=0, dot22=0;
      for(size_t j=0;j<N;j++){
        dot11+=M1[j][i]*M1[j][i];
        dot12+=M1[j][i]*M2[j][i];
        dot22+=M2[j][i]*M2[j][i];
      }
      if(dot11>max_val*eps &&
         dot22>max_val*eps ){
        T s=dot12/dot11;
        M_scal[0][i]=sctl::log<T>(s)/sctl::log<T>(2.0);
        T err=sctl::sqrt<T>(0.5*(dot22/dot11)/(s*s)-0.5);
        if(err>eps_){
          scale_invar=false;
          M_scal[0][i]=0.0;
        }
        //assert(M_scal[0][i]>=0.0); // Kernel function must decay
      }else if(dot11>max_val*eps ||
               dot22>max_val*eps ){
        scale_invar=false;
        M_scal[0][i]=0.0;
      }else{
        M_scal[0][i]=-1;
      }
    }

    src_scal.Resize(ker_dim[0]); src_scal.SetZero();
    trg_scal.Resize(ker_dim[1]); trg_scal.SetZero();
    if(scale_invar){
      Matrix<T> b(ker_dim[0]*ker_dim[1]+1,1); b.SetZero();
      mem::copy<T>(&b[0][0],&M_scal[0][0],ker_dim[0]*ker_dim[1]);

      Matrix<T> M(ker_dim[0]*ker_dim[1]+1,ker_dim[0]+ker_dim[1]); M.SetZero();
      M[ker_dim[0]*ker_dim[1]][0]=1;
      for(size_t i0=0;i0<ker_dim[0];i0++)
      for(size_t i1=0;i1<ker_dim[1];i1++){
        size_t j=i0*ker_dim[1]+i1;
        if(b[j][0]>0){
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
          if(sctl::fabs<T>(src_scal[i0]+trg_scal[i1]-M_scal[i0][i1])>eps_){
            scale_invar=false;
          }
        }
      }
    }

    if(!scale_invar){
      src_scal.SetZero();
      trg_scal.SetZero();
      //std::cout<<ker_name<<" not-scale-invariant\n";
    }
  }
  if(ker_dim[0]*ker_dim[1]>0){ // Determine symmetry
    size_t N=1024;
    T eps_=N*eps;
    T src_coord[3]={0,0,0};
    std::vector<T> trg_coord1(N*PVFMM_COORD_DIM);
    std::vector<T> trg_coord2(N*PVFMM_COORD_DIM);
    for(size_t i=0;i<N/2;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=sctl::sqrt<T>(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*PVFMM_COORD_DIM+0]=x*scal;
      trg_coord1[i*PVFMM_COORD_DIM+1]=y*scal;
      trg_coord1[i*PVFMM_COORD_DIM+2]=z*scal;
    }
    for(size_t i=N/2;i<N;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=sctl::sqrt<T>(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*PVFMM_COORD_DIM+0]=x*1.0/scal;
      trg_coord1[i*PVFMM_COORD_DIM+1]=y*1.0/scal;
      trg_coord1[i*PVFMM_COORD_DIM+2]=z*1.0/scal;
    }

    for(size_t p_type=0;p_type<C_Perm;p_type++){ // For each symmetry transform

      switch(p_type){ // Set trg_coord2
        case ReflecX:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*PVFMM_COORD_DIM+0]=-trg_coord1[i*PVFMM_COORD_DIM+0];
            trg_coord2[i*PVFMM_COORD_DIM+1]= trg_coord1[i*PVFMM_COORD_DIM+1];
            trg_coord2[i*PVFMM_COORD_DIM+2]= trg_coord1[i*PVFMM_COORD_DIM+2];
          }
          break;
        case ReflecY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*PVFMM_COORD_DIM+0]= trg_coord1[i*PVFMM_COORD_DIM+0];
            trg_coord2[i*PVFMM_COORD_DIM+1]=-trg_coord1[i*PVFMM_COORD_DIM+1];
            trg_coord2[i*PVFMM_COORD_DIM+2]= trg_coord1[i*PVFMM_COORD_DIM+2];
          }
          break;
        case ReflecZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*PVFMM_COORD_DIM+0]= trg_coord1[i*PVFMM_COORD_DIM+0];
            trg_coord2[i*PVFMM_COORD_DIM+1]= trg_coord1[i*PVFMM_COORD_DIM+1];
            trg_coord2[i*PVFMM_COORD_DIM+2]=-trg_coord1[i*PVFMM_COORD_DIM+2];
          }
          break;
        case SwapXY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*PVFMM_COORD_DIM+0]= trg_coord1[i*PVFMM_COORD_DIM+1];
            trg_coord2[i*PVFMM_COORD_DIM+1]= trg_coord1[i*PVFMM_COORD_DIM+0];
            trg_coord2[i*PVFMM_COORD_DIM+2]= trg_coord1[i*PVFMM_COORD_DIM+2];
          }
          break;
        case SwapXZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*PVFMM_COORD_DIM+0]= trg_coord1[i*PVFMM_COORD_DIM+2];
            trg_coord2[i*PVFMM_COORD_DIM+1]= trg_coord1[i*PVFMM_COORD_DIM+1];
            trg_coord2[i*PVFMM_COORD_DIM+2]= trg_coord1[i*PVFMM_COORD_DIM+0];
          }
          break;
        default:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*PVFMM_COORD_DIM+0]= trg_coord1[i*PVFMM_COORD_DIM+0];
            trg_coord2[i*PVFMM_COORD_DIM+1]= trg_coord1[i*PVFMM_COORD_DIM+1];
            trg_coord2[i*PVFMM_COORD_DIM+2]= trg_coord1[i*PVFMM_COORD_DIM+2];
          }
      }

      Matrix<long long> M11, M22;
      {
        Matrix<T> M1(N,ker_dim[0]*ker_dim[1]); M1.SetZero();
        Matrix<T> M2(N,ker_dim[0]*ker_dim[1]); M2.SetZero();
        for(size_t i=0;i<N;i++){
          BuildMatrix(&src_coord [          0], 1,
                      &trg_coord1[i*PVFMM_COORD_DIM], 1, &(M1[i][0]));
          BuildMatrix(&src_coord [          0], 1,
                      &trg_coord2[i*PVFMM_COORD_DIM], 1, &(M2[i][0]));
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
            norm1[i]=sctl::sqrt<T>(dot11[i][i]);
            norm2[i]=sctl::sqrt<T>(dot22[i][i]);
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
              if(sctl::fabs<T>(norm1[i]-norm1[j])<eps_ && sctl::fabs<T>(sctl::fabs<T>(dot11[i][j])-1.0)<eps_){
                M11[0][j]=(dot11[i][j]>0?flag:-flag);
              }
              if(sctl::fabs<T>(norm1[i]-norm2[j])<eps_ && sctl::fabs<T>(sctl::fabs<T>(dot12[i][j])-1.0)<eps_){
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
      { // P1vec
        Matrix<long long>& Pmat=P1;
        std::vector<Permutation<long long> >& Pvec=P1vec;

        Permutation<long long> P(Pmat.Dim(0));
        Vector<PVFMM_PERM_INT_T>& perm=P.perm;
        perm.SetZero();

        // First permutation
        for(size_t i=0;i<P.Dim();i++)
        for(size_t j=0;j<P.Dim();j++){
          if(Pmat[i][j]){
            perm[i]=j;
            break;
          }
        }

        Vector<PVFMM_PERM_INT_T> perm_tmp;
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
            PVFMM_PERM_INT_T tmp=perm[i];
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
        Vector<PVFMM_PERM_INT_T>& perm=P.perm;
        perm.SetZero();

        // First permutation
        for(size_t i=0;i<P.Dim();i++)
        for(size_t j=0;j<P.Dim();j++){
          if(Pmat[i][j]){
            perm[i]=j;
            break;
          }
        }

        Vector<PVFMM_PERM_INT_T> perm_tmp;
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
            PVFMM_PERM_INT_T tmp=perm[i];
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
          { // Check if permutation is symmetric
            Permutation<long long> P1_=P1.Transpose();
            Permutation<long long> P2_=P2.Transpose();
            for(size_t i=0;i<P1.Dim();i++){
              if(P1_.perm[i]!=P1.perm[i] || P1_.scal[i]!=P1.scal[i]){
                done=false;
                break;
              }
            }
            for(size_t i=0;i<P2.Dim();i++){
              if(P2_.perm[i]!=P2.perm[i] || P2_.scal[i]!=P2.scal[i]){
                done=false;
                break;
              }
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
        assert(P1_.Dim() && P2_.Dim());
      }

      //std::cout<<P1_<<'\n';
      //std::cout<<P2_<<'\n';
      perm_vec[p_type       ]=P1_;
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
    std::cout<<"Scale Invariant: "<<(scale_invar?"yes":"no")<<'\n';
    if(scale_invar && ker_dim[0]*ker_dim[1]>0){
      std::cout<<"Scaling Matrix :\n";
      Matrix<T> Src(ker_dim[0],1);
      Matrix<T> Trg(1,ker_dim[1]);
      for(size_t i=0;i<ker_dim[0];i++) Src[i][0]=sctl::pow<T>(2.0,src_scal[i]);
      for(size_t i=0;i<ker_dim[1];i++) Trg[0][i]=sctl::pow<T>(2.0,trg_scal[i]);
      std::cout<<Src*Trg;
    }
    if(ker_dim[0]*ker_dim[1]>0){ // Accuracy of multipole expansion
      std::cout<<"Multipole Error: ";
      for(T rad=1.0; rad>1.0e-2; rad*=0.5){
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

                equiv_surf.push_back(x*PVFMM_RAD0*rad);
                equiv_surf.push_back(y*PVFMM_RAD0*rad);
                equiv_surf.push_back(z*PVFMM_RAD0*rad);

                check_surf.push_back(x*PVFMM_RAD1*rad);
                check_surf.push_back(y*PVFMM_RAD1*rad);
                check_surf.push_back(z*PVFMM_RAD1*rad);
              }
            }
          }
        }
        size_t n_equiv=equiv_surf.size()/PVFMM_COORD_DIM;
        size_t n_check=equiv_surf.size()/PVFMM_COORD_DIM;

        size_t n_src=m*m;
        size_t n_trg=m*m;
        std::vector<T> src_coord;
        std::vector<T> trg_coord;
        for(size_t i=0;i<n_src*PVFMM_COORD_DIM;i++){
          src_coord.push_back((2*drand48()-1)/3*rad);
        }
        for(size_t i=0;i<n_trg;i++){
          T x,y,z,r;
          do{
            x=(drand48()-0.5);
            y=(drand48()-0.5);
            z=(drand48()-0.5);
            r=sctl::sqrt<T>(x*x+y*y+z*z);
          }while(r==0.0);
          trg_coord.push_back(x/r*sctl::sqrt<T>((T)PVFMM_COORD_DIM)*rad*(1.0+drand48()));
          trg_coord.push_back(y/r*sctl::sqrt<T>((T)PVFMM_COORD_DIM)*rad*(1.0+drand48()));
          trg_coord.push_back(z/r*sctl::sqrt<T>((T)PVFMM_COORD_DIM)*rad*(1.0+drand48()));
        }

        Matrix<T> M_s2c(n_src*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix( &src_coord[0], n_src,
                    &check_surf[0], n_check, &(M_s2c[0][0]));

        Matrix<T> M_e2c(n_equiv*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                    &check_surf[0], n_check, &(M_e2c[0][0]));
        Matrix<T> M_c2e0, M_c2e1;
        {
          Matrix<T> U,S,V;
          M_e2c.SVD(U,S,V);
          T eps=1, max_S=0;
          while(eps*(T)0.5+(T)1.0>1.0) eps*=0.5;
          for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
            if(sctl::fabs<T>(S[i][i])>max_S) max_S=sctl::fabs<T>(S[i][i]);
          }
          for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
          M_c2e0=V.Transpose()*S;
          M_c2e1=U.Transpose();
        }

        Matrix<T> M_e2t(n_equiv*ker_dim[0],n_trg*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                     &trg_coord[0], n_trg  , &(M_e2t[0][0]));

        Matrix<T> M_s2t(n_src*ker_dim[0],n_trg*ker_dim[1]);
        BuildMatrix( &src_coord[0], n_src,
                     &trg_coord[0], n_trg  , &(M_s2t[0][0]));

        Matrix<T> M=(M_s2c*M_c2e0)*(M_c2e1*M_e2t)-M_s2t;
        T max_error=0, max_value=0;
        for(size_t i=0;i<M.Dim(0);i++)
        for(size_t j=0;j<M.Dim(1);j++){
          max_error=std::max<T>(max_error,sctl::fabs<T>(M    [i][j]));
          max_value=std::max<T>(max_value,sctl::fabs<T>(M_s2t[i][j]));
        }

        std::cout<<(double)(max_error/max_value)<<' ';
        if(scale_invar) break;
      }
      std::cout<<"\n";
    }
    if(ker_dim[0]*ker_dim[1]>0){ // Accuracy of local expansion
      std::cout<<"Local-exp Error: ";
      for(T rad=1.0; rad>1.0e-2; rad*=0.5){
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

                equiv_surf.push_back(x*PVFMM_RAD1*rad);
                equiv_surf.push_back(y*PVFMM_RAD1*rad);
                equiv_surf.push_back(z*PVFMM_RAD1*rad);

                check_surf.push_back(x*PVFMM_RAD0*rad);
                check_surf.push_back(y*PVFMM_RAD0*rad);
                check_surf.push_back(z*PVFMM_RAD0*rad);
              }
            }
          }
        }
        size_t n_equiv=equiv_surf.size()/PVFMM_COORD_DIM;
        size_t n_check=equiv_surf.size()/PVFMM_COORD_DIM;

        size_t n_src=m*m;
        size_t n_trg=m*m;
        std::vector<T> src_coord;
        std::vector<T> trg_coord;
        for(size_t i=0;i<n_trg*PVFMM_COORD_DIM;i++){
          trg_coord.push_back((2*drand48()-1)/3*rad);
        }
        for(size_t i=0;i<n_src;i++){
          T x,y,z,r;
          do{
            x=(drand48()-0.5);
            y=(drand48()-0.5);
            z=(drand48()-0.5);
            r=sctl::sqrt<T>(x*x+y*y+z*z);
          }while(r==0.0);
          src_coord.push_back(x/r*sctl::sqrt<T>((T)PVFMM_COORD_DIM)*rad*(1.0+drand48()));
          src_coord.push_back(y/r*sctl::sqrt<T>((T)PVFMM_COORD_DIM)*rad*(1.0+drand48()));
          src_coord.push_back(z/r*sctl::sqrt<T>((T)PVFMM_COORD_DIM)*rad*(1.0+drand48()));
        }

        Matrix<T> M_s2c(n_src*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix( &src_coord[0], n_src,
                    &check_surf[0], n_check, &(M_s2c[0][0]));

        Matrix<T> M_e2c(n_equiv*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                    &check_surf[0], n_check, &(M_e2c[0][0]));
        Matrix<T> M_c2e0, M_c2e1;
        {
          Matrix<T> U,S,V;
          M_e2c.SVD(U,S,V);
          T eps=1, max_S=0;
          while(eps*(T)0.5+(T)1.0>1.0) eps*=0.5;
          for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
            if(sctl::fabs<T>(S[i][i])>max_S) max_S=sctl::fabs<T>(S[i][i]);
          }
          for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
          M_c2e0=V.Transpose()*S;
          M_c2e1=U.Transpose();
        }

        Matrix<T> M_e2t(n_equiv*ker_dim[0],n_trg*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                     &trg_coord[0], n_trg  , &(M_e2t[0][0]));

        Matrix<T> M_s2t(n_src*ker_dim[0],n_trg*ker_dim[1]);
        BuildMatrix( &src_coord[0], n_src,
                     &trg_coord[0], n_trg  , &(M_s2t[0][0]));

        Matrix<T> M=(M_s2c*M_c2e0)*(M_c2e1*M_e2t)-M_s2t;
        T max_error=0, max_value=0;
        for(size_t i=0;i<M.Dim(0);i++)
        for(size_t j=0;j<M.Dim(1);j++){
          max_error=std::max<T>(max_error,sctl::fabs<T>(M    [i][j]));
          max_value=std::max<T>(max_value,sctl::fabs<T>(M_s2t[i][j]));
        }

        std::cout<<(double)(max_error/max_value)<<' ';
        if(scale_invar) break;
      }
      std::cout<<"\n";
    }
    if(vol_poten && ker_dim[0]*ker_dim[1]>0){ // Check if the volume potential is consistent with integral of kernel.
      int m=8; // multipole order
      std::vector<T> equiv_surf;
      std::vector<T> check_surf;
      std::vector<T> trg_coord;
      for(size_t i=0;i<m*PVFMM_COORD_DIM;i++){
        trg_coord.push_back(drand48()+1.0);
      }
      for(int i0=0;i0<m;i0++){
        for(int i1=0;i1<m;i1++){
          for(int i2=0;i2<m;i2++){
            if(i0==  0 || i1==  0 || i2==  0 ||
               i0==m-1 || i1==m-1 || i2==m-1){

              // Range: [-1/2,1/2]^3
              T x=((T)2*i0-(m-1))/(m-1)/2;
              T y=((T)2*i1-(m-1))/(m-1)/2;
              T z=((T)2*i2-(m-1))/(m-1)/2;

              equiv_surf.push_back(x*PVFMM_RAD1+1.5);
              equiv_surf.push_back(y*PVFMM_RAD1+1.5);
              equiv_surf.push_back(z*PVFMM_RAD1+1.5);

              check_surf.push_back(x*PVFMM_RAD0+1.5);
              check_surf.push_back(y*PVFMM_RAD0+1.5);
              check_surf.push_back(z*PVFMM_RAD0+1.5);
            }
          }
        }
      }
      size_t n_equiv=equiv_surf.size()/PVFMM_COORD_DIM;
      size_t n_check=equiv_surf.size()/PVFMM_COORD_DIM;
      size_t n_trg  =trg_coord .size()/PVFMM_COORD_DIM;

      Matrix<T> M_local, M_analytic;
      Matrix<T> T_local, T_analytic;
      { // Compute local expansions M_local, T_local
        Matrix<T> M_near(ker_dim[0],n_check*ker_dim[1]);
        Matrix<T> T_near(ker_dim[0],n_trg  *ker_dim[1]);
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0;i<n_check;i++){ // Compute near-interaction for operator M_near
          std::vector<T> M_=cheb_integ<T>(0, &check_surf[i*3], 3.0, *this);
          for(size_t j=0; j<ker_dim[0]; j++)
            for(int k=0; k<ker_dim[1]; k++)
              M_near[j][i*ker_dim[1]+k] = M_[j+k*ker_dim[0]];
        }
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0;i<n_trg;i++){ // Compute near-interaction for targets T_near
          std::vector<T> M_=cheb_integ<T>(0, &trg_coord[i*3], 3.0, *this);
          for(size_t j=0; j<ker_dim[0]; j++)
            for(int k=0; k<ker_dim[1]; k++)
              T_near[j][i*ker_dim[1]+k] = M_[j+k*ker_dim[0]];
        }

        { // M_local = M_analytic - M_near
          M_analytic.ReInit(ker_dim[0],n_check*ker_dim[1]); M_analytic.SetZero();
          vol_poten(&check_surf[0],n_check,&M_analytic[0][0]);
          M_local=M_analytic-M_near;
        }
        { // T_local = T_analytic - T_near
          T_analytic.ReInit(ker_dim[0],n_trg  *ker_dim[1]); T_analytic.SetZero();
          vol_poten(&trg_coord[0],n_trg,&T_analytic[0][0]);
          T_local=T_analytic-T_near;
        }
      }

      Matrix<T> T_err;
      { // Now we should be able to compute T_local from M_local
        Matrix<T> M_e2c(n_equiv*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                    &check_surf[0], n_check, &(M_e2c[0][0]));

        Matrix<T> M_e2t(n_equiv*ker_dim[0],n_trg  *ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                    &trg_coord [0], n_trg  , &(M_e2t[0][0]));

        Matrix<T> M_c2e0, M_c2e1;
        {
          Matrix<T> U,S,V;
          M_e2c.SVD(U,S,V);
          T eps=1, max_S=0;
          while(eps*(T)0.5+(T)1.0>1.0) eps*=0.5;
          for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
            if(sctl::fabs<T>(S[i][i])>max_S) max_S=sctl::fabs<T>(S[i][i]);
          }
          for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
          M_c2e0=V.Transpose()*S;
          M_c2e1=U.Transpose();
        }

        T_err=(M_local*M_c2e0)*(M_c2e1*M_e2t)-T_local;
      }
      { // Print relative error
        T err_sum=0, analytic_sum=0;
        for(size_t i=0;i<T_err     .Dim(0)*T_err     .Dim(1);i++)      err_sum+=sctl::fabs<T>(T_err     [0][i]);
        for(size_t i=0;i<T_analytic.Dim(0)*T_analytic.Dim(1);i++) analytic_sum+=sctl::fabs<T>(T_analytic[0][i]);
        std::cout<<"Volume Error   : "<<err_sum/analytic_sum<<"\n";
      }
    }
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
  #pragma omp parallel for
  for(int i=0;i<src_cnt;i++) //TODO Optimize this.
    for(int j=0;j<ker_dim[0];j++){
      std::vector<T> v_src(ker_dim[0],0);
      v_src[j]=1.0;
      ker_poten(&r_src[i*dim], 1, &v_src[0], 1, r_trg, trg_cnt,
                &k_out[(i*ker_dim[0]+j)*trg_cnt*ker_dim[1]], NULL);
    }
}


/**
 * \brief Generic kernel which rearranges data for vectorization, calls the
 * actual uKernel and copies data to the output array in the original order.
 */
template <class Real_t, int SRC_DIM, int TRG_DIM, void (*uKernel)(Matrix<Real_t>&, Matrix<Real_t>&, Matrix<Real_t>&, Matrix<Real_t>&)>
[[deprecated("generic_kernel interface now replaced by easier/cleaner/potentially faster GenericKernel struct")]]
void generic_kernel(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg, mem::MemoryManager* mem_mgr){
  assert(dof==1);
  int VecLen=8;
  if(sizeof(Real_t)==sizeof( float)) VecLen=8;
  if(sizeof(Real_t)==sizeof(double)) VecLen=4;

  #define STACK_BUFF_SIZE 4096
  Real_t stack_buff[STACK_BUFF_SIZE+PVFMM_MEM_ALIGN];
  Real_t* buff=NULL;

  Matrix<Real_t> src_coord;
  Matrix<Real_t> src_value;
  Matrix<Real_t> trg_coord;
  Matrix<Real_t> trg_value;
  { // Rearrange data in src_coord, src_coord, trg_coord, trg_value
    size_t src_cnt_, trg_cnt_; // counts after zero padding
    src_cnt_=((src_cnt+VecLen-1)/VecLen)*VecLen;
    trg_cnt_=((trg_cnt+VecLen-1)/VecLen)*VecLen;

    size_t buff_size=src_cnt_*(PVFMM_COORD_DIM+SRC_DIM)+
                     trg_cnt_*(PVFMM_COORD_DIM+TRG_DIM);
    if(buff_size>STACK_BUFF_SIZE){ // Allocate buff
      buff=mem::aligned_new<Real_t>(buff_size, mem_mgr);
    }

    Real_t* buff_ptr=buff;
    if(!buff_ptr){ // use stack_buff
      uintptr_t ptr=(uintptr_t)stack_buff;
      static uintptr_t     ALIGN_MINUS_ONE=PVFMM_MEM_ALIGN-1;
      static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
      ptr=((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
      buff_ptr=(Real_t*)ptr;
    }
    src_coord.ReInit(PVFMM_COORD_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=PVFMM_COORD_DIM*src_cnt_;
    src_value.ReInit(  SRC_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=  SRC_DIM*src_cnt_;
    trg_coord.ReInit(PVFMM_COORD_DIM, trg_cnt_,buff_ptr,false);  buff_ptr+=PVFMM_COORD_DIM*trg_cnt_;
    trg_value.ReInit(  TRG_DIM, trg_cnt_,buff_ptr,false);//buff_ptr+=  TRG_DIM*trg_cnt_;
    { // Set src_coord
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<PVFMM_COORD_DIM;j++){
          src_coord[j][i]=r_src[i*PVFMM_COORD_DIM+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<PVFMM_COORD_DIM;j++){
          src_coord[j][i]=0;
        }
      }
    }
    { // Set src_value
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<SRC_DIM;j++){
          src_value[j][i]=v_src[i*SRC_DIM+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<SRC_DIM;j++){
          src_value[j][i]=0;
        }
      }
    }
    { // Set trg_coord
      size_t i=0;
      for(   ;i<trg_cnt ;i++){
        for(size_t j=0;j<PVFMM_COORD_DIM;j++){
          trg_coord[j][i]=r_trg[i*PVFMM_COORD_DIM+j];
        }
      }
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<PVFMM_COORD_DIM;j++){
          trg_coord[j][i]=0;
        }
      }
    }
    { // Set trg_value
      size_t i=0;
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<TRG_DIM;j++){
          trg_value[j][i]=0;
        }
      }
    }
  }
  uKernel(src_coord,src_value,trg_coord,trg_value);
  { // Set v_trg
    for(size_t i=0;i<trg_cnt ;i++){
      for(size_t j=0;j<TRG_DIM;j++){
        v_trg[i*TRG_DIM+j]+=trg_value[j][i];
      }
    }
  }
  if(buff){ // Free memory: buff
    mem::aligned_delete<Real_t>(buff);
  }
}

template <class uKernel> template <class Real, int digits> void GenericKernel<uKernel>::Eval(Real* r_src, int src_cnt, Real* v_src, int dof, Real* r_trg, int trg_cnt, Real* v_trg, mem::MemoryManager* mem_mgr) {
  static constexpr int digits_ = (digits==-1 ? (int)(sctl::TypeTraits<Real>::SigBits*0.3010299957) : digits); // log(2)/log(10) = 0.3010299957
  static constexpr int VecLen = sctl::DefaultVecLen<Real>();
  using RealVec = sctl::Vec<Real, VecLen>;
  assert(dof==1);

  #define STACK_BUFF_SIZE 4096
  alignas(sizeof(RealVec)) Real stack_buff[STACK_BUFF_SIZE];
  Real* buff=nullptr;

  Matrix<Real> src_coord;
  Matrix<Real> src_value;
  Matrix<Real> trg_coord;
  Matrix<Real> trg_value;

  const int src_cnt_ = ((src_cnt + VecLen-1)/VecLen)*VecLen; // count after zero padding
  const int trg_cnt_ = ((trg_cnt + VecLen-1)/VecLen)*VecLen; // count after zero padding
  { // Rearrange data in src_coord, src_coord, trg_coord, trg_value
    int buff_size = src_cnt_*(DIM + KDIM0)+
                    trg_cnt_*(DIM + KDIM1);
    if (buff_size > STACK_BUFF_SIZE) { // Allocate buff
      buff = mem::aligned_new<Real>(buff_size, mem_mgr);
    }

    Real* buff_ptr = buff;
    if (!buff_ptr) buff_ptr = (Real*)stack_buff;
    src_coord.ReInit(  DIM, src_cnt_,buff_ptr,false);  buff_ptr+=DIM  *src_cnt_;
    src_value.ReInit(KDIM0, src_cnt_,buff_ptr,false);  buff_ptr+=KDIM0*src_cnt_;
    trg_coord.ReInit(  DIM, trg_cnt_,buff_ptr,false);  buff_ptr+=DIM  *trg_cnt_;
    trg_value.ReInit(KDIM1, trg_cnt_,buff_ptr,false);//buff_ptr+=KDIM1*trg_cnt_;
    { // Set src_coord
      int i=0;
      for(   ;i<src_cnt ;i++){
        for(int j=0;j<DIM;j++){
          src_coord[j][i]=r_src[i*DIM+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(int j=0;j<DIM;j++){
          src_coord[j][i]=0;
        }
      }
    }
    { // Set src_value
      int i=0;
      for(   ;i<src_cnt ;i++){
        for(int j=0;j<KDIM0;j++){
          src_value[j][i]=v_src[i*KDIM0+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(int j=0;j<KDIM0;j++){
          src_value[j][i]=0;
        }
      }
    }
    { // Set trg_coord
      int i=0;
      for(   ;i<trg_cnt ;i++){
        for(int j=0;j<DIM;j++){
          trg_coord[j][i]=r_trg[i*DIM+j];
        }
      }
      for(   ;i<trg_cnt_;i++){
        for(int j=0;j<DIM;j++){
          trg_coord[j][i]=0;
        }
      }
    }
    { // Set trg_value
      int i=0;
      for(   ;i<trg_cnt_;i++){
        for(int j=0;j<KDIM1;j++){
          trg_value[j][i]=0;
        }
      }
    }
  }

  constexpr int SRC_BLK = 500;
  const RealVec scale = RealVec(uKernel::template ScaleFactor<Real>());
  for (int sblk = 0; sblk < src_cnt_; sblk += SRC_BLK){
    int src_cnt = std::min<int>(src_cnt_-sblk, SRC_BLK);
    for (int t = 0; t < trg_cnt_; t += VecLen) {
      const RealVec tx = RealVec::LoadAligned(&trg_coord[0][t]);
      const RealVec ty = RealVec::LoadAligned(&trg_coord[1][t]);
      const RealVec tz = RealVec::LoadAligned(&trg_coord[2][t]);

      RealVec tv[KDIM1];
      for (int k = 0; k < KDIM1; k++) {
        tv[k] = RealVec::Zero();
      }

      for (int s = sblk; s < sblk + src_cnt; s++) {
        RealVec r[DIM];
        r[0] = tx - RealVec::Load1(&src_coord[0][s]);
        r[1] = ty - RealVec::Load1(&src_coord[1][s]);
        r[2] = tz - RealVec::Load1(&src_coord[2][s]);

        RealVec sv[KDIM0];
        for (int k = 0; k < KDIM0; k++) {
          sv[k] = RealVec::Load1(&src_value[k][s]);
        }
        uKernel::template uKerEval<RealVec, digits_>(tv, r, sv, nullptr);
      }

      for (int k = 0; k < KDIM1; k++) {
        tv[k] = FMA(tv[k], scale, RealVec::LoadAligned(&trg_value[k][t]));
        tv[k].StoreAligned(&trg_value[k][t]);
      }
    }
  }
  { // Add FLOPS
    #ifndef __MIC__
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_* uKernel::FLOPS);
    #endif
  }

  { // Set v_trg
    for(int i=0;i<trg_cnt ;i++){
      for(int j=0;j<KDIM1;j++){
        v_trg[i*KDIM1+j]+=trg_value[j][i];
      }
    }
  }
  if(buff){ // Free memory: buff
    mem::aligned_delete<Real>(buff);
  }
}


////////////////////////////////////////////////////////////////////////////////
////////                   LAPLACE KERNEL                               ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Poisson's equation. Kernel tensor
 * dimension = 1x1.
 */
struct laplace_poten : public GenericKernel<laplace_poten> {
  static const int FLOPS = 9;
  template <class Real> static Real ScaleFactor() {
    return 1.0/(4*sctl::const_pi<Real>());
  }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[1], const VecType (&r)[3], const VecType (&f)[1], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    u[0] = FMA(rinv, f[0], u[0]);
  }
};

template <class Real> void laplace_vol_poten(const Real* coord, int n, Real* out){
  for(int i=0;i<n;i++){
    const Real* c=&coord[i*PVFMM_COORD_DIM];
    Real r_2=c[0]*c[0]+c[1]*c[1]+c[2]*c[2];
    out[i]=-r_2/6;
  }
}


// Laplace double layer potential.
struct laplace_dbl_poten : public GenericKernel<laplace_dbl_poten> {
  static const int FLOPS = 17;
  template <class Real> static Real ScaleFactor() {
    return 1.0/(4*sctl::const_pi<Real>());
  }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[4], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rdotn = r[0]*f[0] + r[1]*f[1] + r[2]*f[2];
    VecType rinv3 = rinv * rinv * rinv;
    u[0] = FMA(rdotn * rinv3, f[3], u[0]);
  }
};


// Laplace grdient kernel.
struct laplace_grad : public GenericKernel<laplace_grad> {
  static const int FLOPS = 16;
  template <class Real> static Real ScaleFactor() {
    return 1.0/(4*sctl::const_pi<Real>());
  }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[1], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType f_rinv3 = rinv * rinv * rinv * f[0];
    u[0] = FMA(r[0], f_rinv3, u[0]);
    u[1] = FMA(r[1], f_rinv3, u[1]);
    u[2] = FMA(r[2], f_rinv3, u[2]);
  }
};


template<class T> const Kernel<T>& LaplaceKernel<T>::potential(){
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten::Eval<T>, laplace_dbl_poten::Eval<T> >("laplace"     , 3, std::pair<int,int>(1,1),
      NULL,NULL,NULL, NULL,NULL,NULL, NULL,NULL, &laplace_vol_poten<T>);
  return potn_ker;
}
template<class T> const Kernel<T>& LaplaceKernel<T>::gradient(){
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten::Eval<T>, laplace_dbl_poten::Eval<T> >("laplace"     , 3, std::pair<int,int>(1,1));
  static Kernel<T> grad_ker=BuildKernel<T, laplace_grad ::Eval<T>                             >("laplace_grad", 3, std::pair<int,int>(1,3),
      &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);
  return grad_ker;
}


////////////////////////////////////////////////////////////////////////////////
////////                   STOKES KERNEL                                ////////
////////////////////////////////////////////////////////////////////////////////

struct stokes_vel : public GenericKernel<stokes_vel> {
  static const int FLOPS = 29;
  template <class Real> static Real ScaleFactor() { return 1.0 / (8 * sctl::const_pi<Real>()); }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[3], const void* ctx_ptr) {
      VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
      VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
      VecType rinv2 = rinv*rinv;
      VecType inner_prod = (f[0]*r[0] + f[1]*r[1] + f[2]*r[2]) * rinv2;
      u[0] = FMA(rinv, f[0] + r[0] * inner_prod, u[0]);
      u[1] = FMA(rinv, f[1] + r[1] * inner_prod, u[1]);
      u[2] = FMA(rinv, f[2] + r[2] * inner_prod, u[2]);
  }
};

struct stokes_press : public GenericKernel<stokes_press> {
  static const int FLOPS = 16;
  template <class Real> static Real ScaleFactor() {
    return 1.0/(4.0*sctl::const_pi<Real>());
  }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[1], const VecType (&r)[3], const VecType (&f)[3], const void* ctx_ptr) {
    VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    VecType dot_sum = r[0] * f[0] + r[1] * f[1] + r[2] * f[2];
    VecType rinv3 = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    rinv3 = rinv3 * rinv3 * rinv3;
    u[0] = FMA(dot_sum, rinv3, u[0]);
  }
};

struct stokes_stress : public GenericKernel<stokes_stress> {
  static const int FLOPS = 43;
  template <class Real> static Real ScaleFactor() {
      return -3.0/(4.0*sctl::const_pi<Real>());
  }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[9], const VecType (&r)[3], const VecType (&f)[3], const void* ctx_ptr) {
    VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rinv2 = rinv * rinv;
    VecType inner_prod = (f[0] * r[0] + f[1] * r[1] + f[2] * r[2]) * rinv2 * rinv2 * rinv;

    u[0] += inner_prod * r[0] * r[0];
    u[1] += inner_prod * r[1] * r[0];
    u[2] += inner_prod * r[2] * r[0];
    u[3] += inner_prod * r[0] * r[1];
    u[4] += inner_prod * r[1] * r[1];
    u[5] += inner_prod * r[2] * r[1];
    u[6] += inner_prod * r[0] * r[2];
    u[7] += inner_prod * r[1] * r[2];
    u[8] += inner_prod * r[2] * r[2];
  }
};

struct stokes_grad : public GenericKernel<stokes_grad> {
  static const int FLOPS = 94;
  template <class Real> static Real ScaleFactor() {
      return 1.0/(8.0*sctl::const_pi<Real>());
  }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[9], const VecType (&r)[3], const VecType (&f)[3], const void* ctx_ptr) {
    VecType r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rinv2 = rinv * rinv;
    VecType rinv3 = rinv2 * rinv;
    VecType inner_prod = (f[0] * r[0] + f[1] * r[1] + f[2] * r[2]);
    const VecType one = (typename VecType::ScalarType)(1.0);
    const VecType three = (typename VecType::ScalarType)(3.0);

    u[0] +=                        (inner_prod * (one - three * r[0] * r[0] * rinv2)) * rinv3;  // 6
    u[1] += (r[1] * f[0] - f[1] * r[0] + inner_prod * (-three * r[1] * r[0] * rinv2)) * rinv3;  // 9
    u[2] += (r[2] * f[0] - f[2] * r[0] + inner_prod * (-three * r[2] * r[0] * rinv2)) * rinv3;

    u[3] += (r[0] * f[1] - f[0] * r[1] + inner_prod * (-three * r[0] * r[1] * rinv2)) * rinv3;
    u[4] +=                        (inner_prod * (one - three * r[1] * r[1] * rinv2)) * rinv3;
    u[5] += (r[2] * f[1] - f[2] * r[1] + inner_prod * (-three * r[2] * r[1] * rinv2)) * rinv3;

    u[6] += (r[0] * f[2] - f[0] * r[2] + inner_prod * (-three * r[0] * r[2] * rinv2)) * rinv3;
    u[7] += (r[1] * f[2] - f[1] * r[2] + inner_prod * (-three * r[1] * r[2] * rinv2)) * rinv3;
    u[8] +=                        (inner_prod * (one - three * r[2] * r[2] * rinv2)) * rinv3;
  }
};

struct stokes_sym_dip : public GenericKernel<stokes_sym_dip> {
  static const int FLOPS = 35;
  template <class Real> static Real ScaleFactor() { return -1.0 / (8 * sctl::const_pi<Real>()); }
  template <class VecType, int digits> static void uKerEval(VecType (&k)[3], const VecType (&r)[3], const VecType (&v_src)[6], const void* ctx_ptr) {
      VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
      VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
      VecType rinv2 = rinv*rinv;
      VecType rinv3 = rinv2*rinv;
      VecType r_dot_f = (v_src[0]*r[0] + v_src[1]*r[1] + v_src[2]*r[2]);
      VecType r_dot_n = (v_src[3]*r[0] + v_src[4]*r[1] + v_src[5]*r[2]);
      VecType n_dot_f = (v_src[0]*v_src[3] + v_src[1]*v_src[1] + v_src[2]*v_src[2]);
      VecType three = (typename VecType::ScalarType)(3.0);

      VecType common = (n_dot_f - three * r_dot_n*r_dot_f*rinv2)*rinv3;
      k[0] += r[0] * common;
      k[1] += r[1] * common;
      k[2] += r[2] * common;
  }
};

template <class Real_t>
void stokes_vol_poten(const Real_t* coord, int n, Real_t* out){
  for(int i=0;i<n;i++){
    const Real_t* c=&coord[i*PVFMM_COORD_DIM];
    Real_t rx_2=c[1]*c[1]+c[2]*c[2];
    Real_t ry_2=c[0]*c[0]+c[2]*c[2];
    Real_t rz_2=c[0]*c[0]+c[1]*c[1];
    out[(0*n+i)*3+0]=-rx_2/4; out[(0*n+i)*3+1]=      0; out[(0*n+i)*3+2]=      0;
    out[(1*n+i)*3+0]=      0; out[(1*n+i)*3+1]=-ry_2/4; out[(1*n+i)*3+2]=      0;
    out[(2*n+i)*3+0]=      0; out[(2*n+i)*3+1]=      0; out[(2*n+i)*3+2]=-rz_2/4;
  }
}


template<class T> const Kernel<T>& StokesKernel<T>::velocity(){
  static Kernel<T> ker=BuildKernel<T, stokes_vel::Eval<T>, stokes_sym_dip::Eval<T>>("stokes_vel"   , 3, std::pair<int,int>(3,3),
      NULL,NULL,NULL, NULL,NULL,NULL, NULL,NULL, &stokes_vol_poten<T>);
  return ker;
}
template<class T> const Kernel<T>& StokesKernel<T>::pressure(){
  static Kernel<T> ker = BuildKernel<T, stokes_press::Eval<T>>("stokes_press", 3, std::pair<int, int>(3, 1));
  return ker;
}
template<class T> const Kernel<T>& StokesKernel<T>::stress(){
  static Kernel<T> ker = BuildKernel<T, stokes_stress::Eval<T>>("stokes_stress", 3, std::pair<int, int>(3, 9));
  return ker;
}
template<class T> const Kernel<T>& StokesKernel<T>::vel_grad(){
    static Kernel<T> ker = BuildKernel<T, stokes_grad::Eval<T>>("stokes_grad", 3, std::pair<int, int>(3, 9));
  return ker;
}


////////////////////////////////////////////////////////////////////////////////
////////                   BIOT-SAVART KERNEL                           ////////
////////////////////////////////////////////////////////////////////////////////

struct biot_savart : public GenericKernel<biot_savart> {
  static const int FLOPS = 24;
  template <class Real> static Real ScaleFactor() { return 1.0 / (4 * sctl::const_pi<Real>()); }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[3], const VecType (&r)[3], const VecType (&f)[3], const void* ctx_ptr) {
      VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
      VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
      VecType rinv3 = rinv*rinv*rinv;

      u[0] = FMA(rinv3, f[1]*r[2] - f[2]*r[1], u[0]);
      u[1] = FMA(rinv3, f[2]*r[0] - f[0]*r[2], u[1]);
      u[2] = FMA(rinv3, f[0]*r[1] - f[1]*r[0], u[2]);
  }
};


template<class T> const Kernel<T>& BiotSavartKernel<T>::potential(){
  static Kernel<T> ker = BuildKernel<T, biot_savart::Eval<T>>("biot_savart", 3, std::pair<int, int>(3, 3));
  return ker;
}


////////////////////////////////////////////////////////////////////////////////
////////                   HELMHOLTZ KERNEL                             ////////
////////////////////////////////////////////////////////////////////////////////

struct helmholtz_poten : public GenericKernel<helmholtz_poten> {
  static const int FLOPS = 20;
  template <class Real> static Real ScaleFactor() { return 1.0 / (4.0 * sctl::const_pi<Real>()); }
  template <class VecType, int digits> static void uKerEval(VecType (&u)[2], const VecType (&r)[3], const VecType (&f)[2], const void* ctx_ptr) {
      const VecType mu = (typename VecType::ScalarType)(20.0 * sctl::const_pi<double>());
      VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
      VecType rinv = sctl::approx_rsqrt<digits>(r2, r2 > VecType::Zero());
      VecType r_mag = rinv * r2;
      VecType mu_r = mu * r_mag;
      VecType G0, G1;
      sincos(G1, G0, mu_r);

      u[0] += (f[0] * G0 - f[1] * G1) * rinv;
      u[1] += (f[0] * G1 + f[1] * G0) * rinv;
  }
};


template<class T> const Kernel<T>& HelmholtzKernel<T>::potential(){
  static Kernel<T> ker = BuildKernel<T, helmholtz_poten::Eval<T>>("helmholtz", 3, std::pair<int, int>(2, 2));
  return ker;
}


}//end namespace
