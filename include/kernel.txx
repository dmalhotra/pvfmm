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
#include <intrin_wrapper.hpp>
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
    std::vector<T> trg_coord1(N*COORD_DIM);
    Matrix<T> M1(N,ker_dim[0]*ker_dim[1]);
    while(true){
      T abs_sum=0;
      for(size_t i=0;i<N/2;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=pvfmm::sqrt<T>(x*x+y*y+z*z);
        }while(r<0.25);
        trg_coord1[i*COORD_DIM+0]=x*scal;
        trg_coord1[i*COORD_DIM+1]=y*scal;
        trg_coord1[i*COORD_DIM+2]=z*scal;
      }
      for(size_t i=N/2;i<N;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=pvfmm::sqrt<T>(x*x+y*y+z*z);
        }while(r<0.25);
        trg_coord1[i*COORD_DIM+0]=x*1.0/scal;
        trg_coord1[i*COORD_DIM+1]=y*1.0/scal;
        trg_coord1[i*COORD_DIM+2]=z*1.0/scal;
      }
      for(size_t i=0;i<N;i++){
        BuildMatrix(&src_coord [          0], 1,
                    &trg_coord1[i*COORD_DIM], 1, &(M1[i][0]));
        for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
          abs_sum+=pvfmm::fabs<T>(M1[i][j]);
        }
      }
      if(abs_sum>pvfmm::sqrt<T>(eps) || scal<eps) break;
      scal=scal*0.5;
    }

    std::vector<T> trg_coord2(N*COORD_DIM);
    Matrix<T> M2(N,ker_dim[0]*ker_dim[1]);
    for(size_t i=0;i<N*COORD_DIM;i++){
      trg_coord2[i]=trg_coord1[i]*0.5;
    }
    for(size_t i=0;i<N;i++){
      BuildMatrix(&src_coord [          0], 1,
                  &trg_coord2[i*COORD_DIM], 1, &(M2[i][0]));
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
        M_scal[0][i]=pvfmm::log<T>(s)/pvfmm::log<T>(2.0);
        T err=pvfmm::sqrt<T>(0.5*(dot22/dot11)/(s*s)-0.5);
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
      mem::memcopy(&b[0][0],&M_scal[0][0],ker_dim[0]*ker_dim[1]*sizeof(T));

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
          if(pvfmm::fabs<T>(src_scal[i0]+trg_scal[i1]-M_scal[i0][i1])>eps_){
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
    std::vector<T> trg_coord1(N*COORD_DIM);
    std::vector<T> trg_coord2(N*COORD_DIM);
    for(size_t i=0;i<N/2;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=pvfmm::sqrt<T>(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*COORD_DIM+0]=x*scal;
      trg_coord1[i*COORD_DIM+1]=y*scal;
      trg_coord1[i*COORD_DIM+2]=z*scal;
    }
    for(size_t i=N/2;i<N;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=pvfmm::sqrt<T>(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*COORD_DIM+0]=x*1.0/scal;
      trg_coord1[i*COORD_DIM+1]=y*1.0/scal;
      trg_coord1[i*COORD_DIM+2]=z*1.0/scal;
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
            norm1[i]=pvfmm::sqrt<T>(dot11[i][i]);
            norm2[i]=pvfmm::sqrt<T>(dot22[i][i]);
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
              if(pvfmm::fabs<T>(norm1[i]-norm1[j])<eps_ && pvfmm::fabs<T>(pvfmm::fabs<T>(dot11[i][j])-1.0)<eps_){
                M11[0][j]=(dot11[i][j]>0?flag:-flag);
              }
              if(pvfmm::fabs<T>(norm1[i]-norm2[j])<eps_ && pvfmm::fabs<T>(pvfmm::fabs<T>(dot12[i][j])-1.0)<eps_){
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
      for(size_t i=0;i<ker_dim[0];i++) Src[i][0]=pvfmm::pow<T>(2.0,src_scal[i]);
      for(size_t i=0;i<ker_dim[1];i++) Trg[0][i]=pvfmm::pow<T>(2.0,trg_scal[i]);
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

        size_t n_src=m*m;
        size_t n_trg=m*m;
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
            r=pvfmm::sqrt<T>(x*x+y*y+z*z);
          }while(r==0.0);
          trg_coord.push_back(x/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
          trg_coord.push_back(y/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
          trg_coord.push_back(z/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
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
            if(pvfmm::fabs<T>(S[i][i])>max_S) max_S=pvfmm::fabs<T>(S[i][i]);
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
          max_error=std::max<T>(max_error,pvfmm::fabs<T>(M    [i][j]));
          max_value=std::max<T>(max_value,pvfmm::fabs<T>(M_s2t[i][j]));
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

                equiv_surf.push_back(x*RAD1*rad);
                equiv_surf.push_back(y*RAD1*rad);
                equiv_surf.push_back(z*RAD1*rad);

                check_surf.push_back(x*RAD0*rad);
                check_surf.push_back(y*RAD0*rad);
                check_surf.push_back(z*RAD0*rad);
              }
            }
          }
        }
        size_t n_equiv=equiv_surf.size()/COORD_DIM;
        size_t n_check=equiv_surf.size()/COORD_DIM;

        size_t n_src=m*m;
        size_t n_trg=m*m;
        std::vector<T> src_coord;
        std::vector<T> trg_coord;
        for(size_t i=0;i<n_trg*COORD_DIM;i++){
          trg_coord.push_back((2*drand48()-1)/3*rad);
        }
        for(size_t i=0;i<n_src;i++){
          T x,y,z,r;
          do{
            x=(drand48()-0.5);
            y=(drand48()-0.5);
            z=(drand48()-0.5);
            r=pvfmm::sqrt<T>(x*x+y*y+z*z);
          }while(r==0.0);
          src_coord.push_back(x/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
          src_coord.push_back(y/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
          src_coord.push_back(z/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
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
            if(pvfmm::fabs<T>(S[i][i])>max_S) max_S=pvfmm::fabs<T>(S[i][i]);
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
          max_error=std::max<T>(max_error,pvfmm::fabs<T>(M    [i][j]));
          max_value=std::max<T>(max_value,pvfmm::fabs<T>(M_s2t[i][j]));
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
      for(size_t i=0;i<m*COORD_DIM;i++){
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

              equiv_surf.push_back(x*RAD1+1.5);
              equiv_surf.push_back(y*RAD1+1.5);
              equiv_surf.push_back(z*RAD1+1.5);

              check_surf.push_back(x*RAD0+1.5);
              check_surf.push_back(y*RAD0+1.5);
              check_surf.push_back(z*RAD0+1.5);
            }
          }
        }
      }
      size_t n_equiv=equiv_surf.size()/COORD_DIM;
      size_t n_check=equiv_surf.size()/COORD_DIM;
      size_t n_trg  =trg_coord .size()/COORD_DIM;

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
            if(pvfmm::fabs<T>(S[i][i])>max_S) max_S=pvfmm::fabs<T>(S[i][i]);
          }
          for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
          M_c2e0=V.Transpose()*S;
          M_c2e1=U.Transpose();
        }

        T_err=(M_local*M_c2e0)*(M_c2e1*M_e2t)-T_local;
      }
      { // Print relative error
        T err_sum=0, analytic_sum=0;
        for(size_t i=0;i<T_err     .Dim(0)*T_err     .Dim(1);i++)      err_sum+=pvfmm::fabs<T>(T_err     [0][i]);
        for(size_t i=0;i<T_analytic.Dim(0)*T_analytic.Dim(1);i++) analytic_sum+=pvfmm::fabs<T>(T_analytic[0][i]);
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
void generic_kernel(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg, mem::MemoryManager* mem_mgr){
  assert(dof==1);
  int VecLen=8;
  if(sizeof(Real_t)==sizeof( float)) VecLen=8;
  if(sizeof(Real_t)==sizeof(double)) VecLen=4;

  #define STACK_BUFF_SIZE 4096
  Real_t stack_buff[STACK_BUFF_SIZE+MEM_ALIGN];
  Real_t* buff=NULL;

  Matrix<Real_t> src_coord;
  Matrix<Real_t> src_value;
  Matrix<Real_t> trg_coord;
  Matrix<Real_t> trg_value;
  { // Rearrange data in src_coord, src_coord, trg_coord, trg_value
    size_t src_cnt_, trg_cnt_; // counts after zero padding
    src_cnt_=((src_cnt+VecLen-1)/VecLen)*VecLen;
    trg_cnt_=((trg_cnt+VecLen-1)/VecLen)*VecLen;

    size_t buff_size=src_cnt_*(COORD_DIM+SRC_DIM)+
                     trg_cnt_*(COORD_DIM+TRG_DIM);
    if(buff_size>STACK_BUFF_SIZE){ // Allocate buff
      buff=mem::aligned_new<Real_t>(buff_size, mem_mgr);
    }

    Real_t* buff_ptr=buff;
    if(!buff_ptr){ // use stack_buff
      uintptr_t ptr=(uintptr_t)stack_buff;
      static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
      static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
      ptr=((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
      buff_ptr=(Real_t*)ptr;
    }
    src_coord.ReInit(COORD_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=COORD_DIM*src_cnt_;
    src_value.ReInit(  SRC_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=  SRC_DIM*src_cnt_;
    trg_coord.ReInit(COORD_DIM, trg_cnt_,buff_ptr,false);  buff_ptr+=COORD_DIM*trg_cnt_;
    trg_value.ReInit(  TRG_DIM, trg_cnt_,buff_ptr,false);//buff_ptr+=  TRG_DIM*trg_cnt_;
    { // Set src_coord
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<COORD_DIM;j++){
          src_coord[j][i]=r_src[i*COORD_DIM+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<COORD_DIM;j++){
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
        for(size_t j=0;j<COORD_DIM;j++){
          trg_coord[j][i]=r_trg[i*COORD_DIM+j];
        }
      }
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<COORD_DIM;j++){
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


////////////////////////////////////////////////////////////////////////////////
////////                   LAPLACE KERNEL                               ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Poisson's equation. Kernel tensor
 * dimension = 1x1.
 */
template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void laplace_poten_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 1000
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = 1.0/(4*nwtn_scal*const_pi<Real_t>());

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);
      Vec_t tv=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));
        Vec_t sv=              bcast_intrin<Vec_t>(&src_value[0][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));

        Vec_t rinv=RSQRT_INTRIN(r2);
        tv=add_intrin(tv,mul_intrin(rinv,sv));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);
      tv=add_intrin(mul_intrin(tv,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      store_intrin(&trg_value[0][t],tv);
    }
  }

  { // Add FLOPS
    #ifndef __MIC__
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(12+4*(NWTN_ITER)));
    #endif
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void laplace_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define LAP_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 1, 1, laplace_poten_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define LAPLACE_KERNEL LAP_KER_NWTN(0); LAP_KER_NWTN(1); LAP_KER_NWTN(2); LAP_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    LAPLACE_KERNEL;
    #undef Vec_t
  }

  #undef LAP_KER_NWTN
  #undef LAPLACE_KERNEL
}

template <class Real_t>
void laplace_vol_poten(const Real_t* coord, int n, Real_t* out){
  for(int i=0;i<n;i++){
    const Real_t* c=&coord[i*COORD_DIM];
    Real_t r_2=c[0]*c[0]+c[1]*c[1]+c[2]*c[2];
    out[i]=-r_2/6;
  }
}


// Laplace double layer potential.
template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void laplace_dbl_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = -1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*const_pi<Real_t>());

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);
      Vec_t tv=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));
        Vec_t sn0=             bcast_intrin<Vec_t>(&src_value[0][s]) ;
        Vec_t sn1=             bcast_intrin<Vec_t>(&src_value[1][s]) ;
        Vec_t sn2=             bcast_intrin<Vec_t>(&src_value[2][s]) ;
        Vec_t sv=              bcast_intrin<Vec_t>(&src_value[3][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));

        Vec_t rinv=RSQRT_INTRIN(r2);
        Vec_t r3inv=mul_intrin(mul_intrin(rinv,rinv),rinv);

        Vec_t rdotn=            mul_intrin(sn0,dx);
        rdotn=add_intrin(rdotn, mul_intrin(sn1,dy));
        rdotn=add_intrin(rdotn, mul_intrin(sn2,dz));

        sv=mul_intrin(sv,rdotn);
        tv=add_intrin(tv,mul_intrin(r3inv,sv));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);
      tv=add_intrin(mul_intrin(tv,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      store_intrin(&trg_value[0][t],tv);
    }
  }

  { // Add FLOPS
    #ifndef __MIC__
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(20+4*(NWTN_ITER)));
    #endif
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void laplace_dbl_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define LAP_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 4, 1, laplace_dbl_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define LAPLACE_KERNEL LAP_KER_NWTN(0); LAP_KER_NWTN(1); LAP_KER_NWTN(2); LAP_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    LAPLACE_KERNEL;
    #undef Vec_t
  }

  #undef LAP_KER_NWTN
  #undef LAPLACE_KERNEL
}


// Laplace grdient kernel.
template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void laplace_grad_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = -1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*const_pi<Real_t>());

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);
      Vec_t tv0=zero_intrin<Vec_t>();
      Vec_t tv1=zero_intrin<Vec_t>();
      Vec_t tv2=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));
        Vec_t sv=              bcast_intrin<Vec_t>(&src_value[0][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));

        Vec_t rinv=RSQRT_INTRIN(r2);
        Vec_t r3inv=mul_intrin(mul_intrin(rinv,rinv),rinv);

        sv=mul_intrin(sv,r3inv);
        tv0=add_intrin(tv0,mul_intrin(sv,dx));
        tv1=add_intrin(tv1,mul_intrin(sv,dy));
        tv2=add_intrin(tv2,mul_intrin(sv,dz));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);
      tv0=add_intrin(mul_intrin(tv0,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      tv1=add_intrin(mul_intrin(tv1,oofp),load_intrin<Vec_t>(&trg_value[1][t]));
      tv2=add_intrin(mul_intrin(tv2,oofp),load_intrin<Vec_t>(&trg_value[2][t]));
      store_intrin(&trg_value[0][t],tv0);
      store_intrin(&trg_value[1][t],tv1);
      store_intrin(&trg_value[2][t],tv2);
    }
  }

  { // Add FLOPS
    #ifndef __MIC__
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(19+4*(NWTN_ITER)));
    #endif
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void laplace_grad(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define LAP_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 1, 3, laplace_grad_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define LAPLACE_KERNEL LAP_KER_NWTN(0); LAP_KER_NWTN(1); LAP_KER_NWTN(2); LAP_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    LAPLACE_KERNEL;
    #undef Vec_t
  }

  #undef LAP_KER_NWTN
  #undef LAPLACE_KERNEL
}


template<class T> const Kernel<T>& LaplaceKernel<T>::potential(){
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten<T,1>, laplace_dbl_poten<T,1> >("laplace"     , 3, std::pair<int,int>(1,1),
      NULL,NULL,NULL, NULL,NULL,NULL, NULL,NULL, &laplace_vol_poten<T>);
  return potn_ker;
}
template<class T> const Kernel<T>& LaplaceKernel<T>::gradient(){
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten<T,1>, laplace_dbl_poten<T,1> >("laplace"     , 3, std::pair<int,int>(1,1));
  static Kernel<T> grad_ker=BuildKernel<T, laplace_grad <T,1>                         >("laplace_grad", 3, std::pair<int,int>(1,3),
      &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);
  return grad_ker;
}

template<> inline const Kernel<double>& LaplaceKernel<double>::potential(){
  typedef double T;
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten<T,2>, laplace_dbl_poten<T,2> >("laplace"     , 3, std::pair<int,int>(1,1),
      NULL,NULL,NULL, NULL,NULL,NULL, NULL,NULL, &laplace_vol_poten<double>);
  return potn_ker;
}
template<> inline const Kernel<double>& LaplaceKernel<double>::gradient(){
  typedef double T;
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten<T,2>, laplace_dbl_poten<T,2> >("laplace"     , 3, std::pair<int,int>(1,1));
  static Kernel<T> grad_ker=BuildKernel<T, laplace_grad <T,2>                         >("laplace_grad", 3, std::pair<int,int>(1,3),
      &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);
  return grad_ker;
}


////////////////////////////////////////////////////////////////////////////////
////////                     STOKES KERNEL                              ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Stokes's equation. Kernel tensor
 * dimension = 3x3.
 */
template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void stokes_vel_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOEP = 1.0/(8*nwtn_scal*const_pi<Real_t>());
  Vec_t inv_nwtn_scal2=set_intrin<Vec_t,Real_t>(1.0/(nwtn_scal*nwtn_scal));

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);

      Vec_t tvx=zero_intrin<Vec_t>();
      Vec_t tvy=zero_intrin<Vec_t>();
      Vec_t tvz=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));

        Vec_t svx=             bcast_intrin<Vec_t>(&src_value[0][s]) ;
        Vec_t svy=             bcast_intrin<Vec_t>(&src_value[1][s]) ;
        Vec_t svz=             bcast_intrin<Vec_t>(&src_value[2][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));

        Vec_t rinv=RSQRT_INTRIN(r2);
        Vec_t rinv2=mul_intrin(mul_intrin(rinv,rinv),inv_nwtn_scal2);

        Vec_t inner_prod=                mul_intrin(svx,dx) ;
        inner_prod=add_intrin(inner_prod,mul_intrin(svy,dy));
        inner_prod=add_intrin(inner_prod,mul_intrin(svz,dz));
        inner_prod=mul_intrin(inner_prod,rinv2);

        tvx=add_intrin(tvx,mul_intrin(rinv,add_intrin(svx,mul_intrin(dx,inner_prod))));
        tvy=add_intrin(tvy,mul_intrin(rinv,add_intrin(svy,mul_intrin(dy,inner_prod))));
        tvz=add_intrin(tvz,mul_intrin(rinv,add_intrin(svz,mul_intrin(dz,inner_prod))));
      }
      Vec_t ooep=set_intrin<Vec_t,Real_t>(OOEP);

      tvx=add_intrin(mul_intrin(tvx,ooep),load_intrin<Vec_t>(&trg_value[0][t]));
      tvy=add_intrin(mul_intrin(tvy,ooep),load_intrin<Vec_t>(&trg_value[1][t]));
      tvz=add_intrin(mul_intrin(tvz,ooep),load_intrin<Vec_t>(&trg_value[2][t]));

      store_intrin(&trg_value[0][t],tvx);
      store_intrin(&trg_value[1][t],tvy);
      store_intrin(&trg_value[2][t],tvz);
    }
  }

  { // Add FLOPS
    #ifndef __MIC__
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(29+4*(NWTN_ITER)));
    #endif
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void stokes_vel(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define STK_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 3, 3, stokes_vel_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define STOKES_KERNEL STK_KER_NWTN(0); STK_KER_NWTN(1); STK_KER_NWTN(2); STK_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    STOKES_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    STOKES_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    STOKES_KERNEL;
    #undef Vec_t
  }

  #undef STK_KER_NWTN
  #undef STOKES_KERNEL
}

template <class Real_t>
void stokes_vol_poten(const Real_t* coord, int n, Real_t* out){
  for(int i=0;i<n;i++){
    const Real_t* c=&coord[i*COORD_DIM];
    Real_t rx_2=c[1]*c[1]+c[2]*c[2];
    Real_t ry_2=c[0]*c[0]+c[2]*c[2];
    Real_t rz_2=c[0]*c[0]+c[1]*c[1];
    out[(0*n+i)*3+0]=-rx_2/4; out[(0*n+i)*3+1]=      0; out[(0*n+i)*3+2]=      0;
    out[(1*n+i)*3+0]=      0; out[(1*n+i)*3+1]=-ry_2/4; out[(1*n+i)*3+2]=      0;
    out[(2*n+i)*3+0]=      0; out[(2*n+i)*3+1]=      0; out[(2*n+i)*3+2]=-rz_2/4;
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
          T invR=pvfmm::sqrt<T>(invR2);
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
          T invR=pvfmm::sqrt<T>(invR2);
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
          T invR=pvfmm::sqrt<T>(invR2);
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
          T invR=pvfmm::sqrt<T>(invR2);
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
#if defined __SSE3__
namespace
{
#define IDEAL_ALIGNMENT 16
#define SIMD_LEN (int)(IDEAL_ALIGNMENT / sizeof(double))
#define DECL_SIMD_ALIGNED  __declspec(align(IDEAL_ALIGNMENT))

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
        double r = pvfmm::sqrt<double>(r2);
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
        double r = pvfmm::sqrt<double>(r2);
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
        double r = pvfmm::sqrt<double>(r2);
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
inline void stokes_press<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(17*dof));

  stokesPressureSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mem_mgr);
  return;
}

template <>
inline void stokes_stress<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(45*dof));

  stokesStressSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mem_mgr);
}

template <>
inline void stokes_grad<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(89*dof));

  const double mu=1.0;
  stokesGradSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mu, mem_mgr);
}
#endif
#endif

template<class T> const Kernel<T>& StokesKernel<T>::velocity(){
  static Kernel<T> ker=BuildKernel<T, stokes_vel<T,1>, stokes_sym_dip>("stokes_vel"   , 3, std::pair<int,int>(3,3),
      NULL,NULL,NULL, NULL,NULL,NULL, NULL,NULL, &stokes_vol_poten<T>);
  return ker;
}
template<class T> const Kernel<T>& StokesKernel<T>::pressure(){
  static Kernel<T> ker=BuildKernel<T, stokes_press              >("stokes_press" , 3, std::pair<int,int>(3,1));
  return ker;
}
template<class T> const Kernel<T>& StokesKernel<T>::stress(){
  static Kernel<T> ker=BuildKernel<T, stokes_stress             >("stokes_stress", 3, std::pair<int,int>(3,9));
  return ker;
}
template<class T> const Kernel<T>& StokesKernel<T>::vel_grad(){
  static Kernel<T> ker=BuildKernel<T, stokes_grad               >("stokes_grad"  , 3, std::pair<int,int>(3,9));
  return ker;
}

template<> inline const Kernel<double>& StokesKernel<double>::velocity(){
  typedef double T;
  static Kernel<T> ker=BuildKernel<T, stokes_vel<T,2>, stokes_sym_dip>("stokes_vel"   , 3, std::pair<int,int>(3,3),
      NULL,NULL,NULL, NULL,NULL,NULL, NULL,NULL, &stokes_vol_poten<double>);
  return ker;
}


////////////////////////////////////////////////////////////////////////////////
////////                  BIOT-SAVART KERNEL                            ////////
////////////////////////////////////////////////////////////////////////////////

template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void biot_savart_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = 1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*const_pi<Real_t>());

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);

      Vec_t tvx=zero_intrin<Vec_t>();
      Vec_t tvy=zero_intrin<Vec_t>();
      Vec_t tvz=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));

        Vec_t svx=             bcast_intrin<Vec_t>(&src_value[0][s]) ;
        Vec_t svy=             bcast_intrin<Vec_t>(&src_value[1][s]) ;
        Vec_t svz=             bcast_intrin<Vec_t>(&src_value[2][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));

        Vec_t rinv=RSQRT_INTRIN(r2);
        Vec_t rinv3=mul_intrin(mul_intrin(rinv,rinv),rinv);

        tvx=add_intrin(tvx,mul_intrin(rinv3,sub_intrin(mul_intrin(svy,dz),mul_intrin(svz,dy))));
        tvy=add_intrin(tvy,mul_intrin(rinv3,sub_intrin(mul_intrin(svz,dx),mul_intrin(svx,dz))));
        tvz=add_intrin(tvz,mul_intrin(rinv3,sub_intrin(mul_intrin(svx,dy),mul_intrin(svy,dx))));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);

      tvx=add_intrin(mul_intrin(tvx,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      tvy=add_intrin(mul_intrin(tvy,oofp),load_intrin<Vec_t>(&trg_value[1][t]));
      tvz=add_intrin(mul_intrin(tvz,oofp),load_intrin<Vec_t>(&trg_value[2][t]));

      store_intrin(&trg_value[0][t],tvx);
      store_intrin(&trg_value[1][t],tvy);
      store_intrin(&trg_value[2][t],tvz);
    }
  }

  { // Add FLOPS
    #ifndef __MIC__
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(29+4*(NWTN_ITER)));
    #endif
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void biot_savart(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define BS_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 3, 3, biot_savart_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define BIOTSAVART_KERNEL BS_KER_NWTN(0); BS_KER_NWTN(1); BS_KER_NWTN(2); BS_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    BIOTSAVART_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    BIOTSAVART_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    BIOTSAVART_KERNEL;
    #undef Vec_t
  }

  #undef BS_KER_NWTN
  #undef BIOTSAVART_KERNEL
}

template<class T> const Kernel<T>& BiotSavartKernel<T>::potential(){
  static Kernel<T> ker=BuildKernel<T, biot_savart<T,1> >("biot_savart", 3, std::pair<int,int>(3,3));
  return ker;
}
template<> inline const Kernel<double>& BiotSavartKernel<double>::potential(){
  typedef double T;
  static Kernel<T> ker=BuildKernel<T, biot_savart<T,2> >("biot_savart", 3, std::pair<int,int>(3,3));
  return ker;
}


////////////////////////////////////////////////////////////////////////////////
////////                   HELMHOLTZ KERNEL                             ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Helmholtz's equation. Kernel tensor
 * dimension = 2x2.
 */
template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void helmholtz_poten_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){

  #define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = 1.0/(4*nwtn_scal*const_pi<Real_t>());
  const Vec_t mu = set_intrin<Vec_t,Real_t>(20.0*const_pi<Real_t>()/nwtn_scal);

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);

      Vec_t tvx=zero_intrin<Vec_t>();
      Vec_t tvy=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));

        Vec_t svx=             bcast_intrin<Vec_t>(&src_value[0][s]) ;
        Vec_t svy=             bcast_intrin<Vec_t>(&src_value[1][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));
        Vec_t rinv=RSQRT_INTRIN(r2);

        Vec_t mu_r=mul_intrin(mu,mul_intrin(r2,rinv));
        Vec_t G0=mul_intrin(cos_intrin(mu_r),rinv);
        Vec_t G1=mul_intrin(sin_intrin(mu_r),rinv);

        tvx=add_intrin(tvx,sub_intrin(mul_intrin(svx,G0),mul_intrin(svy,G1)));
        tvy=add_intrin(tvy,add_intrin(mul_intrin(svx,G1),mul_intrin(svy,G0)));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);

      tvx=add_intrin(mul_intrin(tvx,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      tvy=add_intrin(mul_intrin(tvy,oofp),load_intrin<Vec_t>(&trg_value[1][t]));

      store_intrin(&trg_value[0][t],tvx);
      store_intrin(&trg_value[1][t],tvy);
    }
  }

  { // Add FLOPS
    #ifndef __MIC__
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(24+4*(NWTN_ITER)));
    #endif
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void helmholtz_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define HELM_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 2, 2, helmholtz_poten_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define HELMHOLTZ_KERNEL HELM_KER_NWTN(0); HELM_KER_NWTN(1); HELM_KER_NWTN(2); HELM_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    HELMHOLTZ_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __MIC__
      #define Vec_t Real_t
    #elif defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    HELMHOLTZ_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    HELMHOLTZ_KERNEL;
    #undef Vec_t
  }

  #undef HELM_KER_NWTN
  #undef HELMHOLTZ_KERNEL
}

template<class T> const Kernel<T>& HelmholtzKernel<T>::potential(){
  static Kernel<T> ker=BuildKernel<T, helmholtz_poten<T,1> >("helmholtz"     , 3, std::pair<int,int>(2,2));
  return ker;
}
template<> inline const Kernel<double>& HelmholtzKernel<double>::potential(){
  typedef double T;
  static Kernel<T> ker=BuildKernel<T, helmholtz_poten<T,3> >("helmholtz"     , 3, std::pair<int,int>(2,2));
  return ker;
}

}//end namespace
