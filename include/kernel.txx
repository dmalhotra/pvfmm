/**
 * \file kernel.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-20-2011
 * \brief This file contains the implementation of the struct Kernel and also the
 * implementation of various kernels for FMM.
 */

#ifdef USE_SSE
#include <emmintrin.h>
#endif

#include <math.h>
#include <assert.h>
#include <vector>
#include <profile.hpp>

namespace pvfmm{

/**
 * \brief Constructor.
 */
template <class T>
Kernel<T>::Kernel(): dim(0){
  ker_dim[0]=0;
  ker_dim[1]=0;
}

/**
 * \brief Constructor.
 */
template <class T>
Kernel<T>::Kernel(Ker_t poten, Ker_t dbl_poten, const char* name, int dim_,
                  const int (&k_dim)[2], bool homogen_, T ker_scale,
                  size_t dev_poten, size_t dev_dbl_poten){
  dim=dim_;
  ker_dim[0]=k_dim[0];
  ker_dim[1]=k_dim[1];
  ker_poten=poten;
  dbl_layer_poten=dbl_poten;
  homogen=homogen_;
  poten_scale=ker_scale;
  ker_name=std::string(name);

  dev_ker_poten=dev_poten;
  dev_dbl_layer_poten=dev_dbl_poten;
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
                 T* r_trg, int trg_cnt, T* k_out){
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
//  T* r_src=new T[src_cnt*dim];
//  T* r_trg=new T[trg_cnt*dim];
//  T* v_src=new T[src_cnt    ];
//  T* k_out=new T[trg_cnt    ];
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
//  delete[] r_src;
//  delete[] r_trg;
//  delete[] v_src;
//  delete[] k_out;
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
    if(mem_mgr) buff=(double*)mem_mgr->malloc((ns+1+nt)*3*sizeof(double));
    else buff=(double*)malloc((ns+1+nt)*3*sizeof(double));

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

    if(mem_mgr) mem_mgr->free(buff);
    else free(buff);
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
void laplace_poten<double>(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(12*dof));

  if(dof==1){
    laplaceSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src, k_out, mem_mgr);
    return;
  }
}

template <>
void laplace_dbl_poten<double>(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(19*dof));

  if(dof==1){
    laplaceDblSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src, k_out, mem_mgr);
    return;
  }
}

template <>
void laplace_grad<double>(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
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
void stokes_dbl_vel(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
#ifndef __MIC__
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(32*dof));
#endif

  const T mu=1.0;
  const T SOEPMU = -6.0/(8.0*const_pi<T>()*mu);
  for(int t=0;t<trg_cnt;t++){
    for(int i=0;i<dof;i++){
      T p[3]={0,0,0};
      for(int s=0;s<src_cnt;s++){
        T dX_reg=r_trg[3*t  ]-r_src[3*s  ];
        T dY_reg=r_trg[3*t+1]-r_src[3*s+1];
        T dZ_reg=r_trg[3*t+2]-r_src[3*s+2];
        T R = (dX_reg*dX_reg+dY_reg*dY_reg+dZ_reg*dZ_reg);
        if (R!=0){
          R = sqrt(R);
          T invR=1.0/R;
          T invR5=invR*invR*invR*invR*invR;
          T inner_prod =(v_src[(s*dof+i)*6+0]*dX_reg +
                         v_src[(s*dof+i)*6+1]*dY_reg +
                         v_src[(s*dof+i)*6+2]*dZ_reg)*
                        (v_src[(s*dof+i)*6+3]*dX_reg +
                         v_src[(s*dof+i)*6+4]*dY_reg +
                         v_src[(s*dof+i)*6+5]*dZ_reg)*invR5;
          p[0] += dX_reg*inner_prod;
          p[1] += dY_reg*inner_prod;
          p[2] += dZ_reg*inner_prod;
        }
      }
      k_out[(t*dof+i)*3+0] += p[0]*SOEPMU;
      k_out[(t*dof+i)*3+1] += p[1]*SOEPMU;
      k_out[(t*dof+i)*3+2] += p[2]*SOEPMU;
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
void stokes_vel<double>(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(28*dof));

  const T mu=1.0;
  stokesDirectSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mu, mem_mgr);
}

template <>
void stokes_press<double>(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(17*dof));

  stokesPressureSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mem_mgr);
  return;
}

template <>
void stokes_stress<double>(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(45*dof));

  stokesStressSSEShuffle(src_cnt, trg_cnt, r_src, r_trg, v_src_, k_out, mem_mgr);
}

template <>
void stokes_grad<double>(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr){
  Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*(89*dof));

  const T mu=1.0;
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
