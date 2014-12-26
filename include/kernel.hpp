/**
 * \file kernel.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-20-2011
 * \brief This file contains the definition of the struct Kernel and also the
 * implementation of various kernels for FMM.
 */

#include <string>
#include <cstdlib>

#include <pvfmm_common.hpp>
#include <mem_mgr.hpp>
#include <vector.hpp>
#include <matrix.hpp>

#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_

namespace pvfmm{

template <class T>
struct Kernel{
  public:

  /**
   * \brief Evaluate potential due to source points at target coordinates.
   * \param[in] r_src Coordinates of source points.
   * \param[in] src_cnt Number of source points.
   * \param[in] v_src Strength of source points.
   * \param[in] r_trg Coordinates of target points.
   * \param[in] trg_cnt Number of target points.
   * \param[out] k_out Output array with potential values.
   */
  typedef void (*Ker_t)(T* r_src, int src_cnt, T* v_src, int dof,
                        T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

  /**
   * \brief Constructor.
   */
  Kernel(Ker_t poten, Ker_t dbl_poten, const char* name, int dim_, std::pair<int,int> k_dim,
         size_t dev_poten=(size_t)NULL, size_t dev_dbl_poten=(size_t)NULL);

  /**
   * \brief Initialize the kernel.
   */
  void Initialize(bool verbose=false) const;

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
  void BuildMatrix(T* r_src, int src_cnt,
                   T* r_trg, int trg_cnt, T* k_out) const;

  int dim;
  int ker_dim[2];
  std::string ker_name;

  Ker_t ker_poten;
  Ker_t dbl_layer_poten;

  size_t dev_ker_poten;
  size_t dev_dbl_layer_poten;

  mutable bool init;
  mutable bool homogen;
  mutable Vector<T> src_scal;
  mutable Vector<T> trg_scal;
  mutable Vector<Permutation<T> > perm_vec;

  mutable const Kernel<T>* k_s2m;
  mutable const Kernel<T>* k_s2l;
  mutable const Kernel<T>* k_s2t;
  mutable const Kernel<T>* k_m2m;
  mutable const Kernel<T>* k_m2l;
  mutable const Kernel<T>* k_m2t;
  mutable const Kernel<T>* k_l2l;
  mutable const Kernel<T>* k_l2t;

  private:

  Kernel();

};

template<typename T, void (*A)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr),
                     void (*B)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr)>
Kernel<T> BuildKernel(const char* name, int dim, std::pair<int,int> k_dim,
    const Kernel<T>* k_s2m=NULL, const Kernel<T>* k_s2l=NULL, const Kernel<T>* k_s2t=NULL,
    const Kernel<T>* k_m2m=NULL, const Kernel<T>* k_m2l=NULL, const Kernel<T>* k_m2t=NULL,
    const Kernel<T>* k_l2l=NULL, const Kernel<T>* k_l2t=NULL){
  size_t dev_ker_poten      ;
  size_t dev_dbl_layer_poten;
  #ifdef __INTEL_OFFLOAD
  #pragma offload target(mic:0)
  #endif
  {
    dev_ker_poten      =(size_t)((typename Kernel<T>::Ker_t)A);
    dev_dbl_layer_poten=(size_t)((typename Kernel<T>::Ker_t)B);
  }

  Kernel<T> K(A, B, name, dim, k_dim,
              dev_ker_poten, dev_dbl_layer_poten);

  K.k_s2m=k_s2m;
  K.k_s2l=k_s2l;
  K.k_s2t=k_s2t;
  K.k_m2m=k_m2m;
  K.k_m2l=k_m2l;
  K.k_m2t=k_m2t;
  K.k_l2l=k_l2l;
  K.k_l2t=k_l2t;

  return K;
}

template<typename T, void (*A)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr)>
Kernel<T> BuildKernel(const char* name, int dim, std::pair<int,int> k_dim,
    const Kernel<T>* k_s2m=NULL, const Kernel<T>* k_s2l=NULL, const Kernel<T>* k_s2t=NULL,
    const Kernel<T>* k_m2m=NULL, const Kernel<T>* k_m2l=NULL, const Kernel<T>* k_m2t=NULL,
    const Kernel<T>* k_l2l=NULL, const Kernel<T>* k_l2t=NULL){
  size_t dev_ker_poten      ;
  #ifdef __INTEL_OFFLOAD
  #pragma offload target(mic:0)
  #endif
  {
    dev_ker_poten      =(size_t)((typename Kernel<T>::Ker_t)A);
  }

  Kernel<T> K(A, NULL, name, dim, k_dim,
              dev_ker_poten, (size_t)NULL);

  K.k_s2m=k_s2m;
  K.k_s2l=k_s2l;
  K.k_s2t=k_s2t;
  K.k_m2m=k_m2m;
  K.k_m2l=k_m2l;
  K.k_m2t=k_m2t;
  K.k_l2l=k_l2l;
  K.k_l2t=k_l2t;

  return K;
}

}//end namespace

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{ // Predefined Kernel-functions

////////////////////////////////////////////////////////////////////////////////
////////                   LAPLACE KERNEL                               ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Poisson's equation. Kernel tensor
 * dimension = 1x1.
 */
template <class T, int newton_iter=0>
void laplace_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

// Laplace double layer potential.
template <class T, int newton_iter=0>
void laplace_dbl_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

// Laplace grdient kernel.
template <class T, int newton_iter=0>
void laplace_grad(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);


//#ifdef PVFMM_QUAD_T
//const Kernel<QuadReal_t> laplace_potn_q=BuildKernel<QuadReal_t, laplace_poten, laplace_dbl_poten>("laplace"     , 3, std::pair<int,int>(1,1));
//const Kernel<QuadReal_t> laplace_grad_q=BuildKernel<QuadReal_t, laplace_grad                    >("laplace_grad", 3, std::pair<int,int>(1,3));
//#endif

const Kernel<double    > laplace_potn_d=BuildKernel<double    , laplace_poten<double,2>, laplace_dbl_poten<double,2> >("laplace"     , 3, std::pair<int,int>(1,1));
const Kernel<double    > laplace_grad_d=BuildKernel<double    , laplace_grad <double,2>                              >("laplace_grad", 3, std::pair<int,int>(1,3),
  &laplace_potn_d, &laplace_potn_d, NULL, &laplace_potn_d, &laplace_potn_d, NULL, &laplace_potn_d, NULL);

const Kernel<float     > laplace_potn_f=BuildKernel<float     , laplace_poten<float,1>, laplace_dbl_poten<float,1> >("laplace"     , 3, std::pair<int,int>(1,1));
const Kernel<float     > laplace_grad_f=BuildKernel<float     , laplace_grad <float,1>                             >("laplace_grad", 3, std::pair<int,int>(1,3),
  &laplace_potn_f, &laplace_potn_f, NULL, &laplace_potn_f, &laplace_potn_f, NULL, &laplace_potn_f, NULL);

template<class T>
struct LaplaceKernel{
  inline static const Kernel<T>& potn_ker();
  inline static const Kernel<T>& grad_ker();
};

//#ifdef PVFMM_QUAD_T
//template<> const Kernel<QuadReal_t>& LaplaceKernel<QuadReal_t>::potn_ker(){ return laplace_potn_q; };
//template<> const Kernel<QuadReal_t>& LaplaceKernel<QuadReal_t>::grad_ker(){ return laplace_grad_q; };
//#endif

template<> const Kernel<double>& LaplaceKernel<double>::potn_ker(){ return laplace_potn_d; };
template<> const Kernel<double>& LaplaceKernel<double>::grad_ker(){ return laplace_grad_d; };

template<> const Kernel<float>& LaplaceKernel<float>::potn_ker(){ return laplace_potn_f; };
template<> const Kernel<float>& LaplaceKernel<float>::grad_ker(){ return laplace_grad_f; };

////////////////////////////////////////////////////////////////////////////////
////////                   STOKES KERNEL                             ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Stokes's equation. Kernel tensor
 * dimension = 3x3.
 */
template <class T>
void stokes_vel(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

template <class T>
void stokes_sym_dip(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

template <class T>
void stokes_press(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

template <class T>
void stokes_stress(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

template <class T>
void stokes_grad(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);


#ifndef __MIC__
#ifdef USE_SSE
template <>
void stokes_vel<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr);

template <>
void stokes_press<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr);

template <>
void stokes_stress<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr);

template <>
void stokes_grad<double>(double* r_src, int src_cnt, double* v_src_, int dof, double* r_trg, int trg_cnt, double* k_out, mem::MemoryManager* mem_mgr);
#endif
#endif

const Kernel<double> ker_stokes_vel   =BuildKernel<double, stokes_vel, stokes_sym_dip>("stokes_vel"   , 3, std::pair<int,int>(3,3));

const Kernel<double> ker_stokes_press =BuildKernel<double, stokes_press              >("stokes_press" , 3, std::pair<int,int>(3,1));

const Kernel<double> ker_stokes_stress=BuildKernel<double, stokes_stress             >("stokes_stress", 3, std::pair<int,int>(3,9));

const Kernel<double> ker_stokes_grad  =BuildKernel<double, stokes_grad               >("stokes_grad"  , 3, std::pair<int,int>(3,9));

////////////////////////////////////////////////////////////////////////////////
////////                  BIOT-SAVART KERNEL                            ////////
////////////////////////////////////////////////////////////////////////////////

template <class T>
void biot_savart(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

const Kernel<double> ker_biot_savart=BuildKernel<double, biot_savart>("biot_savart", 3, std::pair<int,int>(3,3));

////////////////////////////////////////////////////////////////////////////////
////////                   HELMHOLTZ KERNEL                             ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Helmholtz's equation. Kernel tensor
 * dimension = 2x2.
 */
template <class T>
void helmholtz_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

template <class T>
void helmholtz_grad(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);



const Kernel<double> ker_helmholtz     =BuildKernel<double, helmholtz_poten>("helmholtz"     , 3, std::pair<int,int>(2,2));

const Kernel<double> ker_helmholtz_grad=BuildKernel<double, helmholtz_grad >("helmholtz_grad", 3, std::pair<int,int>(2,6));

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <kernel.txx>

#endif //_PVFMM_FMM_KERNEL_HPP_

