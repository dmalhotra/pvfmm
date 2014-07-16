/**
 * \file kernel.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-20-2011
 * \brief This file contains the definition of the struct Kernel and also the
 * implementation of various kernels for FMM.
 */

#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_

#include <pvfmm_common.hpp>
#include <mem_mgr.hpp>
#include <string>

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif

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
  Kernel();

  /**
   * \brief Constructor.
   */
  Kernel(Ker_t poten, Ker_t dbl_poten, const char* name, int dim_,
         const int (&k_dim)[2], bool homogen_=false, T ker_scale=0,
         size_t dev_poten=(size_t)NULL, size_t dev_dbl_poten=(size_t)NULL);

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
                   T* r_trg, int trg_cnt, T* k_out);

  int dim;
  int ker_dim[2];
  Ker_t ker_poten;
  Ker_t dbl_layer_poten;

  size_t dev_ker_poten;
  size_t dev_dbl_layer_poten;

  bool homogen;
  T poten_scale;
  std::string ker_name;
};

template<typename T, void (*A)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr),
                     void (*B)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr)>
Kernel<T> BuildKernel(const char* name, int dim,
         const int (&k_dim)[2], bool homogen=false, T ker_scale=0){
  size_t dev_ker_poten      ;
  size_t dev_dbl_layer_poten;
  #ifdef __INTEL_OFFLOAD
  #pragma offload target(mic:0)
  #endif
  {
    dev_ker_poten      =(size_t)((typename Kernel<T>::Ker_t)A);
    dev_dbl_layer_poten=(size_t)((typename Kernel<T>::Ker_t)B);
  }

  return Kernel<T>(A, B,
                   name, dim, k_dim, homogen, ker_scale,
                   dev_ker_poten, dev_dbl_layer_poten);
}

template<typename T, void (*A)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr)>
Kernel<T> BuildKernel(const char* name, int dim,
         const int (&k_dim)[2], bool homogen=false, T ker_scale=0){
  size_t dev_ker_poten      ;
  #ifdef __INTEL_OFFLOAD
  #pragma offload target(mic:0)
  #endif
  {
    dev_ker_poten      =(size_t)((typename Kernel<T>::Ker_t)A);
  }

  return Kernel<T>(A, NULL,
                   name, dim, k_dim, homogen, ker_scale,
                   dev_ker_poten, (size_t)NULL);
}

////////////////////////////////////////////////////////////////////////////////
////////                   LAPLACE KERNEL                               ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Poisson's equation. Kernel tensor
 * dimension = 1x1.
 */
template <class T>
void laplace_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

// Laplace double layer potential.
template <class T>
void laplace_dbl_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

// Laplace grdient kernel.
template <class T>
void laplace_grad(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);



int dim_laplace_poten[2]={1,1};
const Kernel<double> laplace_potn_d=BuildKernel<double, laplace_poten, laplace_dbl_poten>("laplace"     , 3, dim_laplace_poten, true, 1.0);
const Kernel<float > laplace_potn_f=BuildKernel<float , laplace_poten, laplace_dbl_poten>("laplace"     , 3, dim_laplace_poten, true, 1.0);

int dim_laplace_grad [2]={1,3};
const Kernel<double> laplace_grad_d=BuildKernel<double, laplace_grad                    >("laplace_grad", 3, dim_laplace_grad , true, 2.0);
const Kernel<float > laplace_grad_f=BuildKernel<float , laplace_grad                    >("laplace_grad", 3, dim_laplace_grad , true, 2.0);

template<class T>
struct LaplaceKernel{
  static Kernel<T>* potn_ker;
  static Kernel<T>* grad_ker;
};

template<> Kernel<double>* LaplaceKernel<double>::potn_ker=(Kernel<double>*)&laplace_potn_d;
template<> Kernel<double>* LaplaceKernel<double>::grad_ker=(Kernel<double>*)&laplace_grad_d;

template<> Kernel<float>* LaplaceKernel<float>::potn_ker=(Kernel<float>*)&laplace_potn_f;
template<> Kernel<float>* LaplaceKernel<float>::grad_ker=(Kernel<float>*)&laplace_grad_f;

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
void stokes_dbl_vel(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

template <class T>
void stokes_press(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

template <class T>
void stokes_stress(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

template <class T>
void stokes_grad(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);



int dim_stokes_vel   [2]={3,3};
const Kernel<double> ker_stokes_vel   =BuildKernel<double, stokes_vel, stokes_dbl_vel>("stokes_vel"   , 3, dim_stokes_vel   ,true,1.0);

int dim_stokes_press [2]={3,1};
const Kernel<double> ker_stokes_press =BuildKernel<double, stokes_press              >("stokes_press" , 3, dim_stokes_press ,true,2.0);

int dim_stokes_stress[2]={3,9};
const Kernel<double> ker_stokes_stress=BuildKernel<double, stokes_stress             >("stokes_stress", 3, dim_stokes_stress,true,2.0);

int dim_stokes_grad  [2]={3,9};
const Kernel<double> ker_stokes_grad  =BuildKernel<double, stokes_grad               >("stokes_grad"  , 3, dim_stokes_grad  ,true,2.0);

////////////////////////////////////////////////////////////////////////////////
////////                  BIOT-SAVART KERNEL                            ////////
////////////////////////////////////////////////////////////////////////////////

template <class T>
void biot_savart(T* r_src, int src_cnt, T* v_src_, int dof, T* r_trg, int trg_cnt, T* k_out, mem::MemoryManager* mem_mgr);

int dim_biot_savart[2]={3,3};
const Kernel<double> ker_biot_savart=BuildKernel<double, biot_savart>("biot_savart", 3, dim_biot_savart,true,2.0);

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



int dim_helmholtz     [2]={2,2};
const Kernel<double> ker_helmholtz     =BuildKernel<double, helmholtz_poten>("helmholtz"     , 3, dim_helmholtz     );

int dim_helmholtz_grad[2]={2,6};
const Kernel<double> ker_helmholtz_grad=BuildKernel<double, helmholtz_grad >("helmholtz_grad", 3, dim_helmholtz_grad);

}//end namespace

#include <kernel.txx>

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_FMM_KERNEL_HPP_

