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
   * \brief Volume potential solution for a constant density f
   * \param[in] coord Coordinates of target points.
   * \param[in] n Number of target points.
   * \param[out] out Elements of a matrix M of size (ker_dim0 x n*ker_dim1),
   * such that fxM gives the target potential.
   */
  typedef void (*VolPoten)(const T* coord, int n, T* out);

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
  int surf_dim; // dimension of source term for double-layer kernel
  std::string ker_name;

  Ker_t ker_poten;
  Ker_t dbl_layer_poten;

  size_t dev_ker_poten;
  size_t dev_dbl_layer_poten;

  mutable bool init;
  mutable bool scale_invar;
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
  mutable VolPoten vol_poten;

  private:

  Kernel();

};

template<typename T, void (*A)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr),
                     void (*B)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr)>
Kernel<T> BuildKernel(const char* name, int dim, std::pair<int,int> k_dim,
    const Kernel<T>* k_s2m=NULL, const Kernel<T>* k_s2l=NULL, const Kernel<T>* k_s2t=NULL,
    const Kernel<T>* k_m2m=NULL, const Kernel<T>* k_m2l=NULL, const Kernel<T>* k_m2t=NULL,
    const Kernel<T>* k_l2l=NULL, const Kernel<T>* k_l2t=NULL, typename Kernel<T>::VolPoten vol_poten=NULL, bool scale_invar_=true){
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
  K.vol_poten=vol_poten;
  K.scale_invar=scale_invar_;

  return K;
}

template<typename T, void (*A)(T*, int, T*, int, T*, int, T*, mem::MemoryManager* mem_mgr)>
Kernel<T> BuildKernel(const char* name, int dim, std::pair<int,int> k_dim,
    const Kernel<T>* k_s2m=NULL, const Kernel<T>* k_s2l=NULL, const Kernel<T>* k_s2t=NULL,
    const Kernel<T>* k_m2m=NULL, const Kernel<T>* k_m2l=NULL, const Kernel<T>* k_m2t=NULL,
    const Kernel<T>* k_l2l=NULL, const Kernel<T>* k_l2t=NULL, typename Kernel<T>::VolPoten vol_poten=NULL, bool scale_invar_=true){
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
  K.vol_poten=vol_poten;
  K.scale_invar=scale_invar_;

  return K;
}

}//end namespace

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{ // Predefined Kernel-functions

template<class T>
struct LaplaceKernel{
  inline static const Kernel<T>& potential();
  inline static const Kernel<T>& gradient();
};

template<class T>
struct StokesKernel{
  inline static const Kernel<T>& velocity();
  inline static const Kernel<T>& pressure();
  inline static const Kernel<T>& stress  ();
  inline static const Kernel<T>& vel_grad();
};

template<class T>
struct BiotSavartKernel{
  inline static const Kernel<T>& potential();
};


template<class T>
struct HelmholtzKernel{
  inline static const Kernel<T>& potential();
};


}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <kernel.txx>

#endif //_PVFMM_FMM_KERNEL_HPP_

