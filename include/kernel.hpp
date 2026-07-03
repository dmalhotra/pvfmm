/**
 * \file kernel.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-20-2011
 * \brief This file contains the definition of the struct Kernel and also the
 * implementation of various kernels for FMM.
 */

#include <string>
#include <cstdlib>
#include <type_traits>

#include <pvfmm_common.hpp>

#include <mat_utils.hpp>

#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_

namespace sctl {
template <class Real, long N> class Vec;
};

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
                        T* r_trg, int trg_cnt, T* k_out);

  typedef void (*BuildMat_t)(T* r_src, int src_cnt,
                             T* r_trg, int trg_cnt, T* k_out);

  /**
   * \brief Volume potential solution for a constant density f
   * \param[in] coord Coordinates of target points.
   * \param[in] n Number of target points.
   * \param[out] out Elements of a matrix M of size (ker_dim0 x n*ker_dim1),
   * such that fxM gives the target potential.
   */
  using VolPoten = std::function<void(const T* coord, int n, T* out)>;

  /**
   * \brief Constructor.
   */
  Kernel(Ker_t poten = nullptr, Ker_t dbl_poten = nullptr, const char* name = "", int dim_ = 0, std::pair<int,int> k_dim = std::make_pair<int,int>(0,0),
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
  BuildMat_t build_matrix = nullptr;

  size_t dev_ker_poten;
  size_t dev_dbl_layer_poten;

  mutable bool init;
  mutable bool scale_invar;
  mutable sctl::Vector<T> src_scal;
  mutable sctl::Vector<T> trg_scal;
  mutable sctl::Vector<sctl::Permutation<T> > perm_vec;

  mutable const Kernel<T>* k_s2m;
  mutable const Kernel<T>* k_s2l;
  mutable const Kernel<T>* k_s2t;
  mutable const Kernel<T>* k_m2m;
  mutable const Kernel<T>* k_m2l;
  mutable const Kernel<T>* k_m2t;
  mutable const Kernel<T>* k_l2l;
  mutable const Kernel<T>* k_l2t;
  mutable VolPoten vol_poten;
};

template<typename T, class K1, class K2 = void>
Kernel<T> BuildKernel(const char* name, int dim, std::pair<int,int> k_dim,
    const Kernel<T>* k_s2m=NULL, const Kernel<T>* k_s2l=NULL, const Kernel<T>* k_s2t=NULL,
    const Kernel<T>* k_m2m=NULL, const Kernel<T>* k_m2l=NULL, const Kernel<T>* k_m2t=NULL,
    const Kernel<T>* k_l2l=NULL, const Kernel<T>* k_l2t=NULL, typename Kernel<T>::VolPoten vol_poten={}, bool scale_invar_=true){
  using Ker_t = typename Kernel<T>::Ker_t;
  const Ker_t A = &K1::template Eval<T>;
  Ker_t B = nullptr;
  if constexpr (!std::is_same<K2, void>::value) B = &K2::template Eval<T>;

  size_t dev_ker_poten;
  size_t dev_dbl_layer_poten;
  {
    dev_ker_poten      =(size_t)A;
    dev_dbl_layer_poten=(size_t)B;
  }

  Kernel<T> K(A, B, name, dim, k_dim, dev_ker_poten, dev_dbl_layer_poten);
  K.k_s2m=k_s2m; K.k_s2l=k_s2l; K.k_s2t=k_s2t;
  K.k_m2m=k_m2m; K.k_m2l=k_m2l; K.k_m2t=k_m2t;
  K.k_l2l=k_l2l; K.k_l2t=k_l2t;
  K.vol_poten=vol_poten;
  K.scale_invar=scale_invar_;
  K.build_matrix = &K1::template BuildMatrix<T>;
  return K;
}

template <class uKernel> class GenericKernel {
  template <class VecType, int D, int K0, int K1> static constexpr int get_DIM  (void (*uKer)(VecType (&u)[K1], const VecType (&r)[D], const VecType (&f)[K0], const void* ctx_ptr)) { return D; }
  template <class VecType, int D, int K0, int K1> static constexpr int get_KDIM0(void (*uKer)(VecType (&u)[K1], const VecType (&r)[D], const VecType (&f)[K0], const void* ctx_ptr)) { return K0; }
  template <class VecType, int D, int K0, int K1> static constexpr int get_KDIM1(void (*uKer)(VecType (&u)[K1], const VecType (&r)[D], const VecType (&f)[K0], const void* ctx_ptr)) { return K1; }

  static constexpr int DIM   = get_DIM  (uKernel::template uKerEval<sctl::Vec<double,1>,0>);
  static constexpr int KDIM0 = get_KDIM0(uKernel::template uKerEval<sctl::Vec<double,1>,0>);
  static constexpr int KDIM1 = get_KDIM1(uKernel::template uKerEval<sctl::Vec<double,1>,0>);

  public:

  template <class Real, int digits = -1> static void Eval(Real* r_src, int src_cnt, Real* v_src, int dof, Real* r_trg, int trg_cnt, Real* v_trg);

  template <class Real, int digits = -1> static void BuildMatrix(Real* r_src, int src_cnt, Real* r_trg, int trg_cnt, Real* k_out);
};

}//end namespace

namespace pvfmm{ // Predefined Kernel-functions

/**
 * \brief Green's function of the Poisson equation \f$-\Delta u = f\f$.
 */
template<class T>
struct LaplaceKernel{
  inline static const Kernel<T>& potential(); ///< \f$u(x)=\frac{1}{4\pi}\sum_j f_j/|x-y_j|\f$; kernel dimensions (1,1)
  inline static const Kernel<T>& gradient();  ///< \f$\nabla u(x)=-\frac{1}{4\pi}\sum_j f_j\,r_j/|r_j|^3\f$, \f$r_j=x-y_j\f$; kernel dimensions (1,3)
};

/**
 * \brief Green's functions of the Stokes equations (unit viscosity).
 */
template<class T>
struct StokesKernel{
  inline static const Kernel<T>& velocity(); ///< Stokeslet; kernel dimensions (3,3)
  inline static const Kernel<T>& pressure(); ///< associated pressure; kernel dimensions (3,1)
  inline static const Kernel<T>& stress  (); ///< stress tensor (symmetric); kernel dimensions (3,9)
  inline static const Kernel<T>& vel_grad(); ///< velocity gradient, \f$\partial u_k/\partial x_i\f$ at index \f$3i+k\f$; kernel dimensions (3,9)
};

/**
 * \brief Velocity induced by vortex sources (Biot-Savart law).
 */
template<class T>
struct BiotSavartKernel{
  inline static const Kernel<T>& potential(); ///< \f$u(x)=\frac{1}{4\pi}\sum_j \omega_j\times r_j/|r_j|^3\f$; kernel dimensions (3,3)
};

/**
 * \brief Green's function of the Helmholtz equation \f$-\Delta u-\mu^2 u=f\f$
 * with fixed wavenumber \f$\mu=20\pi\f$ (see kernel.txx). Complex values are
 * stored as interleaved (real, imaginary) pairs.
 */
template<class T>
struct HelmholtzKernel{
  inline static const Kernel<T>& potential(); ///< \f$u(x)=\frac{1}{4\pi}\sum_j e^{i\mu|r_j|}/|r_j|\,f_j\f$; kernel dimensions (2,2)
};


}//end namespace

#include <kernel.txx>

#endif //_PVFMM_FMM_KERNEL_HPP_

