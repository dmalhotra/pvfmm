/**
 * \file cheb_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains chebyshev related functions.
 */

#include <vector>

#include <pvfmm_common.hpp>
#include <vector.hpp>

#ifndef _PVFMM_CHEB_UTILS_HPP_
#define _PVFMM_CHEB_UTILS_HPP_

namespace pvfmm{

template <class Real> struct Kernel;

/**
 * \brief Returns the sum of the absolute value of coeffecients of the highest
 * order polynomial as an estimate of error.
 *
 * \param[in] cheb_coeff The coefficient array of size
 * dof*(deg+1)*(deg+2)*(deg+3)/6.
 *
 * \param[in] deg The degree of the Chebyshev approximation.
 *
 * \param[in] dof The number of real scalar values associated with each
 * Chebyshev node point.
 *
 * \return The estimated truction error.
 */
template <class T>
T cheb_err(T* cheb_coeff, int deg, int dof);

/**
 * \brief Computes Chebyshev approximation from function values at cheb node
 * points.
 *
 * \param[in] fn_v The array of function values on a tensor product Chebyshev
 * grid (first kind nodes) of size dof*(deg+1)^3.
 *
 * \param[in] deg The degree of the Chebyshev approximation.
 *
 * \param[in] dof The number of real scalar values associated with each
 * Chebyshev node point.
 *
 * \param[out] cheb_coeff The coefficient array of size
 * dof*(deg+1)*(deg+2)*(deg+3)/6.
 *
 * \return Estimate of the truncation error.
 */
template <class T, class Y>
T cheb_approx(const T* fn_v, int deg, int dof, T* cheb_coeff, mem::MemoryManager* mem_mgr=NULL);

/**
 * \brief Evaluates polynomial values from input coefficients at points on
 * a tensor product grid defined by in_x, in_y, in_z vectors.
 *
 * \param[in] coeff_ The vector of Chebshev coefficients.
 *
 * \param[in] cheb_deg The degree of the Chebyshev approximation.
 *
 * \param[in] in_x The nodes in [0,1] in the X-direction.
 *
 * \param[in] in_y The nodes in [0,1] in the Y-direction.
 *
 * \param[in] in_z The nodes in [0,1] in the Z-direction.
 */
template <class T>
void cheb_eval(const Vector<T>& coeff_, int cheb_deg, const std::vector<T>& in_x, const std::vector<T>& in_y, const std::vector<T>& in_z, Vector<T>& out, mem::MemoryManager* mem_mgr=NULL);

/**
 * \brief Evaluates polynomial values from input coefficients at points
 * defined by the values in the coord vector.
 */
template <class T>
void cheb_eval(Vector<T>& coeff_, int cheb_deg, std::vector<T>& coord, Vector<T>& out);

/**
 * \brief Computes a least squares solution for Chebyshev approximation over a
 * cube from point samples.
 *
 * \param[in] deg Maximum degree of the polynomial.
 *
 * \param[in] coord Coordinates of points (x,y,z interleaved).
 *
 * \param[in] node_coord Coordinates of the octant.
 *
 * \param[in] node_size Length of the side of the octant.
 *
 * \param[out] cheb_coeff Output coefficients.
 */
template <class T>
void points2cheb(int deg, T* coord, T* val, int n, int dim, T* node_coord, T node_size, Vector<T>& cheb_coeff);

/**
 * \brief Returns an n-point quadrature rule with points 'x' and weights 'w'.
 * Gauss-Legendre quadrature rule for double precision and Chebyshev quadrature
 * rule for other data types.
 */
template <class T>
void quad_rule(int n, T* x, T* w);

/**
 * \brief
 * \param[in] r Length of the side of cubic region.
 */
template <class T>
std::vector<T> cheb_integ(int m, T* s, T r, const Kernel<T>& kernel);

/**
 * \brief Returns coordinates of Chebyshev node points in 'dim' dimensional
 * space.
 */
template <class T>
std::vector<T> cheb_nodes(int deg, int dim);

template <class T>
void cheb_grad(const Vector<T>& A, int deg, Vector<T>& B, mem::MemoryManager* mem_mgr=NULL);

template <class T>
void cheb_laplacian(T* A, int deg, T* B);

template <class T>
void cheb_curl(T* A, int deg, T* B);

/*
 * \brief Computes image of the chebyshev interpolation along the specified axis.
 */
template <class T>
void cheb_img(T* A, T* B, int cheb_deg, int dir, bool neg_);

}//end namespace

#include <cheb_utils.txx>

#endif //_PVFMM_CHEB_UTILS_HPP_

