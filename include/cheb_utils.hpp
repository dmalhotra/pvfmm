/**
 * \file cheb_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains chebyshev related functions.
 */

#include <vector>

#include <pvfmm_common.hpp>
#include <vector.hpp>
#include <kernel.hpp>

#ifndef _PVFMM_CHEB_UTILS_HPP_
#define _PVFMM_CHEB_UTILS_HPP_

namespace pvfmm{

/**
 * \brief Returns the sum of the absolute value of coeffecients of the highest
 * order polynomial as an estimate of error.
 */
template <class T>
T cheb_err(T* cheb_coeff, int deg, int dof);

/**
 * \brief Computes Chebyshev approximation from function values at cheb node points.
 */
template <class T, class Y>
T cheb_approx(T* fn_v, int d, int dof, T* out, mem::MemoryManager* mem_mgr=NULL);

/**
 * \brief Evaluates polynomial values from input coefficients at points on
 * a regular grid defined by in_x, in_y, in_z vectors.
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
 * \param[in] deg Maximum degree of the polynomial.
 * \param[in] coord Coordinates of points (x,y,z interleaved).
 * \param[in] node_coord Coordinates of the octant.
 * \param[in] node_size Length of the side of the octant.
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

