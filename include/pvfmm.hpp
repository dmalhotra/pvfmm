/**
 * \file pvfmm.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 1-2-2014
 * \brief This file contains the declaration of wrapper functions for PVFMM.
 */

#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <cmath>

#include <pvfmm_common.hpp>
#include <fmm_node.hpp>
#include <cheb_node.hpp>
#include <fmm_cheb.hpp>

#ifndef _PVFMM_HPP_
#define _PVFMM_HPP_

namespace pvfmm{ // Volume FMM interface

// Volume FMM data types
template <class Real> using ChebFMM_Node = FMM_Node<Cheb_Node<Real>>;
template <class Real> using ChebFMM      = FMM_Cheb<ChebFMM_Node<Real>>;
template <class Real> using ChebFMM_Tree = FMM_Tree<ChebFMM<Real>>;
template <class Real> using ChebFMM_Data = typename ChebFMM_Node<Real>::NodeData;
template <class Real> using ChebFn       = typename ChebFMM_Node<Real>::Function_t;

/**
 * \brief Construct a piecewise Chebyshev volume discretization in [0,1]^3.  It
 * first constructs a uniform tree of depth init_depth and then adaptively
 * refines each leaf node until the tails of the Chebyshev coefficients in each
 * leaf node are smaller than tol and the number of target points in each leaf
 * node are less than max_pts. It further refines the tree to satisfy the 2:1
 * balance constraint.
 *
 * \param[in] cheb_deg The degree of the Chebyshev polynomials. The number of
 * coefficients in each leaf node is
 * data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6.
 *
 * \param[in] data_dim The number of scalar values per point in the evaluation
 * of the input function pointer (fn_ptr).
 *
 * \param[in] fn_ptr The input function pointer.
 *
 * \param[in] trg_coord The target coordinate vector with values: [x1 y1 z1 ...
 * xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 *
 * \param[in] comm MPI communicator.
 *
 * \param[in] tol the tolerance for adaptive refinement.
 *
 * \param[in] max_ptr The maximum number of target points per leaf node.
 *
 * \param[in] bndry The type of boundary conditions (FreeSpace or Periodic)
 *
 * \param[in] init_depth The depth of the initial tree defore adaptive
 * refinement. If zero then the depth is the minimum depth so that the number
 * of leaf nodes is greater than the size of the MPI communicator.
 *
 * \return The pointer to the constructed tree. It must be destroyed using
 * delete to free the resources.
 */
template <class Real>
ChebFMM_Tree<Real>* ChebFMM_CreateTree(int cheb_deg, int data_dim, ChebFn<Real> fn_ptr, std::vector<Real>& trg_coord, MPI_Comm comm,
                                      Real tol=1e-6, int max_pts=100, BoundaryType bndry=FreeSpace, int init_depth=0);


/**
 * \brief Construct a piecewise Chebyshev volume discretization in [0,1]^3.  It
 * first constructs a tree with the given leaf node coordinates and then adds
 * the Chebyshev coefficient to each leaf node.
 *
 * \param[in] cheb_deg The degree of the Chebyshev polynomials.
 *
 * \param[in] leaf_coord A vector of points [x1 y1 z1 ...  xn yn zn] where each
 * point corresponds to a leaf node in the tree.
 *
 * \param[in] fn_coeff The vector of Chebyshev coefficients of size
 * Nleaf*data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6, where
 * Nleaf=leaf_coord.size()/3 is the number of leaf nodes.
 *
 * \param[in] trg_coord The target coordinate vector with values: [x1 y1 z1 ...
 * xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 *
 * \param[in] comm MPI communicator.
 *
 * \param[in] bndry The type of boundary conditions (FreeSpace or Periodic)
 *
 * \return The pointer to the constructed tree. It must be destroyed using
 * delete to free the resources.
 */
template <class Real>
ChebFMM_Tree<Real>* ChebFMM_CreateTree(int cheb_deg, std::vector<Real>& leaf_coord, std::vector<Real>& fn_coeff, std::vector<Real>& trg_coord, MPI_Comm comm, BoundaryType bndry);


/**
 * \brief Run volume FMM and evaluate the result at the target points.
 *
 * \note ChebFMM_Tree::SetupFMM(ChebFMM*) must be called before evaluating FMM for
 * the first time with a tree, or when a tree has changed.
 *
 * \param[out] trg_value The computed potential at the target points (in
 * array-of-structure order).
 *
 * \param[in] tree The pointer to the Chebyshev tree.
 *
 * \param[in] loc_size The local size of the output vector (used to partition
 * it among the MPI ranks).
 */
template <class Real>
void ChebFMM_Evaluate(std::vector<Real>& trg_val, ChebFMM_Tree<Real>* tree, size_t loc_size=0);

/**
 * \brief Get the Chebyshev coefficients for the potential.
 *
 * \param[out] coeff The vector of Chebyshev coefficients for the potential of
 * size Nleaf*data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6, where Nleaf is
 * the number of leaf nodes.
 */
template <class Real>
void ChebFMM_GetPotentialCoeff(std::vector<Real>& coeff, ChebFMM_Tree<Real>* tree);

/**
 * \brief Get the leaf node coordinates.
 *
 * \param[in] leaf_coord A vector of points [x1 y1 z1 ...  xn yn zn] where each
 * point corresponds to a leaf node in the tree.
 */
template <class Real>
void ChebFMM_GetLeafCoord(std::vector<Real>& leaf_coord, ChebFMM_Tree<Real>* tree);

/**
 * \brief Evaluate Chebyshev coefficients at tensor product Chebyshev nodes of
 * first kind.
 *
 * \param[out] node_val node_val The function values at tensor product Chebyshev nodes.
 *
 * \param[in] ChebDeg The degree of Chebyshev polynomials.
 *
 * \param[in] dof The number of scalar values at each node point.
 *
 * \param[in] coeff The vector of Chebyshev coefficients.
 */
template <class Real>
void ChebFMM_Coeff2Nodes(std::vector<Real>& node_val, int ChebDeg, int dof, const std::vector<Real>& coeff);


/**
 * \brief Convert function values on tensor product Chebyshev nodes (first
 * kind nodes) to coefficients.
 *
 * \param[out] coeff The vector of Chebyshev coefficients.
 *
 * \param[in] ChebDeg The degree of Chebyshev polynomials.
 *
 * \param[in] dof The number of scalar values at each node point.
 *
 * \param[in] node_val The function values at tensor product Chebyshev nodes.
 */
template <class Real>
void ChebFMM_Nodes2Coeff(std::vector<Real>& coeff, int ChebDeg, int dof, const std::vector<Real>& node_val);

}

namespace pvfmm{ // Particle FMM interface

/**
 * \brief Manages all the precomputed matrices and implements all the
 * translation operations in FMM. An instance of PtFMM must be created and
 * initialized with the kernel function to be used:
 * PtFMM::Initialize(int mult_order, MPI_Comm comm, const Kernel<Real_t>* kernel, const Kernel<Real_t>* aux_kernel=NULL);
 *
 * mult_order    [in]: the multipole order to be used (even number 2,4,6,8,10,12)
 * comm          [in]: MPI communicator.
 * kernel_fn     [in]: the kernel function pointer to be used.
 * kernel_fn_aux [in]: (optional) Used for V-list translations.
 */
template <class Real> using PtFMM = FMM_Pts<FMM_Node<MPI_Node<Real>>>;

/**
 * \brief  The FMM tree data structure.
 */
template <class Real> using PtFMM_Tree = FMM_Tree<PtFMM<Real>>;

/**
 * \brief The node data structure used in the tree.
 */
template <class Real> using PtFMM_Node = FMM_Node<MPI_Node<Real>>;

/**
 * The data used to initialize the tree.
 */
template <class Real> using PtFMM_Data = typename PtFMM_Node<Real>::NodeData;



/**
 * \brief Create a new instance of the tree and return a pointer. The tree must
 * eventually be be destroyed by calling delete.
 *
 * \param[in] sl_coord The single-layer source coordinate vector with values:
 * [x1 y1 z1 ... xn yn zn] where (x1 y1 z1) are the coordinates of the first
 * source point. The coordinates must be in [0,1]^3.
 *
 * \param[in] sl_density The single-layer source density vector with values:
 * [u1 v1 w1 ... un vn wn] where (u1 v1 w1) is the density vector for the first
 * particle.
 *
 * \param[in] dl_coord The double-layer source coordinate vector with values:
 * [x1 y1 z1 ... xn yn zn] where (x1 y1 z1) are the coordinates of the first
 * source point. The coordinates must be in [0,1]^3.
 *
 * \param[in] dl_density The double-layer source density vector with values:
 * [u1 v1 w1 nx1 ny1 nz1 ... un vn wn nxn nyn nzn] where (u1 v1 w1) is the
 * density vector for the first particle and (nx1 ny1 nz1) is the normal vector
 * for the first particle.
 *
 * \param[in] trg_coord The target coordinate vector with values: [x1 y1 z1 ...
 * xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 *
 * \param[in] comm MPI communicator.
 *
 * \param[in] max_pts Maximum number of source points per octant.
 *
 * \param[in] bndry Boundary type (FreeSpace or Periodic)
 *
 * \param[in] init_depth Minimum depth for any octant
 */
template <class Real>
PtFMM_Tree<Real>* PtFMM_CreateTree(const std::vector<Real>& sl_coord, const std::vector<Real>& sl_density,
                                   const std::vector<Real>& dl_coord, const std::vector<Real>& dl_density,
                                   const std::vector<Real>& trg_coord, MPI_Comm comm, int max_pts=100,
                                   BoundaryType bndry=FreeSpace, int init_depth=0);

/**
 * \brief Create a new instance of the tree and return a pointer. The tree must
 * eventually be be destroyed by calling delete.
 *
 * \param[in] sl_coord The single-layer source coordinate vector with values:
 * [x1 y1 z1 ... xn yn zn] where (x1 y1 z1) are the coordinates of the first
 * source point. The coordinates must be in [0,1]^3.
 *
 * \param[in] sl_density The single-layer source density vector with values:
 * [u1 v1 w1 u2 ... un vn wn] where (u1 v1 w1) is the density vector for the
 * first particle.
 *
 * \param[in] trg_coord The target coordinate vector with values: [x1 y1 z1 ...
 * xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 *
 * \param[in] comm MPI communicator.
 *
 * \param[in] max_pts Maximum number of source points per octant.
 *
 * \param[in] bndry Boundary type (FreeSpace or Periodic)
 *
 * \param[in] init_depth Minimum depth for any octant.
 *
 * \return The pointer to the constructed tree.
 */
template <class Real>
PtFMM_Tree<Real>* PtFMM_CreateTree(const std::vector<Real>& sl_coord, const std::vector<Real>& sl_density,
                                   const std::vector<Real>& trg_coord, MPI_Comm comm, int max_pts=100,
                                   BoundaryType bndry=FreeSpace, int init_depth=0);

/**
 * \brief Run FMM on the input octree and return the potential at the target
 * points. The setup function PtFMM_Tree::SetupFMM(PtFMM_Tree* fmm_mat) must be
 * called before evaluating FMM for the first time with a tree, or if a tree
 * has changed.
 *
 * \note PtFMM_Tree::SetupFMM(PtFMM*) must be called before evaluating FMM for
 * the first time with a tree, or when a tree has changed.
 *
 * \param[in] tree Pointer to the octree.
 *
 * \param[out] trg_val The target potential vector with values: [p1 q1 r1 ...
 * pn qn rn] where (p1 q1 r1) is the potential at the first target point.
 *
 * \param[in] loc_size Number of local target points.
 *
 * \param[in] sl_density The new single-layer source density vector.
 *
 * \param[in] dl_density The new double-layer source density vector.
 */
template <class Real>
void PtFMM_Evaluate(PtFMM_Tree<Real>* tree, std::vector<Real>& trg_val, size_t loc_size=0, const std::vector<Real>* sl_density=NULL, const std::vector<Real>* dl_density=NULL);

}

#include <pvfmm.txx>

#endif //_PVFMM_HPP_
