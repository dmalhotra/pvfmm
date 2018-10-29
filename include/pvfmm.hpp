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
#include <cheb_node.hpp>
#include <mpi_node.hpp>
#include <fmm_tree.hpp>
#include <fmm_node.hpp>
#include <fmm_cheb.hpp>
#include <fmm_pts.hpp>
#include <vector.hpp>
#include <parUtils.h>

#ifndef _PVFMM_HPP_
#define _PVFMM_HPP_

namespace pvfmm{

// Volume FMM data types
template <class Real> using ChebFMM_Node = FMM_Node<Cheb_Node<Real>>;
template <class Real> using ChebFMM      = FMM_Cheb<ChebFMM_Node<Real>>;
template <class Real> using ChebFMM_Tree = FMM_Tree<ChebFMM<Real>>;
template <class Real> using ChebFMM_Data = typename ChebFMM_Node<Real>::NodeData;
template <class Real> using ChebFn       = typename ChebFMM_Node<Real>::Function_t;

template <class Real>
ChebFMM_Tree<Real>* ChebFMM_CreateTree(int cheb_deg, int data_dim, ChebFn<Real> fn_ptr, std::vector<Real>& trg_coord, MPI_Comm& comm,
                                      Real tol=1e-6, int max_pts=100, BoundaryType bndry=FreeSpace, int init_depth=0);

template <class Real>
void ChebFMM_Evaluate(ChebFMM_Tree<Real>* tree, std::vector<Real>& trg_val, size_t loc_size=0);




// Particle FMM data types
template <class Real> using PtFMM_Node = FMM_Node<MPI_Node<Real>>;
template <class Real> using PtFMM      = FMM_Pts<PtFMM_Node<Real>>;
template <class Real> using PtFMM_Tree = FMM_Tree<PtFMM<Real>>;
template <class Real> using PtFMM_Data = typename PtFMM_Node<Real>::NodeData;

/**
 * \brief Create a new instance of the tree and return a pointer. The tree must
 * eventually be be destroyed by calling delete.
 * \param[in] sl_coord   The single-layer source coordinate vector with values: [x1 y1 z1 ... xn yn zn] where (x1 y1 z1) are the coordinates of the first source point. The coordinates must be in [0,1]^3.
 * \param[in] sl_density The single-layer source density vector with values: [u1 v1 w1 ... un vn wn] where (u1 v1 w1) is the density vector for the first particle.
 * \param[in] dl_coord   The double-layer source coordinate vector with values: [x1 y1 z1 ... xn yn zn] where (x1 y1 z1) are the coordinates of the first source point. The coordinates must be in [0,1]^3.
 * \param[in] dl_density The double-layer source density vector with values: [u1 v1 w1 nx1 ny1 nz1 ... un vn wn nxn nyn nzn] where (u1 v1 w1) is the density vector for the first particle and (nx1 ny1 nz1) is the normal vector for the first particle.
 * \param[in] trg_coord  The target coordinate vector with values: [x1 y1 z1 ...  xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 * \param[in] comm       MPI communicator.
 * \param[in] max_pts    Maximum number of source points per octant.
 * \param[in] bndry      Boundary type (FreeSpace or Periodic)
 * \param[in] init_depth Minimum depth for any octant
 */
template <class Real>
PtFMM_Tree<Real>* PtFMM_CreateTree(const std::vector<Real>& sl_coord, const std::vector<Real>& sl_density,
                                   const std::vector<Real>& dl_coord, const std::vector<Real>& dl_density,
                                   const std::vector<Real>& trg_coord, const MPI_Comm& comm, int max_pts=100,
                                   BoundaryType bndry=FreeSpace, int init_depth=0);

/**
 * \brief Create a new instance of the tree and return a pointer.
 * \param[in] sl_coord   The single-layer source coordinate vector with values: [x1 y1 z1 ... xn yn zn] where (x1 y1 z1) are the coordinates of the first source point. The coordinates must be in [0,1]^3.
 * \param[in] sl_density The single-layer source density vector with values: [u1 v1 w1 u2 ... un vn wn] where (u1 v1 w1) is the density vector for the first particle.
 * \param[in] trg_coord  The target coordinate vector with values: [x1 y1 z1 ...  xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 * \param[in] comm       MPI communicator.
 * \param[in] max_pts    Maximum number of source points per octant.
 * \param[in] bndry      Boundary type (FreeSpace or Periodic)
 * \param[in] init_depth Minimum depth for any octant
 */
template <class Real>
PtFMM_Tree<Real>* PtFMM_CreateTree(const std::vector<Real>& sl_coord, const std::vector<Real>& sl_density,
                                   const std::vector<Real>& trg_coord, const MPI_Comm& comm, int max_pts=100,
                                   BoundaryType bndry=FreeSpace, int init_depth=0);

/**
 * \brief Run FMM on the input octree and return the potential at the target
 * points. The setup function PtFMM_Tree::SetupFMM(PtFMM_Tree* fmm_mat) must be
 * called before evaluating FMM for the first time with a tree, or if a tree
 * has changed.
 * \param[in]  tree       Pointer to the octree.
 * \param[out] trg_val    The target potential vector with values: [p1 q1 r1 ... pn qn rn] where (p1 q1 r1) is the potential at the first target point.
 * \param[in]  loc_size   Number of local target points.
 * \param[in]  sl_density The new single-layer source density vector.
 * \param[in]  dl_density The new double-layer source density vector.
 */
template <class Real>
void PtFMM_Evaluate(PtFMM_Tree<Real>* tree, std::vector<Real>& trg_val, size_t loc_size=0, const std::vector<Real>* sl_density=NULL, const std::vector<Real>* dl_density=NULL);

}//end namespace

#include <pvfmm.txx>

#endif //_PVFMM_HPP_
