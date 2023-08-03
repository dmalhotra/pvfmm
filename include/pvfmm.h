/**
 * \file pvfmm.h
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 8-5-2018
 * \brief This file contains the declarations for the C interface to PVFMM.
 */

#ifndef _PVFMM_H_
#define _PVFMM_H_

#include <mpi.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Kernel functions
 */
enum PVFMMKernel{
  PVFMMLaplacePotential    = 0,
  PVFMMLaplaceGradient     = 1,
  PVFMMStokesPressure      = 2,
  PVFMMStokesVelocity      = 3,
  PVFMMStokesVelocityGrad  = 4,
  PVFMMBiotSavartPotential = 5
};

/**
 * \brief Build FMM translation operators.
 *
 * \param[in] m the multipole order (positive, even integer).
 *
 * \param[in] q the degree of the Chebyshev polynomials.
 *
 * \param[in] kernel the kernel function.
 *
 * \param[in] comm the MPI communicator.
 *
 * \return the volume FMM context pointer.
 */
void* PVFMMCreateVolumeFMMD(int m, int q, enum PVFMMKernel kernel, MPI_Comm comm);
void* PVFMMCreateVolumeFMMF(int m, int q, enum PVFMMKernel kernel, MPI_Comm comm);

/**
 * \brief Construct a piecewise Chebyshev volume discretization in [0,1]^3.  It
 * first constructs a uniform tree of depth init_depth and then adaptively
 * refines each leaf node until the tails of the Chebyshev coefficients in each
 * leaf node are smaller than tol and the number of target points in each leaf
 * node are less than max_pts. It further refines the tree to satisfy the 2:1
 * balance constraint.
 *
 * \param[in] cheb_deg the degree of the Chebyshev polynomials. The number of
 * coefficients in each leaf node is
 * data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6.
 *
 * \param[in] data_dim the number of scalar values per point in the evaluation
 * of the input function pointer (fn_ptr).
 *
 * \param[in] fn_ptr the input function pointer.
 *
 * \param[in] fn_ctx a context pointer to be passed to fn_ptr.
 *
 * \param[in] trg_coord the target coordinate vector with values: [x1 y1 z1 ...
 * xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 *
 * \param[in] n_trg number of target points.
 *
 * \param[in] comm MPI communicator.
 *
 * \param[in] tol the tolerance for adaptive refinement.
 *
 * \param[in] max_pts the maximum number of target points per leaf node.
 *
 * \param[in] periodic whether to use periodic boundary conditions.
 *
 * \param[in] init_depth the depth of the initial tree before adaptive
 * refinement. If zero then the depth is the minimum depth so that the number
 * of leaf nodes is greater than the size of the MPI communicator.
 *
 * \return the pointer to the constructed tree. It must be destroyed using
 * PVFMMDestroyVolumeTreeD to free the resources.
 */
void* PVFMMCreateVolumeTreeD(int cheb_deg, int data_dim, void (*fn_ptr)(const double* coord, long n, double* out, const void* ctx), const void* fn_ctx, const double* trg_coord, long n_trg, MPI_Comm comm, double tol, int max_pts, bool periodic, int init_depth);
void* PVFMMCreateVolumeTreeF(int cheb_deg, int data_dim, void (*fn_ptr)(const float* coord, long n, float* out, const void* ctx), const void* fn_ctx, const float* trg_coord, long n_trg, MPI_Comm comm, float tol, int max_pts, bool periodic, int init_depth);

/**
 * \brief Construct a piecewise Chebyshev volume discretization in [0,1]^3.  It
 * first constructs a tree with the given leaf node coordinates and then adds
 * the Chebyshev coefficient to each leaf node.
 *
 * \param[in] Nleaf the number of leaf nodes.
 *
 * \param[in] cheb_deg the degree of the Chebyshev polynomials. The number of
 * coefficients in each leaf node is
 * data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6.
 *
 * \param[in] data_dim the number of scalar values per point in the input density.
 *
 * \param[in] leaf_coord A vector of points [x1 y1 z1 ...  xn yn zn] where each
 * point corresponds to a leaf node in the tree.
 *
 * \param[in] fn_coeff the vector of Chebyshev coefficients of size
 * Nleaf*data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6, where
 * Nleaf=leaf_coord.size()/3 is the number of leaf nodes.
 *
 * \param[in] trg_coord the target coordinate vector with values: [x1 y1 z1 ...
 * xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 *
 * \param[in] n_trg number of target points.
 *
 * \param[in] comm MPI communicator.
 *
 * \param[in] periodic whether to use periodic boundary conditions.
 *
 * \return the pointer to the constructed tree. It must be destroyed using
 * PVFMMDestroyVolumeTreeD to free the resources.
 */
void* PVFMMCreateVolumeTreeFromCoeffD(long Nleaf, int cheb_deg, int data_dim, const double* leaf_coord, const double* fn_coeff, const double* trg_coord, long n_trg, MPI_Comm comm, bool periodic);
void* PVFMMCreateVolumeTreeFromCoeffF(long Nleaf, int cheb_deg, int data_dim, const float* leaf_coord, const float* fn_coeff, const float* trg_coord, long n_trg, MPI_Comm comm, bool periodic);



/**
 * \brief Run volume FMM and evaluate the result at the target points.
 *
 * \param[out] trg_value the computed potential at the target points (in
 * array-of-structure order).
 *
 * \param[in,out] tree the pointer to the Chebyshev tree.
 *
 * \param[in] fmm the volume FMM context pointer.
 *
 * \param[in] loc_size the local size of the output vector (used to partition
 * it among the MPI ranks).
 */
void PVFMMEvaluateVolumeFMMD(double* trg_value, void* tree, const void* fmm, long loc_size);
void PVFMMEvaluateVolumeFMMF(float* trg_value, void* tree, const void* fmm, long loc_size);


/**
 * \brief Destroy the volume FMM context.
 *
 * \param[in,out] ctx a pointer to pointer to the FMM. The pointer value is set
 * to NULL.
 */
void PVFMMDestroyVolumeFMMD(void** ctx);
void PVFMMDestroyVolumeFMMF(void** ctx);

/**
 * \brief Destroy the volume FMM tree.
 *
 * \param[in,out] ctx a pointer to pointer to the tree. The pointer value is
 * set to NULL.
 */
void PVFMMDestroyVolumeTreeD(void** ctx);
void PVFMMDestroyVolumeTreeF(void** ctx);




/**
 * \brief Get the number leaf nodes.
 *
 * \param[in] tree the pointer to the Chebyshev tree.
 *
 * \return the number of leaf nodes.
 */
long PVFMMGetLeafCountD(const void* tree);
long PVFMMGetLeafCountF(const void* tree);

/**
 * \brief Get the leaf node coordinates.
 *
 * \param[out] leaf_coord A vector of points [x1 y1 z1 ...  xn yn zn] where each
 * point corresponds to a leaf node in the tree.
 *
 * \param[in] tree the pointer to the Chebyshev tree.
 */
void PVFMMGetLeafCoordD(double* leaf_coord, const void* tree);
void PVFMMGetLeafCoordF(float* leaf_coord, const void* tree);

/**
 * \brief Get the Chebyshev coefficients for the potential.
 *
 * \param[out] coeff the array of Chebyshev coefficients for the potential of
 * size Nleaf*data_dim*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6, where Nleaf is
 * the number of leaf nodes.
 *
 * \param[in] tree the pointer to the Chebyshev tree.
 */
void PVFMMGetPotentialCoeffD(double* coeff, const void* tree);
void PVFMMGetPotentialCoeffF(float* coeff, const void* tree);

/**
 * \brief Evaluate Chebyshev coefficients at tensor product Chebyshev nodes of
 * first kind.
 *
 * \param[out] node_val node_val the function values at tensor product Chebyshev nodes.
 *
 * \param[in] Nleaf the number of leaf nodes.
 *
 * \param[in] ChebDeg the degree of Chebyshev polynomials.
 *
 * \param[in] dof the number of scalar values at each node point.
 *
 * \param[in] coeff the array of Chebyshev coefficients.
 */
void PVFMMCoeff2NodesD(double* node_val, long Nleaf, int ChebDeg, int dof, const double* coeff);
void PVFMMCoeff2NodesF(float* node_val, long Nleaf, int ChebDeg, int dof, const float* coeff);

/**
 * \brief Convert function values on tensor product Chebyshev nodes (first
 * kind nodes) to coefficients.
 *
 * \param[out] coeff the vector of Chebyshev coefficients.
 *
 * \param[in] Nleaf the number of leaf nodes.
 *
 * \param[in] ChebDeg the degree of Chebyshev polynomials.
 *
 * \param[in] dof the number of scalar values at each node point.
 *
 * \param[in] node_val the function values at tensor product Chebyshev nodes.
 */
void PVFMMNodes2CoeffD(double* coeff, long Nleaf, int ChebDeg, int dof, const double* node_val);
void PVFMMNodes2CoeffF(float* coeff, long Nleaf, int ChebDeg, int dof, const float* node_val);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Create particle FMM context.
 *
 * \param[in] box_size the period length for periodic boundary conditions. Set
 * to zero for free-space boundary conditions.
 *
 * \param[in] n maximum number of points per leaf node.
 *
 * \param[in] m the multipole order (positive, even integer).
 *
 * \param[in] kernel the kernel function.
 *
 * \param[in] comm the MPI communicator.
 *
 * \return the particle FMM context pointer.
 */
void* PVFMMCreateContextD(double box_size, int n, int m, enum PVFMMKernel kernel, MPI_Comm comm);
void* PVFMMCreateContextF(float box_size, int n, int m, enum PVFMMKernel kernel, MPI_Comm comm);

/**
 * \brief Evaluate potential in single-precision.
 *
 * \param[in] src_pos the array of source particle positions: [x1 y1 z1 ... xn
 * yn zn] where (x1 y1 z1) are the coordinates of the first source point.
 *
 * \param[in] sl_den the array of single-layer source densities with values:
 * [u1 v1 w1 ... un vn wn] where (u1 v1 w1) is the density vector for the first
 * particle.
 *
 * \param[in] dl_den the array of double-layer source densities with values:
 * [u1 v1 w1 nx1 ny1 nz1 ... un vn wn nxn nyn nzn] where (u1 v1 w1) is the
 * density vector for the first particle and (nx1 ny1 nz1) is the normal vector
 * for the first particle.
 *
 * \param[in] n_src the number of source particles.
 *
 * \param[in] trg_pos the array of target positions with values: [x1 y1 z1 ...
 * xn yn zn] where (x1 y1 z1) are the coordinates of the first target point.
 *
 * \param[out] trg_val the target potential array with values: [p1 q1 r1 ...
 * pn qn rn] where (p1 q1 r1) is the potential at the first target point.
 *
 * \param[in] n_trg the number of target particles.
 *
 * \param[in] ctx the particle FMM context pointer.
 *
 * \param[in] setup a flag to indicate if the source or target particle
 * positions have changed and therefore additional setup must be performed.
 */
void PVFMMEvalD(const double* src_pos, const double* sl_den, const double* dl_den, long n_src, const double* trg_pos, double* trg_val, long n_trg, const void* ctx, int setup);
void PVFMMEvalF(const float* src_pos, const float* sl_den, const float* dl_den, long n_src, const float* trg_pos, float* trg_val, long n_trg, const void* ctx, int setup);

/**
 * \brief Destroy the particle FMM context.
 *
 * \param[in,out] ctx a pointer to pointer to the FMM context. The pointer
 * value is set to NULL.
 */
void PVFMMDestroyContextD(void** ctx);
void PVFMMDestroyContextF(void** ctx);

#ifdef __cplusplus
}
#endif

#endif //_PVFMM_H_

