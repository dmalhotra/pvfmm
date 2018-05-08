/**
 * \file pvfmm.h
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 8-5-2018
 * \brief This file contains the declarations for the C interface to PVFMM.
 */

#ifndef _PVFMM_H_
#define _PVFMM_H_

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

enum PVFMMKernel{
  PVFMMLaplacePotential    = 0,
  PVFMMLaplaceGradient     = 1,
  PVFMMStokesPressure      = 2,
  PVFMMStokesVelocity      = 3,
  PVFMMStokesVelocityGrad  = 4,
  PVFMMBiotSavartPotential = 5
};

// Create single-precision particle FMM context
void* PVFMMCreateContextF(float box_size, int n, int m, enum PVFMMKernel kernel, MPI_Comm comm);

// Evaluate potential in single-precision
void PVFMMEvalF(const float* src_pos, const float* sl_den, const float* dl_den, long n_src, const float* trg_pos, float* trg_val, long n_trg, void* ctx, int setup);

// Destroy single-precision particle FMM context
void PVFMMDestroyContextF(void** ctx);


// Create double-precision particle FMM context
void* PVFMMCreateContextD(double box_size, int n, int m, enum PVFMMKernel kernel, MPI_Comm comm);

// Evaluate potential in double-precision
void PVFMMEvalD(const double* src_pos, const double* sl_den, const double* dl_den, long n_src, const double* trg_pos, double* trg_val, long n_trg, void* ctx, int setup);

// Destroy double-precision particle FMM context
void PVFMMDestroyContextD(void** ctx);

#ifdef __cplusplus
}
#endif

#endif //_PVFMM_H_

