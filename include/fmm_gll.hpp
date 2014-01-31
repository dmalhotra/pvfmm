#ifndef _FMM_GLL_HPP_
#define _FMM_GLL_HPP_

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct{
    MPI_Comm comm;
    int gll_order;
    int cheb_order;
    int multipole_order;

    void* fmm_mat_biotsavart;
    const void* kernel_biotsavart;
    void* tree_biotsavart;

    void* fmm_mat_laplace_grad;
    const void* kernel_laplace_grad;
    void* tree_laplace_grad;

    void* gll_nodes;
  }FMM_GLL_t;

  void fmm_gll_init(FMM_GLL_t* fmm_data, int gll_order, int cheb_order, int multipole_order, MPI_Comm comm);

  void fmm_gll_free(FMM_GLL_t* fmm_data);

  void fmm_gll_run(FMM_GLL_t* fmm_data, size_t K, double* node_coord, unsigned char* node_depth, double** node_gll_data);

  void fmm_gll_laplace_grad(FMM_GLL_t* fmm_data, size_t K, double* node_coord, unsigned char* node_depth, double** node_gll_data);


  void gll_div(FMM_GLL_t* fmm_data, size_t K, double* node_coord, unsigned char* node_depth, double** node_gll_data);

  void gll_divfree(FMM_GLL_t* fmm_data, size_t K, double* node_coord, unsigned char* node_depth, double** node_gll_data);

  void gll_filter(FMM_GLL_t* fmm_data, int cheb_order, size_t node_cnt, double** node_gll_data, double* err);

  void gll_interpolate(FMM_GLL_t* fmm_data, size_t node_cnt, double* node_coord, unsigned char* node_depth, double** node_gll_data);

#ifdef __cplusplus
}
#endif

#endif
