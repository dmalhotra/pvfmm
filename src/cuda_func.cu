#include "stdint.h"

#define DEFAULT_NUM_THREAD 256

/* Case: double */
__global__ void in_perm_k (
  uintptr_t *precomp_data,
  uintptr_t *input_perm,
  uintptr_t *input_data,
  uintptr_t *buff_in,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt )
{
  /* 1-dim thread Id. */
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  /* Convert to ptr. */
  int *perm = (int*) (precomp_data[0] + input_perm[(interac_indx + tid)*4 + 0]);
  double *scal = (double*) (precomp_data[0] + input_perm[(interac_indx + tid)*4 + 1]);
  double *v_in = (double*) (input_data[0] + input_perm[(interac_indx + tid)*4 + 3]);
  double *v_out = (double*) (buff_in + input_perm[(interac_indx + tid)*4 + 2]);

  if (tid < vec_cnt) {
    /* PRAM Model: assuming as many threads as we need. */
    for (int j = 0; j < M_dim0; j++) v_out[j] = v_in[perm[j]]*scal[j];
  }
}

__global__ void out_perm_k (
  uintptr_t *scaling,
  uintptr_t *precomp_data,
  uintptr_t *output_perm,
  uintptr_t *output_data,
  uintptr_t *buff_out,
  size_t interac_indx,
  size_t M_dim1,
  size_t vec_cnt )
{
  /* 1-dim thread Id. */
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  /* Specifing range. */
  int a = tid;
  int b = tid + 1;

  if (tid > 0 && a < vec_cnt) { // Find 'a' independent of other threads.
    size_t out_ptr = output_perm[(interac_indx + a)*4 + 3];
    if (tid > 0) while(a < vec_cnt && out_ptr == output_perm[(interac_indx+a)*4 + 3]) a++;
  }
  if (tid < vec_cnt - 1 && b < vec_cnt) { // Find 'b' independent of other threads.
    size_t out_ptr = output_perm[(interac_indx + b)*4 + 3];
    if (tid < vec_cnt-1) while(b < vec_cnt && out_ptr == output_perm[(interac_indx+b)*4 + 3]) b++;
  }

  if (tid < vec_cnt) {
    /* PRAM Model: assuming as many threads as we need. */
    for(int i = a; i < b; i++) { // Compute permutations.
      double scaling_factor = scaling[interac_indx + i];
      int *perm = (int*) (precomp_data[0] + output_perm[(interac_indx + i)*4 + 0]);
      double *scal = (double*) (precomp_data[0] + output_perm[(interac_indx + i)*4 + 1]);
      double *v_in = (double*) (buff_out + output_perm[(interac_indx + i)*4 + 3]);
      double *v_out = (double*) (output_data[0] + output_perm[(interac_indx + i)*4 + 2]);
      for (int j = 0; j < M_dim1; j++) v_out[j] += v_in[perm[j]]*scal[j]*scaling_factor;
    }
  }
}

void in_perm_d (
  uintptr_t *precomp_data,
  uintptr_t *input_perm,
  uintptr_t *input_data,
  uintptr_t *buff_in,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt,
  cudaStream_t *stream )
{
  int n_thread, n_block;
  n_thread = DEFAULT_NUM_THREAD;
  n_block = vec_cnt/n_thread;
  in_perm_k<<<n_thread, n_block, 0, *stream>>>(precomp_data, input_perm, input_data, buff_in, interac_indx, M_dim0, vec_cnt);
}

void out_perm_d (
  uintptr_t *scaling,
  uintptr_t *precomp_data,
  uintptr_t *output_perm,
  uintptr_t *output_data,
  uintptr_t *buff_out,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt,
  cudaStream_t *stream )
{
  int n_thread, n_block;
  n_thread = DEFAULT_NUM_THREAD;
  n_block = vec_cnt/n_thread;
  out_perm_k<<<n_thread, n_block, 0, *stream>>>(scaling, precomp_data, output_perm, output_data, buff_out, interac_indx, M_dim0, vec_cnt);
}
