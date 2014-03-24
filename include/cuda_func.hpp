#ifndef _CUDA_FUNC_HPP_
#define _CUDA_FUNC_HPP_

#include <pvfmm_common.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cstring>
#include <device_wrapper.hpp>
#include <matrix.hpp>
#include <vector.hpp>

#ifdef __cplusplus
extern "C" {
#endif
  void texture_bind_d (char*, size_t);
  void texture_unbind_d ();
  void in_perm_d (char*, size_t*, char*, char*, size_t, size_t, size_t, cudaStream_t*);
  void out_perm_d (double*, char*, size_t*, char*, char*, size_t, size_t, size_t, cudaStream_t*, size_t*, size_t*, size_t);
#ifdef __cplusplus
}
#endif

template <class Real_t>
class cuda_func {
  public:
    static void texture_bind_h (char *ptr_d, size_t len);
    static void texture_unbind_h ();
    static void in_perm_h (char *precomp_data, char *input_perm, char *input_data, char *buff_in, 
        size_t interac_indx, size_t M_dim0, size_t vec_cnt);
    static void out_perm_h (char *scaling, char *precomp_data, char *output_perm, char *output_data, char *buff_out, 
        size_t interac_indx, size_t M_dim0, size_t vec_cnt, size_t *tmp_a, size_t *tmp_b, size_t counter);
    static void compare_h (Real_t *gold, Real_t *mine, size_t n);
};

template <class Real_t>
void cuda_func<Real_t>::texture_bind_h (char *ptr_d, size_t len) { texture_bind_d(ptr_d, len); };

template <class Real_t>
void cuda_func<Real_t>::texture_unbind_h () { texture_unbind_d(); };

template <class Real_t>
void cuda_func<Real_t>::in_perm_h (
  char *precomp_data,
  char *input_perm,
  char *input_data,
  char *buff_in,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt )
{
  cudaStream_t *stream;
  stream = pvfmm::CUDA_Lock::acquire_stream(0);
  in_perm_d(precomp_data, (size_t *) input_perm, input_data, buff_in, 
	  interac_indx, M_dim0, vec_cnt, stream);
};

template <class Real_t>
void cuda_func<Real_t>::out_perm_h (
  char *scaling,
  char *precomp_data,
  char *output_perm,
  char *output_data,
  char *buff_out,
  size_t interac_indx,
  size_t M_dim1,
  size_t vec_cnt,
  size_t *tmp_a,
  size_t *tmp_b,
  size_t counter )
{
  cudaStream_t *stream;
  stream = pvfmm::CUDA_Lock::acquire_stream(0);
  size_t *a_d, *b_d;
  cudaMalloc((void**)&a_d, sizeof(size_t)*counter);
  cudaMalloc((void**)&b_d, sizeof(size_t)*counter);
  cudaMemcpy(a_d, tmp_a, sizeof(size_t)*counter, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, tmp_b, sizeof(size_t)*counter, cudaMemcpyHostToDevice);
  out_perm_d((double *) scaling, precomp_data, (size_t *) output_perm, output_data, buff_out, 
	  interac_indx, M_dim1, vec_cnt, stream, a_d, b_d, counter);
  cudaFree(a_d);
  cudaFree(b_d);
};

template <class Real_t>
void cuda_func<Real_t>::compare_h (
  Real_t *gold,
  Real_t *mine, 
  size_t n )
{
  cudaError_t error;
  Real_t *mine_h = (Real_t *) malloc(n*sizeof(Real_t));
  error =  cudaMemcpy(mine_h, mine, n*sizeof(Real_t), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) std::cout << "compare_h(): " << cudaGetErrorString(error) << '\n';
  if (n)
  std::cout << "compare_h(): " << n << '\n';
  for (int i = 0; i < n; i++) { 
    if (gold[i] != mine_h[i]) {
      std::cout << "compare_h(): " << i << ", gold[i]: " << gold[i] << ", mine[i]: " << mine_h[i] << '\n';
      //error =  cudaMemcpy(mine, gold, n*sizeof(Real_t), cudaMemcpyHostToDevice);
	    break;
	  } 
  }
  free(mine_h);
};

#endif //_CUDA_FUNC_HPP_
