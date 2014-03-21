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
  void test_d(uintptr_t, uintptr_t, uintptr_t, uintptr_t, int, cudaStream_t*);
  void in_perm_d (char*, size_t*, char*, char*, size_t, size_t, size_t, cudaStream_t*);
  void out_perm_d (double*, char*, size_t*, char*, char*, size_t, size_t, size_t, cudaStream_t*);
#ifdef __cplusplus
}
#endif

template <class Real_t>
class cuda_func {
  public:
    static void in_perm_h (char *precomp_data, char *input_perm, char *input_data, char *buff_in, 
        size_t interac_indx, size_t M_dim0, size_t vec_cnt);
    static void out_perm_h (char *scaling, char *precomp_data, char *output_perm, char *output_data, char *buff_out, 
        size_t interac_indx, size_t M_dim0, size_t vec_cnt);
    static void compare_h (Real_t *gold, Real_t *mine, size_t n);
};

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
  size_t vec_cnt )
{
  cudaStream_t *stream;
  stream = pvfmm::CUDA_Lock::acquire_stream(0);
  out_perm_d((double *) scaling, precomp_data, (size_t *) output_perm, output_data, buff_out, 
	  interac_indx, M_dim1, vec_cnt, stream);
}

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
}

#endif //_CUDA_FUNC_HPP_
