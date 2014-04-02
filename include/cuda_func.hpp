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
  size_t *a_d, *b_d;
  out_perm_d((double *) scaling, precomp_data, (size_t *) output_perm, output_data, buff_out,
      interac_indx, M_dim1, vec_cnt, stream );
};

#endif //_CUDA_FUNC_HPP_
