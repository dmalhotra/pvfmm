#ifndef _CUDA_FUNC_HPP_
#define _CUDA_FUNC_HPP_

#include <pvfmm_common.hpp>
#include <assert.h>
#include <cstring>
#include <device_wrapper.hpp>
#include <matrix.hpp>
#include <vector.hpp>

namespace pvfmm {

// external functions
void in_perm_d (uintptr_t, uintptr_t, uintptr_t, uintptr_t, size_t, size_t, size_t, cudaStream_t*);
void out_perm_d (uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, size_t, size_t, size_t, cudaStream_t*);

template <class Real_t>
class cuda_func {
  public:
    static void in_perm_h (uintptr_t precomp_data, uintptr_t input_perm,
        uintptr_t input_data, uintptr_t buff_in, size_t interac_indx,
        size_t M_dim0, size_t vec_cnt);
    static void out_perm_h (uintptr_t scaling, uintptr_t precomp_data, uintptr_t output_perm,
        uintptr_t output_data, uintptr_t buff_out, size_t interac_indx,
        size_t M_dim0, size_t vec_cnt);
};

template <class Real_t>
void cuda_func<Real_t>::in_perm_h (
  uintptr_t precomp_data,
  uintptr_t input_perm,
  uintptr_t input_data,
  uintptr_t buff_in,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt )
{
  cudaStream_t *stream;
  //stream = DeviceWrapper::CUDA_Lock::acquire_stream(0);
  stream = CUDA_Lock::acquire_stream(0);
  /*
  intptr_t precomp_data_d = precomp_data[0];
  intptr_t input_perm_d = input_perm[0];
  intptr_t input_data_d = input_data[0];
  intptr_t buff_in_d = buff_in[0];
  */
  in_perm_d(precomp_data, input_perm, input_data, buff_in, interac_indx, M_dim0, vec_cnt, stream);
};

template <class Real_t>
void cuda_func<Real_t>::out_perm_h (
  uintptr_t scaling,
  uintptr_t precomp_data,
  uintptr_t output_perm,
  uintptr_t output_data,
  uintptr_t buff_out,
  size_t interac_indx,
  size_t M_dim1,
  size_t vec_cnt )
{
  cudaStream_t *stream;
  //stream = DeviceWrapper::CUDA_Lock::acquire_stream(0);
  stream = CUDA_Lock::acquire_stream(0);
  out_perm_d(scaling, precomp_data, output_perm, output_data, buff_out, interac_indx, M_dim1, vec_cnt, stream);
}

};

#endif //_CUDA_FUNC_HPP_
