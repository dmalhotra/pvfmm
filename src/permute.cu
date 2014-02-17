#include <vector.hpp>
#include <device_wrapper.hpp>

template <class Real_t>
__global__ void in_perm_d (
  Vector *precomp_data,
  Vector *input_perm,
  :Vector *input_data,
  Vector *buff_in,
  size_t interac_indx,
  size_t M_dim0 )
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int *perm = (int*)(precomp_data[0] + input_perm[(interac_indx+i)*4 + 0]);
  const Real_t* scal = (Real_t*)(precomp_data[0] + input_perm[(interac_indx + i)*4 + 1]);
  const Real_t* v_in = (Real_t*)(input_data[0] + input_perm[(interac_indx + i)*4 + 3]);
  Real_t* v_out = (Real_t*)(buff_in + input_perm[(interac_indx + i)*4 + 2]);

  for (size_t j = 0; j < M_dim0; j++) v_out[j] = v_in[perm[j]]*scal[j];
}

template <class Real_t>
__global__ void out_perm_d (
  Vector *precomp_data,
  Vector *input_perm,
  Vector *input_data,
  Vector *buff_in,
  size_t interac_indx,
  size_t M_dim0 )
{


}

template <class Real_t>
void in_perm_h (
  Vector *precomp_data,
  Vector *input_perm,
  Vector *input_data,
  Vector *buff_in,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt )
{
  cudaStream_t *stream;
  int n_thread, n_block;
  n_thread = DEFAULT_NUM_THREAD;
  n_block = vec_cnt/n_thread;
  stream = DeviceWrapper::CUDA_Lock::acquire_stream(0);
  in_perm_d<Real_t><<<n_thread, b_block, 0, *stream>>>
	(precomp_data, input_perm, input_data, buff_in, interac_indx, M_dim0);
}

template <class Real_t>
void out_perm_h (
  Vector *precomp_data,
  Vector *input_perm,
  Vector *input_data,
  Vector *buff_in,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt )
{
  cudaStream_t *stream;
  int n_thread, n_block;
  n_thread = DEFAULT_NUM_THREAD;
  n_block = vec_cnt/n_thread;
  stream = DeviceWrapper::CUDA_Lock::acquire_stream(0);
  out_perm_d<Real_t><<<n_thread, b_block, 0, *stream>>>
	(precomp_data, input_perm, input_data, buff_in, interac_indx, M_dim0);
}

