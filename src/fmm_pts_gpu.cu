#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cassert>
#include <fmm_pts_gpu.hpp>

template <class Real_t>
__global__ void  in_perm_k(char* precomp_data, Real_t*  input_data, char* buff_in , size_t*  input_perm, size_t vec_cnt, size_t M_dim0){
  extern __shared__ char s_[]; Real_t* s=(Real_t*)&s_[0];

  /* Specifing range. */
  int a = ( blockIdx.x     *vec_cnt)/gridDim.x;
  int b = ((blockIdx.x + 1)*vec_cnt)/gridDim.x;

  for(int i = a; i < b; i++) { // Compute permutations.
    const size_t* perm= (size_t*) (precomp_data + input_perm[i*4+0]);
    const Real_t* scal= (Real_t*) (precomp_data + input_perm[i*4+1]);
    const Real_t*v_in = (Real_t*) (input_data   + input_perm[i*4+3]);
    Real_t*      v_out= (Real_t*) (buff_in      + input_perm[i*4+2]);
    for (size_t j = threadIdx.x; j < M_dim0; j+=blockDim.x) s[j] = v_in[j];
    __syncthreads();
    for (size_t j = threadIdx.x; j < M_dim0; j+=blockDim.x) v_out[j] = s[perm[j]]*scal[j];
    __syncthreads();
  }
};

template <class Real_t>
__global__ void out_perm_k(char* precomp_data, Real_t* output_data, char* buff_out, size_t* output_perm, size_t vec_cnt, size_t M_dim1){
  extern __shared__ char s_[]; Real_t* s=(Real_t*)&s_[0];
  for (size_t j = threadIdx.x; j < M_dim1; j+=blockDim.x) s[j] = 0;

  /* Specifing range. */
  int a = ( blockIdx.x     *vec_cnt)/gridDim.x;
  int b = ((blockIdx.x + 1)*vec_cnt)/gridDim.x;

  if (blockIdx.x > 0             && a < vec_cnt) { // Find 'a' independent of other threads.
    size_t out_ptr = output_perm[a*4+3];
    if (blockIdx.x >             0) while (a < vec_cnt && out_ptr == output_perm[a*4+3]) a++;
  }
  if (blockIdx.x < gridDim.x - 1 && b < vec_cnt) { // Find 'b' independent of other threads.
    size_t out_ptr = output_perm[b*4+3];
    if (blockIdx.x < gridDim.x - 1) while (b < vec_cnt && out_ptr == output_perm[b*4+3]) b++;
  }

  for(int i = a; i < b; i++) { // Compute permutations.
    size_t  *perm = (size_t*) (precomp_data + output_perm[i*4+0]);
    Real_t  *scal = (Real_t*) (precomp_data + output_perm[i*4+1]);
    Real_t *v_in  = (Real_t*) (buff_out     + output_perm[i*4+2]);
    Real_t *v_out = (Real_t*) (output_data  + output_perm[i*4+3]);
    for(size_t j = threadIdx.x; j<M_dim1; j+=blockDim.x){
      s[j] += v_in[perm[j]]*scal[j];
    }
    if(output_perm[i*4+3]!=output_perm[(i+1)*4+3])
    for(size_t j = threadIdx.x; j<M_dim1; j+=blockDim.x){
      v_out[j]+=s[j];
      s[j] = 0;
    }
  }
};

template <class Real_t>
void  in_perm_gpu_(char* precomp_data, Real_t*  input_data, char* buff_in , size_t*  input_perm, size_t vec_cnt, size_t M_dim0, cudaStream_t *stream){
  if (vec_cnt == 0) return;
  in_perm_k <Real_t><<<1024, 256, M_dim0*sizeof(Real_t), *stream>>>(precomp_data,  input_data, buff_in ,  input_perm, vec_cnt, M_dim0);
  cudaError_t error = cudaGetLastError();
  assert(error == cudaSuccess);
};

template <class Real_t>
void out_perm_gpu_(char* precomp_data, Real_t* output_data, char* buff_out, size_t* output_perm, size_t vec_cnt, size_t M_dim1, cudaStream_t *stream){
  if (vec_cnt == 0) return;
  out_perm_k<Real_t><<<1024, 256, M_dim1*sizeof(Real_t), *stream>>>(precomp_data, output_data, buff_out, output_perm, vec_cnt, M_dim1);
  cudaError_t error = cudaGetLastError();
  assert(error == cudaSuccess);
};

extern "C" {
  void  in_perm_gpu_f(char* precomp_data,  float*  input_data, char* buff_in , size_t*  input_perm, size_t vec_cnt, size_t M_dim0, cudaStream_t *stream){
    in_perm_gpu_(precomp_data,input_data,buff_in,input_perm,vec_cnt,M_dim0,stream);
  }
  void  in_perm_gpu_d(char* precomp_data, double*  input_data, char* buff_in , size_t*  input_perm, size_t vec_cnt, size_t M_dim0, cudaStream_t *stream){
    in_perm_gpu_(precomp_data,input_data,buff_in,input_perm,vec_cnt,M_dim0,stream);
  }

  void out_perm_gpu_f(char* precomp_data,  float* output_data, char* buff_out, size_t* output_perm, size_t vec_cnt, size_t M_dim1, cudaStream_t *stream){
    out_perm_gpu_(precomp_data,output_data,buff_out,output_perm,vec_cnt,M_dim1,stream);
  }
  void out_perm_gpu_d(char* precomp_data, double* output_data, char* buff_out, size_t* output_perm, size_t vec_cnt, size_t M_dim1, cudaStream_t *stream){
    out_perm_gpu_(precomp_data,output_data,buff_out,output_perm,vec_cnt,M_dim1,stream);
  }
}

