#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cassert>

__global__
void  in_perm_k(char* precomp_data, double*  input_data, char* buff_in , size_t*  input_perm, size_t vec_cnt, size_t M_dim0){
  extern __shared__ double s[];

  /* Specifing range. */
  int a = ( blockIdx.x     *vec_cnt)/gridDim.x;
  int b = ((blockIdx.x + 1)*vec_cnt)/gridDim.x;

  for(int i = a; i < b; i++) { // Compute permutations.
    const size_t* perm= (size_t*) (precomp_data + input_perm[i*4+0]);
    const double* scal= (double*) (precomp_data + input_perm[i*4+1]);
    const double*v_in = (double*) (input_data   + input_perm[i*4+3]);
    double*      v_out= (double*) (buff_in      + input_perm[i*4+2]);
    for (size_t j = threadIdx.x; j < M_dim0; j+=blockDim.x) s[j] = v_in[j];
    __syncthreads();
    for (size_t j = threadIdx.x; j < M_dim0; j+=blockDim.x) v_out[j] = s[perm[j]]*scal[j];
    __syncthreads();
  }
};

__global__
void out_perm_k(char* precomp_data, double* output_data, char* buff_out, size_t* output_perm, size_t vec_cnt, size_t M_dim1){
  extern __shared__ double s[];
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
    double  *scal = (double*) (precomp_data + output_perm[i*4+1]);
    double *v_in  = (double*) (buff_out     + output_perm[i*4+2]);
    double *v_out = (double*) (output_data  + output_perm[i*4+3]);
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

extern "C" {

  void  in_perm_d(char* precomp_data, double*  input_data, char* buff_in , size_t*  input_perm, size_t vec_cnt, size_t M_dim0, cudaStream_t *stream){
    if (vec_cnt == 0) return;
    in_perm_k <<<1024, 256, M_dim0*sizeof(double), *stream>>>(precomp_data,  input_data, buff_in ,  input_perm, vec_cnt, M_dim0);
    cudaError_t error = cudaGetLastError();
    assert(error == cudaSuccess);
  };

  void out_perm_d(char* precomp_data, double* output_data, char* buff_out, size_t* output_perm, size_t vec_cnt, size_t M_dim1, cudaStream_t *stream){
    if (vec_cnt == 0) return;
    out_perm_k<<<1024, 256, M_dim1*sizeof(double), *stream>>>(precomp_data, output_data, buff_out, output_perm, vec_cnt, M_dim1);
    cudaError_t error = cudaGetLastError();
    assert(error == cudaSuccess);
  };

}
