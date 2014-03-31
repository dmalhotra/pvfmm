#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define DEFAULT_NUM_THREAD 128

texture<int, 1, cudaReadModeElementType> tex_in_perm;
texture<double, 1, cudaReadModeElementType> tex_out_perm;


__global__ void in_perm_2d_k (
  char *precomp_data,
  size_t *input_perm,
  char *input_data,
  char *buff_in,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt )
{
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  if (tidy< vec_cnt && tidx < M_dim0) {
    size_t k = (interac_indx + tidy)*4;
    size_t *perm  = (size_t *) (precomp_data + input_perm[k + 0]);
    double *scal  = (double *) (precomp_data + input_perm[k + 1]);
    double *v_in  = (double *) (input_data ) + input_perm[k + 3];
    double *v_out = (double *) (buff_in      + input_perm[k + 2]);
    v_out[tidx] = v_in[perm[tidx]]*scal[tidx];
  }
}

/* Case: double */
__global__ void in_perm_k (
  char *precomp_data,
  size_t *input_perm,
  char *input_data,
  char *buff_in,
  size_t interac_indx,
  size_t M_dim0,
  size_t vec_cnt )
{
  extern __shared__ double s[];
  //__shared__ double s[680];

  /* Specifing range. */
  int a = ( blockIdx.x     *vec_cnt)/gridDim.x;
  int b = ((blockIdx.x + 1)*vec_cnt)/gridDim.x;

  for(int i = a; i < b; i++) { // Compute permutations.
    size_t *perm  = (size_t *) (precomp_data + input_perm[(interac_indx + i)*4 + 0]);
    double *scal  = (double *) (precomp_data + input_perm[(interac_indx + i)*4 + 1]);
    double *v_in  = (double *) (input_data ) + input_perm[(interac_indx + i)*4 + 3];
    double *v_out = (double *) (buff_in      + input_perm[(interac_indx + i)*4 + 2]);
    for (size_t j = threadIdx.x; j < M_dim0; j+=blockDim.x) s[j] = v_in[j];
    __syncthreads();
    for (size_t j = threadIdx.x; j < M_dim0; j+=blockDim.x) v_out[j] = s[perm[j]]*scal[j];
    __syncthreads();
  }
};

__global__ void out_perm_2d_k (
  double *scaling,
  char *precomp_data,
  size_t *output_perm,
  char *output_data,
  char *buff_out,
  size_t interac_indx,
  size_t M_dim1,
  size_t vec_cnt,
  size_t *a_d,
  size_t *b_d,
  size_t counter )
{
  /* 2-dim thread Id. */
  int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  int tidy = blockIdx.y*blockDim.y + threadIdx.y;

  int a = a_d[tidy];
  int b = b_d[tidy];
  /*
  int a = (tidy*vec_cnt)/vec_cnt;
  int b = ((tidy + 1)*vec_cnt)/vec_cnt;

  if (tidy > 0 && a < vec_cnt) {
    size_t out_ptr = output_perm[(interac_indx + a)*4 + 3];
    if (tidy > 0) while (a < vec_cnt && out_ptr == output_perm[(interac_indx + a)*4 + 3]) a++;
  }
  if (tidy < vec_cnt - 1 &&  - 1 && b < vec_cnt) {
    size_t out_ptr = output_perm[(interac_indx + b)*4 + 3];
    if (tidy < vec_cnt - 1) while (b < vec_cnt && out_ptr == output_perm[(interac_indx + b)*4 + 3]) b++;
  }
*/

  //if (tidy < vec_cnt && tidx < M_dim1) {
  if (tidy < counter && tidx < M_dim1) {
    double v = 0.0;
    
    for(int i = a; i < b; i++) { // Compute permutations.
      double scaling_factor = scaling[interac_indx + i];
      size_t k = (interac_indx + i)*4;
      size_t *perm = (size_t*) (precomp_data + output_perm[k + 0]);
      double *scal = (double*) (precomp_data + output_perm[k + 1]);
      double *v_in = (double*) (buff_out     + output_perm[k + 2]);
      //double *v_out = (double*) (output_data)+ output_perm[k + 3];
      //v_out[tidx] += v_in[perm[tidx]]*scal[tidx]*scaling_factor;
      v += v_in[perm[tidx]]*scal[tidx]*scaling_factor;
    }
    double *v_out = (double*) (output_data)+ output_perm[(interac_indx + a)*4 + 3];
    v_out[tidx] += v;

    
    /*
    double *v_out = (double*) (output_data)+ output_perm[(interac_indx + a)*4 + 3];
    for(int i = a; i < b; i++) { // Compute permutations.
      double scaling_factor = scaling[interac_indx + i];
      size_t *perm = (size_t*) (precomp_data + output_perm[(interac_indx + i)*4 + 0]);
      double *scal = (double*) (precomp_data + output_perm[(interac_indx + i)*4 + 1]);
      double *v_in = (double*) (buff_out     + output_perm[(interac_indx + i)*4 + 2]);
      double *v_out = (double*) (output_data)+ output_perm[(interac_indx + a)*4 + 3];
      v += v_in[perm[tidx]]*scal[tidx]*scaling_factor;
    }
    if (a < b) v_out[tidx] = v;
    */
  }
};

__global__ void out_perm_k (
  double *scaling,
  char *precomp_data,
  size_t *output_perm,
  char *output_data,
  char *buff_out,
  size_t interac_indx,
  size_t M_dim1,
  size_t vec_cnt )
{
  extern __shared__ double s[];
  //__shared__ double s[680];
  for (size_t j = threadIdx.x; j < M_dim1; j+=blockDim.x) s[j] = 0;

  /* Specifing range. */
  int a = ( blockIdx.x     *vec_cnt)/gridDim.x;
  int b = ((blockIdx.x + 1)*vec_cnt)/gridDim.x;

  if (blockIdx.x > 0             && a < vec_cnt) { // Find 'a' independent of other threads.
    size_t out_ptr = output_perm[(interac_indx + a)*4 + 3];
    if (blockIdx.x >             0) while (a < vec_cnt && out_ptr == output_perm[(interac_indx + a)*4 + 3]) a++;
  }
  if (blockIdx.x < gridDim.x - 1 && b < vec_cnt) { // Find 'b' independent of other threads.
    size_t out_ptr = output_perm[(interac_indx + b)*4 + 3];
    if (blockIdx.x < gridDim.x - 1) while (b < vec_cnt && out_ptr == output_perm[(interac_indx + b)*4 + 3]) b++;
  }

  for(int i = a; i < b; i++) { // Compute permutations.
    double scaling_factor = scaling[interac_indx + i];
    size_t *perm = (size_t*) (precomp_data + output_perm[(interac_indx + i)*4 + 0]);
    double *scal = (double*) (precomp_data + output_perm[(interac_indx + i)*4 + 1]);
    double *v_in = (double*) (buff_out     + output_perm[(interac_indx + i)*4 + 2]);
    double *v_out = (double*) (output_data)+ output_perm[(interac_indx + i)*4 + 3];
    for (size_t j = threadIdx.x; j < M_dim1; j+=blockDim.x) s[j] += v_in[perm[j]]*scal[j]*scaling_factor;
    if(output_perm[(interac_indx + i)*4 + 3]!=output_perm[(interac_indx + i+1)*4 + 3]){
      for (size_t j = threadIdx.x; j < M_dim1; j+=blockDim.x){
        v_out[j]+=s[j];
        s[j] = 0;
      }
    }
  }
};

extern "C" { 
void texture_bind_d (char *input_perm, size_t len) {
  cudaBindTexture(0, tex_in_perm, input_perm, len);
};

void texture_unbind_d () {
  cudaUnbindTexture(tex_in_perm);
};

void in_perm_d (
    char *precomp_data,
    size_t *input_perm,
    char *input_data,
    char *buff_in,
    size_t interac_indx,
    size_t M_dim0,
    size_t vec_cnt, 
    cudaStream_t *stream )
{
  if (vec_cnt == 0) return;

  float time_ms;
  int n_thread, n_block;
  n_thread = DEFAULT_NUM_THREAD;
  n_block = vec_cnt/n_thread + 1;
/*
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  */
  //cudaBindTexture(0, tex_in_perm, input_perm, );
  /*
  printf("in_perm_k : vec_cnt: %d, M_dim0: %d, n_block: %d, n_thread: %d\n", 
      (int) vec_cnt, (int) M_dim0, n_block, n_thread);
  */
  //cudaEventRecord(beg, 0);
  /*
  in_perm_k<<<1024, 256, M_dim0*sizeof(double)>>>(precomp_data, input_perm, input_data, buff_in, 
      interac_indx, M_dim0, vec_cnt);
      */
  in_perm_k<<<1024, 256, M_dim0*sizeof(double), *stream>>>(precomp_data, input_perm, input_data, buff_in, 
      interac_indx, M_dim0, vec_cnt);
//  dim3 dimBlock(16, 32);
//  dim3 dimGrid(M_dim0/16 + 1, vec_cnt/32 + 1);
//  in_perm_2d_k<<<dimGrid, dimBlock>>>(precomp_data, input_perm, input_data, buff_in, 
//      interac_indx, M_dim0, vec_cnt);
//      
      /*
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_ms, beg, end);
  printf("in_perm_d : %f ms\n", time_ms);	
    
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  */
};

void out_perm_d (
  double *scaling,
  char *precomp_data,
  size_t *output_perm,
  char *output_data,
  char *buff_out,
  size_t interac_indx,
  size_t M_dim1,
  size_t vec_cnt,
  cudaStream_t *stream,
  size_t *a_d,
  size_t *b_d,
  size_t counter )
{
  if (vec_cnt == 0) return;

  float time_ms;
  int n_thread, n_block;
  n_thread = DEFAULT_NUM_THREAD;
  //n_block = vec_cnt/n_thread + 1;
  n_block = counter/n_thread + 1;

//  cudaEvent_t beg, end;
//  cudaEventCreate(&beg);
//  cudaEventCreate(&end);
  /*
  printf("out_perm_k : vec_cnt: %d, M_dim0: %d, n_block: %d, n_thread: %d\n", 
      (int) vec_cnt, (int) M_dim1, n_block, n_thread);
      */
//  cudaEventRecord(beg, 0);
/*
  out_perm_k<<<1024, 256, M_dim1*sizeof(double)>>>(scaling, precomp_data, output_perm, output_data, buff_out, 
    interac_indx, M_dim1, vec_cnt);
    */
  out_perm_k<<<1024, 256, M_dim1*sizeof(double), *stream>>>(scaling, precomp_data, output_perm, output_data, buff_out, 
    interac_indx, M_dim1, vec_cnt);
  
//  dim3 dimBlock(16, 32);
//  dim3 dimGrid(M_dim1/8 + 1, vec_cnt/64 + 1);
//  out_perm_2d_k<<<dimGrid, dimBlock>>>(scaling, precomp_data, output_perm, output_data, buff_out, 
//    interac_indx, M_dim1, vec_cnt, a_d, b_d, counter);

  /*
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time_ms, beg, end);
  printf("out_perm_d : %f ms\n", time_ms);	
    
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  */
};

}
