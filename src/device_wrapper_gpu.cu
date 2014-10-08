#include <stdio.h>
#include <assert.h>

extern "C" {

void* host_malloc_cuda(size_t size){
  void* p;
  cudaError_t error = cudaHostAlloc(&p, size, cudaHostAllocPortable);
  if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
  assert(error == cudaSuccess);
  return p;
}

void host_free_cuda(void* p){
  cudaError_t error = cudaFreeHost(p);
  if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
  assert(error == cudaSuccess);
}

}
