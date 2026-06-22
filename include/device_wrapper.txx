/**
 * \file device_wrapper.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 6-5-2013
 * \brief This file contains implementation of DeviceWrapper.
 *
 * Modified:
 *   editor Chenhan D. Yu
 *   date Juan-28-2014
 *   Add Cuda support. Error handle is available if needed.
 */

#include <omp.h>
#include <cassert>
#include <cstdlib>

// CUDA Stream
#if defined(PVFMM_HAVE_CUDA)
#endif

namespace pvfmm{

namespace DeviceWrapper{

  // CUDA functions
  inline void* host_malloc_cuda(size_t size){
    if(!size) return NULL;
    return malloc(size);
    //void* p;
    //cudaError_t error = cudaHostAlloc(&p, size, cudaHostAllocPortable);
    //if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
    //assert(error == cudaSuccess);
    //return p;
  }

  inline void host_free_cuda(void* p){
    free(p);
    //cudaError_t error = cudaFreeHost(p);
    //if (error != cudaSuccess) fprintf(stderr,"CUDA Error: %s \n", cudaGetErrorString(error));
    //assert(error == cudaSuccess);
  }

  inline uintptr_t alloc_device_cuda(char* dev_handle, size_t len) {
    char *dev_ptr=NULL;
#if defined(PVFMM_HAVE_CUDA)
    cudaError_t error;
    error = cudaHostRegister(dev_handle, len, cudaHostRegisterPortable);
    if (error != cudaSuccess)
      std::cout<<cudaGetErrorString(error)<< '\n';
    assert(error == cudaSuccess);
    if(len) error = cudaMalloc((void**)&dev_ptr, len);
    if (error != cudaSuccess)
      std::cout<<cudaGetErrorString(error)<< '\n';
    assert(error == cudaSuccess);
#else
    PVFMM_UNUSED(dev_handle);
    PVFMM_UNUSED(len);
#endif
    return (uintptr_t)dev_ptr;
  }

  inline void free_device_cuda(char* dev_handle, uintptr_t dev_ptr) {
#if defined(PVFMM_HAVE_CUDA)
    if(dev_handle==NULL || dev_ptr==0) return;
    cudaError_t error;
    error = cudaHostUnregister(dev_handle);
    if (error != cudaSuccess)
      std::cout<<cudaGetErrorString(error)<< '\n';
    assert(error == cudaSuccess);
    error = cudaFree((char*)dev_ptr);
    assert(error == cudaSuccess);
#else
    PVFMM_UNUSED(dev_handle);
    PVFMM_UNUSED(dev_ptr);
#endif
  }

  inline int host2device_cuda(char *host_ptr, char *dev_ptr, size_t len) {
    #if defined(PVFMM_HAVE_CUDA)
    cudaError_t error;
    cudaStream_t *stream = CUDA_Lock::acquire_stream();
    error = cudaMemcpyAsync(dev_ptr, host_ptr, len, cudaMemcpyHostToDevice, *stream);
    if (error != cudaSuccess) std::cout<<cudaGetErrorString(error)<< '\n';
    assert(error == cudaSuccess);
    #else
    PVFMM_UNUSED(host_ptr);
    PVFMM_UNUSED(dev_ptr);
    PVFMM_UNUSED(len);
    #endif
    return 0;
  }

  inline int device2host_cuda(char *dev_ptr, char *host_ptr, size_t len) {
    if(!dev_ptr) return 0;
    #if defined(PVFMM_HAVE_CUDA)
    cudaError_t error;
    cudaStream_t *stream = CUDA_Lock::acquire_stream();
    error = cudaMemcpyAsync(host_ptr, dev_ptr, len, cudaMemcpyDeviceToHost, *stream);
    if (error != cudaSuccess)
      std::cout<<cudaGetErrorString(error)<< '\n';
    assert(error == cudaSuccess);
    #else
    PVFMM_UNUSED(host_ptr);
    PVFMM_UNUSED(len);
    #endif
    return 0;
  }


  // Wrapper functions

  inline void* host_malloc(size_t size){
    if(!size) return NULL;
    #if defined(PVFMM_HAVE_CUDA)
    return host_malloc_cuda(size);
    #else
    return malloc(size);
    #endif
  }

  inline void host_free(void* p){
    #if defined(PVFMM_HAVE_CUDA)
    return host_free_cuda(p);
    #else
    return free(p);
    #endif
  }

  inline uintptr_t alloc_device(char* dev_handle, size_t len){
    #if defined(PVFMM_HAVE_CUDA)
    return alloc_device_cuda(dev_handle,len);
    #else
    PVFMM_UNUSED(len);
    return (uintptr_t)dev_handle;
    #endif
  }

  inline void free_device(char* dev_handle, uintptr_t dev_ptr){
    #if defined(PVFMM_HAVE_CUDA)
    free_device_cuda(dev_handle,dev_ptr);
    #else
    PVFMM_UNUSED(dev_handle);
    PVFMM_UNUSED(dev_ptr);
    #endif
  }

  template <int SYNC>
  inline int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len){
    int lock_idx=-1;
    PVFMM_UNUSED(dev_handle);
    #if defined(PVFMM_HAVE_CUDA)
    lock_idx=host2device_cuda(host_ptr,(char*)dev_ptr,len);
    #else
    PVFMM_UNUSED(host_ptr);
    PVFMM_UNUSED(dev_ptr);
    PVFMM_UNUSED(len);
    #endif
    return lock_idx;
  }

  template <int SYNC>
  inline int device2host(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len){
    int lock_idx=-1;
    PVFMM_UNUSED(dev_handle);
    #if defined(PVFMM_HAVE_CUDA)
    lock_idx=device2host_cuda((char*)dev_ptr, host_ptr, len);
    #else
    PVFMM_UNUSED(host_ptr);
    PVFMM_UNUSED(dev_ptr);
    PVFMM_UNUSED(len);
    #endif
    return lock_idx;
  }

  inline void wait(int lock_idx){
    PVFMM_UNUSED(lock_idx);
    #if defined(PVFMM_HAVE_CUDA)
    CUDA_Lock::wait();
    #endif
  }

}


  // Implementation of DeviceMirror

  template <class T>
  inline DeviceVector<T> DeviceMirror::AllocDevice(sctl::Vector<T>& host, bool copy){
    char* p=(char*)(host.Dim()?&host[0]:nullptr);
    size_t bytes=host.Dim()*sizeof(T);
    if(dev_ptr){ // Already bound: host buffer must not have changed.
      assert(host_ptr==p && len==bytes);
    }else if(bytes){
      host_ptr=p;
      len=bytes;
      dev_ptr=DeviceWrapper::alloc_device(host_ptr,len);
    }
    if(dev_ptr && copy) lock_idx=DeviceWrapper::host2device(host_ptr,host_ptr,dev_ptr,len);

    DeviceVector<T> h;
    h.dim=host.Dim();
    h.dev_ptr=dev_ptr;
    return h;
  }

  template <class T>
  inline DeviceMatrix<T> DeviceMirror::AllocDevice(sctl::Matrix<T>& host, bool copy){
    char* p=(char*)(host.Dim(0)*host.Dim(1)>0?&host[0][0]:nullptr);
    size_t bytes=host.Dim(0)*host.Dim(1)*sizeof(T);
    if(dev_ptr){ // Already bound: host buffer must not have changed.
      assert(host_ptr==p && len==bytes);
    }else if(bytes){
      host_ptr=p;
      len=bytes;
      dev_ptr=DeviceWrapper::alloc_device(host_ptr,len);
    }
    if(dev_ptr && copy) lock_idx=DeviceWrapper::host2device(host_ptr,host_ptr,dev_ptr,len);

    DeviceMatrix<T> h;
    h.dim[0]=host.Dim(0);
    h.dim[1]=host.Dim(1);
    h.dev_ptr=dev_ptr;
    h.lock_idx=lock_idx;
    return h;
  }

  inline void DeviceMirror::Device2Host(char* dst){
    if(!dev_ptr) return;
    lock_idx=DeviceWrapper::device2host(host_ptr,dev_ptr,(dst?dst:host_ptr),len);
  }

  inline void DeviceMirror::Device2HostWait(){
    DeviceWrapper::wait(lock_idx);
    lock_idx=-1;
  }

  inline void DeviceMirror::Free(){
    if(dev_ptr) DeviceWrapper::free_device(host_ptr,dev_ptr);
    host_ptr=NULL;
    len=0;
    dev_ptr=0;
    lock_idx=-1;
  }


#if defined(PVFMM_HAVE_CUDA)
  // Implementation of Simple CUDA_Lock

  inline void CUDA_Lock::init(size_t num_stream) {
    assert(num_stream>0);
    if(num_stream==stream.size()) return;
    cublasStatus_t status;
    cudaError_t error;

    // Delete previous streams
    for(size_t i=0;i<stream.size();i++){
      error = cudaStreamDestroy(stream[i]);
    }

    // Create new streams
    stream.resize(num_stream);
    for (size_t i = 0; i < num_stream; i++) {
      error = cudaStreamCreate(&(stream[i]));
    }

    // Create cuBLAS context
    static bool cuda_init=false;
    if (!cuda_init) {
      cuda_init = true;
      status = cublasCreate(&handle);
    }

    // Set cuBLAS to use stream[0]
    status = cublasSetStream(handle, stream[0]);
    PVFMM_UNUSED(status);
    PVFMM_UNUSED(error);
  }

  inline void CUDA_Lock::finalize () {
    if(stream.size()==0) return;
    for (size_t i = 0; i < stream.size(); i++) {
      cudaError_t error = cudaStreamDestroy(stream[i]);
      PVFMM_UNUSED(error);
    }
    cublasStatus_t status = cublasDestroy(handle);
    PVFMM_UNUSED(status);
  }

  inline cudaStream_t *CUDA_Lock::acquire_stream (int idx) {
    if (stream.size()<=(size_t)idx) init(idx+1);
    return &(stream[idx]);
  }

  inline cublasHandle_t *CUDA_Lock::acquire_handle () {
    if (stream.size()==0) init();
    return &handle;
  }

  inline void CUDA_Lock::wait (int idx) {
    if (stream.size()<=(size_t)idx) init(idx+1);
    cudaError_t error = cudaStreamSynchronize(stream[idx]);
    PVFMM_UNUSED(error);
  }

  // mat::cublasgemm specializations (declared in mat_utils.hpp); defined here
  // because they need CUDA_Lock, which is only declared in device_wrapper.hpp.
  namespace mat{
  template <> inline void cublasgemm<float>(char TransA, char TransB, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    cublasOperation_t cublasTransA, cublasTransB;
    cublasHandle_t *handle = CUDA_Lock::acquire_handle();
    if (TransA == 'T' || TransA == 't')
      cublasTransA = CUBLAS_OP_T;
    else if (TransA == 'N' || TransA == 'n')
      cublasTransA = CUBLAS_OP_N;
    if (TransB == 'T' || TransB == 't')
      cublasTransB = CUBLAS_OP_T;
    else if (TransB == 'N' || TransB == 'n')
      cublasTransB = CUBLAS_OP_N;
    cublasStatus_t status = cublasSgemm(*handle, cublasTransA, cublasTransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    PVFMM_UNUSED(status);
  }

  template <> inline void cublasgemm<double>(char TransA, char TransB, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc) {
    cublasOperation_t cublasTransA, cublasTransB;
    cublasHandle_t *handle = CUDA_Lock::acquire_handle();
    if (TransA == 'T' || TransA == 't')
      cublasTransA = CUBLAS_OP_T;
    else if (TransA == 'N' || TransA == 'n')
      cublasTransA = CUBLAS_OP_N;
    if (TransB == 'T' || TransB == 't')
      cublasTransB = CUBLAS_OP_T;
    else if (TransB == 'N' || TransB == 'n')
      cublasTransB = CUBLAS_OP_N;
    cublasStatus_t status = cublasDgemm(*handle, cublasTransA, cublasTransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    PVFMM_UNUSED(status);
  }
  }//end namespace mat
#endif

}//end namespace
