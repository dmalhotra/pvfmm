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
#ifdef __cplusplus
extern "C" {
#endif
  void* host_malloc_cuda(size_t size);
  void host_free_cuda(void* p);
#ifdef __cplusplus
}
#endif

  inline uintptr_t alloc_device_cuda(char* dev_handle, size_t len) {
    char *dev_ptr=NULL;
#if defined(PVFMM_HAVE_CUDA)
    cudaError_t error;
    error = cudaHostRegister(dev_handle, len, cudaHostRegisterPortable);
    if (error != cudaSuccess)
      std::cout<<cudaGetErrorString(error)<< '\n';
    assert(error == cudaSuccess);
    error = cudaMalloc((void**)&dev_ptr, len);
    if (error != cudaSuccess)
      std::cout<<cudaGetErrorString(error)<< '\n';
    assert(error == cudaSuccess);
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
#endif
  }

  inline int host2device_cuda(char *host_ptr, char *dev_ptr, size_t len) {
    #if defined(PVFMM_HAVE_CUDA)
    cudaError_t error;
    cudaStream_t *stream = CUDA_Lock::acquire_stream();
    error = cudaMemcpyAsync(dev_ptr, host_ptr, len, cudaMemcpyHostToDevice, *stream);
    if (error != cudaSuccess) std::cout<<cudaGetErrorString(error)<< '\n';
    assert(error == cudaSuccess);
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
    #endif
    return 0;
  }


  // MIC functions

  #define ALLOC alloc_if(1) free_if(0)
  #define FREE alloc_if(0) free_if(1)
  #define REUSE alloc_if(0) free_if(0)

  inline uintptr_t alloc_device_mic(char* dev_handle, size_t len){
    assert(dev_handle!=NULL);
    uintptr_t dev_ptr=(uintptr_t)NULL;
    #ifdef __INTEL_OFFLOAD
    #pragma offload target(mic:0) nocopy( dev_handle: length(len) ALLOC) out(dev_ptr)
    #endif
    {dev_ptr=(uintptr_t)dev_handle;}
    return dev_ptr;
  }

  inline void free_device_mic(char* dev_handle, uintptr_t dev_ptr){
    #ifdef __INTEL_OFFLOAD
    #pragma offload          target(mic:0) in( dev_handle: length(0) FREE)
    {
      assert(dev_ptr==(uintptr_t)dev_handle);
    }
    #endif
  }

  inline int host2device_mic(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len){
    #ifdef __INTEL_OFFLOAD
    int wait_lock_idx=MIC_Lock::curr_lock();
    int lock_idx=MIC_Lock::get_lock();
    if(dev_handle==host_ptr){
      #pragma offload target(mic:0)  in( dev_handle        :              length(len)  REUSE ) signal(&MIC_Lock::lock_vec[lock_idx])
      {
        assert(dev_ptr==(uintptr_t)dev_handle);
        MIC_Lock::wait_lock(wait_lock_idx);
        MIC_Lock::release_lock(lock_idx);
      }
    }else{
      #pragma offload target(mic:0)  in(host_ptr   [0:len] : into ( dev_handle[0:len]) REUSE ) signal(&MIC_Lock::lock_vec[lock_idx])
      {
        assert(dev_ptr==(uintptr_t)dev_handle);
        MIC_Lock::wait_lock(wait_lock_idx);
        MIC_Lock::release_lock(lock_idx);
      }
    }
    return lock_idx;
    #endif
    return -1;
  }

  inline int device2host_mic(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len){
    #ifdef __INTEL_OFFLOAD
    int wait_lock_idx=MIC_Lock::curr_lock();
    int lock_idx=MIC_Lock::get_lock();
    if(dev_handle==host_ptr){
      #pragma offload target(mic:0) out( dev_handle        :              length(len)  REUSE ) signal(&MIC_Lock::lock_vec[lock_idx])
      {
        assert(dev_ptr==(uintptr_t)dev_handle);
        MIC_Lock::wait_lock(wait_lock_idx);
        MIC_Lock::release_lock(lock_idx);
      }
    }else{
      #pragma offload target(mic:0) out( dev_handle[0:len] : into (host_ptr   [0:len]) REUSE ) signal(&MIC_Lock::lock_vec[lock_idx])
      {
        assert(dev_ptr==(uintptr_t)dev_handle);
        MIC_Lock::wait_lock(wait_lock_idx);
        MIC_Lock::release_lock(lock_idx);
      }
    }
    return lock_idx;
    #endif
    return -1;
  }

  inline void wait_mic(int lock_idx){
    #ifdef __INTEL_OFFLOAD
    MIC_Lock::wait_lock(lock_idx);
    #endif
  }



  // Wrapper functions

  inline void* host_malloc(size_t size){
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
    #ifdef __INTEL_OFFLOAD
    return alloc_device_mic(dev_handle,len);
    #elif defined(PVFMM_HAVE_CUDA)
    return alloc_device_cuda(dev_handle,len);
    #else
    uintptr_t dev_ptr=(uintptr_t)NULL;
    {dev_ptr=(uintptr_t)dev_handle;}
    return dev_ptr;
    #endif
  }

  inline void free_device(char* dev_handle, uintptr_t dev_ptr){
    #ifdef __INTEL_OFFLOAD
    free_device_mic(dev_handle,dev_ptr);
    #elif defined(PVFMM_HAVE_CUDA)
    free_device_cuda(dev_handle,dev_ptr);
    #else
    ;
    #endif
  }

  template <int SYNC>
  inline int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len){
    int lock_idx=-1;
    #ifdef __INTEL_OFFLOAD
    lock_idx=host2device_mic(host_ptr,dev_handle,dev_ptr,len);
    if(SYNC){
      #pragma offload target(mic:0)
      {MIC_Lock::wait_lock(lock_idx);}
    }
    #elif defined(PVFMM_HAVE_CUDA)
    lock_idx=host2device_cuda(host_ptr,(char*)dev_ptr,len);
    #else
    ;
    #endif
    return lock_idx;
  }

  template <int SYNC>
  inline int device2host(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len){
    int lock_idx=-1;
    #ifdef __INTEL_OFFLOAD
    lock_idx=device2host_mic(dev_handle,dev_ptr, host_ptr, len);
    if(SYNC) MIC_Lock::wait_lock(lock_idx);
    #elif defined(PVFMM_HAVE_CUDA)
    lock_idx=device2host_cuda((char*)dev_ptr, host_ptr, len);
    #else
    ;
    #endif
    return lock_idx;
  }

  inline void wait(int lock_idx){
    #ifdef __INTEL_OFFLOAD
    wait_mic(lock_idx);
    #elif defined(PVFMM_HAVE_CUDA)
    CUDA_Lock::wait();
    #else
    ;
    #endif
  }

}


  // Implementation of MIC_Lock

  #ifdef __MIC__
  #define have_mic 1
  #else
  #define have_mic 0
  #endif

  #define NUM_LOCKS 1000000
  inline void MIC_Lock::init(){
    #ifdef __INTEL_OFFLOAD
    if(have_mic) abort();// Cannot be called from MIC.

    lock_idx=0;
    lock_vec.Resize(NUM_LOCKS);
    lock_vec.SetZero();
    lock_vec_=lock_vec.AllocDevice(false);
    {for(size_t i=0;i<NUM_LOCKS;i++) lock_vec [i]=1;}
    #pragma offload target(mic:0)
    {for(size_t i=0;i<NUM_LOCKS;i++) lock_vec_[i]=1;}
    #endif
  }

  inline int MIC_Lock::get_lock(){
    #ifdef __INTEL_OFFLOAD
    if(have_mic) abort();// Cannot be called from MIC.

    int idx;
    #pragma omp critical
    {
      if(lock_idx==NUM_LOCKS-1){
        int wait_lock_idx=-1;
        wait_lock_idx=MIC_Lock::curr_lock();
        MIC_Lock::wait_lock(wait_lock_idx);
        #pragma offload target(mic:0)
        {MIC_Lock::wait_lock(wait_lock_idx);}
        MIC_Lock::init();
      }
      idx=lock_idx;
      lock_idx++;
      assert(lock_idx<NUM_LOCKS);
    }
    return idx;
    #else
    return -1;
    #endif
  }
  #undef NUM_LOCKS

  inline int MIC_Lock::curr_lock(){
    #ifdef __INTEL_OFFLOAD
    if(have_mic) abort();// Cannot be called from MIC.
    return lock_idx-1;
    #else
    return -1;
    #endif
  }

  inline void MIC_Lock::release_lock(int idx){ // Only call from inside an offload section
    #ifdef __INTEL_OFFLOAD
    #ifdef __MIC__
    if(idx>=0) lock_vec_[idx]=0;
    #endif
    #endif
  }

  inline void MIC_Lock::wait_lock(int idx){
    #ifdef __INTEL_OFFLOAD
    #ifdef __MIC__
    if(idx>=0) while(lock_vec_[idx]==1){
      _mm_delay_32(8192);
    }
    #else
    if(idx<0 || lock_vec[idx]==0) return;
    if(lock_vec[idx]==2){
      while(lock_vec[idx]==2);
      return;
    }
    lock_vec[idx]=2;
    #pragma offload_wait target(mic:0) wait(&lock_vec[idx])
    lock_vec[idx]=0;
    #endif
    #endif
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
    for (int i = 0; i < num_stream; i++) {
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
  }

  inline void CUDA_Lock::finalize () {
    if(stream.size()==0) return;
    for (int i = 0; i < stream.size(); i++) {
      cudaError_t error = cudaStreamDestroy(stream[i]);
    }
    cublasStatus_t status = cublasDestroy(handle);
  }

  inline cudaStream_t *CUDA_Lock::acquire_stream (int idx) {
    if (stream.size()<=idx) init(idx+1);
    return &(stream[idx]);
  }

  inline cublasHandle_t *CUDA_Lock::acquire_handle () {
    if (stream.size()==0) init();
    return &handle;
  }

  inline void CUDA_Lock::wait (int idx) {
    if (stream.size()<=idx) init(idx+1);
    cudaError_t error = cudaStreamSynchronize(stream[idx]);
  }
#endif

}//end namespace
