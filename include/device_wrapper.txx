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

#include <vector.hpp>
#include <device_wrapper.hpp>

// CUDA Stream
#if defined(PVFMM_HAVE_CUDA)
#endif

namespace pvfmm{

namespace DeviceWrapper{

  // CUDA functions
  inline uintptr_t alloc_device_cuda(size_t len) {
    char *dev_ptr=NULL;
#if defined(PVFMM_HAVE_CUDA)
    //std::cout << "cudaMalloc();" << '\n';
    cudaError_t error;
    error = cudaMalloc((void**)&dev_ptr, len);
    /*
    std::cout << cudaGetErrorString(error) << ", "
      << (uintptr_t) dev_ptr << " - " 
      << (uintptr_t) dev_ptr + len 
      << "(" << len << ")" << '\n';
      */
#endif
    return (uintptr_t)dev_ptr;
  }

  inline void free_device_cuda(char *dev_ptr) {
#if defined(PVFMM_HAVE_CUDA)
    //std::cout << "cudaFree();" << '\n';
    cudaFree(dev_ptr);
#endif
  }

  inline int host2device_cuda(char *host_ptr, char *dev_ptr, size_t len) {
    #if defined(PVFMM_HAVE_CUDA)
	//std::cout << "cudaHostRegister(), cudaMemcpyAsync(HostToDevice);" << '\n';
    cudaError_t error;
    cudaStream_t *stream = CUDA_Lock::acquire_stream(0);
    //error = cudaHostRegister(host_ptr, len, cudaHostRegisterPortable);
    //if (error != cudaSuccess) std::cout << "cudaHostRegister(): " << cudaGetErrorString(error) << '\n';
    //error = cudaMemcpyAsync(dev_ptr, host_ptr, len, cudaMemcpyHostToDevice, *stream);
    error = cudaMemcpy(dev_ptr, host_ptr, len, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
	  std::cout << "cudaMemcpyAsync(HostToDevice): " << cudaGetErrorString(error) << ", " 
		<< (uintptr_t) dev_ptr << ", len: "
	    << len << '\n';	
	  return -1;
	  }
    else return (int)len;
    #endif
    return -1;
  }

  inline int device2host_cuda(char *dev_ptr, char *host_ptr, size_t len) {
    #if defined(PVFMM_HAVE_CUDA)
	//std::cout << "cudaHostRegister(), cudaMemcpyAsync(DeviceToHost);" << '\n';
    cudaError_t error;
    cudaStream_t *stream = CUDA_Lock::acquire_stream(0);
    //error = cudaHostRegister(host_ptr, len, cudaHostRegisterPortable);
    //if (error != cudaSuccess) std::cout << "cudaHostRegister(): " << cudaGetErrorString(error) << '\n';
    //error = cudaMemcpyAsync(host_ptr, dev_ptr, len, cudaMemcpyDeviceToHost, *stream);
    error = cudaMemcpy(host_ptr, dev_ptr, len, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
	  std::cout << "cudaMemcpyAsnc(DeviceToHost): " << cudaGetErrorString(error) << ", " 
		<< (uintptr_t) dev_ptr << ", len: "
	    << len << '\n';	
	  return -1;
	}
    else return (int)len;
    #endif
    return -1;
  }


  // MIC functions

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
    #ifndef __MIC_ASYNCH__ // Wait
    #pragma offload target(mic:0)
    {MIC_Lock::wait_lock(lock_idx);}
    #endif
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
    #ifndef __MIC_ASYNCH__ // Wait
    MIC_Lock::wait_lock(lock_idx);
    #endif
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

  inline uintptr_t alloc_device(char* dev_handle, size_t len){
    #ifdef __INTEL_OFFLOAD
    return alloc_device_mic(dev_handle,len);
    #elif defined(PVFMM_HAVE_CUDA)
    return alloc_device_cuda(len);
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
    free_device_cuda((char*)dev_ptr);
    #else
    ;
    #endif
  }

  inline int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len){
    int lock_idx=-1;
    #ifdef __INTEL_OFFLOAD
    lock_idx=host2device_mic(host_ptr,dev_handle,dev_ptr,len);
    #elif defined(PVFMM_HAVE_CUDA)
    //lock_idx is len if success.
    lock_idx=host2device_cuda(host_ptr,(char*)dev_ptr,len);
    #else
    ;
    #endif
    return lock_idx;
  }

  inline int device2host(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len){
    int lock_idx=-1;
    #ifdef __INTEL_OFFLOAD
    lock_idx=device2host_mic(dev_handle,dev_ptr, host_ptr, len);
    #elif defined(PVFMM_HAVE_CUDA)
    //lock_idx is len if success.
    lock_idx=device2host_cuda((char*)dev_ptr, host_ptr, len);
    #else
    ;
    #endif
    return lock_idx;
  }

  inline void wait(int lock_idx){
    #ifdef __INTEL_OFFLOAD
    wait_mic(lock_idx);
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

  inline void MIC_Lock::init(){
    if(have_mic) abort();// Cannot be called from MIC.

    lock_idx=0;
    lock_vec.Resize(NUM_LOCKS);
    lock_vec.SetZero();
    lock_vec_=lock_vec.AllocDevice(false);
    {for(size_t i=0;i<NUM_LOCKS;i++) lock_vec [i]=1;}
    #ifdef __INTEL_OFFLOAD
    #pragma offload target(mic:0)
    {for(size_t i=0;i<NUM_LOCKS;i++) lock_vec_[i]=1;}
    #endif
  }

  inline int MIC_Lock::get_lock(){
    if(have_mic) abort();// Cannot be called from MIC.

    int idx;
    #pragma omp critical
    {
      if(lock_idx==NUM_LOCKS-1){
        int wait_lock_idx=-1;
        wait_lock_idx=MIC_Lock::curr_lock();
        MIC_Lock::wait_lock(wait_lock_idx);
        #ifdef __INTEL_OFFLOAD
        #pragma offload target(mic:0)
        {MIC_Lock::wait_lock(wait_lock_idx);}
        #endif
        MIC_Lock::init();
      }
      idx=lock_idx;
      lock_idx++;
      assert(lock_idx<NUM_LOCKS);
    }
    return idx;
  }

  inline int MIC_Lock::curr_lock(){
    if(have_mic) abort();// Cannot be called from MIC.
    return lock_idx-1;
  }

  inline void MIC_Lock::release_lock(int idx){ // Only call from inside an offload section
    #ifdef __MIC__
    if(idx>=0) lock_vec_[idx]=0;
    #endif
  }

  inline void MIC_Lock::wait_lock(int idx){
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
    #ifdef __INTEL_OFFLOAD
    #pragma offload_wait target(mic:0) wait(&lock_vec[idx])
    #endif
    lock_vec[idx]=0;
#endif
  }

  Vector<char> MIC_Lock::lock_vec;
  Vector<char>::Device MIC_Lock::lock_vec_;
  int MIC_Lock::lock_idx;



  // Implementation of Simple CUDA_Lock

  #if defined(PVFMM_HAVE_CUDA)
  CUDA_Lock::CUDA_Lock () {
    cuda_init = false;
  }

  inline void CUDA_Lock::init () {
    cudaError_t error;
    cublasStatus_t status;
    if (!cuda_init) {
      for (int i = 0; i < NUM_STREAM; i++) {
        error = cudaStreamCreate(&(stream[i]));
      }
      status = cublasCreate(&handle);
      //status = cublasSetStream(handle, stream[0]);
      cuda_init = true;
    }
  }

  inline void CUDA_Lock::terminate () {
    cudaError_t error;
    cublasStatus_t status;
    if (!cuda_init) init();
    for (int i = 0; i < NUM_STREAM; i++) {
      error = cudaStreamDestroy(stream[i]);
    }
    status = cublasDestroy(handle);
    cuda_init = false;
  }

  inline cudaStream_t *CUDA_Lock::acquire_stream (int idx) {
    if (!cuda_init) init();
    if (idx < NUM_STREAM) return &(stream[idx]);
    else return NULL;
  }

  inline cublasHandle_t *CUDA_Lock::acquire_handle () {
    if (!cuda_init) init();
    return &handle;
  }

  inline void CUDA_Lock::wait (int idx) {
    cudaError_t error;
    if (!cuda_init) init();
    if (idx < NUM_STREAM) error = cudaStreamSynchronize(stream[idx]);
  }

  cudaStream_t CUDA_Lock::stream[NUM_STREAM];
  cublasHandle_t CUDA_Lock::handle;
  bool CUDA_Lock::cuda_init;

  #endif

}//end namespace
