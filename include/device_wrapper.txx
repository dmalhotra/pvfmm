/**
 * \file device_wrapper.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 6-5-2013
 * \brief This file contains implementation of DeviceWrapper.
 */

#include <vector.hpp>
#include <device_wrapper.hpp>

namespace pvfmm{

namespace DeviceWrapper{

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
    #else
    uintptr_t dev_ptr=(uintptr_t)NULL;
    {dev_ptr=(uintptr_t)dev_handle;}
    return dev_ptr;
    #endif
  }

  inline void free_device(char* dev_handle, uintptr_t dev_ptr){
    #ifdef __INTEL_OFFLOAD
    free_device_mic(dev_handle,dev_ptr);
    #else
    ;
    #endif
  }

  inline int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len){
    int lock_idx=-1;
    #ifdef __INTEL_OFFLOAD
    lock_idx=host2device_mic(host_ptr,dev_handle,dev_ptr,len);
    #else
    ;
    #endif
    return lock_idx;
  }

  inline int device2host(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len){
    int lock_idx=-1;
    #ifdef __INTEL_OFFLOAD
    lock_idx=device2host_mic(dev_handle,dev_ptr, host_ptr, len);
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

}//end namespace
