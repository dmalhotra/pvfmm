/**
 * \file device_wrapper.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 6-5-2013
 * \brief This file contains definition of DeviceWrapper.
 */

#include <cstdlib>
#include <stdint.h>

// Cuda Headers
#if defined(PVFMM_HAVE_CUDA)
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#include <pvfmm_common.hpp>
#include <vector.hpp>

#ifndef _PVFMM_DEVICE_WRAPPER_HPP_
#define _PVFMM_DEVICE_WRAPPER_HPP_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{

namespace DeviceWrapper{

  void* host_malloc(size_t size);

  void host_free(void*);

  uintptr_t alloc_device(char* dev_handle, size_t len);

  void free_device(char* dev_handle, uintptr_t dev_ptr);

  template <int SYNC=__DEVICE_SYNC__>
  int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len);

  template <int SYNC=__DEVICE_SYNC__>
  int device2host(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len);

  void wait(int lock_idx);

}//end namespace



/*
   Usage of 'MIC_Lock' in Asynchronous Offloads
   --------------------------------------------

Note: Any MIC offload section should look like this:

    int wait_lock_idx=MIC_Lock::curr_lock();
    int lock_idx=MIC_Lock::get_lock();
    #pragma offload target(mic:0) signal(&MIC_Lock::lock_vec[lock_idx])
    {
      MIC_Lock::wait_lock(wait_lock_idx);

      // Offload code here...

      MIC_Lock::release_lock(lock_idx);
    }

    #ifdef __DEVICE_SYNC__
    MIC_Lock::wait_lock(lock_idx);
    #endif

   This ensures the execution of offloaded code does not overlap with other
asynchronous offloaded code and that data transfers from host to mic have
completed before the data is accessed.  You will however, need to be careful
not to overwrite data on mic which may be transferring to the host, or data on
the host which may be transferring to the mic.

On the host, to wait for the last asynchronous offload section or data
transfer, use:

    int wait_lock_idx=MIC_Lock::curr_lock();
    MIC_Lock::wait_lock(wait_lock_idx);
*/

  class MIC_Lock{
    public:

      static void init();

      static int get_lock();

      static void release_lock(int idx);

      static void wait_lock(int idx);

      static int curr_lock();

      static Vector<char> lock_vec;
      static Vector<char>::Device lock_vec_;

    private:
      MIC_Lock(){}; // private constructor for static class.
      static int lock_idx;
  };

#if defined(PVFMM_HAVE_CUDA)
  class CUDA_Lock {
    public:
      static void init(size_t num_stream=1);
      static void finalize();

      static cudaStream_t *acquire_stream(int idx=0);
      static cublasHandle_t *acquire_handle();
      static void wait(int idx=0);
    private:
      CUDA_Lock();
      static std::vector<cudaStream_t> stream;
      static cublasHandle_t handle;
  };
#endif

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <device_wrapper.txx>

#endif //_PVFMM_DEVICE_WRAPPER_HPP_
