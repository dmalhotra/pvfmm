/**
 * \file device_wrapper.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 6-5-2013
 * \brief This file contains definition of DeviceWrapper.
 */

#ifndef _PVFMM_DEVICE_WRAPPER_HPP_
#define _PVFMM_DEVICE_WRAPPER_HPP_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif

#include <cstdlib>
#include <cassert>
#include <stdint.h>
#include <pvfmm_common.hpp>

namespace pvfmm{

namespace DeviceWrapper{

  uintptr_t alloc_device(char* dev_handle, size_t len);

  void free_device(char* dev_handle, uintptr_t dev_ptr);

  int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len);

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

    #ifdef __MIC_ASYNCH__
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

  #define NUM_LOCKS 1000000
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

}//end namespace

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <device_wrapper.txx>

#endif //_PVFMM_DEVICE_WRAPPER_HPP_
