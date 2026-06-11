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

  template <int SYNC=PVFMM_DEVICE_SYNC>
  int host2device(char* host_ptr, char* dev_handle, uintptr_t dev_ptr, size_t len);

  template <int SYNC=PVFMM_DEVICE_SYNC>
  int device2host(char* dev_handle, uintptr_t dev_ptr, char* host_ptr, size_t len);

  void wait(int lock_idx);

}//end namespace

template <class T> class Matrix;

/**
 * \brief Lightweight handle describing a vector buffer (device or host) to
 * kernel/offload code. Formerly nested as Vector<T>::Device.
 */
template <class T>
struct DeviceVector{

  DeviceVector(): dim(0), dev_ptr(0) {}

  // Bind a host-side view (CPU/MIC fallback path).
  DeviceVector& operator=(Vector<T>& V){
    dim=V.Dim();
    dev_ptr=(uintptr_t)V.Begin();
    return *this;
  }

  inline T& operator[](size_t j) const{
    return ((T*)dev_ptr)[j];
  }

  size_t dim;
  uintptr_t dev_ptr;
};

/**
 * \brief Lightweight handle describing a matrix buffer (device or host) to
 * kernel/offload code. Formerly nested as Matrix<T>::Device.
 */
template <class T>
struct DeviceMatrix{

  DeviceMatrix(){
    dim[0]=0;
    dim[1]=0;
    dev_ptr=0;
    lock_idx=-1;
  }

  // Bind a host-side view (CPU/MIC fallback path).
  DeviceMatrix& operator=(Matrix<T>& M){
    dim[0]=M.Dim(0);
    dim[1]=M.Dim(1);
    dev_ptr=(uintptr_t)M.Begin();
    return *this;
  }

  inline T* operator[](size_t j) const{
    assert(j<dim[0]);
    return &((T*)dev_ptr)[j*dim[1]];
  }

  size_t dim[2];
  uintptr_t dev_ptr;
  int lock_idx;
};

/**
 * \brief Owns the device-side mirror of one host buffer: the page-locked
 * registration of the host range (cudaHostRegister) and the device
 * allocation. This lifecycle state previously lived inside Vector/Matrix
 * as the `dev` member and the AllocDevice/FreeDevice/Device2Host methods.
 *
 * Contract: while bound, the host buffer must not be freed, resized, or
 * reallocated — call Free() first. Releasing the pinned registration
 * requires the host pages to still be mapped, so a stale binding cannot be
 * released after the host buffer is gone; AllocDevice asserts if called
 * with a different host range while still bound.
 */
class DeviceMirror{
  public:

  DeviceMirror(): host_ptr(NULL), len(0), dev_ptr(0), lock_idx(-1) {}

  ~DeviceMirror(){ Free(); }

  DeviceMirror(const DeviceMirror&) = delete;
  DeviceMirror& operator=(const DeviceMirror&) = delete;

  DeviceMirror(DeviceMirror&& m) noexcept: host_ptr(m.host_ptr), len(m.len), dev_ptr(m.dev_ptr), lock_idx(m.lock_idx){
    m.host_ptr=NULL; m.len=0; m.dev_ptr=0; m.lock_idx=-1;
  }

  DeviceMirror& operator=(DeviceMirror&& m) noexcept{
    if(this!=&m){
      Free();
      host_ptr=m.host_ptr; len=m.len; dev_ptr=m.dev_ptr; lock_idx=m.lock_idx;
      m.host_ptr=NULL; m.len=0; m.dev_ptr=0; m.lock_idx=-1;
    }
    return *this;
  }

  /**
   * Bind to `host` and allocate the device block if not already bound
   * (no-op when already bound to the same range). If copy, enqueue an
   * asynchronous host-to-device copy of the full range.
   */
  template <class T> DeviceVector<T> AllocDevice(Vector<T>& host, bool copy);
  template <class T> DeviceMatrix<T> AllocDevice(Matrix<T>& host, bool copy);

  /**
   * Asynchronous device-to-host copy of the bound range, into `dst` if
   * given (default: back into the bound host range itself).
   */
  void Device2Host(char* dst=NULL);

  /** Wait for the last asynchronous copy to complete. */
  void Device2HostWait();

  /** Release the device allocation and the host pinning. */
  void Free();

  bool Allocated() const{ return dev_ptr!=0; }

  private:

  char* host_ptr;
  size_t len;
  uintptr_t dev_ptr;
  int lock_idx;
};



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

    #ifdef PVFMM_DEVICE_SYNC
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
      static DeviceVector<char> lock_vec_;

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
