/**
 * \file device_wrapper.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 7-30-2014
 * \brief This file contains implementation of DeviceWrapper.
 */

#include <mpi.h>

#include <device_wrapper.hpp>
#include <vector.hpp>

namespace pvfmm{

  Vector<char> MIC_Lock::lock_vec;
  Vector<char>::Device MIC_Lock::lock_vec_;
  int MIC_Lock::lock_idx;

#if defined(PVFMM_HAVE_CUDA)
  std::vector<cudaStream_t> CUDA_Lock::stream;
  cublasHandle_t CUDA_Lock::handle;
#endif

}//end namespace
