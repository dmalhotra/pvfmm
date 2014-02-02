/**
 * \file mat_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains FFTW3 wrapper functions.
 */

#ifndef _PVFMM_FFT_WRAPPER_
#define _PVFMM_FFT_WRAPPER_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif

#include <fftw3.h>
#ifdef FFTW3_MKL
#include <fftw3_mkl.h>
#endif

namespace pvfmm{

template<class T>
struct FFTW_t{};

#ifdef PVFMM_HAVE_FFTW
template<>
struct FFTW_t<double>{
  typedef fftw_plan plan;
  typedef fftw_complex cplx;

  static plan fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
      double *in, const int *inembed, int istride, int idist,
      fftw_complex *out, const int *onembed, int ostride, int odist, unsigned flags){
    #ifdef FFTW3_MKL
    int omp_p0=omp_get_num_threads();
    int omp_p1=omp_get_max_threads();
    fftw3_mkl.number_of_user_threads = (omp_p0>omp_p1?omp_p0:omp_p1);
    #endif
    return fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
        idist, out, onembed, ostride, odist, flags);
  }

  static plan fft_plan_many_dft_c2r(int rank, const int *n, int howmany,
      fftw_complex *in, const int *inembed, int istride, int idist,
      double *out, const int *onembed, int ostride, int odist, unsigned flags){
    #ifdef FFTW3_MKL
    int omp_p0=omp_get_num_threads();
    int omp_p1=omp_get_max_threads();
    fftw3_mkl.number_of_user_threads = (omp_p0>omp_p1?omp_p0:omp_p1);
    #endif
    return fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
        out, onembed, ostride, odist, flags);
  }

  static void fft_execute_dft_r2c(const fftw_plan p, double *in, fftw_complex *out){
    fftw_execute_dft_r2c(p, in, out);
  }

  static void fft_execute_dft_c2r(const fftw_plan p, fftw_complex *in, double *out){
    fftw_execute_dft_c2r(p, in, out);
  }

  static void fft_destroy_plan(fftw_plan plan){
    fftw_destroy_plan(plan);
  }

};
#endif

#ifdef PVFMM_HAVE_FFTWF
template<>
struct FFTW_t<float>{
  typedef fftwf_plan plan;
  typedef fftwf_complex cplx;

  static plan fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
      float *in, const int *inembed, int istride, int idist,
      fftwf_complex *out, const int *onembed, int ostride, int odist, unsigned flags){
    return fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
        idist, out, onembed, ostride, odist, flags);
  }

  static plan fft_plan_many_dft_c2r(int rank, const int *n, int howmany,
      fftwf_complex *in, const int *inembed, int istride, int idist,
      float *out, const int *onembed, int ostride, int odist, unsigned flags){
    return fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
        out, onembed, ostride, odist, flags);
  }

  static void fft_execute_dft_r2c(const fftwf_plan p, float *in, fftwf_complex *out){
    fftwf_execute_dft_r2c(p, in, out);
  }

  static void fft_execute_dft_c2r(const fftwf_plan p, fftwf_complex *in, float *out){
    fftwf_execute_dft_c2r(p, in, out);
  }

  static void fft_destroy_plan(fftwf_plan plan){
    fftwf_destroy_plan(plan);
  }

};
#endif

}//end namespace

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_FFT_WRAPPER_

