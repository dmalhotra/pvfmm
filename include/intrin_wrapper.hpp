/**
 * \file intrin_wrapper.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-19-2014
 * \brief This file contains the templated wrappers for vector intrinsics.
 */

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#if defined(__MIC__)
#include <immintrin.h>
#endif

#ifndef _PVFMM_INTRIN_WRAPPER_HPP_
#define _PVFMM_INTRIN_WRAPPER_HPP_

namespace pvfmm{

template <class T>
inline T zero_intrin(){
  return (T)0;
}

template <class T, class Real_t>
inline T set_intrin(const Real_t& a){
  return a;
}

template <class T, class Real_t>
inline T load_intrin(Real_t const* a){
  return a[0];
}

template <class T, class Real_t>
inline T bcast_intrin(Real_t const* a){
  return a[0];
}

template <class T, class Real_t>
inline void store_intrin(Real_t* a, const T& b){
  a[0]=b;
}

template <class T>
inline T mul_intrin(const T& a, const T& b){
  return a*b;
}

template <class T>
inline T add_intrin(const T& a, const T& b){
  return a+b;
}

template <class T>
inline T sub_intrin(const T& a, const T& b){
  return a-b;
}

template <class T>
inline T rinv_approx_intrin(const T& r2){
  if(r2!=0) return 1.0/pvfmm::sqrt<T>(r2);
  return 0;
}

template <class T, class Real_t>
inline void rinv_newton_intrin(T& rinv, const T& r2, const Real_t& nwtn_const){
  rinv=rinv*(nwtn_const-r2*rinv*rinv);
}

template <class T>
inline T rinv_single_intrin(const T& r2){
  if(r2!=0) return 1.0/pvfmm::sqrt<T>(r2);
  return 0;
}

template <class T>
inline T sin_intrin(const T& t){
  return pvfmm::sin<T>(t);
}

template <class T>
inline T cos_intrin(const T& t){
  return pvfmm::cos<T>(t);
}



#ifdef __SSE3__
template <>
inline __m128 zero_intrin(){
  return _mm_setzero_ps();
}

template <>
inline __m128d zero_intrin(){
  return _mm_setzero_pd();
}

template <>
inline __m128 set_intrin(const float& a){
  return _mm_set_ps1(a);
}

template <>
inline __m128d set_intrin(const double& a){
  return _mm_set_pd1(a);
}

template <>
inline __m128 load_intrin(float const* a){
  return _mm_load_ps(a);
}

template <>
inline __m128d load_intrin(double const* a){
  return _mm_load_pd(a);
}

template <>
inline __m128 bcast_intrin(float const* a){
  return _mm_set_ps1(a[0]);
}

template <>
inline __m128d bcast_intrin(double const* a){
  return _mm_load_pd1(a);
}

template <>
inline void store_intrin(float* a, const __m128& b){
  return _mm_store_ps(a,b);
}

template <>
inline void store_intrin(double* a, const __m128d& b){
  return _mm_store_pd(a,b);
}

template <>
inline __m128 mul_intrin(const __m128& a, const __m128& b){
  return _mm_mul_ps(a,b);
}

template <>
inline __m128d mul_intrin(const __m128d& a, const __m128d& b){
  return _mm_mul_pd(a,b);
}

template <>
inline __m128 add_intrin(const __m128& a, const __m128& b){
  return _mm_add_ps(a,b);
}

template <>
inline __m128d add_intrin(const __m128d& a, const __m128d& b){
  return _mm_add_pd(a,b);
}

template <>
inline __m128 sub_intrin(const __m128& a, const __m128& b){
  return _mm_sub_ps(a,b);
}

template <>
inline __m128d sub_intrin(const __m128d& a, const __m128d& b){
  return _mm_sub_pd(a,b);
}

template <>
inline __m128 rinv_approx_intrin(const __m128& r2){
  #define VEC_INTRIN          __m128
  #define RSQRT_INTRIN(a)     _mm_rsqrt_ps(a)
  #define CMPEQ_INTRIN(a,b)   _mm_cmpeq_ps(a,b)
  #define ANDNOT_INTRIN(a,b)  _mm_andnot_ps(a,b)

  // Approx inverse square root which returns zero for r2=0
  return ANDNOT_INTRIN(CMPEQ_INTRIN(r2,zero_intrin<VEC_INTRIN>()),RSQRT_INTRIN(r2));

  #undef VEC_INTRIN
  #undef RSQRT_INTRIN
  #undef CMPEQ_INTRIN
  #undef ANDNOT_INTRIN
}

template <>
inline __m128d rinv_approx_intrin(const __m128d& r2){
  #define PD2PS(a) _mm_cvtpd_ps(a)
  #define PS2PD(a) _mm_cvtps_pd(a)
  return PS2PD(rinv_approx_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

template <>
inline void rinv_newton_intrin(__m128& rinv, const __m128& r2, const float& nwtn_const){
  #define VEC_INTRIN       __m128
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  // We do not compute the product with 0.5 and this needs to be adjusted later
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline void rinv_newton_intrin(__m128d& rinv, const __m128d& r2, const double& nwtn_const){
  #define VEC_INTRIN       __m128d
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  // We do not compute the product with 0.5 and this needs to be adjusted later
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline __m128 rinv_single_intrin(const __m128& r2){
  #define VEC_INTRIN __m128
  VEC_INTRIN rinv=rinv_approx_intrin(r2);
  rinv_newton_intrin(rinv,r2,(float)3.0);
  return rinv;
  #undef VEC_INTRIN
}

template <>
inline __m128d rinv_single_intrin(const __m128d& r2){
  #define PD2PS(a) _mm_cvtpd_ps(a)
  #define PS2PD(a) _mm_cvtps_pd(a)
  return PS2PD(rinv_single_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

#ifdef PVFMM_HAVE_INTEL_SVML
template <>
inline __m128 sin_intrin(const __m128& t){
  return _mm_sin_ps(t);
}

template <>
inline __m128 cos_intrin(const __m128& t){
  return _mm_cos_ps(t);
}

template <>
inline __m128d sin_intrin(const __m128d& t){
  return _mm_sin_pd(t);
}

template <>
inline __m128d cos_intrin(const __m128d& t){
  return _mm_cos_pd(t);
}
#else
template <>
inline __m128 sin_intrin(const __m128& t_){
  union{float e[4];__m128 d;} t; store_intrin(t.e, t_);
  return _mm_set_ps(pvfmm::sin<float>(t.e[3]),pvfmm::sin<float>(t.e[2]),pvfmm::sin<float>(t.e[1]),pvfmm::sin<float>(t.e[0]));
}

template <>
inline __m128 cos_intrin(const __m128& t_){
  union{float e[4];__m128 d;} t; store_intrin(t.e, t_);
  return _mm_set_ps(pvfmm::cos<float>(t.e[3]),pvfmm::cos<float>(t.e[2]),pvfmm::cos<float>(t.e[1]),pvfmm::cos<float>(t.e[0]));
}

template <>
inline __m128d sin_intrin(const __m128d& t_){
  union{double e[2];__m128d d;} t; store_intrin(t.e, t_);
  return _mm_set_pd(pvfmm::sin<double>(t.e[1]),pvfmm::sin<double>(t.e[0]));
}

template <>
inline __m128d cos_intrin(const __m128d& t_){
  union{double e[2];__m128d d;} t; store_intrin(t.e, t_);
  return _mm_set_pd(pvfmm::cos<double>(t.e[1]),pvfmm::cos<double>(t.e[0]));
}
#endif
#endif



#ifdef __AVX__
template <>
inline __m256 zero_intrin(){
  return _mm256_setzero_ps();
}

template <>
inline __m256d zero_intrin(){
  return _mm256_setzero_pd();
}

template <>
inline __m256 set_intrin(const float& a){
  return _mm256_set_ps(a,a,a,a,a,a,a,a);
}

template <>
inline __m256d set_intrin(const double& a){
  return _mm256_set_pd(a,a,a,a);
}

template <>
inline __m256 load_intrin(float const* a){
  return _mm256_load_ps(a);
}

template <>
inline __m256d load_intrin(double const* a){
  return _mm256_load_pd(a);
}

template <>
inline __m256 bcast_intrin(float const* a){
  return _mm256_broadcast_ss(a);
}

template <>
inline __m256d bcast_intrin(double const* a){
  return _mm256_broadcast_sd(a);
}

template <>
inline void store_intrin(float* a, const __m256& b){
  return _mm256_store_ps(a,b);
}

template <>
inline void store_intrin(double* a, const __m256d& b){
  return _mm256_store_pd(a,b);
}

template <>
inline __m256 mul_intrin(const __m256& a, const __m256& b){
  return _mm256_mul_ps(a,b);
}

template <>
inline __m256d mul_intrin(const __m256d& a, const __m256d& b){
  return _mm256_mul_pd(a,b);
}

template <>
inline __m256 add_intrin(const __m256& a, const __m256& b){
  return _mm256_add_ps(a,b);
}

template <>
inline __m256d add_intrin(const __m256d& a, const __m256d& b){
  return _mm256_add_pd(a,b);
}

template <>
inline __m256 sub_intrin(const __m256& a, const __m256& b){
  return _mm256_sub_ps(a,b);
}

template <>
inline __m256d sub_intrin(const __m256d& a, const __m256d& b){
  return _mm256_sub_pd(a,b);
}

template <>
inline __m256 rinv_approx_intrin(const __m256& r2){
  #define VEC_INTRIN          __m256
  #define RSQRT_INTRIN(a)     _mm256_rsqrt_ps(a)
  #define CMPEQ_INTRIN(a,b)   _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cmpeq_ps(_mm256_extractf128_ps(a,0),_mm256_extractf128_ps(b,0))),\
                                                                    (_mm_cmpeq_ps(_mm256_extractf128_ps(a,1),_mm256_extractf128_ps(b,1))), 1)
  #define ANDNOT_INTRIN(a,b)  _mm256_andnot_ps(a,b)

  // Approx inverse square root which returns zero for r2=0
  return ANDNOT_INTRIN(CMPEQ_INTRIN(r2,zero_intrin<VEC_INTRIN>()),RSQRT_INTRIN(r2));

  #undef VEC_INTRIN
  #undef RSQRT_INTRIN
  #undef CMPEQ_INTRIN
  #undef ANDNOT_INTRIN
}

template <>
inline __m256d rinv_approx_intrin(const __m256d& r2){
  #define PD2PS(a) _mm256_cvtpd_ps(a)
  #define PS2PD(a) _mm256_cvtps_pd(a)
  return PS2PD(rinv_approx_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

template <>
inline void rinv_newton_intrin(__m256& rinv, const __m256& r2, const float& nwtn_const){
  #define VEC_INTRIN       __m256
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  // We do not compute the product with 0.5 and this needs to be adjusted later
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline void rinv_newton_intrin(__m256d& rinv, const __m256d& r2, const double& nwtn_const){
  #define VEC_INTRIN       __m256d
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  // We do not compute the product with 0.5 and this needs to be adjusted later
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline __m256 rinv_single_intrin(const __m256& r2){
  #define VEC_INTRIN __m256
  VEC_INTRIN rinv=rinv_approx_intrin(r2);
  rinv_newton_intrin(rinv,r2,(float)3.0);
  return rinv;
  #undef VEC_INTRIN
}

template <>
inline __m256d rinv_single_intrin(const __m256d& r2){
  #define PD2PS(a) _mm256_cvtpd_ps(a)
  #define PS2PD(a) _mm256_cvtps_pd(a)
  return PS2PD(rinv_single_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

#ifdef PVFMM_HAVE_INTEL_SVML
template <>
inline __m256 sin_intrin(const __m256& t){
  return _mm256_sin_ps(t);
}

template <>
inline __m256 cos_intrin(const __m256& t){
  return _mm256_cos_ps(t);
}

template <>
inline __m256d sin_intrin(const __m256d& t){
  return _mm256_sin_pd(t);
}

template <>
inline __m256d cos_intrin(const __m256d& t){
  return _mm256_cos_pd(t);
}
#else
template <>
inline __m256 sin_intrin(const __m256& t_){
  union{float e[8];__m256 d;} t; store_intrin(t.e, t_);//t.d=t_;
  return _mm256_set_ps(pvfmm::sin<float>(t.e[7]),pvfmm::sin<float>(t.e[6]),pvfmm::sin<float>(t.e[5]),pvfmm::sin<float>(t.e[4]),pvfmm::sin<float>(t.e[3]),pvfmm::sin<float>(t.e[2]),pvfmm::sin<float>(t.e[1]),pvfmm::sin<float>(t.e[0]));
}

template <>
inline __m256 cos_intrin(const __m256& t_){
  union{float e[8];__m256 d;} t; store_intrin(t.e, t_);//t.d=t_;
  return _mm256_set_ps(pvfmm::cos<float>(t.e[7]),pvfmm::cos<float>(t.e[6]),pvfmm::cos<float>(t.e[5]),pvfmm::cos<float>(t.e[4]),pvfmm::cos<float>(t.e[3]),pvfmm::cos<float>(t.e[2]),pvfmm::cos<float>(t.e[1]),pvfmm::cos<float>(t.e[0]));
}

template <>
inline __m256d sin_intrin(const __m256d& t_){
  union{double e[4];__m256d d;} t; store_intrin(t.e, t_);//t.d=t_;
  return _mm256_set_pd(pvfmm::sin<double>(t.e[3]),pvfmm::sin<double>(t.e[2]),pvfmm::sin<double>(t.e[1]),pvfmm::sin<double>(t.e[0]));
}

template <>
inline __m256d cos_intrin(const __m256d& t_){
  union{double e[4];__m256d d;} t; store_intrin(t.e, t_);//t.d=t_;
  return _mm256_set_pd(pvfmm::cos<double>(t.e[3]),pvfmm::cos<double>(t.e[2]),pvfmm::cos<double>(t.e[1]),pvfmm::cos<double>(t.e[0]));
}
#endif
#endif



template <class VEC, class Real_t>
inline VEC rinv_intrin0(VEC r2){
  #define NWTN0 0
  #define NWTN1 0
  #define NWTN2 0
  #define NWTN3 0

  //Real_t scal=1;                      Real_t const_nwtn0=3*scal*scal;
  //scal=(NWTN0?2*scal*scal*scal:scal); Real_t const_nwtn1=3*scal*scal;
  //scal=(NWTN1?2*scal*scal*scal:scal); Real_t const_nwtn2=3*scal*scal;
  //scal=(NWTN2?2*scal*scal*scal:scal); Real_t const_nwtn3=3*scal*scal;

  VEC rinv;
  #if NWTN0
  rinv=rinv_single_intrin(r2);
  #else
  rinv=rinv_approx_intrin(r2);
  #endif

  #if NWTN1
  rinv_newton_intrin(rinv,r2,const_nwtn1);
  #endif
  #if NWTN2
  rinv_newton_intrin(rinv,r2,const_nwtn2);
  #endif
  #if NWTN3
  rinv_newton_intrin(rinv,r2,const_nwtn3);
  #endif

  return rinv;

  #undef NWTN0
  #undef NWTN1
  #undef NWTN2
  #undef NWTN3
}

template <class VEC, class Real_t>
inline VEC rinv_intrin1(VEC r2){
  #define NWTN0 0
  #define NWTN1 1
  #define NWTN2 0
  #define NWTN3 0

  Real_t scal=1;                      //Real_t const_nwtn0=3*scal*scal;
  scal=(NWTN0?2*scal*scal*scal:scal); Real_t const_nwtn1=3*scal*scal;
  //scal=(NWTN1?2*scal*scal*scal:scal); Real_t const_nwtn2=3*scal*scal;
  //scal=(NWTN2?2*scal*scal*scal:scal); Real_t const_nwtn3=3*scal*scal;

  VEC rinv;
  #if NWTN0
  rinv=rinv_single_intrin(r2);
  #else
  rinv=rinv_approx_intrin(r2);
  #endif

  #if NWTN1
  rinv_newton_intrin(rinv,r2,const_nwtn1);
  #endif
  #if NWTN2
  rinv_newton_intrin(rinv,r2,const_nwtn2);
  #endif
  #if NWTN3
  rinv_newton_intrin(rinv,r2,const_nwtn3);
  #endif

  return rinv;

  #undef NWTN0
  #undef NWTN1
  #undef NWTN2
  #undef NWTN3
}

template <class VEC, class Real_t>
inline VEC rinv_intrin2(VEC r2){
  #define NWTN0 0
  #define NWTN1 1
  #define NWTN2 1
  #define NWTN3 0

  Real_t scal=1;                      //Real_t const_nwtn0=3*scal*scal;
  scal=(NWTN0?2*scal*scal*scal:scal); Real_t const_nwtn1=3*scal*scal;
  scal=(NWTN1?2*scal*scal*scal:scal); Real_t const_nwtn2=3*scal*scal;
  //scal=(NWTN2?2*scal*scal*scal:scal); Real_t const_nwtn3=3*scal*scal;

  VEC rinv;
  #if NWTN0
  rinv=rinv_single_intrin(r2);
  #else
  rinv=rinv_approx_intrin(r2);
  #endif

  #if NWTN1
  rinv_newton_intrin(rinv,r2,const_nwtn1);
  #endif
  #if NWTN2
  rinv_newton_intrin(rinv,r2,const_nwtn2);
  #endif
  #if NWTN3
  rinv_newton_intrin(rinv,r2,const_nwtn3);
  #endif

  return rinv;

  #undef NWTN0
  #undef NWTN1
  #undef NWTN2
  #undef NWTN3
}

template <class VEC, class Real_t>
inline VEC rinv_intrin3(VEC r2){
  #define NWTN0 0
  #define NWTN1 1
  #define NWTN2 1
  #define NWTN3 1

  Real_t scal=1;                      //Real_t const_nwtn0=3*scal*scal;
  scal=(NWTN0?2*scal*scal*scal:scal); Real_t const_nwtn1=3*scal*scal;
  scal=(NWTN1?2*scal*scal*scal:scal); Real_t const_nwtn2=3*scal*scal;
  scal=(NWTN2?2*scal*scal*scal:scal); Real_t const_nwtn3=3*scal*scal;

  VEC rinv;
  #if NWTN0
  rinv=rinv_single_intrin(r2);
  #else
  rinv=rinv_approx_intrin(r2);
  #endif

  #if NWTN1
  rinv_newton_intrin(rinv,r2,const_nwtn1);
  #endif
  #if NWTN2
  rinv_newton_intrin(rinv,r2,const_nwtn2);
  #endif
  #if NWTN3
  rinv_newton_intrin(rinv,r2,const_nwtn3);
  #endif

  return rinv;

  #undef NWTN0
  #undef NWTN1
  #undef NWTN2
  #undef NWTN3
}

}

#endif //_PVFMM_INTRIN_WRAPPER_HPP_
