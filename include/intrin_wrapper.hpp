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

// #ifdef __clang__ // alternative function names for clang
// #define _mm_set_pd1 _mm_set1_pd
// #define _mm_store_pd1 _mm_store1_pd
// #endif

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
inline T cmplt_intrin(const T& a, const T& b){
  T r=0;
  uint8_t* r_=reinterpret_cast<uint8_t*>(&r);
  if(a<b) for(size_t i=0;i<sizeof(T);i++) r_[i] = ~(uint8_t)0;
  return r;
}

template <class T>
inline T and_intrin(const T& a, const T& b){
  T r=0;
  const uint8_t* a_=reinterpret_cast<const uint8_t*>(&a);
  const uint8_t* b_=reinterpret_cast<const uint8_t*>(&b);
  uint8_t*       r_=reinterpret_cast<      uint8_t*>(&r);
  for(size_t i=0;i<sizeof(T);i++) r_[i] = a_[i] & b_[i];
  return r;
}

template <class T>
inline T rsqrt_approx_intrin(const T& r2){
  if(r2!=0) return 1.0/pvfmm::sqrt<T>(r2);
  return 0;
}

template <class T, class Real_t>
inline void rsqrt_newton_intrin(T& rinv, const T& r2, const Real_t& nwtn_const){
  rinv=rinv*(nwtn_const-r2*rinv*rinv);
}

template <class T>
inline T rsqrt_single_intrin(const T& r2){
  if(r2!=0) return 1.0/pvfmm::sqrt<T>(r2);
  return 0;
}

template <class T>
inline T max_intrin(const T& a, const T& b){
  if(a>b) return a;
  else return b;
}

template <class T>
inline T min_intrin(const T& a, const T& b){
  if(a>b) return b;
  else return a;
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
  return _mm_set1_ps(a);
}

template <>
inline __m128d set_intrin(const double& a){
  return _mm_set1_pd(a);
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
  return _mm_set1_ps(a[0]);
}

template <>
inline __m128d bcast_intrin(double const* a){
  return _mm_load1_pd(a);
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
inline __m128 cmplt_intrin(const __m128& a, const __m128& b){
  return _mm_cmplt_ps(a,b);
}

template <>
inline __m128d cmplt_intrin(const __m128d& a, const __m128d& b){
  return _mm_cmplt_pd(a,b);
}

template <>
inline __m128 and_intrin(const __m128& a, const __m128& b){
  return _mm_and_ps(a,b);
}

template <>
inline __m128d and_intrin(const __m128d& a, const __m128d& b){
  return _mm_and_pd(a,b);
}

template <>
inline __m128 rsqrt_approx_intrin(const __m128& r2){
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
inline __m128d rsqrt_approx_intrin(const __m128d& r2){
  #define PD2PS(a) _mm_cvtpd_ps(a)
  #define PS2PD(a) _mm_cvtps_pd(a)
  return PS2PD(rsqrt_approx_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

template <>
inline void rsqrt_newton_intrin(__m128& rinv, const __m128& r2, const float& nwtn_const){
  #define VEC_INTRIN       __m128
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  // We do not compute the product with 0.5 and this needs to be adjusted later
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline void rsqrt_newton_intrin(__m128d& rinv, const __m128d& r2, const double& nwtn_const){
  #define VEC_INTRIN       __m128d
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  // We do not compute the product with 0.5 and this needs to be adjusted later
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline __m128 rsqrt_single_intrin(const __m128& r2){
  #define VEC_INTRIN __m128
  VEC_INTRIN rinv=rsqrt_approx_intrin(r2);
  rsqrt_newton_intrin(rinv,r2,(float)3.0);
  return rinv;
  #undef VEC_INTRIN
}

template <>
inline __m128d rsqrt_single_intrin(const __m128d& r2){
  #define PD2PS(a) _mm_cvtpd_ps(a)
  #define PS2PD(a) _mm_cvtps_pd(a)
  return PS2PD(rsqrt_single_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

template <>
inline __m128 max_intrin(const __m128& a, const __m128& b){
  return _mm_max_ps(a,b);
}

template <>
inline __m128d max_intrin(const __m128d& a, const __m128d& b){
  return _mm_max_pd(a,b);
}

template <>
inline __m128 min_intrin(const __m128& a, const __m128& b){
  return _mm_min_ps(a,b);
}

template <>
inline __m128d min_intrin(const __m128d& a, const __m128d& b){
  return _mm_min_pd(a,b);
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
  return _mm256_set1_ps(a);
}

template <>
inline __m256d set_intrin(const double& a){
  return _mm256_set1_pd(a);
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
inline __m256 cmplt_intrin(const __m256& a, const __m256& b){
  return _mm256_cmp_ps(a,b,_CMP_LT_OS);
}

template <>
inline __m256d cmplt_intrin(const __m256d& a, const __m256d& b){
  return _mm256_cmp_pd(a,b,_CMP_LT_OS);
}

template <>
inline __m256 and_intrin(const __m256& a, const __m256& b){
  return _mm256_and_ps(a,b);
}

template <>
inline __m256d and_intrin(const __m256d& a, const __m256d& b){
  return _mm256_and_pd(a,b);
}

template <>
inline __m256 rsqrt_approx_intrin(const __m256& r2){
  #define VEC_INTRIN          __m256
  #define RSQRT_INTRIN(a)     _mm256_rsqrt_ps(a)
  #define CMPEQ_INTRIN(a,b)   _mm256_cmp_ps(a,b,_CMP_EQ_OS)
  #define ANDNOT_INTRIN(a,b)  _mm256_andnot_ps(a,b)

  // Approx inverse square root which returns zero for r2=0
  return ANDNOT_INTRIN(CMPEQ_INTRIN(r2,zero_intrin<VEC_INTRIN>()),RSQRT_INTRIN(r2));

  #undef VEC_INTRIN
  #undef RSQRT_INTRIN
  #undef CMPEQ_INTRIN
  #undef ANDNOT_INTRIN
}

template <>
inline __m256d rsqrt_approx_intrin(const __m256d& r2){
  #define PD2PS(a) _mm256_cvtpd_ps(a)
  #define PS2PD(a) _mm256_cvtps_pd(a)
  return PS2PD(rsqrt_approx_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

template <>
inline void rsqrt_newton_intrin(__m256& rinv, const __m256& r2, const float& nwtn_const){
  #define VEC_INTRIN       __m256
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  // We do not compute the product with 0.5 and this needs to be adjusted later
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline void rsqrt_newton_intrin(__m256d& rinv, const __m256d& r2, const double& nwtn_const){
  #define VEC_INTRIN       __m256d
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  // We do not compute the product with 0.5 and this needs to be adjusted later
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline __m256 rsqrt_single_intrin(const __m256& r2){
  #define VEC_INTRIN __m256
  VEC_INTRIN rinv=rsqrt_approx_intrin(r2);
  rsqrt_newton_intrin(rinv,r2,(float)3.0);
  return rinv;
  #undef VEC_INTRIN
}

template <>
inline __m256d rsqrt_single_intrin(const __m256d& r2){
  #define PD2PS(a) _mm256_cvtpd_ps(a)
  #define PS2PD(a) _mm256_cvtps_pd(a)
  return PS2PD(rsqrt_single_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

template <>
inline __m256 max_intrin(const __m256& a, const __m256& b){
  return _mm256_max_ps(a,b);
}

template <>
inline __m256d max_intrin(const __m256d& a, const __m256d& b){
  return _mm256_max_pd(a,b);
}

template <>
inline __m256 min_intrin(const __m256& a, const __m256& b){
  return _mm256_min_ps(a,b);
}

template <>
inline __m256d min_intrin(const __m256d& a, const __m256d& b){
  return _mm256_min_pd(a,b);
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
inline VEC rsqrt_intrin0(VEC r2){
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
  rinv=rsqrt_single_intrin(r2);
  #else
  rinv=rsqrt_approx_intrin(r2);
  #endif

  #if NWTN1
  rsqrt_newton_intrin(rinv,r2,const_nwtn1);
  #endif
  #if NWTN2
  rsqrt_newton_intrin(rinv,r2,const_nwtn2);
  #endif
  #if NWTN3
  rsqrt_newton_intrin(rinv,r2,const_nwtn3);
  #endif

  return rinv;

  #undef NWTN0
  #undef NWTN1
  #undef NWTN2
  #undef NWTN3
}

template <class VEC, class Real_t>
inline VEC rsqrt_intrin1(VEC r2){
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
  rinv=rsqrt_single_intrin(r2);
  #else
  rinv=rsqrt_approx_intrin(r2);
  #endif

  #if NWTN1
  rsqrt_newton_intrin(rinv,r2,const_nwtn1);
  #endif
  #if NWTN2
  rsqrt_newton_intrin(rinv,r2,const_nwtn2);
  #endif
  #if NWTN3
  rsqrt_newton_intrin(rinv,r2,const_nwtn3);
  #endif

  return rinv;

  #undef NWTN0
  #undef NWTN1
  #undef NWTN2
  #undef NWTN3
}

template <class VEC, class Real_t>
inline VEC rsqrt_intrin2(VEC r2){
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
  rinv=rsqrt_single_intrin(r2);
  #else
  rinv=rsqrt_approx_intrin(r2);
  #endif

  #if NWTN1
  rsqrt_newton_intrin(rinv,r2,const_nwtn1);
  #endif
  #if NWTN2
  rsqrt_newton_intrin(rinv,r2,const_nwtn2);
  #endif
  #if NWTN3
  rsqrt_newton_intrin(rinv,r2,const_nwtn3);
  #endif

  return rinv;

  #undef NWTN0
  #undef NWTN1
  #undef NWTN2
  #undef NWTN3
}

template <class VEC, class Real_t>
inline VEC rsqrt_intrin3(VEC r2){
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
  rinv=rsqrt_single_intrin(r2);
  #else
  rinv=rsqrt_approx_intrin(r2);
  #endif

  #if NWTN1
  rsqrt_newton_intrin(rinv,r2,const_nwtn1);
  #endif
  #if NWTN2
  rsqrt_newton_intrin(rinv,r2,const_nwtn2);
  #endif
  #if NWTN3
  rsqrt_newton_intrin(rinv,r2,const_nwtn3);
  #endif

  return rinv;

  #undef NWTN0
  #undef NWTN1
  #undef NWTN2
  #undef NWTN3
}

}

#endif //_PVFMM_INTRIN_WRAPPER_HPP_
