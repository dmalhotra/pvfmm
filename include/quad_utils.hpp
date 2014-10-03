/**
 * \file quad_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 7-16-2014
 * \brief This file contains definition of QuadReal_t.
 */

#include <cmath>
#include <ostream>

#ifndef _QUAD_UTILS_
#define _QUAD_UTILS_

#ifdef PVFMM_QUAD_T

typedef PVFMM_QUAD_T QuadReal_t;

inline std::ostream& operator<<(std::ostream& output, const QuadReal_t& q_);

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif

inline QuadReal_t atoquad(const char* str);

inline QuadReal_t fabs(const QuadReal_t& f);

inline QuadReal_t sqrt(const QuadReal_t& a);

inline QuadReal_t sin(const QuadReal_t& a);

inline QuadReal_t cos(const QuadReal_t& a);

inline QuadReal_t exp(const QuadReal_t& a);

inline QuadReal_t log(const QuadReal_t& a);

template<>
inline QuadReal_t const_pi<QuadReal_t>(){
  static QuadReal_t pi=atoquad("3.1415926535897932384626433832795028841");
  return pi;
}

template<>
inline QuadReal_t const_e<QuadReal_t>(){
  static QuadReal_t e =atoquad("2.7182818284590452353602874713526624977");
  return e;
}

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <quad_utils.txx>

#endif //PVFMM_QUAD_T

#endif //_QUAD_UTILS_HPP_

