/**
 * \file quad_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 7-16-2014
 * \brief This file contains definition of QuadReal_t.
 */

#ifndef _QUAD_UTILS_
#define _QUAD_UTILS_

#include <pvfmm_common.hpp>
#include <iostream>
#include <vector>

#if defined __INTEL_COMPILER
#define QuadReal_t _Quad
#elif defined __GNUC__
#define QuadReal_t __float128
#endif

#ifdef QuadReal_t

inline QuadReal_t atoquad(const char* str);

inline QuadReal_t fabs(const QuadReal_t& f);

inline QuadReal_t sqrt(const QuadReal_t& a);

inline QuadReal_t sin(const QuadReal_t& a);

inline QuadReal_t cos(const QuadReal_t& a);

inline std::ostream& operator<<(std::ostream& output, const QuadReal_t& q_);

template<>
inline QuadReal_t const_pi<QuadReal_t>(){
  return atoquad("3.1415926535897932384626433832795028841");
}

#include <quad_utils.txx>

#endif //QuadReal_t

#endif //_QUAD_UTILS_HPP_

