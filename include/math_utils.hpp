/**
 * \file math_utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 7-16-2014
 * \brief This file contains wrappers for functions in math.h
 */

#include <cmath>
#include <ostream>

#ifndef _MATH_UTILS_
#define _MATH_UTILS_

namespace pvfmm{

template <class Real_t>
inline Real_t const_pi(){return 3.1415926535897932384626433832795028841;}

template <class Real_t>
inline Real_t const_e (){return 2.7182818284590452353602874713526624977;}

//template <class Real_t>
//inline std::ostream& operator<<(std::ostream& output, const Real_t q_);

template <class Real_t>
inline Real_t fabs(const Real_t f){return ::fabs(f);}

template <class Real_t>
inline Real_t sqrt(const Real_t a){return ::sqrt(a);}

template <class Real_t>
inline Real_t sin(const Real_t a){return ::sin(a);}

template <class Real_t>
inline Real_t cos(const Real_t a){return ::cos(a);}

template <class Real_t>
inline Real_t exp(const Real_t a){return ::exp(a);}

template <class Real_t>
inline Real_t log(const Real_t a){return ::log(a);}

template <class Real_t>
inline Real_t pow(const Real_t b, const Real_t e){return ::pow(b,e);}

}//end namespace



#ifdef PVFMM_QUAD_T

typedef PVFMM_QUAD_T QuadReal_t;

namespace pvfmm{
inline QuadReal_t atoquad(const char* str);
}

inline std::ostream& operator<<(std::ostream& output, const QuadReal_t q_);

#endif //PVFMM_QUAD_T

#endif //_MATH_UTILS_HPP_

