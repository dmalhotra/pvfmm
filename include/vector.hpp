/**
 * \file vector.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief pvfmm::Vector is an alias for sctl::Vector<T>.
 *
 * Call sites use sctl::Vector directly. The only historical conveniences kept
 * (as free helpers) are Resize() — resize-if-needed without preserving
 * contents — and VecBegin(), a null-safe raw-pointer view for terminal
 * MPI/memcpy/device use.
 */

#include <vector>
#include <cstdlib>
#include <stdint.h>

#include <pvfmm_common.hpp>
#include <sctl.hpp>

#ifndef _PVFMM_VECTOR_HPP_
#define _PVFMM_VECTOR_HPP_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{

template <class T> using Vector = sctl::Vector<T>;

// Resize-if-needed: matches the historical pvfmm::Vector::Resize (a no-op when
// the size is unchanged; otherwise reallocates, NOT preserving contents).
template <class T> inline void Resize(sctl::Vector<T>& v, size_t n){ if((size_t)v.Dim()!=n) v.ReInit((sctl::Long)n); }

// Null-safe raw-pointer view: NULL for empty vectors and for the
// dim>0/null-storage placeholder state (ReInit(n,NULL,false)) that
// FMM_Pts::CollectNodeData uses to request buffer space. Use only at
// terminal consumption points (MPI, memcpy, device copies); carry
// iterators (v.begin()) everywhere else.
template <class T>
T* VecBegin(sctl::Vector<T>& v){ sctl::Iterator<T> it=v.begin(); return (v.Dim()>0 && it!=sctl::NullIterator<T>() ? &it[0] : (T*)NULL); }
template <class T>
const T* VecBegin(const sctl::Vector<T>& v){ sctl::ConstIterator<T> it=v.begin(); return (v.Dim()>0 && it!=sctl::NullIterator<T>() ? &it[0] : (const T*)NULL); }

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_VECTOR_HPP_
