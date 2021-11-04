/**
 * \file mortonid.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains definition of the class MortonId.
 */

#include <vector>
#include <ostream>
#include <stdint.h>

#include <pvfmm_common.hpp>

#ifndef _PVFMM_MORTONID_HPP_
#define _PVFMM_MORTONID_HPP_

namespace pvfmm{

#ifndef PVFMM_MAX_DEPTH
#define PVFMM_MAX_DEPTH 30
#endif

#if PVFMM_MAX_DEPTH < 7
#define PVFMM_MID_UINT_T uint8_t
#define  PVFMM_MID_INT_T  int8_t
#elif PVFMM_MAX_DEPTH < 15
#define PVFMM_MID_UINT_T uint16_t
#define  PVFMM_MID_INT_T  int16_t
#elif PVFMM_MAX_DEPTH < 31
#define PVFMM_MID_UINT_T uint32_t
#define  PVFMM_MID_INT_T  int32_t
#elif PVFMM_MAX_DEPTH < 63
#define PVFMM_MID_UINT_T uint64_t
#define  PVFMM_MID_INT_T  int64_t
#endif

class MortonId{

 public:

  MortonId();

  MortonId(MortonId m, uint8_t depth);

  template <class T>
  MortonId(T x_f,T y_f, T z_f, uint8_t depth=PVFMM_MAX_DEPTH);

  template <class T>
  MortonId(T* coord, uint8_t depth=PVFMM_MAX_DEPTH);

  unsigned int GetDepth() const;

  template <class T>
  void GetCoord(T* coord);

  MortonId NextId() const;

  MortonId getAncestor(uint8_t ancestor_level) const;

  /**
   * \brief Returns the deepest first descendant.
   */
  MortonId getDFD(uint8_t level=PVFMM_MAX_DEPTH) const;

  void NbrList(std::vector<MortonId>& nbrs,uint8_t level, int periodic) const;

  std::vector<MortonId> Children() const;

  int operator<(const MortonId& m) const;

  int operator>(const MortonId& m) const;

  int operator==(const MortonId& m) const;

  int operator!=(const MortonId& m) const;

  int operator<=(const MortonId& m) const;

  int operator>=(const MortonId& m) const;

  int isAncestor(MortonId const & other) const;

  friend std::ostream& operator<<(std::ostream& out, const MortonId & mid);

 private:

  PVFMM_MID_UINT_T x,y,z;
  uint8_t depth;

};

}//end namespace

#include <mortonid.txx>

#endif //_PVFMM_MORTONID_HPP_
