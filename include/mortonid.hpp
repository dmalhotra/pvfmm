/**
 * \file mortonid.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains definition of the class MortonId.
 */

#include <vector>
#include <stdint.h>

#include <pvfmm_common.hpp>

#ifndef _PVFMM_MORTONID_HPP_
#define _PVFMM_MORTONID_HPP_

namespace pvfmm{

#ifndef MAX_DEPTH
#define MAX_DEPTH 30
#endif

#if MAX_DEPTH < 7
#define UINT_T uint8_t
#define  INT_T  int8_t
#elif MAX_DEPTH < 15
#define UINT_T uint16_t
#define  INT_T  int16_t
#elif MAX_DEPTH < 31
#define UINT_T uint32_t
#define  INT_T  int32_t
#elif MAX_DEPTH < 63
#define UINT_T uint64_t
#define  INT_T  int64_t
#endif

class MortonId{

 public:

  MortonId();

  MortonId(MortonId m, uint8_t depth);

  template <class T>
  MortonId(T x_f,T y_f, T z_f, uint8_t depth=MAX_DEPTH);

  template <class T>
  MortonId(T* coord, uint8_t depth=MAX_DEPTH);

  unsigned int GetDepth() const;

  template <class T>
  void GetCoord(T* coord);

  MortonId NextId() const;

  MortonId getAncestor(uint8_t ancestor_level) const;

  /**
   * \brief Returns the deepest first descendant.
   */
  MortonId getDFD(uint8_t level=MAX_DEPTH) const;

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

  UINT_T x,y,z;
  uint8_t depth;

};

}//end namespace

#include <mortonid.txx>

#endif //_PVFMM_MORTONID_HPP_
