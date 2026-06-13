/**
 * \file mortonid.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief pvfmm::MortonId is a thin compatibility shim over sctl::Morton<3>.
 *
 * The Morton-code arithmetic now lives in sctl::Morton<3>; this class only
 * adds the constructor signatures and method names that pvfmm's tree code
 * uses (MortonId(x,y,z,depth), GetDepth/NextId/getAncestor/getDFD/GetCoord,
 * Children/NbrList returning std::vector). Build with -DSCTL_MAX_DEPTH=30 so
 * sctl::Morton<3> matches PVFMM_MAX_DEPTH (the wide multi-word code path).
 */

#include <vector>
#include <ostream>
#include <stdint.h>

#include <pvfmm_common.hpp>   // pulls in sctl.hpp -> sctl::Morton

#ifndef _PVFMM_MORTONID_HPP_
#define _PVFMM_MORTONID_HPP_

namespace pvfmm{

#ifndef PVFMM_MAX_DEPTH
#define PVFMM_MAX_DEPTH 30
#endif

class MortonId : public sctl::Morton<3> {
  typedef sctl::Morton<3> Base;

 public:

  MortonId() : Base() {}

  // Implicit conversion from the base type so the aliased methods below can
  // return MortonId from sctl::Morton<3>-returning calls.
  MortonId(const Base& m) : Base(m) {}

  // Truncate m to the given depth (matches the old "copy code, mask to depth").
  MortonId(MortonId m, uint8_t depth) : Base(m.Ancestor(depth)) {}

  template <class T>
  MortonId(T x_f, T y_f, T z_f, uint8_t depth=PVFMM_MAX_DEPTH) {
    const T coord[3] = {x_f, y_f, z_f};
    *static_cast<Base*>(this) = Base(sctl::Ptr2ConstItr<T>(&coord[0], 3), depth);
  }

  template <class T>
  MortonId(T* coord, uint8_t depth=PVFMM_MAX_DEPTH)
    : Base(sctl::Ptr2ConstItr<T>(coord, 3), depth) {}

  unsigned int GetDepth() const { return (unsigned int)this->Depth(); }

  template <class T>
  void GetCoord(T* coord) const { this->Coord(coord); }

  MortonId NextId() const { return MortonId(this->Next()); }

  MortonId getAncestor(uint8_t ancestor_level) const { return MortonId(this->Ancestor(ancestor_level)); }

  /** \brief Returns the deepest first descendant. */
  MortonId getDFD(uint8_t level=PVFMM_MAX_DEPTH) const { return MortonId(this->DFD(level)); }

  void NbrList(std::vector<MortonId>& nbrs, uint8_t level, int periodic) const {
    nbrs.clear();
    // pvfmm's `periodic` is isotropic (all axes or none) -> XYZ / NONE.
    const auto arr = this->Base::NbrList(level, periodic ? sctl::Periodicity::XYZ : sctl::Periodicity::NONE);
    nbrs.reserve(arr.size());
    for (const auto& m : arr) {
      if (periodic || m.Depth() != Base::INVALID_DEPTH) nbrs.push_back(MortonId(m));
    }
  }

  std::vector<MortonId> Children() const {
    const auto arr = this->Base::Children();
    std::vector<MortonId> child;
    child.reserve(arr.size());
    for (const auto& m : arr) child.push_back(MortonId(m));
    return child;
  }

  // Comparison operators, isAncestor and operator<< are inherited from
  // sctl::Morton<3> (a MortonId binds to a const sctl::Morton<3>& argument).
};

}//end namespace

#endif //_PVFMM_MORTONID_HPP_
