/**
 * \file mortonid.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief pvfmm::MortonId is an alias for sctl::Morton<3>.
 *
 * The Morton-code arithmetic lives in sctl::Morton<3> and call sites use it
 * directly. The only pvfmm-specific bit left is the neighbor-list helper
 * below: sctl::Morton::NbrList returns all 3^DIM entries (flagging
 * out-of-domain neighbors with INVALID_DEPTH), whereas pvfmm's tree code
 * expects only the in-domain neighbors unless periodic. Build with
 * -DSCTL_MAX_DEPTH=30 so sctl::Morton<3> matches PVFMM_MAX_DEPTH.
 */

#include <vector>

#include <pvfmm_common.hpp>   // pulls in sctl.hpp -> sctl::Morton

#ifndef _PVFMM_MORTONID_HPP_
#define _PVFMM_MORTONID_HPP_

namespace pvfmm{

#ifndef PVFMM_MAX_DEPTH
#define PVFMM_MAX_DEPTH 30
#endif

using MortonId = sctl::Morton<3>;

// In-domain neighbor list. `periodic` is isotropic (all axes or none).
inline void NbrList(const MortonId& m, std::vector<MortonId>& nbrs, uint8_t level, int periodic){
  nbrs.clear();
  const auto arr = m.NbrList(level, periodic ? sctl::Periodicity::XYZ : sctl::Periodicity::NONE);
  nbrs.reserve(arr.size());
  for(const auto& n : arr){
    if(periodic || n.Depth()!=MortonId::INVALID_DEPTH) nbrs.push_back(n);
  }
}

}//end namespace

#endif //_PVFMM_MORTONID_HPP_
