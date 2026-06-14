/**
 * \file mortonid.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief pvfmm::MortonId is an alias for sctl::Morton<3>.
 *
 * The Morton-code arithmetic lives in sctl::Morton<3> and call sites use it
 * directly, including NbrList (which returns all 3^DIM neighbors, flagging
 * out-of-domain ones with INVALID_DEPTH; tree code skips those unless
 * periodic). Build with -DSCTL_MAX_DEPTH=30 so sctl::Morton<3> matches
 * PVFMM_MAX_DEPTH.
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

}//end namespace

#endif //_PVFMM_MORTONID_HPP_
