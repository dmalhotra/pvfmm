/**
 * \file profile.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief Backward-compatibility stub.
 *
 * pvfmm::Profile and pvfmm::GetSctlComm have both been retired. Profiling
 * now goes through sctl::Profile directly, and pvfmm classes hold an
 * sctl::Comm member so no MPI_Comm → sctl::Comm conversion helper is needed.
 *
 * Preserved (with #include <sctl.hpp> for convenience) so the many
 * `#include <profile.hpp>` lines scattered through pvfmm continue to compile
 * without source churn — callers transitively pick up sctl::Profile through
 * the sctl.hpp include below.
 */

#ifndef _PVFMM_PROFILE_HPP_
#define _PVFMM_PROFILE_HPP_

#include <pvfmm_common.hpp>
#include <sctl.hpp>

#endif //_PVFMM_PROFILE_HPP_
