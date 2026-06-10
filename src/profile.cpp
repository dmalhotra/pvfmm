/**
 * \file profile.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief Translation-unit placeholder.
 *
 * pvfmm::Profile and pvfmm::GetSctlComm have both been retired (all
 * profiling now flows through sctl::Profile directly, and pvfmm classes
 * hold an sctl::Comm member). This file is kept solely so the existing
 * Makefile build rule (`src/profile.lo`) still has a translation unit to
 * compile; once the build is regenerated this file can be deleted.
 */

#include <profile.hpp>
