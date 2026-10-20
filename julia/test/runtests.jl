# Copyright (C) 2010-2026 The Simons Foundation, Inc. and the PVFMM authors (see NOTICE)
# Author: Xuanzhao Gao
# SPDX-License-Identifier: LGPL-3.0-only

using Test
using PVFMM

@testset "PVFMM Julia interface parity" begin
    @test isdefined(PVFMM, :FMMKernel)
    @test isdefined(PVFMM, :FMMVolumeContext)
    @test isdefined(PVFMM, :FMMParticleContext)
    @test isdefined(PVFMM, :FMMVolumeTree)
    @test isdefined(PVFMM, :nodes_to_coeff)

    expected_exports = (
        :FMMKernel,
        :FMMVolumeContext,
        :FMMParticleContext,
        :FMMVolumeTree,
        :nodes_to_coeff,
    )
    for name in expected_exports
        @test name in names(PVFMM)
    end
end

include("reference_comparison.jl")
