using Test
using LinearAlgebra
using Random

const _PVFMM_BUILD_DIR = normpath(joinpath(@__DIR__, "..", "..", "build"))
const _PVFMM_DYLIB = joinpath(_PVFMM_BUILD_DIR, "libpvfmm.dylib")
ENV["PVFMM"] = _PVFMM_BUILD_DIR

function _aos_flat(coords::AbstractMatrix{T}) where {T}
    n = size(coords, 2)
    out = Vector{T}(undef, 3 * n)
    for i in 1:n
        out[3i - 2] = coords[1, i]
        out[3i - 1] = coords[2, i]
        out[3i] = coords[3, i]
    end
    return out
end

function _best_scale(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:Real}
    denom = dot(a, a)
    denom == 0 && return one(T)
    return dot(a, b) / denom
end

function _assert_close_up_to_scale(a::AbstractVector{Float64}, b::AbstractVector{Float64}; atol=5e-7, rtol=5e-2)
    s = _best_scale(a, b)
    @test isfinite(s)
    @test isapprox(s .* a, b; atol=atol, rtol=rtol)
end

function _direct_laplace_potential_3d(sources::Matrix{Float64}, charges::Vector{Float64}, targets::Matrix{Float64})
    nsrc = size(sources, 2)
    ntrg = size(targets, 2)
    pot = zeros(ntrg)
    for t in 1:ntrg
        xt = targets[:, t]
        for s in 1:nsrc
            r = xt - sources[:, s]
            r2 = dot(r, r)
            if r2 > 1e-24
                pot[t] += charges[s] / sqrt(r2)
            end
        end
    end
    return pot
end

function _direct_laplace_gradient_3d(sources::Matrix{Float64}, charges::Vector{Float64}, targets::Matrix{Float64})
    nsrc = size(sources, 2)
    ntrg = size(targets, 2)
    grad = zeros(3, ntrg)
    for t in 1:ntrg
        xt = targets[:, t]
        for s in 1:nsrc
            r = xt - sources[:, s]
            r2 = dot(r, r)
            if r2 > 1e-24
                invr3 = inv(r2 * sqrt(r2))
                grad[:, t] .+= (-charges[s] * invr3) .* r
            end
        end
    end
    return grad
end

function _direct_stokes_velocity_3d(sources::Matrix{Float64}, forces::Matrix{Float64}, targets::Matrix{Float64})
    nsrc = size(sources, 2)
    ntrg = size(targets, 2)
    vel = zeros(3, ntrg)
    for t in 1:ntrg
        xt = targets[:, t]
        for s in 1:nsrc
            r = xt - sources[:, s]
            r2 = dot(r, r)
            if r2 > 1e-24
                invr = inv(sqrt(r2))
                invr3 = inv(r2 * sqrt(r2))
                f = forces[:, s]
                vel[:, t] .+= invr .* f .+ (dot(r, f) * invr3) .* r
            end
        end
    end
    return vel
end

function _direct_stokes_pressure_3d(sources::Matrix{Float64}, forces::Matrix{Float64}, targets::Matrix{Float64})
    nsrc = size(sources, 2)
    ntrg = size(targets, 2)
    pre = zeros(ntrg)
    for t in 1:ntrg
        xt = targets[:, t]
        for s in 1:nsrc
            r = xt - sources[:, s]
            r2 = dot(r, r)
            if r2 > 1e-24
                invr3 = inv(r2 * sqrt(r2))
                pre[t] += dot(r, forces[:, s]) * invr3
            end
        end
    end
    return pre
end

@testset "Reference comparisons" begin
    Random.seed!(11)
    @test isfile(_PVFMM_DYLIB)

    nsrc = 40
    ntrg = 35
    # keep source/target sets separated to avoid near-singular evaluations
    sources = rand(3, nsrc)
    targets = rand(3, ntrg) .+ 1.5

    src_flat = _aos_flat(sources)
    trg_flat = _aos_flat(targets)

    charges = randn(nsrc)
    forces = randn(3, nsrc)

    @testset "Laplace potential" begin
        sl = zeros(3 * nsrc)
        sl[1:nsrc] .= charges
        ctx = PVFMM.FMMParticleContext(0.0, 50, 8, PVFMM.LaplacePotential)
        pv_out = PVFMM.evaluate(ctx, src_flat, sl, nothing, trg_flat; setup=true)
        ref = _direct_laplace_potential_3d(sources, charges, targets)
        _assert_close_up_to_scale(ref, pv_out[1:ntrg])
    end

    @testset "Laplace gradient" begin
        sl = zeros(3 * nsrc)
        sl[1:nsrc] .= charges
        ctx = PVFMM.FMMParticleContext(0.0, 50, 8, PVFMM.LaplaceGradient)
        pv_out = PVFMM.evaluate(ctx, src_flat, sl, nothing, trg_flat; setup=true)
        ref = _direct_laplace_gradient_3d(sources, charges, targets)
        _assert_close_up_to_scale(vec(ref), vec(reshape(pv_out, 3, ntrg)))
    end

    @testset "Stokes velocity" begin
        sl = _aos_flat(forces)
        ctx = PVFMM.FMMParticleContext(0.0, 50, 8, PVFMM.StokesVelocity)
        pv_out = PVFMM.evaluate(ctx, src_flat, sl, nothing, trg_flat; setup=true)
        ref = _direct_stokes_velocity_3d(sources, forces, targets)
        _assert_close_up_to_scale(vec(ref), vec(reshape(pv_out, 3, ntrg)))
    end

    @testset "Stokes pressure" begin
        sl = _aos_flat(forces)
        ctx = PVFMM.FMMParticleContext(0.0, 50, 8, PVFMM.StokesPressure)
        pv_out = PVFMM.evaluate(ctx, src_flat, sl, nothing, trg_flat; setup=true)
        ref = _direct_stokes_pressure_3d(sources, forces, targets)
        _assert_close_up_to_scale(ref, pv_out[1:ntrg])
    end
end
