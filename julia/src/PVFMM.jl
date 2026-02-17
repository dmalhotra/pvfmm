module PVFMM

using Libdl

export FMMKernel
export FMMVolumeContext
export FMMParticleContext
export FMMVolumeTree
export nodes_to_coeff
export from_function
export from_coefficients
export evaluate
export leaf_count
export get_leaf_coordinates
export get_coefficients
export get_values

@enum FMMKernel begin
    LaplacePotential = 0
    LaplaceGradient = 1
    StokesPressure = 2
    StokesVelocity = 3
    StokesVelocityGrad = 4
    BiotSavartPotential = 5
end

const KERNEL_DIMS = Dict(
    LaplacePotential => (1, 1),
    LaplaceGradient => (1, 3),
    StokesPressure => (3, 1),
    StokesVelocity => (3, 3),
    StokesVelocityGrad => (3, 9),
    BiotSavartPotential => (3, 3),
)

const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function _resolve_library_path()
    if haskey(ENV, "PVFMM")
        root = ENV["PVFMM"]
        if isfile(root)
            return root
        end
        for candidate in ("libpvfmm.so", "libpvfmm.dylib", "libpvfmm.dll")
            path = joinpath(root, candidate)
            if isfile(path)
                return path
            end
        end
        throw(ArgumentError("PVFMM was set but no PVFMM library was found in $root"))
    end

    found = Libdl.find_library(["pvfmm"])
    found == "" && throw(ArgumentError("Failed to find libpvfmm. Set ENV[\"PVFMM\"] to the install/build path."))
    return found
end

function _libpvfmm()
    if _LIB_HANDLE[] == C_NULL
        _LIB_HANDLE[] = Libdl.dlopen(_resolve_library_path())
    end
    return _LIB_HANDLE[]
end

_suffix(::Type{Float64}) = "D"
_suffix(::Type{Float32}) = "F"
_suffix(::Type{T}) where {T} = throw(ArgumentError("Unsupported element type $T. Use Float32 or Float64."))

_check_multipole_order(m::Integer) = (m > 0 && iseven(m)) || throw(ArgumentError("multipole_order must be positive and even"))

function _comm_kind(comm)
    if comm isa Ptr{Cvoid}
        return (:ptr, comm)
    end
    if comm isa Integer
        return (:int, Cint(comm))
    end
    if hasproperty(comm, :val)
        val = getproperty(comm, :val)
        if val isa Integer
            return (:int, Cint(val))
        elseif val isa Ptr{Cvoid}
            return (:ptr, val)
        end
    end
    throw(ArgumentError("Unsupported MPI communicator type $(typeof(comm)). Pass Cint, Ptr{Cvoid}, or an object with integer/ptr field `val`."))
end

mutable struct FMMVolumeContext{T<:AbstractFloat}
    ptr::Ptr{Cvoid}
    kernel::FMMKernel
end

mutable struct FMMParticleContext{T<:AbstractFloat}
    ptr::Ptr{Cvoid}
    kernel::FMMKernel
end

mutable struct FMMVolumeTree{T<:AbstractFloat}
    ptr::Ptr{Cvoid}
    cheb_deg::Int
    n_cheb::Int
    n_coeff::Int
    data_dim::Int
    n_trg::Int
    used_kernel::Union{Nothing,FMMKernel}
end

function _destroy_ptr!(ctx_ref::Ref{Ptr{Cvoid}}, T::Type{<:AbstractFloat}, base::String)
    ctx_ref[] == C_NULL && return nothing
    symbol = Symbol(base * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), symbol)
    ccall(fp, Cvoid, (Ref{Ptr{Cvoid}},), ctx_ref)
    return nothing
end

function FMMVolumeContext(
    multipole_order::Integer,
    chebyshev_degree::Integer,
    kernel::FMMKernel,
    comm;
    T::Type{<:AbstractFloat}=Float64,
)
    _check_multipole_order(multipole_order)
    sym = Symbol("PVFMMCreateVolumeFMM" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    kind, comm_arg = _comm_kind(comm)
    ptr = if kind === :int
        ccall(fp, Ptr{Cvoid}, (Cint, Cint, Cuint, Cint), Cint(multipole_order), Cint(chebyshev_degree), Cuint(kernel), comm_arg)
    else
        ccall(fp, Ptr{Cvoid}, (Cint, Cint, Cuint, Ptr{Cvoid}), Cint(multipole_order), Cint(chebyshev_degree), Cuint(kernel), comm_arg)
    end
    ptr == C_NULL && error("PVFMMCreateVolumeFMM returned NULL")
    ctx = FMMVolumeContext{T}(ptr, kernel)
    finalizer(ctx) do obj
        ref = Ref(obj.ptr)
        _destroy_ptr!(ref, T, "PVFMMDestroyVolumeFMM")
        obj.ptr = C_NULL
    end
    return ctx
end

function FMMParticleContext(
    box_size::Real,
    max_points::Integer,
    multipole_order::Integer,
    kernel::FMMKernel,
    comm=nothing;
    T::Type{<:AbstractFloat}=Float64,
)
    _check_multipole_order(multipole_order)
    sym = comm === nothing ? Symbol("PVFMMCreateContext" * _suffix(T) * "World") : Symbol("PVFMMCreateContext" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    ptr = if T === Float64
        if comm === nothing
            ccall(fp, Ptr{Cvoid}, (Cdouble, Cint, Cint, Cuint), Cdouble(box_size), Cint(max_points), Cint(multipole_order), Cuint(kernel))
        else
            kind, comm_arg = _comm_kind(comm)
            if kind === :int
                ccall(fp, Ptr{Cvoid}, (Cdouble, Cint, Cint, Cuint, Cint), Cdouble(box_size), Cint(max_points), Cint(multipole_order), Cuint(kernel), comm_arg)
            else
                ccall(fp, Ptr{Cvoid}, (Cdouble, Cint, Cint, Cuint, Ptr{Cvoid}), Cdouble(box_size), Cint(max_points), Cint(multipole_order), Cuint(kernel), comm_arg)
            end
        end
    else
        if comm === nothing
            ccall(fp, Ptr{Cvoid}, (Cfloat, Cint, Cint, Cuint), Cfloat(box_size), Cint(max_points), Cint(multipole_order), Cuint(kernel))
        else
            kind, comm_arg = _comm_kind(comm)
            if kind === :int
                ccall(fp, Ptr{Cvoid}, (Cfloat, Cint, Cint, Cuint, Cint), Cfloat(box_size), Cint(max_points), Cint(multipole_order), Cuint(kernel), comm_arg)
            else
                ccall(fp, Ptr{Cvoid}, (Cfloat, Cint, Cint, Cuint, Ptr{Cvoid}), Cfloat(box_size), Cint(max_points), Cint(multipole_order), Cuint(kernel), comm_arg)
            end
        end
    end
    ptr == C_NULL && error("PVFMMCreateContext returned NULL")
    ctx = FMMParticleContext{T}(ptr, kernel)
    finalizer(ctx) do obj
        ref = Ref(obj.ptr)
        _destroy_ptr!(ref, T, "PVFMMDestroyContext")
        obj.ptr = C_NULL
    end
    return ctx
end

function evaluate(
    ctx::FMMParticleContext{T},
    src_pos::AbstractVector{T},
    sl_den::Union{Nothing,AbstractVector{T}},
    dl_den::Union{Nothing,AbstractVector{T}},
    trg_pos::AbstractVector{T};
    setup::Bool=true,
) where {T<:AbstractFloat}
    length(src_pos) % 3 == 0 || throw(ArgumentError("Source positions length must be a multiple of 3"))
    n_src = length(src_pos) ÷ 3
    if sl_den !== nothing
        length(sl_den) == length(src_pos) || throw(ArgumentError("Single-layer density length must match source positions length"))
    end
    if dl_den !== nothing
        length(dl_den) == 2 * length(src_pos) || throw(ArgumentError("Double-layer density length must be 2x source positions length"))
    end

    length(trg_pos) % 3 == 0 || throw(ArgumentError("Target positions length must be a multiple of 3"))
    n_trg = length(trg_pos) ÷ 3
    trg_val = Vector{T}(undef, length(trg_pos))

    sym = Symbol("PVFMMEval" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    ccall(
        fp,
        Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Clong, Ptr{Cvoid}, Ptr{Cvoid}, Clong, Ptr{Cvoid}, Cint),
        pointer(src_pos),
        sl_den === nothing ? C_NULL : Ptr{Cvoid}(pointer(sl_den)),
        dl_den === nothing ? C_NULL : Ptr{Cvoid}(pointer(dl_den)),
        Clong(n_src),
        pointer(trg_pos),
        pointer(trg_val),
        Clong(n_trg),
        ctx.ptr,
        Cint(setup),
    )
    return trg_val
end

function nodes_to_coeff(
    N_leaf::Integer,
    cheb_deg::Integer,
    dof::Integer,
    node_val::AbstractVector{T},
) where {T<:AbstractFloat}
    n_coeff = (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) ÷ 6
    coeff = Vector{T}(undef, n_coeff * N_leaf * dof)
    sym = Symbol("PVFMMNodes2Coeff" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    ccall(fp, Cvoid, (Ptr{Cvoid}, Clong, Cint, Cint, Ptr{Cvoid}), pointer(coeff), Clong(N_leaf), Cint(cheb_deg), Cint(dof), pointer(node_val))
    return coeff
end

function _coeff_to_nodes(
    N_leaf::Integer,
    cheb_deg::Integer,
    dof::Integer,
    coeff::AbstractVector{T},
) where {T<:AbstractFloat}
    n_cheb = (cheb_deg + 1)^3
    node_val = Vector{T}(undef, n_cheb * N_leaf * dof)
    sym = Symbol("PVFMMCoeff2Nodes" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    ccall(fp, Cvoid, (Ptr{Cvoid}, Clong, Cint, Cint, Ptr{Cvoid}), pointer(node_val), Clong(N_leaf), Cint(cheb_deg), Cint(dof), pointer(coeff))
    return node_val
end

function from_function(
    ::Type{FMMVolumeTree{T}},
    cheb_deg::Integer,
    data_dim::Integer,
    fn_ptr::Ptr{Cvoid},
    fn_ctx::Ptr{Cvoid},
    trg_coord::AbstractVector{T},
    comm,
    tol::Real,
    max_pts::Integer,
    periodic::Bool,
    init_depth::Integer,
) where {T<:AbstractFloat}
    length(trg_coord) % 3 == 0 || throw(ArgumentError("Target coordinates length must be a multiple of 3"))
    n_trg = length(trg_coord) ÷ 3
    sym = Symbol("PVFMMCreateVolumeTree" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    kind, comm_arg = _comm_kind(comm)
    ptr = if T === Float64
        if kind === :int
            ccall(fp, Ptr{Cvoid}, (Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Clong, Cint, Cdouble, Cint, Bool, Cint), Cint(cheb_deg), Cint(data_dim), fn_ptr, fn_ctx, pointer(trg_coord), Clong(n_trg), comm_arg, Cdouble(tol), Cint(max_pts), periodic, Cint(init_depth))
        else
            ccall(fp, Ptr{Cvoid}, (Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Clong, Ptr{Cvoid}, Cdouble, Cint, Bool, Cint), Cint(cheb_deg), Cint(data_dim), fn_ptr, fn_ctx, pointer(trg_coord), Clong(n_trg), comm_arg, Cdouble(tol), Cint(max_pts), periodic, Cint(init_depth))
        end
    else
        if kind === :int
            ccall(fp, Ptr{Cvoid}, (Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Clong, Cint, Cfloat, Cint, Bool, Cint), Cint(cheb_deg), Cint(data_dim), fn_ptr, fn_ctx, pointer(trg_coord), Clong(n_trg), comm_arg, Cfloat(tol), Cint(max_pts), periodic, Cint(init_depth))
        else
            ccall(fp, Ptr{Cvoid}, (Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Clong, Ptr{Cvoid}, Cfloat, Cint, Bool, Cint), Cint(cheb_deg), Cint(data_dim), fn_ptr, fn_ctx, pointer(trg_coord), Clong(n_trg), comm_arg, Cfloat(tol), Cint(max_pts), periodic, Cint(init_depth))
        end
    end
    ptr == C_NULL && error("PVFMMCreateVolumeTree returned NULL")
    tree = FMMVolumeTree{T}(ptr, cheb_deg, (cheb_deg + 1)^3, (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) ÷ 6, data_dim, n_trg, nothing)
    finalizer(tree) do obj
        ref = Ref(obj.ptr)
        _destroy_ptr!(ref, T, "PVFMMDestroyVolumeTree")
        obj.ptr = C_NULL
    end
    return tree
end

function from_coefficients(
    ::Type{FMMVolumeTree{T}},
    cheb_deg::Integer,
    data_dim::Integer,
    leaf_coord::AbstractVector{T},
    fn_coeff::AbstractVector{T},
    trg_coord::Union{Nothing,AbstractVector{T}},
    comm,
    periodic::Bool,
) where {T<:AbstractFloat}
    length(leaf_coord) % 3 == 0 || throw(ArgumentError("Leaf coordinates length must be a multiple of 3"))
    N_leaf = length(leaf_coord) ÷ 3
    coeff_size = N_leaf * data_dim * (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) ÷ 6
    length(fn_coeff) == coeff_size || throw(ArgumentError("Function coefficients have wrong length, expected $coeff_size"))
    n_trg = trg_coord === nothing ? 0 : (length(trg_coord) ÷ 3)
    trg_coord !== nothing && (length(trg_coord) % 3 == 0 || throw(ArgumentError("Target coordinates length must be a multiple of 3")))
    trg_buf = trg_coord === nothing ? Vector{T}(undef, 0) : trg_coord

    sym = Symbol("PVFMMCreateVolumeTreeFromCoeff" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    kind, comm_arg = _comm_kind(comm)
    ptr = if kind === :int
        ccall(fp, Ptr{Cvoid}, (Clong, Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Clong, Cint, Bool), Clong(N_leaf), Cint(cheb_deg), Cint(data_dim), pointer(leaf_coord), pointer(fn_coeff), pointer(trg_buf), Clong(n_trg), comm_arg, periodic)
    else
        ccall(fp, Ptr{Cvoid}, (Clong, Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Clong, Ptr{Cvoid}, Bool), Clong(N_leaf), Cint(cheb_deg), Cint(data_dim), pointer(leaf_coord), pointer(fn_coeff), pointer(trg_buf), Clong(n_trg), comm_arg, periodic)
    end
    ptr == C_NULL && error("PVFMMCreateVolumeTreeFromCoeff returned NULL")
    tree = FMMVolumeTree{T}(ptr, cheb_deg, (cheb_deg + 1)^3, (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) ÷ 6, data_dim, n_trg, nothing)
    finalizer(tree) do obj
        ref = Ref(obj.ptr)
        _destroy_ptr!(ref, T, "PVFMMDestroyVolumeTree")
        obj.ptr = C_NULL
    end
    return tree
end

function evaluate(tree::FMMVolumeTree{T}, fmm::FMMVolumeContext{T}, loc_size::Integer) where {T<:AbstractFloat}
    _, kdim1 = KERNEL_DIMS[fmm.kernel]
    trg_val = Vector{T}(undef, tree.n_trg * kdim1)
    sym = Symbol("PVFMMEvaluateVolumeFMM" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    ccall(fp, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Clong), pointer(trg_val), tree.ptr, fmm.ptr, Clong(loc_size))
    tree.used_kernel = fmm.kernel
    return trg_val
end

function leaf_count(tree::FMMVolumeTree{T}) where {T<:AbstractFloat}
    sym = Symbol("PVFMMGetLeafCount" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    return Int(ccall(fp, Clong, (Ptr{Cvoid},), tree.ptr))
end

function get_leaf_coordinates(tree::FMMVolumeTree{T}) where {T<:AbstractFloat}
    n_leaf = leaf_count(tree)
    leaf_coord = Vector{T}(undef, 3 * n_leaf)
    sym = Symbol("PVFMMGetLeafCoord" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    ccall(fp, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), pointer(leaf_coord), tree.ptr)
    return leaf_coord
end

function get_coefficients(tree::FMMVolumeTree{T}) where {T<:AbstractFloat}
    tree.used_kernel === nothing && throw(ArgumentError("Cannot get coefficients of an unevaluated tree"))
    n_leaf = leaf_count(tree)
    _, kdim1 = KERNEL_DIMS[tree.used_kernel]
    coeff = Vector{T}(undef, n_leaf * tree.n_coeff * kdim1)
    sym = Symbol("PVFMMGetPotentialCoeff" * _suffix(T))
    fp = Libdl.dlsym(_libpvfmm(), sym)
    ccall(fp, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), pointer(coeff), tree.ptr)
    return coeff
end

function get_values(tree::FMMVolumeTree{T}) where {T<:AbstractFloat}
    coeff = get_coefficients(tree)
    n_leaf = leaf_count(tree)
    _kdim0, kdim1 = KERNEL_DIMS[tree.used_kernel]
    return _coeff_to_nodes(n_leaf, tree.cheb_deg, kdim1, coeff)
end

end
