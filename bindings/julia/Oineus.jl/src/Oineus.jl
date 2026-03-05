module Oineus

using CxxWrap
using Libdl

function _candidate_paths()
    libname = "liboineus_julia." * Libdl.dlext
    return [
        get(ENV, "OINEUS_JULIA_LIBRARY", ""),
        joinpath(@__DIR__, "..", "lib", libname),
        joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia", libname),
        joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia", "Release", libname),
        joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia", "Debug", libname),
    ]
end

function _library_path()
    for path in _candidate_paths()
        if !isempty(path) && isfile(path)
            return path
        end
    end
    error(
        "Could not locate liboineus_julia. Set OINEUS_JULIA_LIBRARY " *
        "to the absolute path of the built library."
    )
end

@wrapmodule(() -> _library_path())

const _CxxLong = CxxWrap.CxxWrapCore.CxxLong
const _CxxLongVector = CxxWrap.StdLib.StdVector{_CxxLong}

_to_cxxlong_vector(vertices::AbstractVector{<:Integer}) =
    _CxxLongVector(_CxxLong.(collect(vertices)))

# Convenience constructors from native Julia vectors.
CombinatorialSimplex(vertices::AbstractVector{<:Integer}) =
    CombinatorialSimplex(_to_cxxlong_vector(vertices))

CombinatorialSimplex(id::Integer, vertices::AbstractVector{<:Integer}) =
    CombinatorialSimplex(Int(id), _to_cxxlong_vector(vertices))

make_simplex(vertices::AbstractVector{<:Integer}, value::Real) =
    make_simplex(_to_cxxlong_vector(vertices), Float64(value))

make_simplex_with_id(id::Integer, vertices::AbstractVector{<:Integer}, value::Real) =
    make_simplex_with_id(Int(id), _to_cxxlong_vector(vertices), Float64(value))

Simplex(vertices::AbstractVector{<:Integer}, value::Real) =
    make_simplex(vertices, value)

Simplex(id::Integer, vertices::AbstractVector{<:Integer}, value::Real) =
    make_simplex_with_id(id, vertices, value)

Filtration(cells::AbstractVector{<:Simplex}, negate::Bool=false, n_threads::Integer=1) =
    make_filtration(CxxWrap.StdLib.StdVector{Simplex}(Vector{Simplex}(cells)), negate, Int(n_threads))

function diagram_array(dgms, dim::Integer)
    pts = diagram_in_dimension(dgms, Int(dim))
    n = length(pts)
    out = Matrix{Float64}(undef, n, 2)
    @inbounds for i in 1:n
        p = pts[i]
        out[i, 1] = birth(p)
        out[i, 2] = death(p)
    end
    return out
end

function index_diagram_array(dgms, dim::Integer)
    pts = index_diagram_in_dimension(dgms, Int(dim))
    n = length(pts)
    out = Matrix{UInt64}(undef, n, 2)
    @inbounds for i in 1:n
        p = pts[i]
        out[i, 1] = UInt64(birth_index(p))
        out[i, 2] = UInt64(death_index(p))
    end
    return out
end

function __init__()
    @initcxx
end

end # module Oineus
