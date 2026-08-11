"""
    TheEggman

A Julia port of [thewalrus](https://github.com/XanaduAI/thewalrus), Xanadu's library of matrix
functions for Gaussian quantum optics.

Currently implements the hafnian: [`hafnian`](@ref) and [`hafnian_repeated`](@ref).
"""
module TheEggman

using LinearAlgebra

export hafnian, hafnian_repeated

include("unrolled.jl")
include("dp.jl")
include("hafnian.jl")

# Emitting the unrolled kernels is the one genuinely slow piece of compilation in this package
# (~1s at UNROLL_MAX, growing steeply with degree). Doing it here spends that once at
# precompilation for the element types that matter, instead of stalling somebody's first call.
for T in (Float64, ComplexF64)
    for K in 2:2:UNROLL_MAX
        precompile(_haf_unrolled, (Matrix{T}, NTuple{K,Int}))
    end
    precompile(hafnian, (Matrix{T},))
    precompile(hafnian_repeated, (Matrix{T}, Vector{Int}))
end

end
