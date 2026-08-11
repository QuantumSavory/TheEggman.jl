"""
Fully unrolled hafnians for small total degree.

For a small enough matrix the sieve in `hafnian.jl` is the wrong tool: it pays for workspace
allocation, a binomial table and a mixed-radix odometer, all to amortise an asymptotic win that has
not kicked in yet. Below `UNROLL_MAX` it is beaten outright by the *definition* — the sum over
perfect matchings — provided that sum is emitted as branch-free straight-line code with every index
resolved at compile time.

That is what [`_haf_unrolled`](@ref) does, via a `@generated` function keyed on the total degree.

# Sharing subexpressions

Expanding `haf(pos) = Σₜ A[pos₁, posₜ] · haf(pos ∖ {pos₁, posₜ})` naively produces `(K-1)!!`
products — 10395 at `K = 12`, which is far too much code to hand to the compiler even though the
*result* is much smaller: the recursion only ever reaches subsets of the original index set, and
there are few of those. The generator therefore memoises on the remaining-index bitmask and emits
one named temporary per distinct subset, which collapses the emitted code to the size of the
underlying DAG rather than of its unfolded tree:

| K  | subsets emitted | multiplies | multiplies if unfolded |
|----|-----------------|------------|------------------------|
| 8  | 33              | 87         | 315                    |
| 10 | 88              | 317        | 3780                   |
| 12 | 232             | 1055       | 51975                  |

An unfolded tree would reach the same machine code — LLVM's CSE finds the same sharing, and both
forms were measured to run identically at `K ≤ 10` — but it makes the compiler rediscover the
structure from an exponentially larger input. Memoising in the generator keeps compile time
proportional to the DAG, which is what makes `K = 12` affordable at all.

# Why the cap is 12

`UNROLL_MAX` is set by compile time, not by run time; the unrolled kernel keeps winning well past
it. Measured on the development machine (one-time cost, per degree and element type, paid at
precompilation for the common types):

| K  | compile | run     | sieve (1 thread) |
|----|---------|---------|------------------|
| 12 | 1.1 s   | 0.8 µs  | 44 µs            |
| 14 | 5.5 s   | 2.9 µs  | 134 µs           |
| 16 | 48 s    | 11.0 µs | 338 µs           |

`K = 14` and `K = 16` are still 15–30× faster than the sieve, but a 48-second first call is not a
reasonable thing for a library to do. Raise `UNROLL_MAX` if you are willing to pay for it.
"""

"""
Largest total degree for which a fully unrolled kernel is emitted. See the discussion above — this
is a compile-time budget, not a crossover point.
"""
const UNROLL_MAX = 12

"""
    _pair_index(i, j, K)

Position of the pair `(i < j)` within the flattened upper triangle of a `K × K` matrix, which is
how [`_haf_unrolled`](@ref) addresses its gathered entries.
"""
_pair_index(i::Int, j::Int, K::Int) = (i - 1) * K - (i * (i + 1)) ÷ 2 + j

"""
    _unrolled_expansion(K) -> (statements, result_symbol)

Build the straight-line body of the degree-`K` hafnian expansion.

Subsets of the `K` positions are represented as bitmasks. Each distinct reachable subset gets one
assignment, emitted only after everything it depends on, so the returned statements can be spliced
in order. Entries are read from a tuple `S` laid out by [`_pair_index`](@ref).
"""
function _unrolled_expansion(K::Int)
    statements = Expr[]
    emitted = Dict{UInt32,Symbol}()

    function gen(mask::UInt32)
        sym = get(emitted, mask, nothing)
        sym === nothing || return sym

        # Always pair off the lowest remaining index; that is what keeps the set of reachable
        # subsets small enough to enumerate.
        i = trailing_zeros(mask) + 1
        rest = mask & ~(UInt32(1) << (i - 1))

        terms = Any[]
        r = rest
        while r != 0
            j = trailing_zeros(r) + 1
            r &= ~(UInt32(1) << (j - 1))
            sub = rest & ~(UInt32(1) << (j - 1))
            p = _pair_index(i, j, K)
            push!(terms, sub == 0 ? :(S[$p]) : :(S[$p] * $(gen(sub))))
        end

        sym = Symbol("h", string(mask; base = 16))
        push!(statements, :($sym = $(Expr(:call, :+, terms...))))
        emitted[mask] = sym
        return sym
    end

    top = gen(UInt32((1 << K) - 1))
    return statements, top
end

"""
    _unrolled_cost(K)

Number of multiplications the degree-`K` unrolled kernel performs.

Used by the callers in `hafnian.jl` to decide between this kernel and the sieve; counted by walking
the same memoised recursion as [`_unrolled_expansion`](@ref) without building any expressions.
"""
function _unrolled_cost(K::Int)
    seen = Set{UInt32}()
    muls = 0
    function walk(mask::UInt32)
        mask in seen && return
        push!(seen, mask)
        i = trailing_zeros(mask) + 1
        rest = mask & ~(UInt32(1) << (i - 1))
        r = rest
        while r != 0
            j = trailing_zeros(r) + 1
            r &= ~(UInt32(1) << (j - 1))
            sub = rest & ~(UInt32(1) << (j - 1))
            if sub != 0
                muls += 1
                walk(sub)
            end
        end
    end
    K >= 2 && walk(UInt32((1 << K) - 1))
    return muls
end

# Cost of every unrolled degree, resolved once at load time so the dispatch decision is a lookup.
const _UNROLL_COSTS = ntuple(k -> _unrolled_cost(2 * (k - 1)), UNROLL_MAX ÷ 2 + 1)

"""
    _unrolled_muls(K)

Multiplication count of the degree-`K` unrolled kernel, for `K` even and `≤ UNROLL_MAX`.
"""
@inline _unrolled_muls(K::Int) = @inbounds _UNROLL_COSTS[K÷2+1]

"""
    _haf_unrolled(A, idx::NTuple{K,Int})

Hafnian of `A[idx, idx]` as one branch-free expression, with all `K` indices resolved at compile
time.

`idx` may repeat an index, in which case this is the hafnian of the corresponding expanded matrix
and diagonal entries of `A` legitimately participate — the same convention as
[`hafnian_repeated`](@ref).

Being a direct evaluation of the definition, this is also the most accurate route available: there
is none of the cancellation between large signed terms that any sieve relies on.
"""
@inline _haf_unrolled(A::AbstractMatrix, ::Tuple{}) = one(eltype(A))
@inline _haf_unrolled(A::AbstractMatrix, idx::NTuple{2,Int}) = @inbounds A[idx[1], idx[2]]

@generated function _haf_unrolled(A::AbstractMatrix, idx::NTuple{K,Int}) where {K}
    isodd(K) && return :(zero(eltype(A)))
    # Gather the upper triangle once, so the expansion runs entirely on tuple elements.
    gathers = [:(@inbounds A[idx[$i], idx[$j]]) for i in 1:K-1 for j in i+1:K]
    statements, top = _unrolled_expansion(K)
    quote
        S = $(Expr(:tuple, gathers...))
        $(statements...)
        return $top
    end
end

@inline _index_tuple(idx::AbstractVector{Int}, ::Val{K}) where {K} =
    ntuple(i -> @inbounds(idx[i]), Val(K))

"""
    _haf_direct(A, idx) -> Number

Hafnian of `A[idx, idx]` for a runtime-length `idx`, dispatched into the matching
[`_haf_unrolled`](@ref) kernel.

Generated so that `UNROLL_MAX` stays the single place the supported degrees are declared. The
branches compare against compile-time constants, so the call that survives is a direct one rather
than a dynamic dispatch on `Val(length(idx))`. Throws if no kernel covers `length(idx)`.
"""
@generated function _haf_direct(A::AbstractMatrix, idx::AbstractVector{Int})
    branches = Expr(:block)
    for K in 0:2:UNROLL_MAX
        push!(branches.args, :(K == $K && return _haf_unrolled(A, _index_tuple(idx, Val($K)))))
    end
    quote
        K = length(idx)
        $branches
        throw(ArgumentError("degree $K has no unrolled kernel (UNROLL_MAX = $(UNROLL_MAX))"))
    end
end
