"""
Hafnians by dynamic programming over reachable index subsets.

The recursion behind the unrolled kernels in `unrolled.jl` —

```
haf(S) = Σ_{t ∈ S∖{min S}} A[min S, t] · haf(S ∖ {min S, t}),    haf(∅) = 1
```

— is not restricted to small degrees. What limits `unrolled.jl` is emitting it as code, not
evaluating it. Evaluated at runtime with memoisation it stays useful far past `UNROLL_MAX`, and in
the range this package cares about it beats the sieve outright.

# Why there are so few subproblems

Because the recursion always pairs off the *smallest* remaining index, the subsets it reaches are
heavily constrained. Writing `p = min(S)` and `r = K - |S|` for the number of indices already
removed, a non-empty `S` is reachable exactly when

```
p - 1 ≤ r ≤ 2(p - 1)
```

— everything below `p` must already be gone (that costs `p-1`), and each removal step consumes one
minimum plus one partner (so `r` cannot exceed twice the number of minima available below `p`).
Counting those gives `F(K+1)` subsets including the empty one, a Fibonacci number: the state space
grows like `φ^K ≈ 1.618^K`, not `2^K`.

That is *worse* asymptotically than the sieve's `2^{K/2} K³ ≈ 1.414^K K³` — the two cross only
around `K ≈ 105` — but across the entire range that is actually computable, the sieve does 65–100×
more arithmetic:

| K              | 12  | 16   | 20    | 24     | 28      |
|----------------|-----|------|-------|--------|---------|
| DP transitions | 1076| 10226| 89665 | 748776 | 6052062 |
| sieve work     | 106k| 1.0M | 7.9M  | 54M    | 393M    |

# Making the transitions cheap

The subset structure depends only on `K`, never on the matrix, so all of it is precomputed once per
degree into a [`HafnianDPPlan`](@ref) and cached. States are numbered in order of increasing
`|S|`, so a single forward pass visits every subproblem after its children, and the transitions are
stored in CSR form. Evaluation is then a flat loop of `acc += P[pair[k]] * H[child[k]]` with no
hashing, no bitmask arithmetic and no recursion — which is what turns the operation-count advantage
into a real one.

The plan also keeps the state space compact enough to stay in cache: at `K = 20` the value array is
175 KB, where indexing 2^K masks directly would need 16 MB of scattered access and lose to the sieve
outright.

# Limits

Plans cost memory (`≈ 8` bytes per transition) and time to build, both growing like `φ^K`: 0.7 MB
and 3 ms at `K = 20`, 48 MB and 250 ms at `K = 28`. They are cached for the lifetime of the session,
so the build is amortised across calls, but `DP_MAX` caps the degree to keep a single plan from
running away. Above it, [`hafnian`](@ref) falls back to the sieve, whose memory is negligible.

Accuracy is that of the definition — the DP sums products of matrix entries with no cancellation
between large signed terms, so it is more accurate than either sieve variant.
"""

using Base.Threads: ReentrantLock

"""
Largest total degree for which a DP plan will be built. Set by plan size — `K = 28` needs 48 MB and
about 250 ms to construct, and the next even degree would need roughly 2.9× both.
"""
const DP_MAX = 28

"""
    HafnianDPPlan

Precomputed subset structure for degree `K`, independent of any matrix.

States are numbered `1:nstates` in order of increasing subset size, with the empty set first and the
full set last, so evaluating them in order visits every subproblem after its children. Transitions
are stored in CSR form: state `s` owns `starts[s]:starts[s+1]-1`, and entry `k` contributes
`P[pair[k]] * H[child[k]]`, where `P` holds the flattened upper triangle of the gathered matrix (see
[`_pair_index`](@ref)).
"""
struct HafnianDPPlan
    K::Int
    nstates::Int
    starts::Vector{Int32}
    child::Vector{Int32}
    pair::Vector{Int32}
end

"""
    _dp_counts(K) -> (nstates, ntransitions)

Size of the degree-`K` plan, without building it.

Counted from the reachability characterisation in this file's docstring: subsets with `min(S) = p`
and `r` indices removed number `binomial(K - p, K - r - 1)`, and each contributes `|S| - 1`
transitions. Needed before the plan exists, so that [`_prefer_dp`](@ref) can decide whether building
one is worth it.
"""
function _dp_counts(K::Int)
    nstates = 1                     # the empty set
    ntrans = 0
    for p in 1:K
        lo = p - 1
        hi = min(2 * (p - 1), K - 1)
        r = iseven(lo) ? lo : lo + 1
        while r <= hi
            k = K - r - 1           # |S| - 1, and also the number of transitions out of S
            c = binomial(K - p, k)
            nstates += c
            ntrans += c * k
            r += 2
        end
    end
    return nstates, ntrans
end

"""
    _build_dp_plan(K) -> HafnianDPPlan

Enumerate the reachable subsets for degree `K` and lay out their transitions.

Walks the recursion from the full set, sorts the subsets by size (then by mask, for determinism),
and rewrites every transition in terms of the resulting state numbering. The `Dict` used for that
rewrite lives only as long as the build.
"""
function _build_dp_plan(K::Int)
    full = UInt32((1 << K) - 1)
    reached = Set{UInt32}([full])
    stack = UInt32[full]
    while !isempty(stack)
        m = pop!(stack)
        m == 0 && continue
        i = trailing_zeros(m) + 1
        rest = m & ~(UInt32(1) << (i - 1))
        r = rest
        while r != 0
            j = trailing_zeros(r) + 1
            r &= ~(UInt32(1) << (j - 1))
            sub = rest & ~(UInt32(1) << (j - 1))
            if !(sub in reached)
                push!(reached, sub)
                push!(stack, sub)
            end
        end
    end

    masks = collect(reached)
    sort!(masks; by = m -> (count_ones(m), m))
    nstates = length(masks)
    number = Dict{UInt32,Int32}(m => Int32(s) for (s, m) in enumerate(masks))

    _, ntrans = _dp_counts(K)
    starts = Vector{Int32}(undef, nstates + 1)
    child = Vector{Int32}(undef, ntrans)
    pair = Vector{Int32}(undef, ntrans)

    n = 0
    starts[1] = 1
    for (s, m) in enumerate(masks)
        if m != 0
            i = trailing_zeros(m) + 1
            rest = m & ~(UInt32(1) << (i - 1))
            r = rest
            while r != 0
                j = trailing_zeros(r) + 1
                r &= ~(UInt32(1) << (j - 1))
                n += 1
                child[n] = number[rest&~(UInt32(1)<<(j-1))]
                pair[n] = Int32(_pair_index(i, j, K))
            end
        end
        starts[s+1] = Int32(n + 1)
    end

    return HafnianDPPlan(K, nstates, starts, child, pair)
end

const _DP_PLANS = Dict{Int,HafnianDPPlan}()
const _DP_PLAN_LOCK = ReentrantLock()

"""
    _dp_plan(K) -> HafnianDPPlan

The degree-`K` plan, built on first use and cached for the rest of the session.

Plans are pure functions of `K`, so caching them is what makes the DP worth using at the larger
degrees: the build is a few hundred milliseconds at `DP_MAX` but is paid once. The cache is never
evicted; see [`DP_MAX`](@ref) for the size it is allowed to reach.
"""
function _dp_plan(K::Int)
    lock(_DP_PLAN_LOCK) do
        get!(() -> _build_dp_plan(K), _DP_PLANS, K)
    end
end

"""
    _haf_dp(A, idx, plan) -> Number

Hafnian of `A[idx, idx]` by a forward pass over `plan`.

`idx` may repeat an index, in which case this is the hafnian of the corresponding expanded matrix,
exactly as for [`_haf_unrolled`](@ref). `length(idx)` must equal `plan.K`.
"""
function _haf_dp(A::AbstractMatrix{T}, idx::AbstractVector{<:Integer}, plan::HafnianDPPlan) where {T}
    K = plan.K
    length(idx) == K || throw(DimensionMismatch("idx has length $(length(idx)), plan expects $K"))

    # Gather the upper triangle once; every transition then reads a matrix entry by flat index.
    P = Vector{T}(undef, K * (K - 1) ÷ 2)
    @inbounds for i in 1:K-1, j in i+1:K
        P[_pair_index(i, j, K)] = A[idx[i], idx[j]]
    end

    H = Vector{T}(undef, plan.nstates)
    starts, child, pair = plan.starts, plan.child, plan.pair
    @inbounds begin
        H[1] = one(T)                  # the empty set
        for s in 2:plan.nstates
            acc = zero(T)
            for k in starts[s]:starts[s+1]-1
                acc += P[pair[k]] * H[child[k]]
            end
            H[s] = acc
        end
        return H[plan.nstates]         # the full set, sorted last
    end
end

"""
    _prefer_dp(K, sieve_work, nchunks) -> Bool

Decide between the DP and the sieve for total degree `K`.

`sieve_work` is the sieve's serial multiply-add count and `nchunks` the number of tasks it would
split into, so `sieve_work / nchunks` is what it actually costs in wall clock. The DP runs on one
thread, and a transition was measured at about twice the cost of a sieve work unit — both indirect
loads, but the sieve's reduction vectorises.

This matters most for repeated rows: enough repetition shrinks the sieve below the DP even at large
`K` (`rpt = [2,2,…]` at `N = 28` sieves), while for distinct rows the DP wins at every degree it
covers.
"""
function _prefer_dp(K::Int, sieve_work::Int, nchunks::Int)
    K <= DP_MAX || return false
    _, ntrans = _dp_counts(K)
    return 2 * ntrans < sieve_work ÷ max(nchunks, 1)
end
