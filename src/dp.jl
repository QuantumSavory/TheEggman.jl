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

Plans cost memory (4 bytes per transition) and time to build, both growing like `φ^K`: 0.4 MB and
3 ms at `K = 20`, 24 MB and 250 ms at `K = 28`. They are cached for the lifetime of the session, so
the build is amortised across calls, but `DP_MAX` caps the degree to keep a single plan from running
away. Above it, [`hafnian`](@ref) falls back to the sieve, whose memory is negligible.

Accuracy is that of the definition — the DP sums products of matrix entries with no cancellation
between large signed terms, so it is more accurate than either sieve variant.
"""

using Base.Threads: ReentrantLock, @spawn

"""
Largest total degree for which a DP plan will be built. Set by plan size — `K = 28` needs 24 MB and
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

Each transition is one packed `Int32`: the target state in the high bits and the matrix-entry index
in the low `_DP_PAIR_BITS`. The evaluation loop is memory-bound on this array — 24 MB of it at
`K = 28` — so packing the two indices together rather than storing parallel arrays halves its
traffic, which is worth far more than the shift and mask it costs to unpack.

`levels` records where each subset size begins, so `levels[L]:levels[L+1]-1` are the states of one
size. Every transition out of such a state lands two sizes below, so states within a level are
mutually independent and can be evaluated in parallel — see [`_haf_dp`](@ref).
"""
struct HafnianDPPlan
    K::Int
    nstates::Int
    starts::Vector{Int32}
    trans::Vector{Int32}
    levels::Vector{Int32}
end

# A packed transition is `(child << _DP_PAIR_BITS) | pair`. At `DP_MAX = 28` the pair index reaches
# 378 (9 bits) and the state index 514229 (20 bits), so both fit in an Int32 with room to spare;
# `_build_dp_plan` asserts this rather than trusting it.
const _DP_PAIR_BITS = 9
const _DP_PAIR_MASK = Int32(1 << _DP_PAIR_BITS - 1)

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
    npairs = K * (K - 1) ÷ 2
    npairs <= _DP_PAIR_MASK || error("degree $K needs more than $_DP_PAIR_BITS bits per pair index")
    nstates <= (typemax(Int32) >> _DP_PAIR_BITS) ||
        error("degree $K has too many states to pack into an Int32 transition")

    starts = Vector{Int32}(undef, nstates + 1)
    trans = Vector{Int32}(undef, ntrans)

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
                c = number[rest&~(UInt32(1)<<(j-1))]
                trans[n] = (c << _DP_PAIR_BITS) | Int32(_pair_index(i, j, K))
            end
        end
        starts[s+1] = Int32(n + 1)
    end

    levels = Int32[1]
    for s in 2:nstates
        count_ones(masks[s]) != count_ones(masks[s-1]) && push!(levels, Int32(s))
    end
    push!(levels, Int32(nstates + 1))

    return HafnianDPPlan(K, nstates, starts, trans, levels)
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

# Evaluate states `lo:hi`, all of which must have had their children evaluated already.
function _dp_states!(
    H::Vector{T},
    P::Vector{T},
    starts::Vector{Int32},
    trans::Vector{Int32},
    lo::Int,
    hi::Int,
) where {T}
    @inbounds for s in lo:hi
        acc = zero(T)
        for k in starts[s]:starts[s+1]-1
            t = trans[k]
            acc += P[t&_DP_PAIR_MASK] * H[t>>_DP_PAIR_BITS]
        end
        H[s] = acc
    end
    return nothing
end

"""
Transitions per task when a level is split. Measured: handing out chunks proportional to each
level's work beat splitting every level across all threads at every degree tested, because the
levels vary in size by three orders of magnitude and the small ones cannot repay a 12-way spawn.
"""
const _DP_CHUNK_WORK = 2_000

"""
Total transitions below which the whole DP runs serially. Under this, the levels are individually
too small for the spawn-and-join to pay for itself — at `K = 16` and `K = 18` every threaded variant
measured was slower than plain serial evaluation.
"""
const _DP_PARALLEL_MIN = 60_000

"""
    _dp_level_chunks(ntrans, nstates_in_level, nthreads) -> Int

How many tasks to split one level over.

Every state in a level has the same number of transitions, so equal-sized chunks are perfectly
balanced and only the total matters.
"""
@inline _dp_level_chunks(ntrans::Int, nstates::Int, nthreads::Int) =
    clamp(ntrans ÷ _DP_CHUNK_WORK, 1, min(nthreads, nstates))

"""
    _haf_dp(A, idx, plan; nthreads=1) -> Number

Hafnian of `A[idx, idx]` by a forward pass over `plan`.

`idx` may repeat an index, in which case this is the hafnian of the corresponding expanded matrix,
exactly as for [`_haf_unrolled`](@ref). `length(idx)` must equal `plan.K`.

States are evaluated one subset size at a time. Within a size they depend only on the size two
below, so each level is handed out in chunks across tasks and joined before the next begins. The
barriers are cheap because the levels are few (`K/2 + 1`) and the work is concentrated in the middle
ones — at `K = 28` the five largest levels carry 86% of the transitions and the levels too small to
be worth splitting carry under 1%.

Scaling tops out around 4× on 12 threads rather than approaching the thread count: the inner loop
does two indirect loads per multiply, so once several cores are running it is limited by memory
rather than arithmetic. Small degrees stay serial entirely (see [`_DP_PARALLEL_MIN`](@ref)).
"""
function _haf_dp(
    A::AbstractMatrix{T},
    idx::AbstractVector{<:Integer},
    plan::HafnianDPPlan;
    nthreads::Int = 1,
) where {T}
    K = plan.K
    length(idx) == K || throw(DimensionMismatch("idx has length $(length(idx)), plan expects $K"))

    # Gather the upper triangle once; every transition then reads a matrix entry by flat index.
    P = Vector{T}(undef, K * (K - 1) ÷ 2)
    @inbounds for i in 1:K-1, j in i+1:K
        P[_pair_index(i, j, K)] = A[idx[i], idx[j]]
    end

    H = Vector{T}(undef, plan.nstates)
    @inbounds H[1] = one(T)            # the empty set
    starts, trans, levels = plan.starts, plan.trans, plan.levels
    parallel = nthreads > 1 && length(trans) >= _DP_PARALLEL_MIN

    @inbounds for L in 2:length(levels)-1
        lo = Int(levels[L])
        hi = Int(levels[L+1]) - 1
        n = hi - lo + 1
        ntrans = Int(starts[hi+1]) - Int(starts[lo])
        nchunks = parallel ? _dp_level_chunks(ntrans, n, nthreads) : 1
        if nchunks == 1
            _dp_states!(H, P, starts, trans, lo, hi)
        else
            @sync for c in 1:nchunks
                clo = lo + div((c - 1) * n, nchunks)
                chi = lo + div(c * n, nchunks) - 1
                clo <= chi && @spawn _dp_states!(H, P, starts, trans, clo, chi)
            end
        end
    end
    return @inbounds H[plan.nstates]   # the full set, sorted last
end

"""
    _prefer_dp(K, sieve_work, nchunks) -> Bool

Decide between the DP and the sieve for total degree `K`.

`sieve_work` is the sieve's serial multiply-add count and `nchunks` the number of tasks it would
split into, so `sieve_work / nchunks` is what it actually costs in wall clock. The DP is compared
against that at its *serial* cost, with a transition weighted at about twice a sieve work unit —
both are indirect loads, but the sieve's reduction vectorises. Since the DP also threads (less well:
~4× against the sieve's ~5×), that makes the comparison conservative in the sieve's favour, which is
where the two are closest and mispicking would cost the most.

This matters most for repeated rows: enough repetition shrinks the sieve below the DP even at large
`K` (`rpt = [2,2,…]` at `N = 28` sieves), while for distinct rows the DP wins at every degree it
covers.
"""
function _prefer_dp(K::Int, sieve_work::Int, nchunks::Int)
    K <= DP_MAX || return false
    _, ntrans = _dp_counts(K)
    return 2 * ntrans < sieve_work ÷ max(nchunks, 1)
end
