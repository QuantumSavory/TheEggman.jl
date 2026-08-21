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

Transitions are `Θ(K φ^K)`: `Θ(φ^K)` states, each with `O(K)` of them. That is *worse*
asymptotically than the sieve's `Θ(K³ 2^{K/2}) = Θ(K³ 1.414^K)` — their exact counts cross at
`K ≈ 62` — but across the entire range that is actually computable, the sieve does 65–100× more
arithmetic:

| K              | 12  | 16   | 20    | 24     | 28      | 32       |
|----------------|-----|------|-------|--------|---------|----------|
| DP transitions | 1076| 10226| 89665 | 748776 | 6052062 | 47786401 |
| sieve work     | 106k| 1.0M | 7.9M  | 54M    | 393M    | 2.1G     |

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
3 ms at `K = 20`, 25 MB and 0.2 s at `K = 28`, 196 MB and 2.1 s at `K = 32`. They are cached for the
lifetime of the session, so the build is amortised across calls. `DP_MAX` caps the degree — not
because the sieve catches up there (it does not; see [`DP_MAX`](@ref)) but because a transition
stops fitting in an `Int32` above it. Past that, [`hafnian`](@ref) falls back to the sieve, whose
memory is negligible.

Accuracy is that of the definition — the DP sums products of matrix entries with no cancellation
between large signed terms, so it is more accurate than either sieve variant.
"""

using Base.Threads: ReentrantLock, @spawn

"""
Largest total degree for which a DP plan will be built.

This is a *representation* limit, not a performance crossover. The sieve never overtakes the DP
anywhere reachable — measured on 12 threads, the DP is still 9.6× ahead at `K = 32`:

| K              | 24   | 26   | 28   | 30   | 32    |
|----------------|------|------|------|------|-------|
| DP (ms)        | 0.16 | 0.40 | 1.06 | 4.19 | 11.34 |
| sieve (ms)     | 2.88 | 11.0 | 19.3 | 45.1 | 108.7 |
| plan (MB)      | 3.1  | 8.9  | 25.0 | 70.2 | 195.7 |
| plan build (s) | 0.02 | 0.05 | 0.20 | 0.64 | 2.13  |

The DP's advantage does shrink — its states grow like `φ^K ≈ 1.618^K` against the sieve's
`1.414^K K³`, costing about 13% of the ratio per two degrees — but extrapolating the measured times
puts parity near `K ≈ 60`, far past anything either method could hold in memory.

What actually binds is the packed `Int32` transition (see [`_pair_bits`](@ref)): at `K = 32` it needs
9 bits of pair index plus 22 of state index, exactly the 31 available. `K = 34` needs 10 + 24 = 34
bits, so it cannot be packed into an `Int32` under any split.

Above this the DP is still the faster algorithm, but a plan stops being something to build casually
— 1 GB at `K = 34`, 21 GB at `K = 40` — so it is never chosen automatically. `method = :dp` opts in
explicitly and switches to the wide `Int64` layout; see [`DP_HARD_MAX`](@ref).

Plans are cached for the session, so their build cost is amortised — but a *single* cold call at
`K = 32` pays 2.13 s of construction to save 0.10 s of evaluation, and only comes out ahead after
about 22 calls (11 at `K = 28`, 16 at `K = 30`). Pass `method = :sieve` for one-shot large hafnians.
"""
const DP_MAX = 32

"""
Largest degree at which `method = :dp` will build a plan at all.

Past [`DP_MAX`](@ref) the DP is still the faster algorithm — it is memory, not the sieve, that
stops it — so explicitly requesting `:dp` is allowed well beyond the automatic cap, for machines
where that memory exists. Those plans use the wide `Int64` layout (see [`_dp_index_type`](@ref)).

The cap itself is where the *representation* runs out: subsets are enumerated as bitmasks, and a
`UInt64` holds 64 of them. Everything else has headroom at that point — the packed transition needs
44 + 12 = 56 of its 63 bits at `K = 64`.

It stays worth it: at `K = 34`, measured on 12 threads, the DP runs in 56.5 ms against the sieve's
249.9 ms — 4.4× — off a 1.06 GB plan that takes 6.5 s to build once.

Memory is what will actually stop you, long before. Plan size, and the per-call state array:

| K            | 34   | 38   | 40   | 44    | 46    | 50     |
|--------------|------|------|------|-------|-------|--------|
| plan         | 1 GB | 8 GB | 21 GB| 160 GB| 439 GB| 3.3 TB |
| state array  | 0.1  | 0.9  | 2.5  | 17    | 44    | 303 GB |

Building transiently needs about twice the plan — the mask set and index map are live at the same
time as the finished arrays — and each concurrent call needs its own state array on top.
[`_build_dp_plan`](@ref) refuses up front, with the projected size, rather than letting the
allocation take the machine down.
"""
const DP_HARD_MAX = 64

"""
    HafnianDPPlan

Precomputed subset structure for degree `K`, independent of any matrix.

States are numbered `1:nstates` in order of increasing subset size, with the empty set first and the
full set last, so evaluating them in order visits every subproblem after its children. Transitions
are stored in CSR form: state `s` owns `starts[s]:starts[s+1]-1`, and entry `k` contributes
`P[pair[k]] * H[child[k]]`, where `P` holds the flattened upper triangle of the gathered matrix (see
[`_pair_index`](@ref)).

Each transition is one packed integer `I`: the target state in the high bits and the matrix-entry
index in the low `_pair_bits(I)`. The evaluation loop is memory-bound on this array — 24 MB of it at
`K = 28` — so packing the two indices together rather than storing parallel arrays halves its
traffic, which is worth far more than the shift and mask it costs to unpack.

`I` is `Int32` up to [`DP_MAX`](@ref) and `Int64` above, which also widens `starts` and `levels`:
past `K = 40` a plan has more than `typemax(Int32)` transitions, and past `K = 46` more than that
many states. See [`_dp_index_type`](@ref).

`levels` records where each subset size begins, so `levels[L]:levels[L+1]-1` are the states of one
size. Every transition out of such a state lands two sizes below, so states within a level are
mutually independent and can be evaluated in parallel — see [`_haf_dp`](@ref).
"""
struct HafnianDPPlan{I<:Signed}
    K::Int
    nstates::Int
    starts::Vector{I}
    trans::Vector{I}
    levels::Vector{I}
end

"""
    _pair_bits(I) -> Int

Bits reserved for the matrix-entry index in a transition packed into `I`.

Nine is exactly enough for the 496 pairs at `DP_MAX`, and leaves the 22 an `Int32` state index needs
there. The wide layout spends twelve — more than the 2016 pairs at [`DP_HARD_MAX`](@ref) require —
because with 51 bits left for the state index there is nothing to gain by being tighter.

Dispatching on the type rather than storing this per plan keeps the shift and mask compile-time
constants inside [`_dp_states!`](@ref).
"""
_pair_bits(::Type{Int32}) = 9
_pair_bits(::Type{Int64}) = 12
@inline _pair_mask(::Type{I}) where {I} = I(1) << _pair_bits(I) - I(1)

"""
    _dp_index_type(K) -> Type

Integer type for a degree-`K` plan's transitions and offsets: `Int32` through [`DP_MAX`](@ref),
`Int64` above, where neither the packed transition nor the offsets would still fit.
"""
_dp_index_type(K::Int) = K <= DP_MAX ? Int32 : Int64

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
    _build_dp_plan(K, I = _dp_index_type(K)) -> HafnianDPPlan{I}

Enumerate the reachable subsets for degree `K` and lay out their transitions.

Walks the recursion from the full set, sorts the subsets by size (then by mask, for determinism),
and rewrites every transition in terms of the resulting state numbering. The `Dict` used for that
rewrite lives only as long as the build.

Masks are `UInt32` where they fit and `UInt64` beyond, independently of `I`, so that the dominant
build-time structures stay narrow at the sizes that are actually common. Refuses up front if the
plan cannot be represented, or if it would not fit in physical memory.
"""
function _build_dp_plan(K::Int, ::Type{I} = _dp_index_type(K)) where {I<:Signed}
    K <= DP_HARD_MAX ||
        throw(ArgumentError("degree $K exceeds DP_HARD_MAX = $DP_HARD_MAX; subsets are enumerated " *
                            "as bitmasks and a UInt64 holds only $DP_HARD_MAX of them"))
    nstates, ntrans = _dp_counts(K)
    npairs = K * (K - 1) ÷ 2
    npairs <= _pair_mask(I) ||
        throw(ArgumentError("degree $K needs more than $(_pair_bits(I)) bits per pair index in $I"))
    nstates <= (typemax(I) >> _pair_bits(I)) ||
        throw(ArgumentError("degree $K has too many states to pack into an $I transition"))
    ntrans <= typemax(I) ||
        throw(ArgumentError("degree $K has too many transitions to index with $I"))

    # The mask set and index map live alongside the finished arrays, so the build peaks near twice
    # the plan: measured 1.88 GB resident for the 1.06 GB plan at `K = 34`. Checking the plan size
    # alone would wave through builds that then run the machine out of memory.
    bytes = (ntrans + nstates + 1) * sizeof(I)
    2 * bytes < Sys.total_memory() ||
        throw(ArgumentError("a degree-$K plan needs $(round(bytes / 2^30, digits = 1)) GB, and " *
                            "about twice that while building — more than this machine has; use " *
                            "method = :sieve"))

    return K <= 32 ? _build_dp_plan(K, UInt32, I) : _build_dp_plan(K, UInt64, I)
end

function _build_dp_plan(K::Int, ::Type{M}, ::Type{I}) where {M<:Unsigned,I<:Signed}
    full = ~M(0) >> (8 * sizeof(M) - K)     # `M(1) << K` would be undefined at the full width
    reached = Set{M}([full])
    stack = M[full]
    while !isempty(stack)
        m = pop!(stack)
        m == 0 && continue
        i = trailing_zeros(m) + 1
        rest = m & ~(M(1) << (i - 1))
        r = rest
        while r != 0
            j = trailing_zeros(r) + 1
            r &= ~(M(1) << (j - 1))
            sub = rest & ~(M(1) << (j - 1))
            if !(sub in reached)
                push!(reached, sub)
                push!(stack, sub)
            end
        end
    end

    masks = collect(reached)
    sort!(masks; by = m -> (count_ones(m), m))
    nstates = length(masks)
    number = Dict{M,I}(m => I(s) for (s, m) in enumerate(masks))

    _, ntrans = _dp_counts(K)
    bits = _pair_bits(I)
    starts = Vector{I}(undef, nstates + 1)
    trans = Vector{I}(undef, ntrans)

    n = 0
    starts[1] = 1
    for (s, m) in enumerate(masks)
        if m != 0
            i = trailing_zeros(m) + 1
            rest = m & ~(M(1) << (i - 1))
            r = rest
            while r != 0
                j = trailing_zeros(r) + 1
                r &= ~(M(1) << (j - 1))
                n += 1
                c = number[rest&~(M(1)<<(j-1))]
                trans[n] = (c << bits) | I(_pair_index(i, j, K))
            end
        end
        starts[s+1] = I(n + 1)
    end

    levels = I[1]
    for s in 2:nstates
        count_ones(masks[s]) != count_ones(masks[s-1]) && push!(levels, I(s))
    end
    push!(levels, I(nstates + 1))

    return HafnianDPPlan{I}(K, nstates, starts, trans, levels)
end

# One cache per layout, rather than a single `Dict{Int,HafnianDPPlan}`: an abstractly-typed cache
# would make every `_haf_dp` call a dynamic dispatch and leave `hafnian`'s return type uninferrable.
const _DP_PLANS32 = Dict{Int,HafnianDPPlan{Int32}}()
const _DP_PLANS64 = Dict{Int,HafnianDPPlan{Int64}}()
const _DP_PLAN_LOCK = ReentrantLock()

_dp_cache(::Type{Int32}) = _DP_PLANS32
_dp_cache(::Type{Int64}) = _DP_PLANS64

"""
    _dp_plan(K, I = _dp_index_type(K)) -> HafnianDPPlan{I}

The degree-`K` plan in layout `I`, built on first use and cached for the rest of the session.

Plans are pure functions of `K`, so caching them is what makes the DP worth using at the larger
degrees: the build is a few hundred milliseconds at `DP_MAX`, and seconds to minutes above it, but
is paid once. The cache is never evicted; see [`DP_MAX`](@ref) and [`DP_HARD_MAX`](@ref) for the
sizes it is allowed to reach.
"""
function _dp_plan(K::Int, ::Type{I} = _dp_index_type(K)) where {I<:Signed}
    lock(_DP_PLAN_LOCK) do
        get!(() -> _build_dp_plan(K, I), _dp_cache(I), K)::HafnianDPPlan{I}
    end
end

"""
    _dp_states!(H, P, starts, trans, lo, hi)

Evaluate states `lo:hi` into `H`, reading gathered matrix entries from `P`.

The inner loop of the whole package: one multiply-add per transition, with both operands reached
through indirection. Every state in `lo:hi` must have had its children evaluated already, which
holds because states are numbered by subset size and transitions always descend two sizes — see
[`HafnianDPPlan`](@ref).

The shift and mask come from the transition type via [`_pair_bits`](@ref), so they are compile-time
constants in each specialisation rather than loads from the plan.
"""
function _dp_states!(
    H::Vector{T},
    P::Vector{T},
    starts::Vector{I},
    trans::Vector{I},
    lo::Int,
    hi::Int,
) where {T,I}
    bits = _pair_bits(I)            # compile-time constants: `I` fixes both
    mask = _pair_mask(I)
    @inbounds for s in lo:hi
        acc = zero(T)
        for k in starts[s]:starts[s+1]-1
            t = trans[k]
            acc += P[t&mask] * H[t>>bits]
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
    plan::HafnianDPPlan{I};
    nthreads::Int = 1,
) where {T,I}
    K = plan.K
    length(idx) == K || throw(DimensionMismatch("idx has length $(length(idx)), plan expects $K"))

    # Gather the upper triangle once; every transition then reads a matrix entry by flat index.
    P = _gather_pairs!(Vector{T}(undef, K * (K - 1) ÷ 2), A, idx, K)

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
    _gather_pairs!(P, A, idx, K) -> P

Fill `P` with the flattened upper triangle of `A[idx, idx]`, laid out by [`_pair_index`](@ref).

This is the only place the kernels touch the caller's matrix: afterwards every transition reads a
matrix entry by flat index, which is what lets the evaluation be a pure gather over `P` and `H`.
"""
function _gather_pairs!(
    P::AbstractVector,
    A::AbstractMatrix,
    idx::AbstractVector{<:Integer},
    K::Int,
)
    @inbounds for i in 1:K-1, j in i+1:K
        P[_pair_index(i, j, K)] = A[idx[i], idx[j]]
    end
    return P
end

"""
    _gather_pairs_batch(T, A, idxs, K) -> Matrix{T}
    _gather_pairs_batch(T, As, idxs, K) -> Matrix{T}

Gather one batch of instances into a `B × npairs` matrix — **instance index first**, so it is the
fastest-varying one.

That layout is the point of batching on a GPU. A warp handling 32 consecutive instances of the same
state reads one contiguous run of `P` (and of `H`, laid out the same way), while the transition it
is following is a single value broadcast across the warp. A `npairs × B` layout would scatter every
one of those reads.

The first form shares one matrix across instances that differ only in `idxs` — the Gaussian
boson-sampling case, where the batch is many photon patterns against a single `A`.
"""
function _gather_pairs_batch(
    ::Type{T},
    A::AbstractMatrix,
    idxs::AbstractVector{<:AbstractVector{<:Integer}},
    K::Int,
) where {T}
    B = length(idxs)
    P = Matrix{T}(undef, B, K * (K - 1) ÷ 2)
    @inbounds for b in 1:B
        idx = idxs[b]
        for i in 1:K-1, j in i+1:K
            P[b, _pair_index(i, j, K)] = A[idx[i], idx[j]]
        end
    end
    return P
end

function _gather_pairs_batch(
    ::Type{T},
    As::AbstractVector{<:AbstractMatrix},
    idxs::AbstractVector{<:AbstractVector{<:Integer}},
    K::Int,
) where {T}
    B = length(idxs)
    P = Matrix{T}(undef, B, K * (K - 1) ÷ 2)
    @inbounds for b in 1:B
        A = As[b]
        idx = idxs[b]
        for i in 1:K-1, j in i+1:K
            P[b, _pair_index(i, j, K)] = A[idx[i], idx[j]]
        end
    end
    return P
end

"""
    _haf_dp_backend(backend, P, plan) -> Vector

Evaluate a whole batch of degree-`plan.K` hafnians on `backend`, given the gathered pair matrix `P`
from [`_gather_pairs_batch`](@ref).

This is a hook with no method in the base package: loading `KernelAbstractions` brings in
`ext/TheEggmanKernelAbstractionsExt.jl`, which adds the method that actually runs. Keeping it out of
the base package keeps `TheEggman` dependent on nothing but `LinearAlgebra`, which matters because
its downstream consumers are CPU-only.
"""
function _haf_dp_backend end

# Reached only if someone conjures a backend without KernelAbstractions loaded, which should be
# impossible in practice; a named error still beats a MethodError.
_haf_dp_backend(backend, P, plan; kwargs...) = throw(ArgumentError(
    "no GPU backend method for $(typeof(backend)); run `using KernelAbstractions` (and a backend " *
    "package such as CUDA.jl) before passing `backend`"))

# Device-resident copies of plans, populated by the KernelAbstractions extension and keyed by
# (backend, degree, layout). The cache lives here rather than in the extension so that its lifecycle
# is inspectable without a GPU loaded, and so the extension adds methods rather than overwriting any.
const _DEVICE_PLANS = Dict{Any,Any}()
const _DEVICE_PLAN_LOCK = ReentrantLock()

"""
    gpu_cache_bytes() -> Int

Device memory held by cached DP plans, across every backend. Zero until a GPU plan is built.

Plans are never evicted — they are pure functions of the degree and expensive to rebuild — so at
`N = 32` this reaches ~196 MB per backend. [`empty_gpu_cache!`](@ref) releases it.
"""
function gpu_cache_bytes()
    lock(_DEVICE_PLAN_LOCK) do
        total = 0
        for (_, dev) in _DEVICE_PLANS, arr in dev
            total += length(arr) * sizeof(eltype(arr))
        end
        return total
    end
end

"""
    empty_gpu_cache!()

Drop every cached device plan, releasing its memory. Later calls re-upload on demand.
"""
function empty_gpu_cache!()
    lock(_DEVICE_PLAN_LOCK) do
        empty!(_DEVICE_PLANS)
    end
    return nothing
end

"""
    dp_batch_bytes(N, T, B) -> Int

Device memory one batch needs, excluding the shared plan: the `B × nstates` state array plus the
`B × npairs` gathered pairs.

`KernelAbstractions` exposes no portable free-memory query, so batches are not chunked
automatically — pass `max_batch` to [`hafnian`](@ref) if a batch will not fit. Use this to size it.
"""
function dp_batch_bytes(N::Int, ::Type{T}, B::Int) where {T}
    nstates, _ = _dp_counts(N)
    return B * (nstates + N * (N - 1) ÷ 2) * sizeof(T)
end

"""
    _prefer_dp(K, sieve_work, nchunks) -> Bool

Decide between the DP and the sieve for total degree `K`.

`sieve_work` is the sieve's serial multiply-add count and `nchunks` the number of tasks it would
split into, so `sieve_work / nchunks` is what it actually costs in wall clock. The DP is compared
against that at its *serial* cost, since it threads less well (~4× against the sieve's ~5×), which
keeps the comparison conservative in the sieve's favour — where the two are closest and mispicking
would cost the most.

A DP transition is weighted at one sieve work unit. Unlike the other constants in this package that
is not a comfortable threshold with slack on both sides — the two classes genuinely overlap. Over a
19-case sweep the cases the DP should win go down to a ratio of 0.37 while the cases the sieve should
win reach 0.89, so *no* single weight separates them. One is the least any weight achieves, and
weights in `[0.89, 1.25]` all achieve it; 1 sits in the middle of that. The surviving miss is
`rpt = fill(2, 9)`, where the sieve gets chosen and costs 1.34×.

The overlap is the model's fault, not the data's: it prices both strategies as a single work count
times a constant, but the sieve carries a few microseconds of fixed setup that dominates its
smallest cases, and neither strategy's cost per unit is really constant across cache regimes. A
model with an intercept would separate them; a linear one cannot.

The weight is deliberately *not* the 2 that was correct before [`_sieve_variant_cost`](@ref) existed.
Pricing the sieve as `steps × _term_cost(E, n)` overstated it — every term was charged at full size,
though terms shrink whenever a multiplicity vanishes — so a transition had to be weighted at 2 to
compensate. Now that the sieve is costed honestly the compensation has to come out, or the DP loses
distinct-row cases it wins by 12×.

This matters most for repeated rows: enough repetition shrinks the sieve below the DP even at large
`K` (`rpt = [2,2,…]` at `N = 28` sieves), while for distinct rows the DP wins at every degree it
covers.
"""
function _prefer_dp(K::Int, sieve_work::Int, nchunks::Int)
    K <= DP_MAX || return false
    _, ntrans = _dp_counts(K)
    return ntrans < sieve_work ÷ max(nchunks, 1)
end
