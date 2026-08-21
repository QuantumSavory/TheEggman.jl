"""
GPU evaluation of TheEggman's subset DP, via KernelAbstractions.

Loading `KernelAbstractions` (together with a backend package such as CUDA.jl) supplies the method
behind `TheEggman._haf_dp_backend`, which the `backend` keyword on [`hafnian`](@ref) and
[`hafnian_repeated`](@ref) routes to.

Only the subset DP is ported. The unrolled kernels finish in tens of nanoseconds, less than a single
kernel launch; the sieve needs roughly a 40 KB workspace per term, which would leave almost no
occupancy. The DP, by contrast, is memory-bandwidth-bound on the CPU — two indirect loads per
multiply, with thread scaling stalling near 4× — which is exactly the shape a GPU improves.

# Why the batch axis comes first

`P` and `H` are stored `B × ·`, so the instance index is fastest-varying. A warp then covers 32
consecutive instances of the *same* subproblem: the transition it follows is one value broadcast
across the warp, and the reads it makes from `P` and `H` are contiguous. Laying the arrays out the
other way would scatter every one of them. Batching is therefore what makes the port worth doing,
not merely a convenience — and it also amortises the `K/2` kernel launches each call needs across
the whole batch instead of paying them per hafnian.
"""
module TheEggmanKernelAbstractionsExt

using TheEggman
using KernelAbstractions
const KA = KernelAbstractions

# Device copies of a plan's CSR arrays, cached in the base package (see `TheEggman._DEVICE_PLANS`)
# so its lifecycle is inspectable without a GPU. `levels` deliberately stays on the host: it is only
# ever read to compute launch bounds.
function _device_plan(backend, plan::TheEggman.HafnianDPPlan{I}) where {I}
    key = (backend, plan.K, I)
    lock(TheEggman._DEVICE_PLAN_LOCK) do
        get!(TheEggman._DEVICE_PLANS, key) do
            starts = KA.allocate(backend, I, length(plan.starts))
            trans = KA.allocate(backend, I, length(plan.trans))
            copyto!(starts, plan.starts)
            copyto!(trans, plan.trans)
            (starts = starts, trans = trans)
        end
    end
end

# One level of the DP. Every state in a level is independent and reads only levels at least two
# below, so `H` is safely read and written in the same launch — it must NOT be marked `@Const`.
#
# Each state's sum is accumulated by a single thread in plan order — the same order the CPU uses —
# so nothing is reordered. Results are still not bit-identical to the CPU: device compilers contract
# `a*b + c` into a single-rounding FMA, which shifts the last ulp. Measured against an extended
# precision reference the contracted result is the *more* accurate of the two, and the gap runs
# ~1e-16 to 2e-15 relative. Compare across backends with a tolerance, never with `===`.
@kernel function _dp_level_kernel!(H, @Const(P), @Const(starts), @Const(trans), lo, B, bits, mask)
    b, i = @index(Global, NTuple)      # b: instance (fastest), i: state within the level
    s = lo + i - 1
    acc = zero(eltype(H))
    # `H` and `P` are flat with the instance index fastest, so `(col - 1) * B + b`. Flat rather than
    # 2-D so the buffers can be reused across calls at whatever batch size turns up, without
    # reshaping a device array.
    @inbounds begin
        for k in starts[s]:starts[s+1]-1
            t = trans[k]
            acc += P[(Int(t & mask) - 1) * B + b] * H[(Int(t >> bits) - 1) * B + b]
        end
        H[(s - 1) * B + b] = acc
    end
end

"""
    _scratch(backend, T, nH, nP) -> (H, P)

Reusable device work buffers, grown on demand and never shrunk.

Allocating these per call cost ~21 µs of fixed overhead, which dominates every small batch and shows
up again as allocator churn on large ones. They are cached in the base package alongside the plans
(see `TheEggman._DEVICE_SCRATCH`) so that `gpu_cache_bytes` accounts for them.
"""
function _scratch(backend, ::Type{T}, nH::Int, nP::Int) where {T}
    cache = TheEggman._DEVICE_SCRATCH
    key = (backend, T)
    cur = get(cache, key, nothing)
    if cur === nothing || length(cur.H) < nH || length(cur.P) < nP
        H = KA.allocate(backend, T, max(nH, cur === nothing ? 0 : length(cur.H)))
        P = KA.allocate(backend, T, max(nP, cur === nothing ? 0 : length(cur.P)))
        cur = (H = H, P = P)
        cache[key] = cur
    end
    return cur
end

function TheEggman._haf_dp_backend(
    backend::KA.Backend,
    P::Matrix{T},
    plan::TheEggman.HafnianDPPlan{I};
    max_batch::Union{Nothing,Int} = nothing,
) where {T,I}
    B, npairs = size(P)
    bits = TheEggman._pair_bits(I)
    mask = TheEggman._pair_mask(I)
    dev = _device_plan(backend, plan)
    levels = plan.levels
    nstates = plan.nstates

    out = Vector{T}(undef, B)
    chunk = max_batch === nothing ? B : max(1, min(B, max_batch))
    kernel = _dp_level_kernel!(backend)
    host = Vector{T}(undef, min(chunk, B))

    # The scratch buffers are shared, so one call at a time per backend. Device work serialises
    # anyway, and this keeps concurrent callers from writing over each other's state array.
    lock(TheEggman._DEVICE_PLAN_LOCK) do
        for first_b in 1:chunk:B
            last_b = min(first_b + chunk - 1, B)
            nb = last_b - first_b + 1
            sc = _scratch(backend, T, nb * nstates, nb * npairs)
            Hd, Pd = sc.H, sc.P

            copyto!(view(Pd, 1:nb*npairs), vec(P[first_b:last_b, :]))
            fill!(view(Hd, 1:nb), one(T))      # the empty set, one entry per instance

            for L in 2:length(levels)-1
                lo = Int(levels[L])
                hi = Int(levels[L+1]) - 1
                kernel(Hd, Pd, dev.starts, dev.trans, lo, nb, bits, mask;
                       ndrange = (nb, hi - lo + 1))
            end
            KA.synchronize(backend)

            copyto!(view(host, 1:nb), view(Hd, (nstates-1)*nb+1:nstates*nb))
            copyto!(view(out, first_b:last_b), view(host, 1:nb))
        end
    end
    return out
end

end # module
