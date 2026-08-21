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
# Each state's sum is accumulated by a single thread in plan order, which is the same order the CPU
# uses, so results are bit-identical rather than merely close.
@kernel function _dp_level_kernel!(H, @Const(P), @Const(starts), @Const(trans), lo, bits, mask)
    b, i = @index(Global, NTuple)      # b: instance (fastest), i: state within the level
    s = lo + i - 1
    acc = zero(eltype(H))
    @inbounds begin
        for k in starts[s]:starts[s+1]-1
            t = trans[k]
            acc += P[b, t&mask] * H[b, t>>bits]
        end
        H[b, s] = acc
    end
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

    for first_b in 1:chunk:B
        last_b = min(first_b + chunk - 1, B)
        nb = last_b - first_b + 1

        Pd = KA.allocate(backend, T, nb, npairs)
        copyto!(Pd, P[first_b:last_b, :])
        Hd = KA.allocate(backend, T, nb, nstates)
        fill!(view(Hd, :, 1), one(T))          # the empty set

        for L in 2:length(levels)-1
            lo = Int(levels[L])
            hi = Int(levels[L+1]) - 1
            kernel(Hd, Pd, dev.starts, dev.trans, lo, bits, mask; ndrange = (nb, hi - lo + 1))
        end
        KA.synchronize(backend)

        copyto!(view(out, first_b:last_b), Array(Hd[:, nstates]))
    end
    return out
end

end # module
