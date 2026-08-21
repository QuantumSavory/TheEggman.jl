"""
Hafnians via the Björklund/Glynn finite-difference sieve.

The hafnian of a symmetric ``2n × 2n`` matrix ``A`` is the sum, over all perfect matchings ``M`` of
the complete graph on ``2n`` vertices, of ``∏_{(i,j) ∈ M} A[i,j]``. Enumerating those ``(2n-1)!!``
matchings directly is the naive algorithm; this file implements the exponentially better
``O(n³ 2ⁿ)`` sieve of [Björklund, Gupt & Quesada](https://arxiv.org/abs/2108.01622), the same
algorithm used by Xanadu's `thewalrus`.

# The identity

Fix a perfect matching of the ``2n`` vertices — the "edges" — and let ``X`` be its adjacency matrix
(a permutation swapping each vertex with its partner). For a subset ``S`` of edges write ``A_S`` for
the principal submatrix on the endpoints of ``S``. Then

```
haf(A) = Σ_{S ⊆ edges} (-1)^{n-|S|} [λⁿ] exp( Σ_{k≥1} tr((A_S X)ᵏ) λᵏ / 2k )
```

The generating function counts vertex-disjoint cycles alternating between ``A``-edges and
matching-edges; the ``λⁿ`` coefficient selects those using exactly ``n`` ``A``-edges, and
inclusion–exclusion over ``S`` kills everything that is not a full cover, i.e. a perfect matching.

Two changes make this practical, both taken from the reference implementation:

  * **Glynn form.** Replacing the 0/1 subset indicator by a ``±1`` sign vector ``δ`` (a finite
    difference sieve rather than inclusion–exclusion) makes the summand invariant under the global
    flip ``δ → -δ``, so only half the ``2ⁿ`` terms need evaluating.
  * **Repeated rows.** If row/column ``i`` is repeated ``rᵢ`` times, matching up the repeats
    ([`matched_reps`](@ref)) turns the sum over subsets of edges into a mixed-radix sum over
    *multiplicities* of only ``E`` distinct edges, weighted by binomial coefficients. The cost
    becomes ``∏(rᵢ+1)`` terms on ``2E × 2E`` matrices instead of ``2^{N/2}`` terms on ``N × N``
    ones — a large win whenever the same row appears many times.

# Deviation from `thewalrus`

`thewalrus` evaluates the inner ``[λⁿ]`` coefficient by first forming the power traces
``tr(Mᵏ)``, `k = 1…n`, through ``n`` explicit matrix products — ``O(m³)`` each, so ``O(n⁴)`` per
sieve term. We instead use

```
exp( Σ_{k≥1} tr(Mᵏ) λᵏ / 2k ) = det(I - λM)^{-1/2}
```

and get all the coefficients of ``det(I - λM)`` at once from the characteristic polynomial of ``M``
(reduction to Hessenberg form, then La Budde's recurrence over leading principal minors). That is a
single ``O(m³)`` pass per sieve term, restoring the ``O(n³ 2ⁿ)`` complexity the algorithm is
supposed to have, and the ``[λⁿ]`` extraction from the polynomial is a plain ``O(n²)`` recurrence.

Two further choices matter for the constant factor, both measured rather than assumed:

  * the Hessenberg reduction uses Gaussian similarity transforms with partial pivoting
    ([`_hessenberg!`](@ref)) rather than Householder reflections, halving its flop count for a
    negligible accuracy cost at these matrix sizes;
  * for hardware complex types it runs on separate real and imaginary arrays
    ([`_hessenberg_split!`](@ref)), which vectorises where interleaved complex storage does not.

See [`_glynn_term!`](@ref) for the per-term pipeline.
"""

using Base.Threads: @spawn, nthreads

# ---------------------------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------------------------

"""
    HafnianWorkspace{T}

Scratch buffers for one sieve worker, sized from the largest submatrix it can encounter. The sieve
visits up to millions of terms, so every buffer the inner loop touches is allocated once here and
reused; `_glynn_term!` performs no allocation at all.

`E` is the number of distinct edges, `m = 2E` the largest submatrix side, and `n = N/2` the
coefficient index being extracted.

The `Mr`/`Mi`/`ur`/`ui` buffers back the split-storage Hessenberg reduction (see
[`_reduce_hessenberg!`](@ref)) and are allocated empty for element types that do not use it.
"""
struct HafnianWorkspace{T,R}
    M::Matrix{T}        # m×m working matrix, overwritten with its Hessenberg form
    v::Vector{T}        # Householder reflector / elimination multipliers
    w::Vector{T}        # matrix-vector scratch for the right-hand reflector application
    C::Matrix{T}        # (m+1)×(n+1) table of characteristic polynomials of leading minors
    f::Vector{T}        # n+1 coefficients of det(I - λM)^{-1/2}
    kept::Vector{Int}   # per-edge multiplicity for the current sieve term
    idx::Vector{Int}    # edges with nonzero multiplicity
    rows::Vector{Int}   # rows of `Ax` those edges select
    Mr::Matrix{R}       # real part of M, for the split-storage reduction
    Mi::Matrix{R}       # imaginary part of M
    ur::Vector{R}       # real part of the elimination multipliers
    ui::Vector{R}       # imaginary part of the elimination multipliers
end

# Split storage pays off only for the hardware complex types, where it lets the reduction run as
# plain real FMAs that LLVM can vectorise. Everything else keeps the generic in-place path.
_use_split(::Type) = false
_use_split(::Type{Complex{R}}) where {R<:Union{Float32,Float64}} = true
_split_size(::Type{T}, m::Int) where {T} = _use_split(T) ? m : 0

function HafnianWorkspace{T}(E::Int, n::Int) where {T}
    m = 2E
    R = real(T)
    ms = _split_size(T, m)
    HafnianWorkspace{T,R}(
        Matrix{T}(undef, m, m),
        Vector{T}(undef, m),
        Vector{T}(undef, m),
        Matrix{T}(undef, m + 1, n + 1),
        Vector{T}(undef, n + 1),
        Vector{Int}(undef, E),
        Vector{Int}(undef, E),
        Vector{Int}(undef, m),
        Matrix{R}(undef, ms, ms),
        Matrix{R}(undef, ms, ms),
        Vector{R}(undef, ms),
        Vector{R}(undef, ms),
    )
end

# ---------------------------------------------------------------------------------------------
# Edge matching for repeated rows/columns
# ---------------------------------------------------------------------------------------------

"""
    matched_reps(rpt) -> (x, edge_reps)

Pair up repeated rows into as few distinct edges as possible.

Given repetition counts `rpt` with an even total, returns a permutation `x` of length `2E`
(position `i` is matched with position `i + E`) and the multiplicity `edge_reps[i]` of each of the
`E` resulting edges. Reordering the matrix by `x` puts it in the layout `_calc_hafnian` expects: the
fixed perfect matching is `X = [0 I; I 0]`.

Repeatedly pairing the two most-repeated rows keeps `E` small, which is what makes
[`hafnian_repeated`](@ref) cheap: the sieve runs over `∏(edge_reps .+ 1)` terms rather than
`2^{N/2}`. A row whose count exceeds twice the next-largest is paired with itself instead, which is
why the diagonal of `A` can legitimately enter the result when some `rpt[i] ≥ 2`.

Assumes `sum(rpt)` is even, in which case every vertex gets matched.
"""
function matched_reps(rpt::AbstractVector{<:Integer})
    # Work on the nonzero counts only, as `(count, row)` pairs sorted descending — which for tuples
    # orders by count and then by row, matching thewalrus so the two implementations pick the same
    # pairing on identical input. Insertion sort keeps this allocation-free; the list is short and
    # already nearly sorted on every pass after the first.
    v = Tuple{Int,Int}[]
    sizehint!(v, length(rpt))
    for i in eachindex(rpt)
        rpt[i] > 0 && push!(v, (Int(rpt[i]), Int(i)))
    end

    edgesA = Int[]
    edgesB = Int[]
    edge_reps = Int[]
    isempty(v) && return (edgesA, edge_reps)
    sort!(v; rev = true, alg = InsertionSort)

    while length(v) > 1 || (length(v) == 1 && v[1][1] > 1)
        sort!(v; rev = true, alg = InsertionSort)
        reps1, x1 = v[1]

        if length(v) == 1 || reps1 > 2 * v[2][1]
            # Nothing big enough to soak up all of `reps1`: pair the row with itself.
            push!(edgesA, x1)
            push!(edgesB, x1)
            push!(edge_reps, reps1 ÷ 2)
            if iseven(reps1)
                popfirst!(v)
            else
                v[1] = (1, x1)
            end
        else
            reps2, x2 = v[2]
            push!(edgesA, x1)
            push!(edgesB, x2)
            push!(edge_reps, reps2)
            if reps1 > reps2
                v[1] = (reps1 - reps2, x1)
                deleteat!(v, 2)
            else
                deleteat!(v, 1:2)
            end
        end
    end

    return (append!(edgesA, edgesB), edge_reps)
end

# ---------------------------------------------------------------------------------------------
# Inner kernels
# ---------------------------------------------------------------------------------------------

"""
    _hessenberg!(M, m, mult)

Reduce the leading `m × m` block of `M` to upper Hessenberg form in place by Gaussian similarity
transforms with partial pivoting (EISPACK's `elmhes`), using `mult` as scratch for the elimination
multipliers.

At `(5/6)m³` multiply-adds this is half the cost of the Householder reduction in
[`_hessenberg_householder!`](@ref), and reducing to Hessenberg form is the single most expensive
thing the sieve does. Pivoting bounds the multipliers by 1, and the matrices here are small
(`m ≤ N`), so the accuracy cost is negligible in practice — see the `glynn` accuracy test in
`test/runtests.jl`.

Only the Hessenberg entries are meaningful afterwards; the strictly-lower part is zeroed as it goes
so [`_charpoly_hessenberg!`](@ref) can read the subdiagonal directly.
"""
function _hessenberg!(M::AbstractMatrix{T}, m::Int, mult::AbstractVector{T}) where {T}
    @inbounds for k in 2:m-1
        # Partial pivot on column k-1 over rows k:m, applied as a similarity (swap both).
        p = k
        big = abs(M[k, k-1])
        for i in k+1:m
            a = abs(M[i, k-1])
            if a > big
                big = a
                p = i
            end
        end
        if p != k
            for j in k-1:m
                M[p, j], M[k, j] = M[k, j], M[p, j]
            end
            for i in 1:m
                M[i, p], M[i, k] = M[i, k], M[i, p]
            end
        end

        x = M[k, k-1]
        iszero(x) && continue      # column already clear below the subdiagonal

        for i in k+1:m
            mult[i] = M[i, k-1] / x
            M[i, k-1] = zero(T)
        end

        # Row elimination as a rank-1 update, so the inner loop runs down columns.
        for j in k:m
            akj = M[k, j]
            iszero(akj) && continue
            for i in k+1:m
                M[i, j] -= mult[i] * akj
            end
        end

        # Compensating column operation that makes the elimination a similarity.
        for i in k+1:m
            mi = mult[i]
            iszero(mi) && continue
            for l in 1:m
                M[l, k] += mi * M[l, i]
            end
        end
    end
    return nothing
end

"""
    _hessenberg_split!(Mr, Mi, m, ur, ui)

Same Gaussian-similarity reduction as [`_hessenberg!`](@ref), but on a complex matrix held as
separate real and imaginary parts.

Interleaved `Complex{Float64}` storage forces the compiler to shuffle lanes for every complex
multiply; with the parts split apart both inner loops become plain real FMAs over contiguous
columns, which vectorise cleanly and run about 1.75× faster. This is the hot loop of the whole
package.
"""
function _hessenberg_split!(
    Mr::Matrix{R},
    Mi::Matrix{R},
    m::Int,
    ur::Vector{R},
    ui::Vector{R},
) where {R<:AbstractFloat}
    @inbounds for k in 2:m-1
        # Partial pivot on column k-1 over rows k:m, comparing squared moduli.
        p = k
        big = Mr[k, k-1]^2 + Mi[k, k-1]^2
        for i in k+1:m
            a = Mr[i, k-1]^2 + Mi[i, k-1]^2
            if a > big
                big = a
                p = i
            end
        end
        if p != k
            for j in k-1:m
                Mr[p, j], Mr[k, j] = Mr[k, j], Mr[p, j]
                Mi[p, j], Mi[k, j] = Mi[k, j], Mi[p, j]
            end
            for i in 1:m
                Mr[i, p], Mr[i, k] = Mr[i, k], Mr[i, p]
                Mi[i, p], Mi[i, k] = Mi[i, k], Mi[i, p]
            end
        end

        xr = Mr[k, k-1]
        xi = Mi[k, k-1]
        d = xr * xr + xi * xi
        iszero(d) && continue

        for i in k+1:m
            ar = Mr[i, k-1]
            ai = Mi[i, k-1]
            ur[i] = (ar * xr + ai * xi) / d
            ui[i] = (ai * xr - ar * xi) / d
            Mr[i, k-1] = zero(R)
            Mi[i, k-1] = zero(R)
        end

        # Row elimination as a rank-1 update, down columns.
        for j in k:m
            ar = Mr[k, j]
            ai = Mi[k, j]
            @simd for i in k+1:m
                Mr[i, j] -= ur[i] * ar - ui[i] * ai
                Mi[i, j] -= ur[i] * ai + ui[i] * ar
            end
        end

        # Compensating column operation that makes the elimination a similarity.
        for i in k+1:m
            mr = ur[i]
            mi = ui[i]
            @simd for l in 1:m
                Mr[l, k] += mr * Mr[l, i] - mi * Mi[l, i]
                Mi[l, k] += mr * Mi[l, i] + mi * Mr[l, i]
            end
        end
    end
    return nothing
end

"""
    _reduce_hessenberg!(ws, m)

Reduce `ws.M[1:m, 1:m]` to upper Hessenberg form, picking the fastest kernel for the element type.

For hardware complex types this splits into `ws.Mr`/`ws.Mi`, runs [`_hessenberg_split!`](@ref), and
copies just the Hessenberg band back — the split and writeback together cost far less than the
1.75× the vectorised reduction saves. Everything else reduces in place via [`_hessenberg!`](@ref).
"""
_reduce_hessenberg!(ws::HafnianWorkspace{T}, m::Int) where {T} = _hessenberg!(ws.M, m, ws.v)

function _reduce_hessenberg!(ws::HafnianWorkspace{Complex{R},R}, m::Int) where {R<:Union{Float32,Float64}}
    M, Mr, Mi = ws.M, ws.Mr, ws.Mi
    @inbounds for j in 1:m, i in 1:m
        z = M[i, j]
        Mr[i, j] = real(z)
        Mi[i, j] = imag(z)
    end

    _hessenberg_split!(Mr, Mi, m, ws.ur, ws.ui)

    # Only the Hessenberg band is read afterwards.
    @inbounds for j in 1:m
        for i in 1:min(j + 1, m)
            M[i, j] = Complex{R}(Mr[i, j], Mi[i, j])
        end
    end
    return nothing
end

"""
    _hessenberg_householder!(M, m, v, w)

Reduce the leading `m × m` block of `M` to upper Hessenberg form in place by Householder
similarity transforms, using `v` and `w` as scratch.

Unconditionally backward stable, but at `(5/3)m³` multiply-adds it is twice the cost of
[`_hessenberg!`](@ref), which is what the sieve actually uses. Kept as the reference against which
the Gaussian reduction's accuracy is checked.
"""
function _hessenberg_householder!(M::AbstractMatrix{T}, m::Int, v::AbstractVector{T}, w::AbstractVector{T}) where {T}
    R = real(T)
    @inbounds for k in 1:m-2
        # Reflector annihilating M[k+2:m, k].
        nrm2 = zero(R)
        for i in k+1:m
            nrm2 += abs2(M[i, k])
        end
        iszero(nrm2) && continue

        x1 = M[k+1, k]
        nrm = sqrt(nrm2)
        # Choose the sign of σ away from x1 to avoid cancellation in v[k+1].
        σ = iszero(x1) ? T(nrm) : T(nrm * (x1 / abs(x1)))
        for i in k+1:m
            v[i] = M[i, k]
        end
        v[k+1] += σ
        # conj(σ)*x1 = nrm*|x1| is real, so ‖v‖² = 2(nrm² + nrm|x1|) exactly.
        vn2 = 2 * (nrm2 + nrm * abs(x1))
        iszero(vn2) && continue
        c = 2 / vn2

        # Column k is known analytically: the reflector maps it to -σ e₁.
        M[k+1, k] = -σ
        for i in k+2:m
            M[i, k] = zero(T)
        end

        # Left application: M[k+1:m, k+1:m] -= c * v * (vᴴ M[k+1:m, k+1:m]).
        for j in k+1:m
            s = zero(T)
            for i in k+1:m
                s += conj(v[i]) * M[i, j]
            end
            s *= c
            for i in k+1:m
                M[i, j] -= v[i] * s
            end
        end

        # Right application: M[1:m, k+1:m] -= c * (M[1:m, k+1:m] v) * vᴴ.
        # Accumulated column-wise so both passes stay contiguous in memory.
        for i in 1:m
            w[i] = zero(T)
        end
        for j in k+1:m
            vj = v[j]
            for i in 1:m
                w[i] += M[i, j] * vj
            end
        end
        for j in k+1:m
            cv = c * conj(v[j])
            for i in 1:m
                M[i, j] -= w[i] * cv
            end
        end
    end
    return nothing
end

"""
    _charpoly_hessenberg!(C, H, m, n)

Fill `C` with the characteristic polynomials of the leading principal submatrices of the upper
Hessenberg matrix `H`, truncated to the `n+1` highest-order coefficients.

Row `k+1` of `C` holds ``p_k(λ) = det(λI - H[1:k, 1:k])`` with `C[k+1, t+1]` the coefficient of
``λ^{k-t}``. La Budde's recurrence
``p_k = (λ - h_{kk}) p_{k-1} - Σ_{i<k} h_{ik} (∏_{j=i+1}^{k} h_{j,j-1}) p_{i-1}``
builds each row from the previous ones.

Truncating at `t ≤ n` is what keeps this cheap: a term contributes only from ``λ^{k-i+1}`` down, so
`i` never has to run below `k-n+1`, and the whole pass costs ``O(m n²)`` rather than ``O(m³)``. The
answer we ultimately want, ``det(I - λH) = 1 + Σ_j C[m+1, j+1] λʲ``, needs only ``j ≤ n``.
"""
function _charpoly_hessenberg!(C::AbstractMatrix{T}, H::AbstractMatrix{T}, m::Int, n::Int) where {T}
    @inbounds begin
        C[1, 1] = one(T)
        for t in 1:n
            C[1, t+1] = zero(T)
        end

        for k in 1:m
            hkk = H[k, k]
            tmax = min(k, n)
            prevmax = min(k - 1, n)
            # (λ - h_kk) * p_{k-1}
            C[k+1, 1] = C[k, 1]
            for t in 1:tmax
                a = t <= prevmax ? C[k, t+1] : zero(T)
                C[k+1, t+1] = a - hkk * C[k, t]
            end
            for t in tmax+1:n
                C[k+1, t+1] = zero(T)
            end

            # Subtract the h_{ik} β_{ik} p_{i-1} terms, walking `i` down so β accumulates.
            β = one(T)
            ilo = max(1, k - n + 1)
            for i in k-1:-1:ilo
                β *= H[i+1, i]
                iszero(β) && break          # a zero subdiagonal kills every remaining term
                coef = H[i, k] * β
                d = k - i + 1               # p_{i-1} starts at λ^{k-d}
                if !iszero(coef)
                    for t in d:min(n, d + i - 1)
                        C[k+1, t+1] -= coef * C[i, t-d+1]
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _inv_sqrt_det_coeff!(f, q, m, n)

Return ``[λⁿ] det(I - λM)^{-1/2}`` given `q[j+1] = [λʲ] det(I - λM)` for `j = 0…min(m, n)`.

With ``g = q^{-1/2}`` the ODE ``2 q g' + q' g = 0`` gives the ``O(n²)`` recurrence
``g_t = -(1/2t) Σ_{j=1}^{t} (2t - j) q_j g_{t-j}``, `g_0 = 1`. `f` holds the running coefficients.
"""
function _inv_sqrt_det_coeff!(f::AbstractVector{T}, q::AbstractVector{T}, m::Int, n::Int) where {T}
    @inbounds begin
        f[1] = one(T)
        for t in 1:n
            s = zero(T)
            for j in 1:min(t, m)
                s += (2t - j) * q[j+1] * f[t-j+1]
            end
            f[t+1] = -s / (2t)
        end
        return f[n+1]
    end
end

"""
    _glynn_term!(ws, Ax, edge_reps, n, j, binoms) -> T

Evaluate sieve term `j` of the Glynn sum: decode the mixed-radix multiplicity vector, build the
scaled submatrix ``M = A_S X D``, and return its signed contribution to the hafnian.

`j` indexes the mixed-radix odometer over `edge_reps .+ 1` (edge 1 most significant, and restricted
to its lower half — see [`_calc_hafnian`](@ref)). Under `glynn` the substitution
`kept → 2*kept - edge_reps` moves the multiplicities into the symmetric range used by the
finite-difference sieve; otherwise they are left as plain inclusion–exclusion multiplicities. Either
way, edges whose multiplicity lands on zero drop out of the submatrix entirely.
"""
function _glynn_term!(
    ws::HafnianWorkspace{T},
    Ax::Matrix{T},
    edge_reps::Vector{Int},
    n::Int,
    j::Int,
    binoms::Matrix{R},
    ::Val{glynn},
) where {T,R,glynn}
    E = length(edge_reps)
    kept = ws.kept
    idx = ws.idx
    rows = ws.rows

    # Mixed-radix decode of `j`, least significant digit last.
    num = j
    edge_sum = 0
    binom_prod = one(R)
    @inbounds for i in E:-1:1
        r = edge_reps[i]
        b = r + 1
        k = num % b
        num ÷= b
        kept[i] = k
        edge_sum += k
        binom_prod *= binoms[r+1, k+1]
    end

    # Glynn substitution, and collect the surviving edges.
    mE = 0
    @inbounds for i in 1:E
        k = glynn ? 2 * kept[i] - edge_reps[i] : kept[i]
        kept[i] = k
        if k != 0
            mE += 1
            idx[mE] = i
        end
    end
    m = 2mE
    m == 0 && return zero(T)     # empty submatrix: det(I - λ·[]) = 1 has no λⁿ term for n ≥ 1

    prefac = isodd(n - edge_sum) ? -binom_prod : binom_prod
    # The δ → -δ symmetry pairs term j with its mirror; the fixed point (leading digit 0) is its
    # own mirror and must not be double counted.
    @inbounds if glynn && kept[1] == 0
        prefac /= 2
    end

    # Rows/cols of `Ax` kept by this term: edge i owns row i and row i+E.
    @inbounds for a in 1:mE
        rows[a] = idx[a]
        rows[mE+a] = idx[a] + E
    end

    # M = A_S X D: swap the two halves of the columns and scale column pair `b` by its multiplicity.
    M = ws.M
    @inbounds for b in 1:mE
        kb = kept[idx[b]]
        cb1 = idx[b] + E
        cb2 = idx[b]
        for a in 1:m
            ra = rows[a]
            M[a, b] = kb * Ax[ra, cb1]
            M[a, mE+b] = kb * Ax[ra, cb2]
        end
    end

    _reduce_hessenberg!(ws, m)
    _charpoly_hessenberg!(ws.C, M, m, n)
    # Row m+1 of C is det(λI - M) in descending powers, i.e. det(I - λM) in ascending ones.
    q = @view ws.C[m+1, :]
    val = _inv_sqrt_det_coeff!(ws.f, q, m, n)

    return prefac * val
end

# ---------------------------------------------------------------------------------------------
# Sieve driver
# ---------------------------------------------------------------------------------------------

function _haf_range(
    ws::HafnianWorkspace{T},
    Ax::Matrix{T},
    edge_reps::Vector{Int},
    n::Int,
    lo::Int,
    hi::Int,
    binoms::Matrix{R},
    glynn::Val,
) where {T,R}
    H = zero(T)
    for j in lo:hi
        H += _glynn_term!(ws, Ax, edge_reps, n, j, binoms, glynn)
    end
    return H
end

# Rough multiply-add count of one sieve term, used to decide whether threading is worth it.
_term_cost(E::Int, n::Int) = (5 * (2E)^3) ÷ 3 + 2E * n * n

"""
    _sieve_chunks(work, steps, nthreads) -> Int

How many tasks the sieve would split into, given its exact multiply-add count and term count.

Spawning costs a few microseconds per task and the chunks are equal-sized, so on a machine with
uneven cores a short parallel run waits on its slowest chunk and loses to running serially. The
threshold was measured on the `benchmark/competitors` sweep: below it, every thread count tested was
slower than one; above it, using all of them was fastest.

`work` must come from [`_sieve_variant_cost`](@ref), not from `steps × _term_cost(E, n)`. The two
differ whenever multiplicities can vanish, and using the cruder figure here made the *threading*
decision disagree between the two sieve variants: at `rpt = fill(2, 8)` Glynn does a third less work
yet fell below the threshold while inclusion–exclusion rose above it, so Glynn ran serially, lost on
wall clock, and dragged the variant choice with it.

Also consulted by [`_choose_method`](@ref), since the sieve is the only strategy that threads.
"""
@inline _sieve_chunks(work::Int, steps::Int, nthreads::Int) =
    work < 90_000 ? 1 : min(nthreads, steps)

"""
    _calc_hafnian(Ax, edge_reps; nthreads, glynn = false) -> T

Run the sieve over `Ax`, a matrix already permuted so that the fixed perfect matching pairs row `i`
with row `i + E`.

The odometer runs over `∏(edge_reps .+ 1)` multiplicity vectors. Under `glynn` the leading digit is
clipped to its lower half to exploit the `δ → -δ` symmetry and the sum is scaled by `2^{-(n-1)}` at
the end; otherwise the multiplicities are plain inclusion–exclusion counts.

# Choosing between them

Which is faster depends entirely on whether rows repeat, and the two effects pull opposite ways.
Inclusion–exclusion always runs more terms — 2× for distinct rows, 1.5× at `rpt = 2` — but its
multiplicities can hit zero, which drops edges and shrinks the submatrix each term works on. For
distinct rows that shrinkage is decisive (Glynn's `±1` multiplicities never vanish, so every term
runs at full size); once rows repeat, Glynn's multiplicities can vanish too and only the term count
is left, which it wins.

Measured on 12 threads:

| case      | rpt1 N=20 | rpt1 N=24 | rpt1 N=28 | rpt2 N=20 | rpt2 N=24 | rpt2 N=28 |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| inclexcl  | 1.32×     | 1.47×     | 1.71×     | 0.60×     | 0.62×     | 0.63×     |

Accuracy runs the other way. Inclusion–exclusion sums terms far larger than the answer and relies on
cancellation, so it loses one to two decimal digits: against a `BigFloat` reference, Glynn holds
`1e-14`–`5e-14` while inclusion–exclusion runs `3e-13`–`3e-11`, a 24–744× spread and 105× on the
worst of twelve draws at `N = 20`. Both are far behind the direct strategies, which sit at machine
precision (`2e-16`) because they never cancel at all.

Which is why the caller does not pick: [`_choose_method`](@ref) costs both and returns the cheaper,
so distinct rows get inclusion–exclusion and repeated rows get Glynn, which there is both faster
*and* more accurate. Pass `glynn = true` explicitly when the extra digits matter more than the time.
"""
function _calc_hafnian(
    Ax::Matrix{T},
    edge_reps::Vector{Int};
    nthreads::Int = nthreads(),
    glynn::Bool = false,
) where {T}
    E = length(edge_reps)
    E == 0 && return one(T)
    n = sum(edge_reps)          # half the total number of rows, i.e. the coefficient we extract

    work, steps = _sieve_variant_cost(edge_reps, n, glynn)

    R = real(T)
    maxrep = maximum(edge_reps)
    binoms = Matrix{R}(undef, maxrep + 1, maxrep + 1)
    for a in 0:maxrep, b in 0:maxrep
        binoms[a+1, b+1] = b <= a ? R(binomial(a, b)) : zero(R)
    end

    nchunks = _sieve_chunks(work, steps, nthreads)

    # Branch here rather than computing `glynn ? Val(true) : Val(false)`: that would be a
    # `Union{Val{true},Val{false}}`, which makes the sieve call a dynamic dispatch and leaves the
    # whole return type uninferrable all the way out through `hafnian`.
    return if glynn
        _sieve_sum(Ax, edge_reps, n, steps, binoms, nchunks, Val(true))
    else
        _sieve_sum(Ax, edge_reps, n, steps, binoms, nchunks, Val(false))
    end
end

# Sum the sieve, with `glynn` a compile-time constant so the final rescaling folds away.
function _sieve_sum(
    Ax::Matrix{T},
    edge_reps::Vector{Int},
    n::Int,
    steps::Int,
    binoms::Matrix{R},
    nchunks::Int,
    ::Val{glynn},
) where {T,R,glynn}
    E = length(edge_reps)
    H = if nchunks == 1
        _haf_range(HafnianWorkspace{T}(E, n), Ax, edge_reps, n, 0, steps - 1, binoms, Val(glynn))
    else
        tasks = map(1:nchunks) do c
            lo = div((c - 1) * steps, nchunks)
            hi = div(c * steps, nchunks) - 1
            @spawn _haf_range(HafnianWorkspace{T}(E, n), Ax, edge_reps, n, lo, hi, binoms, Val(glynn))
        end
        # `fetch` is untyped, so annotate rather than let the sum widen to `Any`.
        sum(t -> fetch(t)::T, tasks)
    end
    return glynn ? H * R(0.5)^(n - 1) : H
end

# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------

_haf_eltype(A::AbstractMatrix) = float(eltype(A))

"""
    _check_symmetric(A)

Validate that `A` is symmetric to within `isapprox`'s default tolerance, and 1-based.

This runs before every hafnian and is `O(N²)` against kernels that can be as short as 50 ns, so it
is written to be cheap rather than obvious:

  * an exact `==` fast path, which is what actually fires in practice — a matrix built the usual way
    as `B + transpose(B)` has bitwise-identical mirrored entries, since floating-point addition is
    commutative. It also keeps `Inf` entries comparing equal, as `isapprox` does.
  * for anything else, the tolerance test squared. `isapprox` with the default `rtol` and `atol = 0`
    asks `|x - y| ≤ √eps · max(|x|, |y|)`; both sides are non-negative, so comparing squares is
    equivalent and replaces three `hypot` calls per entry pair with plain multiplications. The
    squared tolerance is exactly `eps`.

Symmetry is a property of the mathematical object here, not merely of the storage, so it is checked
rather than assumed — but see the `Symmetric` method below, which knows the answer already.
"""
function _check_symmetric(A::AbstractMatrix)
    # Every index in this package is 1-based, and the kernels read `A` under `@inbounds`, so an
    # offset array would silently read out of bounds rather than merely give a wrong answer.
    Base.require_one_based_indexing(A)
    n = size(A, 1)
    tol = eps(float(real(eltype(A))))
    @inbounds for j in 1:n, i in 1:j-1
        x = A[i, j]
        y = A[j, i]
        x == y && continue
        abs2(x - y) <= tol * max(abs2(x), abs2(y)) ||
            throw(ArgumentError("matrix is not symmetric at ($i, $j)"))
    end
    return nothing
end

# `Symmetric` mirrors one triangle on read, so it satisfies this by construction. `Hermitian` does
# not (conjugation is not transposition for complex entries) and deliberately falls through above.
_check_symmetric(A::Symmetric) = (Base.require_one_based_indexing(A); nothing)

# Build the reordered matrix `Ax[a, b] = A[x[a], x[b]]` the sieve works on.
function _permuted(::Type{T}, A::AbstractMatrix, x::Vector{Int}) where {T}
    M = Matrix{T}(undef, length(x), length(x))
    @inbounds for b in eachindex(x), a in eachindex(x)
        M[a, b] = A[x[a], x[b]]
    end
    return M
end

"""
    _kernel_matrix(T, A) -> AbstractMatrix

The array the direct kernels should read, promoted to `T` only if it is not already that element
type.

Anything already holding `T` is handed through untouched — in particular views, which the kernels
index perfectly well and which must not be silently copied. Only a genuine element-type change
materialises anything, and there it is unavoidable: computing an integer matrix's hafnian in `Int`
and converting at the end would overflow where the promoted arithmetic does not.

Dispatching on the element type rather than branching keeps the result concretely typed at every
call site, so no dynamic dispatch leaks into the kernels.
"""
_kernel_matrix(::Type{T}, A::AbstractMatrix{T}) where {T} = A
_kernel_matrix(::Type{T}, A::AbstractMatrix) where {T} = Matrix{T}(A)

"""
    _prefer_unrolled(K, sieve_work) -> Bool

Decide between the unrolled kernel of [`_haf_direct`](@ref) and the sieve, given the total degree
`K` and the sieve's wall-clock-equivalent multiply-add count.

Comparing the two counts directly is a calibration, not a cost model: their per-operation costs
differ (the unrolled kernel spills its live subset values to L1-resident stack, the sieve's
reduction vectorises) and the sieve also carries a fixed ~0.5 µs of setup that no work count sees.
Parity between the two counts is simply where the measured crossover sits, checked against the
sweep in `test/runtests.jl`.

This only ever matters for heavily repeated inputs, where enough repetition shrinks the sieve below
the cost of enumerating matchings — `rpt = [6, 6]` sieves, `rpt = [5, 5, 1, 1]` does not. For
distinct rows the unrolled kernel wins by one to two orders of magnitude at every degree it covers.
"""
@inline _prefer_unrolled(K::Int, sieve_work::Int) =
    K <= UNROLL_MAX && _unrolled_muls(K) < sieve_work

# Work counts can run past `typemax(Int)` for degrees nobody will ever sieve; saturate rather than
# wrap, since every consumer only compares them.
_clamp_work(x::Float64) = x >= 9.0e18 ? typemax(Int) : round(Int, x)

"""
    _sieve_variant_cost(edge_reps, n, glynn) -> (work, steps)

Exact multiply-add count and term count for one sieve variant.

Not `steps × _term_cost(E, n)`: terms do *not* all cost the same. An edge whose multiplicity lands
on zero drops out of that term's submatrix, and `_term_cost` is cubic in how many survive, so the
distribution matters more than the count. That distribution is a product over edges, so this
convolves it exactly in `O(E²)` — `counts[c+1]` ends up holding how many terms keep `c` edges.

The two variants differ in how often a multiplicity vanishes:

  * inclusion–exclusion draws each multiplicity from `0:rᵢ`, so exactly one of the `rᵢ+1` choices is
    zero, always;
  * Glynn draws `2k - rᵢ`, which is zero only when `rᵢ` is *even*. With odd multiplicities — the
    all-distinct case, where every `rᵢ = 1` — no edge ever drops and every term runs at full size.

That is the whole story of which variant wins. For distinct rows Glynn pays full size on every term
and loses despite running half as many; once rows repeat with even counts both drop edges at the
same rate, the sizes match, and Glynn's smaller term count decides it.
"""
function _sieve_variant_cost(edge_reps::Vector{Int}, n::Int, glynn::Bool)
    E = length(edge_reps)
    E == 0 && return (0, 0)

    counts = zeros(Float64, E + 1)          # counts[c+1]: terms keeping c edges so far
    counts[1] = 1.0
    for i in 1:E
        r = edge_reps[i]
        # Glynn clips the leading digit to its lower half; the zero multiplicity, when it exists,
        # sits at the top of that clipped range and so survives the clipping.
        total = glynn ? (i == 1 ? (r + 2) ÷ 2 : r + 1) : r + 1
        nzero = glynn ? (iseven(r) ? 1 : 0) : 1
        nkeep = total - nzero
        for c in i-1:-1:0
            v = counts[c+1]
            v == 0.0 && continue
            counts[c+1] = v * nzero
            counts[c+2] += v * nkeep
        end
    end

    work = 0.0
    steps = 0.0
    for c in 0:E
        counts[c+1] == 0.0 && continue
        steps += counts[c+1]
        work += counts[c+1] * _term_cost(c, n)   # c = 0 costs nothing: the term is empty
    end
    return (_clamp_work(work), _clamp_work(steps))
end

# `hafnian` has all-ones multiplicities and prices them without materialising the vector, which is
# what keeps its unrolled path allocation-free. Same quantities as `_sieve_variant_cost`, closed
# form: Glynn never drops an edge, inclusion–exclusion drops each independently with probability ½.
function _sieve_variant_cost_distinct(E::Int, glynn::Bool)
    E == 0 && return (0, 0)
    if glynn
        steps = 2.0^(E - 1)
        return (_clamp_work(steps * _term_cost(E, E)), _clamp_work(steps))
    end
    work = 0.0
    for c in 0:E
        work += binomial(E, c) * Float64(_term_cost(c, E))
    end
    return (_clamp_work(work), _clamp_work(2.0^E))
end

"""
    _best_sieve_variant(costs...) -> Bool

Whether Glynn beats inclusion–exclusion, given each one's exact work.
"""
@inline _best_sieve_variant(work_glynn::Int, work_inclexcl::Int) = work_glynn <= work_inclexcl

"""
    _choose_method(K, edge_reps, nthreads) -> (method, glynn)

Pick between `:unrolled`, `:dp` and `:sieve` for a problem of total degree `K` whose sieve would run
over `edge_reps`, and pick which sieve variant to use if it comes to that.

All three compute the same thing; only their costs differ, and those costs are known in advance
(see [`_prefer_unrolled`](@ref) and [`_prefer_dp`](@ref)). The sieve's count is divided by the
number of tasks it would spread over, since it is the only one of the three that threads — with one
thread available the other two win more often, which is the intended behaviour.

The sieve is priced at whichever of its two variants is cheaper, and that variant is returned
alongside the method so the caller runs the one that was costed. See [`_sieve_variant_cost`](@ref).

Known limits, from a 24-case measured sweep — two picks are wrong, both by under 1.6×:

  * `rpt = fill(2, 9)` takes the sieve where the DP is 1.58× faster. Unavoidable with a linear cost
    model; see [`_prefer_dp`](@ref) for why the two classes overlap.
  * `rpt = fill(7, 4)` takes Glynn where inclusion–exclusion is 1.37× faster. With only two edges
    and large odd multiplicities, `_term_cost`'s cubic-in-size shape misjudges terms that shrink to
    a 2×2 submatrix, where per-term overhead rather than arithmetic dominates.

The unrolled kernel is tested first and so keeps every degree it covers, even though the DP edges it
out by ~13% on a `min`-of-many microbenchmark at `K = 12`. That microbenchmark is misleading: the DP
allocates its state array on every call (5 KB at `K = 12`, against 144 bytes), and once the same
call is made repeatedly with GC time counted the unrolled kernel is ahead again at both `K = 10` and
`K = 12`.
"""
function _choose_method(K::Int, edge_reps::Vector{Int}, nthreads::Int)
    isempty(edge_reps) && return (method = :sieve, glynn = false)
    n = sum(edge_reps)
    wg, sg = _sieve_variant_cost(edge_reps, n, true)
    wi, si = _sieve_variant_cost(edge_reps, n, false)
    return _decide(K, wg, sg, wi, si, length(edge_reps), n, nthreads)
end

"""
    _choose_method_distinct(K, E, nthreads) -> (method, glynn)

As above for all-ones multiplicities, priced in closed form so [`hafnian`](@ref) never has to
materialise the vector — which is what keeps its unrolled path allocation-free.
"""
function _choose_method_distinct(K::Int, E::Int, nthreads::Int)
    E == 0 && return (method = :sieve, glynn = false)
    wg, sg = _sieve_variant_cost_distinct(E, true)
    wi, si = _sieve_variant_cost_distinct(E, false)
    return _decide(K, wg, sg, wi, si, E, E, nthreads)
end

"""
Modelled ceiling on each strategy's parallel speedup.

Neither scales with the core count: both are limited by memory rather than arithmetic, and on twelve
threads the sieve reached ~5× and the DP ~4×. Treating the sieve as scaling linearly in `nthreads` —
as this did originally — over-credits it more and more as the machine gets bigger, and on a 32-thread
box it flipped every distinct-row problem from `:dp` to `:sieve`, which is roughly 4× slower there.
Capping both makes the choice essentially thread-count independent, which is the property worth
having: the same problem should not change strategy because somebody exported `JULIA_NUM_THREADS`.
"""
const _SIEVE_PARALLEL_CAP = 5
const _DP_PARALLEL_CAP = 4

@inline _sieve_speedup(nthreads::Int, steps::Int) = max(1, min(nthreads, steps, _SIEVE_PARALLEL_CAP))
@inline _dp_speedup(nthreads::Int) = max(1, min(nthreads, _DP_PARALLEL_CAP))

function _decide(K::Int, wg::Int, sg::Int, wi::Int, si::Int, E::Int, n::Int, nthreads::Int)
    glynn = _best_sieve_variant(wg, wi)
    work, steps = glynn ? (wg, sg) : (wi, si)
    # The sieve only actually threads once it is big enough to be worth chunking.
    parallel = _sieve_chunks(work, steps, nthreads) > 1
    effective = parallel ? work ÷ _sieve_speedup(nthreads, steps) : work
    _prefer_unrolled(K, effective) && return (method = :unrolled, glynn = glynn)
    _prefer_dp(K, effective, parallel ? nthreads : 1) && return (method = :dp, glynn = glynn)
    return (method = :sieve, glynn = glynn)
end

"""
    _steer_backend(chosen, glynn, backend, K) -> Symbol

Reconcile the cost-model's choice with the presence of a GPU backend.

Only the DP has a device path, so a backend genuinely changes which strategy is right — the cost
model compares CPU costs and cannot see the device at all. When the caller supplies a backend and
has not forced a `method`, the DP is used wherever the degree allows it.

The warning matters as much as the steering. Before this existed, a backend that could not be used
was ignored in silence: on a 32-thread machine the cost model flipped every distinct-row problem to
`:sieve`, so an entire GPU benchmark run measured the CPU sieve against itself and reported ~1.0×
speedups with sieve-level accuracy. Silence turned a one-line policy gap into a whole invalid run.
"""
function _steer_backend(chosen::Symbol, backend, K::Int, method::Symbol)
    backend === nothing && return chosen
    # Never override the unrolled kernel. It finishes in tens of nanoseconds, against ~21 µs for a
    # single kernel launch, so sending those degrees to a device is a three-order-of-magnitude loss.
    chosen === :unrolled && return chosen
    method === :auto && K <= DP_MAX && return :dp
    if chosen !== :dp
        @warn "`backend` was given but the $(chosen === :unrolled ? "unrolled kernel" : "sieve") " *
              "was selected, which has no device path; this call ran on the CPU. Degree $K, " *
              "method=:$method." maxlog = 1
    end
    return chosen
end

# Expanded index list `idx` with row `i` repeated `rpt[i]` times, as both direct kernels want it.
function _expanded_indices(rpt::AbstractVector{<:Integer}, total::Int)
    idx = Vector{Int}(undef, total)
    n = 0
    @inbounds for i in eachindex(rpt), _ in 1:rpt[i]
        idx[n+=1] = Int(i)
    end
    return idx
end

# Run whichever direct (non-sieve) kernel was selected. Only the DP uses `nthreads`; the unrolled
# kernels are a single straight-line expression with nothing to spread.
function _haf_direct_method(
    ::Type{T},
    method::Symbol,
    A::AbstractMatrix,
    idx::AbstractVector{Int},
    nthreads::Int,
    backend = nothing,
) where {T}
    Am = _kernel_matrix(T, A)
    if method === :unrolled
        # Tens of nanoseconds of straight-line code: a kernel launch alone would cost more, so a
        # backend is ignored here rather than honoured.
        return T(_haf_direct(Am, idx))
    end
    # Branch on the plan layout so each call is concretely typed; a single `_dp_plan(K)` would be
    # abstractly typed and cost an inferrable return type. See `_dp_index_type`.
    K = length(idx)
    if backend !== nothing
        # One instance is the degenerate batch. Sharing the batched path keeps a single kernel.
        P = _gather_pairs_batch(T, Am, [idx], K)
        if K <= DP_MAX
            return T(_haf_dp_backend(backend, P, _dp_plan(K, Int32))[1])
        else
            return T(_haf_dp_backend(backend, P, _dp_plan(K, Int64))[1])
        end
    end
    if K <= DP_MAX
        return T(_haf_dp(Am, idx, _dp_plan(K, Int32); nthreads))
    else
        return T(_haf_dp(Am, idx, _dp_plan(K, Int64); nthreads))
    end
end

function _check_method(method::Symbol, K::Int)
    method in (:auto, :unrolled, :dp, :sieve) ||
        throw(ArgumentError("method must be :auto, :unrolled, :dp or :sieve, got :$method"))
    method === :unrolled && K > UNROLL_MAX &&
        throw(ArgumentError("method=:unrolled needs degree ≤ $UNROLL_MAX, got $K"))
    method === :dp && K > DP_HARD_MAX &&
        throw(ArgumentError("method=:dp needs degree ≤ $DP_HARD_MAX, got $K"))
    return nothing
end

"""
    hafnian(A; nthreads=Threads.nthreads(), glynn=nothing, method=:auto, check_symmetric=true) -> Number

Hafnian of the square symmetric matrix `A`: the sum over all perfect matchings of
``∏_{(i,j)} A[i,j]``.

The diagonal of `A` is ignored (this is the plain hafnian, not the loop hafnian). An odd-sized
matrix has no perfect matching and gives `0`; a `0 × 0` matrix gives `1`.

Three strategies compute this, chosen automatically by comparing their known costs, and `method`
overrides the choice:

  * `:unrolled` — the matching sum emitted as straight-line code, for degrees up to `UNROLL_MAX`.
  * `:dp` — the same recursion evaluated over memoised subsets. Usually the fastest option in
    between, by a wide margin. Chosen automatically up to `DP_MAX`; requesting it explicitly goes
    up to `DP_HARD_MAX` on a wider layout, which stays faster than the sieve but needs a plan
    measured in gigabytes (see [`DP_HARD_MAX`](@ref)).
  * `:sieve` — the ``O(n³ 2ⁿ)`` Björklund/Glynn sieve, parallelised over `nthreads` tasks when the
    problem is large enough to pay for them. The fallback above `DP_MAX`, and the best choice when
    repeated rows shrink it far enough.

`glynn` selects the sieve variant and has no effect on the other two. Left at `nothing` it is chosen
per problem like the strategy is: inclusion–exclusion for distinct rows, Glynn once rows repeat.
Setting it forces one — `true` for the Glynn finite-difference sieve, which is also one to two
decimal digits more accurate. See [`_calc_hafnian`](@ref) for the measured trade.

`A` is read in place: views, `Symmetric` wrappers and other `AbstractMatrix`es holding the result
element type are never copied. Indices must be 1-based.

`check_symmetric=false` skips the `O(N²)` symmetry validation, which at small `N` costs a real fraction of the
whole call — about a third of it at `N = 8`. Pass it only when the caller already knows `A` is
symmetric; with it, only the entries the chosen strategy happens to read are consulted, so an
asymmetric matrix gives a silently strategy-dependent answer rather than an error.

# Examples
```jldoctest
julia> hafnian([0 1 2 3; 1 0 4 5; 2 4 0 6; 3 5 6 0])
28.0
```
which is `A[1,2]*A[3,4] + A[1,3]*A[2,4] + A[1,4]*A[2,3] = 1*6 + 2*5 + 3*4`.

See also [`hafnian_repeated`](@ref).
"""
function hafnian(
    A::AbstractMatrix;
    nthreads::Int = nthreads(),
    glynn::Union{Nothing,Bool} = nothing,
    method::Symbol = :auto,
    check_symmetric::Bool = true,
    backend = nothing,
)
    N = LinearAlgebra.checksquare(A)
    T = _haf_eltype(A)
    N == 0 && return one(T)
    isodd(N) && return zero(T)
    _check_method(method, N)
    check_symmetric ? _check_symmetric(A) : Base.require_one_based_indexing(A)

    # All edge multiplicities are 1 here, so the sieve would run `2^(E-1)` terms; pricing that
    # directly avoids allocating the vector on the paths that never sieve.
    E = N ÷ 2
    sel = _choose_method_distinct(N, E, nthreads)
    chosen = _steer_backend(method === :auto ? sel.method : method, backend, N, method)
    use_glynn = glynn === nothing ? sel.glynn : glynn
    if chosen === :unrolled
        # Distinct rows in their natural order, so no index vector needs materialising.
        return T(_haf_unrolled_range(_kernel_matrix(T, A), N))
    elseif chosen === :dp
        # `1:N` rather than a materialised vector: the kernels only ever index it.
        return _haf_direct_method(T, :dp, A, Base.OneTo(N), nthreads, backend)
    end

    # Match vertex 2i-1 with 2i, then reorder into the [first halves; second halves] layout.
    x = Vector{Int}(undef, N)
    @inbounds for i in 1:E
        x[i] = 2i - 1
        x[E+i] = 2i
    end
    return _calc_hafnian(_permuted(T, A, x), ones(Int, E); nthreads, glynn = use_glynn)
end

# hafnian(A) on the full index range 1:N — no index vector needs materialising.
@generated function _haf_unrolled_range(A::AbstractMatrix, N::Int)
    branches = Expr(:block)
    for K in 0:2:UNROLL_MAX
        push!(branches.args, :(N == $K && return _haf_unrolled(A, ntuple(identity, Val($K)))))
    end
    quote
        $branches
        throw(ArgumentError("degree $N has no unrolled kernel (UNROLL_MAX = $(UNROLL_MAX))"))
    end
end

"""
    hafnian_repeated(A, rpt; nthreads=Threads.nthreads(), glynn=nothing, method=:auto, check_symmetric=true) -> Number

Hafnian of the matrix obtained by repeating row and column `i` of `A` exactly `rpt[i]` times, i.e.
`hafnian(reduction(A, rpt))`, but without ever forming that larger matrix.

The cost is driven by the number of *distinct* rows rather than the total ``N = Σ rpt``, so this is
dramatically faster than [`hafnian`](@ref) on the expanded matrix whenever rows repeat. Unlike the
plain hafnian the diagonal of `A` matters here: with `rpt[i] ≥ 2`, `A[i,i]` is an off-diagonal entry
of the expanded matrix.

Returns `1` when `sum(rpt) == 0` and `0` when `sum(rpt)` is odd.

`method` selects among the same three strategies as [`hafnian`](@ref). Repetition is what makes the
sieve cheap, so it wins here far more often than it does for distinct rows: `rpt = [6, 6]` sieves
where `rpt = [5, 5, 1, 1]` does not, and `fill(2, 14)` sieves where 28 distinct rows would not.

`A` is read in place and `check_symmetric=false` skips symmetry validation, exactly as for
[`hafnian`](@ref).

# Examples
```jldoctest
julia> A = [1.0 2.0; 2.0 3.0];

julia> hafnian_repeated(A, [2, 2])
11.0

julia> hafnian(TheEggman.reduction(A, [2, 2]))
11.0
```

See also [`hafnian`](@ref), [`reduction`](@ref).
"""
function hafnian_repeated(
    A::AbstractMatrix,
    rpt::AbstractVector{<:Integer};
    nthreads::Int = nthreads(),
    glynn::Union{Nothing,Bool} = nothing,
    method::Symbol = :auto,
    check_symmetric::Bool = true,
    backend = nothing,
)
    N = LinearAlgebra.checksquare(A)
    length(rpt) == N || throw(DimensionMismatch("rpt has length $(length(rpt)), expected $N"))
    any(<(0), rpt) && throw(ArgumentError("rpt must contain non-negative integers"))
    T = _haf_eltype(A)

    total = Int(sum(rpt; init = 0))
    total == 0 && return one(T)
    isodd(total) && return zero(T)
    _check_method(method, total)
    check_symmetric ? _check_symmetric(A) : Base.require_one_based_indexing(A)

    x, edge_reps = matched_reps(rpt)
    sel = _choose_method(total, edge_reps, nthreads)
    chosen = _steer_backend(method === :auto ? sel.method : method, backend, total, method)
    use_glynn = glynn === nothing ? sel.glynn : glynn
    if chosen !== :sieve
        return _haf_direct_method(T, chosen, A, _expanded_indices(rpt, total), nthreads, backend)
    end
    return _calc_hafnian(_permuted(T, A, x), edge_reps; nthreads, glynn = use_glynn)
end

# Evaluate `f(b)` for `b in 1:B` into `out`, spread across tasks when that is worth it. Each
# instance runs single-threaded, so the two levels of parallelism never oversubscribe.
function _batch_cpu!(f, out::Vector{T}, B::Int, nthreads::Int) where {T}
    nchunks = min(nthreads, B)
    if nchunks <= 1
        @inbounds for b in 1:B
            out[b] = f(b)
        end
        return out
    end
    @sync for c in 1:nchunks
        lo = div((c - 1) * B, nchunks) + 1
        hi = div(c * B, nchunks)
        lo <= hi && @spawn for b in lo:hi
            @inbounds out[b] = f(b)
        end
    end
    return out
end

_batch_eltype(As::AbstractVector{<:AbstractMatrix}) = float(eltype(eltype(As)))

"""
    hafnian(As; backend=nothing, kwargs...) -> Vector

Hafnians of a batch of equally-sized matrices, returned in order.

Every matrix must be the same size, since a batch shares one strategy and — on a GPU — one plan.
Without `backend` this is a threaded loop over [`hafnian`](@ref) and accepts all its keywords.

With `backend` set to a `KernelAbstractions` backend it becomes one batched GPU evaluation, which is
where batching earns its keep: the instance axis is the fastest-varying one in the device layout, so
a warp covering 32 instances of the same subproblem reads contiguous memory, and the per-level
kernel launches are amortised across the whole batch rather than paid per hafnian. Requires
`KernelAbstractions` (and a backend package) to be loaded, and falls back to the threaded loop for
any degree the subset DP does not cover.

See also [`hafnian_repeated`](@ref), which batches many repetition patterns against one matrix.
"""
function hafnian(
    As::AbstractVector{<:AbstractMatrix};
    nthreads::Int = nthreads(),
    glynn::Union{Nothing,Bool} = nothing,
    method::Symbol = :auto,
    check_symmetric::Bool = true,
    backend = nothing,
    max_batch::Union{Nothing,Int} = nothing,
)
    T = _batch_eltype(As)
    B = length(As)
    B == 0 && return T[]

    N = LinearAlgebra.checksquare(first(As))
    for (b, A) in enumerate(As)
        LinearAlgebra.checksquare(A) == N ||
            throw(DimensionMismatch("matrix $b is $(size(A, 1))×$(size(A, 2)), expected $N×$N; a " *
                                    "batch shares one strategy and one plan"))
        check_symmetric ? _check_symmetric(A) : Base.require_one_based_indexing(A)
    end

    N == 0 && return fill(one(T), B)
    isodd(N) && return fill(zero(T), B)
    _check_method(method, N)

    chosen = _steer_backend(method === :auto ? _choose_method_distinct(N, N ÷ 2, nthreads).method :
                            method, backend, N, method)
    if backend !== nothing && chosen === :dp
        idxs = fill(Base.OneTo(N), B)
        P = _gather_pairs_batch(T, As, idxs, N)
        return N <= DP_MAX ? _haf_dp_backend(backend, P, _dp_plan(N, Int32); max_batch) :
                             _haf_dp_backend(backend, P, _dp_plan(N, Int64); max_batch)
    end

    out = Vector{T}(undef, B)
    return _batch_cpu!(out, B, nthreads) do b
        hafnian(As[b]; nthreads = 1, glynn, method, check_symmetric = false)
    end
end

"""
    hafnian_repeated(A, rpts; backend=nothing, kwargs...) -> Vector

Hafnians of one matrix under many repetition patterns, returned in order.

This is the Gaussian boson-sampling shape: a single `A`, one `rpt` per photon pattern. Every pattern
must have the same total `sum(rpt)`, because that total is the degree and a batch shares one plan —
a mismatch throws and names the offending index. Group patterns by total if they differ.

Without `backend` this is a threaded loop over [`hafnian_repeated`](@ref), so each pattern still
picks its own strategy. With `backend` the whole batch runs the subset DP on the GPU, since one
shared plan is what makes batching worth doing; for heavily repeated patterns the CPU sieve may
still be faster, so compare before assuming the GPU wins.
"""
function hafnian_repeated(
    A::AbstractMatrix,
    rpts::AbstractVector{<:AbstractVector{<:Integer}};
    nthreads::Int = nthreads(),
    glynn::Union{Nothing,Bool} = nothing,
    method::Symbol = :auto,
    check_symmetric::Bool = true,
    backend = nothing,
    max_batch::Union{Nothing,Int} = nothing,
)
    T = _haf_eltype(A)
    B = length(rpts)
    B == 0 && return T[]

    N = LinearAlgebra.checksquare(A)
    total = Int(sum(first(rpts); init = 0))
    for (b, rpt) in enumerate(rpts)
        length(rpt) == N ||
            throw(DimensionMismatch("rpt $b has length $(length(rpt)), expected $N"))
        any(<(0), rpt) && throw(ArgumentError("rpt $b contains a negative entry"))
        Int(sum(rpt; init = 0)) == total ||
            throw(ArgumentError("rpt $b sums to $(Int(sum(rpt; init = 0))), but rpt 1 sums to " *
                                "$total; a batch shares one plan, so every pattern must have the " *
                                "same total"))
    end
    check_symmetric ? _check_symmetric(A) : Base.require_one_based_indexing(A)

    total == 0 && return fill(one(T), B)
    isodd(total) && return fill(zero(T), B)
    _check_method(method, total)

    if backend !== nothing && method in (:auto, :dp) && total <= DP_HARD_MAX
        idxs = [_expanded_indices(rpt, total) for rpt in rpts]
        P = _gather_pairs_batch(T, A, idxs, total)
        return total <= DP_MAX ? _haf_dp_backend(backend, P, _dp_plan(total, Int32); max_batch) :
                                 _haf_dp_backend(backend, P, _dp_plan(total, Int64); max_batch)
    end

    out = Vector{T}(undef, B)
    return _batch_cpu!(out, B, nthreads) do b
        hafnian_repeated(A, rpts[b]; nthreads = 1, glynn, method, check_symmetric = false)
    end
end

"""
    reduction(A, rpt) -> Matrix

Expand `A` by repeating row and column `i` exactly `rpt[i]` times, so that
`hafnian(reduction(A, rpt)) == hafnian_repeated(A, rpt)`.

Mostly useful for testing; the whole point of [`hafnian_repeated`](@ref) is to avoid building this.
"""
function reduction(A::AbstractMatrix, rpt::AbstractVector{<:Integer})
    LinearAlgebra.checksquare(A) == length(rpt) ||
        throw(DimensionMismatch("rpt must have one entry per row of A"))
    idx = reduce(vcat, [fill(i, rpt[i]) for i in eachindex(rpt)]; init = Int[])
    return A[idx, idx]
end
