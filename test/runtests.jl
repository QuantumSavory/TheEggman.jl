using TheEggman
using TheEggman: reduction, matched_reps
using LinearAlgebra
using Random
using Test

"""
Hafnian by direct enumeration of perfect matchings — exponentially slower than anything in the
package, but a completely independent definition to check against.
"""
function brute_hafnian(A::AbstractMatrix{T}) where {T}
    N = size(A, 1)
    N == 0 && return one(T)
    isodd(N) && return zero(T)
    function rec(rem)
        isempty(rem) && return one(T)
        i = rem[1]
        s = zero(T)
        for t in 2:length(rem)
            s += A[i, rem[t]] * rec([rem[k] for k in 2:length(rem) if k != t])
        end
        return s
    end
    return rec(collect(1:N))
end

randsym(rng, T, n) = (B = randn(rng, T, n, n); B + transpose(B))

"""
Allocations of one `f(args...)` call, after a warmup and behind a function barrier.

The barrier matters: measuring `@allocated f(container[k])` instead would count the boxing of the
result caused by the *test's* own dynamic dispatch, not anything `f` did.
"""
function alloc_of(f, args...)
    f(args...)
    return @allocated f(args...)
end

"""Count `*` calls anywhere in an expression tree."""
count_muls(x) = 0
function count_muls(e::Expr)
    n = (e.head === :call && e.args[1] === :*) ? 1 : 0
    for a in e.args
        n += count_muls(a)
    end
    return n
end

@testset "TheEggman.jl" begin
    @testset "hafnian: small closed forms" begin
        @test hafnian(zeros(0, 0)) == 1
        @test hafnian([0.0 1.0; 1.0 0.0]) == 1.0

        A = [0 1 2 3; 1 0 4 5; 2 4 0 6; 3 5 6 0]
        @test hafnian(A) ≈ 1 * 6 + 2 * 5 + 3 * 4

        # The plain hafnian never touches the diagonal.
        rng = MersenneTwister(1)
        B = randsym(rng, ComplexF64, 8)
        C = copy(B)
        for i in 1:8
            C[i, i] = 3B[i, i] + 7
        end
        @test hafnian(B) ≈ hafnian(C)
    end

    @testset "hafnian: odd sizes vanish" begin
        rng = MersenneTwister(2)
        for N in (1, 3, 5, 7)
            @test hafnian(randsym(rng, Float64, N)) == 0
        end
    end

    @testset "hafnian vs brute force" begin
        rng = MersenneTwister(3)
        for N in (2, 4, 6, 8, 10, 12)
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A) ≈ brute_hafnian(A)
            R = randsym(rng, Float64, N)
            @test hafnian(R) ≈ brute_hafnian(R)
        end
    end

    @testset "hafnian: element types" begin
        rng = MersenneTwister(4)
        A = randsym(rng, Float64, 8)
        @test hafnian(A) isa Float64
        @test hafnian(ComplexF64.(A)) isa ComplexF64
        @test hafnian(Float32.(A)) isa Float32
        @test hafnian(ComplexF32.(A)) isa ComplexF32
        # Integer input is promoted, not rejected.
        I8 = round.(Int, 10 .* A)
        @test hafnian(I8) ≈ brute_hafnian(Float64.(I8))
        # The generic (non-split-storage) kernels must agree with the fast ones.
        @test hafnian(Complex{BigFloat}.(A)) ≈ hafnian(ComplexF64.(A)) rtol = 1e-10
    end

    @testset "hafnian: inclusion–exclusion agrees with Glynn" begin
        rng = MersenneTwister(5)
        for N in (4, 8, 12)
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A; glynn = false, method = :sieve) ≈
                  hafnian(A; method = :sieve) rtol = 1e-9
        end
    end

    @testset "hafnian: threading is deterministic in value" begin
        rng = MersenneTwister(6)
        A = randsym(rng, ComplexF64, 14)
        ref = hafnian(A; nthreads = 1, method = :sieve)
        for nt in (2, 3, 8)
            @test hafnian(A; nthreads = nt, method = :sieve) ≈ ref rtol = 1e-10
        end
    end

    @testset "hafnian: Gaussian reduction matches Householder" begin
        # The sieve reduces with Gaussian similarity transforms (half the flops); confirm that
        # choice does not cost meaningful accuracy against the backward-stable Householder path
        # and against an extended-precision reference.
        rng = MersenneTwister(7)
        setprecision(BigFloat, 256) do
            for N in (10, 14, 18)
                A = randsym(rng, ComplexF64, N)
                ref = hafnian(Complex{BigFloat}.(A); nthreads = 1, method = :sieve)
                err = abs(hafnian(A; method = :sieve) - ref) / abs(ref)
                @test err < 1e-11
            end
        end
    end

    @testset "Hessenberg reductions agree on the characteristic polynomial" begin
        # `_hessenberg!` (Gaussian, half the flops) is what the sieve runs; check it against the
        # unconditionally stable Householder reduction on the quantity that actually gets used.
        rng = MersenneTwister(13)
        for m in (4, 8, 12, 20)
            n = m ÷ 2
            M0 = randn(rng, ComplexF64, m, m)
            v = Vector{ComplexF64}(undef, m)
            w = Vector{ComplexF64}(undef, m)

            Cg = Matrix{ComplexF64}(undef, m + 1, n + 1)
            Mg = copy(M0)
            TheEggman._hessenberg!(Mg, m, v)
            TheEggman._charpoly_hessenberg!(Cg, Mg, m, n)

            Ch = Matrix{ComplexF64}(undef, m + 1, n + 1)
            Mh = copy(M0)
            TheEggman._hessenberg_householder!(Mh, m, v, w)
            TheEggman._charpoly_hessenberg!(Ch, Mh, m, n)

            @test Cg[m+1, :] ≈ Ch[m+1, :] rtol = 1e-9

            # ... and against the coefficients LinearAlgebra's eigenvalues imply, since
            # det(I - λM) = ∏(1 - λ μᵢ).
            μ = eigvals(M0)
            poly = [1.0 + 0im]
            for z in μ
                poly = vcat(poly, 0im) .+ vcat(0im, -z .* poly)
            end
            @test Cg[m+1, 1:n+1] ≈ poly[1:n+1] rtol = 1e-8
        end
    end

    @testset "all three strategies agree" begin
        rng = MersenneTwister(14)
        for N in 2:2:TheEggman.UNROLL_MAX
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A; method = :unrolled) ≈ hafnian(A; method = :sieve) rtol = 1e-10
            @test hafnian(A; method = :dp) ≈ hafnian(A; method = :sieve) rtol = 1e-10
            R = randsym(rng, Float64, N)
            @test hafnian(R; method = :unrolled) ≈ hafnian(R; method = :sieve) rtol = 1e-10
            @test hafnian(R; method = :dp) ≈ hafnian(R; method = :sieve) rtol = 1e-10
        end
        # Above the unrolled cap only the DP and the sieve remain.
        for N in (14, 16, 18)
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A; method = :dp) ≈ hafnian(A; method = :sieve) rtol = 1e-9
        end
        # Repeated indices exercise the same kernels through a different entry point, and are the
        # only case where the unrolled path reads the diagonal of `A`.
        for rpt in ([2, 2, 2], [1, 2, 3, 2], [2, 2, 2, 2, 2, 2], [6, 6], [3, 3, 3, 3], [8, 4],
                    [5, 5, 1, 1], [10, 2], [12], [4, 4], [9, 3], [6, 2, 2, 2])
            d = length(rpt)
            A = randsym(rng, ComplexF64, d)
            @test hafnian_repeated(A, rpt; method = :unrolled) ≈
                  hafnian_repeated(A, rpt; method = :sieve) rtol = 1e-10
            @test hafnian_repeated(A, rpt; method = :dp) ≈
                  hafnian_repeated(A, rpt; method = :sieve) rtol = 1e-10
            @test hafnian_repeated(A, rpt) ≈ brute_hafnian(reduction(A, rpt)) rtol = 1e-10
        end
    end

    @testset "DP_MAX is exactly the packing limit" begin
        # `DP_MAX` is set by the packed-Int32 transition, not by the sieve catching up, so assert
        # that boundary directly — cheaply, from the closed-form counts rather than by building the
        # 196 MB plan.
        bits(x) = ndigits(x, base = 2)
        K = TheEggman.DP_MAX
        ns, _ = TheEggman._dp_counts(K)
        npairs = K * (K - 1) ÷ 2
        @test npairs <= TheEggman._pair_mask(Int32)
        @test ns <= typemax(Int32) >> TheEggman._pair_bits(Int32)
        @test bits(npairs) + bits(ns) <= 31

        # The next even degree overflows an Int32 transition under *any* split of the bits, which
        # is what makes this the cap rather than a tunable.
        ns2, _ = TheEggman._dp_counts(K + 2)
        npairs2 = (K + 2) * (K + 1) ÷ 2
        @test bits(npairs2) + bits(ns2) > 31
    end

    @testset "DP plan structure" begin
        # Both layouts must satisfy the same invariants; only the packing width differs.
        for I in (Int32, Int64), K in (4, 8, 12, 16)
            plan = TheEggman._dp_plan(K, I)
            nstates, ntrans = TheEggman._dp_counts(K)
            @test plan isa TheEggman.HafnianDPPlan{I}
            @test eltype(plan.trans) === I
            @test eltype(plan.starts) === I
            @test eltype(plan.levels) === I
            # The closed-form counts drive the strategy choice before any plan exists, so they must
            # match what enumeration actually produces.
            @test plan.nstates == nstates
            @test length(plan.trans) == ntrans
            @test plan.starts[1] == 1
            @test plan.starts[end] == ntrans + 1
            @test issorted(plan.starts)
            # A single forward pass is only valid if every state's children precede it.
            children(s) = plan.trans[plan.starts[s]:plan.starts[s+1]-1] .>> TheEggman._pair_bits(I)
            @test all(s -> all(children(s) .< s), 2:nstates)
            # State 1 is the empty set (no transitions) and the last state is the full set.
            @test plan.starts[2] == 1
            @test plan.starts[nstates+1] - plan.starts[nstates] == K - 1
            @test all(1 .<= (plan.trans .& TheEggman._pair_mask(I)) .<= K * (K - 1) ÷ 2)
        end
        @test TheEggman._dp_plan(8) === TheEggman._dp_plan(8)   # cached, not rebuilt
        @test TheEggman._dp_plan(8, Int32) !== TheEggman._dp_plan(8, Int64)  # separate caches
    end

    @testset "wide Int64 plans" begin
        # The wide layout exists for degrees past `DP_MAX`, where a transition no longer fits an
        # Int32. Its arithmetic must be identical to the narrow one, which is checkable at a small
        # degree where building both is cheap.
        rng = MersenneTwister(25)
        for K in (8, 12, 16)
            A = randsym(rng, ComplexF64, K)
            idx = collect(1:K)
            v32 = TheEggman._haf_dp(A, idx, TheEggman._dp_plan(K, Int32))
            v64 = TheEggman._haf_dp(A, idx, TheEggman._dp_plan(K, Int64))
            @test v32 === v64
            @test v32 ≈ hafnian(A; method = :sieve) rtol = 1e-10
        end

        # Layout is chosen by degree, and the wide one is never selected automatically.
        @test TheEggman._dp_index_type(TheEggman.DP_MAX) === Int32
        @test TheEggman._dp_index_type(TheEggman.DP_MAX + 2) === Int64
        @test TheEggman._choose_method(34, ones(Int, 17), 12) === :sieve
        @test !TheEggman._prefer_dp(34, typemax(Int), 1)

        # ...but asking for it explicitly is allowed, right up to the bitmask width.
        @test TheEggman._check_method(:dp, TheEggman.DP_HARD_MAX) === nothing
        @test_throws ArgumentError TheEggman._check_method(:dp, TheEggman.DP_HARD_MAX + 2)

        # A plan too large to represent, or to fit in memory, is refused up front rather than
        # attempted — the guards run off the closed-form counts, so this builds nothing.
        @test_throws ArgumentError TheEggman._build_dp_plan(TheEggman.DP_HARD_MAX + 2, Int64)
        @test_throws ArgumentError TheEggman._build_dp_plan(50, Int64)
        @test_throws ArgumentError TheEggman._build_dp_plan(34, Int32)   # will not pack
    end

    @testset "DP threading is exact and level-safe" begin
        # Each state is written by exactly one task and reads only levels already joined, so the
        # arithmetic is identical regardless of how the levels get chunked — not merely close.
        rng = MersenneTwister(18)
        for N in (16, 20, 24)
            A = randsym(rng, ComplexF64, N)
            ref = hafnian(A; method = :dp, nthreads = 1)
            for nt in (2, 3, 7, 12)
                @test hafnian(A; method = :dp, nthreads = nt) === ref
            end
        end
        # Chunking never outruns the level it is splitting, and never returns a useless zero.
        for (ntrans, n, nt) in ((10, 5, 12), (10^6, 3, 12), (10^6, 10^5, 12), (0, 1, 8))
            c = TheEggman._dp_level_chunks(ntrans, n, nt)
            @test 1 <= c <= min(nt, n)
        end
    end

    @testset "DP is at least as accurate as the sieve" begin
        # The DP sums products of matrix entries directly, with none of the cancellation between
        # large signed terms that the sieve depends on.
        rng = MersenneTwister(15)
        setprecision(BigFloat, 256) do
            for N in (14, 18, 22)
                A = randsym(rng, ComplexF64, N)
                ref = hafnian(Complex{BigFloat}.(A); nthreads = 1, method = :sieve)
                dp_err = abs(hafnian(A; method = :dp) - ref) / abs(ref)
                sieve_err = abs(hafnian(A; method = :sieve) - ref) / abs(ref)
                @test dp_err < 1e-13
                @test dp_err <= max(sieve_err, 1e-15)
            end
        end
    end

    @testset "method selection" begin
        # Distinct rows: unrolled up to its cap, then the DP up to its own.
        @test TheEggman._choose_method(8, ones(Int, 4), 12) === :unrolled
        @test TheEggman._choose_method(12, ones(Int, 6), 12) === :unrolled
        @test TheEggman._choose_method(20, ones(Int, 10), 12) === :dp
        @test TheEggman._choose_method(28, ones(Int, 14), 12) === :dp
        @test TheEggman._choose_method(32, ones(Int, 16), 12) === :dp
        @test TheEggman._choose_method(34, ones(Int, 17), 12) === :sieve   # past DP_MAX
        # Repetition is what makes the sieve cheap, so it takes over as reps grow.
        @test TheEggman._choose_method(28, TheEggman.matched_reps(fill(2, 14))[2], 12) === :sieve
        @test TheEggman._choose_method(16, TheEggman.matched_reps(fill(2, 8))[2], 12) === :dp
        # Forcing a strategy it cannot serve is an error, not a silent fallback.
        A = randsym(MersenneTwister(16), ComplexF64, 16)
        @test_throws ArgumentError hafnian(A; method = :unrolled)
        @test_throws ArgumentError hafnian(A; method = :nonsense)
        # Past DP_HARD_MAX nothing can be built; below it `:dp` is allowed but would cost gigabytes,
        # so this deliberately probes the rejected side only.
        @test_throws ArgumentError hafnian(randsym(MersenneTwister(17), ComplexF64, 66); method = :dp)
    end

    @testset "unrolled crossover picks the faster path" begin
        # `_prefer_unrolled` compares two work counts whose units differ, so the threshold is a
        # calibration. Guard the two decisions it is actually calibrated against: enough repetition
        # shrinks the sieve below the unrolled kernel, a little does not.
        heavy = TheEggman.matched_reps([6, 6])[2]
        light = TheEggman.matched_reps([5, 5, 1, 1])[2]
        @test !TheEggman._prefer_unrolled(12, TheEggman._sieve_work(heavy))
        @test TheEggman._prefer_unrolled(12, TheEggman._sieve_work(light))
        # Distinct rows always favour the unrolled kernel, at every degree it covers.
        for N in 2:2:TheEggman.UNROLL_MAX
            @test TheEggman._prefer_unrolled(N, TheEggman._sieve_work(ones(Int, N ÷ 2)))
        end
        # Nothing above the cap may claim an unrolled kernel.
        @test !TheEggman._prefer_unrolled(TheEggman.UNROLL_MAX + 2, typemax(Int))
        @test_throws ArgumentError TheEggman._haf_direct(zeros(20, 20), collect(1:14))
    end

    @testset "unrolled expansion bookkeeping" begin
        # The emitted multiply count must match what `_prefer_unrolled` is told, and must stay far
        # below the (K-1)!! products an unshared expansion would emit.
        for K in 2:2:TheEggman.UNROLL_MAX
            statements, _ = TheEggman._unrolled_expansion(K)
            emitted = sum(count_muls, statements; init = 0)
            @test emitted == TheEggman._unrolled_muls(K)
            @test length(statements) < prod(1:2:(K-1)) || K <= 4
        end
    end

    @testset "matched_reps" begin
        for rpt in ([1, 1], [2, 2], [3, 1], [4, 1, 1], [2, 4], [5, 3], [1, 2, 3, 2], [0, 2, 2], [6, 2], [8])
            x, edge_reps = matched_reps(rpt)
            E = length(edge_reps)
            @test length(x) == 2E
            # Every vertex is matched: counting both endpoints of every edge reproduces `rpt`.
            counts = zeros(Int, length(rpt))
            for i in 1:E
                counts[x[i]] += edge_reps[i]
                counts[x[E+i]] += edge_reps[i]
            end
            @test counts == collect(rpt)
        end
    end

    @testset "hafnian_repeated vs brute force" begin
        rng = MersenneTwister(8)
        for rpt in ([1, 1], [2, 2], [3, 1], [2, 2, 2], [1, 2, 3, 2], [4, 1, 1], [2, 4], [5, 3], [0, 2, 2], [6, 2], [1, 1, 1, 1, 1, 1])
            d = length(rpt)
            A = randsym(rng, ComplexF64, d)
            @test hafnian_repeated(A, rpt) ≈ brute_hafnian(reduction(A, rpt))
        end
    end

    @testset "hafnian_repeated: consistency with hafnian" begin
        rng = MersenneTwister(9)
        A = randsym(rng, ComplexF64, 6)
        @test hafnian_repeated(A, ones(Int, 6)) ≈ hafnian(A)
        for rpt in ([2, 2, 2, 1, 1, 0], [3, 3, 2, 0, 0, 0])
            @test hafnian_repeated(A, rpt) ≈ hafnian(reduction(A, rpt))
        end
        # Unlike the plain hafnian, repeats do see the diagonal.
        C = copy(A)
        C[1, 1] += 5
        @test !isapprox(hafnian_repeated(A, [2, 2, 1, 1, 0, 0]), hafnian_repeated(C, [2, 2, 1, 1, 0, 0]))
    end

    @testset "hafnian_repeated: edge cases" begin
        rng = MersenneTwister(10)
        A = randsym(rng, ComplexF64, 4)
        @test hafnian_repeated(A, [0, 0, 0, 0]) == 1
        @test hafnian_repeated(A, [1, 0, 0, 0]) == 0     # odd total
        @test hafnian_repeated(A, [3, 1, 1, 0]) ≈ brute_hafnian(reduction(A, [3, 1, 1, 0]))
        @test hafnian_repeated(A, [2, 0, 0, 0]) ≈ A[1, 1]
    end

    @testset "views and wrappers cost nothing extra" begin
        rng = MersenneTwister(19)
        for N in (8, 12, 16, 20)
            F = randsym(rng, ComplexF64, 2N + 4)
            A = Matrix(F[1:N, 1:N])
            rpt = fill(1, N)
            base = hafnian(A)
            base_alloc = alloc_of(hafnian, A)
            base_alloc_rep = alloc_of(hafnian_repeated, A, rpt)

            for W in (view(F, 1:N, 1:N), view(F, 3:N+2, 3:N+2), view(F, 1:2:2N, 1:2:2N), Symmetric(A))
                @test hafnian(W) ≈ hafnian(Matrix(W))
                # Wrapping must not make the algorithm copy the matrix. Comparing against the
                # `Matrix` case rather than a fixed number keeps this robust to whatever the
                # chosen strategy legitimately allocates for itself.
                @test alloc_of(hafnian, W) == base_alloc
                @test hafnian_repeated(W, rpt) ≈ hafnian_repeated(Matrix(W), rpt)
                @test alloc_of(hafnian_repeated, W, rpt) == base_alloc_rep
            end
        end
    end

    @testset "unrolled path allocates nothing" begin
        rng = MersenneTwister(20)
        for N in 2:2:TheEggman.UNROLL_MAX
            A = randsym(rng, ComplexF64, N)
            @test alloc_of(hafnian, A) == 0
            @test alloc_of(hafnian, view(A, 1:N, 1:N)) == 0
        end
    end

    @testset "return type is inferrable" begin
        # A `Union{Val{true},Val{false}}` reaching the sieve once made this `Any`, which propagates
        # into every caller and boxes the result.
        rng = MersenneTwister(21)
        A = randsym(rng, ComplexF64, 12)
        R = randsym(rng, Float64, 12)
        I8 = round.(Int, 10 .* R)
        @test (@inferred hafnian(A)) isa ComplexF64
        @test (@inferred hafnian(view(A, 1:12, 1:12))) isa ComplexF64
        @test (@inferred hafnian(Symmetric(A))) isa ComplexF64
        @test (@inferred hafnian(I8)) isa Float64
        @test (@inferred hafnian(A; method = :sieve)) isa ComplexF64
        @test (@inferred hafnian(A; method = :sieve, glynn = false)) isa ComplexF64
        @test (@inferred hafnian(A; method = :dp)) isa ComplexF64
        @test (@inferred hafnian_repeated(A, fill(1, 12))) isa ComplexF64
        @test (@inferred hafnian_repeated(A, fill(2, 12); method = :sieve)) isa ComplexF64
    end

    @testset "symmetry check: tolerance matches isapprox" begin
        # The check compares squared magnitudes to avoid three `hypot` calls per entry pair, so its
        # accept/reject boundary must still be exactly `isapprox`'s.
        rng = MersenneTwister(22)
        for T in (Float64, ComplexF64)
            A = randsym(rng, T, 6)
            inside = copy(A)
            inside[2, 5] = A[5, 2] * (1 + sqrt(eps(Float64)) / 8)
            @test isapprox(inside[2, 5], inside[5, 2])
            @test TheEggman._check_symmetric(inside) === nothing

            outside = copy(A)
            outside[2, 5] = A[5, 2] * (1 + 8 * sqrt(eps(Float64)))
            @test !isapprox(outside[2, 5], outside[5, 2])
            @test_throws ArgumentError TheEggman._check_symmetric(outside)
        end
        # Infinities compare equal under `isapprox` and must keep doing so; NaN never does.
        B = fill(1.0, 4, 4)
        B[1, 2] = B[2, 1] = Inf
        @test TheEggman._check_symmetric(B) === nothing
        B[1, 2] = B[2, 1] = NaN
        @test_throws ArgumentError TheEggman._check_symmetric(B)
    end

    @testset "check_symmetric=false skips validation only" begin
        rng = MersenneTwister(23)
        for N in (8, 12, 16)
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A; check_symmetric = false) === hafnian(A)
            @test hafnian_repeated(A, fill(1, N); check_symmetric = false) === hafnian_repeated(A, fill(1, N))
        end
        # Asymmetric input is rejected with the check on, and not consulted with it off.
        bad = randsym(rng, ComplexF64, 8)
        bad[1, 2] += 1
        @test_throws ArgumentError hafnian(bad)
        @test hafnian(bad; check_symmetric = false) isa ComplexF64
        # With the check off only the entries the strategy reads are consulted; at this size that
        # is the upper triangle, so the answer is the one for `bad`'s symmetrisation-from-above.
        upper = copy(bad)
        for j in 1:8, i in 1:j-1
            upper[j, i] = upper[i, j]
        end
        @test hafnian(bad; check_symmetric = false) ≈ hafnian(upper)
    end

    @testset "Symmetric wrappers skip the check but agree" begin
        rng = MersenneTwister(24)
        for N in (8, 12, 16)
            raw = randn(rng, ComplexF64, N, N)      # deliberately asymmetric storage
            S = Symmetric(raw)
            # `Symmetric` mirrors one triangle on read, so it is symmetric by construction and the
            # check is skipped — but the answer must still be that of the mirrored matrix.
            @test TheEggman._check_symmetric(S) === nothing
            @test hafnian(S) ≈ hafnian(Matrix(S))
            # Complex `Hermitian` is *not* symmetric and must not slip through.
            H = Hermitian(raw)
            @test_throws ArgumentError hafnian(H)
        end
    end

    @testset "input validation" begin
        rng = MersenneTwister(11)
        A = randsym(rng, ComplexF64, 4)
        @test_throws DimensionMismatch hafnian(randn(3, 4))
        @test_throws ArgumentError hafnian([1.0 2.0; 3.0 4.0])
        @test_throws DimensionMismatch hafnian_repeated(A, [1, 1, 1])
        @test_throws ArgumentError hafnian_repeated(A, [-1, 1, 1, 1])
    end

    @testset "no allocation in the sieve inner loop" begin
        rng = MersenneTwister(12)
        A = randsym(rng, ComplexF64, 12)
        hafnian(A; nthreads = 1, method = :sieve)
        base = @allocated hafnian(A; nthreads = 1, method = :sieve)
        # Setup (permuted copy, binomials, one workspace) allocates; the 32 sieve terms must not.
        A2 = randsym(rng, ComplexF64, 16)
        hafnian(A2; nthreads = 1, method = :sieve)
        grown = @allocated hafnian(A2; nthreads = 1, method = :sieve)
        # 4× the terms and a larger workspace, but nothing that scales with the term count.
        @test grown < 6 * base
    end
end
