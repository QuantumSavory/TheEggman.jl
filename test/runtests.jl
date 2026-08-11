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
            @test hafnian(A; glynn = false, unrolled = false) ≈
                  hafnian(A; unrolled = false) rtol = 1e-9
        end
    end

    @testset "hafnian: threading is deterministic in value" begin
        rng = MersenneTwister(6)
        A = randsym(rng, ComplexF64, 14)
        ref = hafnian(A; nthreads = 1, unrolled = false)
        for nt in (2, 3, 8)
            @test hafnian(A; nthreads = nt, unrolled = false) ≈ ref rtol = 1e-10
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
                ref = hafnian(Complex{BigFloat}.(A); nthreads = 1, unrolled = false)
                err = abs(hafnian(A; unrolled = false) - ref) / abs(ref)
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

    @testset "unrolled kernels agree with the sieve" begin
        rng = MersenneTwister(14)
        for N in 2:2:TheEggman.UNROLL_MAX
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A) ≈ hafnian(A; unrolled = false) rtol = 1e-10
            R = randsym(rng, Float64, N)
            @test hafnian(R) ≈ hafnian(R; unrolled = false) rtol = 1e-10
        end
        # Repeated indices exercise the same kernels through a different entry point, and are the
        # only case where the unrolled path reads the diagonal of `A`.
        for rpt in ([2, 2, 2], [1, 2, 3, 2], [2, 2, 2, 2, 2, 2], [6, 6], [3, 3, 3, 3], [8, 4],
                    [5, 5, 1, 1], [10, 2], [12], [4, 4], [9, 3], [6, 2, 2, 2])
            d = length(rpt)
            A = randsym(rng, ComplexF64, d)
            @test hafnian_repeated(A, rpt) ≈ hafnian_repeated(A, rpt; unrolled = false) rtol = 1e-10
            @test hafnian_repeated(A, rpt) ≈ brute_hafnian(reduction(A, rpt)) rtol = 1e-10
        end
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
        hafnian(A; nthreads = 1, unrolled = false)
        base = @allocated hafnian(A; nthreads = 1, unrolled = false)
        # Setup (permuted copy, binomials, one workspace) allocates; the 32 sieve terms must not.
        A2 = randsym(rng, ComplexF64, 16)
        hafnian(A2; nthreads = 1, unrolled = false)
        grown = @allocated hafnian(A2; nthreads = 1, unrolled = false)
        # 4× the terms and a larger workspace, but nothing that scales with the term count.
        @test grown < 6 * base
    end
end
