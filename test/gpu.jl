# Exercises the KernelAbstractions extension on the `CPU()` backend. That backend runs the real
# kernel through the real device-plan and batching machinery, so everything except CUDA codegen and
# actual GPU performance is covered here — no GPU required, in CI or locally.
#
# The GPU-only checks live in benchmark/gpu/run_gpu_validation.jl.

using KernelAbstractions

@testset "GPU backend (KernelAbstractions, CPU backend)" begin
    be = CPU()

    @testset "bit-identical to the CPU DP" begin
        # Each state is summed by one thread in plan order, the same order the CPU uses, so this is
        # exact equality rather than a tolerance. A failure means the kernel reordered a sum.
        rng = MersenneTwister(101)
        for N in (14, 16, 20, 24)
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A; backend = be) === hafnian(A; method = :dp, nthreads = 1)
            R = randsym(rng, Float64, N)
            @test hafnian(R; backend = be) === hafnian(R; method = :dp, nthreads = 1)
        end
    end

    @testset "batched matches the scalar loop" begin
        rng = MersenneTwister(102)
        As = [randsym(rng, ComplexF64, 16) for _ in 1:7]
        ref = [hafnian(A; method = :dp, nthreads = 1) for A in As]
        @test all(hafnian(As; backend = be) .=== ref)
        # Chunking must not change a single bit, whatever the chunk size.
        for mb in (1, 2, 3, 100)
            @test hafnian(As; backend = be, max_batch = mb) == ref
        end
        @test hafnian(As[1:1]; backend = be) == ref[1:1]
        @test hafnian(Matrix{ComplexF64}[]; backend = be) == ComplexF64[]
    end

    @testset "batched repeated matches the scalar loop" begin
        rng = MersenneTwister(103)
        A = randsym(rng, ComplexF64, 16)
        rpts = [[2, 2, 2, 1, 1, 0, 0, 0, 1, 1, 2, 2, 1, 1, 0, 0],
                ones(Int, 16),
                [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ref = [hafnian_repeated(A, r; method = :dp, nthreads = 1) for r in rpts]
        @test all(hafnian_repeated(A, rpts; backend = be) .=== ref)
        @test hafnian_repeated(A, rpts; backend = be, max_batch = 2) == ref
    end

    @testset "ComplexF32 path" begin
        # fp32 matters on consumer cards, where fp64 runs at 1/32-1/64 rate. The DP never cancels,
        # so its fp32 error stays near machine epsilon instead of growing with N.
        rng = MersenneTwister(104)
        for N in (12, 16, 20)
            A = randsym(rng, ComplexF64, N)
            v32 = hafnian(ComplexF32.(A); backend = be)
            v64 = hafnian(A; backend = be)
            @test v32 isa ComplexF32
            @test abs(v32 - v64) / abs(v64) < 1e-5
        end
    end

    @testset "falls back rather than forcing the GPU" begin
        rng = MersenneTwister(105)
        # Below UNROLL_MAX the whole call is shorter than a kernel launch, so `backend` is ignored.
        for N in (4, 8, 12)
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A; backend = be) === hafnian(A)
        end
        # Degenerate sizes never reach a kernel.
        @test hafnian(zeros(0, 0); backend = be) == 1
        @test hafnian(randsym(rng, Float64, 5); backend = be) == 0
        # Heavy repetition selects the sieve, which has no GPU path; the answer must still be right.
        A = randsym(rng, ComplexF64, 4)
        rpt = [7, 7, 7, 7]
        @test hafnian_repeated(A, rpt; backend = be) ≈ hafnian_repeated(A, rpt) rtol = 1e-10
    end

    @testset "determinism" begin
        rng = MersenneTwister(106)
        A = randsym(rng, ComplexF64, 18)
        @test hafnian(A; backend = be) === hafnian(A; backend = be)
        As = [randsym(rng, ComplexF64, 18) for _ in 1:4]
        @test hafnian(As; backend = be) == hafnian(As; backend = be)
    end

    @testset "device plan cache" begin
        TheEggman.empty_gpu_cache!()
        @test TheEggman.gpu_cache_bytes() == 0
        A = randsym(MersenneTwister(107), ComplexF64, 16)
        hafnian(A; backend = be)
        used = TheEggman.gpu_cache_bytes()
        @test used > 0
        # A second call at the same degree reuses the upload rather than repeating it.
        hafnian(A; backend = be)
        @test TheEggman.gpu_cache_bytes() == used
        TheEggman.empty_gpu_cache!()
        @test TheEggman.gpu_cache_bytes() == 0
    end

    @testset "batch validation" begin
        rng = MersenneTwister(108)
        A = randsym(rng, ComplexF64, 10)
        # A batch shares one plan, so every pattern needs the same total.
        @test_throws ArgumentError hafnian_repeated(A, [ones(Int, 10), [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
        @test_throws ArgumentError hafnian_repeated(A, [ones(Int, 10), [-1; ones(Int, 9)]])
        @test_throws DimensionMismatch hafnian_repeated(A, [ones(Int, 10), ones(Int, 9)])
        # ...and one size.
        @test_throws DimensionMismatch hafnian([A, randsym(rng, ComplexF64, 12)])
    end

    @testset "batched return types are inferrable" begin
        rng = MersenneTwister(109)
        As = [randsym(rng, ComplexF64, 12) for _ in 1:3]
        @test (@inferred hafnian(As)) isa Vector{ComplexF64}
        @test (@inferred hafnian(As; backend = be)) isa Vector{ComplexF64}
        A = As[1]
        rpts = [ones(Int, 12), [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]]
        @test (@inferred hafnian_repeated(A, rpts)) isa Vector{ComplexF64}
        @test (@inferred hafnian_repeated(A, rpts; backend = be)) isa Vector{ComplexF64}
    end

    @testset "dp_batch_bytes accounting" begin
        nstates, _ = TheEggman._dp_counts(20)
        npairs = 20 * 19 ÷ 2
        @test TheEggman.dp_batch_bytes(20, ComplexF64, 1) == (nstates + npairs) * 16
        @test TheEggman.dp_batch_bytes(20, ComplexF32, 8) == 8 * (nstates + npairs) * 8
    end
end
