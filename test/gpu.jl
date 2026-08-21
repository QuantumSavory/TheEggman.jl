# Exercises the KernelAbstractions extension on the `CPU()` backend. That backend runs the real
# kernel through the real device-plan and batching machinery, so everything except CUDA codegen and
# actual GPU performance is covered here — no GPU required, in CI or locally.
#
# The GPU-only checks live in benchmark/gpu/run_gpu_validation.jl.

using KernelAbstractions

@testset "GPU backend (KernelAbstractions, CPU backend)" begin
    be = CPU()

    @testset "agrees with the CPU DP" begin
        # Nothing in the kernel reorders a sum — each state is accumulated by one thread in plan
        # order — but a device compiler may contract `a*b + c` into a single-rounding FMA, which
        # moves the last ulp. So this is a tight tolerance across backends, not `===`. Measured gap
        # on a real GPU: ~1e-16 to 2e-15, with the contracted result the more accurate one.
        rng = MersenneTwister(101)
        for N in (14, 16, 20, 24)
            A = randsym(rng, ComplexF64, N)
            @test hafnian(A; backend = be) ≈ hafnian(A; method = :dp, nthreads = 1) rtol = 1e-12
            R = randsym(rng, Float64, N)
            @test hafnian(R; backend = be) ≈ hafnian(R; method = :dp, nthreads = 1) rtol = 1e-12
        end
    end

    @testset "batched matches the scalar loop exactly" begin
        # Within one backend nothing changes how a sum is evaluated, so batching, chunking and
        # repetition are all exactly reproducible. These stay `===`.
        rng = MersenneTwister(102)
        As = [randsym(rng, ComplexF64, 16) for _ in 1:7]
        scalar = [hafnian(A; backend = be) for A in As]
        @test all(hafnian(As; backend = be) .=== scalar)
        for mb in (1, 2, 3, 100)
            @test hafnian(As; backend = be, max_batch = mb) == scalar
        end
        @test hafnian(As[1:1]; backend = be) == scalar[1:1]
        @test hafnian(Matrix{ComplexF64}[]; backend = be) == ComplexF64[]
        # ...and it still agrees with the CPU to a tolerance.
        @test hafnian(As; backend = be) ≈ [hafnian(A; method = :dp, nthreads = 1) for A in As] rtol = 1e-12
    end

    @testset "batched repeated matches the scalar loop" begin
        rng = MersenneTwister(103)
        A = randsym(rng, ComplexF64, 16)
        rpts = [[2, 2, 2, 1, 1, 0, 0, 0, 1, 1, 2, 2, 1, 1, 0, 0],
                ones(Int, 16),
                [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # The batched path runs the DP for the whole batch (one shared plan), while a *scalar* call
        # picks per pattern — `[4,4,4,4,...]` is repeated enough to select the sieve. Force `:dp` on
        # the scalar side so this compares like with like.
        scalar = [hafnian_repeated(A, r; backend = be, method = :dp) for r in rpts]
        @test all(hafnian_repeated(A, rpts; backend = be) .=== scalar)
        @test hafnian_repeated(A, rpts; backend = be, max_batch = 2) == scalar
        @test hafnian_repeated(A, rpts; backend = be) ≈
              [hafnian_repeated(A, r; method = :dp, nthreads = 1) for r in rpts] rtol = 1e-12
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
            # These never reach a kernel at all, so they really are the same computation.
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

    @testset "no views in host<->device transfers" begin
        # A source lint, not a behavioural test, and deliberately so: the CPU backend allocates
        # plain `Array`s, so it cannot reproduce device dispatch at all. The bug this guards
        # against only appears on a real GPU — `copyto!` with a view on either side misses the
        # host<->device methods, falls through to Base's generic implementation, and scalar-indexes
        # the device array ("Scalar indexing is disallowed"). Every transfer must therefore use the
        # five-argument `copyto!(dest, doffs, src, soffs, n)` form on plain arrays.
        src = read(joinpath(pkgdir(TheEggman), "ext", "TheEggmanKernelAbstractionsExt.jl"), String)
        body = split(src, "function TheEggman._haf_dp_backend")[end]
        for line in split(body, '\n')
            code = first(split(line, '#'))          # ignore the comment that explains this rule
            if occursin("copyto!", code) || occursin("fill!", code)
                @test !occursin("view(", code)
            end
        end
    end

    @testset "dp_batch_bytes accounting" begin
        nstates, _ = TheEggman._dp_counts(20)
        npairs = 20 * 19 ÷ 2
        @test TheEggman.dp_batch_bytes(20, ComplexF64, 1) == (nstates + npairs) * 16
        @test TheEggman.dp_batch_bytes(20, ComplexF32, 8) == 8 * (nstates + npairs) * 8
    end
end
