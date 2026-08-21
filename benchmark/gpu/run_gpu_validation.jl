# Validate and quantify TheEggman.jl's GPU subset-DP port. Run on a GPU-enabled machine:
#
#     julia --project=benchmark/gpu -t auto benchmark/gpu/run_gpu_validation.jl
#     julia --project=benchmark/gpu -t auto benchmark/gpu/run_gpu_validation.jl --quick
#
# Writes gpu-validation.json next to itself, prints a summary, and ends with a fill-in template.
# Nothing here mutates the repository. Stages are independent: a failure in one is recorded and the
# rest still run.
#
# Stages: A environment, B correctness, C performance, D memory, E sieve viability.

using TheEggman, KernelAbstractions, Random, Printf, JSON

const QUICK = "--quick" in ARGS

# EGGMAN_GPU_DRYRUN=1 runs every stage against KernelAbstractions' CPU backend with stubbed device
# queries. The performance numbers are then meaningless, but it exercises the whole script end to
# end, so it can be smoke-tested on a machine with no GPU before being run for real.
const DRYRUN = get(ENV, "EGGMAN_GPU_DRYRUN", "0") == "1"
DRYRUN || @eval using CUDA

const RESULTS = Dict{String,Any}()

randsym(rng, T, n) = (B = randn(rng, T, n, n); B + transpose(B))

# --- the only places this script touches a vendor API -------------------------------------------
if DRYRUN
    backend() = CPU()
    devsync() = nothing
    avail_mem() = Int(Sys.free_memory())
    reclaim!() = GC.gc()
    smem_per_sm() = 100 * 1024                     # a plausible modern value, for the arithmetic
    functional() = true
    device_info() = Dict{String,Any}(
        "device" => "DRYRUN (KernelAbstractions CPU backend)", "capability" => "n/a",
        "fp64_fp32_ratio_hint" => "n/a",
        "vram_total_GB" => round(Sys.total_memory() / 2^30, digits = 2),
        "vram_free_GB" => round(Sys.free_memory() / 2^30, digits = 2),
        "multiprocessors" => 0, "shared_mem_per_block_KB" => 0,
        "shared_mem_per_sm_KB" => smem_per_sm() ÷ 1024,
        "cuda_jl" => "n/a", "driver" => "n/a")
else
    backend() = CUDA.CUDABackend()
    devsync() = CUDA.synchronize()
    avail_mem() = CUDA.free_memory()
    reclaim!() = CUDA.reclaim()
    functional() = CUDA.functional()
    smem_per_sm() =
        CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    function device_info()
        dev = CUDA.device()
        cap = CUDA.capability(dev)
        # The fp64:fp32 ratio is fixed per architecture and is the single most important number for
        # reading stage C3: 1/2 on datacenter parts, 1/32 or worse on consumer ones.
        hint = get(Dict(6 => "1/2 (P100) or 1/32", 7 => "1/2 (V100) or 1/32 (Turing)",
                        8 => "1/2 (A100) or 1/64 (Ampere consumer)", 9 => "1/2 (H100)",
                        10 => "consumer: ~1/64"), cap.major, "unknown")
        Dict{String,Any}(
            "device" => CUDA.name(dev), "capability" => string(cap),
            "fp64_fp32_ratio_hint" => hint,
            "vram_total_GB" => round(CUDA.totalmem(dev) / 2^30, digits = 2),
            "vram_free_GB" => round(CUDA.free_memory() / 2^30, digits = 2),
            "multiprocessors" => CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            "shared_mem_per_block_KB" =>
                CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) ÷ 1024,
            "shared_mem_per_sm_KB" => smem_per_sm() ÷ 1024,
            "cuda_jl" => string(pkgversion(CUDA)),
            "driver" => (try string(CUDA.driver_version()) catch; "unknown" end))
    end
end

"""Minimum wall time of `f` over `reps`, in milliseconds. Synchronises the device each rep."""
function timed(f, reps; sync = true)
    f(); sync && devsync()
    t = Inf
    for _ in 1:reps
        t0 = time_ns()
        f()
        sync && devsync()
        t = min(t, (time_ns() - t0) / 1e6)
    end
    t
end

section(name) = (println("\n", "="^78); println(name); println("="^78); flush(stdout))

# ---------------------------------------------------------------------------------------------
section("A. Environment")

if !functional()
    @error "CUDA is not functional here; nothing below can run."
    exit(1)
end
env = device_info()
env["ka_jl"] = string(pkgversion(KernelAbstractions))
env["julia"] = string(VERSION)
env["nthreads"] = Threads.nthreads()
env["dryrun"] = DRYRUN
RESULTS["A_environment"] = env
for k in sort(collect(keys(env)))
    @printf("  %-26s %s\n", k, env[k])
end
DRYRUN && @warn "DRY RUN: CPU backend with stubbed device queries. Timings are not GPU numbers."
if Threads.nthreads() == 1
    @warn """Julia has ONE thread, so every CPU baseline below is single-threaded and every GPU
             speedup in stages C1/C2 is overstated — the CPU DP scales about 4x across cores.
             Re-run with `-t auto` (or JULIA_NUM_THREADS) for a fair comparison."""
end

const BE = backend()

# ---------------------------------------------------------------------------------------------
section("B. Correctness — all must pass")

bres = Dict{String,Any}()
rng = MersenneTwister(20260810)

# B1: agreement with the CPU DP. Nothing in the kernel reorders a sum, but device compilers contract
# `a*b + c` into a single-rounding FMA, which moves the last ulp — so this is a tolerance, not `===`.
# The threshold sits far above that FP noise and far below anything a real bug would hide under.
const B1_RTOL = 1e-12
b1 = Dict{String,Any}()
b1ok = true
for N in (QUICK ? (14, 20) : (14, 16, 20, 24, 28, 32))
    A = randsym(rng, ComplexF64, N)
    g = hafnian(A; backend = BE)
    c = hafnian(A; method = :dp, nthreads = 1)
    rel = abs(g - c) / abs(c)
    ok = rel < B1_RTOL
    global b1ok &= ok
    b1["N=$N"] = Dict("ok" => ok, "rel_diff" => rel, "exact" => g === c)
    @printf("  B1 N=%-3d agrees (rel %.3e): %-6s%s\n", N, rel, ok,
            g === c ? "  [exact]" : "")
    flush(stdout)
end
bres["B1_agrees_with_cpu"] = b1

As = [randsym(rng, ComplexF64, 20) for _ in 1:16]

# B2: within one backend, batching must not change how any sum is evaluated. This one IS exact.
b2 = all(hafnian(As; backend = BE) .=== [hafnian(A; backend = BE) for A in As])
bres["B2_batched_eq_scalar_same_backend"] = b2
println("  B2 batched === scalar, same backend: ", b2)

# B3: across backends, tolerance again — the CPU backend need not contract the same way.
b3rel = maximum(abs.(hafnian(As; backend = BE) .- hafnian(As; backend = CPU())) ./
                abs.(hafnian(As; backend = CPU())))
b3 = b3rel < B1_RTOL
bres["B3_backend_agreement"] = Dict("ok" => b3, "max_rel_diff" => b3rel)
@printf("  B3 device vs KA CPU backend (max rel %.3e): %s\n", b3rel, b3)

# B4: ComplexF32 should stay near fp32 epsilon; the DP never cancels, so it must not drift with N.
b4 = Dict{String,Any}(); b4ok = true
for N in (16, 24)
    A = randsym(rng, ComplexF64, N)
    v64 = hafnian(A; backend = BE)
    e = abs(hafnian(ComplexF32.(A); backend = BE) - v64) / abs(v64)
    b4["N=$N"] = e
    global b4ok &= e < 1e-5
    @printf("  B4 N=%-3d ComplexF32 rel err: %.3e  %s\n", N, e, e < 1e-5 ? "ok" : "TOO LARGE")
end
bres["B4_complexf32_relerr"] = b4

# B5: things that must never reach a kernel.
Asmall = randsym(rng, ComplexF64, 8)
b5 = (hafnian(Asmall; backend = BE) === hafnian(Asmall)) &&
     (hafnian(zeros(0, 0); backend = BE) == 1) &&
     (hafnian(randsym(rng, Float64, 5); backend = BE) == 0)
bres["B5_fallbacks"] = b5
println("  B5 unrolled / empty / odd fall back correctly: ", b5)

Ad = randsym(rng, ComplexF64, 22)
b6 = hafnian(Ad; backend = BE) === hafnian(Ad; backend = BE)
bres["B6_determinism"] = b6
println("  B6 determinism: ", b6)

b7 = hafnian(As; backend = BE, max_batch = 3) == hafnian(As; backend = BE)
bres["B7_chunking_invariant"] = b7
println("  B7 chunking invariance: ", b7)

# B8: a batch shares one plan, so mismatched totals must be refused.
b8 = try
    hafnian_repeated(As[1], [ones(Int, 20), [ones(Int, 19); 0]]; backend = BE)
    false
catch e
    e isa ArgumentError
end
bres["B8_mismatched_totals_rejected"] = b8
println("  B8 mismatched totals rejected: ", b8)

allb = b1ok && b2 && b3 && b4ok && b5 && b6 && b7 && b8
bres["overall"] = allb
RESULTS["B_correctness"] = bres
println("\n  B overall: ", allb ? "PASS" : "FAIL — investigate before trusting stage C")

# ---------------------------------------------------------------------------------------------
section("C. Performance")

# C0 — path consistency. `hafnian(A; backend)` and `hafnian([A]; backend)` do identical work, so they
# must take the same time. They did not before device buffers were reused: an earlier run measured
# 23.1 ms for the scalar path against 1.2 ms for the batch-of-one at N=32, which pointed at the
# allocator being churned by per-call allocations rather than at anything algorithmic.
println("\nC0 scalar vs batch-of-one — identical work, so the ratio should be ~1")
@printf("  %-5s %-13s %-13s %s\n", "N", "scalar (ms)", "batch-1 (ms)", "ratio")
c0 = Dict{String,Any}()
c0worst = 1.0
for N in (QUICK ? (24,) : (24, 28, 32))
    A = randsym(MersenneTwister(N), ComplexF64, N)
    reps = N >= 30 ? 3 : 10
    ts = timed(() -> hafnian(A; backend = BE), reps)
    tb = timed(() -> hafnian([A]; backend = BE), reps)
    r = max(ts, tb) / min(ts, tb)
    global c0worst = max(c0worst, r)
    c0["N=$N"] = Dict("scalar_ms" => ts, "batch1_ms" => tb, "ratio" => r)
    @printf("  %-5d %-13.3f %-13.3f %.2fx%s\n", N, ts, tb, r, r > 1.5 ? "  <-- INVESTIGATE" : "")
    flush(stdout)
end
RESULTS["C0_path_consistency"] = c0
c0worst > 1.5 && @warn "scalar and batch-of-one disagree by $(round(c0worst, digits=2))x; the two " *
                       "paths do the same work, so one of them is paying an avoidable cost."

# C1 — single instance. Expect the GPU to lose at small N: roughly K/2 kernel launches per call is a
# floor that a single instance cannot amortise.
println("\nC1 single-instance sweep (ComplexF64, B=1)")
@printf("  %-5s %-12s %-12s %s\n", "N", "GPU (ms)", "CPU (ms)", "speedup")
c1 = Dict{String,Any}()
crossover = nothing
for N in (QUICK ? (20, 28) : (16, 20, 24, 26, 28, 30, 32))
    A = randsym(MersenneTwister(N), ComplexF64, N)
    reps = N >= 30 ? 3 : 20
    tg = timed(() -> hafnian(A; backend = BE), reps)
    tc = timed(() -> hafnian(A; method = :dp), reps; sync = false)
    c1["N=$N"] = Dict("gpu_ms" => tg, "cpu_ms" => tc, "speedup" => tc / tg)
    if crossover === nothing && tc / tg > 1
        global crossover = N
    end
    @printf("  %-5d %-12.3f %-12.3f %.2fx\n", N, tg, tc, tc / tg)
    flush(stdout)
end
c1["crossover_N"] = crossover
RESULTS["C1_single"] = c1
println("  crossover N (GPU first wins): ", something(crossover, "none in range"))

# C2 — batched throughput. The headline: per-level launches are paid once for the whole batch, so
# small N should improve dramatically.
println("\nC2 batched throughput (ComplexF64) — hafnians/second")
@printf("  %-5s %-7s %-14s %-14s %s\n", "N", "B", "GPU (haf/s)", "CPU (haf/s)", "speedup")
c2 = Dict{String,Any}()
best_batched = 0.0
for N in (QUICK ? (20,) : (20, 24, 28, 32)), Bsz in (1, 8, 64, 256)
    # Keep each measurement bounded in memory and time.
    TheEggman.dp_batch_bytes(N, ComplexF64, Bsz) > 0.4 * avail_mem() && continue
    Ab = [randsym(MersenneTwister(N * 1000 + b), ComplexF64, N) for b in 1:Bsz]
    reps = N >= 28 ? 2 : 5
    tg = timed(() -> hafnian(Ab; backend = BE), reps)
    tc = timed(() -> hafnian(Ab), reps; sync = false)
    c2["N=$N,B=$Bsz"] = Dict("gpu_ms" => tg, "cpu_ms" => tc, "gpu_per_s" => 1000Bsz / tg,
                             "cpu_per_s" => 1000Bsz / tc, "speedup" => tc / tg)
    global best_batched = max(best_batched, tc / tg)
    @printf("  %-5d %-7d %-14.1f %-14.1f %.2fx\n", N, Bsz, 1000Bsz / tg, 1000Bsz / tc, tc / tg)
    flush(stdout)
end
RESULTS["C2_batched"] = c2

# C2b — chunk sweep. C2 showed throughput can *fall* as the batch grows (N=28 peaked at B=64 and
# dropped by ~2.8x at B=256). `max_batch` splits a batch into chunks without changing the answer, so
# sweeping it on one large batch separates the two candidate causes: if a smaller chunk recovers the
# peak, the cost is per-chunk (allocation churn, or a working set that has outgrown the cache); if it
# does not, the cost is elsewhere. Whatever wins here is the value to pass in production.
println("\nC2b chunk sweep at a fixed large batch — finds the best max_batch")
@printf("  %-5s %-7s %-11s %-14s %s\n", "N", "B", "max_batch", "GPU (haf/s)", "vs unchunked")
c2b = Dict{String,Any}()
for N in (QUICK ? (24,) : (24, 28))
    Bsz = 256
    TheEggman.dp_batch_bytes(N, ComplexF64, Bsz) > 0.4 * avail_mem() && continue
    Ab = [randsym(MersenneTwister(N * 7 + b), ComplexF64, N) for b in 1:Bsz]
    reps = N >= 28 ? 2 : 4
    base = timed(() -> hafnian(Ab; backend = BE), reps)
    best_mb, best_t = nothing, base
    for mb in (16, 32, 64, 128, 256)
        mb > Bsz && continue
        t = timed(() -> hafnian(Ab; backend = BE, max_batch = mb), reps)
        c2b["N=$N,max_batch=$mb"] = Dict("gpu_ms" => t, "haf_per_s" => 1000Bsz / t,
                                         "vs_unchunked" => base / t)
        if t < best_t
            best_mb, best_t = mb, t
        end
        @printf("  %-5d %-7d %-11d %-14.1f %.2fx\n", N, Bsz, mb, 1000Bsz / t, base / t)
        flush(stdout)
    end
    c2b["N=$N,best"] = Dict("max_batch" => something(best_mb, Bsz), "speedup" => base / best_t)
    @printf("  -> N=%d best max_batch: %s (%.2fx over unchunked)\n",
            N, something(best_mb, "unchunked"), base / best_t)
end
RESULTS["C2b_chunk_sweep"] = c2b

# C3 — precision. A ratio near 1 means the kernel is bandwidth-bound rather than fp64-throughput
# bound, which is itself worth knowing on a consumer card.
println("\nC3 ComplexF32 vs ComplexF64 (batched, B=64)")
@printf("  %-5s %-14s %-14s %s\n", "N", "f64 (ms)", "f32 (ms)", "f32 speedup")
c3 = Dict{String,Any}()
for N in (QUICK ? (20,) : (20, 24, 28))
    Bsz = 64
    A64 = [randsym(MersenneTwister(N + b), ComplexF64, N) for b in 1:Bsz]
    A32 = [ComplexF32.(A) for A in A64]
    reps = N >= 28 ? 2 : 5
    t64 = timed(() -> hafnian(A64; backend = BE), reps)
    t32 = timed(() -> hafnian(A32; backend = BE), reps)
    c3["N=$N"] = Dict("f64_ms" => t64, "f32_ms" => t32, "f32_speedup" => t64 / t32)
    @printf("  %-5d %-14.3f %-14.3f %.2fx\n", N, t64, t32, t64 / t32)
    flush(stdout)
end
RESULTS["C3_precision"] = c3

# C4 — plan upload amortisation. The first call at a degree pays the H2D transfer (~196 MB at N=32);
# everything after reuses it.
println("\nC4 plan upload amortisation")
@printf("  %-5s %-14s %-14s %s\n", "N", "first (ms)", "steady (ms)", "break-even calls")
c4 = Dict{String,Any}()
for N in (QUICK ? (24,) : (24, 28, 32))
    TheEggman.empty_gpu_cache!()
    reclaim!()
    A = randsym(MersenneTwister(N), ComplexF64, N)
    t0 = time_ns(); hafnian(A; backend = BE); devsync()
    tfirst = (time_ns() - t0) / 1e6
    tsteady = timed(() -> hafnian(A; backend = BE), 5)
    tcpu = timed(() -> hafnian(A; method = :dp), 3; sync = false)
    saved = tcpu - tsteady
    be_calls = saved > 0 ? ceil(Int, (tfirst - tsteady) / saved) : -1
    c4["N=$N"] = Dict("first_ms" => tfirst, "steady_ms" => tsteady, "cpu_ms" => tcpu,
                      "break_even_calls" => be_calls)
    @printf("  %-5d %-14.2f %-14.3f %s\n", N, tfirst, tsteady,
            be_calls < 0 ? "never (CPU faster)" : string(be_calls))
    flush(stdout)
end
RESULTS["C4_upload"] = c4

# C5 — launch overhead. An N=16 call is nearly all launch overhead, so dividing by the level count
# estimates the per-launch cost.
A16 = randsym(MersenneTwister(16), ComplexF64, 16)
nlevels = length(TheEggman._dp_plan(16, Int32).levels) - 2
t16 = timed(() -> hafnian(A16; backend = BE), 50)
launch_us = 1000 * t16 / nlevels
RESULTS["C5_launch_overhead_us"] = launch_us
@printf("\nC5 fixed cost per call: %.3f ms at N=16 (%.1f us per level launch over %d launches)\n",
        t16, launch_us, nlevels)
println("     — an N=16 call is almost all fixed cost, so this bounds the launch + transfer floor.")

# ---------------------------------------------------------------------------------------------
section("D. Memory")

d = Dict{String,Any}()
TheEggman.empty_gpu_cache!(); reclaim!()
for N in (QUICK ? (24,) : (24, 28, 32))
    A = randsym(MersenneTwister(N), ComplexF64, N)
    hafnian(A; backend = BE)
    plan_bytes = TheEggman.gpu_cache_bytes()
    maxB32 = floor(Int, 0.7 * avail_mem() / TheEggman.dp_batch_bytes(N, ComplexF32, 1))
    maxB64 = floor(Int, 0.7 * avail_mem() / TheEggman.dp_batch_bytes(N, ComplexF64, 1))
    d["N=$N"] = Dict("plan_cache_MB" => plan_bytes / 2^20, "max_B_f32" => maxB32,
                     "max_B_f64" => maxB64)
    @printf("  N=%-3d plan cache %.1f MB   max B: %d (f32) / %d (f64)\n",
            N, plan_bytes / 2^20, maxB32, maxB64)
    TheEggman.empty_gpu_cache!(); reclaim!()
end
RESULTS["D_memory"] = d

# ---------------------------------------------------------------------------------------------
section("E. Sieve viability — measurement only, no sieve GPU code exists")

# A sieve term needs a private workspace: the m x m working matrix, its split real/imaginary copies,
# and the (m+1) x (n+1) characteristic-polynomial table. At m = 32 that is roughly 40 KB, which is
# why the sieve was not ported — it would leave almost no occupancy.
smem_sm = smem_per_sm()
m, n = 32, 16
ws_bytes = (m * m * 16) + 2 * (m * m * 8) + ((m + 1) * (n + 1) * 16)
terms_per_sm = smem_sm ÷ ws_bytes
verdict = terms_per_sm >= 2 ? "worth investigating" : "do not build"
RESULTS["E_sieve"] = Dict("workspace_bytes_m32" => ws_bytes, "shared_mem_per_sm" => smem_sm,
                          "terms_per_sm" => terms_per_sm, "verdict" => verdict)
@printf("  sieve workspace at m=32: %.1f KB\n", ws_bytes / 1024)
@printf("  shared memory per SM:    %.1f KB\n", smem_sm / 1024)
@printf("  concurrent terms per SM: %d  -> %s\n", terms_per_sm, uppercase(verdict))

# ---------------------------------------------------------------------------------------------
out = joinpath(@__DIR__, "gpu-validation.json")
open(io -> JSON.print(io, RESULTS, 2), out, "w")

section("Summary — paste this back")
println("""
Device / CC / VRAM / driver : $(env["device"]) / $(env["capability"]) / $(env["vram_total_GB"]) GB / $(env["driver"])
B1-B8 correctness           : $(allb ? "PASS" : "FAIL")
C1 crossover N              : $(something(crossover, "none in range"))
C2 best batched speedup     : $(round(best_batched, digits = 2))x
C0 worst path ratio         : $(round(c0worst, digits = 2))x
C2b best max_batch          : $(join(["$k=$(v["max_batch"]) ($(round(v["speedup"], digits=2))x)" for (k, v) in sort(collect(c2b), by = first) if endswith(k, "best")], " "))
C3 f32 speedup              : $(join(["$k:$(round(v["f32_speedup"], digits=2))x" for (k, v) in sort(collect(c3), by = first)], " "))
C4 break-even calls         : $(join(["$k:$(v["break_even_calls"] < 0 ? "never" : string(v["break_even_calls"]))" for (k, v) in sort(collect(c4), by = first)], " "))
C5 fixed cost/call (N=16)   : $(round(t16, digits = 3)) ms ($(round(launch_us, digits = 1)) us per launch)
D  max B (f32)              : $(join(["$k:$(v["max_B_f32"])" for (k, v) in sort(collect(d), by = first)], " "))
E  terms/SM -> sieve verdict: $terms_per_sm -> $verdict

Full results: $out""")
