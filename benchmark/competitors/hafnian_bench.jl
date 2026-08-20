# Benchmark TheEggman.jl's `hafnian` / `hafnian_repeated` against Xanadu's thewalrus, Budapest
# QCG's piquasso and Quandela's perceval. Counterpart to `bench_thewalrus.py`,
# `bench_piquasso.py` and `bench_perceval.py` in this folder; run them and feed the JSON output to
# `plot_comparison.py`, or just run `benchmark/run_comparison.jl` which drives everything.
#
# thewalrus and piquasso are independent implementations of the same Björklund/Glynn
# O(N^3 2^(N/2)) sieve (arXiv:2108.01622) that TheEggman.jl falls back to, so those comparisons
# measure implementation quality rather than asymptotics wherever the Julia side also sieves.
# Three regimes are covered:
#
#   rpt=1: an N x N matrix of N distinct rows -- plain `hafnian(A)`.
#   rpt=2: N/2 distinct rows, each doubled (e.g. 2 photons detected per mode) --
#          `hafnian_repeated(A, rpt)`. Pairing the repeats up front shrinks the sieve from
#          2^(N/2) terms on N x N matrices to 2*3^(N/4-1) terms on (N/2) x (N/2) ones, so this
#          regime is far cheaper than rpt=1 at the same total degree N.
#   perm:  an N x N block-antidiagonal matrix [0 B; Bᵀ 0] built from a d x d B, d = N/2, whose
#          hafnian is exactly perm(B). Only run when --perm is passed, since it exists solely to
#          give perceval something to be compared against -- perceval has no hafnian, only
#          permanents -- and it adds half again to the runtime of this script. It is the
#          general-purpose hafnian's worst case, since neither Julia nor thewalrus detects the
#          structure that lets a permanent routine get the same number in O(2^d d^2).
#
# Runnable directly:
#     julia --project=benchmark -t auto benchmark/competitors/hafnian_bench.jl [bench_dir] [--perm]

using TheEggman
using BenchmarkTools
using Random
using JSON

with_perm = "--perm" in ARGS  # the perceval comparison regime; off by default
args = filter(!=("--perm"), ARGS)
bench_dir = length(args) > 0 ? args[1] : ".benchmarks"

const NS = (4, 8, 12, 16, 20, 24, 28, 32, 36)

# BenchmarkTools' 5s-per-case default would keep every core pegged for over a minute, leaving the
# CPU hot (and clocked down) for whichever stage runs next. Keep the total short enough that the
# two libraries are measured under comparable thermal conditions.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0

const SUITE = BenchmarkGroup()

for N in NS
    # rpt=1: N distinct rows, each used once.
    Random.seed!(N)
    A1 = let B = randn(ComplexF64, N, N)
        B + transpose(B)
    end
    SUITE["haf.eggman.rpt1.N=$N"] = @benchmarkable hafnian($A1; check_symmetric=false)

    # rpt=2: N/2 distinct rows, each doubled.
    d = N ÷ 2
    Random.seed!(N + 1)
    A2 = let B = randn(ComplexF64, d, d)
        B + transpose(B)
    end
    rpt = fill(2, d)
    SUITE["haf.eggman.rpt2.N=$N"] = @benchmarkable hafnian_repeated($A2, $rpt; check_symmetric=false)

    if with_perm
        # perm: haf([0 B; Bᵀ 0]) == perm(B), the regime perceval competes in.
        Random.seed!(N + 2)
        A3 = let B = randn(ComplexF64, d, d), Z = zeros(ComplexF64, d, d)
            [Z B; transpose(B) Z]
        end
        SUITE["haf.eggman.perm.N=$N"] = @benchmarkable hafnian($A3; check_symmetric=false)
    end
end

# To prevent very fast samples from hitting the time_ns() granularity floor
tune!(SUITE)

results = run(SUITE)
for name in sort(collect(keys(results)))
    println("$name:")
    display(results[name])
    println()
end

mkpath(bench_dir)
BenchmarkTools.save(joinpath(bench_dir, "jl-eggman-hafnian-bench.json"), results)

# thewalrus parallelises its sieve with numba's `prange` and we parallelise ours with Julia tasks,
# so the thread count is part of the result and belongs in the plot.
open(joinpath(bench_dir, "jl-eggman-hafnian-meta.json"), "w") do io
    JSON.print(io, Dict(
        "nthreads" => Threads.nthreads(),
        "julia_version" => string(VERSION),
        "cpu" => Sys.cpu_info()[1].model,
    ))
end
println("Saved to $(joinpath(bench_dir, "jl-eggman-hafnian-bench.json"))")
