# Benchmark TheEggman.jl's `hafnian` / `hafnian_repeated` against Xanadu's thewalrus. Counterpart
# to `bench_thewalrus.py` in this folder; run both and feed their JSON output to
# `plot_comparison.py`, or just run `benchmark/run_comparison.jl` which drives all three.
#
# Both libraries implement the same Björklund/Glynn O(N^3 2^(N/2)) sieve (arXiv:2108.01622), so
# this measures implementation quality rather than asymptotics. Two regimes are covered:
#
#   rpt=1: an N x N matrix of N distinct rows -- plain `hafnian(A)`.
#   rpt=2: N/2 distinct rows, each doubled (e.g. 2 photons detected per mode) --
#          `hafnian_repeated(A, rpt)`. Pairing the repeats up front shrinks the sieve from
#          2^(N/2) terms on N x N matrices to 2*3^(N/4-1) terms on (N/2) x (N/2) ones, so this
#          regime is far cheaper than rpt=1 at the same total degree N.
#
# Runnable directly:
#     julia --project=benchmark -t auto benchmark/thewalrus/hafnian_bench.jl [bench_dir]

using TheEggman
using BenchmarkTools
using Random
using JSON

bench_dir = length(ARGS) > 0 ? ARGS[1] : ".benchmarks"

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
BenchmarkTools.save(joinpath(bench_dir, "jl-thewalrus-hafnian-bench.json"), results)

# thewalrus parallelises its sieve with numba's `prange` and we parallelise ours with Julia tasks,
# so the thread count is part of the result and belongs in the plot.
open(joinpath(bench_dir, "jl-thewalrus-hafnian-meta.json"), "w") do io
    JSON.print(io, Dict(
        "nthreads" => Threads.nthreads(),
        "julia_version" => string(VERSION),
        "cpu" => Sys.cpu_info()[1].model,
    ))
end
println("Saved to $(joinpath(bench_dir, "jl-thewalrus-hafnian-bench.json"))")
