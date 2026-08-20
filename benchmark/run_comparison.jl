#!/usr/bin/env julia

# One-shot benchmark driver: times TheEggman.jl against thewalrus and piquasso, which run in a
# shared uv-managed virtualenv, and plots the three against each other.
#
#     julia benchmark/run_comparison.jl [bench_dir] [--cooldown SECONDS] [--perceval]
#
# With no argument the results land in a timestamped directory under `.benchmarks/`. Requires
# `uv` (https://docs.astral.sh/uv/) on PATH; the Python venv is created and synced on first run.
#
# `--perceval` additionally runs the permanent-as-hafnian regime and writes a *second* image,
# `perceval_benchmark_comparison.svg`. It is off by default because perceval has no hafnian: it
# answers a different question, on a regime that exists only to give it something to be compared
# against, and benchmarking that regime roughly doubles the two hafnian stages. See
# benchmark/thewalrus/README.md.
#
# Each stage runs as its own subprocess so the Julia benchmark gets `-t auto` regardless of how
# this script was started, and so a failure in one stage reports which one. The timing stages are
# separated by an idle cooldown: every library saturates every core, and on a laptop the ones
# running later would otherwise be measured on an already-throttled CPU.

using Dates

const ROOT = dirname(@__DIR__)
const BENCH = joinpath(ROOT, "benchmark")
const PYPROJ = joinpath(BENCH, "thewalrus")

args = copy(ARGS)
with_perceval = "--perceval" in args
filter!(!=("--perceval"), args)

cooldown = 30.0
let i = findfirst(==("--cooldown"), args)
    if i !== nothing
        i < length(args) || error("--cooldown needs a value in seconds")
        cooldown = parse(Float64, args[i+1])
        deleteat!(args, i:i+1)
    end
end

bench_dir = if length(args) > 0
    args[1]
else
    joinpath(ROOT, ".benchmarks", Dates.format(now(), "yyyy-mm-dd-HHMMSS"))
end
mkpath(bench_dir)

function cool_down(seconds)
    seconds > 0 || return
    println("\n[cooldown] idling $(seconds)s so the next stage starts from the same thermal state")
    sleep(seconds)
end

function run_stage(name, cmd)
    println("\n", "="^88)
    println("[$name] ", join(cmd.exec, " "))
    println("="^88)
    try
        run(cmd)
    catch err
        err isa ProcessFailedException || rethrow()
        error("benchmark stage '$name' failed; see the output above")
    end
end

if Sys.which("uv") === nothing
    error("`uv` was not found on PATH. Install it from https://docs.astral.sh/uv/ — it manages " *
          "the isolated Python environment the Python libraries are benchmarked in.")
end

# The `--perm` regime only earns its runtime if the perceval plot is going to be drawn from it.
perm_flag = with_perceval ? ["--perm"] : String[]

# thewalrus and piquasso both parallelise with numba over every core, and perceval's permanent uses
# its own C++ thread pool, so give Julia the whole machine too.
cool_down(cooldown)
run_stage("julia", `$(Base.julia_cmd()[1]) --project=$BENCH -t auto --startup-file=no
                    $(joinpath(PYPROJ, "hafnian_bench.jl")) $bench_dir $perm_flag`)

cool_down(cooldown)
run_stage("thewalrus", `uv run --project $PYPROJ python $(joinpath(PYPROJ, "bench_thewalrus.py")) $bench_dir $perm_flag`)

cool_down(cooldown)
run_stage("piquasso", `uv run --project $PYPROJ python $(joinpath(PYPROJ, "bench_piquasso.py")) $bench_dir`)

if with_perceval
    cool_down(cooldown)
    run_stage("perceval", `uv run --project $PYPROJ python $(joinpath(PYPROJ, "bench_perceval.py")) $bench_dir`)
end

run_stage("plot", `uv run --project $PYPROJ python $(joinpath(PYPROJ, "plot_comparison.py")) $bench_dir`)

if with_perceval
    run_stage("plot-perceval", `uv run --project $PYPROJ python $(joinpath(PYPROJ, "plot_perceval.py")) $bench_dir`)
end

println("\nResults in $bench_dir")
for f in sort(readdir(bench_dir))
    println("  ", f)
end
