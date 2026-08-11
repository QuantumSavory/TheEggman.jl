#!/usr/bin/env julia

# One-shot benchmark driver: times TheEggman.jl, times thewalrus in its own uv-managed virtualenv,
# and plots the two against each other.
#
#     julia benchmark/run_comparison.jl [bench_dir] [--cooldown SECONDS]
#
# With no argument the results land in a timestamped directory under `.benchmarks/`. Requires
# `uv` (https://docs.astral.sh/uv/) on PATH; the Python venv is created and synced on first run.
#
# Each stage runs as its own subprocess so the Julia benchmark gets `-t auto` regardless of how
# this script was started, and so a failure in one stage reports which one. The two timing stages
# are separated by an idle cooldown: both libraries saturate every core, and on a laptop the
# second one to run would otherwise be measured on an already-throttled CPU.

using Dates

const ROOT = dirname(@__DIR__)
const BENCH = joinpath(ROOT, "benchmark")
const PYPROJ = joinpath(BENCH, "thewalrus")

args = copy(ARGS)
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
          "the isolated Python environment thewalrus is benchmarked in.")
end

# thewalrus parallelises with numba over every core, so give Julia the whole machine too.
cool_down(cooldown)
run_stage("julia", `$(Base.julia_cmd()[1]) --project=$BENCH -t auto --startup-file=no
                    $(joinpath(PYPROJ, "hafnian_bench.jl")) $bench_dir`)

cool_down(cooldown)
run_stage("thewalrus", `uv run --project $PYPROJ python $(joinpath(PYPROJ, "bench_thewalrus.py")) $bench_dir`)

run_stage("plot", `uv run --project $PYPROJ python $(joinpath(PYPROJ, "plot_comparison.py")) $bench_dir`)

println("\nResults in $bench_dir")
for f in sort(readdir(bench_dir))
    println("  ", f)
end
