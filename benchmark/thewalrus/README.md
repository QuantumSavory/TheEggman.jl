# thewalrus benchmark

Compares TheEggman.jl's `hafnian` / `hafnian_repeated` against
[thewalrus](https://github.com/XanaduAI/thewalrus), the Xanadu package this one ports.

thewalrus always runs the Björklund/Glynn `O(N^3 2^(N/2))` finite-difference sieve
([arXiv:2108.01622](https://arxiv.org/abs/2108.01622)). TheEggman.jl chooses per problem between an
unrolled matching sum (N ≤ 12), a subset DP (N ≤ 32) and the same sieve, so most groups in this
sweep compare *different algorithms* rather than two implementations of one. The exception is the
rpt=2 panel from N=20 on, where repetition has made the sieve the cheapest option for both libraries
and the comparison is sieve-vs-sieve.

Both saturate every core — thewalrus through numba's `prange`, TheEggman.jl only in its sieve — so
the numbers are whole-machine wall clock.

Neither side's warmup is timed: the Python script calls each function once before measuring so
numba's JIT is done, and BenchmarkTools does the same, which also builds TheEggman.jl's DP plan (up
to ~250 ms at N=28). A cold one-shot call at a new degree pays that; a benchmark loop does not.

Two regimes are benchmarked at total degree N = 8, 12, 16, 20, 24, 28:

- **rpt=1** (distinct rows): an N×N random complex symmetric matrix — `hafnian(A)` against
  `thewalrus.hafnian(A)`.
- **rpt=2** (repeated rows): N/2 distinct rows, each doubled (e.g. 2 photons detected per mode) —
  `hafnian_repeated(A, rpt)` against `thewalrus.hafnian_repeated(A, rpt)`. Pairing the repeats up
  front shrinks the sieve from `2^(N/2)` terms on N×N matrices to `2·3^(N/4-1)` terms on
  (N/2)×(N/2) ones, so this regime is far cheaper than rpt=1 at the same N in both libraries.

The Python side lives in its own [uv](https://docs.astral.sh/uv/)-managed venv because thewalrus
pins `numba`, which caps the interpreter at Python ≤3.12.

## Usage

```sh
julia benchmark/run_comparison.jl [bench_dir] [--cooldown SECONDS]
```

which runs, from the repo root:

```sh
julia --project=benchmark -t auto benchmark/thewalrus/hafnian_bench.jl "$BENCH_DIR"
uv run --project benchmark/thewalrus python benchmark/thewalrus/bench_thewalrus.py "$BENCH_DIR"
uv run --project benchmark/thewalrus python benchmark/thewalrus/plot_comparison.py "$BENCH_DIR"
```

writing `jl-thewalrus-hafnian-bench.json`, `jl-thewalrus-hafnian-meta.json`,
`py-thewalrus-hafnian-bench.json`, and `thewalrus_benchmark_comparison.svg` to a timestamped
directory under `.benchmarks/`.

`uv run --project benchmark/thewalrus` creates and syncs the isolated venv on first use; run
`uv sync --project benchmark/thewalrus` to set it up ahead of time.

## Reading the results

The plot shows every individual sample as a jittered dot, with a bar at the median of each group
and the median speedup annotated above it.

Three things are worth knowing before trusting a single run:

- **Thermal coupling.** Both stages peg every core, so whichever runs second is measured on a
  hotter, lower-clocked CPU. `run_comparison.jl` idles for `--cooldown` seconds (default 30) before
  each timing stage, and `hafnian_bench.jl` caps BenchmarkTools at 2s per case, to keep the two
  sides comparable. On a laptop this reduces the bias rather than removing it.
- **Spread.** The Julia samples have a long right tail: thread spawn latency and GC pauses affect
  short runs, and this machine has both performance and efficiency cores, so an unlucky chunk
  assignment costs more than the kernel does. Medians and minima can differ by 2x on the same data.
  The relative ordering of the two libraries is stable across runs; the exact multiplier is not.
- **Timer resolution.** The smallest cases run in tens of nanoseconds, so a single call is at or
  below the granularity of `time_ns()` on some machines. `hafnian_bench.jl` calls `tune!` before
  `run` so each sample averages over enough evaluations (hundreds, at N=4) to be resolution-
  independent; without it those groups collapse onto the clock tick and onto BenchmarkTools'
  0.001ns floor for samples that measure zero. Tuning is why the Julia stage spends its first
  ~30s not reporting anything.
