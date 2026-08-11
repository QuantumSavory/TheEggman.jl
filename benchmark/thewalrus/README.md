# thewalrus benchmark

Compares TheEggman.jl's `hafnian` / `hafnian_repeated` against
[thewalrus](https://github.com/XanaduAI/thewalrus), the Xanadu package this one ports.

thewalrus always runs the Björklund/Glynn `O(N^3 2^(N/2))` finite-difference sieve
([arXiv:2108.01622](https://arxiv.org/abs/2108.01622)). TheEggman.jl runs the same sieve above total
degree `UNROLL_MAX` (12) — there the comparison is one of implementations, not asymptotics — and
below it switches to a compile-time-unrolled sum over perfect matchings, which is a different
algorithm and wins by two orders of magnitude. The N=8 and N=12 groups therefore measure something
different from the rest of the sweep, and the plot's subtitle says so.

Both saturate every core — thewalrus through numba's `prange`, TheEggman.jl through Julia tasks —
so the numbers are whole-machine wall clock.

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

Two things are worth knowing before trusting a single run:

- **Thermal coupling.** Both stages peg every core, so whichever runs second is measured on a
  hotter, lower-clocked CPU. `run_comparison.jl` idles for `--cooldown` seconds (default 30) before
  each timing stage, and `hafnian_bench.jl` caps BenchmarkTools at 2s per case, to keep the two
  sides comparable. On a laptop this reduces the bias rather than removing it.
- **Spread.** The Julia samples have a long right tail: thread spawn latency and GC pauses affect
  short runs, and this machine has both performance and efficiency cores, so an unlucky chunk
  assignment costs more than the kernel does. Medians and minima can differ by 2x on the same data.
  The relative ordering of the two libraries is stable across runs; the exact multiplier is not.
