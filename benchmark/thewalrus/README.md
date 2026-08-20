# Hafnian benchmark

Compares TheEggman.jl's `hafnian` / `hafnian_repeated` against other open-source implementations.
Two images come out of it:

| image | libraries | produced by |
|---|---|---|
| `thewalrus_benchmark_comparison.svg` | thewalrus, piquasso, TheEggman.jl | always |
| `perceval_benchmark_comparison.svg`  | thewalrus, perceval, TheEggman.jl | `--perceval` only |

## Who is in the main plot, and why

- **[thewalrus](https://github.com/XanaduAI/thewalrus)** (Xanadu) — the package TheEggman.jl ports.
  Always runs the Björklund/Glynn `O(N^3 2^(N/2))` finite-difference sieve
  ([arXiv:2108.01622](https://arxiv.org/abs/2108.01622)), parallelised with numba's `prange`.
- **[piquasso](https://github.com/Budapest-Quantum-Computing-Group/piquasso)** (Budapest QCG) — an
  independent implementation of that same sieve: a numba translation of the PiquassoBoost C++ code,
  `@njit(parallel=True)`. Its single entry point `hafnian_with_reduction(matrix, occupation_numbers)`
  covers both regimes here — occupation numbers all 1 is the plain hafnian, and repeat counts fold
  into the sieve exactly as `hafnian_repeated` does — so it appears in both panels.
- **TheEggman.jl** — chooses per problem between an unrolled matching sum (N ≤ 12), a subset DP
  (N ≤ 32) and the same sieve, so most groups compare *different algorithms* rather than
  implementations of one. The exception is the rpt=2 panel from N=20 on, where repetition has made
  the sieve the cheapest option for everyone and the comparison is sieve-vs-sieve.

`bench_piquasso.py` checks each piquasso result against thewalrus in the same venv before timing it;
they agree to ~1e-14 relative.

## Who is not, and why

**[perceval](https://github.com/Quandela/Perceval)** (Quandela) has **no hafnian**. It is a
Fock-state linear-optics simulator, and every one of its back-ends — `SLOS`, `Naive`, `MPS`,
`Stepper`, `SLAP` — is built on the **permanent**; there is no Gaussian/GBS module, and no `haf`
symbol anywhere in the Python package or in the compiled `exqalibur` core.

It does meet the others on one task, because the permanent is a hafnian of a block-antidiagonal
matrix:

```
perm(B) = haf([0 B; Bᵀ 0])
```

so at total degree `N` a `d = N/2` permanent and an `N×N` hafnian return the same number. That is
the `perm` regime, and it gets **its own image** because it asks a different question from the main
plot — not "whose hafnian is faster" but "what does a hafnian cost you when the problem was really a
permanent". It is a general-purpose hafnian's worst case by construction: perceval reads `perm(B)`
off the `d×d` `B` in `O(2^d d²)`, while neither `hafnian` implementation detects the block structure
and both pay full price on the `N×N` matrix they are handed. `bench_perceval.py` asserts the
identity against `thewalrus.hafnian` before timing anything, so the panel is known to be measuring
one quantity.

What is timed is `exqalibur.permanent_cx`, the exact multithreaded C++ Ryser/Glynn permanent (Glynn
at 1–2 threads, Ryser above). That is the kernel of perceval's `Naive` back-end verbatim —
`NaiveBackend.prob_amplitude` builds a submatrix and returns `xq.permanent_cx(M)` up to a
normalisation constant (`perceval/backends/_naive.py`). It is also the only entry point that can
take the matrix at all: every perceval back-end is configured with a *circuit* and simulates its
`compute_unitary()` — `SLOS` calls `self._slos.set_unitary(self._umat)` — so none of them accepts a
general complex `B`. Going straight to the kernel is therefore both the fairest option (no
Fock-state bookkeeping counted against perceval) and the only general one.

**[StrawberryFields](https://github.com/XanaduAI/strawberryfields)** (Xanadu) is not benchmarked at
all, because it has no hafnian to benchmark: it defines no hafnian or permanent kernel of its own,
ships no compiled extension, and every hafnian in the package is imported from thewalrus
(`from thewalrus.samples import hafnian_sample_state`, `from thewalrus._hafnian import reduction`,
`import thewalrus.quantum as twq`). Timing it would be timing thewalrus a second time with an extra
Python frame on top. A separate plot was considered and dropped for the same reason — unlike
perceval, there is no different-but-valid quantity it could be compared on. (If what you want is
the cost of a hafnian *inside a GBS simulator's public API*, that is a different benchmark and not
one this folder does.)

## Regimes

Three regimes at total degree N = 4, 8, 12, 16, 20, 24, 28, 32, 36:

- **rpt=1** (distinct rows): an N×N random complex symmetric matrix — `hafnian(A)` against
  `thewalrus.hafnian(A)` and `piquasso.hafnian_with_reduction(A, ones(N))`.
- **rpt=2** (repeated rows): N/2 distinct rows, each doubled (e.g. 2 photons detected per mode) —
  `hafnian_repeated(A, rpt)` against `thewalrus.hafnian_repeated(A, rpt)` and
  `piquasso.hafnian_with_reduction(A, rpt)`. Pairing the repeats up front shrinks the sieve from
  `2^(N/2)` terms on N×N matrices to `2·3^(N/4-1)` terms on (N/2)×(N/2) ones, so this regime is far
  cheaper than rpt=1 at the same N in every library.
- **perm** (permanent as hafnian, `--perceval` only): `haf([0 B; Bᵀ 0])` for a random complex
  (N/2)×(N/2) `B`, against `thewalrus.hafnian` of the same block matrix and
  `exqalibur.permanent_cx(B)`. All three return the same complex scalar. This regime is skipped by
  default — it exists only to give perceval something to be compared against, and running it adds
  roughly half again to the Julia stage and doubles the thewalrus one.

All the Python libraries saturate every core — thewalrus and piquasso through numba, perceval
through exqalibur's own thread pool — and TheEggman.jl does in its sieve, so the numbers are
whole-machine wall clock.

No side's warmup is timed: the Python scripts call each function once before measuring so numba's
JIT and exqalibur's thread-pool spin-up are done, and BenchmarkTools does the same, which also
builds TheEggman.jl's DP plan (up to ~250 ms at N=28). A cold one-shot call at a new degree pays
that; a benchmark loop does not.

The Python side lives in one [uv](https://docs.astral.sh/uv/)-managed venv, separate from any other
project venv because thewalrus pins `numba`, which caps the interpreter at Python ≤3.12. piquasso
and perceval share it; both resolve against the same numpy and add no conflicting pins.

## Usage

```sh
julia benchmark/run_comparison.jl [bench_dir] [--cooldown SECONDS] [--perceval]
```

which runs, from the repo root:

```sh
julia --project=benchmark -t auto benchmark/thewalrus/hafnian_bench.jl "$BENCH_DIR" [--perm]
uv run --project benchmark/thewalrus python benchmark/thewalrus/bench_thewalrus.py "$BENCH_DIR" [--perm]
uv run --project benchmark/thewalrus python benchmark/thewalrus/bench_piquasso.py "$BENCH_DIR"
uv run --project benchmark/thewalrus python benchmark/thewalrus/bench_perceval.py "$BENCH_DIR"   # --perceval only
uv run --project benchmark/thewalrus python benchmark/thewalrus/plot_comparison.py "$BENCH_DIR"
uv run --project benchmark/thewalrus python benchmark/thewalrus/plot_perceval.py "$BENCH_DIR"    # --perceval only
```

writing `jl-thewalrus-hafnian-bench.json`, `jl-thewalrus-hafnian-meta.json`,
`py-thewalrus-hafnian-bench.json`, `py-piquasso-hafnian-bench.json`,
`py-piquasso-hafnian-meta.json` and `thewalrus_benchmark_comparison.svg` to a timestamped directory
under `.benchmarks/`, plus `py-perceval-perm-bench.json`, `py-perceval-perm-meta.json` and
`perceval_benchmark_comparison.svg` under `--perceval`.

`uv run --project benchmark/thewalrus` creates and syncs the venv on first use; run
`uv sync --project benchmark/thewalrus` to set it up ahead of time. Both plot scripts treat every
input file as optional, so they still work on a bench dir produced before a stage existed — the
missing series is simply absent. `plot_perceval.py` exits with a message if the dir holds no `perm`
results at all.

## Reading the results

The plots show every individual sample as a jittered dot, with a bar at the mean of each group.
Under each panel is a table of mean ratios against TheEggman.jl — thewalrus and piquasso in the
main plot, thewalrus and perceval in the perceval plot — with one column per group, aligned to the
groups above it.

Four things are worth knowing before trusting a single run:

- **Thermal coupling.** Every stage pegs every core, so whichever runs later is measured on a
  hotter, lower-clocked CPU. `run_comparison.jl` idles for `--cooldown` seconds (default 30) before
  each timing stage, and `hafnian_bench.jl` caps BenchmarkTools at 2s per case, to keep the sides
  comparable. On a laptop this reduces the bias rather than removing it.
- **Spread.** The Julia samples have a long right tail: thread spawn latency and GC pauses affect
  short runs, and this machine has both performance and efficiency cores, so an unlucky chunk
  assignment costs more than the kernel does. Means and minima can differ by 2x on the same data.
  The relative ordering of the libraries is stable across runs; the exact multiplier is not.
- **Timer resolution.** The smallest cases run in tens of nanoseconds, so a single call is at or
  below the granularity of `time_ns()` on some machines. `hafnian_bench.jl` calls `tune!` before
  `run` so each sample averages over enough evaluations (hundreds, at N=4) to be resolution-
  independent; without it those groups collapse onto the clock tick and onto BenchmarkTools'
  0.001ns floor for samples that measure zero. Tuning is why the Julia stage spends its first
  ~30s not reporting anything.
- **Fixed overheads dominate the small end.** Every Python call carries a floor of tens to hundreds
  of microseconds — interpreter dispatch, and for perceval handing work to a thread pool that a
  small permanent cannot repay — while TheEggman.jl's small cases are tens of nanoseconds. The
  three- and four-digit ratios at low N are largely that floor, not the algorithms. perceval's
  floor is the least reproducible number here: on the same laptop it has measured anywhere from
  0.15 ms to 1.7 ms per call depending on what else had just been running, which alone moves the
  crossover in the perceval plot between N ≈ 18 and N ≈ 22. Both hafnian sieves are far less
  sensitive, because their cost is arithmetic rather than dispatch.
