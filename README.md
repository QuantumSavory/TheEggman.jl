# TheEggman

[![Build Status](https://github.com/JacobGunnell/TheEggman.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JacobGunnell/TheEggman.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia port of [thewalrus](https://github.com/XanaduAI/thewalrus), Xanadu's library of matrix
functions for Gaussian quantum optics.

So far it implements the hafnian:

```julia
using TheEggman

A = [0 1 2 3
     1 0 4 5
     2 4 0 6
     3 5 6 0]

hafnian(A)                    # 28.0 == 1*6 + 2*5 + 3*4

# Hafnian of the matrix with row/col i repeated rpt[i] times, without ever building it.
B = [1.0 2.0; 2.0 3.0]
hafnian_repeated(B, [2, 2])   # 11.0
```

Loop hafnians are not implemented yet.

## Algorithm

`hafnian` sums `∏ A[i,j]` over all perfect matchings of the row indices. Three strategies cover the
range, chosen per problem by comparing costs that are all known in advance. `method=:unrolled`,
`:dp` or `:sieve` overrides the choice.

**`:unrolled` — degrees up to `TheEggman.UNROLL_MAX` (12).** Evaluates the definition as a single
branch-free expression emitted by a `@generated` function. Expanding the matching sum naively would
produce `(N-1)!!` products — 10395 at `N = 12` — so the generator memoises on the remaining-index
bitmask and emits one temporary per distinct subset, collapsing the code to the size of the
underlying DAG (232 subsets, 1055 multiplies at `N = 12`). Allocation-free, and the fastest option
at these sizes once repeated-call GC is counted.

**`:dp` — degrees up to `TheEggman.DP_MAX` (28).** The same recursion, evaluated at runtime over
memoised subsets. Because it always pairs off the *smallest* remaining index, the subsets it reaches
are constrained to `F(K+1)` of them — a Fibonacci number, so the state space grows like `φ^K ≈
1.618^K`, not `2^K`. That is worse asymptotically than the sieve's `1.414^K K³` (they cross around
`K ≈ 105`) but across the whole computable range the sieve does 65–100× more arithmetic:

| K              | 12   | 16    | 20    | 24     | 28      |
|----------------|------|-------|-------|--------|---------|
| DP transitions | 1076 | 10226 | 89665 | 748776 | 6052062 |
| sieve work     | 106k | 1.0M  | 7.9M  | 54M    | 393M    |

The subset structure depends only on the degree, so it is precomputed once per `K` into a cached
plan, and the evaluation is a flat CSR walk with no hashing or bitmask arithmetic — each transition
one packed `Int32`. Levels are evaluated in order and split across tasks within a level, where the
states are mutually independent; results are bit-identical regardless of thread count. Plans cost
4 bytes per transition and are built on first use: 0.4 MB / 3 ms at `N = 20`, 24 MB / 250 ms at
`N = 28`, then cached for the session. `DP_MAX` is where that stops being reasonable.

Parallel scaling tops out near 4× on 12 threads rather than approaching the thread count: two
indirect loads per multiply make the loop memory-bound once several cores are running. Degrees below
`N ≈ 20` run serially, where the levels are too small to repay a spawn.

**`:sieve` — everything else.** The `O(N³ 2^(N/2))` finite-difference sieve of
[Björklund, Gupt & Quesada](https://arxiv.org/abs/2108.01622), the algorithm `thewalrus` uses. It is
the fallback above `DP_MAX`, and it is also the *best* choice whenever repeated rows shrink it far
enough — `hafnian_repeated` pairs repeats into as few distinct edges as possible, turning the sieve
into a short mixed-radix sum over multiplicities, so e.g. `rpt = fill(2, 14)` sieves where 28
distinct rows would use the DP.

Where the sieve diverges from `thewalrus` is in its inner loop. `thewalrus` extracts the
generating-function coefficient from power traces `tr(Mᵏ)` built by `k` explicit matrix products,
`O(m⁴)` per sieve term. TheEggman.jl instead uses

```
exp( Σ tr(Mᵏ) λᵏ / 2k ) = det(I - λM)^(-1/2)
```

and reads every coefficient off the characteristic polynomial of `M` in one `O(m³)` pass
(Hessenberg reduction, then La Budde's recurrence). Two further choices, both measured rather than
assumed: the Hessenberg reduction uses Gaussian similarity transforms with partial pivoting
(half the flops of Householder, with no meaningful accuracy cost at these sizes), and for
`Complex{Float64}`/`Complex{Float32}` it runs on split real/imaginary arrays, which vectorise where
interleaved complex storage does not. The sieve is spread across threads once it is large enough to
be worth it; the other two strategies are single-threaded.

Both direct strategies are more accurate than either sieve variant, since they sum products of
matrix entries with no cancellation between large signed terms. Results agree with `thewalrus` to
~1e-14 relative on random complex symmetric matrices.

## Benchmarks

```sh
julia benchmark/run_comparison.jl
```

times TheEggman.jl, times `thewalrus` in an isolated [uv](https://docs.astral.sh/uv/)-managed
virtualenv, and writes a comparison plot to a timestamped directory under `.benchmarks/`. See
[`benchmark/thewalrus/README.md`](benchmark/thewalrus/README.md) for details and for how to run the
stages individually.

On a 12-thread i7-1365U, median speedup over `thewalrus` at total degree `N`:

| N  | `hafnian` (distinct rows) | `hafnian_repeated` (rpt = 2) |
|----|---------------------------|------------------------------|
| 8  | 388x                      | 418x                         |
| 12 | 138x                      | 72x                          |
| 16 | 67x                       | 19x                          |
| 20 | 54x                       | 21x                          |
| 24 | 92x                       | 15x                          |
| 28 | 72x                       | 13x                          |

The `hafnian` column is unrolled at N=8/12 and DP above; the `hafnian_repeated` column falls back to
the sieve from N=20 on, where repetition has made it the cheapest option, so those entries are
sieve-vs-sieve.

Both libraries use every core, so these are wall-clock ratios on a thermally-constrained laptop and
the run-to-run spread is wide — treat the exact multipliers as indicative and the ordering as the
stable part. Timings exclude one-time warmup on both sides: numba's JIT for thewalrus, DP plan
construction for TheEggman.jl.
