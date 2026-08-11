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

`hafnian` sums `∏ A[i,j]` over all perfect matchings of the row indices. Two algorithms cover the
range, chosen automatically by a cost comparison.

**Small degrees (`N ≤ TheEggman.UNROLL_MAX`, currently 12)** evaluate the definition directly, as a
single branch-free expression emitted by a `@generated` function. Expanding the matching sum naively
would produce `(N-1)!!` products — 10395 at `N = 12` — so the generator memoises on the
remaining-index bitmask and emits one temporary per distinct subset instead, collapsing the code to
the size of the underlying DAG:

| N  | subsets emitted | multiplies | multiplies if unshared |
|----|-----------------|------------|------------------------|
| 8  | 33              | 87         | 315                    |
| 10 | 88              | 317        | 3780                   |
| 12 | 232             | 1055       | 51975                  |

This is 7×–52× faster than the sieve below the cap, and more accurate too — it is the definition,
with none of the cancellation between large signed terms that a sieve relies on. The cap is set by
compile time, not by where the kernel stops winning; the kernels are precompiled for `Float64` and
`ComplexF64`, which is most of the package's ~3s precompile.

**Larger degrees** use the `O(N³ 2^(N/2))` finite-difference sieve of
[Björklund, Gupt & Quesada](https://arxiv.org/abs/2108.01622), the same algorithm as `thewalrus`:
fix a perfect matching, sum over sign patterns on its edges, and read off a single coefficient of a
generating function for each.

`hafnian_repeated` exploits repeated rows by pairing the repeats into as few distinct edges as
possible, which turns the sieve into a much shorter mixed-radix sum over *multiplicities*. Enough
repetition shrinks the sieve below even the unrolled kernel, so the crossover is a cost comparison
rather than a size cutoff — `rpt = [6, 6]` sieves, `rpt = [5, 5, 1, 1]` does not.

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
be worth it.

Results agree with `thewalrus` to ~1e-14 relative on random complex symmetric matrices.

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
| 8  | 657x                      | 234x                         |
| 12 | 85x                       | 52x                          |
| 16 | 6.3x                      | 6.4x                         |
| 20 | 3.0x                      | 15.8x                        |
| 24 | 2.7x                      | 16.3x                        |
| 28 | 6.1x                      | 6.9x                         |

The step at N=16 is where the unrolled kernels stop and both libraries are running the same sieve.

Both libraries use every core, so these are wall-clock ratios on a thermally-constrained laptop and
the run-to-run spread is wide (the sieve-only sizes range from 2.7x to 6.6x depending on whether
medians or best-case times are compared). The ordering is the stable part: TheEggman.jl was faster
at every size in both regimes.
