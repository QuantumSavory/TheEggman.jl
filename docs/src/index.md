```@meta
CurrentModule = TheEggman
```

# TheEggman

Documentation for [TheEggman](https://github.com/JacobGunnell/TheEggman.jl), a Julia port of
[thewalrus](https://github.com/XanaduAI/thewalrus).

## Hafnians

The hafnian of a symmetric matrix `A` is the sum, over all perfect matchings of its row indices, of
the product of the matched entries. It plays the role for symmetric matrices that the permanent
plays for general ones, and it is the quantity that shows up in Gaussian boson sampling
probabilities.

```jldoctest
julia> using TheEggman

julia> A = [0 1 2 3; 1 0 4 5; 2 4 0 6; 3 5 6 0];

julia> hafnian(A)          # 1*6 + 2*5 + 3*4
28.0
```

When rows repeat, [`hafnian_repeated`](@ref) is much faster than expanding the matrix and calling
[`hafnian`](@ref), because its cost is driven by the number of *distinct* rows:

```jldoctest
julia> using TheEggman

julia> B = [1.0 2.0; 2.0 3.0];

julia> hafnian_repeated(B, [2, 2]) ≈ hafnian(TheEggman.reduction(B, [2, 2]))
true
```

Which algorithm runs is chosen automatically, by comparing costs that are all known before any work
starts. `method=:unrolled` uses a `@generated` branch-free expansion of the definition (degrees up to
`TheEggman.UNROLL_MAX`); `method=:dp` evaluates the same recursion over memoised subsets, whose count
grows only like `φ^K` (chosen automatically up to `TheEggman.DP_MAX`, and available on request up to
`TheEggman.DP_HARD_MAX` on a wider layout, for machines with the memory for a multi-gigabyte plan); `method=:sieve` runs the `O(N³ 2^(N/2))`
finite-difference sieve of [Björklund, Gupt & Quesada](https://arxiv.org/abs/2108.01622), which is
the fallback at large degrees and the best choice when repeated rows shrink it. See `src/unrolled.jl`,
`src/dp.jl` and `src/hafnian.jl` respectively.

The sieve has two variants, and which one is faster depends on whether rows repeat, so it is chosen
per problem alongside the strategy. Inclusion–exclusion wins on distinct rows — 1.3× at `N=20`
widening to 2.05× at `N=36` — while Glynn wins by 1.5–1.8× once rows repeat, where it is also one to
two decimal digits more accurate. `glynn=true` or `false` forces one; the keyword has no effect on
the other two strategies.

Matrices are read in place — views and `Symmetric` wrappers are never copied — and must be 1-based.
Symmetry is validated on every call, which is `O(N²)` and so a noticeable fraction of the total at
small `N`; `check_symmetric=false` skips it for callers that already know their input is symmetric.

## Batches and GPUs

Both entry points take batches, which on a GPU is what makes the port worthwhile:

```julia
hafnian(As)                                        # a vector of equally-sized matrices
hafnian_repeated(A, rpts)                          # one matrix, many repetition patterns
hafnian(As; backend = CUDABackend())               # ...on a GPU
```

Loading `KernelAbstractions` (with a backend package such as CUDA.jl) enables `backend`, which runs
the subset DP on the device; every other strategy falls back to the CPU. Device results differ from
the CPU in the last ulp or two (device compilers fuse multiply-add, which is slightly *more*
accurate), so compare across backends with a tolerance; within one backend results are exactly
reproducible. Passing `ComplexF32` matrices halves the memory traffic for about seven digits of
accuracy — worthwhile on consumer cards, where `Float64` runs at a fraction of `Float32` rate.

`TheEggman.gpu_cache_bytes()`, `TheEggman.empty_gpu_cache!()` and `TheEggman.dp_batch_bytes` manage
the device-side plan cache and batch sizing.

Loop hafnians are not implemented yet.

## Index

```@index
```

## API

```@autodocs
Modules = [TheEggman]
```
