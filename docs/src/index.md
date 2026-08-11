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
grows only like `φ^K` (degrees up to `TheEggman.DP_MAX`); `method=:sieve` runs the `O(N³ 2^(N/2))`
finite-difference sieve of [Björklund, Gupt & Quesada](https://arxiv.org/abs/2108.01622), which is
the fallback at large degrees and the best choice when repeated rows shrink it. See `src/unrolled.jl`,
`src/dp.jl` and `src/hafnian.jl` respectively.

`glynn=false` selects the inclusion–exclusion sieve variant instead of the default Glynn one (about
2.5× faster, about 1000× less accurate), and has no effect on the other two strategies.

Loop hafnians are not implemented yet.

## Index

```@index
```

## API

```@autodocs
Modules = [TheEggman]
```
