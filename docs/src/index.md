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

Which algorithm runs is chosen automatically. Above total degree `TheEggman.UNROLL_MAX` (12) it is
the `O(N³ 2^(N/2))` finite-difference sieve of
[Björklund, Gupt & Quesada](https://arxiv.org/abs/2108.01622) — see `src/hafnian.jl` for the identity
being evaluated and where this implementation diverges from `thewalrus`. At or below the cap, a
`@generated` function emits the sum over perfect matchings as one branch-free expression, which is
both faster and more accurate; see `src/unrolled.jl`. Repeated rows can shrink the sieve below even
that, so the choice is a cost comparison rather than a size cutoff.

Pass `unrolled=false` to force the sieve, or `glynn=false` to select the inclusion–exclusion sieve
variant instead of the default Glynn one (about 2.5× faster, about 1000× less accurate).

Loop hafnians are not implemented yet.

## Index

```@index
```

## API

```@autodocs
Modules = [TheEggman]
```
