# GPU validation

Verifies TheEggman.jl's GPU subset-DP port and quantifies its speedup. Run on a GPU-enabled
machine:

```sh
julia --project=benchmark/gpu -t auto benchmark/gpu/run_gpu_validation.jl
julia --project=benchmark/gpu -t auto benchmark/gpu/run_gpu_validation.jl --quick   # shorter sweep
```

It writes `gpu-validation.json` here, prints a summary, and ends with a fill-in template to send
back. Nothing it does mutates the repository.

## Smoke-testing without a GPU

```sh
EGGMAN_GPU_DRYRUN=1 julia --project=benchmark/gpu -t auto benchmark/gpu/run_gpu_validation.jl --quick
```

runs every stage against KernelAbstractions' `CPU()` backend with stubbed device queries (and skips
loading CUDA entirely). The timings are then meaningless, but it proves the script itself works.
This is how it was developed, since the development machine has no GPU.

**What the dry run cannot catch.** The CPU backend allocates plain `Array`s, so it exercises kernel
logic, batching, chunking and the plan cache — but not device dispatch. Two classes of bug are
invisible to it and have both actually occurred:

- *Transfers.* `copyto!` with a view on either side misses the host↔device methods, falls through to
  Base's generic implementation and scalar-indexes the device array. All transfers therefore use the
  five-argument `copyto!(dest, doffs, src, soffs, n)` form on plain arrays; `test/gpu.jl` lints for
  this, since no CPU test can.
- *Arithmetic.* Device compilers fuse multiply-add, so results differ from the CPU in the last ulp.
  Compare across backends with a tolerance.

## Stages

| stage | what it establishes |
|---|---|
| **A** environment | device, compute capability, VRAM, and the fp64:fp32 ratio — every later number is meaningless without these |
| **B** correctness | eight checks that must all pass before any timing is worth reading |
| **C** performance | crossover `N`, batched throughput, fp32 gain, plan-upload amortisation, per-launch overhead |
| **D** memory | plan cache size and the largest batch that fits |
| **E** sieve viability | measures whether a future sieve port could reach useful occupancy |

## What the results mean

**B1 checks agreement with the CPU DP to a tight tolerance, not exactly.** Nothing in the kernel
reorders a sum — each state is accumulated by one thread in plan order — but device compilers
contract `a*b + c` into a single-rounding FMA, which moves the last ulp. Measured, that leaves a
~1e-16 to 2e-15 relative gap, and the contracted result is the *more* accurate one. The threshold is
1e-12: far above the FP noise, far below anything a real bug would hide under.

The exactness that does hold is *within* a backend: B2 (batched versus scalar), B6 (determinism) and
B7 (chunking invariance) all use `===`, because none of them changes how any single sum is
evaluated. A failure in one of those is a structural bug.

**C1 is expected to lose at small N.** A call needs roughly `N/2` kernel launches, a floor a single
instance cannot amortise. That is what **C2** is for: batching pays those launches once for the
whole batch, which is where the port should earn its keep.

**C3 near 1.0× would be informative, not disappointing.** It would mean the kernel is bandwidth-bound
rather than limited by fp64 throughput — worth knowing on a consumer card, where fp64 runs at
1/32–1/64 rate.

**E is a go/no-go input, not a result.** A sieve term needs roughly a 40 KB workspace at `m = 32`.
If fewer than two terms fit per SM, a sieve port would have too little occupancy to repay its
barrier-heavy, pivoting-dependent reduction, and should not be attempted.
