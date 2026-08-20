"""Benchmark Quandela's perceval against TheEggman.jl and thewalrus on the permanent-as-hafnian
regime. Counterpart to hafnian_bench.jl and bench_thewalrus.py in this folder; run all three and
feed their JSON output to plot_comparison.py, or just run benchmark/run_comparison.jl.

perceval has no hafnian -- it is a Fock-state linear-optics simulator, and every one of its
back-ends (SLOS, Naive, MPS, Stepper, SLAP) is built on the *permanent*. The two libraries do meet
on one task, because the permanent is a hafnian of a block-antidiagonal matrix:

    perm(B) = haf([[0, B], [B.T, 0]])

so at total degree N a d = N/2 permanent and an N x N hafnian return the same number. That is the
regime benchmarked here, and it is the one perceval is built for: it computes perm(B) directly in
`O(2^d d^2)` while a general-purpose hafnian sees only an N x N symmetric matrix and pays the full
`O(N^3 2^(N/2))` (or subset-DP) cost for structure it cannot exploit.

What is timed is `exqalibur.permanent_cx`, the exact multithreaded C++ Ryser/Glynn permanent. That
is the kernel of perceval's `Naive` back-end verbatim -- `NaiveBackend.prob_amplitude` builds a
submatrix and returns `xq.permanent_cx(M)` up to a normalisation constant (see
perceval/backends/_naive.py) -- and it is also the only entry point that can take a general complex
matrix, since every perceval back-end is configured with a circuit and simulates its
`compute_unitary()`. Calling the kernel directly is therefore both the fairest option (none of
perceval's Fock-state bookkeeping is counted against it) and the only general one.

Runs in the same isolated uv-managed venv as thewalrus, which is also used here to check the
reduction actually holds before any timing is done:

    uv run --project benchmark/thewalrus python benchmark/thewalrus/bench_perceval.py [bench_dir]
"""

import json
import os
import sys
import time

import exqalibur as xq
import numpy as np
import perceval
from thewalrus import hafnian

bench_dir = sys.argv[1] if len(sys.argv) > 1 else ".benchmarks"

# Matches the N sweep in hafnian_bench.jl / bench_thewalrus.py. perceval only appears in the perm
# regime, but it is timed at every N so the panel lines up with the other two.
ROUNDS = {4: 200, 8: 200, 12: 200, 16: 100, 20: 50, 24: 20, 28: 10, 32: 5, 36: 3}


def block(B):
    """The symmetric N x N matrix whose hafnian is perm(B), for d x d B and N = 2d."""
    Z = np.zeros_like(B)
    return np.block([[Z, B], [B.T, Z]])


def _time(fn, rounds):
    fn()  # warmup: first call pays exqalibur's thread-pool spin-up
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


# Sanity-check the reduction the whole panel rests on, against the other library in this venv.
_rng = np.random.default_rng(0)
_B = np.ascontiguousarray(_rng.standard_normal((5, 5)) + 1j * _rng.standard_normal((5, 5)))
_p, _h = xq.permanent_cx(_B), hafnian(block(_B))
assert abs(_p - _h) <= 1e-10 * abs(_h), f"perm/haf reduction disagrees: {_p} vs {_h}"
print(f"perm(B) == haf([[0,B],[B.T,0]]) to {abs(_p - _h) / abs(_h):.1e} relative at d=5")

results = {}
for N, rounds in ROUNDS.items():
    # Same construction as the perm regime in hafnian_bench.jl and bench_thewalrus.py: a d x d
    # complex matrix B, timed here as perm(B) and there as haf of the 2d x 2d block matrix.
    d = N // 2
    rng = np.random.default_rng(N + 2)
    B = np.ascontiguousarray(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
    times = _time(lambda: xq.permanent_cx(B), rounds)
    results[f"haf.perceval.perm.N={N}"] = times
    print(f"haf.perceval.perm.N={N}: mean={np.mean(times) * 1e3:.4f}ms  std={np.std(times) * 1e3:.4f}ms  n={rounds}")

os.makedirs(bench_dir, exist_ok=True)
out_path = os.path.join(bench_dir, "py-perceval-perm-bench.json")
with open(out_path, "w") as f:
    json.dump(results, f)

# exqalibur parallelises the permanent over every core by default, and its algorithm choice depends
# on the thread count (Glynn at 1-2 threads, Ryser above), so both belong in the result.
threads = xq.Config.compute_max_thread_count or (os.cpu_count() or 0)
with open(os.path.join(bench_dir, "py-perceval-perm-meta.json"), "w") as f:
    json.dump({
        "perceval_version": perceval.__version__,
        "nthreads": threads,
        "algorithm": "glynn" if threads <= 2 else "ryser",
    }, f)
print(f"Saved to {out_path}")
