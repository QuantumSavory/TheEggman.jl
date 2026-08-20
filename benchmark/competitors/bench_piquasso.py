"""Benchmark piquasso's hafnian against TheEggman.jl's and thewalrus's. Counterpart to
hafnian_bench.jl and bench_thewalrus.py in this folder; run them and feed the JSON output to
plot_comparison.py, or just run benchmark/run_comparison.jl which drives everything.

piquasso (Budapest Quantum Computing Group) is the third independent implementation of the same
Bjoerklund/Glynn O(N^3 2^(N/2)) power-trace sieve (arXiv:2108.01622) in this comparison -- a numba
translation of the PiquassoBoost C++ code, `@njit(parallel=True)` over every core, like thewalrus.
Its one entry point covers both regimes benchmarked here:

    hafnian_with_reduction(matrix, occupation_numbers)

is `hafnian(A)` when the occupation numbers are all 1, and `hafnian_repeated(A, rpt)` when they are
the repeat counts -- it folds repetitions into the sieve exactly as thewalrus's `hafnian_repeated`
and TheEggman.jl's `hafnian_repeated` do. So piquasso appears in both panels of the main plot:

  rpt=1: an N x N matrix of N distinct rows, occupation numbers all 1.
  rpt=2: N/2 distinct rows, each doubled (e.g. 2 photons detected per mode).

The matrices match bench_thewalrus.py seed for seed, and the results are checked against
thewalrus in this same venv before any timing is done.

Runs in the isolated uv-managed venv at benchmark/competitors/:

    uv run --project benchmark/competitors python benchmark/competitors/bench_piquasso.py [bench_dir]
"""

import json
import os
import sys
import time

import numba
import numpy as np
import piquasso
from piquasso._math.hafnian import hafnian_with_reduction
from thewalrus import hafnian, hafnian_repeated

bench_dir = sys.argv[1] if len(sys.argv) > 1 else ".benchmarks"

# Matches the N sweep in hafnian_bench.jl. Round counts fall off with N because the rpt=1 cost
# doubles with every step of 2 in N.
ROUNDS = {4: 200, 8: 200, 12: 200, 16: 100, 20: 50, 24: 20, 28: 10, 32: 5, 36: 3}


def _time(fn, rounds):
    fn()  # warmup: numba JIT-compiles the kernel on first call per signature
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _check(got, want, what):
    assert abs(got - want) <= 1e-8 * abs(want), f"{what}: piquasso {got} vs thewalrus {want}"


results = {}
for N, rounds in ROUNDS.items():
    # rpt=1: N distinct rows, occupation numbers all 1. Same matrix as bench_thewalrus.py.
    rng = np.random.default_rng(N)
    B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A1 = B + B.T
    occ1 = np.ones(N, dtype=np.int64)
    _check(hafnian_with_reduction(A1, occ1), hafnian(A1), f"rpt1 N={N}")
    times1 = _time(lambda: hafnian_with_reduction(A1, occ1), rounds)
    results[f"haf.piquasso.rpt1.N={N}"] = times1
    print(f"haf.piquasso.rpt1.N={N}: mean={np.mean(times1) * 1e3:.4f}ms  std={np.std(times1) * 1e3:.4f}ms  n={rounds}")

    # rpt=2: N/2 distinct rows, each repeated twice.
    d = N // 2
    rng2 = np.random.default_rng(N + 1)
    B2 = rng2.standard_normal((d, d)) + 1j * rng2.standard_normal((d, d))
    A2 = B2 + B2.T
    occ2 = np.full(d, 2, dtype=np.int64)
    _check(hafnian_with_reduction(A2, occ2), hafnian_repeated(A2, [2] * d), f"rpt2 N={N}")
    times2 = _time(lambda: hafnian_with_reduction(A2, occ2), rounds)
    results[f"haf.piquasso.rpt2.N={N}"] = times2
    print(f"haf.piquasso.rpt2.N={N}: mean={np.mean(times2) * 1e3:.4f}ms  std={np.std(times2) * 1e3:.4f}ms  n={rounds}")

os.makedirs(bench_dir, exist_ok=True)
out_path = os.path.join(bench_dir, "py-piquasso-hafnian-bench.json")
with open(out_path, "w") as f:
    json.dump(results, f)

# piquasso's sieve is `parallel=True`, so numba's thread count is part of the result.
with open(os.path.join(bench_dir, "py-piquasso-hafnian-meta.json"), "w") as f:
    json.dump({
        "piquasso_version": piquasso.__version__,
        "numba_threads": numba.get_num_threads(),
    }, f)
print(f"Saved to {out_path}")
