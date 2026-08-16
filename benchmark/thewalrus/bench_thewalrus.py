"""Benchmark thewalrus.hafnian / hafnian_repeated against TheEggman.jl's equivalents. Counterpart
to hafnian_bench.jl in this folder; run both and feed their JSON output to plot_comparison.py, or
just run benchmark/run_comparison.jl which drives all three.

Both libraries implement the same Bjoerklund/Glynn O(N^3 2^(N/2)) sieve (arXiv:2108.01622), so the
comparison is one of implementation quality. Two regimes, matching hafnian_bench.jl:

  rpt=1: an N x N matrix of N distinct rows -- plain `hafnian(A)`.
  rpt=2: N/2 distinct rows, each doubled (e.g. 2 photons detected per mode) --
         `hafnian_repeated(A, rpt)`, which pairs the repeats up front and runs a much smaller
         sieve.

thewalrus parallelises its sieve internally with numba's `prange`, so these timings already use
every core; the Julia side records its own thread count in jl-thewalrus-hafnian-meta.json.

Runs in the isolated uv-managed venv at benchmark/thewalrus/ (kept separate from any other project
venv because thewalrus pins numba, which caps the interpreter at Python <=3.12):

    uv run --project benchmark/thewalrus python benchmark/thewalrus/bench_thewalrus.py [bench_dir]
"""

import json
import os
import sys
import time

import numpy as np
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


results = {}
for N, rounds in ROUNDS.items():
    # rpt=1: plain hafnian on an N x N matrix of N distinct rows.
    rng = np.random.default_rng(N)
    B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A1 = B + B.T
    times1 = _time(lambda: hafnian(A1), rounds)
    results[f"haf.thewalrus.rpt1.N={N}"] = times1
    print(f"haf.thewalrus.rpt1.N={N}: mean={np.mean(times1) * 1e3:.4f}ms  std={np.std(times1) * 1e3:.4f}ms  n={rounds}")

    # rpt=2: N/2 distinct rows, each repeated twice.
    d = N // 2
    rng2 = np.random.default_rng(N + 1)
    B2 = rng2.standard_normal((d, d)) + 1j * rng2.standard_normal((d, d))
    A2 = B2 + B2.T
    rpt = [2] * d
    times2 = _time(lambda: hafnian_repeated(A2, rpt), rounds)
    results[f"haf.thewalrus.rpt2.N={N}"] = times2
    print(f"haf.thewalrus.rpt2.N={N}: mean={np.mean(times2) * 1e3:.4f}ms  std={np.std(times2) * 1e3:.4f}ms  n={rounds}")

os.makedirs(bench_dir, exist_ok=True)
out_path = os.path.join(bench_dir, "py-thewalrus-hafnian-bench.json")
with open(out_path, "w") as f:
    json.dump(results, f)
print(f"Saved to {out_path}")
