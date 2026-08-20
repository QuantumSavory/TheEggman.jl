"""Benchmark thewalrus.hafnian / hafnian_repeated against TheEggman.jl's equivalents. Counterpart
to hafnian_bench.jl in this folder; run them and feed the JSON output to plot_comparison.py, or
just run benchmark/run_comparison.jl which drives everything.

thewalrus and TheEggman.jl implement the same Bjoerklund/Glynn O(N^3 2^(N/2)) sieve
(arXiv:2108.01622), so the comparison between those two is one of implementation quality. Three
regimes, matching hafnian_bench.jl:

  rpt=1: an N x N matrix of N distinct rows -- plain `hafnian(A)`.
  rpt=2: N/2 distinct rows, each doubled (e.g. 2 photons detected per mode) --
         `hafnian_repeated(A, rpt)`, which pairs the repeats up front and runs a much smaller
         sieve.
  perm:  an N x N block-antidiagonal matrix [[0, B], [B.T, 0]] built from a d x d B, d = N/2,
         whose hafnian is perm(B). Only run when --perm is passed, since it exists solely to give
         perceval something to be compared against -- see bench_perceval.py -- and it roughly
         doubles the cost of this script. It is a general-purpose hafnian's worst case, since
         neither library detects the structure that lets a permanent routine do the same work in
         O(2^d d^2).

thewalrus parallelises its sieve internally with numba's `prange`, so these timings already use
every core; the version and thread count land in py-thewalrus-hafnian-meta.json, and the Julia side
records its own in jl-eggman-hafnian-meta.json.

Runs in the isolated uv-managed venv at benchmark/competitors/ (kept separate from any other project
venv because thewalrus pins numba, which caps the interpreter at Python <=3.12):

    uv run --project benchmark/competitors python benchmark/competitors/bench_thewalrus.py \
        [bench_dir] [--perm]
"""

import json
import os
import sys
import time

import numba
import numpy as np
import thewalrus
from thewalrus import hafnian, hafnian_repeated

args = sys.argv[1:]
with_perm = "--perm" in args  # the perceval comparison regime; off by default
args = [a for a in args if a != "--perm"]
bench_dir = args[0] if args else ".benchmarks"

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

    if not with_perm:
        continue

    # perm: haf([[0, B], [B.T, 0]]) == perm(B), the regime perceval competes in.
    rng3 = np.random.default_rng(N + 2)
    B3 = rng3.standard_normal((d, d)) + 1j * rng3.standard_normal((d, d))
    Z = np.zeros((d, d), dtype=complex)
    A3 = np.block([[Z, B3], [B3.T, Z]])
    times3 = _time(lambda: hafnian(A3), rounds)
    results[f"haf.thewalrus.perm.N={N}"] = times3
    print(f"haf.thewalrus.perm.N={N}: mean={np.mean(times3) * 1e3:.4f}ms  std={np.std(times3) * 1e3:.4f}ms  n={rounds}")

os.makedirs(bench_dir, exist_ok=True)
out_path = os.path.join(bench_dir, "py-thewalrus-hafnian-bench.json")
with open(out_path, "w") as f:
    json.dump(results, f)

# thewalrus parallelises its sieve with numba's `prange`, so the thread count is part of the
# result, as is the version -- the sieve has been reworked across releases.
with open(os.path.join(bench_dir, "py-thewalrus-hafnian-meta.json"), "w") as f:
    json.dump({
        "thewalrus_version": thewalrus.__version__,
        "numba_threads": numba.get_num_threads(),
    }, f)
print(f"Saved to {out_path}")
