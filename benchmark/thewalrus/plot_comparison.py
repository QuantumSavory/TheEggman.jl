"""Grouped scatter comparing thewalrus.hafnian/hafnian_repeated (Python + numba) against
TheEggman.jl's hafnian/hafnian_repeated at matching total degrees N, in two side-by-side panels:

  rpt=1 (left):  N distinct rows, each used once -- plain `hafnian(A)`.
  rpt=2 (right): N/2 distinct rows, each doubled -- `hafnian_repeated(A, rpt)`.

thewalrus always runs the Bjoerklund/Glynn O(N^3 2^(N/2)) sieve. TheEggman.jl picks between three
strategies by cost, so most of these groups are comparing different algorithms rather than different
implementations of one: an unrolled matching sum at N<=12, a subset DP up to N=28, and the same
sieve when repeated rows make it cheap (which is why the rpt=2 panel falls back to sieve-vs-sieve at
the larger N). The mean speedup is annotated above each group.

Usage: python plot_comparison.py [bench_dir]
"""

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

bench_dir = sys.argv[1] if len(sys.argv) > 1 else ".benchmarks"

plt.rcParams.update({"font.size": 8})

with open(f"{bench_dir}/py-thewalrus-hafnian-bench.json") as f:
    py = json.load(f)
with open(f"{bench_dir}/jl-thewalrus-hafnian-bench.json") as f:
    jl = json.load(f)

meta = {}
meta_path = f"{bench_dir}/jl-thewalrus-hafnian-meta.json"
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)

py_times = {name: np.array(times) for name, times in py.items()}
jl_times = {name: np.array(bm[1]["times"]) / 1e9 for name, bm in jl[1][0][1]["data"].items()}

py_pattern = re.compile(r"^haf\.thewalrus\.rpt(\d+)\.N=(\d+)$")
jl_pattern = re.compile(r"^haf\.eggman\.rpt(\d+)\.N=(\d+)$")

# rpt -> {N -> {variant: times}}
panels = {}
for name, times in py_times.items():
    m = py_pattern.match(name)
    if m:
        rpt, N = int(m.group(1)), int(m.group(2))
        panels.setdefault(rpt, {}).setdefault(N, {})["thewalrus"] = times
for name, times in jl_times.items():
    m = jl_pattern.match(name)
    if m:
        rpt, N = int(m.group(1)), int(m.group(2))
        panels.setdefault(rpt, {}).setdefault(N, {})["eggman"] = times

rpts = sorted(panels.keys())
variant_colors = {"eggman": "coral", "thewalrus": "steelblue"}
variant_labels = {
    "eggman": "TheEggman.jl",
    "thewalrus": "thewalrus (Python + numba)",
}
panel_titles = {
    1: "rpt=1: distinct rows (hafnian)",
    2: "rpt=2: repeated rows (hafnian_repeated)",
}
variants = ["eggman", "thewalrus"]

fig, axes = plt.subplots(1, len(rpts), figsize=(10, 7), sharey=True)
if len(rpts) == 1:
    axes = [axes]

rng = np.random.default_rng(0)
max_points = 300  # subsample dense series for a legible/lightweight plot
for ax, rpt in zip(axes, rpts):
    groups = panels[rpt]
    Ns = sorted(groups.keys())

    positions = []
    data = []
    colors = []
    group_centers = []
    for i, N in enumerate(Ns):
        present = [v for v in variants if v in groups[N]]
        base = 3 * i
        for j, v in enumerate(present):
            positions.append(base + j + 1)
            data.append(groups[N][v])
            colors.append(variant_colors[v])
        group_centers.append(base + (len(present) + 1) / 2)

    for pos, values, color in zip(positions, data, colors):
        mean = np.mean(values)
        if len(values) > max_points:
            values = rng.choice(values, size=max_points, replace=False)
        jitter = rng.uniform(-0.3, 0.3, size=len(values))
        ax.scatter(pos + jitter, values, s=6, color=color, alpha=0.5, edgecolors="none")
        ax.plot([pos - 0.38, pos + 0.38], [mean, mean], color="black", lw=1.1, zorder=3)

    # Speedup label above each group, using means.
    for center, N in zip(group_centers, Ns):
        if not {"eggman", "thewalrus"} <= groups[N].keys():
            continue
        speedup = np.mean(groups[N]["thewalrus"]) / np.mean(groups[N]["eggman"])
        top = max(np.mean(groups[N][v]) for v in variants if v in groups[N])
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(center, top),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="darkred",
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"N={N}" for N in Ns])
    ax.set_xlabel("N (total degree)")
    ax.set_yscale("log")
    ax.set_title(panel_titles.get(rpt, f"rpt={rpt}"))
    ax.grid(axis="y", alpha=0.3)
    # Headroom so the speedup label above the tallest group stays inside the axes.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 3)

axes[0].set_ylabel("Execution Time (s)")

subtitle = "TheEggman.jl picks per problem: unrolled sum / subset DP / sieve"
if meta:
    subtitle += f" - Julia {meta.get('julia_version', '?')}, {meta.get('nthreads', '?')} threads"
fig.suptitle(f"Hafnian: TheEggman.jl vs thewalrus\n{subtitle}", fontsize=10)
fig.legend(
    handles=[
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=variant_colors[v],
               markeredgecolor="none", markersize=6, label=variant_labels[v])
        for v in variants
    ],
    loc="lower center", ncol=2, frameon=False,
)

fig.tight_layout(rect=(0, 0.04, 1, 1))
out = f"{bench_dir}/thewalrus_benchmark_comparison.svg"
fig.savefig(out)
print(f"Saved plot to {out}")
