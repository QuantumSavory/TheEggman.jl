"""Grouped scatter comparing TheEggman.jl's hafnian/hafnian_repeated against Xanadu's thewalrus and
Budapest QCG's piquasso at matching total degrees N, in two side-by-side panels:

  rpt=1 (left):  N distinct rows, each used once -- plain `hafnian(A)`.
  rpt=2 (right): N/2 distinct rows, each doubled -- `hafnian_repeated(A, rpt)`.

Both Python libraries always run the Bjoerklund/Glynn O(N^3 2^(N/2)) sieve (arXiv:2108.01622) --
thewalrus and piquasso are independent numba implementations of it, and piquasso folds repetitions
into the sieve the same way `hafnian_repeated` does. TheEggman.jl picks between three strategies by
cost, so most of these groups compare different algorithms rather than different implementations of
one: an unrolled matching sum at N<=12, a subset DP up to N=28, and the same sieve when repeated
rows make it cheap (which is why the rpt=2 panel falls back to sieve-vs-sieve at the larger N).

Under each panel is a small table of mean ratios against TheEggman.jl, one column per group.

perceval is not here: it has no hafnian, and is plotted separately by plot_perceval.py on the one
regime it can be compared on. See benchmark/thewalrus/README.md.

Usage: python plot_comparison.py [bench_dir]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import plot_common as pc

bench_dir = sys.argv[1] if len(sys.argv) > 1 else ".benchmarks"

plt.rcParams.update({"font.size": 8})

panels = pc.panels(pc.load_times(bench_dir))
variants = ["thewalrus", "piquasso", "eggman"]
regimes = [r for r in ("rpt1", "rpt2") if r in panels]

fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 8), sharey=True)
if len(regimes) == 1:
    axes = [axes]

rng = np.random.default_rng(0)
for i, (ax, regime) in enumerate(zip(axes, regimes)):
    pc.draw_panel(ax, panels[regime], variants, pc.PANEL_TITLES[regime], rng)
    # Row labels only under the leftmost panel: both panels carry the same two rows, and only
    # that one has figure margin to the left of it.
    pc.draw_ratio_table(ax, panels[regime], variants, ["thewalrus", "piquasso"],
                        with_labels=(i == 0))

axes[0].set_ylabel("Execution Time (s)")

subtitle = "TheEggman.jl picks per problem: unrolled sum / subset DP / sieve"
hardware = pc.hardware_subtitle(bench_dir)
if hardware:
    subtitle += f" - {hardware}"
fig.suptitle(f"Hafnian: TheEggman.jl vs thewalrus vs piquasso\n{subtitle}", fontsize=10)
pc.add_legend(fig, variants)

fig.tight_layout(rect=(0, 0.04, 1, 1))
# Room under the axes for the ratio table and its row labels, which tight_layout cannot see.
fig.subplots_adjust(bottom=0.21, left=0.175)
out = f"{bench_dir}/thewalrus_benchmark_comparison.svg"
fig.savefig(out)
print(f"Saved plot to {out}")
