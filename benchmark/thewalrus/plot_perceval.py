"""Single-panel scatter comparing Quandela's perceval against TheEggman.jl and thewalrus on the one
regime the three libraries share: the permanent as a hafnian.

perceval has no hafnian -- it is a Fock-state linear-optics simulator built entirely on permanents.
The libraries do meet on one task, because the permanent is a hafnian of a block-antidiagonal
matrix:

    perm(B) = haf([[0, B], [B.T, 0]])

so at total degree N a d = N/2 permanent and an N x N hafnian return the same number. That is the
regime plotted here, and it is a general-purpose hafnian's worst case by construction: perceval
reads perm(B) off the d x d B in O(2^d d^2), while neither hafnian implementation detects the block
structure and both pay full price on the N x N matrix they are handed.

This lives in its own image rather than alongside the hafnian panels because it is a different
question -- not "whose hafnian is faster" but "what does a hafnian cost you when the problem was
really a permanent". Produced only when benchmark/run_comparison.jl is passed --perceval.

Usage: python plot_perceval.py [bench_dir]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import plot_common as pc

bench_dir = sys.argv[1] if len(sys.argv) > 1 else ".benchmarks"

plt.rcParams.update({"font.size": 8})

panels = pc.panels(pc.load_times(bench_dir))
if "perm" not in panels:
    sys.exit(f"no perm-regime results in {bench_dir}; "
             "run `julia benchmark/run_comparison.jl --perceval` to produce them")

variants = ["thewalrus", "perceval", "eggman"]

fig, ax = plt.subplots(1, 1, figsize=(9, 8))
# Single panel, so the regime is named in the figure title instead of repeated over the axes.
pc.draw_panel(ax, panels["perm"], variants, "", np.random.default_rng(0))
pc.draw_ratio_table(ax, panels["perm"], variants, ["thewalrus", "perceval"], with_labels=True)
ax.set_ylabel("Execution Time (s)")

subtitle = ("haf([0 B; Bᵀ 0]) = perm(B), so all three return the same number "
            "— only perceval can exploit the block structure")
hardware = pc.hardware_subtitle(bench_dir, include_perceval=True)
if hardware:
    subtitle += f"\n{hardware}"
fig.suptitle(f"Permanent as hafnian: TheEggman.jl vs thewalrus vs perceval\n{subtitle}", fontsize=10)
pc.add_legend(fig, variants)

fig.tight_layout(rect=(0, 0.05, 1, 1))
# Room under the axes for the ratio table and its row labels, which tight_layout cannot see.
fig.subplots_adjust(bottom=0.21, left=0.245)
out = f"{bench_dir}/perceval_benchmark_comparison.svg"
fig.savefig(out)
print(f"Saved plot to {out}")
