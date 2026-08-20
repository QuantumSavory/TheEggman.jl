"""Shared machinery for the benchmark comparison images.

Two plots are produced from the same bench directory, by plot_comparison.py and plot_perceval.py.
They draw the same kind of grouped scatter -- one column of jittered samples per library, grouped
by total degree N, with a mean bar and a speedup annotation -- so the drawing lives here and the
scripts only choose which regimes and which libraries go in which figure.
"""

import json
import os
import re

import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D

VARIANT_COLORS = {
    "thewalrus": "steelblue",
    "piquasso": "forestgreen",
    "perceval": "darkviolet",
    "eggman": "coral",
}
VARIANT_LABELS = {
    "thewalrus": "thewalrus (Python + numba)",
    "piquasso": "piquasso (Python + numba)",
    "perceval": "perceval (Naive / exqalibur permanent)",
    "eggman": "TheEggman.jl",
}
# Short forms, for the ratio table's row labels.
VARIANT_SHORT = {
    "thewalrus": "thewalrus",
    "piquasso": "piquasso",
    "perceval": "perceval",
    "eggman": "TheEggman.jl",
}
PANEL_TITLES = {
    "rpt1": "rpt=1: distinct rows (hafnian)",
    "rpt2": "rpt=2: repeated rows (hafnian_repeated)",
    "perm": "perm: haf([0 B; Bᵀ 0]) = perm(B)",
}

# Every result file a bench dir may hold. Each is optional: a dir written before a stage existed,
# or by a run that skipped an optional stage, simply contributes no samples for that library.
PY_FILES = (
    "py-thewalrus-hafnian-bench.json",
    "py-piquasso-hafnian-bench.json",
    "py-perceval-perm-bench.json",
)
JL_FILE = "jl-eggman-hafnian-bench.json"

PATTERN = re.compile(r"^haf\.([a-z]+)\.(rpt1|rpt2|perm)\.N=(\d+)$")


def load(bench_dir, name):
    """Read one JSON file from the bench dir, or return None if the stage never ran."""
    path = os.path.join(bench_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_times(bench_dir):
    """All samples in the bench dir as {benchmark name -> seconds}, across every library."""
    times = {}
    for name in PY_FILES:
        src = load(bench_dir, name)
        if src:  # Python stages record raw seconds.
            times.update({k: np.array(v) for k, v in src.items()})
    jl = load(bench_dir, JL_FILE)
    if jl:  # BenchmarkTools records nanoseconds inside its own envelope.
        times.update({k: np.array(b[1]["times"]) / 1e9 for k, b in jl[1][0][1]["data"].items()})
    return times


def panels(times):
    """Group samples as {regime -> {N -> {variant -> samples}}}."""
    out = {}
    for name, ts in times.items():
        m = PATTERN.match(name)
        if m:
            variant, regime, N = m.group(1), m.group(2), int(m.group(3))
            out.setdefault(regime, {}).setdefault(N, {})[variant] = ts
    return out


def fmt_speedup(x):
    """Speedups here span 1000x to 0.001x, so a fixed number of decimals cannot cover them."""
    if x >= 10:
        return f"{x:.0f}x"
    if x >= 1:
        return f"{x:.1f}x"
    return f"{x:.2g}x"


def layout(groups, variants):
    """Where each group sits on the x axis: (Ns, group centers, per-variant column positions).

    draw_panel and draw_ratio_table both call this, so the table's columns land exactly under the
    groups they describe.
    """
    Ns = sorted(groups.keys())
    # One slot per variant present anywhere in the panel, plus a blank column between groups.
    stride = max(len([v for v in variants if v in groups[N]]) for N in Ns) + 2

    centers, columns = [], []
    for i, N in enumerate(Ns):
        present = [v for v in variants if v in groups[N]]
        base = stride * i
        columns.append({v: base + j + 1 for j, v in enumerate(present)})
        centers.append(base + (len(present) + 1) / 2)
    return Ns, centers, columns


def draw_panel(ax, groups, variants, title, rng, max_points=300):
    """One panel: grouped jittered scatter with a mean bar over each column.

    `groups` is {N -> {variant -> samples}}; `variants` fixes the left-to-right order within a
    group. The numbers themselves go in the ratio table underneath -- see draw_ratio_table.
    """
    Ns, group_centers, columns = layout(groups, variants)

    positions, data, colors = [], [], []
    for N, cols in zip(Ns, columns):
        for v, pos in cols.items():
            positions.append(pos)
            data.append(groups[N][v])
            colors.append(VARIANT_COLORS[v])

    for pos, values, color in zip(positions, data, colors):
        mean = np.mean(values)
        if len(values) > max_points:  # subsample dense series for a legible/lightweight plot
            values = rng.choice(values, size=max_points, replace=False)
        jitter = rng.uniform(-0.3, 0.3, size=len(values))
        ax.scatter(pos + jitter, values, s=6, color=color, alpha=0.5, edgecolors="none")
        ax.plot([pos - 0.38, pos + 0.38], [mean, mean], color="black", lw=1.1, zorder=3)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"N={N}" for N in Ns])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.5)


# Axes-fraction y of each ratio row, below the axes. The x tick labels sit just above these.
RATIO_ROW_Y0 = -0.075
RATIO_ROW_DY = -0.042


def draw_ratio_table(ax, groups, variants, numerators, denominator="eggman", with_labels=False):
    """A small table under the panel: one row per numerator, one column per N group.

    Each cell is mean(numerator) / mean(denominator) for that N, so the columns line up with the
    groups above them. Row labels are drawn once per figure -- pass with_labels only for the
    leftmost panel, which has the figure margin to put them in.
    """
    Ns, group_centers, _ = layout(groups, variants)
    # x in data coordinates (so cells track the groups), y in axes fractions (so rows sit below).
    trans = ax.get_xaxis_transform()

    for row, numerator in enumerate(numerators):
        y = RATIO_ROW_Y0 + RATIO_ROW_DY * row
        for center, N in zip(group_centers, Ns):
            have = {numerator, denominator} <= groups[N].keys()
            text = (fmt_speedup(np.mean(groups[N][numerator]) / np.mean(groups[N][denominator]))
                    if have else "–")
            ax.text(center, y, text, transform=trans, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="darkred", clip_on=False)
        if with_labels:
            ax.text(-0.015, y, f"{VARIANT_SHORT[numerator]} / {VARIANT_SHORT[denominator]}",
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=8, fontweight="bold", color="darkred", clip_on=False)

    # The x label goes under the table rather than between it and the tick labels.
    ax.set_xlabel("N (total degree)")
    ax.xaxis.set_label_coords(0.5, RATIO_ROW_Y0 + RATIO_ROW_DY * (len(numerators) + 0.4))


def add_legend(fig, variants):
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=VARIANT_COLORS[v],
                   markeredgecolor="none", markersize=6, label=VARIANT_LABELS[v])
            for v in variants
        ],
        loc="lower center", ncol=len(variants), frameon=False,
    )


def hardware_subtitle(bench_dir, include_perceval=False):
    """The versions and thread counts the timings depend on, for the figure subtitle.

    One entry per library actually drawn in that figure -- perceval and piquasso never share one.
    Each is skipped if its stage did not write a meta file.
    """
    parts = []
    jl = load(bench_dir, "jl-eggman-hafnian-meta.json") or {}
    if jl:
        parts.append(f"Julia {jl.get('julia_version', '?')}, {jl.get('nthreads', '?')} threads")
    tw = load(bench_dir, "py-thewalrus-hafnian-meta.json") or {}
    if tw:
        parts.append(f"thewalrus {tw.get('thewalrus_version', '?')}, "
                     f"{tw.get('numba_threads', '?')} numba threads")
    pq = load(bench_dir, "py-piquasso-hafnian-meta.json") or {}
    if pq and not include_perceval:
        parts.append(f"piquasso {pq.get('piquasso_version', '?')}, "
                     f"{pq.get('numba_threads', '?')} numba threads")
    pc = load(bench_dir, "py-perceval-perm-meta.json") or {}
    if pc and include_perceval:
        parts.append(f"perceval {pc.get('perceval_version', '?')} "
                     f"({pc.get('algorithm', '?')}, {pc.get('nthreads', '?')} threads)")
    return " - ".join(parts)
