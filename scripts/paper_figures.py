"""Draws the paper's figures from the result files.

    python scripts/paper_figures.py
Writes paper/figures/*.pdf. Colours follow one rule: blue for a run whose forgetting held, red diamonds (always labelled) for a relapse.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
HELD, RELAPSE = "#2a78d6", "#d03b3b"
WINDOWS = ("3-5", "5-7", "7-9", "10-12", "13-15")
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "pdf.fonttype": 42})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.tick_params(length=2, width=0.5)


def relapse_map() -> None:
    runs = [r for r in json.load(open(os.path.join(ROOT, "results", "map", "summary.json")))["runs"] if r["effective"]]
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 1, figsize=(3.03, 2.45), sharex=True, sharey=True)
    for ax, (fmt, label) in zip(axes, (("Q4_0", "Q4_0"), ("nf4", "NF4"))):
        style(ax)
        ax.axhline(0, color=AXIS, linewidth=0.6)
        ax.axhline(5, color=RELAPSE, linewidth=0.6, linestyle=(0, (3, 2)))
        for i, w in enumerate(WINDOWS):
            rs = [r for r in runs if r["window"] == w]
            x = i + rng.uniform(-0.26, 0.26, len(rs)); y = np.array([100 * r[fmt]["delta"] for r in rs]); rel = np.array([r[fmt]["relapse"] for r in rs])
            ax.scatter(x[~rel], y[~rel], s=9, color=HELD, alpha=0.75, linewidths=0.3, edgecolors="white")
            ax.scatter(x[rel], y[rel], s=22, color=RELAPSE, marker="D", linewidths=0.4, edgecolors="white", zorder=3)
            if rel.any():
                ax.annotate("Tiny Aya Base,\nseed 12", (x[rel].mean(), y[rel].max()), xytext=(i - 1.65, 8.6), fontsize=6.5, color=INK, va="center",
                            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5})
        ax.text(4.42, 5.35, "relapse line", fontsize=6, color=RELAPSE, ha="right")
        ax.text(-0.4, 9.2, label, fontsize=8, color=INK, weight="bold", va="top")
        ax.set_ylim(-8, 10); ax.set_yticks((-5, 0, 5)); ax.set_yticklabels(("−5", "0", "+5"))
    axes[1].set_xticks(range(len(WINDOWS))); axes[1].set_xticklabels([w.replace("-", "–") for w in WINDOWS]); axes[1].set_xlabel("Edited layers", color=INK)
    fig.supylabel("WMDP-bio in the 4-bit copy minus fp16 (points)", fontsize=7.5, color=INK, x=0.02)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.98, bottom=0.13, hspace=0.12)
    fig.savefig(os.path.join(OUT, "map.pdf")); fig.savefig(os.path.join(OUT, "map.png"), dpi=220); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    relapse_map()
