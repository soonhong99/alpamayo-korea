#!/usr/bin/env python3
"""
Alpamayo 1.5 — Per-Stage Estimated Bandwidth Figure
Output: docs/2605_5주차/figures/260605_bw_pipeline_analysis.png
"""

import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit("pip install matplotlib numpy")

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "docs" / "2605_5주차" / "figures" / "260605_bw_pipeline_analysis.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ─── Data ────────────────────────────────────────────────────────────────────
STAGES = ["VE\n(728 ms)", "LM Prefill\n(1,423 ms)", "Decode\n(79 ms/step)", "Flow\n(87 ms/step)"]
BW     = [5, 45, 207, 53]
COLORS = ["#27ae60", "#2980b9", "#e74c3c", "#e67e22"]

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.spines.left" : False,
    "xtick.direction"  : "out",
    "ytick.major.size" : 0,
})

fig, ax = plt.subplots(figsize=(7, 3.8))
fig.subplots_adjust(left=0.24, right=0.85, top=0.95, bottom=0.15)

y = np.arange(len(STAGES))

# ─── Bars ────────────────────────────────────────────────────────────────────
for i, (bw, color) in enumerate(zip(BW, COLORS)):
    ax.barh(y[i], bw, height=0.5, color=color, alpha=0.85, zorder=3)
    ax.text(bw + 4, y[i], f"{bw} GB/s",
            va="center", ha="left", fontsize=11,
            color=color, fontweight="bold")

# ─── Axes ────────────────────────────────────────────────────────────────────
ax.set_yticks(y)
ax.set_yticklabels(STAGES, fontsize=10.5)
ax.invert_yaxis()
ax.set_xlim(0, 260)
ax.set_xlabel("Estimated Bandwidth  (GB/s)", fontsize=11, labelpad=6)
ax.xaxis.grid(True, linestyle=":", alpha=0.35, zorder=0)
ax.set_axisbelow(True)

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"[OK] {OUT}")
