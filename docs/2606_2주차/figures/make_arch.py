# -*- coding: utf-8 -*-
"""UMIC overview — what UMIC applies to each of Alpamayo's 4 stages to reach
-29.8% with no model change and no quantization. Paper-style matplotlib schematic."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

BLUE, BLUEE = "#dbe7f3", "#1f4e79"
TEAL, TEALE = "#dcefe6", "#0f6e56"
GRAY, GRAYE = "#ebe9e2", "#5f5e5a"
GREEN, GREENE = "#e7f1d8", "#3b6d11"

fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")


def box(x, y, w, h, title, sub, fc, ec, tc, ts=12, ss=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    if sub:
        ax.text(x + w/2, y + h*0.63, title, ha="center", va="center", color=tc,
                fontsize=ts, fontweight="bold")
        ax.text(x + w/2, y + h*0.27, sub, ha="center", va="center", color=ec, fontsize=ss,
                style="italic")
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", color=tc, fontsize=ts,
                fontweight="bold")


def arrow(x1, y1, x2, y2, color="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.6, color=color))


# ---- pipeline (top) ----
py, ph = 4.25, 0.95
stages = [
    (0.2, 1.5, "Multi-camera\nvideo", "", GRAY, GRAYE, "#3b3a36"),
    (2.0, 1.95, "Vision Encoder", "perceive", BLUE, BLUEE, BLUEE),
    (4.15, 1.95, "LM Prefill", "read scene", BLUE, BLUEE, BLUEE),
    (6.3, 1.95, "LM Decode", "reason (tokens)", BLUE, BLUEE, BLUEE),
    (8.45, 1.95, "Flow Matching", "draw trajectory", BLUE, BLUEE, BLUEE),
    (10.7, 2.1, "6.4 s trajectory\n(64 waypoints)", "", GRAY, GRAYE, "#3b3a36"),
]
for x, w, t, s, fc, ec, tc in stages:
    box(x, py, w, ph, t, s, fc, ec, tc, ts=12, ss=9.5)
xs = [1.7, 3.95, 6.1, 8.25, 10.4, 12.8]
for i in range(5):
    arrow(xs[i], py + ph/2, xs[i] + 0.28, py + ph/2)

# ---- per-stage UMIC technique (below) ----
ty, th = 2.0, 1.55
tech = [
    (2.0, 1.95, "fuse LayerNorm\n& RoPE", "$-$43%"),
    (4.15, 1.95, "fuse FFN/RMSNorm/RoPE\n+ custom GEMM", "$-$46%"),
    (6.3, 1.95, "in-place KV cache\n+ CUDA Graph", "$-$11%  (at floor)"),
    (8.45, 1.95, "in-place KV\n+ fusion reuse", "$-$38%"),
]
for x, w, t, pct in tech:
    ax.add_patch(FancyBboxPatch((x, ty), w, th, boxstyle="round,pad=0.015,rounding_size=0.12",
                                fc=TEAL, ec=TEALE, lw=1.3))
    ax.text(x + w/2, ty + th*0.67, t, ha="center", va="center", color=TEALE, fontsize=10)
    ax.text(x + w/2, ty + th*0.24, pct, ha="center", va="center", color="#0a4d3c",
            fontsize=14, fontweight="bold")
    ax.add_patch(FancyArrowPatch((x + w/2, py), (x + w/2, ty + th), arrowstyle="-",
                                 lw=1.0, ls=(0, (3, 2)), color="#9aa"))

ax.text(2.0, 3.80, "What UMIC applies at each stage  $\\downarrow$", ha="left", va="center",
        fontsize=11, color="#666", style="italic")

# ---- principle + result banner (bottom) ----
ax.add_patch(FancyBboxPatch((0.2, 0.35), 12.2, 1.05, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc=GREEN, ec=GREENE, lw=1.4))
ax.text(6.3, 1.04, "Measurement-guided (ncu) $\\cdot$ model unmodified $\\cdot$ no quantization "
        "$\\cdot$ output bit-identical", ha="center", va="center", fontsize=12.5, color=GREENE,
        fontweight="bold")
ax.text(6.3, 0.62, "end-to-end inference   3,846 ms  $\\rightarrow$  2,701 ms    "
        "($-$29.8%, same conditions vs eager)", ha="center", va="center", fontsize=12.5,
        color="#1c3a06")

fig.tight_layout()
fig.savefig("260614_fig9_umic_architecture.png", dpi=200, bbox_inches="tight")
print("saved 260614_fig9_umic_architecture.png")
