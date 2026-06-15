# -*- coding: utf-8 -*-
"""FlashDrive vs UMIC (ours): where we differ. Paper-style comparison card."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

rows = [
    ("Quantization", "W4A8 (4-bit weights)", "none (full precision)"),
    ("Extra training", "trained DFlash drafter\n+ streaming fine-tune", "none (training-free)"),
    ("Model weights", "modified (fine-tuned)", "untouched (forward swap)"),
    ("Hardware focus", "discrete GPU (RTX)", "iGPU (Jetson Thor)"),
    ("Output", "approximate ($\\leq$0.08 m)", "bit-identical (core $-$29.8%)"),
]
FD, FDE = "#f3e3d2", "#9a6314"
US, USE = "#dcefe6", "#0f6e56"

fig, ax = plt.subplots(figsize=(10, 4.3))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")
cx = [0.2, 3.5, 6.75]; cw = [3.1, 3.05, 3.05]
rh = 0.92


def cell(x, y, w, h, txt, fc, ec, tc, bold=False, fs=11):
    if fc:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)


# header
hy = 5.5
cell(cx[1], hy, cw[1], rh, "FlashDrive (2026)", FD, FDE, FDE, bold=True, fs=12.5)
cell(cx[2], hy, cw[2], rh, "UMIC (ours)", US, USE, USE, bold=True, fs=12.5)
for i, (lab, fd, us) in enumerate(rows):
    y = hy - (i + 1) * (rh + 0.08)
    ax.text(cx[0] + 0.05, y + rh/2, lab, ha="left", va="center", fontsize=11.5,
            color="#333", fontweight="bold")
    cell(cx[1], y, cw[1], rh, fd, "#faf4ec", FDE, "#5a3a0c")
    cell(cx[2], y, cw[2], rh, us, "#eef8f2", USE, "#0a4d3c")

fig.savefig("260614_fig12_flashdrive_compare.png", dpi=200, bbox_inches="tight")
print("saved 260614_fig12_flashdrive_compare.png")
