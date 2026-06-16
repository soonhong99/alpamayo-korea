# -*- coding: utf-8 -*-
"""Block-verify speculative decoding, told in two pictures and almost no words.
Left: 16 passes vs 1 pass. Right: accept the matching prefix, cut at the first
mismatch (output stays identical)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
GREEN, RED, GRAY, INK = "#1f7a52", "#c0392b", "#c2c8cf", "#222"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3),
                               gridspec_kw={"width_ratios": [1, 1.05]})
for ax in (axL, axR):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.invert_yaxis()


def cell(ax, x, y, w, h, fc, ec=None, alpha=1.0, r=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={r}",
                 fc=fc, ec=ec or fc, lw=1.4, alpha=alpha))


# ============ LEFT: 16 passes  vs  1 pass ============
axL.text(50, 8, "Speed", ha="center", fontsize=14, fontweight="bold", color=INK)

# normal: 16 separate narrow passes (a comb)
axL.text(8, 26, "Normal decode", fontsize=12, color=INK, fontweight="bold")
n = 16
x0, x1 = 8, 92
cw = (x1 - x0) / n * 0.62
for i in range(n):
    cx = x0 + (x1 - x0) * (i + 0.5) / n
    cell(axL, cx - cw/2, 32, cw, 12, GRAY)
axL.annotate("", xy=(x1, 53), xytext=(x0, 53),
             arrowprops=dict(arrowstyle="-", color="#999", lw=1))
axL.text(50, 58, "16 passes  ·  reads the 15 GB weights 16×",
         ha="center", fontsize=10.5, color="#666")

# speculative: one wide pass
axL.text(8, 74, "Speculative", fontsize=12, color=INK, fontweight="bold")
cell(axL, x0, 80, x1 - x0, 12, GREEN, alpha=0.85)
axL.text(50, 86, "1 pass", ha="center", fontsize=12.5, color="white", fontweight="bold")

# big 16x
axL.text(50, 67, "16×", ha="center", fontsize=20, color=GREEN, fontweight="bold")


# ============ RIGHT: accept prefix, cut at first mismatch ============
axR.text(50, 8, "Exactness", ha="center", fontsize=14, fontweight="bold", color=INK)

# the guessed sentence (draft) as a strip; verified in ONE pass
m = 8
gx0, gx1 = 8, 92
gw = (gx1 - gx0) / m * 0.82
cut = 5  # first 5 match, 6th mismatches
for i in range(m):
    cx = gx0 + (gx1 - gx0) * (i + 0.5) / m
    if i < cut:
        cell(axR, cx - gw/2, 40, gw, 13, GREEN, alpha=0.88)
    elif i == cut:
        cell(axR, cx - gw/2, 40, gw, 13, RED, alpha=0.9)
    else:
        cell(axR, cx - gw/2, 40, gw, 13, GRAY, alpha=0.45)

# cut line right after the matching prefix
cutx = gx0 + (gx1 - gx0) * cut / m
redcx = gx0 + (gx1 - gx0) * (cut + 0.5) / m
axR.plot([cutx, cutx], [34, 59], color=RED, lw=1.8, ls=(0, (4, 3)))

# top labels: keep (over green span) + first mismatch (over red cell)
axR.annotate("", xy=(cutx - 1, 35), xytext=(gx0, 35),
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=2.2))
axR.text((gx0 + cutx) / 2, 29, "keep — identical", ha="center",
         fontsize=11, color=GREEN, fontweight="bold")
axR.text(redcx, 29, "mismatch", ha="center", fontsize=10.5, color=RED, fontweight="bold")

# bottom label: re-decode (under faded span)
axR.annotate("", xy=(gx1, 59), xytext=(cutx + 1, 59),
             arrowprops=dict(arrowstyle="-", color="#bbb", lw=2.2))
axR.text((cutx + gx1) / 2 + 3, 65, "re-decode from here", ha="center",
         fontsize=10.5, color="#999")

fig.savefig("260616_fig15_block_verify.png", dpi=200, bbox_inches="tight")
print("saved 260616_fig15_block_verify.png")
