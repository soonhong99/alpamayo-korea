# -*- coding: utf-8 -*-
"""DRAM traffic per stage, eager vs UMIC, with the decode 'floor' highlighted."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 13, "mathtext.fontset": "dejavuserif"})

stages = ["Vision Enc.", "Prefill", "Decode", "Flow"]
eager = [98.1, 232.0, 323.4, 122.1]
umic = [50.7, 78.3, 290.0, 83.9]
red = ["$-$48%", "$-$66%", "$-$10%", "$-$31%"]
x = np.arange(4); w = 0.38
C_E, C_U, C_FLOOR = "#9ca3af", "#1f4e79", "#d9952b"

fig, ax = plt.subplots(figsize=(8.2, 4.7))
ax.bar(x - w/2, eager, w, color=C_E, edgecolor="black", lw=1, hatch="///", label="eager (baseline)")
ucolors = [C_U, C_U, C_FLOOR, C_U]
ax.bar(x + w/2, umic, w, color=ucolors, edgecolor="black", lw=1, label="UMIC")

for xi, (e, u, r) in enumerate(zip(eager, umic, red)):
    ax.text(xi - w/2, e + 6, f"{e:.0f}", ha="center", va="bottom", fontsize=10.5, color="#555")
    rc = "#8a5a10" if xi == 2 else "#1f4e79"
    ax.text(xi + w/2, u + 8, f"{u:.0f} GB\n{r}", ha="center", va="bottom", fontsize=10.5,
            color=rc, fontweight="bold", linespacing=1.3)

ax.annotate("hardware floor:\nweights re-read every token,\ncannot be reduced",
            xy=(2 + w/2, 250), xytext=(0.45, 245), fontsize=9.5, color="#8a5a10",
            ha="center", arrowprops=dict(arrowstyle="->", color="#b07a1f", lw=1.3))

ax.set_ylabel("DRAM traffic (GB)")
ax.set_xticks(x); ax.set_xticklabels(stages)
ax.set_ylim(0, 380); ax.set_xlim(-0.7, 3.7)
ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0.0, 0.86))
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Memory traffic cut where avoidable; decode is at the hardware floor",
             fontsize=12.5, pad=12)
fig.savefig("260614_fig10_dram.png", dpi=200, bbox_inches="tight")
print("saved 260614_fig10_dram.png")
