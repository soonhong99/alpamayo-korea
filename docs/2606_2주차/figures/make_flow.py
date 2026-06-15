# -*- coding: utf-8 -*-
"""Adaptive-step flow (FlashDrive technique, implemented on Thor): skipping the
redundant middle ODE steps cuts flow latency a lot for a few-cm trajectory cost."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 13, "mathtext.fontset": "dejavuserif"})

labels = ["Full\n(10 steps)", "NFE 6\n(adopted)", "NFE 5", "NFE 4"]
lat = [412, 248, 207, 166]
dev = [0.0, 4.2, 8.4, 12.4]
x = np.arange(4)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
colors = ["#9ca3af", "#1f4e79", "#5b86b3", "#9bb4d1"]
bars = ax.bar(x, lat, 0.55, color=colors, edgecolor="black", lw=1)
for xi, v in zip(x, lat):
    ax.text(xi, v + 6, f"{v} ms", ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Flow latency (ms)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 470); ax.spines[["top"]].set_visible(False)
ax.set_title("Adaptive-step flow: skip redundant middle ODE steps", fontsize=12.5, pad=10)

ax2 = ax.twinx()
ax2.plot(x, dev, "o--", color="#c0392b", lw=1.8, ms=7, label="trajectory deviation")
for xi, d in zip(x, dev):
    if d > 0:
        ax2.text(xi + 0.06, d + 0.5, f"{d} cm", color="#a02", fontsize=10, va="bottom")
ax2.set_ylabel("Trajectory deviation (cm)", color="#a02")
ax2.tick_params(axis="y", colors="#a02")
ax2.set_ylim(0, 26); ax2.spines[["top"]].set_visible(False)

ax.annotate("$-$41% flow at ~4 cm\n(< 1% of a lane width)", xy=(1, 248), xytext=(1.7, 360),
            fontsize=10.5, color="#0f6e56", ha="center",
            arrowprops=dict(arrowstyle="->", color="#0f6e56", lw=1.3))
fig.savefig("260614_fig11_adaptive_flow.png", dpi=200, bbox_inches="tight")
print("saved 260614_fig11_adaptive_flow.png")
