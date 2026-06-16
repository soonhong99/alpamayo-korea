# -*- coding: utf-8 -*-
"""Delta-t sensitivity (analysis only; deployment = 100 ms). Acceptance alpha is
uniformly high at 100 ms and decays as the interval grows, because the chance of
a large scene change accumulates. This shows WHY 100 ms is the right operating
point -- not a proposal to reuse old context. 8 clips, Thor, bit-identical."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 12, "mathtext.fontset": "dejavuserif"})
GREEN, GRAY, RED, INK = "#1f7a52", "#b9c0c8", "#c0392b", "#222"

dt = [100, 200, 300, 500, 1000]
# alpha per clip across dt (260617_dtsweep.csv)
clips = {
 "a": [.933, 0, 0, 0, 0], "b": [.929, .929, .929, .929, .929],
 "c": [.900, .900, .900, .900, .400], "d": [.900, .900, .900, .900, .900],
 "e": [.938, .938, .938, .938, .938], "f": [.938, .938, .938, .938, .938],
 "g": [.900, .900, .900, .900, 0], "h": [.938, .938, .938, .938, .938],
}
A = np.array(list(clips.values()))
mean = A.mean(0)
x = np.arange(len(dt))

fig, ax = plt.subplots(figsize=(8.4, 4.6))
# per-clip faint lines (show the spread / accumulating risk)
for row in A:
    ax.plot(x, row, "-", color=GRAY, lw=1.2, alpha=0.7, zorder=1)
ax.plot(x, mean, "o-", color=GREEN, lw=2.6, ms=8, zorder=3, label="mean acceptance $\\alpha$")
for xi, m in zip(x, mean):
    ax.text(xi, m + 0.03, f"{m:.2f}", ha="center", fontsize=11, color=GREEN, fontweight="bold")

# highlight 100 ms operating point
ax.axvspan(-0.25, 0.25, color=GREEN, alpha=0.08, zorder=0)
ax.annotate("deployment: 10 Hz / 100 ms\n(NVIDIA-fixed) — $\\alpha=0.92$",
            xy=(0, 0.922), xytext=(1.15, 0.985), fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
ax.text(3.5, 0.18, "larger $\\Delta t$: some scenes change →\nacceptance collapses (fallback)",
        fontsize=10, color=RED, ha="center", style="italic")

ax.set_xticks(x); ax.set_xticklabels([f"{d}" for d in dt])
ax.set_xlabel("inter-frame interval  $\\Delta t$  (ms)   — analysis only, not a deployment knob")
ax.set_ylabel("temporal-draft acceptance  $\\alpha$")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Acceptance is highest at 100 ms and decays with $\\Delta t$  "
             "($\\mathbb{E}[d]\\!\\uparrow$ as $\\Delta t\\!\\uparrow$)", fontsize=12.5, pad=10)
ax.legend(frameon=False, fontsize=10.5, loc="center right")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("260617_fig19_dtsweep.png", dpi=200, bbox_inches="tight")
print("saved 260617_fig19_dtsweep.png | mean", [round(m, 3) for m in mean])
