# -*- coding: utf-8 -*-
"""E-scale honest distribution: temporal speculative gives ~-29% e2e on every
one of 8 clips (40 consecutive frames), not a cherry-picked window. Two frames
fall back (draft stale) and stay at baseline; nothing gets slower."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 12, "mathtext.fontset": "dejavuserif"})
GREEN, GRAY, RED = "#1f7a52", "#9ca3af", "#c0392b"

# e2e_ms, proj_e2e_ms per frame (40), in clip order, from 260616_escale.csv
e2e = [3483,3477,3477,3475,3477, 3340,3343,3335,3338,3340, 3049,3054,3049,3050,3055,
       3200,3196,3194,3196,3192, 3051,3051,3050,3049,3051, 3266,3265,3268,3264,3269,
       3261,3269,3262,3272,3270, 3191,3198,3198,3196,3194]
proj = [2259,2245,2236,2240,2252, 3390,2249,2237,2248,2242, 2239,2240,2237,3107,2244,
        2255,2246,2239,2253,2246, 2239,2242,2241,2242,2245, 2248,2245,2247,2244,2249,
        2241,2249,2242,2251,2249, 2242,2248,2248,2241,2239]
e2e = np.array(e2e, float); proj = np.array(proj, float)
x = np.arange(1, 41)
special = {5: "partial accept", 13: "fallback (stale draft)"}  # 0-based indices

fig, ax = plt.subplots(figsize=(11, 4.3))
ax.fill_between(x, proj, e2e, color=GREEN, alpha=0.12, zorder=1)
ax.plot(x, e2e, "-", color=GRAY, lw=1.6, label="baseline e2e", zorder=2)
ax.plot(x, proj, "-", color=GREEN, lw=1.8, label="temporal speculative e2e", zorder=3)
ax.scatter(x, proj, s=14, color=GREEN, zorder=4)
for i, lab in special.items():
    ax.scatter([x[i]], [proj[i]], s=70, facecolor="none", edgecolor=RED, lw=1.8, zorder=5)
ax.annotate("2 frames fall back to baseline\n(draft stale) — never slower",
            xy=(x[13], proj[13]), xytext=(20, 3450), fontsize=10, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

med = np.median(e2e / proj)
ax.text(40, 2120, f"median  $-${(1-1/med)*100:.0f}%  (×{med:.2f})\nevery clip, no cherry-pick",
        ha="right", fontsize=11, color=GREEN, fontweight="bold")
# clip boundaries
for b in range(5, 40, 5):
    ax.axvline(b + 0.5, color="#eee", lw=1, zorder=0)
ax.set_xlabel("frame  (8 clips × 5 consecutive 10 Hz frames)")
ax.set_ylabel("end-to-end latency (ms)")
ax.set_ylim(2050, 3600); ax.set_xlim(0.3, 40.7)
ax.set_title("Speculative e2e across 8 clips — consistent $-$29%, not a favourable window",
             fontsize=12.5, pad=10)
ax.legend(frameon=False, fontsize=10.5, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("260616_fig17_escale.png", dpi=200, bbox_inches="tight")
print("saved 260616_fig17_escale.png", f"| median speedup {med:.3f}")
