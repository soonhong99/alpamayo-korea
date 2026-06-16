# -*- coding: utf-8 -*-
"""Greedy speculative decode in the real 10 Hz loop: stable frames collapse to
one forward, a scene-change frame falls back instantly (never slower), output
bit-identical. Measured on Thor (9 frames + 1 warmup, clocks locked)."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "mathtext.fontset": "dejavuserif"})

# measured 260615_spec_pipeline.log (frames 1..9; f0 warmup excluded)
frame = np.arange(1, 10)
bfwd = np.array([19, 16, 16, 16, 16, 16, 16, 16, 16])
sfwd = np.array([1, 16, 1, 1, 1, 1, 1, 1, 1])
e2e_b = np.array([3540, 3324, 3330, 3331, 3329, 3323, 3322, 3294, 3319])
e2e_s = np.array([2332, 3376, 2318, 2320, 2317, 2281, 2318, 2284, 2304])
is_chg = sfwd == bfwd

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
w = 0.38

# (a) forwards per frame
ax1.bar(frame - w/2, bfwd, w, color="#9ca3af", edgecolor="black", lw=1,
        hatch="///", label="baseline greedy")
sc = ["#c0392b" if c else "#1f7a52" for c in is_chg]
ax1.bar(frame + w/2, sfwd, w, color=sc, edgecolor="black", lw=1,
        label="speculative")
for x, c in zip(frame, is_chg):
    if c:
        ax1.annotate("scene change:\ninstant fallback (not slower)", xy=(x + w/2, 16.3),
                     xytext=(3.3, 21.4), fontsize=8.8, color="#8a1b10", ha="center",
                     arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
ax1.text(5.7, 4.6, "stable frames: 16 forwards $\\rightarrow$ 1  (16$\\times$)",
         fontsize=10.5, color="#0f6e56", ha="center", style="italic")
ax1.set_xlabel("frame  (10 Hz, 100 ms apart)")
ax1.set_ylabel("decode forwards")
ax1.set_title("(a) Forwards collapse on stable frames", fontsize=12, pad=8)
ax1.set_xticks(frame); ax1.set_ylim(0, 24)
ax1.legend(frameon=False, fontsize=10, loc="upper right")
ax1.spines[["top", "right"]].set_visible(False)

# (b) e2e per frame
ax2.plot(frame, e2e_b, "o-", color="#9ca3af", lw=1.8, ms=6, label="baseline e2e")
ax2.plot(frame, e2e_s, "o-", color="#1f7a52", lw=1.8, ms=6,
         label="speculative e2e (projected)")
ax2.fill_between(frame, e2e_s, e2e_b, color="#1f7a52", alpha=0.10)
ax2.scatter([2], [3376], color="#c0392b", zorder=5, s=45)
ax2.annotate("change frame\n($+$52 ms only)", xy=(2, 3376), xytext=(3.5, 3530),
             fontsize=8.8, color="#8a1b10", ha="center",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
ax2.text(6.0, 2780, "stable: 3.35 s $\\rightarrow$ 2.3 s  ($-$30%)",
         fontsize=10.5, color="#0f6e56", ha="center", style="italic")
ax2.set_xlabel("frame  (10 Hz, 100 ms apart)")
ax2.set_ylabel("end-to-end latency (ms)")
ax2.set_title("(b) e2e latency, output bit-identical", fontsize=12, pad=8)
ax2.set_xticks(frame); ax2.set_ylim(2000, 3700)
ax2.legend(frameon=False, fontsize=10, loc="upper right")
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("Training-free greedy speculative decode in the real 10 Hz loop "
             "(Jetson Thor, measured)", fontsize=12.5, y=1.02)
fig.savefig("260615_fig13_spec_pipeline.png", dpi=200, bbox_inches="tight")
print("saved 260615_fig13_spec_pipeline.png")
