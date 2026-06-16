# -*- coding: utf-8 -*-
"""Why a TEMPORAL draft wins (2b). Left: speedup by draft source — the previous
frame beats a fixed template and crushes prompt-lookup (~nothing, echoing
MMSpec). Right, at a glance: if the reasoning is unchanged we reuse almost all
of it; once it changes we can't. 30 frames, Thor, every source bit-identical."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 12, "mathtext.fontset": "dejavuserif"})
GREEN, BLUE, RED, INK = "#1f7a52", "#3b6ea5", "#c0392b", "#222"

# columns from 260615_draftsrc.csv: cot_edit, fwd_none(=N-1), fwd_t, fwd_s, fwd_p, acc_t
rows = [
 (8,16,16,16,16,0),(1,16,1,16,16,15),(2,16,2,16,16,10),(0,16,1,16,16,16),(0,16,1,16,16,16),
 (0,16,1,1,16,16),(0,16,1,1,16,16),(0,16,1,1,16,16),(0,16,1,1,16,16),(0,16,1,1,16,16),
 (1,13,1,1,13,12),(1,13,1,1,13,12),(1,13,1,1,13,12),(1,13,1,1,13,12),(3,14,6,6,14,8),
 (1,13,1,1,13,12),(1,13,1,1,13,12),(3,14,5,5,14,9),(6,10,4,4,10,6),(1,10,1,4,10,9),
 (1,9,1,1,9,8),(1,9,1,1,9,8),(1,9,1,1,9,8),(1,9,1,1,9,8),(16,19,17,17,19,0),
 (1,16,1,1,16,15),(20,24,24,24,23,0),(0,24,1,24,23,24),(2,24,2,24,23,12),(34,40,40,38,38,0),
]
edit = np.array([r[0] for r in rows], float)
fn = np.array([r[1] for r in rows], float)
ft = np.array([r[2] for r in rows], float)
fs = np.array([r[3] for r in rows], float)
fp = np.array([r[4] for r in rows], float)
acc = np.array([r[5] for r in rows], float)
frac = acc / np.maximum(fn, 1)          # fraction of the sentence reused
sp_t = np.mean(fn / np.maximum(ft, 1)); sp_s = np.mean(fn / np.maximum(fs, 1))
sp_p = np.mean(fn / np.maximum(fp, 1)); sp_n = np.mean(fn / np.maximum(fn, 1))  # none = 1.0 by def

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.4),
                               gridspec_kw={"width_ratios": [1.12, 1.0]})

# ---- (a) speedup by source (all 4 tested sources, incl. baseline) ----
labels = ["temporal\n(prev frame)", "static\n(fixed templ.)", "prompt-lookup\n(context)",
          "none\n(baseline)"]
GRAY = "#9ca3af"
vals = [sp_t, sp_s, sp_p, sp_n]; cols = [GREEN, BLUE, RED, GRAY]; xb = np.arange(4)
axA.bar(xb, vals, 0.66, color=cols, edgecolor="black", lw=1)
for x, v in zip(xb, vals):
    axA.text(x, v + 0.25, f"{v:.2f}×" if v < 2 else f"{v:.1f}×", ha="center",
             fontsize=14, fontweight="bold", color=cols[x] if x != 3 else "#555")
axA.axhline(1.0, color="#bbb", lw=1, ls=(0, (4, 3)), zorder=0)
axA.annotate("prompt-lookup ≈ doing nothing\n(matches MMSpec)", xy=(2, 1.0), xytext=(2.5, 4.3),
             fontsize=9.5, color=RED, ha="center",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
axA.set_xticks(xb); axA.set_xticklabels(labels, fontsize=10)
axA.set_ylabel("decode speedup  (forwards vs greedy)")
axA.set_ylim(0, 13); axA.set_title("(a) The previous frame is the best free draft", fontsize=12, pad=8)
axA.spines[["top", "right"]].set_visible(False)

# ---- (b) at a glance: reuse collapses once the reasoning changes ----
bins = [("same\n(0 changed)", edit == 0),
        ("minor\n(1–3 changed)", (edit >= 1) & (edit <= 3)),
        ("major\n(≥4 changed)", edit >= 4)]
pcts = [100 * frac[mask].mean() for _, mask in bins]
bcolors = [GREEN, "#7fb89e", "#d9b3b3"]
xb2 = np.arange(3)
axB.bar(xb2, pcts, 0.6, color=bcolors, edgecolor="black", lw=1)
for x, p in zip(xb2, pcts):
    axB.text(x, p + 2.5, f"{p:.0f}%", ha="center", fontsize=15, fontweight="bold",
             color=INK if x < 2 else RED)
axB.annotate("", xy=(2.05, 18), xytext=(0.0, 104),
             arrowprops=dict(arrowstyle="->", color="#999", lw=2, ls="--"))
axB.text(1.0, 70, "more change\n→ less reuse", fontsize=10.5, color="#666", ha="center", style="italic")
axB.set_xticks(xb2); axB.set_xticklabels([b[0] for b in bins], fontsize=10.5)
axB.set_ylabel("share of the sentence reused  (%)")
axB.set_ylim(0, 118); axB.set_yticks([0, 50, 100])
axB.set_title("(b) Reasoning unchanged → reuse it; changed → can't", fontsize=12, pad=8)
axB.spines[["top", "right"]].set_visible(False)

fig.suptitle("A temporal draft is the right free draft for a 10 Hz VLA "
             "(30 frames, output bit-identical)", fontsize=12.5, y=1.02)
fig.savefig("260616_fig16_draft_source.png", dpi=200, bbox_inches="tight")
print("saved 260616_fig16_draft_source.png",
      f"| temporal {sp_t:.2f} static {sp_s:.2f} pld {sp_p:.2f} | bins {[round(p) for p in pcts]}")
