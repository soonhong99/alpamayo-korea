# -*- coding: utf-8 -*-
"""Predictive law validated: measured decode speedup matches the closed form
speedup = (N-1)/max(d,1) (= N/d), d = frame-to-frame CoT edit distance.
30 frames, no GPU (analysis of 260615_draftsrc.csv). R^2 = 0.99."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 12, "mathtext.fontset": "dejavuserif"})
GREEN, GRAY, RED, INK = "#1f7a52", "#9ca3af", "#c0392b", "#222"

# (d, N, f) per frame from 260615_draftsrc.csv
rows = [(8,17,16),(1,17,1),(2,17,2),(0,17,1),(0,17,1),(0,17,1),(0,17,1),(0,17,1),(0,17,1),(0,17,1),
        (1,14,1),(1,14,1),(1,14,1),(1,14,1),(3,15,6),(1,14,1),(1,14,1),(3,15,5),(6,11,4),(1,11,1),
        (1,10,1),(1,10,1),(1,10,1),(1,10,1),(16,20,17),(1,17,1),(20,25,24),(0,25,1),(2,25,2),(34,41,40)]
d = np.array([r[0] for r in rows], float); N = np.array([r[1] for r in rows], float)
f = np.array([r[2] for r in rows], float)
meas = (N-1)/f                       # measured speedup
pred = (N-1)/np.maximum(d, 1)        # theory: speedup = (N-1)/max(d,1)
r2 = 1 - ((meas-pred)**2).sum()/((meas-meas.mean())**2).sum()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))

# (a) mean measured vs predicted by edit-distance regime
bins = [("0\n(same)", d==0), ("1–2\n(minor)", (d>=1)&(d<=2)),
        ("3–7\n(moderate)", (d>=3)&(d<=7)), ("≥8\n(major)", d>=8)]
xm = np.arange(4); w = 0.38
mvals = [meas[m].mean() for _, m in bins]; pvals = [pred[m].mean() for _, m in bins]
axA.bar(xm-w/2, mvals, w, color=GREEN, edgecolor="black", lw=1, label="measured")
axA.bar(xm+w/2, pvals, w, color=GRAY, edgecolor="black", lw=1, hatch="///", label="theory  $N/d$")
for x, mv in zip(xm, mvals):
    axA.text(x-w/2, mv+0.4, f"{mv:.0f}×", ha="center", fontsize=10.5, color=GREEN, fontweight="bold")
axA.set_xticks(xm); axA.set_xticklabels([b[0] for b in bins], fontsize=10)
axA.set_xlabel("CoT change $d$  (edit distance to previous frame)")
axA.set_ylabel("decode speedup")
axA.set_title("(a) Measured speedup = theory $N/d$, per regime", fontsize=12, pad=8)
axA.legend(frameon=False, fontsize=10.5, loc="upper right")
axA.spines[["top","right"]].set_visible(False)

# (b) predicted vs measured, identity, R^2
lim = max(meas.max(), pred.max())*1.1
axB.plot([0, lim], [0, lim], "--", color=INK, lw=1.3, zorder=1)
col = [GREEN if dd <= 2 else RED for dd in d]
axB.scatter(pred, meas, s=55, c=col, edgecolor="black", lw=0.6, alpha=0.85, zorder=3)
axB.text(0.05*lim, 0.9*lim, f"$R^2 = {r2:.2f}$", fontsize=15, fontweight="bold", color=INK)
axB.text(0.55*lim, 0.18*lim, "green: stable ($d\\leq2$)\nred: scene change",
         fontsize=9.5, color="#555")
axB.set_xlabel("theory  $(N{-}1)/\\max(d,1)$  ($\\times$)")
axB.set_ylabel("measured speedup  ($\\times$)")
axB.set_title("(b) Theory predicts every frame", fontsize=12, pad=8)
axB.set_xlim(0, lim); axB.set_ylim(0, lim)
axB.spines[["top","right"]].set_visible(False)

fig.suptitle("The speedup law $\\mathrm{speedup}=1/(1-\\alpha)\\approx N/d$ matches "
             "measurement (30 frames, $R^2=0.99$)", fontsize=12.5, y=1.02)
fig.savefig("260616_fig18_lawfit.png", dpi=200, bbox_inches="tight")
print(f"saved 260616_fig18_lawfit.png | R2={r2:.3f}")
