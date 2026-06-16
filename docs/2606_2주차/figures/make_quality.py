# -*- coding: utf-8 -*-
"""Greedy vs sampled trajectory quality (minADE6 vs GT) over 40 clips, paired.
No statistically significant difference (Wilcoxon p>0.05); greedy trends better,
more so in the long-tail (dynamic) half. Measured on Thor."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "mathtext.fontset": "dejavuserif"})

# (lat, dsp, greedy, sampled) from 260615_largeN.csv (n=40)
data = [
    (10.428,0.131,0.4604,0.4522),(3.113,0.029,0.8390,0.8390),(1.132,0.134,0.0241,0.0241),
    (11.422,0.161,0.3135,0.7290),(54.452,0.861,4.4537,5.8363),(0.571,0.038,0.2591,0.2591),
    (9.262,0.039,0.5306,0.5306),(13.018,0.069,0.2625,0.2866),(0.401,0.145,0.9248,0.9248),
    (0.418,0.771,2.6637,2.6637),(0.760,0.413,0.4269,0.4269),(26.051,1.384,0.5117,1.1782),
    (0.667,0.131,1.2833,1.5073),(3.223,0.488,0.6142,0.6142),(1.811,0.455,0.6378,0.5453),
    (2.577,0.520,0.3493,0.3541),(0.326,0.067,0.1377,0.1690),(0.014,0.158,2.7170,2.7170),
    (20.414,0.460,1.4632,0.9030),(6.388,0.067,0.9369,0.8874),(0.000,0.026,0.0195,0.0195),
    (0.699,0.189,1.7800,1.3369),(36.083,0.258,0.3538,1.3340),(1.035,0.532,0.4878,0.4878),
    (21.179,0.382,1.1211,1.1211),(44.274,0.897,1.1259,1.6266),(2.025,0.102,0.5127,0.5127),
    (0.994,0.046,0.2348,0.2348),(0.481,0.851,0.5331,0.3478),(0.125,0.350,0.6869,0.5232),
    (1.040,0.022,0.1735,0.1735),(1.122,0.082,0.3134,0.3134),(0.629,0.324,0.5053,0.5693),
    (34.008,0.160,0.9761,0.9761),(34.721,0.455,0.5193,0.3496),(23.035,0.020,1.0908,0.8519),
    (15.090,0.118,0.9869,1.2173),(0.422,0.139,0.7318,0.7318),(7.660,0.346,1.3320,2.2541),
    (12.243,0.022,0.2028,0.2028),
]
g = np.array([r[2] for r in data]); s = np.array([r[3] for r in data])
dynsc = np.array([r[0] + r[1] for r in data])
med = np.median(dynsc); dyn = dynsc >= med
d = g - s  # greedy - sampled (negative => greedy better)


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.5),
                               gridspec_kw={"width_ratios": [1, 1.25]})

# ---- (a) grouped means with SE, all vs long-tail ----
groups = [("All\n(n=40)", g, s), ("Long-tail half\n(n=20)", g[dyn], s[dyn])]
x = np.arange(2); w = 0.34
for i, (lab, gg, ss) in enumerate(groups):
    axA.bar(x[i] - w/2, gg.mean(), w, yerr=se(gg), capsize=4, color="#1f7a52",
            edgecolor="black", lw=1, label="greedy" if i == 0 else None)
    axA.bar(x[i] + w/2, ss.mean(), w, yerr=se(ss), capsize=4, color="#9ca3af",
            edgecolor="black", lw=1, hatch="///", label="sampled" if i == 0 else None)
    top = max(gg.mean() + se(gg), ss.mean() + se(ss)) + 0.12
    axA.plot([x[i]-w/2, x[i]-w/2, x[i]+w/2, x[i]+w/2],
             [top, top+0.04, top+0.04, top], color="#444", lw=1)
    p = "p=0.31" if i == 0 else "p=0.12"
    axA.text(x[i], top + 0.07, f"n.s. ({p})", ha="center", fontsize=9.5, color="#444")
axA.set_xticks(x); axA.set_xticklabels([gname for gname, _, _ in groups])
axA.set_ylabel("mean minADE6 vs GT (m)   (lower = better)")
axA.set_ylim(0, 1.75)
axA.set_title("(a) Greedy is no worse than sampled", fontsize=12, pad=8)
axA.legend(frameon=False, fontsize=10.5, loc="upper right")
axA.spines[["top", "right"]].set_visible(False)

# ---- (b) paired per-clip difference, sorted ----
order = np.argsort(d)
dd = d[order]; dyo = dyn[order]
xb = np.arange(len(dd))
cols = []
for v in dd:
    if v < -0.02:
        cols.append("#1f7a52")      # greedy better
    elif v > 0.02:
        cols.append("#c0392b")      # sampled better
    else:
        cols.append("#b8c0c8")      # tie
axB.bar(xb, dd, 0.8, color=cols, edgecolor="none")
axB.axhline(0, color="#333", lw=1)
mean_d = d.mean(); ci = 1.96 * se(d)
axB.axhline(mean_d, color="#0f3d2e", lw=1.4, ls="--")
axB.fill_between([-1, len(dd)], mean_d - ci, mean_d + ci, color="#1f7a52", alpha=0.12)
axB.set_xlim(-1, len(dd))
axB.text(1.5, mean_d - 0.18, f"mean $-$0.09 m\n95% CI [$-$0.20, $+$0.02]",
         fontsize=9.5, color="#0f3d2e", va="top")
axB.set_ylabel("greedy $-$ sampled minADE6 (m)")
axB.set_xlabel("clips, sorted by paired difference")
axB.set_title("(b) Per-clip paired difference (negative = greedy better)",
              fontsize=12, pad=8)
axB.set_ylim(-1.05, 0.65)
axB.spines[["top", "right"]].set_visible(False)
leg = [Patch(fc="#1f7a52", label="greedy better (11)"),
       Patch(fc="#b8c0c8", label="tie within 0.02 (21)"),
       Patch(fc="#c0392b", label="sampled better (8)")]
axB.legend(handles=leg, frameon=False, fontsize=9.5, loc="lower right")

fig.suptitle("Trajectory quality: training-free greedy CoT vs stochastic sampling "
             "(40 clips, minADE6 vs ground truth)", fontsize=12.5, y=1.02)
fig.savefig("260615_fig14_quality.png", dpi=200, bbox_inches="tight")
print("saved 260615_fig14_quality.png")
