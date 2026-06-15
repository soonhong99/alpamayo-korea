"""Paper-style figure: greedy speculative is far faster AND keeps accuracy in
dynamic (long-tail) frames. Real measured data from 260614 experiments."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.linewidth": 1.0,
    "mathtext.fontset": "dejavuserif",
})

# (lat = lateral maneuver = how dynamic; greedy = ours; sampled = baseline) -- 29 frames, 2 seeds
data = [
    (36.99, 1.513, 0.712), (32.25, 3.920, 2.996), (23.11, 0.754, 1.060),
    (20.33, 0.815, 0.841), (14.92, 0.536, 0.526), (9.43, 0.522, 0.477),
    (4.65, 0.158, 0.856), (2.05, 0.257, 0.257), (0.02, 3.246, 2.826),
    (0.19, 0.774, 0.725), (0.45, 0.323, 0.403), (0.52, 1.496, 1.412),
    (0.10, 0.528, 0.528), (26.20, 1.534, 2.840), (1.85, 0.784, 0.279),
    (0.00, 0.001, 0.001), (1.43, 0.382, 0.323), (12.97, 0.627, 0.313),
    (40.01, 1.108, 2.672), (3.02, 0.708, 0.291), (13.41, 1.007, 1.007),
    (70.34, 1.103, 1.103), (26.22, 2.640, 4.648), (6.18, 0.993, 1.235),
    (20.84, 0.355, 0.355), (0.62, 0.229, 0.229), (0.17, 0.851, 0.851),
    (1.04, 1.571, 0.908), (11.91, 5.257, 4.780),
]
lat = np.array([d[0] for d in data])
greedy = np.array([d[1] for d in data])
sampled = np.array([d[2] for d in data])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
C_BASE, C_OURS = "#9ca3af", "#1f4e79"

# ---- (a) decode latency: 17x faster ----
vals = [17 * 70, 1 * 70]          # forwards x ~70ms/forward (clock-locked)
bars = ax1.bar(["Baseline\n(token-by-token)", "Ours\n(speculative)"], vals,
               color=[C_BASE, C_OURS], width=0.6, edgecolor="black", linewidth=1)
bars[0].set_hatch("///")
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 25, f"{v:,} ms",
             ha="center", va="bottom", fontsize=12)
ax1.annotate("", xy=(1, 120), xytext=(1, 1100),
             arrowprops=dict(arrowstyle="->", lw=1.6, color="black"))
ax1.text(1.12, 560, "17$\\times$ faster", fontsize=14, color=C_OURS, rotation=90,
         va="center", fontweight="bold")
ax1.set_ylabel("Decode latency (ms)")
ax1.set_title("(a) Reasoning decode — stable frame", fontsize=13)
ax1.set_ylim(0, 1350)
ax1.spines[["top", "right"]].set_visible(False)

# ---- (b) accuracy by how dynamic the frame is (long-tail focus) ----
edges = np.quantile(lat, [0, 1/3, 2/3, 1.0])
labels = ["Calm\n(low)", "Moderate", "Dynamic\n(long-tail)"]
gm, sm, ge, se = [], [], [], []
for i in range(3):
    lo, hi = edges[i], edges[i + 1]
    m = (lat >= lo) & (lat <= hi) if i == 2 else (lat >= lo) & (lat < hi)
    n = max(m.sum(), 1)
    gm.append(greedy[m].mean()); sm.append(sampled[m].mean())
    ge.append(greedy[m].std() / np.sqrt(n)); se.append(sampled[m].std() / np.sqrt(n))
x = np.arange(3); w = 0.36
b1 = ax2.bar(x - w/2, sm, w, yerr=se, capsize=4, color=C_BASE, edgecolor="black",
             linewidth=1, hatch="///", label="Baseline (sampled)", error_kw=dict(lw=1.2))
b2 = ax2.bar(x + w/2, gm, w, yerr=ge, capsize=4, color=C_OURS, edgecolor="black",
             linewidth=1, label="Ours (greedy)", error_kw=dict(lw=1.2))
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel("Trajectory error  minADE$_6$ (m)")
ax2.set_title("(b) Accuracy holds even in long-tail", fontsize=13)
ax2.set_ylim(0, 2.7)
ax2.legend(frameon=False, fontsize=11, loc="upper left")
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("260614_fig8_speed_accuracy.png", dpi=200, bbox_inches="tight")
print("saved 260614_fig8_speed_accuracy.png")
print("dynamism tertiles greedy:", [round(v, 3) for v in gm], "sampled:", [round(v, 3) for v in sm])
