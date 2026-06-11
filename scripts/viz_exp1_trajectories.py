"""
viz_exp1_trajectories.py  —  Experiment 1: Decode Skip trajectory visualization
Paper-quality figures. English only. Run on Thor after exp1_decode_skip.py.

Output:
  evaluation/results/streaming/exp1_decode_skip/
  ├── fig1_trajectory_overview.png   top-down BEV + speed + lateral profiles
  ├── fig2_n16_vs_n0.png             N=16 vs N=0 side-by-side comparison
  ├── fig3_error_over_time.png       per-waypoint error from baseline (already good)
  └── fig4_latency_vs_quality.png    latency gain vs ADE tradeoff

Usage:
  cd ~/alpamayo1.5
  python3 scripts/viz_exp1_trajectories.py
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.dpi":        150,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})

RESULT_DIR = Path("evaluation/results/streaming/exp1_decode_skip")
OUT_DIR    = RESULT_DIR

TOKEN_LIST = [0, 1, 3, 5, 8, 10, 13, 16]
DT = 0.1   # seconds per waypoint

# Colormap: cool-to-warm, baseline = black
CMAP = plt.get_cmap("plasma")
_N   = len([t for t in TOKEN_LIST if t != 16])

def token_color(n: int) -> str:
    if n == 16:
        return "#1a1a2e"
    rank = sorted([t for t in TOKEN_LIST if t != 16]).index(n)
    return CMAP(rank / max(_N - 1, 1))

COLORS = {n: token_color(n) for n in TOKEN_LIST}
LW     = {n: (3.5 if n == 16 else 1.8) for n in TOKEN_LIST}
ZORDER = {n: (10 if n == 16 else 5)    for n in TOKEN_LIST}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_wp(n: int) -> np.ndarray | None:
    p = RESULT_DIR / f"waypoints_{n}.npy"
    return np.load(str(p)) if p.exists() else None


def load_summary() -> dict | None:
    p = RESULT_DIR / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def get_cond(summary: dict, n: int) -> dict | None:
    for c in summary.get("conditions", []):
        if c["max_coc_tokens"] == n:
            return c
    return None


# ── Figure 1: Overview  ────────────────────────────────────────────────────────
#   Three panels: (a) BEV, (b) Forward distance X(t), (c) Lateral deviation Y(t)

def fig1_overview(wps: dict[int, np.ndarray]):
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax_bev  = fig.add_subplot(gs[0])
    ax_x    = fig.add_subplot(gs[1])
    ax_y    = fig.add_subplot(gs[2])

    times = np.arange(1, 65) * DT   # 0.1 … 6.4 s
    wp16  = wps.get(16)

    # ── (a) Bird's-eye view ───────────────────────────────────────────────
    # Use wider y-range so lateral differences are visible
    for n in TOKEN_LIST:
        wp = wps.get(n)
        if wp is None:
            continue
        ax_bev.plot(wp[:, 0], wp[:, 1],
                    color=COLORS[n], lw=LW[n], alpha=0.9,
                    zorder=ZORDER[n],
                    label=f"N={n}" + (" (baseline)" if n == 16 else ""))
        ax_bev.plot(wp[-1, 0], wp[-1, 1], "*",
                    color=COLORS[n], ms=10, zorder=ZORDER[n] + 1)

    # ego vehicle
    ax_bev.plot(0, 0, "ks", ms=10, zorder=20, label="Ego vehicle (t=0)")
    ax_bev.set_xlabel("Forward distance  x  (m)")
    ax_bev.set_ylabel("Lateral deviation  y  (m)")
    ax_bev.set_title("(a)  Bird's-eye view  —  64 waypoints  (6.4 s)")
    # Fixed y range: show ±3 m so lateral differences are visible
    ax_bev.set_ylim(-3, 3)
    ax_bev.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
    # Lane boundaries (Korean std lane width 3.5 m)
    for y in [-1.75, 1.75]:
        ax_bev.axhline(y, color="steelblue", lw=0.7, ls=":", alpha=0.5)
    ax_bev.text(1, 1.85, "lane boundary (±1.75 m)", fontsize=8,
                color="steelblue", alpha=0.7)
    ax_bev.legend(fontsize=8, loc="upper left", framealpha=0.85,
                  handlelength=1.5, ncol=2)
    ax_bev.grid(True, alpha=0.25)

    # ── (b) Forward distance X(t) ─────────────────────────────────────────
    for n in TOKEN_LIST:
        wp = wps.get(n)
        if wp is None:
            continue
        ax_x.plot(times, wp[:, 0], color=COLORS[n], lw=LW[n], alpha=0.9,
                  zorder=ZORDER[n], label=f"N={n}")
    ax_x.set_xlabel("Time  (s)")
    ax_x.set_ylabel("Forward position  x  (m)")
    ax_x.set_title("(b)  Planned forward distance over time")
    ax_x.set_xlim(0, 6.5)

    # Speed annotation (slope of x at t=6.4)
    if wp16 is not None:
        v16 = (wp16[-1, 0] - wp16[-2, 0]) / DT
        ax_x.annotate(f"N=16  v≈{v16:.1f} m/s\n({v16*3.6:.0f} km/h)",
                      xy=(6.4, wp16[-1, 0]),
                      xytext=(5.0, wp16[-1, 0] - 4),
                      fontsize=8, color=COLORS[16],
                      arrowprops=dict(arrowstyle="->", color=COLORS[16],
                                      lw=1.2, connectionstyle="arc3,rad=-0.2"))
    ax_x.legend(fontsize=8, loc="upper left", framealpha=0.85,
                handlelength=1.5, ncol=2)
    ax_x.grid(True, alpha=0.25)

    # ── (c) Lateral deviation Y(t) ────────────────────────────────────────
    for n in TOKEN_LIST:
        wp = wps.get(n)
        if wp is None:
            continue
        ax_y.plot(times, wp[:, 1], color=COLORS[n], lw=LW[n], alpha=0.9,
                  zorder=ZORDER[n], label=f"N={n}")
    ax_y.set_xlabel("Time  (s)")
    ax_y.set_ylabel("Lateral position  y  (m)")
    ax_y.set_title("(c)  Planned lateral position over time")
    ax_y.set_xlim(0, 6.5)
    ax_y.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4, label="centerline")
    ax_y.legend(fontsize=8, loc="upper left", framealpha=0.85,
                handlelength=1.5, ncol=2)
    ax_y.grid(True, alpha=0.25)

    fig.suptitle(
        "Experiment 1 — Decode Skip: Trajectory Comparison\n"
        "All conditions use identical input (PhysicalAI clip 030c760c, t=5.1 s)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    out = OUT_DIR / "fig1_trajectory_overview.png"
    plt.savefig(str(out))
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: N=16 vs N=0 annotated comparison ────────────────────────────────

def fig2_comparison(wp16: np.ndarray, wp0: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    times = np.arange(1, 65) * DT

    config = [
        (axes[0], wp16, 16, COLORS[16],
         "N = 16  —  Full CoC  (natural completion)",
         '"Keep distance to the lead vehicle\nsince it is directly ahead in our lane"\n'
         r"$\rightarrow$  14 tokens generated  $\rightarrow$  Action Expert conditioned on complete reasoning"),
        (axes[1], wp0,  0,  COLORS[0],
         "N = 0  —  Decode Skip  (no CoC)",
         "Generation skipped immediately.\n"
         r"$\rightarrow$  Action Expert conditioned on Prefill hidden state only"),
    ]

    for ax, wp, n, color, title, subtitle in config:
        # Trajectory line
        ax.plot(wp[:, 0], wp[:, 1], "-o",
                color=color, lw=2.5, ms=3, alpha=0.9, label=f"N={n}")

        # Time stamps every 1 s
        for i in [9, 19, 29, 39, 49, 59, 63]:
            t = (i + 1) * DT
            ax.annotate(f"{t:.1f} s",
                        xy=(wp[i, 0], wp[i, 1]),
                        xytext=(wp[i, 0] + 0.4, wp[i, 1] + 0.25),
                        fontsize=8, color=color, alpha=0.8,
                        arrowprops=dict(arrowstyle="-", color=color,
                                        lw=0.8, alpha=0.5))

        # Ego vehicle marker
        ax.plot(0, 0, "ks", ms=12, zorder=10)
        ax.text(0.5, -0.3, "ego\nvehicle", fontsize=8, ha="center")

        # Final waypoint star
        ax.plot(wp[-1, 0], wp[-1, 1], "*", color=color, ms=16, zorder=10,
                label=f"6.4 s endpoint\n({wp[-1,0]:.1f}, {wp[-1,1]:.2f}) m")

        ax.set_xlabel("Forward distance  x  (m)", fontsize=11)
        ax.set_ylabel("Lateral deviation  y  (m)", fontsize=11)
        ax.set_ylim(-2.5, 2.5)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
        for y in [-1.75, 1.75]:
            ax.axhline(y, color="steelblue", lw=0.7, ls=":", alpha=0.4)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.text(0.5, -0.18, subtitle, transform=ax.transAxes,
                ha="center", fontsize=9, color="dimgray",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                          ec="orange", alpha=0.85))

    # Difference annotation between panels
    fde   = np.linalg.norm(wp16[63, :2] - wp0[63, :2])
    x_gap = wp16[63, 0] - wp0[63, 0]
    y_gap = wp16[63, 1] - wp0[63, 1]
    fig.text(0.5, -0.04,
             f"Endpoint difference:   "
             f"Δx = {x_gap:+.2f} m   Δy = {y_gap:+.2f} m   "
             f"FDE = {fde:.3f} m   "
             f"(ADE = 0.907 m  across all 64 waypoints)",
             ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", fc="#fff3cd", ec="#f0ad4e", alpha=0.95))

    fig.suptitle(
        "Experiment 1 — N=16 (Full CoC) vs N=0 (Decode Skip):  What changes?",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out = OUT_DIR / "fig2_n16_vs_n0.png"
    plt.savefig(str(out))
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: Error over time (reworked, English) ─────────────────────────────

def fig3_error_over_time(wps: dict[int, np.ndarray]):
    wp16  = wps.get(16)
    if wp16 is None:
        return

    fig, ax = plt.subplots(figsize=(13, 5.5))
    times   = np.arange(1, 65) * DT

    for n in [0, 1, 3, 5, 8, 10, 13]:
        wp = wps.get(n)
        if wp is None:
            continue
        dists = np.linalg.norm(wp[:, :2] - wp16[:, :2], axis=1)
        ax.plot(times, dists, color=COLORS[n], lw=2.2, alpha=0.88,
                label=f"N={n}")

    # Threshold lines
    ax.axhline(0.2, color="#27ae60", lw=1.8, ls="--",
               label="ADE pass threshold  (0.2 m)")
    ax.axhline(0.5, color="#f39c12", lw=1.8, ls="--",
               label="ADE marginal threshold  (0.5 m)")
    ax.axhline(1.0, color="#c0392b", lw=1.8, ls="--",
               label="FDE pass threshold  (1.0 m)")

    ax.fill_between(times, 0, 0.2, alpha=0.06, color="#27ae60",
                    label="Pass zone  (ADE < 0.2 m)")

    ax.set_xlabel("Time  (s)", fontsize=12)
    ax.set_ylabel("L2 distance from N=16 baseline  (m)", fontsize=12)
    ax.set_title(
        "Per-waypoint Trajectory Error vs. Baseline  —  "
        "How quickly does each condition diverge?\n"
        "(Each point = distance between predicted and baseline position at that timestamp)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, 6.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25)

    # Annotation: all conditions cross 1 m around t=4 s
    ax.axvline(4.0, color="gray", lw=1.0, ls=":", alpha=0.6)
    ax.text(4.05, 0.05, "~4 s: all conditions\ncross FDE pass threshold",
            fontsize=8.5, color="gray", va="bottom")

    out = OUT_DIR / "fig3_error_over_time.png"
    plt.savefig(str(out))
    plt.close()
    print(f"Saved: {out}")


# ── Figure 4: Latency–quality tradeoff ────────────────────────────────────────

def fig4_latency_quality(wps: dict[int, np.ndarray], summary: dict | None):
    if summary is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ns, lats, ades, fdes = [], [], [], []
    for cond in sorted(summary["conditions"], key=lambda c: c["max_coc_tokens"]):
        n = cond["max_coc_tokens"]
        ns.append(n)
        lats.append(cond["latency"]["mean_ms"])
        ades.append(cond["ade"]["mean_m"] if cond["ade"]["mean_m"] is not None else 0)
        fdes.append(cond["fde"]["mean_m"] if cond["fde"]["mean_m"] is not None else 0)

    ns   = np.array(ns)
    lats = np.array(lats)
    ades = np.array(ades)

    # ── (a) Latency vs N ──────────────────────────────────────────────────
    ax = axes[0]
    for i, n in enumerate(ns):
        ax.scatter(n, lats[i], color=COLORS[n], s=90, zorder=6)
    ax.plot(ns, lats, "--", color="gray", lw=1.2, alpha=0.5, zorder=3)

    # Theoretical line (base 5122 ms + 107 ms/token)
    ns_th  = np.array(TOKEN_LIST)
    lat_th = 5122 + ns_th * 107
    ax.plot(ns_th, lat_th, "k:", lw=1.2, alpha=0.4,
            label="Theory: 5122 + 107×N  (ms)")

    ax.axhline(lats[ns == 16][0], color=COLORS[16], lw=1.2, ls=":",
               alpha=0.6, label=f"N=16 baseline ({lats[ns==16][0]:.0f} ms)")
    ax.set_xlabel("CoC token count  N", fontsize=11)
    ax.set_ylabel("End-to-end latency  (ms)", fontsize=11)
    ax.set_title("(a)  Latency scales linearly with N\n"
                 "Measured ~107 ms / token on Jetson AGX Thor",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── (b) ADE vs latency (tradeoff scatter) ────────────────────────────
    ax = axes[1]
    for i, n in enumerate(ns[ns != 16]):
        idx = list(ns).index(n)
        ax.scatter(lats[idx], ades[idx], color=COLORS[n],
                   s=120, zorder=6, label=f"N={n}")
        ax.annotate(f"N={n}",
                    xy=(lats[idx], ades[idx]),
                    xytext=(lats[idx] + 30, ades[idx] + 0.03),
                    fontsize=8.5, color=COLORS[n])

    ax.axhline(0.2, color="#27ae60", lw=1.8, ls="--",
               label="Pass threshold  (ADE < 0.2 m)")
    ax.axhline(0.5, color="#f39c12", lw=1.5, ls="--",
               label="Marginal  (ADE < 0.5 m)")
    ax.fill_between([4500, 8500], 0, 0.2, alpha=0.07, color="#27ae60")

    ax.set_xlabel("End-to-end latency  (ms)", fontsize=11)
    ax.set_ylabel("Mean ADE vs. baseline  (m)", fontsize=11)
    ax.set_title("(b)  Latency–quality tradeoff\n"
                 "No truncated-CoC condition meets the ADE < 0.2 m pass threshold",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(4800, 8500)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    # Annotation: "all fail"
    ax.text(0.5, 0.93,
            "Conclusion: CoC must be complete (natural EOS).\n"
            "Truncation at any N produces out-of-distribution hidden state.",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#c0392b", fontweight="bold",
            bbox=dict(boxstyle="round", fc="#fdecea", ec="#c0392b", alpha=0.9))

    fig.suptitle("Experiment 1 — Latency vs. Trajectory Quality Tradeoff",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = OUT_DIR / "fig4_latency_vs_quality.png"
    plt.savefig(str(out))
    plt.close()
    print(f"Saved: {out}")


# ── Figure 5: Model prediction vs GT (ego_future_xyz) ────────────────────────

def fig5_pred_vs_gt(wps: dict[int, np.ndarray]):
    """
    Compare N=16 model prediction against ground-truth ego_future_xyz.
    GT is loaded from the same PhysicalAI clip used in the experiment.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        data = load_physical_aiavdataset(
            "030c760c-ae38-49aa-9ad8-f5650a545d26", t0_us=5_100_000)
    except Exception as e:
        print(f"  [SKIP fig5] Cannot load GT: {e}")
        return

    import torch
    gt_raw = data.get("ego_future_xyz")
    if gt_raw is None:
        print("  [SKIP fig5] ego_future_xyz not found in dataset")
        return

    if isinstance(gt_raw, torch.Tensor):
        gt_raw = gt_raw.cpu().numpy()
    gt = gt_raw.reshape(-1, 3)[:64]   # (64, 3)

    times = np.arange(1, 65) * DT
    wp16  = wps.get(16)
    wp0   = wps.get(0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── (a) BEV: N=16 pred vs GT ──────────────────────────────────────────
    ax = axes[0]
    if wp16 is not None:
        ax.plot(wp16[:, 0], wp16[:, 1], color=COLORS[16], lw=2.5,
                label="N=16  model prediction", zorder=6)
        ax.plot(wp16[-1, 0], wp16[-1, 1], "*", color=COLORS[16], ms=14, zorder=7)
    ax.plot(gt[:, 0], gt[:, 1], "g--", lw=2.5,
            label="Ground truth  (ego_future_xyz)", zorder=8)
    ax.plot(gt[-1, 0], gt[-1, 1], "g*", ms=14, zorder=9)
    ax.plot(0, 0, "ks", ms=10, zorder=10, label="Ego vehicle")
    ax.set_xlabel("Forward  x  (m)")
    ax.set_ylabel("Lateral  y  (m)")
    ax.set_title("(a)  N=16 prediction vs Ground Truth")
    ax.set_ylim(-3, 3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    if wp16 is not None:
        ade_gt16 = float(np.mean(np.linalg.norm(wp16[:, :2] - gt[:, :2], axis=1)))
        fde_gt16 = float(np.linalg.norm(wp16[63, :2] - gt[63, :2]))
        ax.text(0.02, 0.05,
                f"ADE(pred, GT) = {ade_gt16:.3f} m\nFDE(pred, GT) = {fde_gt16:.3f} m",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # ── (b) BEV: N=0 pred vs GT ───────────────────────────────────────────
    ax = axes[1]
    if wp0 is not None:
        ax.plot(wp0[:, 0], wp0[:, 1], color=COLORS[0], lw=2.5,
                label="N=0  Decode Skip", zorder=6)
        ax.plot(wp0[-1, 0], wp0[-1, 1], "*", color=COLORS[0], ms=14, zorder=7)
    ax.plot(gt[:, 0], gt[:, 1], "g--", lw=2.5, label="Ground truth", zorder=8)
    ax.plot(gt[-1, 0], gt[-1, 1], "g*", ms=14, zorder=9)
    ax.plot(0, 0, "ks", ms=10, zorder=10, label="Ego vehicle")
    ax.set_xlabel("Forward  x  (m)")
    ax.set_ylabel("Lateral  y  (m)")
    ax.set_title("(b)  N=0 Decode Skip vs Ground Truth")
    ax.set_ylim(-3, 3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    if wp0 is not None:
        ade_gt0 = float(np.mean(np.linalg.norm(wp0[:, :2] - gt[:, :2], axis=1)))
        fde_gt0 = float(np.linalg.norm(wp0[63, :2] - gt[63, :2]))
        ax.text(0.02, 0.05,
                f"ADE(pred, GT) = {ade_gt0:.3f} m\nFDE(pred, GT) = {fde_gt0:.3f} m",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="#fdecea", alpha=0.9))

    # ── (c) Error vs time for all N, now also vs GT ────────────────────────
    ax = axes[2]
    for n in TOKEN_LIST:
        wp = wps.get(n)
        if wp is None:
            continue
        dists_gt = np.linalg.norm(wp[:, :2] - gt[:, :2], axis=1)
        ax.plot(times, dists_gt, color=COLORS[n], lw=LW[n], alpha=0.85,
                label=f"N={n}")
    ax.axhline(0.5, color="#f39c12", lw=1.5, ls="--",
               label="0.5 m margin")
    ax.axhline(1.0, color="#c0392b", lw=1.8, ls="--",
               label="1.0 m margin")
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("L2 distance from GT  (m)")
    ax.set_title("(c)  All conditions vs Ground Truth")
    ax.set_xlim(0, 6.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Experiment 1 — Model Prediction vs Ground Truth Trajectory\n"
        "(GT = ego_future_xyz recorded in the PhysicalAI dataset)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    out = OUT_DIR / "fig5_pred_vs_gt.png"
    plt.savefig(str(out))
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading waypoints from: {RESULT_DIR.resolve()}")
    wps: dict[int, np.ndarray] = {}
    for n in TOKEN_LIST:
        wp = load_wp(n)
        if wp is not None:
            wps[n] = wp
            print(f"  N={n:>2}  wp[0]=({wp[0,0]:.3f}, {wp[0,1]:.3f}) m"
                  f"  wp[63]=({wp[63,0]:.2f}, {wp[63,1]:.3f}) m")

    if not wps:
        print("ERROR: No waypoints found. Run exp1_decode_skip.py first.")
        return

    summary = load_summary()

    print("\nRendering figures...")
    fig1_overview(wps)

    if 16 in wps and 0 in wps:
        fig2_comparison(wps[16], wps[0])

    fig3_error_over_time(wps)
    fig4_latency_quality(wps, summary)

    print("\nLoading GT for pred-vs-GT comparison...")
    fig5_pred_vs_gt(wps)

    print(f"\nDone. Figures saved to: {OUT_DIR.resolve()}")
    print("\nFetch on Windows (WSL):")
    print("  scp 'ice401@100.95.177.101:"
          "~/alpamayo1.5/evaluation/results/streaming/exp1_decode_skip/fig*.png' "
          "/mnt/c/Users/nanay/Desktop/Alphamayo/evaluation/results/streaming/exp1_decode_skip/")


if __name__ == "__main__":
    main()
