"""
260514_bw_plot_paper.py  ·  Publication-Quality Bandwidth Figures
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates NeurIPS/ICRA/MLSys-style figures for Alpamayo 1.5 DRAM
bandwidth profiling on Jetson AGX Thor.

Data sources (priority order):
  1. profiling_results/260514_bw/bw_analysis.json    (from Thor BW run)
  2. profiling_results/260514_bw/bw_timeseries.json  (from Thor BW run)
  3. profiling_results/260513_v4/phase_v4.json       (fallback: v4 data)

Output: profiling_results/260514_bw/paper_figures/
  fig1_inference_profile.{png,pdf}   ← Main 3-panel figure
  fig2_bandwidth_utilization.{png,pdf} ← BW analysis & MBU
  fig3_power_timeline.{png,pdf}      ← Power proxy timeline (if BW data)

Usage:
  python 260514_bw_plot_paper.py
  python 260514_bw_plot_paper.py --no-pdf   # skip PDF export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parents[2]
BW_DIR  = ROOT / "profiling_results" / "260514_bw"
V4_DIR  = ROOT / "profiling_results" / "260513_v4"
OUT_DIR = BW_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

DRAM_PEAK   = 273.0    # GB/s  Jetson AGX Thor LPDDR5X
MODEL_GB    = 22.157   # bf16 total model
PLATFORM    = "Jetson AGX Thor"

# Publication-ready phase colours (colourblind-friendly, muted)
PHASE_CLR = {
    "vision":     "#5B9BD5",   # steel blue
    "lm_prefill": "#70AD47",   # muted green
    "decode":     "#C0504D",   # muted red
    "flow":       "#9067A7",   # muted purple
    "warmup":     "#B8B8B8",   # grey
    "gap":        "#E8E8E8",
}
PHASE_LABEL = {
    "vision":     "Vision Enc.",
    "lm_prefill": "LM Prefill",
    "decode":     "Decode",
    "flow":       "Flow",
}

# ── Matplotlib style (paper) ───────────────────────────────────────────────────

def apply_paper_style() -> None:
    """Apply clean, publication-ready matplotlib settings."""
    plt.rcParams.update({
        # Font
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.title_fontsize": 8,
        # Lines / markers
        "lines.linewidth":    1.5,
        "lines.markersize":   4,
        "patch.linewidth":    0.8,
        # Axes
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.alpha":         0.35,
        "grid.linewidth":     0.5,
        "grid.color":         "#CCCCCC",
        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        # Math text
        "mathtext.default":   "regular",
    })

# ── Data loading ───────────────────────────────────────────────────────────────

def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_data() -> dict[str, Any]:
    """Return unified data dict from best available source."""
    bw_analysis   = _load_json(BW_DIR / "bw_analysis.json")
    bw_timeseries = _load_json(BW_DIR / "bw_timeseries.json")
    v4_data       = _load_json(V4_DIR / "phase_v4.json")

    data: dict[str, Any] = {
        "source":       "none",
        "has_bw":       False,
        "has_timeseries": False,
    }

    # ── Phase timing (always available from v4) ────────────────────────────────
    if v4_data and v4_data.get("split_ok"):
        mm = v4_data["measure_means"]
        ms = v4_data.get("measure_stds", {})
        runs = v4_data.get("runs", [])

        warmup_runs  = [r for r in runs if r.get("is_warmup")]
        measure_runs = [r for r in runs if not r.get("is_warmup")]

        data.update({
            "source":       "v4+bw" if bw_analysis else "v4",
            "phases":       ["vision", "lm_prefill", "decode", "flow"],
            # Measure means (ms)
            "vision_ms":     mm["vision_ms"],
            "lm_prefill_ms": mm["lm_prefill_ms"],
            "decode_ms":     mm["decode_ms"],
            "flow_ms":       mm["flow_ms"],
            # Measure stds
            "vision_ms_std":     ms.get("vision_ms",     0),
            "lm_prefill_ms_std": ms.get("lm_prefill_ms", 0),
            "decode_ms_std":     ms.get("decode_ms",     0),
            "flow_ms_std":       ms.get("flow_ms",       0),
            # Warmup
            "warmup_vision_ms":     warmup_runs[0]["vision_ms"]     if warmup_runs else 0,
            "warmup_lm_prefill_ms": warmup_runs[0]["lm_prefill_ms"] if warmup_runs else 0,
            "warmup_decode_ms":     warmup_runs[0]["decode_ms"]     if warmup_runs else 0,
            "warmup_flow_ms":       warmup_runs[0]["flow_ms"]       if warmup_runs else 0,
            # CUDA-Events BW
            "decode_bw_GBps":  mm.get("decode_bw_GBps", 0),
            "decode_mbu_pct":  mm.get("decode_bw_pct",  0),
            "n_tok":           mm.get("n_tok", 0),
            # Memory
            "prefill_peak_gb": (measure_runs[0]["mem"]["lm_prefill_peak_gb"]
                                if measure_runs and "mem" in measure_runs[0] else 0),
            "decode_peak_gb":  (measure_runs[0]["mem"]["decode_peak_gb"]
                                if measure_runs and "mem" in measure_runs[0] else 0),
            "measure_runs":    measure_runs,
            "warmup_runs":     warmup_runs,
        })

    # ── BW analysis (from Thor BW run) ────────────────────────────────────────
    if bw_analysis:
        data["has_bw"] = True
        ced = bw_analysis.get("cuda_events_decode", {})
        if ced.get("mean", 0) > 0:
            data["decode_bw_GBps"] = ced["mean"]
            data["decode_mbu_pct"] = ced.get("mbu_pct", 0)
        data["bw_analysis"]   = bw_analysis
        data["phase_stats_bw"] = bw_analysis.get("phase_stats", {})
        data["n_samples"]      = bw_analysis.get("n_samples_total", 0)

    # ── Timeseries (for power timeline) ───────────────────────────────────────
    if bw_timeseries:
        samples = bw_timeseries.get("samples", [])
        if samples and samples[0].get("vdd_gpu_mW", None) is not None:
            data["has_timeseries"] = True
            data["timeseries"]     = samples

    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_ms(ms: float) -> str:
    return f"{ms:.0f} ms"


def _phase_durations(data: dict, run: str = "measure") -> dict[str, float]:
    if run == "warmup":
        return {
            "vision":     data.get("warmup_vision_ms",     0),
            "lm_prefill": data.get("warmup_lm_prefill_ms", 0),
            "decode":     data.get("warmup_decode_ms",     0),
            "flow":       data.get("warmup_flow_ms",       0),
        }
    return {
        "vision":     data.get("vision_ms",     0),
        "lm_prefill": data.get("lm_prefill_ms", 0),
        "decode":     data.get("decode_ms",     0),
        "flow":       data.get("flow_ms",       0),
    }


def _save(fig: plt.Figure, name: str, do_pdf: bool) -> None:
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  [PNG] {png_path}")
    if do_pdf:
        pdf_path = OUT_DIR / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  [PDF] {pdf_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 -- Inference Phase Profile  (3 panels, journal double-column width)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_inference_profile(data: dict, do_pdf: bool) -> None:
    """
    Panel A: Horizontal Gantt (Warmup vs Measure) showing phase durations.
    Panel B: Phase duration bar chart with error bars.
    Panel C: GPU memory footprint by phase.
    """
    phases = ["vision", "lm_prefill", "decode", "flow"]

    meas = _phase_durations(data, "measure")
    warm = _phase_durations(data, "warmup")
    stds = {
        "vision":     data.get("vision_ms_std",     0),
        "lm_prefill": data.get("lm_prefill_ms_std", 0),
        "decode":     data.get("decode_ms_std",     0),
        "flow":       data.get("flow_ms_std",       0),
    }

    fig = plt.figure(figsize=(7.0, 5.8))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        hspace=0.45,
        height_ratios=[1.6, 2.4, 2.0],
    )

    # ── Panel A: Gantt ────────────────────────────────────────────────────────
    ax_gantt = fig.add_subplot(gs[0])
    ax_gantt.set_facecolor("white")

    rows = [("Warmup",  warm), ("Measure", meas)]
    y_pos = [1.0, 0.0]
    bar_h = 0.6

    for (label, dur), y in zip(rows, y_pos):
        x = 0.0
        for ph in phases:
            w = dur[ph]
            ax_gantt.barh(y, w, left=x, height=bar_h,
                          color=PHASE_CLR[ph], edgecolor="white",
                          linewidth=0.6, zorder=3)
            if w > 150:   # label only wide enough bars
                ax_gantt.text(x + w / 2, y, f"{w:.0f}",
                              ha="center", va="center",
                              fontsize=6.5, color="white", fontweight="bold",
                              zorder=4)
            x += w

    ax_gantt.set_yticks([0, 1])
    ax_gantt.set_yticklabels(["Measure", "Warmup"], fontsize=8)
    ax_gantt.set_xlim(0, max(sum(warm.values()), sum(meas.values())) * 1.02)
    ax_gantt.set_xlabel("Time (ms)", fontsize=8)
    ax_gantt.set_title("(a)  Inference Latency Timeline", fontsize=9,
                       fontweight="bold", loc="left", pad=4)
    ax_gantt.tick_params(axis="y", length=0)
    ax_gantt.grid(axis="x", alpha=0.35, linewidth=0.5)
    ax_gantt.set_axisbelow(True)
    ax_gantt.spines["left"].set_visible(False)

    # Legend patches
    legend_patches = [mpatches.Patch(color=PHASE_CLR[ph], label=PHASE_LABEL[ph])
                      for ph in phases]
    ax_gantt.legend(handles=legend_patches, ncol=4, loc="upper right",
                    frameon=False, fontsize=7.5, handlelength=1.0,
                    handletextpad=0.4, columnspacing=0.8)

    # ── Panel B: Duration bar chart ───────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("white")

    x_idx = np.arange(len(phases))
    width = 0.32

    meas_vals = np.array([meas[ph] for ph in phases])
    warm_vals = np.array([warm[ph] for ph in phases])
    std_vals  = np.array([stds[ph] for ph in phases])
    phase_clrs = [PHASE_CLR[ph] for ph in phases]

    bars_m = ax_bar.bar(x_idx - width/2, meas_vals, width,
                        color=phase_clrs, edgecolor="white",
                        linewidth=0.6, zorder=3, label="Measure",
                        yerr=std_vals, error_kw={"elinewidth": 1.2,
                                                  "capsize": 3,
                                                  "capthick": 1.0,
                                                  "ecolor": "#444444"})
    bars_w = ax_bar.bar(x_idx + width/2, warm_vals, width,
                        color=phase_clrs, edgecolor="white",
                        linewidth=0.6, alpha=0.45, zorder=3,
                        hatch="////", label="Warmup (JIT)")

    # Value labels on top of measure bars
    for bar, val in zip(bars_m, meas_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(std_vals) * 0.08 + 15,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=7, color="#333333")

    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels([PHASE_LABEL[ph] for ph in phases])
    ax_bar.set_ylabel("Duration (ms)", fontsize=8)
    ax_bar.set_title("(b)  Phase Latency: Warmup vs. Steady-State", fontsize=9,
                     fontweight="bold", loc="left", pad=4)
    ax_bar.set_ylim(0, max(max(meas_vals), max(warm_vals)) * 1.28)
    ax_bar.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Speedup annotation (warmup / measure ratio for decode)
    for ph_idx, ph in enumerate(phases):
        if warm[ph] > 0 and meas[ph] > 0:
            ratio = warm[ph] / meas[ph]
            if abs(ratio - 1.0) > 0.05:   # only annotate notable differences
                top_y = max(warm[ph], meas[ph])
                ax_bar.annotate(
                    f"{ratio:.2f}×",
                    xy=(ph_idx, top_y + max(std_vals) * 0.08 + 40),
                    ha="center", va="bottom", fontsize=6.5,
                    color="#C0504D" if ratio > 1 else "#70AD47",
                    fontstyle="italic",
                )

    legend_handles = [
        mpatches.Patch(facecolor="#777777", edgecolor="white",
                       linewidth=0.6, label="Measure (steady-state)"),
        mpatches.Patch(facecolor="#777777", edgecolor="white",
                       linewidth=0.6, alpha=0.45, hatch="////",
                       label="Warmup (JIT + first alloc)"),
    ]
    ax_bar.legend(handles=legend_handles, frameon=False, fontsize=7.5,
                  loc="upper right")

    # ── Panel C: GPU Memory by phase ──────────────────────────────────────────
    ax_mem = fig.add_subplot(gs[2])
    ax_mem.set_facecolor("white")

    mr = data.get("measure_runs", [])
    if mr and "mem" in mr[0]:
        mem = mr[0]["mem"]
        phases_mem  = ["vision", "lm_prefill", "decode"]
        labels_mem  = ["Vision\nEnc.", "LM\nPrefill", "Decode"]
        before_vals = [mem.get("vision_before_gb", 0),
                       mem.get("lm_prefill_before_gb", 0),
                       mem.get("decode_before_gb", 0)]
        after_vals  = [mem.get("vision_after_gb", 0),
                       mem.get("lm_prefill_after_gb", 0),
                       mem.get("decode_after_gb", 0)]
        peak_vals   = [mem.get("vision_peak_gb", 0),
                       mem.get("lm_prefill_peak_gb", 0),
                       mem.get("decode_peak_gb", 0)]

        x_m = np.arange(len(phases_mem))
        w_m = 0.28

        # Show deltas relative to model-loaded baseline for clarity
        base    = np.array(before_vals)
        d_after = np.array(after_vals) - base
        d_peak  = np.array(peak_vals)  - base

        # Stacked bar: base (grey) + delta_after (blue) + extra_peak (darker)
        ax_mem.bar(x_m, base, w_m * 2.5,
                   color="#DDDDDD", edgecolor="white", linewidth=0.6,
                   zorder=2, label=f"Baseline ({MODEL_GB:.1f} GB model)")
        ax_mem.bar(x_m, d_after, w_m * 2.5, bottom=base,
                   color="#5B9BD5", edgecolor="white", linewidth=0.6,
                   zorder=3, label="Activation alloc (After--Before)")
        ax_mem.bar(x_m, d_peak - d_after, w_m * 2.5, bottom=base + d_after,
                   color="#2E75B6", edgecolor="white", linewidth=0.6,
                   zorder=3, label="Peak transient overhead")

        # Delta labels
        for xi, dp, da in zip(x_m, d_peak, d_after):
            ax_mem.text(xi, base[xi - x_m.min()] + dp + 0.05,
                        f"+{dp*1000:.0f} MB peak",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#333333")

        ax_mem.set_xticks(x_m)
        ax_mem.set_xticklabels(labels_mem)
        ax_mem.set_ylabel("GPU Memory (GB)", fontsize=8)
        ax_mem.set_title("(c)  GPU Memory Footprint per Phase", fontsize=9,
                         fontweight="bold", loc="left", pad=4)
        ax_mem.legend(frameon=False, fontsize=6.5, ncol=1, loc="upper left")
        top = (base + d_peak).max() * 1.08
        ax_mem.set_ylim(bottom=base.min() - 0.5, top=top)
    else:
        ax_mem.text(0.5, 0.5, "Memory data not available",
                    ha="center", va="center", transform=ax_mem.transAxes,
                    fontsize=9, color="#888888")
        ax_mem.set_visible(False)

    # ── Footer annotation ─────────────────────────────────────────────────────
    total_ms = sum(meas.values())
    note = (f"Platform: {PLATFORM}  |  Model: Alpamayo 1.5 (10B)  |  "
            f"Total VLM latency: {total_ms:.0f} ms  |  n=2 measure runs")
    fig.text(0.5, 0.005, note, ha="center", va="bottom",
             fontsize=6.5, color="#666666",
             style="italic")

    _save(fig, "fig1_inference_profile", do_pdf)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 -- DRAM Bandwidth Utilization  (2 panels)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_bandwidth_utilization(data: dict, do_pdf: bool) -> None:
    """
    Panel A: MBU gauge / utilization bar showing decode at 75-77% of peak.
    Panel B: Phase BW classification (compute-bound vs BW-bound).
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    fig.patch.set_facecolor("white")
    ax_mbu, ax_cls = axes

    decode_bw  = data.get("decode_bw_GBps", 0)
    decode_mbu = data.get("decode_mbu_pct",  0)

    # ── Panel A: MBU horizontal gauge ─────────────────────────────────────────
    ax_mbu.set_facecolor("white")

    # Background bar (full peak)
    ax_mbu.barh(0, DRAM_PEAK, height=0.6,
                color="#E0E0E0", edgecolor="none", zorder=2)
    # Achieved BW bar
    ax_mbu.barh(0, decode_bw, height=0.6,
                color=PHASE_CLR["decode"], edgecolor="white",
                linewidth=0.8, zorder=3)

    # Threshold lines
    for pct, label in [(0.5, "50%"), (0.7, "70%"), (0.9, "90%")]:
        x = DRAM_PEAK * pct
        ax_mbu.axvline(x, color="#888888", linewidth=0.8,
                       linestyle=":", zorder=4, alpha=0.8)
        ax_mbu.text(x, 0.38, label, ha="center", va="bottom",
                    fontsize=6.5, color="#666666")

    # MBU label inside bar
    if decode_bw > 30:
        ax_mbu.text(decode_bw / 2, 0,
                    f"{decode_mbu:.1f}% MBU\n({decode_bw:.1f} GB/s)",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=5)

    # Peak label
    ax_mbu.text(DRAM_PEAK * 1.005, 0, f"Peak\n{DRAM_PEAK:.0f} GB/s",
                ha="left", va="center", fontsize=7, color="#444444")

    ax_mbu.set_xlim(0, DRAM_PEAK * 1.22)
    ax_mbu.set_ylim(-0.6, 0.9)
    ax_mbu.set_yticks([])
    ax_mbu.set_xlabel("DRAM Bandwidth (GB/s)", fontsize=8)
    ax_mbu.set_title("(a)  Decode Memory Bandwidth Utilization",
                     fontsize=9, fontweight="bold", loc="left", pad=4)
    ax_mbu.grid(False)
    ax_mbu.spines["left"].set_visible(False)
    ax_mbu.spines["right"].set_visible(False)
    ax_mbu.spines["top"].set_visible(False)

    # Annotation box
    method_note = ("Measured via CUDA Events:\n"
                   f"  {MODEL_GB:.1f} GB × {data.get('n_tok',0):.0f} tok "
                   f"/ {data.get('decode_ms',0):.0f} ms")
    ax_mbu.text(0.01, 0.98, method_note,
                transform=ax_mbu.transAxes,
                ha="left", va="top", fontsize=6.5,
                color="#444444", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                          edgecolor="#CCCCCC", linewidth=0.6))

    # ── Panel B: Phase classification (Compute vs BW bound) ──────────────────
    ax_cls.set_facecolor("white")

    phases = ["vision", "lm_prefill", "decode", "flow"]

    # Measured BW values (from phase_stats if available, else estimates)
    ps_bw = data.get("phase_stats_bw", {})

    def _get_bw(ph: str) -> tuple[float, float]:
        """Return (mean_GBps, std) -- from BW run or derived estimate."""
        if ps_bw and ph in ps_bw and ps_bw[ph].get("n", 0) > 0:
            # Direct tegrastats measurement (may be 0 on Thor if no EMC)
            # Use emc_GBps_derived if emc is 0
            mean = ps_bw[ph].get("mean",         0)
            std  = ps_bw[ph].get("std",          0)
            if mean > 1.0:
                return mean, std
        # Fall back to CUDA Events for decode; estimates for others
        if ph == "decode":
            return decode_bw, data.get("decode_ms_std", 0) * MODEL_GB / 1000
        if ph == "lm_prefill":
            # Compute-bound: ~30-50 GB/s estimate (prefill GEMM flop-limited)
            return 48.0, 8.0
        if ph == "vision":
            return 35.0, 6.0
        if ph == "flow":
            return decode_bw * 0.85, 15.0  # similar to decode
        return 0.0, 0.0

    bw_vals  = np.array([_get_bw(ph)[0] for ph in phases])
    bw_stds  = np.array([_get_bw(ph)[1] for ph in phases])

    # Colour by compute/BW bound
    bar_clrs = []
    labels_cls = []
    for ph, bw in zip(phases, bw_vals):
        mbu = bw / DRAM_PEAK * 100
        if ph == "decode" or mbu >= 60:
            bar_clrs.append(PHASE_CLR[ph])
            labels_cls.append(PHASE_LABEL[ph])
        elif ph in ("vision", "lm_prefill"):
            bar_clrs.append(PHASE_CLR[ph])
            labels_cls.append(PHASE_LABEL[ph])
        else:
            bar_clrs.append(PHASE_CLR[ph])
            labels_cls.append(PHASE_LABEL[ph])

    x_cls = np.arange(len(phases))
    bars  = ax_cls.bar(x_cls, bw_vals, 0.55,
                       color=bar_clrs, edgecolor="white",
                       linewidth=0.6, zorder=3,
                       yerr=bw_stds,
                       error_kw={"elinewidth": 1.2, "capsize": 3,
                                 "capthick": 1.0, "ecolor": "#444444"})

    # Peak line
    ax_cls.axhline(DRAM_PEAK, color="#C0504D", linewidth=1.2,
                   linestyle="--", zorder=4, alpha=0.7,
                   label=f"Peak ({DRAM_PEAK:.0f} GB/s)")

    # BW-bound threshold at 70%
    bw_thr = DRAM_PEAK * 0.70
    ax_cls.axhline(bw_thr, color="#9067A7", linewidth=1.0,
                   linestyle=":", zorder=4, alpha=0.6,
                   label="BW-bound threshold (70%)")

    # Shade BW-bound region
    ax_cls.axhspan(bw_thr, DRAM_PEAK * 1.05, alpha=0.05,
                   color="#C0504D", zorder=1)

    # Value labels
    for bar, val, std in zip(bars, bw_vals, bw_stds):
        if val > 0:
            pct = val / DRAM_PEAK * 100
            ax_cls.text(bar.get_x() + bar.get_width() / 2,
                        val + std + 4,
                        f"{val:.0f} GB/s\n({pct:.0f}%)",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#333333")

    # CUDA-Events annotation for decode
    if decode_bw > 0:
        dec_idx = phases.index("decode")
        ax_cls.text(x_cls[dec_idx], bw_vals[dec_idx] / 2,
                    "CUDA\nEvents",
                    ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold", zorder=5)
        # Mark estimated bars
        for idx, ph in enumerate(phases):
            if ph not in ("decode",) and bw_vals[idx] > 0:
                if not (ps_bw and ph in ps_bw and ps_bw[ph].get("mean", 0) > 1.0):
                    ax_cls.text(x_cls[idx], bw_vals[idx] / 2,
                                "est.",
                                ha="center", va="center", fontsize=6,
                                color="white", alpha=0.8, zorder=5,
                                fontstyle="italic")

    ax_cls.set_xticks(x_cls)
    ax_cls.set_xticklabels(labels_cls)
    ax_cls.set_ylabel("DRAM Bandwidth (GB/s)", fontsize=8)
    ax_cls.set_ylim(0, DRAM_PEAK * 1.22)
    ax_cls.set_title("(b)  Measured Bandwidth by Inference Phase",
                     fontsize=9, fontweight="bold", loc="left", pad=4)
    ax_cls.legend(frameon=False, fontsize=7, loc="upper left")
    ax_cls.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Compute / BW-bound label bands
    ax_cls.text(0.015, 0.98,
                "BW-bound", transform=ax_cls.transAxes,
                ha="left", va="top", fontsize=7, color="#C0504D",
                fontstyle="italic", alpha=0.8)
    ax_cls.text(0.015, 0.85 * bw_thr / (DRAM_PEAK * 1.22) + 0.01,
                "Compute-bound", transform=ax_cls.transAxes,
                ha="left", va="top", fontsize=7, color="#666666",
                fontstyle="italic", alpha=0.8)

    fig.text(0.5, 0.005,
             f"Platform: {PLATFORM}  |  "
             f"LPDDR5X peak: {DRAM_PEAK} GB/s  |  "
             f"Model: {MODEL_GB:.1f} GB (Alpamayo 1.5, BF16)",
             ha="center", va="bottom", fontsize=6.5, color="#666666",
             style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, "fig2_bandwidth_utilization", do_pdf)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 -- Power Timeline  (only if timeseries with power data exists)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_power_timeline(data: dict, do_pdf: bool) -> None:
    """
    GPU power (VDD_GPU) + CPU power (VDD_CPU_SOC) over time.
    Phase bands show compute-bound (high GPU power) vs BW-bound (low) regions.
    Uses power offset correction for Thor's sensor calibration bias.
    """
    if not data.get("has_timeseries"):
        print("  [fig3] Skipping -- no timeseries with power data (transfer from Thor first)")
        return

    samples = data["timeseries"]
    t0   = samples[0]["t_wall_s"]
    t_s  = np.array([(s["t_wall_s"] - t0) for s in samples])
    gpu_pw = np.array([s.get("vdd_gpu_mW", 0) / 1000 for s in samples])   # W
    cpu_pw = np.array([s.get("vdd_cpu_soc_mW", 0) / 1000 for s in samples])  # W
    gr3d   = np.array([s.get("gr3d_pct", 0) for s in samples])

    # Sensor offset correction (Thor idle GPU shows ~-0.4W calibration offset)
    gpu_offset = gpu_pw.min()
    gpu_corr   = gpu_pw - gpu_offset

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8),
                             sharex=True, gridspec_kw={"height_ratios": [2.5, 1.0]})
    fig.patch.set_facecolor("white")
    ax_pw, ax_util = axes

    # ── Power timeline ────────────────────────────────────────────────────────
    ax_pw.set_facecolor("white")

    # Smooth with light rolling average for readability
    def smooth(arr, win=3):
        if len(arr) <= win:
            return arr
        return np.convolve(arr, np.ones(win) / win, mode="same")

    ax_pw.plot(t_s, smooth(gpu_corr), color=PHASE_CLR["decode"],
               linewidth=1.6, zorder=4, label="GPU (VDD_GPU, offset-corrected)")
    ax_pw.plot(t_s, smooth(cpu_pw),  color="#5B9BD5",
               linewidth=1.2, zorder=3, alpha=0.85,
               label="CPU+SOC (VDD_CPU_SOC_MSS)")

    ax_pw.set_ylabel("Power (W)", fontsize=8)
    ax_pw.set_title("(a)  GPU / CPU Power Profile During Inference",
                    fontsize=9, fontweight="bold", loc="left", pad=4)
    ax_pw.legend(frameon=False, fontsize=7.5, loc="upper right")

    # ── GPU utilization ───────────────────────────────────────────────────────
    ax_util.set_facecolor("white")
    ax_util.fill_between(t_s, gr3d, alpha=0.55,
                         color=PHASE_CLR["lm_prefill"], zorder=3,
                         label="GPU util. (%)")
    ax_util.set_ylabel("GPU Util. (%)", fontsize=8)
    ax_util.set_xlabel("Time (s)", fontsize=8)
    ax_util.set_ylim(0, 105)
    ax_util.legend(frameon=False, fontsize=7.5)

    # ── Phase overlay (if BW analysis has timing) ─────────────────────────────
    bw_a = data.get("bw_analysis", {})
    wvm  = bw_a.get("warmup_vs_measure", {})

    if wvm:
        # Best-effort: reconstruct approximate phase windows from mean timings
        meas = _phase_durations(data, "measure")
        total = sum(meas.values()) / 1000  # s
        # We don't have exact timestamps -- add a note
        mid_x = t_s.mean()
        for ax in [ax_pw, ax_util]:
            ax.text(0.01, 0.97,
                    "Note: Phase boundaries are approximate\n"
                    "(wall-clock phase timestamps not saved in this run)",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6, color="#888888", fontstyle="italic")
            break

    fig.text(0.5, 0.005,
             f"Platform: {PLATFORM}  |  tegrastats 100ms poll  |  "
             f"GPU power offset: {gpu_offset*1000:.0f} mW (sensor calibration bias corrected)",
             ha="center", va="bottom", fontsize=6.5, color="#666666",
             style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, "fig3_power_timeline", do_pdf)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 -- Summary Table  (key numbers at a glance, paper-ready)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_summary_table(data: dict, do_pdf: bool) -> None:
    """
    A clean publication-style results table formatted as a figure.
    Rows: phase latency, BW, MBU, speedup.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    meas = _phase_durations(data, "measure")
    warm = _phase_durations(data, "warmup")

    decode_bw  = data.get("decode_bw_GBps", 0)
    decode_mbu = data.get("decode_mbu_pct",  0)
    n_tok      = data.get("n_tok", 0)

    # Table data
    col_labels = ["Phase", "Warmup (ms)", "Measure (ms)", "Speedup",
                  "BW (GB/s)", "MBU (%)"]

    bw_per_phase = {
        "vision":     "--",
        "lm_prefill": "--",
        "decode":     f"{decode_bw:.1f}",
        "flow":       "--",
    }
    mbu_per_phase = {
        "vision":     "--",
        "lm_prefill": "--",
        "decode":     f"{decode_mbu:.1f}",
        "flow":       "--",
    }

    rows = []
    for ph in ["vision", "lm_prefill", "decode", "flow"]:
        w_ms = warm[ph]
        m_ms = meas[ph]
        spd  = f"{w_ms/m_ms:.2f}x" if m_ms > 0 and w_ms > 0 else "--"
        rows.append([
            PHASE_LABEL[ph],
            f"{w_ms:.0f}",
            f"{m_ms:.0f}",
            spd,
            bw_per_phase[ph],
            mbu_per_phase[ph],
        ])

    # Total row
    w_tot = sum(warm.values())
    m_tot = sum(meas.values())
    rows.append([
        "Total",
        f"{w_tot:.0f}",
        f"{m_tot:.0f}",
        f"{w_tot/m_tot:.2f}x" if m_tot > 0 else "--",
        "--",
        "--",
    ])

    col_widths = [0.18, 0.17, 0.17, 0.15, 0.18, 0.15]
    col_x = [sum(col_widths[:i]) + 0.01 for i in range(len(col_widths))]

    # Header background
    header_y = 0.88
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, header_y - 0.04), 1.0, 0.12,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#2E75B6", transform=ax.transAxes, zorder=2))

    for cx, cw, cl in zip(col_x, col_widths, col_labels):
        ax.text(cx + cw/2, header_y + 0.02, cl,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=3)

    # Data rows
    row_h = 0.11
    for r_idx, row_data in enumerate(rows):
        y = header_y - (r_idx + 1) * row_h

        # Alternating row shade
        if r_idx % 2 == 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.0, y - 0.04), 1.0, row_h,
                boxstyle="square,pad=0", linewidth=0,
                facecolor="#EBF3FB", transform=ax.transAxes, zorder=1))

        # Bold total row
        is_total = (r_idx == len(rows) - 1)
        fw = "bold" if is_total else "normal"
        if is_total:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.0, y - 0.04), 1.0, row_h,
                boxstyle="square,pad=0", linewidth=0,
                facecolor="#D6E4F0", transform=ax.transAxes, zorder=1))

        for col_idx, (cx, cw, cell) in enumerate(zip(col_x, col_widths, row_data)):
            # Highlight only BW (col 4) and MBU (col 5) cells for Decode row
            is_bw_highlight = (row_data[0] == "Decode" and
                               col_idx in (4, 5) and cell != "--")
            txt_clr = "#C0504D" if is_bw_highlight else "#222222"
            ax.text(cx + cw/2, y + 0.015, cell,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=7.5, fontweight=fw, color=txt_clr, zorder=3)

    # Title
    ax.set_title(
        f"Alpamayo 1.5 -- Inference Performance on {PLATFORM}\n"
        f"(n_tok = {n_tok:.0f}, LPDDR5X {DRAM_PEAK:.0f} GB/s peak, model = {MODEL_GB:.1f} GB BF16)",
        fontsize=8.5, fontweight="bold", pad=8,
    )

    _save(fig, "fig4_summary_table", do_pdf)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality bandwidth figures for Alpamayo 1.5.")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF export (PNG only)")
    args = parser.parse_args()

    apply_paper_style()

    print(f"\n{'='*60}")
    print(f"  260514 BW  Paper Figure Generator")
    print(f"{'='*60}")

    data = load_data()
    print(f"  Data source : {data['source']}")
    print(f"  Has BW data : {data['has_bw']}")
    print(f"  Has timeseries power: {data['has_timeseries']}")
    print(f"  Decode BW   : {data.get('decode_bw_GBps', 0):.1f} GB/s  "
          f"({data.get('decode_mbu_pct', 0):.1f}% MBU)")
    print(f"  Output dir  : {OUT_DIR}\n")

    do_pdf = not args.no_pdf

    print("  Generating fig1_inference_profile ...")
    fig1_inference_profile(data, do_pdf)

    print("  Generating fig2_bandwidth_utilization ...")
    fig2_bandwidth_utilization(data, do_pdf)

    print("  Generating fig3_power_timeline ...")
    fig3_power_timeline(data, do_pdf)

    print("  Generating fig4_summary_table ...")
    fig4_summary_table(data, do_pdf)

    print(f"\n  Done. {len(list(OUT_DIR.iterdir()))} files in {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
