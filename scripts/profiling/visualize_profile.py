"""
visualize_profile.py  ·  v4.0
────────────────────────────────────────────────────────────────────────────────
Alpamayo 1.5  논문급 프로파일링 시각화

데이터 소스:
  - profiling_results/summary_v4.json  (v4.0 실측)
  - profiling_results/raw_timings_v4.json
  - profiling_results/cpu_raw_samples.json

생성 Figure:
  Fig 1  stage_breakdown     <- 5단계 수평 스택바 + 도넛 차트
  Fig 2  run_variability     <- 런별 분산 (16 vs 19 step bimodal)
  Fig 3  cpu_gpu_timeline    <- CPU Core 02 + GPU 단계 Gantt
  Fig 4  decode_linearity    <- Decode step linearity (R^2)
  Fig 5  cpu_core_heatmap    <- 14코어 x 8런 히트맵 + 코어별 평균 바
  Fig 6  cpu_phase_bar       <- 단계별 CPU 활용률 + Core 02 추정
  Fig 7  hardware_compare    <- Thor vs A100 vs RTX PRO 6000
  Fig 8  optimization        <- 최적화 로드맵 waterfall
  Fig 9  core02_timeseries   <- Core 02 시계열 + 단계 오버레이
  Fig 10 memory_breakdown    <- 메모리 구성 + BW 컨텍스트
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import LogNorm
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 폰트 설정
# ─────────────────────────────────────────────────────────────────────────────
def _setup_font():
    from matplotlib import font_manager
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    available  = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            return name
    return "DejaVu Sans"

FONT = _setup_font()
matplotlib.rcParams.update({
    "axes.unicode_minus"  : False,
    "figure.dpi"          : 150,
    "savefig.dpi"         : 300,
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "axes.grid"           : True,
    "grid.alpha"          : 0.25,
    "grid.linestyle"      : "--",
    "font.size"           : 10,
    "axes.titlesize"      : 12,
    "axes.labelsize"      : 10,
    "xtick.labelsize"     : 9,
    "ytick.labelsize"     : 9,
    "legend.fontsize"     : 9,
    "legend.framealpha"   : 0.9,
    "figure.constrained_layout.use": True,
})

OUT_DIR  = Path("profiling_results/figures")
DATA_DIR = Path("profiling_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 컬러 팔레트
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "vision"   : "#4878CF",
    "prefill"  : "#6ACC65",
    "decode"   : "#D65F5F",
    "flow"     : "#B47CC7",
    "overhead" : "#C4AD66",
    "cpu_main" : "#E8454A",
    "cpu_sub"  : "#A0C4E8",
    "bg_16"    : "#EBF5FF",
    "bg_19"    : "#FFF0EB",
    "gray"     : "#888888",
    "dark"     : "#222222",
}

STAGE_COLORS       = [C["vision"], C["prefill"], C["decode"], C["flow"], C["overhead"]]
STAGE_LABELS_SHORT = ["Vision", "Prefill", "Decode", "Flow", "Overhead"]

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    summary_path = DATA_DIR / "summary_v4.json"
    raw_path     = DATA_DIR / "raw_timings_v4.json"
    cpu_path     = DATA_DIR / "cpu_raw_samples.json"

    if not summary_path.exists():
        summary_path = DATA_DIR / "summary.json"
    if not raw_path.exists():
        raw_path = DATA_DIR / "raw_timings.json"

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    cpu_raw = []
    if cpu_path.exists():
        with open(cpu_path, encoding="utf-8") as f:
            cpu_raw = json.load(f)
    return summary, raw, cpu_raw


def _ok(name):
    print(f"  [OK] {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1  Stage Breakdown
# ─────────────────────────────────────────────────────────────────────────────
def fig1_stage_breakdown(summary):
    t   = summary["timing_ms"]
    pct = summary["breakdown_pct"]

    stages = [
        ("vision_encoding", "Vision Encoding"),
        ("llm_prefill",     "LLM Prefill"),
        ("llm_decode",      "LLM Decode"),
        ("action_direct",   "Flow Matching"),
        ("action_overhead", "Action Overhead"),
    ]
    means  = [t[k]["mean"] for k, _ in stages]
    stds   = [t[k]["std"]  for k, _ in stages]
    total  = sum(means)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 왼쪽: 수평 스택 바 ──
    ax   = axes[0]
    y, h = 0.5, 0.55
    left = 0.0
    for i, ((k, label), m, s, color) in enumerate(
            zip(stages, means, stds, STAGE_COLORS)):
        ax.barh(y, m, height=h, left=left,
                color=color, edgecolor="white", linewidth=1.5)
        if m / total > 0.07:
            ax.text(left + m / 2, y,
                    f"{label}\n{m:.0f} ms\n({pct[k]:.1f}%)",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold", linespacing=1.4)
        else:
            ax.annotate(f"{label}\n{m:.0f} ms",
                        xy=(left + m, y),
                        xytext=(left + m + total * 0.015, y + 0.28),
                        fontsize=8, color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=color,
                                        lw=0.8, alpha=0.6))
        ax.errorbar(left + m, y, xerr=s, fmt="none",
                    ecolor="black", elinewidth=1.8,
                    capsize=5, capthick=1.8)
        left += m

    ax.set_xlim(0, total * 1.08)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Inference Latency (ms)", fontsize=11)
    ax.set_title("(a) End-to-End Latency Breakdown\n"
                 "Jetson AGX Thor  |  BF16  |  mean +/- std  (n=8 runs)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.text(0.5, -0.12, f"Total:  {total:.0f} ms  per inference",
            ha="center", va="top", fontsize=11,
            fontweight="bold", color=C["dark"],
            transform=ax.transAxes)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(axis="x", which="major", alpha=0.3)
    ax.grid(axis="x", which="minor", alpha=0.1)

    # ── 오른쪽: 도넛 ──
    ax2 = axes[1]
    wedge_props = dict(width=0.45, edgecolor="white", linewidth=2.5)
    _, _, autotexts = ax2.pie(
        means, colors=STAGE_COLORS,
        autopct="%1.1f%%", pctdistance=0.78,
        startangle=90, counterclock=False,
        wedgeprops=wedge_props,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")

    ax2.text(0, 0.10, f"{total:.0f}", ha="center", va="center",
             fontsize=21, fontweight="bold", color=C["dark"])
    ax2.text(0, -0.20, "ms / inference", ha="center", va="center",
             fontsize=9, color=C["gray"])

    handles = [mpatches.Patch(color=c,
                              label=f"{STAGE_LABELS_SHORT[i]}  {means[i]:.0f} ms")
               for i, c in enumerate(STAGE_COLORS)]
    ax2.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
    ax2.set_title("(b) Proportional Distribution",
                  fontsize=11, fontweight="bold", pad=10)

    fig.savefig(OUT_DIR / "fig1_stage_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig1_stage_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2  Run-by-Run Variability
# ─────────────────────────────────────────────────────────────────────────────
def fig2_run_variability(raw):
    vis   = [r["vision_encoding"] for r in raw]
    pre   = [r["llm_prefill"]     for r in raw]
    dec   = [r["llm_decode"]      for r in raw]
    flo   = [r["action_direct"]   for r in raw]
    steps = [r["decode_steps"]    for r in raw]
    tots  = [r["total_gpu"]       for r in raw]
    x     = np.arange(len(raw))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]

    # 배경: 16 vs 19 step
    for i, s in enumerate(steps):
        ax.axvspan(i - 0.45, i + 0.45,
                   color=C["bg_16"] if s == 16 else C["bg_19"],
                   alpha=0.55, zorder=0)

    # 스택바
    b1 = np.zeros(len(raw))
    for vals, color, label in [
        (vis, C["vision"],  "Vision Encoding"),
        (pre, C["prefill"], "LLM Prefill"),
        (dec, C["decode"],  "LLM Decode"),
        (flo, C["flow"],    "Flow Matching"),
    ]:
        ax.bar(x, vals, 0.65, bottom=b1, color=color,
               edgecolor="white", linewidth=1, label=label, zorder=2)
        b1 = b1 + np.array(vals)

    # Decode 내부 레이블 (step count)
    bot_dec = np.array(vis) + np.array(pre)
    for i, (d, s) in enumerate(zip(dec, steps)):
        ax.text(i, bot_dec[i] + d / 2,
                f"{s}x\n110ms",
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

    # 총합 레이블
    for i, t in enumerate(tots):
        ax.text(i, t + 35, f"{t:.0f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=C["dark"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"Run {i+1}" for i in range(len(raw))], fontsize=9)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_ylim(0, max(tots) * 1.13)
    ax.set_title("(a) Per-Run Latency Breakdown\nBackground: blue=16 CoC tokens, orange=19 CoC tokens",
                 fontsize=11, fontweight="bold")
    ax.yaxis.set_minor_locator(MultipleLocator(200))

    handles, labels = ax.get_legend_handles_labels()
    p16 = mpatches.Patch(color=C["bg_16"], alpha=0.8, label="16 CoC steps")
    p19 = mpatches.Patch(color=C["bg_19"], alpha=0.8, label="19 CoC steps")
    ax.legend(handles=handles + [p16, p19], ncol=2,
              fontsize=8.5, loc="upper left")

    # ── 오른쪽: Bimodal boxplot ──
    ax2 = axes[1]
    dec_16 = [dec[i] for i, s in enumerate(steps) if s == 16]
    dec_19 = [dec[i] for i, s in enumerate(steps) if s == 19]
    n16    = len(dec_16)
    n19    = len(dec_19)

    bp = ax2.boxplot([dec_16, dec_19], positions=[1, 2], widths=0.4,
                     patch_artist=True, notch=False,
                     medianprops=dict(color="white", linewidth=2.5),
                     whiskerprops=dict(linewidth=1.8),
                     capprops=dict(linewidth=2))
    bp["boxes"][0].set_facecolor(C["bg_16"])
    bp["boxes"][0].set_edgecolor(C["decode"])
    bp["boxes"][0].set_linewidth(2)
    bp["boxes"][1].set_facecolor(C["bg_19"])
    bp["boxes"][1].set_edgecolor(C["decode"])
    bp["boxes"][1].set_linewidth(2)

    for i, data in enumerate([dec_16, dec_19]):
        ax2.scatter([i + 1] * len(data), data, zorder=4,
                    color=C["decode"], s=65, alpha=0.85,
                    edgecolors="white", linewidths=1.2)

    # 이론값
    ax2.axhline(16 * 110.03, color=C["decode"],
                linestyle="--", linewidth=1.8, alpha=0.7,
                label=f"16 x 110 ms = {16*110:.0f} ms")
    ax2.axhline(19 * 110.03, color="#8B3A3A",
                linestyle=":", linewidth=1.8, alpha=0.7,
                label=f"19 x 110 ms = {19*110:.0f} ms")

    ax2.annotate(f"{16*110:.0f} ms (theory)",
                 xy=(1, 16*110.03), xytext=(1.55, 16*110.03 - 70),
                 fontsize=9, color=C["decode"],
                 arrowprops=dict(arrowstyle="->", color=C["decode"], lw=1.3))
    ax2.annotate(f"{19*110:.0f} ms (theory)",
                 xy=(2, 19*110.03), xytext=(1.52, 19*110.03 + 45),
                 fontsize=9, color="#8B3A3A",
                 arrowprops=dict(arrowstyle="->", color="#8B3A3A", lw=1.3))

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(
        [f"16-step  (n={n16})", f"19-step  (n={n19})"], fontsize=10)
    ax2.set_ylabel("LLM Decode Time (ms)", fontsize=11)
    ax2.set_title("(b) Decode Time by CoC Token Count\n"
                  "1 step = 110.0 ms  (R2 = 0.998)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="lower right")

    fig.savefig(OUT_DIR / "fig2_run_variability.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig2_run_variability.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3  CPU-GPU Execution Timeline
# ─────────────────────────────────────────────────────────────────────────────
def fig3_cpu_gpu_timeline(raw, cpu_raw):
    r       = raw[0]
    cpu_run = next((c for c in cpu_raw if c["run_id"] == 0), None)

    vis   = r["vision_encoding"]
    pre   = r["llm_prefill"]
    dec   = r["llm_decode"]
    flo   = r["action_direct"]
    ovh   = r["action_overhead"]
    total = r["total_gpu"]

    gpu_stages = [
        ("Vision Encoding",  0,           vis,           C["vision"]),
        ("LLM Prefill",      vis,         vis+pre,       C["prefill"]),
        ("LLM Decode\n(16 steps)", vis+pre, vis+pre+dec, C["decode"]),
        ("Flow Matching",    vis+pre+dec, vis+pre+dec+flo, C["flow"]),
    ]

    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, total * 1.03)
    ax.set_ylim(-0.3, 4.8)

    gpu_y = 3.3
    h     = 0.65

    # GPU 단계 바
    for label, t_s, t_e, color in gpu_stages:
        w = t_e - t_s
        ax.barh(gpu_y, w, height=h, left=t_s,
                color=color, edgecolor="white", linewidth=1.8, alpha=0.92)
        if w / total > 0.055:
            ax.text(t_s + w / 2, gpu_y,
                    f"{label}\n{w:.0f} ms",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold", linespacing=1.35)

    # Action Overhead
    ax.barh(gpu_y, ovh, height=h,
            left=vis+pre+dec+flo,
            color=C["overhead"], edgecolor="white", linewidth=1.5, alpha=0.92)

    ax.text(-30, gpu_y, "GPU\nStream",
            ha="right", va="center", fontsize=10, fontweight="bold",
            color=C["dark"])

    # CPU Core 02 + Core 00 시계열
    cpu_y_base = 1.6
    scale      = 1.3 / 100.0

    if cpu_run and cpu_run["raw_samples"]:
        samples = cpu_run["raw_samples"]
        t_arr   = np.array([s["t_ms"]    for s in samples])
        c02_arr = np.array([s["cores"][2] for s in samples])
        c00_arr = np.array([s["cores"][0] for s in samples])

        ax.fill_between(t_arr, cpu_y_base,
                         cpu_y_base + c02_arr * scale,
                         step="post", color=C["cpu_main"],
                         alpha=0.85, label="Core 02 (Python Main Thread)")
        ax.fill_between(t_arr, cpu_y_base,
                         cpu_y_base + c00_arr * scale,
                         step="post", color=C["cpu_sub"],
                         alpha=0.65, label="Core 00 (Sampler Sub-Thread)")

        # Y축 보조선 (0%, 50%, 100%)
        for pct_val, pct_label in [(0, "0%"), (50, "50%"), (100, "100%")]:
            y_line = cpu_y_base + pct_val * scale
            ax.axhline(y_line, color=C["cpu_main"],
                       linewidth=0.6, linestyle=":", alpha=0.35,
                       xmin=0, xmax=total / (total * 1.03))
            ax.text(total * 1.005, y_line, pct_label,
                    va="center", fontsize=7.5, color=C["cpu_main"])

        # 단계 경계 수직선
        markers_map = {m["name"]: m["t_ms"]
                       for m in cpu_run.get("markers", [])}
        for mk_name in ["vlm_start", "vision_end", "prefill_end",
                        "vlm_end", "flow_start", "flow_end"]:
            t_mk = markers_map.get(mk_name)
            if t_mk is not None:
                ax.axvline(t_mk, color=C["dark"], linewidth=0.9,
                           linestyle="--", alpha=0.3,
                           ymin=0.02, ymax=0.95)

    ax.text(-30, cpu_y_base + 0.65, "CPU\nCore 02",
            ha="right", va="center", fontsize=10, fontweight="bold",
            color=C["cpu_main"])

    # 구분선
    ax.axhline(2.55, color=C["gray"], linewidth=0.8,
               linestyle="-", alpha=0.3, xmin=0)

    ax.set_yticks([])
    ax.set_xlabel("Wall Time from Inference Start (ms)", fontsize=11)
    ax.set_title(
        "Fig 3  CPU-GPU Execution Timeline  |  Run 1  (16 CoC tokens, 4,843 ms)\n"
        "GPU: CUDA Event timing  |  CPU: psutil percpu 50 ms sampling  "
        "(Core 02 = Python GIL holder)",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.grid(axis="x", which="major", alpha=0.2)
    ax.grid(axis="x", which="minor", alpha=0.08)
    ax.spines["left"].set_visible(False)

    # 범례
    gpu_patches = [mpatches.Patch(color=c, label=l)
                   for l, _, _, c in gpu_stages]
    cpu_patches = [
        mpatches.Patch(color=C["cpu_main"], label="CPU Core 02 (Main Thread)"),
        mpatches.Patch(color=C["cpu_sub"],  alpha=0.65, label="CPU Core 00 (Sub)"),
    ]
    ax.legend(handles=gpu_patches + cpu_patches,
              loc="upper right", ncol=3, fontsize=8.5,
              bbox_to_anchor=(1.0, 0.99))

    fig.savefig(OUT_DIR / "fig3_cpu_gpu_timeline.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig3_cpu_gpu_timeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4  Decode Linearity
# ─────────────────────────────────────────────────────────────────────────────
def fig4_decode_linearity(raw):
    steps = np.array([r["decode_steps"] for r in raw], dtype=float)
    dec   = np.array([r["llm_decode"]   for r in raw], dtype=float)

    coeffs    = np.polyfit(steps, dec, 1)
    slope, intercept = coeffs
    x_fit     = np.linspace(14.5, 20.5, 200)
    y_fit     = np.polyval(coeffs, x_fit)

    y_pred    = np.polyval(coeffs, steps)
    ss_res    = np.sum((dec - y_pred) ** 2)
    ss_tot    = np.sum((dec - dec.mean()) ** 2)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # 데이터 포인트
    for s_val, bg in [(16, C["bg_16"]), (19, C["bg_19"])]:
        mask = (steps == s_val)
        ax.scatter(steps[mask], dec[mask],
                   color=bg, edgecolors=C["decode"],
                   s=130, linewidths=2.2, zorder=5,
                   label=f"{s_val} CoC steps (n={mask.sum()})")

    # 회귀선
    ax.plot(x_fit, y_fit, color=C["decode"], linewidth=2.2,
            zorder=3, label=f"Linear fit: y = {slope:.1f}x + {intercept:.0f}")

    # 이론 하한선
    ax.plot(x_fit, 81.2 * x_fit, color=C["gray"],
            linewidth=1.8, linestyle="--", alpha=0.7,
            label="BW theory min: 81.2 ms/step\n(22 GB / 273 GB/s)")

    # 음영: 실측 - 이론 = 오버헤드
    ax.fill_between(x_fit, 81.2 * x_fit, y_fit,
                    alpha=0.10, color=C["decode"],
                    label="BW overhead  (73.9% utilization)")

    # R^2 Annotation
    ax.annotate(
        f"Slope = {slope:.1f} ms / step\nR2 = {r2:.5f}",
        xy=(17.5, np.polyval(coeffs, 17.5)),
        xytext=(15.8, 2220),
        fontsize=10.5, fontweight="bold", color=C["decode"],
        arrowprops=dict(arrowstyle="->", color=C["decode"], lw=1.8),
        bbox=dict(boxstyle="round,pad=0.45",
                  fc="white", ec=C["decode"], alpha=0.92)
    )
    ax.annotate(
        "BW limit:\n81.2 ms/step",
        xy=(19.5, 81.2 * 19.5),
        xytext=(17.6, 81.2 * 19.5 - 200),
        fontsize=9, color=C["gray"],
        arrowprops=dict(arrowstyle="->", color=C["gray"], lw=1.3)
    )

    ax.set_xlabel("Decode Steps  (#CoC tokens generated)", fontsize=11)
    ax.set_ylabel("LLM Decode Latency (ms)", fontsize=11)
    ax.set_title(
        "Fig 4  Decode Latency vs. CoC Step Count\n"
        "Memory-bandwidth-bound: 22 GB weights / 273 GB/s LPDDR5x = 81.2 ms/step minimum",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(14.2, 20.8)
    ax.set_ylim(1450, 2450)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    ax.legend(loc="upper left", fontsize=9)

    fig.savefig(OUT_DIR / "fig4_decode_linearity.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig4_decode_linearity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5  CPU Core Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig5_cpu_core_heatmap(raw):
    n_runs  = len(raw)
    n_cores = 14

    matrix        = np.array(
        [r["cpu"]["per_core_mean_pct"] for r in raw], dtype=float
    )
    per_core_mean = matrix.mean(axis=0)
    per_core_max  = matrix.max(axis=0)

    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1.0],
                            figure=fig, wspace=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── 히트맵 (log scale) ──
    mat_log = np.where(matrix < 0.05, 0.05, matrix)
    im = ax1.imshow(mat_log, aspect="auto", cmap="Reds",
                    norm=LogNorm(vmin=0.05, vmax=100),
                    interpolation="nearest")

    ax1.set_xticks(range(n_cores))
    ax1.set_xticklabels([f"C{i:02d}" for i in range(n_cores)], fontsize=9)
    ax1.set_yticks(range(n_runs))
    ax1.set_yticklabels(
        [f"Run {i+1}  ({'16' if raw[i]['decode_steps']==16 else '19'} steps)"
         for i in range(n_runs)],
        fontsize=9
    )
    ax1.set_xlabel("CPU Core Index", fontsize=11)
    ax1.set_title("(a) Per-Core CPU Utilization (%)  —  All Runs\n"
                  "Log color scale  |  Core 02 = Python GIL main thread",
                  fontsize=11, fontweight="bold")

    # 셀 값 (Core 00, 02만)
    for ri in range(n_runs):
        for ci in [0, 2]:
            val      = matrix[ri, ci]
            txt_col  = "white" if val > 25 else "black"
            ax1.text(ci, ri, f"{val:.0f}",
                     ha="center", va="center",
                     fontsize=8.5, fontweight="bold", color=txt_col)

    # Core 02 강조 테두리
    for ri in range(n_runs):
        ax1.add_patch(plt.Rectangle((1.5, ri - 0.5), 1, 1,
                                     fill=False,
                                     edgecolor=C["cpu_main"],
                                     linewidth=2.8))

    cbar = fig.colorbar(im, ax=ax1, fraction=0.022, pad=0.02)
    cbar.set_label("CPU Utilization (%, log)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ── 오른쪽: 코어별 평균 바 ──
    bar_colors = [C["cpu_main"] if i == 2 else
                  ("#E0A0A0" if i == 0 else C["cpu_sub"])
                  for i in range(n_cores)]
    y_pos = np.arange(n_cores)

    ax2.barh(y_pos, per_core_mean, color=bar_colors,
             edgecolor="white", linewidth=0.8, height=0.68, zorder=3)

    for i, v in enumerate(per_core_mean):
        if v > 0.3:
            ax2.text(v + 0.4, i, f"{v:.1f}%",
                     va="center", fontsize=8.5, fontweight="bold",
                     color=C["cpu_main"] if i == 2 else C["dark"])

    # GIL 이론 최대
    ax2.axvline(100 / n_cores, color=C["dark"], linewidth=1.8,
                linestyle="--", alpha=0.55,
                label=f"Single-thread max\n(100%/{n_cores} = {100/n_cores:.1f}%)")

    # Core 02 화살표
    ax2.annotate("GIL Main Thread\n(CUDA dispatch\n+ Python interp.)",
                 xy=(per_core_mean[2], 2),
                 xytext=(30, 2 + 3.0),
                 fontsize=8.5, color=C["cpu_main"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->",
                                 color=C["cpu_main"], lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc="white", ec=C["cpu_main"], alpha=0.92))
    ax2.annotate("CPUSampler\nthread",
                 xy=(per_core_mean[0], 0),
                 xytext=(20, 0 + 2.5),
                 fontsize=8, color=C["cpu_sub"],
                 arrowprops=dict(arrowstyle="->",
                                 color=C["cpu_sub"], lw=1.2))

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Core {i:02d}" for i in range(n_cores)], fontsize=9)
    ax2.set_xlabel("Mean Utilization (%)", fontsize=11)
    ax2.set_xlim(0, 72)
    ax2.set_title("(b) Per-Core Mean\n(8-run average)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")

    fig.savefig(OUT_DIR / "fig5_cpu_core_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig5_cpu_core_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6  Phase-Level CPU Utilization
# ─────────────────────────────────────────────────────────────────────────────
def fig6_cpu_phase_bar(summary, raw):
    cs     = summary["cpu_summary"]
    bp_s   = cs["by_phase"]
    n_cores = cs["n_cores"]

    phase_data = [
        ("Vision\nEncoding",   bp_s["cpu_vision_pct"],  C["vision"]),
        ("LLM Prefill",        bp_s["cpu_prefill_pct"], C["prefill"]),
        ("VLM Total\n(V+P+D)", bp_s["cpu_vlm_pct"],     C["decode"]),
        ("Flow\nMatching",     bp_s["cpu_flow_pct"],     C["flow"]),
        ("Overall\n(inference)", cs["all_core_mean_pct"], C["gray"]),
    ]

    labels = [p[0] for p in phase_data]
    means  = [p[1]["mean"] for p in phase_data]
    stds   = [p[1]["std"]  for p in phase_data]
    colors = [p[2] for p in phase_data]
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── 왼쪽: 단계별 all-core 평균 ──
    ax = axes[0]
    bars = ax.bar(x, means, color=colors,
                  edgecolor="white", linewidth=1.8,
                  width=0.58, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="black", elinewidth=2.0,
                capsize=7, capthick=2.0, zorder=4)

    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.18, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=colors[xi])

    # GIL 이론 최대 기준선
    ax.axhline(100 / n_cores, color=C["dark"], linewidth=2.0,
               linestyle="--", alpha=0.55,
               label=f"GIL max (1/{n_cores} = {100/n_cores:.1f}%)")

    # 설명 annotation
    ax.annotate(
        "Flow: ODE loop\ntorch.randn() x N steps\n-> highest CPU load",
        xy=(3, means[3]),
        xytext=(3.4, means[3] + 2.2),
        fontsize=8.5, color=C["flow"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["flow"], lw=1.4),
        bbox=dict(boxstyle="round,pad=0.3",
                  fc="white", ec=C["flow"], alpha=0.9)
    )
    ax.annotate(
        "Vision: GPU-dominant\nCPU near idle",
        xy=(0, means[0]),
        xytext=(0.4, means[0] + 3.0),
        fontsize=8.5, color=C["vision"],
        arrowprops=dict(arrowstyle="->", color=C["vision"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3",
                  fc="white", ec=C["vision"], alpha=0.9)
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("All-Core Mean CPU Utilization (%)", fontsize=11)
    ax.set_title("(a) CPU Utilization by Inference Phase\n"
                 "psutil percpu  |  50 ms sampling  |  n=8 runs",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 14)
    ax.legend(fontsize=9, loc="upper left")

    # ── 오른쪽: Core 02 추정 (x14 scale-up) ──
    ax2 = axes[1]
    phase_c02      = [m * n_cores for m in means[:-1]]
    phase_labs_s   = ["Vision", "Prefill", "VLM Total", "Flow Match."]
    phase_cols     = colors[:-1]
    x2             = np.arange(len(phase_labs_s))

    ax2.bar(x2, phase_c02, color=phase_cols,
            edgecolor="white", linewidth=1.8,
            width=0.55, alpha=0.85, zorder=3)

    for xi, v in enumerate(phase_c02):
        ax2.text(xi, v + 1.0, f"~{v:.0f}%",
                 ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=phase_cols[xi])

    # 실측 Core 02 평균선
    core02_mean = cs["per_core_mean_pct"][2]
    ax2.axhline(core02_mean, color=C["cpu_main"], linewidth=2.5,
                linestyle="-", alpha=0.85,
                label=f"Core 02 measured mean: {core02_mean:.1f}%")
    ax2.axhline(100, color=C["gray"], linewidth=1.2,
                linestyle="--", alpha=0.5,
                label="100% (core saturation)")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(phase_labs_s, fontsize=10)
    ax2.set_ylabel("Core 02 Estimated Utilization (%)\n[all-core avg x 14]",
                   fontsize=10)
    ax2.set_title("(b) Core 02 (Main Thread) by Phase\n"
                  "Estimated from all-core mean x n_cores",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 125)
    ax2.legend(fontsize=9, loc="upper left")

    fig.savefig(OUT_DIR / "fig6_cpu_phase_bar.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig6_cpu_phase_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7  Hardware Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig7_hardware_compare():
    hw = [
        {"name": "Jetson AGX Thor\n(this work)",
         "bw": 273, "tdp": 60, "decode_ms": 110.0,
         "total_ms": 5008.7, "color": C["decode"], "marker": "*"},
        {"name": "NVIDIA A100 80G\n(PCIe)",
         "bw": 2000, "tdp": 300, "decode_ms": 15.0,
         "total_ms": 15.0*17.5 + 714.9 + 1471.9 + 890.3, "color": "#4878CF", "marker": "o"},
        {"name": "RTX PRO 6000\n(Blackwell)",
         "bw": 1700, "tdp": 300, "decode_ms": 17.6,
         "total_ms": 17.6*17.5 + 714.9 + 1471.9 + 890.3, "color": "#76B7B2", "marker": "s"},
        {"name": "Jetson Orin AGX\n(predecessor)",
         "bw": 204, "tdp": 40, "decode_ms": 147.0,
         "total_ms": 147.0*17.5 + 714.9 + 1471.9 + 890.3, "color": "#EDC948", "marker": "^"},
        {"name": "Apple M4 Max\n(96 GB)",
         "bw": 546, "tdp": 35, "decode_ms": 55.0,
         "total_ms": 55.0*17.5 + 714.9 + 1471.9 + 890.3, "color": "#B07AA1", "marker": "D"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── 왼쪽: BW vs Decode latency (log-log) ──
    ax = axes[0]
    bw_range = np.linspace(80, 2500, 400)
    theory   = 22157 / bw_range
    ax.plot(bw_range, theory, color=C["gray"], linewidth=2.0,
            linestyle="--", alpha=0.6, zorder=1,
            label="Theory: 22 GB / BW (100% util.)")

    for h in hw:
        ax.scatter(h["bw"], h["decode_ms"],
                   s=200, color=h["color"],
                   marker=h["marker"],
                   edgecolors="white", linewidths=1.8,
                   zorder=4)
        y_off = 5 if not h["name"].startswith("Jetson AGX T") else -15
        x_off = 0.05
        ax.annotate(h["name"],
                    xy=(h["bw"], h["decode_ms"]),
                    xytext=(h["bw"] * (1 + x_off), h["decode_ms"] + y_off),
                    fontsize=8.5, color=h["color"], fontweight="bold",
                    va="bottom")

    ax.annotate(
        "Thor: 110ms measured\n81ms theory  ->  73.9% BW",
        xy=(273, 110), xytext=(450, 125),
        fontsize=9, color=C["decode"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["decode"], lw=1.6),
        bbox=dict(boxstyle="round,pad=0.4",
                  fc="white", ec=C["decode"], alpha=0.92)
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Memory Bandwidth (GB/s)  [log]", fontsize=11)
    ax.set_ylabel("Decode Latency / step (ms)  [log]", fontsize=11)
    ax.set_title("(a) Memory BW vs. Decode Step Latency\n"
                 "Alpamayo 1.5  (22 GB, BF16)  —  Various Hardware",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(which="both", alpha=0.2)
    ax.grid(which="major", alpha=0.3)

    # ── 오른쪽: Total latency 수평 바 ──
    ax2 = axes[1]
    names  = [h["name"].replace("\n", " ") for h in hw]
    totals = [h["total_ms"] for h in hw]
    colors = [h["color"]    for h in hw]
    y_pos  = np.arange(len(hw))

    ax2.barh(y_pos, totals, color=colors,
             edgecolor="white", linewidth=1.5,
             height=0.58, alpha=0.88)
    for i, (v, col) in enumerate(zip(totals, colors)):
        ax2.text(v + 40, i, f"{v:.0f} ms",
                 va="center", fontsize=9.5,
                 fontweight="bold", color=col)

    ax2.axvline(100, color="green", linewidth=2.2,
                linestyle="--", alpha=0.75,
                label="Real-time target: 100 ms")
    ax2.text(130, -0.6, "Real-time\ntarget",
             fontsize=8.5, color="green", va="bottom")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9.5)
    ax2.set_xlabel("Estimated Total Inference Latency (ms)", fontsize=11)
    ax2.set_title("(b) Estimated End-to-End Latency by Platform\n"
                  "(17.5 decode steps avg, fixed Vision/Prefill/Flow)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_xlim(0, max(totals) * 1.16)

    fig.savefig(OUT_DIR / "fig7_hardware_compare.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig7_hardware_compare.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8  Optimization Roadmap
# ─────────────────────────────────────────────────────────────────────────────
def fig8_optimization_roadmap():
    # (label, delta, resulting_value, color)
    steps_opt = [
        ("Baseline\n(BF16, Eager)",        0,      5008.7,  "#BBBBBB"),
        ("TensorRT\nEngine",             -1200,    3808.7,  C["vision"]),
        ("FP4 Quant.\n(2x TFLOPS)",       -800,    3008.7,  C["prefill"]),
        ("CUDA Graphs\n(Decode loop)",    -370,    2638.7,  C["decode"]),
        ("Flash Attn.\n(SDPA)",           -320,    2318.7,  C["flow"]),
        ("KV Offload\n+ Prefetch",        -400,    1918.7,  C["overhead"]),
        ("Speculative\nDecoding",        -1800,     118.7,  "#76B7B2"),
        ("Target\n(100 ms)",                0,      100.0,  "#2CA02C"),
    ]

    fig, ax = plt.subplots(figsize=(15, 6))
    running = 5008.7
    x_pos   = list(range(len(steps_opt)))

    for i, (label, delta, result, color) in enumerate(steps_opt):
        if label.startswith("Baseline"):
            ax.bar(i, result, color=color, edgecolor="white",
                   linewidth=1.8, width=0.62, zorder=3, alpha=0.75)
            ax.text(i, result + 60, f"{result:.0f} ms",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        elif label.startswith("Target"):
            ax.bar(i, result, color=color, edgecolor="white",
                   linewidth=2.2, width=0.62, zorder=3, alpha=0.88)
            ax.text(i, result + 60, f"{result:.0f} ms\n(goal)",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#2CA02C")
        else:
            ax.bar(i, abs(delta), bottom=result,
                   color=color, edgecolor="white",
                   linewidth=1.8, width=0.62, zorder=3, alpha=0.88)
            ax.text(i, result - 70, f"{result:.0f} ms",
                    ha="center", va="top", fontsize=8.5, fontweight="bold")
            mid_y = result + abs(delta) / 2
            ax.text(i, mid_y,
                    f"-{abs(delta):.0f}",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
            running = result

        # 이음선
        if 0 < i < len(steps_opt) - 1:
            ax.plot([i - 0.31, i + 0.31], [result, result],
                    color=C["dark"], linewidth=1.5,
                    alpha=0.4, zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s[0] for s in steps_opt], fontsize=9)
    ax.set_ylabel("Inference Latency (ms)", fontsize=11)
    ax.set_title(
        "Fig 8  Optimization Roadmap — Alpamayo 1.5 on Jetson AGX Thor\n"
        f"Baseline: {5008.7:.0f} ms  ->  Target: 100 ms  (50x reduction required)",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 5008.7 * 1.12)
    ax.axhline(100, color="#2CA02C", linewidth=2.2,
               linestyle="--", alpha=0.7, label="100 ms real-time target")
    ax.legend(fontsize=9, loc="upper right")

    fig.savefig(OUT_DIR / "fig8_optimization_roadmap.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig8_optimization_roadmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9  Core 02 Time-Series
# ─────────────────────────────────────────────────────────────────────────────
def fig9_core02_timeseries(raw, cpu_raw):
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)

    # ── 상단: 8런 박스플롯 ──
    ax = axes[0]
    all_c02 = []
    for cpu_run in sorted(cpu_raw, key=lambda x: x["run_id"]):
        c02 = [s["cores"][2] for s in cpu_run["raw_samples"]]
        all_c02.append(c02)

    bp = ax.boxplot(all_c02, positions=range(1, 9),
                    widths=0.52, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.8, color=C["cpu_main"]),
                    capprops=dict(linewidth=2.2, color=C["cpu_main"]),
                    flierprops=dict(marker=".", markersize=5,
                                    color=C["cpu_main"], alpha=0.4))
    for box in bp["boxes"]:
        box.set_facecolor(C["cpu_main"])
        box.set_alpha(0.65)
        box.set_linewidth(2.0)
        box.set_edgecolor(C["cpu_main"])

    means_c02 = [r["cpu"]["per_core_mean_pct"][2] for r in raw]
    ax.scatter(range(1, 9), means_c02, zorder=5,
               color="white", s=80, edgecolors=C["cpu_main"],
               linewidths=2.2, label="Run mean")
    ax.axhline(np.mean(means_c02), color=C["cpu_main"],
               linestyle="--", linewidth=2.2, alpha=0.75,
               label=f"Overall mean: {np.mean(means_c02):.1f}%")
    ax.axhline(100, color=C["gray"], linewidth=1.2,
               linestyle=":", alpha=0.4, label="100% saturation")

    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(
        [f"Run {i}\n({'16' if raw[i-1]['decode_steps']==16 else '19'} steps)"
         for i in range(1, 9)],
        fontsize=9.5
    )
    ax.set_ylabel("Core 02 Utilization (%)", fontsize=11)
    ax.set_title("(a) Core 02 Utilization Distribution — All 8 Runs",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(-5, 115)
    ax.legend(fontsize=9, loc="upper right")

    # ── 하단: Run 2 시계열 (19-step) ──
    ax2  = axes[1]
    r1   = raw[1]
    cpu1 = next((c for c in cpu_raw if c["run_id"] == 1), None)

    if cpu1:
        samples = cpu1["raw_samples"]
        t_arr   = np.array([s["t_ms"]    for s in samples])
        c02_arr = np.array([s["cores"][2] for s in samples])
        c00_arr = np.array([s["cores"][0] for s in samples])

        ax2.step(t_arr, c02_arr, where="post",
                 color=C["cpu_main"], linewidth=2.2,
                 label="Core 02 (Main)", alpha=0.95, zorder=3)
        ax2.fill_between(t_arr, 0, c02_arr, step="post",
                         color=C["cpu_main"], alpha=0.25, zorder=2)
        ax2.step(t_arr, c00_arr, where="post",
                 color=C["cpu_sub"], linewidth=1.8,
                 label="Core 00 (Sub)", alpha=0.75, zorder=3)

        # 단계 배경 음영
        vis2 = r1["vision_encoding"]
        pre2 = r1["llm_prefill"]
        dec2 = r1["llm_decode"]
        flo2 = r1["action_direct"]

        spans = [
            (0,           vis2,           C["vision"],  "Vision"),
            (vis2,        vis2+pre2,       C["prefill"], "Prefill"),
            (vis2+pre2,   vis2+pre2+dec2,  C["decode"],  "Decode"),
            (vis2+pre2+dec2, vis2+pre2+dec2+flo2, C["flow"], "Flow"),
        ]
        for t_s, t_e, col, name in spans:
            ax2.axvspan(t_s, t_e, alpha=0.07, color=col, zorder=1)
            ax2.text((t_s + t_e) / 2, 103, name,
                     ha="center", va="bottom", fontsize=9,
                     color=col, fontweight="bold")

        # 마커 선
        for m in cpu1.get("markers", []):
            ax2.axvline(m["t_ms"], color=C["dark"],
                        linewidth=0.9, linestyle="--",
                        alpha=0.28, zorder=2)

    ax2.set_xlabel("Time from Inference Start (ms)", fontsize=11)
    ax2.set_ylabel("CPU Core Utilization (%)", fontsize=11)
    ax2.set_title("(b) Core 02 Utilization Time-Series — Run 2 (19 CoC tokens, 5,171 ms)\n"
                  "Background = GPU inference stage  |  CPU mark() points shown as dashed lines",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(-3, 115)
    ax2.set_xlim(0, r1["total_gpu"] * 1.01)
    ax2.xaxis.set_minor_locator(MultipleLocator(100))
    ax2.legend(fontsize=9, loc="upper right")

    fig.savefig(OUT_DIR / "fig9_core02_timeseries.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig9_core02_timeseries.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10  Memory Breakdown
# ─────────────────────────────────────────────────────────────────────────────
def fig10_memory_breakdown(summary):
    param_mb = summary["memory_mb"]["param_mem_mb"]
    act_mb   = summary["memory_mb"]["activation"]["mean"]
    total_hw = 131.9 * 1024  # MB

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── 왼쪽: 메모리 구성 도넛 ──
    ax = axes[0]
    segments = [
        ("Model Weights (BF16)", param_mb,              C["decode"]),
        ("Activations + KV",     act_mb,                C["flow"]),
        ("Available (unused)",   total_hw - param_mb - act_mb, "#DDDDDD"),
    ]
    vals   = [s[1] for s in segments]
    colors = [s[2] for s in segments]

    _, _, ats = ax.pie(
        vals, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2.5),
        autopct="%1.1f%%", pctdistance=0.76,
    )
    for at, col in zip(ats, colors[:2] + ["#999999"]):
        at.set_fontsize(9.5)
        at.set_fontweight("bold")
        at.set_color("white" if col != "#DDDDDD" else "#555555")

    ax.text(0, 0.12, f"{(param_mb+act_mb)/1024:.1f} GB",
            ha="center", va="center",
            fontsize=19, fontweight="bold", color=C["dark"])
    ax.text(0, -0.20, "used", ha="center", va="center",
            fontsize=10, color=C["gray"])

    lh = [mpatches.Patch(color=s[2],
                          label=f"{s[0]}:  {s[1]/1024:.1f} GB")
          for s in segments]
    ax.legend(handles=lh, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=9)
    ax.set_title(f"(a) GPU Memory Usage\nTotal HW Unified Memory: {total_hw/1024:.0f} GB",
                 fontsize=11, fontweight="bold")

    # ── 오른쪽: BW 비교 ──
    ax2 = axes[1]
    bw_rows = [
        ("A100 80G (PCIe)",      2000, "#4878CF"),
        ("RTX PRO 6000",          1700, "#76B7B2"),
        ("Apple M4 Max (96G)",     546, "#B07AA1"),
        ("Thor  (spec. 273 GB/s)", 273, C["gray"]),
        ("Thor  (effective 73.9%)", 201.7, C["decode"]),
        ("Jetson Orin AGX",         204, "#EDC948"),
    ]
    y_p    = np.arange(len(bw_rows))
    bw_v   = [b[1] for b in bw_rows]
    bw_c   = [b[2] for b in bw_rows]
    bw_lab = [b[0] for b in bw_rows]

    bars = ax2.barh(y_p, bw_v, color=bw_c,
                    edgecolor="white", linewidth=1.5,
                    height=0.58, alpha=0.87)
    for i, (v, col) in enumerate(zip(bw_v, bw_c)):
        ax2.text(v + 15, i, f"{v:.0f} GB/s",
                 va="center", fontsize=9.5,
                 fontweight="bold", color=col)

    ax2.annotate("73.9% BW utilization\n(110 ms actual / 81 ms theory)",
                 xy=(201.7, 4),
                 xytext=(500, 4 + 0.9),
                 fontsize=9, color=C["decode"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["decode"], lw=1.6),
                 bbox=dict(boxstyle="round,pad=0.35",
                           fc="white", ec=C["decode"], alpha=0.92))

    ax2.set_yticks(y_p)
    ax2.set_yticklabels(bw_lab, fontsize=9.5)
    ax2.set_xlabel("Memory Bandwidth (GB/s)", fontsize=11)
    ax2.set_title("(b) Memory Bandwidth Comparison\n"
                  "Decode bottleneck = memory-bandwidth-limited",
                  fontsize=11, fontweight="bold")

    fig.savefig(OUT_DIR / "fig10_memory_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    _ok("fig10_memory_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[Font] {FONT}")
    summary, raw, cpu_raw = load_data()
    print(f"[Data] {len(raw)} runs  |  CPU raw: {len(cpu_raw)} runs")

    print(f"\n[Generate] -> {OUT_DIR}/")
    fig1_stage_breakdown(summary)
    fig2_run_variability(raw)
    fig3_cpu_gpu_timeline(raw, cpu_raw)
    fig4_decode_linearity(raw)
    fig5_cpu_core_heatmap(raw)
    fig6_cpu_phase_bar(summary, raw)
    fig7_hardware_compare()
    fig8_optimization_roadmap()
    fig9_core02_timeseries(raw, cpu_raw)
    fig10_memory_breakdown(summary)

    print(f"\n[Done]  10 figures  ->  {OUT_DIR}/")
    print()
    print("  fig1   Stage Breakdown (stacked bar + donut)")
    print("  fig2   Run Variability (bimodal decode)")
    print("  fig3   CPU-GPU Timeline (Gantt, Core 02 real)")
    print("  fig4   Decode Linearity (BW limit, R^2)")
    print("  fig5   CPU Core Heatmap (14 cores x 8 runs)")
    print("  fig6   CPU Phase Bar (Vision/Prefill/Flow)")
    print("  fig7   Hardware Comparison (BW vs latency)")
    print("  fig8   Optimization Roadmap (waterfall)")
    print("  fig9   Core 02 Time-Series (phase overlay)")
    print("  fig10  Memory Breakdown (usage + BW context)")


if __name__ == "__main__":
    main()
