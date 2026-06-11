#!/usr/bin/env python3
"""
260611_figure_dram_bw.py
========================
Alpamayo 1.5 on Jetson AGX Thor — DRAM 대역폭 실측 결과 시각화

생성 figure:
  Figure 1 (fig1_stage_bw.png)   : 4단계 DRAM BW 포화도 + 시간 분해
  Figure 2 (fig2_prefill_kernel.png): Prefill 커널별 DRAM BW% + 시간 비중
  Figure 3 (fig3_bubble_map.png) : Bubble 지도 — 각 자원 유휴 시간 + DMA 기회

실행:
  python3 260611_figure_dram_bw.py

출력:
  docs/2606_1주차/figures/fig1_stage_bw.png
  docs/2606_1주차/figures/fig2_prefill_kernel.png
  docs/2606_1주차/figures/fig3_bubble_map.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────
# 출력 디렉토리
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "2606_1주차", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 스타일 설정
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

DRAM_PEAK = 231.0   # GB/s, LPDDR5X 단방향 peak

# ─────────────────────────────────────────────────────────────
# Figure 1: 4단계 DRAM BW 포화도 + 파이프라인 시간
# ─────────────────────────────────────────────────────────────
def figure1_stage_bw():
    stages   = ["VE", "LM Prefill", "LM Decode", "Flow Matching"]
    read_bw  = [80.3, 126.4, 204.6, 203.1]   # GB/s
    peak_pct = [v / DRAM_PEAK * 100 for v in read_bw]

    # 포화도에 따라 초록 → 주황 → 빨강
    def bw_color(p):
        if p < 50:   return "#27ae60"
        if p < 75:   return "#e67e22"
        return              "#e74c3c"
    colors = [bw_color(p) for p in peak_pct]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("white")

    y = np.arange(len(stages))
    bar_h = 0.52

    # 배경 (100% 기준 회색)
    ax.barh(y, [100] * len(stages), height=bar_h,
            color="#f0f0f0", edgecolor="none", zorder=1)

    # 실제 BW 막대
    ax.barh(y, peak_pct, height=bar_h,
            color=colors, edgecolor="none", zorder=2)

    # 수치 라벨 (막대 끝, 굵게)
    for i, (pct, bw) in enumerate(zip(peak_pct, read_bw)):
        # BW% — 막대 안
        ax.text(pct - 1.5, i, f"{pct:.0f}%",
                va="center", ha="right", fontsize=14, fontweight="bold",
                color="white", zorder=3)
        # GB/s — 막대 오른쪽
        ax.text(pct + 1.5, i, f"{bw:.0f} GB/s",
                va="center", ha="left", fontsize=11, color=colors[i],
                fontweight="bold", zorder=3)

    # 100% 기준선
    ax.axvline(100, color="#aaaaaa", linewidth=1, linestyle="--", zorder=0)
    ax.text(100.5, len(stages) - 0.5, "peak\n231 GB/s",
            fontsize=8, color="#999999", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 120)
    ax.set_xlabel("DRAM Read BW  (% of LPDDR5X 231 GB/s)", fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Alpamayo 1.5 — DRAM Bandwidth per Stage",
                 fontsize=13, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig1_stage_bw.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Figure 1] 저장: {out_path}")


# ─────────────────────────────────────────────────────────────
# Figure 2: Prefill 커널별 DRAM BW% + 시간 비중
# ─────────────────────────────────────────────────────────────
def figure2_prefill_kernel():
    """
    Prefill 내 주요 커널 그룹:
      - GEMM (nvjet_256x128): 108 calls, ~70% BW, ~14ms each → 총 ~756ms (가중치 선형 투영)
      - FlashAttention: 18 calls, ~39.5% BW, ~41ms each → 총 ~738ms (SRAM-bound attention)
      - 기타 (norm, elementwise 등): 2070 - 108 - 18*2 = ~1930 kernels, 각 <1ms

    2-pass 보정 및 ncu 실측치 사용.
    """
    kernel_groups = [
        "GEMM\n(weight projections)\n108 calls",
        "FlashAttention\n18 calls",
        "Misc\n(norm, elementwise\netc.)",
    ]
    dram_bw_pct  = [70.0,  39.5,  8.0]     # % of peak
    time_share   = [53.0,  52.0,  12.0]     # % of Prefill time (738+756 > 1423 due to overlap estimation)
    # Note: 53% + 52% > 100% is intentional — showing contribution, not pie
    time_abs_ms  = [756,   738,   170]
    n_calls      = [108,   18,    "~1,900"]
    dram_free    = [DRAM_PEAK * (1 - p/100) for p in dram_bw_pct]

    x = np.arange(len(kernel_groups))
    width = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LM Prefill — Kernel-level DRAM Bandwidth Analysis\n"
                 "(ncu per-kernel measurement, seq_len = 3,086, 18 transformer blocks)",
                 fontsize=12, fontweight="bold", y=1.02)

    # ── 왼쪽: DRAM BW% ──
    ax = axes[0]
    bar_colors = ["#e74c3c", "#2ecc71", "#95a5a6"]
    bars = ax.bar(x, dram_bw_pct, width=0.5, color=bar_colors,
                  edgecolor="white", linewidth=1.5)

    # 여유 BW (DRAM 100% 기준 남은 부분)
    for i, (b, pct, fbw, nc) in enumerate(zip(bars, dram_bw_pct, dram_free, n_calls)):
        ax.text(b.get_x() + b.get_width()/2, pct + 1.5,
                f"{pct:.1f}%\n({fbw:.0f} GB/s free)\n{nc} calls",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color=bar_colors[i])

    # FlashAttention 강조 화살표
    ax.annotate("★ SRAM-bound!\nDRAM 60% free\n= 140 GB/s unused\n→ DMA opportunity",
                xy=(1, 39.5), xytext=(1.55, 62),
                fontsize=9, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#d5f5e3", alpha=0.9))

    ax.axhline(60, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(30, color="#2ecc71", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(-0.45, 62, "DRAM-bound threshold (60%)", fontsize=8, color="#e74c3c")
    ax.text(-0.45, 32, "SRAM/compute-bound threshold (30%)", fontsize=8, color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(kernel_groups, fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("DRAM BW Utilization (% of 231 GB/s peak)")
    ax.set_title("(a) DRAM BW% by Kernel Group")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # ── 오른쪽: 시간 비중 ──
    ax2 = axes[1]
    bars2 = ax2.bar(x, time_abs_ms, width=0.5, color=bar_colors,
                    edgecolor="white", linewidth=1.5)

    for i, (b, t_ms) in enumerate(zip(bars2, time_abs_ms)):
        pct_of_prefill = t_ms / 1423 * 100
        ax2.text(b.get_x() + b.get_width()/2, t_ms + 8,
                 f"{t_ms} ms\n({pct_of_prefill:.0f}% of Prefill)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=bar_colors[i])

    # 총 Prefill 시간 기준선
    ax2.axhline(1423, color="#2c3e50", linestyle="-.", linewidth=1.5)
    ax2.text(-0.4, 1435, "Total Prefill: 1,423 ms", fontsize=9, color="#2c3e50", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(kernel_groups, fontsize=9.5)
    ax2.set_ylim(0, 1700)
    ax2.set_ylabel("Duration (ms)")
    ax2.set_title("(b) Time Duration by Kernel Group")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig2_prefill_kernel.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Figure 2] 저장: {out_path}")


# ─────────────────────────────────────────────────────────────
# Figure 3: Bubble 지도 — Gantt 스타일 파이프라인 시각화
# ─────────────────────────────────────────────────────────────
def figure3_bubble_map():
    """
    Bubble 지도: 4단계 추론 타임라인에서 유휴 시간이 어디 있는지 직관적으로 표시.
    두 행: DRAM 사용 / GPU 유휴
    """
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor("white")

    # ── 단계 데이터 ──
    stages = [
        {"name": "VE",           "t_s":    0, "t_e":  728, "bw": 35, "c": "#27ae60"},
        {"name": "LM Prefill",   "t_s":  728, "t_e": 2151, "bw": 55, "c": "#e67e22"},
        {"name": "LM Decode",    "t_s": 2151, "t_e": 3654, "bw": 89, "c": "#e74c3c"},
        {"name": "Flow Matching","t_s": 3654, "t_e": 4524, "bw": 88, "c": "#c0392b"},
    ]
    TOTAL = 4524

    ROW_DRAM  = 1.6   # DRAM 사용 row y
    ROW_IDLE  = 0.7   # GPU 유휴 row y
    BAR_H     = 0.45

    # ── Row 라벨 ──
    ax.text(-200, ROW_DRAM, "DRAM\nUsage", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#555555")
    ax.text(-200, ROW_IDLE, "GPU\nIdle", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#555555")

    for s in stages:
        dur = s["t_e"] - s["t_s"]
        mid = (s["t_s"] + s["t_e"]) / 2

        # DRAM 사용 막대 (BW% 비례 높이)
        dram_h = BAR_H * s["bw"] / 100
        ax.barh(ROW_DRAM, dur, left=s["t_s"], height=dram_h,
                color=s["c"], alpha=0.85, edgecolor="white", linewidth=1.5)

        # 배경 (100% 기준)
        ax.barh(ROW_DRAM, dur, left=s["t_s"], height=BAR_H,
                color=s["c"], alpha=0.15, edgecolor="white", linewidth=0)

        # 단계명 + BW%
        ax.text(mid, ROW_DRAM + BAR_H * 0.75,
                f"{s['name']}\n{s['bw']}%  {dur} ms",
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="#2c3e50")

    # ── GPU 유휴 row ──
    # Prefill: FlashAttention 구간에서 DRAM idle (DMA 기회)
    fa_t_s, fa_dur = 728, 738
    ax.barh(ROW_IDLE, fa_dur, left=fa_t_s, height=BAR_H,
            color="#27ae60", alpha=0.75, edgecolor="white", linewidth=1.5)
    ax.text(fa_t_s + fa_dur / 2, ROW_IDLE,
            "DRAM idle\n140 GB/s free\n(FlashAttention)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")

    # Decode: 커널 간 유휴
    dec_t_s, dec_dur_idle = 2151, 159   # 10.6% × 1503
    ax.barh(ROW_IDLE, dec_dur_idle, left=dec_t_s, height=BAR_H,
            color="#8e44ad", alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.text(dec_t_s + dec_dur_idle / 2, ROW_IDLE,
            "Inter-kernel\nidle\n159 ms",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")

    # VE / Flow: 미측정 표시
    for t_s, dur_total in [(0, 728), (3654, 870)]:
        ax.barh(ROW_IDLE, dur_total, left=t_s, height=BAR_H,
                color="#bdc3c7", alpha=0.4, edgecolor="#aaaaaa",
                linewidth=1, linestyle="--")
        ax.text(t_s + dur_total / 2, ROW_IDLE, "TBD",
                ha="center", va="center", fontsize=9, color="#888888")

    # Decode 나머지 (커널 실행 중 = 유휴 아님)
    dec_active = 1503 - dec_dur_idle
    ax.barh(ROW_IDLE, dec_active, left=dec_t_s + dec_dur_idle, height=BAR_H,
            color="#e74c3c", alpha=0.2, edgecolor="white", linewidth=0)

    # Prefill 나머지
    prefill_rest = 1423 - fa_dur
    ax.barh(ROW_IDLE, prefill_rest, left=fa_t_s + fa_dur, height=BAR_H,
            color="#e67e22", alpha=0.2, edgecolor="white", linewidth=0)

    # ── 수직 구분선 (단계 경계) ──
    for s in stages[1:]:
        ax.axvline(s["t_s"], color="#cccccc", linewidth=1, linestyle="--", zorder=0)

    # ── x축 ──
    ax.set_xlim(-300, TOTAL + 100)
    ax.set_ylim(0.2, 2.25)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_title("Alpamayo 1.5 — Bubble Map  (Where GPU idles)", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # x tick: 단계 경계 ms
    tick_vals = [0, 728, 2151, 3654, 4524]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{v}" for v in tick_vals], fontsize=9)

    # ── 범례 ──
    legend_patches = [
        mpatches.Patch(color="#27ae60", alpha=0.75, label="DRAM idle — DMA insertion possible (Prefill FlashAttention)"),
        mpatches.Patch(color="#8e44ad", alpha=0.85, label="Inter-kernel idle — removable via CUDA Graph (Decode)"),
        mpatches.Patch(color="#bdc3c7", alpha=0.6,  label="TBD — kernel-level not yet measured (VE / Flow Matching)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8.5,
              framealpha=0.95, bbox_to_anchor=(0.01, 0.98))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig3_bubble_map.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Figure 3] 저장: {out_path}")


# ─────────────────────────────────────────────────────────────
# Figure 4: Prefill vs Decode Layer Feasibility 비교
# ─────────────────────────────────────────────────────────────
def figure4_layer_feasibility():
    """
    Layer prefetch 실현 가능성 직관적 비교:
    - Prefill: 79ms layer time >> 10ms prefetch time → ✅
    - Decode: 4.4ms layer time << 33ms prefetch time → ❌
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Layer-level Async Prefetch Feasibility Analysis\n"
                 "cudaMemPrefetchAsync opportunity: Prefill [OK] vs Decode [NG]",
                 fontsize=12, fontweight="bold", y=1.02)

    params = [
        {
            "stage": "LM Prefill",
            "layer_time_ms": 79.0,      # 1423 / 18
            "prefetch_time_ms": 10.0,   # 0.84 GB / 85 GB/s
            "feasible": True,
            "ax": axes[0],
            "color_compute": "#f39c12",
            "color_dma": "#27ae60",
        },
        {
            "stage": "LM Decode (per step)",
            "layer_time_ms": 4.4,       # 79.1 / 18
            "prefetch_time_ms": 33.0,   # 0.84 GB / 25 GB/s
            "feasible": False,
            "ax": axes[1],
            "color_compute": "#e74c3c",
            "color_dma": "#95a5a6",
        },
    ]

    for idx, p in enumerate(params):
        ax = p["ax"]
        layer_t   = p["layer_time_ms"]
        prefetch_t = p["prefetch_time_ms"]
        feasible  = p["feasible"]

        # 현재 layer 실행
        ax.barh(1.0, layer_t, left=0, height=0.4,
                color=p["color_compute"], alpha=0.9, edgecolor="white", linewidth=2,
                label=f"Layer N compute: {layer_t} ms")

        # DMA prefetch (async)
        if feasible:
            # DMA fits within layer compute time
            ax.barh(0.5, prefetch_t, left=0, height=0.4,
                    color=p["color_dma"], alpha=0.9, edgecolor="white", linewidth=2,
                    label=f"DMA prefetch Layer N+1: {prefetch_t} ms")
            ax.barh(1.0, layer_t, left=layer_t, height=0.4,
                    color=p["color_compute"], alpha=0.5, edgecolor="white", linewidth=2,
                    label=f"Layer N+1 compute (hidden wait)")
        else:
            # DMA extends BEYOND layer compute time (bad)
            ax.barh(0.5, prefetch_t, left=0, height=0.4,
                    color=p["color_dma"], alpha=0.6, edgecolor="white", linewidth=2,
                    label=f"DMA prefetch Layer N+1: {prefetch_t} ms")
            # Extra wait time
            extra_wait = prefetch_t - layer_t
            ax.barh(1.0, extra_wait, left=layer_t, height=0.4,
                    color="#bdc3c7", alpha=0.8, edgecolor="#7f8c8d", linewidth=2,
                    linestyle="--", label=f"STALL (wait for DMA): {extra_wait:.1f} ms")

        # 수치 라벨
        ax.text(layer_t / 2, 1.0, f"{layer_t} ms\n(compute)",
                ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        ax.text(prefetch_t / 2, 0.5, f"{prefetch_t} ms\n(DMA)",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if feasible else "#2c3e50")

        if feasible:
            verdict = f"[OK] FEASIBLE\n{layer_t} ms compute > {prefetch_t} ms DMA\nMargin: {layer_t - prefetch_t:.0f} ms"
            verdict_color = "#1a7a40"
        else:
            verdict = f"[NG] NOT FEASIBLE\n{layer_t} ms compute < {prefetch_t} ms DMA\nShortfall: {prefetch_t - layer_t:.0f} ms"
            verdict_color = "#922b21"

        ax.text(max(layer_t, prefetch_t) + 1, 0.75, verdict,
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                color=verdict_color,
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#d5f5e3" if feasible else "#fadbd8",
                          edgecolor=verdict_color, alpha=0.9))

        ax.set_xlim(0, max(layer_t, prefetch_t) * 1.8)
        ax.set_ylim(0.1, 1.5)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["DMA\n(Stream 2)", "Compute\n(Stream 1)"], fontsize=9.5)
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_title(f"({'ab'[idx]}) {p['stage']}", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig4_layer_feasibility.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Figure 4] 저장: {out_path}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Alpamayo 1.5 DRAM BW 분석 Figure 생성")
    print(f"  출력 디렉토리: {OUT_DIR}")
    print("=" * 60)

    figure1_stage_bw()
    figure2_prefill_kernel()
    figure3_bubble_map()
    figure4_layer_feasibility()

    print()
    print("=" * 60)
    print("  완료! 생성된 파일:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".png"):
            fpath = os.path.join(OUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f}  ({size_kb:.0f} KB)")
    print()
    print("  보고서: docs/2606_1주차/260611_교수님_보고서_DRAM대역폭_측정결과.md")
    print("=" * 60)
