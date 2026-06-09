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
    stages   = ["VE\n(Vision\nEncoder)", "LM Prefill", "LM Decode\n(19 steps)", "Flow ODE"]
    duration = [728,  1423,  1503,  870]   # ms
    read_bw  = [80.3, 126.4, 204.6, 203.1] # GB/s
    peak_pct = [v / DRAM_PEAK * 100 for v in read_bw]
    free_bw  = [DRAM_PEAK - v for v in read_bw]
    l2_hit   = [49.2, 29.6, 37.7, 22.0]   # %

    # 색상: 포화도에 따라 초록→주황→빨강
    colors = []
    for p in peak_pct:
        if p < 50:
            colors.append("#2ecc71")   # 초록 (여유 많음)
        elif p < 75:
            colors.append("#f39c12")   # 주황 (중간)
        else:
            colors.append("#e74c3c")   # 빨강 (포화)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Alpamayo 1.5 on Jetson AGX Thor — 4-Stage DRAM Bandwidth Analysis\n"
                 "(ncu lts__d_sectors_fill_sysmem, LPDDR5X peak = 231 GB/s)",
                 fontsize=12, fontweight="bold", y=1.02)

    # ── 왼쪽: DRAM BW 포화도 (수평 막대) ──
    ax = axes[0]
    y_pos = np.arange(len(stages))
    bars = ax.barh(y_pos, peak_pct, color=colors, height=0.55, edgecolor="white", linewidth=1.5)

    # 여유 BW 표시 (투명 회색)
    ax.barh(y_pos, [100 - p for p in peak_pct], left=peak_pct,
            color="#ecf0f1", height=0.55, edgecolor="white", linewidth=1.5)

    # 수치 라벨
    for i, (bar, pct, fbw) in enumerate(zip(bars, peak_pct, free_bw)):
        ax.text(pct + 1, i, f"{pct:.0f}%   (+{fbw:.0f} GB/s free)",
                va="center", ha="left", fontsize=10, fontweight="bold",
                color=colors[i])

    # 체제 구분선 표시
    ax.axvline(50, color="#bdc3c7", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(75, color="#bdc3c7", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(25, -0.65, "Regime A\n(DMA opportunity)",
            ha="center", va="top", fontsize=9, color="#2ecc71", fontstyle="italic")
    ax.text(87, -0.65, "Regime B\n(BW saturated)",
            ha="center", va="top", fontsize=9, color="#e74c3c", fontstyle="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.set_xlim(0, 130)
    ax.set_xlabel("DRAM Read BW utilization (% of 231 GB/s peak)")
    ax.set_title("(a) DRAM Bandwidth Utilization per Stage")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # 범례
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Regime A: BW headroom ≥50% → DMA insertion possible"),
        mpatches.Patch(color="#f39c12", label="Regime B-mid: BW headroom ~25%"),
        mpatches.Patch(color="#e74c3c", label="Regime B: BW saturated (>75%) → DMA insertion limited"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

    # ── 오른쪽: 단계 시간 분해 (절대 ms) ──
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, duration, color=colors, height=0.55, edgecolor="white", linewidth=1.5)

    for i, (dur, pct) in enumerate(zip(duration, peak_pct)):
        ax2.text(dur + 20, i, f"{dur} ms\n({pct:.0f}% BW)",
                 va="center", ha="left", fontsize=9.5, color=colors[i])

    # 전체 추론 시간 표시
    total_ms = sum(duration)
    ax2.axvline(total_ms, color="none")
    ax2.text(1550, -0.75, f"Total: {total_ms:,} ms", ha="center", fontsize=10,
             fontweight="bold", color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.9))

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(stages)
    ax2.set_xlim(0, 2200)
    ax2.set_xlabel("Duration (ms)")
    ax2.set_title("(b) Inference Stage Duration")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # L2 hit rate 주석
    for i, (hit, dur) in enumerate(zip(l2_hit, duration)):
        ax2.text(dur / 2, i - 0.32, f"L2 hit {hit:.1f}%",
                 ha="center", va="top", fontsize=8.5, color="white", fontweight="bold")

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
    파이프라인 시각화:
    - 현재 직렬 실행 타임라인
    - 각 단계의 DRAM 포화도 (색상)
    - Bubble (유휴 시간) 표시
    - DMA 기회 (Prefill) 표시
    - CUDA Graph 기회 (Decode) 표시
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [1.8, 1.2]})
    fig.suptitle("Alpamayo 1.5 Inference Pipeline — Bubble Map & Async Pipeline Opportunity\n"
                 "(llm.npu ASPLOS 2025 framework applied to Jetson AGX Thor)",
                 fontsize=12, fontweight="bold", y=1.02)

    # ────────────────────────────────────────────────────────
    # 상단: 전체 추론 타임라인 (Gantt)
    # ────────────────────────────────────────────────────────
    ax_top = axes[0]

    stages = [
        {"name": "VE",     "t_start":    0, "t_end":  728, "bw_pct": 35, "color": "#2ecc71"},
        {"name": "LM\nPrefill", "t_start": 728, "t_end": 2151, "bw_pct": 55, "color": "#f39c12"},
        {"name": "LM\nDecode", "t_start": 2151, "t_end": 3654, "bw_pct": 89, "color": "#e74c3c"},
        {"name": "Flow\nODE",  "t_start": 3654, "t_end": 4524, "bw_pct": 88, "color": "#c0392b"},
    ]

    row_y = 0.55
    bar_h = 0.5

    for s in stages:
        dur = s["t_end"] - s["t_start"]
        rect = mpatches.FancyBboxPatch(
            (s["t_start"], row_y - bar_h/2), dur, bar_h,
            boxstyle="round,pad=0", linewidth=1.5,
            edgecolor="white", facecolor=s["color"], alpha=0.85
        )
        ax_top.add_patch(rect)

        # DRAM BW% 라벨
        mid_x = s["t_start"] + dur / 2
        ax_top.text(mid_x, row_y + 0.02,
                    f"{s['name']}\n{s['bw_pct']}% DRAM BW\n{dur} ms",
                    ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if s["bw_pct"] > 50 else "#2c3e50")

    # ── DMA 기회 표시 (Prefill FlashAttention 구간) ──
    fa_start = 728          # Prefill 시작
    fa_dur   = 738          # FA 총 시간 (18 calls × 41ms)
    fa_end   = fa_start + fa_dur

    dma_rect = mpatches.FancyBboxPatch(
        (fa_start, row_y + bar_h/2 + 0.05), fa_dur, 0.25,
        boxstyle="round,pad=0", linewidth=2,
        edgecolor="#27ae60", facecolor="#d5f5e3", alpha=0.9
    )
    ax_top.add_patch(dma_rect)
    ax_top.text(fa_start + fa_dur/2, row_y + bar_h/2 + 0.175,
                "★ DMA Opportunity: FlashAttention (738 ms)\n"
                "140 GB/s free → layer prefetch (~10 ms/layer × 18 = 180 ms hidden)",
                ha="center", va="center", fontsize=8.5, color="#1a7a40", fontweight="bold")

    # ── CUDA Graph 기회 표시 (Decode 커널 간 bubble) ──
    cuda_graph_dur = 159    # 10.6% × 1503 ms
    decode_mid = (2151 + 3654) / 2
    ax_top.annotate("",
                    xy=(decode_mid + 100, row_y - bar_h/2 - 0.1),
                    xytext=(decode_mid - 100, row_y - bar_h/2 - 0.1),
                    arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=2))
    ax_top.text(decode_mid, row_y - bar_h/2 - 0.22,
                f"CUDA Graph target: inter-kernel bubble ~159 ms (10.6%)",
                ha="center", va="top", fontsize=8.5, color="#8e44ad", fontweight="bold")

    ax_top.set_xlim(-50, 4700)
    ax_top.set_ylim(0, 1.5)
    ax_top.set_yticks([])
    ax_top.set_xlabel("Time (ms)", fontsize=10)
    ax_top.set_title("(a) Current Serial Inference Timeline — DRAM BW% and Bubble Locations", fontsize=11)

    # 색상 범례
    legend_elements = [
        mpatches.Patch(color="#2ecc71",  label="Regime A: DRAM < 50% (DMA headroom)"),
        mpatches.Patch(color="#f39c12",  label="Regime A-B: DRAM 55% (partial DMA)"),
        mpatches.Patch(color="#e74c3c",  label="Regime B: DRAM 89-88% (saturated)"),
        mpatches.Patch(color="#d5f5e3",  label="DMA prefetch opportunity (140 GB/s free)"),
        mpatches.Patch(color="#e8daef",  label="CUDA Graph opportunity (inter-kernel bubble)"),
    ]
    ax_top.legend(handles=legend_elements, loc="upper left", fontsize=8,
                  framealpha=0.9, ncol=2)

    # ────────────────────────────────────────────────────────
    # 하단: 목표 파이프라인 (Layer Prefetch 적용 후)
    # ────────────────────────────────────────────────────────
    ax_bot = axes[1]

    # 현재 Prefill
    ax_bot.barh(1.0, 1423, left=728, height=0.35,
                color="#f39c12", alpha=0.5, edgecolor="white", linewidth=1.5,
                label="Current Prefill (1,423 ms)")

    # 목표 Prefill (DMA 중첩으로 ~-180ms)
    target_prefill = 1423 - 180
    ax_bot.barh(0.45, target_prefill, left=728, height=0.35,
                color="#27ae60", alpha=0.85, edgecolor="white", linewidth=1.5,
                label=f"Target Prefill with layer prefetch (~{target_prefill} ms)")

    # DMA 구간 표시
    for i in range(18):
        layer_t = 728 + i * (target_prefill / 18)
        dma_t   = layer_t + (target_prefill / 18) * 0.12
        ax_bot.barh(0.45, 10, left=dma_t, height=0.35,
                    color="#1abc9c", alpha=0.6, edgecolor="none")

    ax_bot.text(728 + target_prefill/2, 0.45,
                f"Target: ~{target_prefill} ms  (-180 ms via DMA prefetch)",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax_bot.text(728 + 1423/2, 1.0,
                "Current: 1,423 ms", ha="center", va="center",
                fontsize=9, color="#2c3e50", fontweight="bold")

    # 절약 표시
    ax_bot.annotate("", xy=(728 + target_prefill, 0.7), xytext=(728 + 1423, 0.7),
                    arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=2))
    ax_bot.text(728 + (target_prefill + 1423)/2, 0.78,
                "−180 ms", ha="center", fontsize=9.5, color="#e74c3c", fontweight="bold")

    ax_bot.set_xlim(-50, 4700)
    ax_bot.set_ylim(0.1, 1.4)
    ax_bot.set_yticks([0.45, 1.0])
    ax_bot.set_yticklabels(["Target\n(layer prefetch)", "Current"], fontsize=9)
    ax_bot.set_xlabel("Time (ms)", fontsize=10)
    ax_bot.set_title("(b) Prefill Layer-level Async DMA Prefetch — Expected Gain", fontsize=11)

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
