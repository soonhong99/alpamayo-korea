"""
260510_plot_thor_hardware_arch.py  —  v2.0
NVIDIA Jetson AGX Thor 하드웨어 Figure 생성기 (논문급, 단순/명료)

출력:
  profiling_results/260510_memory_utilization/figures/
    ├── fig_thor_platform.png/pdf       ← (a) SoC 구조 + Unified Memory
    └── fig_thor_memory_bottleneck.png/pdf  ← (b) 메모리 계층 병목 분석

데이터 출처:
  [M] Measured  — CUDA API / psutil / /sys 직접 측정
  [S] Spec      — NVIDIA Jetson Thor Datasheet DS-11945-001 (Feb 2026)
  [E] Estimated — 유사 아키텍처 문헌 추론 (논문에 "est." 명기 필요)
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# ── 한글 폰트 ────────────────────────────────────────────────────────────────
for _f in ["Malgun Gothic", "NanumGothic", "Apple SD Gothic Neo"]:
    if _f in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 출력 경로 ────────────────────────────────────────────────────────────────
OUT = Path("profiling_results/260510_memory_utilization/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── 검증된 스펙 (출처 표기) ──────────────────────────────────────────────────
SPEC = dict(
    # GPU  [M]easured via torch.cuda.get_device_properties()
    gpu_sm          = 20,           # [M]
    gpu_cuda        = 2560,         # [S] 20×128
    gpu_tensor      = 96,           # [S] 5th-gen Tensor Core
    gpu_fp4_tops    = 2070,         # [S] TFLOPS
    gpu_shared_kb   = 228,          # [M] Shared+L1 per SM
    gpu_l2_mb       = 33.6,         # [M] GPU LLC
    # CPU  [M]+[S]
    cpu_cores       = 14,           # [M]
    cpu_ghz         = 2.6,          # [S]
    cpu_l1_kb       = 64,           # [M]+[S]
    cpu_l2_mb       = 14,           # [M]+[S]  1 MB × 14 cores
    cpu_l3_mb       = 16,           # [S]  /sys 미탐지, 공식 스펙 사용
    # Memory  [M]+[S]
    mem_gb          = 131.9,        # [M]
    mem_bw          = 273,          # [S] GB/s
    mem_bus         = 256,          # [S] bit
    mem_type        = "LPDDR5X",    # [S]
    # Model  [M]
    model_params    = 11.08,        # [M] 실측 파라미터 수 (B)
    model_gb        = 22.16,        # [M] bf16 = 11.08B × 2 bytes
    infer_ms_theory = 81.2,         # [M] 22.16 GB / 273 GB/s × 1000
)


# ═════════════════════════════════════════════════════════════════════════════
#  Figure A  ·  Thor Platform Overview
#  목적: "Thor가 어떤 구조인가?" 한눈에
#  메시지: CPU + GPU가 동일 DRAM을 공유 → 273 GB/s가 유일한 대역폭
# ═════════════════════════════════════════════════════════════════════════════

def fig_platform():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # ── helper ───────────────────────────────────────────────────────────────
    def box(x, y, w, h, fc, ec, lw=1.4, r=0.12, zorder=2, alpha=1.0):
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={r}",
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           zorder=zorder, alpha=alpha)
        ax.add_patch(p)

    def txt(x, y, s, fs=9, color="#1a1a1a", bold=False, ha="center", va="center",
            style="normal", zorder=5):
        ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
                fontweight="bold" if bold else "normal",
                fontstyle=style, zorder=zorder)

    def hline(x0, x1, y, color="#555", lw=1.2, ls="-"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, zorder=3)

    def arrow_v(x, y0, y1, color="#555", lw=1.5):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="<->", color=color,
                                   lw=lw, mutation_scale=12),
                    zorder=4)

    # ── 제목 ─────────────────────────────────────────────────────────────────
    txt(4.5, 4.98,
        "NVIDIA Jetson AGX Thor — System Architecture",
        fs=12, bold=True)
    txt(4.5, 4.72,
        "All measured on Thor board (ice401@100.95.177.101) unless noted [S]",
        fs=8, color="#666", style="italic")

    # ═══════════════════════════════════════════════════════
    # SoC 외곽
    # ═══════════════════════════════════════════════════════
    box(0.25, 1.05, 8.5, 3.45, fc="#F4F6F9", ec="#999", lw=1.8, r=0.2, zorder=1)
    txt(4.5, 4.40, "SoC", fs=8.5, color="#888", bold=False)

    # ═══════════════════════════════════════════════════════
    # GPU 블록 (왼쪽)
    # ═══════════════════════════════════════════════════════
    GPU_X, GPU_W = 0.45, 3.8
    GPU_Y, GPU_H = 1.22, 3.10

    box(GPU_X, GPU_Y, GPU_W, GPU_H, fc="#DDEEFF", ec="#2E6DA4", lw=1.6, zorder=2)

    # GPU 헤더
    box(GPU_X, GPU_Y + GPU_H - 0.54, GPU_W, 0.54,
        fc="#2E6DA4", ec="#2E6DA4", lw=0, zorder=3, r=0.10)
    txt(GPU_X + GPU_W/2, GPU_Y + GPU_H - 0.20,
        "GPU  ·  Blackwell  ·  SM 11.0", fs=10, bold=True, color="white")
    txt(GPU_X + GPU_W/2, GPU_Y + GPU_H - 0.40,
        f"{SPEC['gpu_cuda']} CUDA cores  ·  {SPEC['gpu_tensor']} Tensor cores  ·  {SPEC['gpu_fp4_tops']} TOPS (FP4)  [S]",
        fs=7.5, color="#BDD9FF")

    # 20 SM 그리드 (4열 × 5행)
    sm_x0, sm_y0 = GPU_X + 0.18, GPU_Y + 1.70
    sw, sh, gx, gy = 0.68, 0.29, 0.80, 0.36
    for r in range(5):
        for c in range(4):
            bx = sm_x0 + c * gx
            by = sm_y0 - r * gy
            box(bx, by, sw, sh, fc="#AACCEE", ec="#4A8ABF", lw=0.7, zorder=4, r=0.05)
            txt(bx + sw/2, by + sh*0.62, f"SM{r*4+c:02d}",
                fs=5.8, bold=True, color="#1A3A5A")
            txt(bx + sw/2, by + sh*0.23, "128 CUDA | 228 KB",
                fs=5.0, color="#2A4A6A")

    txt(GPU_X + GPU_W/2, GPU_Y + GPU_H - 0.70,
        f"20 × SM  ·  {SPEC['gpu_shared_kb']} KB Shared+L1 per SM  =  4.45 MB total  [M]",
        fs=7.5, color="#2E6DA4")

    # GPU L2
    box(GPU_X + 0.18, GPU_Y + 0.15, GPU_W - 0.36, 0.60,
        fc="#2E6DA4", ec="#1A4A7A", lw=1.4, zorder=4, r=0.07)
    txt(GPU_X + GPU_W/2, GPU_Y + 0.52,
        f"GPU L2 Cache  {SPEC['gpu_l2_mb']} MB  [M]", fs=9.5, bold=True, color="white")
    txt(GPU_X + GPU_W/2, GPU_Y + 0.28,
        "GPU LLC (SRAM)  ·  on-chip", fs=7.8, color="#CCE4FF")

    # ═══════════════════════════════════════════════════════
    # CPU 블록 (오른쪽)
    # ═══════════════════════════════════════════════════════
    CPU_X = 4.75
    CPU_W = 3.80
    CPU_Y, CPU_H = 1.22, 3.10

    box(CPU_X, CPU_Y, CPU_W, CPU_H, fc="#DDEFDD", ec="#2E8042", lw=1.6, zorder=2)

    # CPU 헤더
    box(CPU_X, CPU_Y + CPU_H - 0.54, CPU_W, 0.54,
        fc="#2E8042", ec="#2E8042", lw=0, zorder=3, r=0.10)
    txt(CPU_X + CPU_W/2, CPU_Y + CPU_H - 0.20,
        "CPU  ·  Arm Neoverse V3AE  [S]", fs=10, bold=True, color="white")
    txt(CPU_X + CPU_W/2, CPU_Y + CPU_H - 0.40,
        f"{SPEC['cpu_cores']} cores  ·  {SPEC['cpu_ghz']} GHz  ·  SVE2  [S]",
        fs=7.5, color="#C8F0C8")

    # 14코어 (2열 × 7행)
    cx0, cy0 = CPU_X + 0.22, CPU_Y + 1.66
    cw, ch, cgx, cgy = 1.55, 0.24, 1.78, 0.30
    for row in range(7):
        for col in range(2):
            bx = cx0 + col * cgx
            by = cy0 - row * cgy
            box(bx, by, cw, ch, fc="#99CC99", ec="#2E8042", lw=0.7, zorder=4, r=0.04)
            txt(bx + cw/2, by + ch/2,
                f"Core {col*7+row:02d}  L1: {SPEC['cpu_l1_kb']}+{SPEC['cpu_l1_kb']} KB  L2: 1 MB",
                fs=5.6, color="#1A3A1A")

    txt(CPU_X + CPU_W/2, CPU_Y + CPU_H - 0.70,
        f"Per-core: L1 {SPEC['cpu_l1_kb']}+{SPEC['cpu_l1_kb']} KB  ·  L2 1 MB  [M+S]",
        fs=7.5, color="#2E8042")

    # CPU L3
    box(CPU_X + 0.18, CPU_Y + 0.78, CPU_W - 0.36, 0.48,
        fc="#2E8042", ec="#1A5A1A", lw=1.4, zorder=4, r=0.07)
    txt(CPU_X + CPU_W/2, CPU_Y + 1.07,
        f"L3 Cache (shared)  {SPEC['cpu_l3_mb']} MB  [S]", fs=9.5, bold=True, color="white")
    txt(CPU_X + CPU_W/2, CPU_Y + 0.88,
        "System LLC (SRAM)  ·  14-core shared  ·  /sys undetected*", fs=7.0, color="#D0FFD0")

    # CPU L2 합산
    box(CPU_X + 0.18, CPU_Y + 0.15, CPU_W - 0.36, 0.54,
        fc="#6AAA6A", ec="#2E8042", lw=1.0, zorder=4, r=0.06)
    txt(CPU_X + CPU_W/2, CPU_Y + 0.44,
        f"L2 Total  {SPEC['cpu_l2_mb']} MB  [M+S]", fs=9.5, bold=True, color="white")
    txt(CPU_X + CPU_W/2, CPU_Y + 0.24,
        "1 MB × 14 cores", fs=7.5, color="#DFFFDF")

    # ═══════════════════════════════════════════════════════
    # 연결 화살표 (GPU ↔ CPU)
    # ═══════════════════════════════════════════════════════
    ax.annotate("", xy=(CPU_X, 2.95), xytext=(GPU_X + GPU_W, 2.95),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1.3,
                                mutation_scale=10), zorder=4)
    txt(4.5, 3.08, "Unified address space (zero-copy)", fs=7.5, color="#555", style="italic")

    # ═══════════════════════════════════════════════════════
    # Unified Memory (하단)
    # ═══════════════════════════════════════════════════════
    MX, MY, MW, MH = 0.25, 0.10, 8.50, 0.85
    box(MX, MY, MW, MH, fc="#FFF5CC", ec="#856404", lw=2.0, zorder=3, r=0.15)

    txt(4.5, MY + MH*0.74,
        f"Unified Memory  ·  {SPEC['mem_gb']} GB  {SPEC['mem_type']}  ·  {SPEC['mem_bus']}-bit  ·  {SPEC['mem_bw']} GB/s  [M+S]",
        fs=11, bold=True, color="#5A4000")
    txt(4.5, MY + MH*0.30,
        "CPU and GPU share the same physical DRAM  —  no PCIe, no data copy",
        fs=8.5, color="#7A5A00", style="italic")

    # GPU L2 → DRAM 화살표
    arrow_v(GPU_X + GPU_W/2, MY + MH, GPU_Y + 0.15, color="#2E6DA4")
    txt(GPU_X + GPU_W/2 - 0.32, (MY + MH + GPU_Y + 0.15)/2,
        f"{SPEC['mem_bw']} GB/s", fs=8, color="#2E6DA4", bold=True, ha="left")

    # CPU L2 → DRAM 화살표
    arrow_v(CPU_X + CPU_W/2, MY + MH, CPU_Y + 0.15, color="#2E8042")
    txt(CPU_X + CPU_W/2 + 0.12, (MY + MH + CPU_Y + 0.15)/2,
        f"{SPEC['mem_bw']} GB/s", fs=8, color="#2E8042", bold=True, ha="left")

    # ═══════════════════════════════════════════════════════
    # 각주
    # ═══════════════════════════════════════════════════════
    txt(0.30, 0.04,
        "* CPU L3 undetected via /sys on JetPack 7 (known issue); 16 MB from official datasheet DS-11945-001",
        fs=6.8, color="#999", ha="left", va="bottom")

    plt.tight_layout(pad=0.1)
    _save(fig, "fig_thor_platform")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure B  ·  Memory Hierarchy & Inference Bottleneck
#  목적: "왜 Decode가 느린가?" 한눈에
#  메시지: 모델(22 GB) >> 모든 캐시 → 매 step DRAM에서 읽어야 함
# ═════════════════════════════════════════════════════════════════════════════

def fig_bottleneck():
    fig, (ax_bar, ax_ladder) = plt.subplots(
        1, 2, figsize=(11, 5.5),
        gridspec_kw={"width_ratios": [1.55, 1]}
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Alpamayo 1.5 (11.08 B)  Inference  ·  Memory Bottleneck Analysis\n"
        "NVIDIA Jetson AGX Thor",
        fontsize=13, fontweight="bold", y=0.99
    )

    # ─────────────────────────────────────────────────────
    # 왼쪽: 로그 스케일 가로 막대 (크기 비교)
    # ─────────────────────────────────────────────────────
    ax = ax_bar
    ax.set_facecolor("white")

    # 데이터: (레이블, 크기 GB, 색, 출처, 크기 표시 문자열)
    data = [
        ("GPU Shared+L1\n(total)",
         SPEC["gpu_shared_kb"] * SPEC["gpu_sm"] / 1024 / 1024,
         "#5B9BD5", "[M]",  "4.45 MB"),
        ("GPU L2 Cache",
         SPEC["gpu_l2_mb"] / 1024,
         "#2E6DA4", "[M]",  "33.6 MB"),
        ("CPU L3 Cache\n(shared)",
         SPEC["cpu_l3_mb"] / 1024,
         "#2E8042", "[S]",  "16 MB"),
        ("CPU L2\n(total)",
         SPEC["cpu_l2_mb"] / 1024,
         "#6AAA6A", "[M+S]","14 MB"),
        ("Alpamayo 1.5\n(bf16 weights)",
         SPEC["model_gb"],
         "#C00000", "[M]",  "22.16 GB"),
        ("Unified Memory",
         SPEC["mem_gb"],
         "#AAAAAA", "[M]",  "131.9 GB"),
    ]

    labels  = [d[0] for d in data]
    values  = [d[1] for d in data]
    colors  = [d[2] for d in data]
    sources = [d[3] for d in data]
    strs    = [d[4] for d in data]

    y_pos = np.arange(len(data))[::-1]          # 위에서 아래 순서
    bars  = ax.barh(y_pos, values, color=colors, edgecolor="white",
                    linewidth=1.4, height=0.62, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Size  (GB, log scale)", fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("(a)  Cache Hierarchy vs. Model Size", fontsize=11, fontweight="bold", pad=8)

    # 값 레이블 (막대 오른쪽)
    for bar, val, src, s in zip(bars, values, sources, strs):
        ax.text(val * 1.18, bar.get_y() + bar.get_height()/2,
                f" {s}  {src}", va="center", ha="left", fontsize=8.2, color="#333")

    # GPU L2 / CPU L3 임계선
    ax.axvline(SPEC["gpu_l2_mb"] / 1024, color="#2E6DA4", ls="--", lw=1.4, alpha=0.7,
               label=f"GPU L2  ({SPEC['gpu_l2_mb']} MB)")
    ax.axvline(SPEC["cpu_l3_mb"] / 1024, color="#2E8042", ls="-.", lw=1.4, alpha=0.7,
               label=f"CPU L3  ({SPEC['cpu_l3_mb']} MB)")

    # "모든 캐시 초과" 영역 음영
    ax.axvspan(SPEC["gpu_l2_mb"] / 1024, SPEC["model_gb"],
               alpha=0.07, color="#C00000", zorder=1,
               label="DRAM-bound region")
    ax.text(1.2, 0.52, "Model exceeds\nall caches\n→ DRAM-bound",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=8, color="#C00000", style="italic",
            bbox=dict(fc="white", ec="#C00000", alpha=0.85, boxstyle="round,pad=0.3"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", ls="--", alpha=0.3, zorder=0)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9, edgecolor="#ccc")

    # ─────────────────────────────────────────────────────
    # 오른쪽: 계층 사다리 + 병목 강조
    # ─────────────────────────────────────────────────────
    ax2 = ax_ladder
    ax2.set_facecolor("white")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axis("off")
    ax2.set_title("(b)  Decode Step  Bottleneck", fontsize=11, fontweight="bold", pad=8)

    # 계층 사다리 데이터 (아래→위 = 느림→빠름)
    # (이름, 크기 문자열, 대역폭 문자열, 색, 출처)
    rungs = [
        ("LPDDR5X DRAM",   "131.9 GB", "273 GB/s",        "#ED7D31", "[M+S]"),
        ("CPU L3 Cache",   "16 MB",    "~200 GB/s [E]",   "#2E8042", "[S]"),
        ("GPU L2 (LLC)",   "33.6 MB",  "~2 TB/s [E]",     "#2E6DA4", "[M]"),
        ("GPU Shared+L1",  "4.45 MB",  "~20 TB/s [E]",    "#5B9BD5", "[M]"),
    ]
    n = len(rungs)
    box_h = 0.185
    gap   = 0.038
    total = n * box_h + (n-1) * gap
    y0    = (1.0 - total) / 2 + 0.04

    from matplotlib.patches import FancyBboxPatch

    for i, (name, sz, bw, col, src) in enumerate(rungs):
        bx_center = 0.5
        # 피라미드: 위로 갈수록 좁게 (빠른 캐시 = 작은 박스)
        bw_ratio  = 0.42 + 0.50 * (i / (n-1))   # 0→0.42, n-1→0.92
        bw2       = bw_ratio
        bx        = bx_center - bw2 / 2
        by        = y0 + i * (box_h + gap)

        rect = FancyBboxPatch((bx, by), bw2, box_h,
                              boxstyle="round,pad=0,rounding_size=0.018",
                              facecolor=col, edgecolor="white", linewidth=1.5,
                              zorder=3)
        ax2.add_patch(rect)

        # 이름 + 크기
        ax2.text(bx_center, by + box_h * 0.65, name,
                 ha="center", va="center", fontsize=9.5, fontweight="bold",
                 color="white", zorder=4)
        ax2.text(bx_center, by + box_h * 0.38, sz,
                 ha="center", va="center", fontsize=8.8, color="#F0F0F0", zorder=4)
        ax2.text(bx_center, by + box_h * 0.13, bw,
                 ha="center", va="center", fontsize=8.0, color="#FFEE99",
                 style="italic", zorder=4)

    # 병목 화살표 (DRAM에서 올라오는 빨간 화살표)
    arrow_y_bot = y0 - 0.005
    ax2.annotate("",
                 xy=(0.5, arrow_y_bot),
                 xytext=(0.5, -0.04),
                 arrowprops=dict(arrowstyle="-|>", color="#C00000",
                                 lw=2.8, mutation_scale=18),
                 zorder=5)

    # 병목 설명 박스
    theory_ms = SPEC["infer_ms_theory"]
    box_txt = (
        f"Decode bottleneck\n"
        f"22.16 GB ÷ 273 GB/s\n"
        f"= {theory_ms:.1f} ms / step\n"
        f"(theoretical lower bound)"
    )
    ax2.text(0.50, -0.04, box_txt,
             ha="center", va="top", fontsize=8.5, color="#C00000",
             fontweight="bold", zorder=5,
             bbox=dict(fc="white", ec="#C00000", alpha=0.92,
                       boxstyle="round,pad=0.4", lw=1.4))

    # 범례 (출처 표기)
    ax2.text(0.5, 0.985,
             "[M] Measured   [S] Spec (DS-11945-001)   [E] Estimated",
             ha="center", va="top", fontsize=7.2, color="#666",
             style="italic", zorder=5)

    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=1.2)
    _save(fig, "fig_thor_memory_bottleneck")


# ── 저장 helper ───────────────────────────────────────────────────────────────
def _save(fig, name: str):
    for fmt, dpi in [("png", 300), ("pdf", None)]:
        p = OUT / f"{name}.{fmt}"
        kw = dict(bbox_inches="tight", facecolor=fig.get_facecolor())
        if fmt == "pdf":
            fig.savefig(p, format="pdf", **kw)
        else:
            fig.savefig(p, dpi=dpi, **kw)
        print(f"  [OK] {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("[1/2] Platform overview figure ...")
    fig_platform()
    print("[2/2] Memory bottleneck figure ...")
    fig_bottleneck()
    print(f"\nDone → {OUT.resolve()}")
