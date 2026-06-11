"""
visualize_pipeline.py  ·  v1.0
────────────────────────────────────────────────────────────────────────────────
CPU vs GPU 파이프라인 시각화 — 교수님 발표 전용

"GPU가 무엇을 할 때 CPU가 기다리고,
 CPU가 무엇을 할 때 GPU가 기다리는가"를 한눈에 보여주는 그림

3개 패널:
  Panel 1 (상단): 전체 파이프라인 — 토크나이징 포함 5,100ms 전체
  Panel 2 (하단 좌): 토크나이징 구간 줌인 — GPU idle 직접 확인
  Panel 3 (하단 우): Decode 1 step 줌인 — CPU launch → GPU run 반복 패턴

데이터:
  - GPU 타이밍: profiling_results/raw_timings_v4.json (Run 1, 실측)
  - CPU 파형:   profiling_results/cpu_raw_samples.json (Run 1, 실측)
  - 토크나이징: nsys 데모 실측 90.5ms (92.1ms, 88.8ms 평균)
"""

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ─── 폰트 ────────────────────────────────────────────────────────────────────
from matplotlib import font_manager
_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
_available  = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in _candidates if n in _available), "DejaVu Sans")
matplotlib.rcParams["font.family"]       = FONT
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"]         = 150
matplotlib.rcParams["savefig.dpi"]        = 300

DATA_DIR = Path("profiling_results")
OUT_DIR  = Path("profiling_results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 컬러 팔레트 ──────────────────────────────────────────────────────────────
C = {
    "tokenize" : "#FF6B6B",   # 빨강  — CPU 전용
    "vision"   : "#4878CF",   # 파랑
    "prefill"  : "#6ACC65",   # 초록
    "decode"   : "#D65F5F",   # 진빨강
    "flow"     : "#B47CC7",   # 보라
    "overhead" : "#C4AD66",   # 황토
    "detok"    : "#FF6B6B",   # 빨강  — CPU 전용
    "cpu_main" : "#E8454A",   # CPU Core 02
    "cpu_idle" : "#DDDDDD",   # 연회색 — idle
    "gpu_idle" : "#F0F0F0",   # 연회색 — GPU idle
    "idle_txt" : "#999999",
    "dark"     : "#222222",
    "white"    : "#FFFFFF",
}

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
def load():
    with open(DATA_DIR / "raw_timings_v4.json") as f:
        raw = json.load(f)
    with open(DATA_DIR / "cpu_raw_samples.json") as f:
        cpu_raw = json.load(f)
    return raw, cpu_raw

# ─────────────────────────────────────────────────────────────────────────────
# 메인 Figure
# ─────────────────────────────────────────────────────────────────────────────
def draw(raw, cpu_raw):
    # Run 1 (16-step, 4,843ms) 사용
    r       = raw[0]
    cpu_run = next(c for c in cpu_raw if c["run_id"] == 0)

    # ── 실측 타이밍 ──────────────────────────────────────────────────────────
    TOK  = 90.5          # 토크나이징 실측 (nsys demo 평균)
    VIS  = r["vision_encoding"]    # 708.6 ms
    PRE  = r["llm_prefill"]        # 1464.1 ms
    DEC  = r["llm_decode"]         # 1771.4 ms
    FLO  = r["action_direct"]      # 891.9 ms
    OVH  = r["action_overhead"]    # 6.5 ms
    DTOK = 2.0                     # 디토크나이징 추정 (이 run에선 미측정)
    N_STEPS = r["decode_steps"]    # 16

    # 시간 원점: 토크나이징 시작 = t=0
    # cpu_raw_samples 는 추론 시작(=VIS 시작)이 t=0
    # → cpu_raw에 TOK를 오프셋으로 더해 정렬

    # 전체 길이
    TOTAL = TOK + VIS + PRE + DEC + FLO + OVH + DTOK

    # 각 단계의 절대 시작/종료 시각
    t_vis_s  = TOK
    t_vis_e  = TOK + VIS
    t_pre_s  = t_vis_e
    t_pre_e  = t_pre_s + PRE
    t_dec_s  = t_pre_e
    t_dec_e  = t_dec_s + DEC
    t_flo_s  = t_dec_e
    t_flo_e  = t_flo_s + FLO
    t_ovh_e  = t_flo_e + OVH
    t_dtok_e = t_ovh_e + DTOK

    # CPU 샘플 (t=0 = VIS 시작 → +TOK 오프셋)
    samples  = cpu_run["raw_samples"]
    cpu_t    = np.array([s["t_ms"] + TOK for s in samples])
    cpu_c02  = np.array([s["cores"][2]    for s in samples])

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.45, wspace=0.30,
    )
    ax_full  = fig.add_subplot(gs[0, :])   # 상단 전체
    ax_zoom1 = fig.add_subplot(gs[1, 0])   # 하단 좌: 토크나이징 줌
    ax_zoom2 = fig.add_subplot(gs[1, 1])   # 하단 우: Decode 1step 줌

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 1 — 전체 파이프라인
    # ══════════════════════════════════════════════════════════════════════════
    ax = ax_full
    ax.set_xlim(-50, TOTAL + 80)
    ax.set_ylim(-0.5, 5.2)
    ax.set_yticks([])
    ax.set_xlabel("Wall Time from Tokenization Start (ms)", fontsize=11)
    ax.set_title(
        "Fig A  CPU vs GPU Full Pipeline\n"
        f"Total: {TOTAL:.0f} ms  |  GPU idle: {TOK + DTOK:.0f} ms (Tokenize+Detokenize)  |"
        f"  CPU serial overhead: ~419 ms (Decode kernel launch)",
        fontsize=11, fontweight="bold", pad=10
    )

    GPU_Y  = 3.5
    CPU_Y  = 1.8
    BAR_H  = 0.65
    SCALE  = 1.1 / 100.0  # 100% → 1.1 높이

    # ── GPU 레인 레이블 ──
    ax.text(-45, GPU_Y, "GPU\nStream", ha="right", va="center",
            fontsize=10, fontweight="bold", color=C["dark"])
    ax.text(-45, CPU_Y + 0.55, "CPU\nCore 02", ha="right", va="center",
            fontsize=10, fontweight="bold", color=C["cpu_main"])

    # ── GPU idle 배경 (토크나이징 구간) ──
    ax.axvspan(0, TOK, color=C["gpu_idle"], alpha=0.6, zorder=0)
    ax.text(TOK / 2, GPU_Y + 0.7,
            f"GPU IDLE\n({TOK:.0f} ms)",
            ha="center", va="bottom", fontsize=9,
            color=C["idle_txt"], fontweight="bold")

    # ── GPU idle 배경 (디토크나이징) ──
    ax.axvspan(t_ovh_e, t_dtok_e, color=C["gpu_idle"], alpha=0.6, zorder=0)

    # ── GPU 단계 바 ──
    gpu_stages = [
        ("Tokenize\n(CPU only)",  0,       TOK,     C["tokenize"], True),
        ("Vision\nEncoding",      t_vis_s, t_vis_e, C["vision"],   False),
        ("LLM\nPrefill",          t_pre_s, t_pre_e, C["prefill"],  False),
        (f"LLM Decode\n({N_STEPS} steps × 110ms)", t_dec_s, t_dec_e, C["decode"], False),
        ("Flow\nMatching",        t_flo_s, t_flo_e, C["flow"],     False),
        ("Detok",                 t_ovh_e, t_dtok_e, C["detok"],   True),
    ]

    for label, ts, te, color, cpu_only in gpu_stages:
        w = te - ts
        if cpu_only:
            # CPU 전용 — GPU 바를 점선으로 표시 (GPU idle)
            ax.barh(GPU_Y, w, height=BAR_H, left=ts,
                    color=C["gpu_idle"], edgecolor=color,
                    linewidth=2.0, linestyle="--", zorder=2)
            ax.text(ts + w / 2, GPU_Y,
                    "GPU\nIDLE", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
        else:
            ax.barh(GPU_Y, w, height=BAR_H, left=ts,
                    color=color, edgecolor="white",
                    linewidth=1.5, alpha=0.92, zorder=2)
            if w > 200:
                ax.text(ts + w / 2, GPU_Y,
                        f"{label}\n{w:.0f} ms",
                        ha="center", va="center",
                        fontsize=8.5, color="white",
                        fontweight="bold", linespacing=1.35)

    # ── CPU Core 02 파형 ──
    # 토크나이징 구간: Core 02 = ~100% (실제 tokenizer 실행)
    tok_t_synth  = np.array([0.0, TOK])
    tok_c02_synth = np.array([95.0, 95.0])
    ax.fill_between(tok_t_synth, CPU_Y, CPU_Y + tok_c02_synth * SCALE,
                    step="post", color=C["tokenize"],
                    alpha=0.85, zorder=3,
                    label="CPU Core 02 (실측 + 토크나이징 합성)")

    # 추론 구간: 실측 psutil 데이터
    ax.fill_between(cpu_t, CPU_Y, CPU_Y + cpu_c02 * SCALE,
                    step="post", color=C["cpu_main"],
                    alpha=0.80, zorder=3)

    # CPU Y축 보조선
    for pct, label in [(0, "0%"), (50, "50%"), (100, "100%")]:
        y_line = CPU_Y + pct * SCALE
        ax.axhline(y_line, color=C["cpu_main"],
                   linewidth=0.5, linestyle=":", alpha=0.4,
                   xmin=0.025, xmax=0.985)
        ax.text(TOTAL + 10, y_line, label,
                va="center", fontsize=7.5, color=C["cpu_main"])

    # ── 단계 경계선 ──
    for t_mark, lbl in [
        (TOK,    "추론\n시작"),
        (t_vis_e,"Prefill\n시작"),
        (t_pre_e,"Decode\n시작"),
        (t_dec_e,"Flow\n시작"),
        (t_flo_e,"완료"),
    ]:
        ax.axvline(t_mark, color=C["dark"], linewidth=0.9,
                   linestyle="--", alpha=0.3, zorder=1)

    # ── 핵심 어노테이션 ──
    # 토크나이징: CPU running, GPU idle
    ax.annotate(
        f"① TOKENIZE {TOK:.0f}ms\nCPU: apply_chat_template()\n   image patch 토큰화\nGPU: 완전 idle",
        xy=(TOK / 2, GPU_Y - 0.35),
        xytext=(TOK / 2, 0.4),
        fontsize=8.5, color=C["tokenize"], fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C["tokenize"], lw=1.4),
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=C["tokenize"], alpha=0.95),
    )

    # Decode: GPU running, CPU mostly idle (kernel launch만)
    ax.annotate(
        f"② DECODE {DEC:.0f}ms\nGPU: {N_STEPS}× forward pass (110ms/step)\nCPU: kernel launch → 대기 반복\n     (~419ms CPU overhead)",
        xy=(t_dec_s + DEC * 0.5, GPU_Y + 0.35),
        xytext=(t_dec_s + DEC * 0.3, GPU_Y + 1.6),
        fontsize=8.5, color=C["decode"], fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C["decode"], lw=1.4),
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=C["decode"], alpha=0.95),
    )

    # 범례
    handles = [
        mpatches.Patch(color=C["tokenize"], label=f"Tokenize (CPU only, GPU idle) — {TOK:.0f} ms"),
        mpatches.Patch(color=C["vision"],   label=f"Vision Encoding — {VIS:.0f} ms"),
        mpatches.Patch(color=C["prefill"],  label=f"LLM Prefill — {PRE:.0f} ms"),
        mpatches.Patch(color=C["decode"],   label=f"LLM Decode — {DEC:.0f} ms"),
        mpatches.Patch(color=C["flow"],     label=f"Flow Matching — {FLO:.0f} ms"),
        mpatches.Patch(color=C["cpu_main"], label="CPU Core 02 utilization (%)"),
    ]
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.0, 0.98), ncol=2,
              fontsize=8.5, framealpha=0.92)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 2 — 토크나이징 줌인 (0ms ~ VIS+200ms)
    # ══════════════════════════════════════════════════════════════════════════
    ax2 = ax_zoom1
    ZOOM1_END = TOK + VIS + 150
    ax2.set_xlim(-10, ZOOM1_END)
    ax2.set_ylim(-0.3, 3.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Time (ms)", fontsize=10)
    ax2.set_title(
        "Fig B  Zoom: Tokenization → Vision Start\n"
        "GPU는 토크나이징 동안 완전히 idle",
        fontsize=10, fontweight="bold"
    )

    G2Y = 2.3
    C2Y = 0.8
    S2  = 1.0 / 100.0

    # GPU idle 배경
    ax2.axvspan(0, TOK, color=C["gpu_idle"], alpha=0.7, zorder=0,
                label="GPU idle zone")
    ax2.text(TOK / 2, G2Y + 0.55,
             "◀ GPU IDLE ▶\n(no CUDA kernel)",
             ha="center", va="bottom", fontsize=9,
             color=C["idle_txt"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc=C["gpu_idle"], ec=C["idle_txt"],
                       alpha=0.9, linewidth=1.5))

    # GPU 바
    # 토크나이징 (GPU idle 표시)
    ax2.barh(G2Y, TOK, height=0.55, left=0,
             color=C["gpu_idle"], edgecolor=C["tokenize"],
             linewidth=2.5, linestyle="--", zorder=2)
    ax2.text(TOK / 2, G2Y, "GPU IDLE", ha="center", va="center",
             fontsize=9, color=C["tokenize"], fontweight="bold")
    # Vision
    ax2.barh(G2Y, VIS + 150, height=0.55, left=TOK,
             color=C["vision"], edgecolor="white",
             linewidth=1.5, alpha=0.92, zorder=2)
    ax2.text(TOK + (VIS + 150) / 2, G2Y, "Vision Encoding (GPU running)",
             ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")

    # CPU 파형
    ax2.fill_between([0, TOK], C2Y, [C2Y + 95 * S2, C2Y + 95 * S2],
                     step="post", color=C["tokenize"], alpha=0.85, zorder=3,
                     label="CPU Core 02")
    mask = cpu_t <= ZOOM1_END
    ax2.fill_between(cpu_t[mask], C2Y, C2Y + cpu_c02[mask] * S2,
                     step="post", color=C["cpu_main"], alpha=0.80, zorder=3)

    # Y축 보조
    for pct, lbl in [(0, "0%"), (100, "100%")]:
        ax2.axhline(C2Y + pct * S2, color=C["cpu_main"],
                    linewidth=0.6, linestyle=":", alpha=0.4)
        ax2.text(ZOOM1_END + 5, C2Y + pct * S2, lbl,
                 va="center", fontsize=7.5, color=C["cpu_main"])

    # 경계선
    ax2.axvline(TOK, color=C["dark"], linewidth=1.5,
                linestyle="-", alpha=0.6, zorder=4)
    ax2.text(TOK + 5, 0.1,
             f"GPU 시작\n(t={TOK:.0f}ms)",
             fontsize=8.5, color=C["dark"], fontweight="bold")

    # 어노테이션
    ax2.annotate(
        f"apply_chat_template()\n이미지 패치 + 텍스트 → 토큰\n{TOK:.0f} ms  |  Core 02 ≈ 95%",
        xy=(TOK / 2, C2Y + 95 * S2),
        xytext=(TOK / 2, 2.9),
        fontsize=8.5, color=C["tokenize"], fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C["tokenize"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec=C["tokenize"], alpha=0.95),
    )
    ax2.annotate(
        "Vision 시작과 동시에\nCPU 활용률 급감\n(GPU가 주도권 가져감)",
        xy=(TOK + 150, C2Y + 10 * S2),
        xytext=(TOK + 300, C2Y + 60 * S2),
        fontsize=8.5, color=C["vision"],
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C["vision"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec=C["vision"], alpha=0.95),
    )

    ax2.text(-8, G2Y, "GPU", ha="right", va="center",
             fontsize=9, fontweight="bold", color=C["dark"])
    ax2.text(-8, C2Y + 0.5, "CPU\nCore 02", ha="right", va="center",
             fontsize=9, fontweight="bold", color=C["cpu_main"])

    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 3 — Decode 1 step 마이크로 패턴
    # ══════════════════════════════════════════════════════════════════════════
    ax3 = ax_zoom2

    # 측정값: step당 110ms, CPU serial overhead 419ms / 16 steps = 26ms/step
    STEP_MS  = 110.03
    CPU_LAUNCH_MS = 419.0 / N_STEPS    # ~26ms
    GPU_COMPUTE_MS = STEP_MS - CPU_LAUNCH_MS  # ~84ms

    N_SHOW = 3   # 3 step 반복 표시
    TOTAL3 = STEP_MS * N_SHOW + 20

    ax3.set_xlim(-5, TOTAL3)
    ax3.set_ylim(-0.3, 3.8)
    ax3.set_yticks([])
    ax3.set_xlabel("Time within Decode phase (ms)", fontsize=10)
    ax3.set_title(
        "Fig C  Zoom: LLM Decode — 1 Step Micro-pattern\n"
        f"Measured: 110 ms/step  |  CPU launch: ~{CPU_LAUNCH_MS:.0f} ms  |  GPU compute: ~{GPU_COMPUTE_MS:.0f} ms",
        fontsize=10, fontweight="bold"
    )

    G3Y = 2.3
    C3Y = 0.8

    for step_i in range(N_SHOW):
        t_step_start = step_i * STEP_MS

        # ── CPU kernel launch 구간 (step 앞부분) ──
        t_launch_s = t_step_start
        t_launch_e = t_step_start + CPU_LAUNCH_MS

        # CPU bar (launch 구간)
        ax3.barh(C3Y, CPU_LAUNCH_MS, height=0.55,
                 left=t_launch_s,
                 color=C["cpu_main"], edgecolor="white",
                 linewidth=1.0, alpha=0.92, zorder=3)
        # GPU idle (launch 구간)
        ax3.barh(G3Y, CPU_LAUNCH_MS, height=0.55,
                 left=t_launch_s,
                 color=C["gpu_idle"], edgecolor=C["decode"],
                 linewidth=1.5, linestyle="--", zorder=2)
        if step_i == 0:
            ax3.text(t_launch_s + CPU_LAUNCH_MS / 2, G3Y,
                     "GPU\nwait", ha="center", va="center",
                     fontsize=7.5, color=C["decode"], fontweight="bold")
            ax3.text(t_launch_s + CPU_LAUNCH_MS / 2, C3Y,
                     "kernel\nlaunch", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold")

        # ── GPU compute 구간 (step 뒷부분) ──
        t_gpu_s = t_launch_e
        t_gpu_e = t_step_start + STEP_MS

        # GPU bar (compute)
        ax3.barh(G3Y, GPU_COMPUTE_MS, height=0.55,
                 left=t_gpu_s,
                 color=C["decode"], edgecolor="white",
                 linewidth=1.0, alpha=0.92, zorder=3)
        if step_i == 0:
            ax3.text(t_gpu_s + GPU_COMPUTE_MS / 2, G3Y,
                     "GPU compute\n(22GB 읽기)",
                     ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold")

        # CPU idle (compute 구간)
        ax3.barh(C3Y, GPU_COMPUTE_MS, height=0.55,
                 left=t_gpu_s,
                 color=C["cpu_idle"], edgecolor=C["cpu_main"],
                 linewidth=1.0, linestyle=":", zorder=2)
        if step_i == 0:
            ax3.text(t_gpu_s + GPU_COMPUTE_MS / 2, C3Y,
                     "CPU wait\n(pthread_cond_wait)",
                     ha="center", va="center",
                     fontsize=7.5, color=C["cpu_main"], fontweight="bold")

        # 스텝 경계선
        ax3.axvline(t_step_start + STEP_MS, color=C["dark"],
                    linewidth=1.0, linestyle="--", alpha=0.25, zorder=1)
        ax3.text(t_step_start + STEP_MS / 2, 3.4,
                 f"Step {step_i + 1}\n{STEP_MS:.0f} ms",
                 ha="center", va="center", fontsize=8,
                 color=C["dark"],
                 bbox=dict(boxstyle="round,pad=0.2",
                           fc="white", ec=C["dark"], alpha=0.7))

    # "... (16회 반복)" 표시
    ax3.text(TOTAL3 - 15, G3Y, "...\n(총 16회)", ha="center",
             va="center", fontsize=9, color=C["decode"],
             fontweight="bold")

    # 핵심 설명 어노테이션
    ax3.annotate(
        f"CPU가 이 구간에서\nnext-token 샘플링 +\nCUDA kernel launch\n→ ~{CPU_LAUNCH_MS:.0f} ms",
        xy=(CPU_LAUNCH_MS / 2, C3Y + 0.3),
        xytext=(CPU_LAUNCH_MS + 30, C3Y + 1.3),
        fontsize=8.5, color=C["cpu_main"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["cpu_main"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec=C["cpu_main"], alpha=0.95),
    )
    ax3.annotate(
        f"nsys에서 보인\n'pthread_cond_wait'\n= CPU가 GPU 완료를\n기다리는 구간\n→ ~{GPU_COMPUTE_MS:.0f} ms",
        xy=(CPU_LAUNCH_MS + GPU_COMPUTE_MS * 0.5, C3Y - 0.3),
        xytext=(CPU_LAUNCH_MS + GPU_COMPUTE_MS * 0.5 + 10, 0.1),
        fontsize=8.5, color=C["idle_txt"], fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C["idle_txt"], lw=1.3),
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec=C["idle_txt"], alpha=0.95),
    )

    ax3.text(-4, G3Y, "GPU", ha="right", va="center",
             fontsize=9, fontweight="bold", color=C["dark"])
    ax3.text(-4, C3Y, "CPU\nCore 02", ha="right", va="center",
             fontsize=9, fontweight="bold", color=C["cpu_main"])

    # 범례
    leg_handles = [
        mpatches.Patch(color=C["cpu_main"],  label=f"CPU active (kernel launch) ~{CPU_LAUNCH_MS:.0f} ms"),
        mpatches.Patch(color=C["cpu_idle"],  label=f"CPU idle (pthread_cond_wait) ~{GPU_COMPUTE_MS:.0f} ms"),
        mpatches.Patch(color=C["decode"],    label=f"GPU compute (22GB 가중치 읽기)"),
        mpatches.Patch(color=C["gpu_idle"],  label="GPU idle (CPU kernel launch 대기)"),
    ]
    ax3.legend(handles=leg_handles, loc="upper right",
               fontsize=8, framealpha=0.92)

    ax3.spines["left"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ── 전체 제목 ─────────────────────────────────────────────────────────────
    fig.suptitle(
        "Alpamayo 1.5  CPU-GPU Pipeline Analysis  |  Jetson AGX Thor (BF16)\n"
        "\"GPU가 무엇을 할 때 CPU가 기다리고, CPU가 무엇을 할 때 GPU가 기다리는가\"",
        fontsize=13, fontweight="bold", y=0.98
    )

    out = OUT_DIR / "fig_pipeline_cpu_gpu.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK]  {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[Load] profiling_results/ 데이터 로드...")
    raw, cpu_raw = load()
    print(f"       {len(raw)} runs, CPU raw: {len(cpu_raw)} runs")
    print("[Draw] CPU-GPU Pipeline Figure 생성 중...")
    out = draw(raw, cpu_raw)
    print(f"\n완료: {out}")
    print("\n교수님 발표 포인트:")
    print("  Fig A: 전체 흐름 — 토크나이징(빨강) 동안 GPU bar가 점선(idle)")
    print("  Fig B: 줌인 — 토크나이징 끝나는 순간 CPU 파형이 급락, GPU 시작")
    print("  Fig C: Decode micro-pattern — pthread_cond_wait의 정체가 무엇인지")


if __name__ == "__main__":
    main()
