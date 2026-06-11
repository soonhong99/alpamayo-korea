"""
report_generator.py
────────────────────
profile_alpamayo.py + tegrastats_monitor.py 결과를 받아
논문·보고서 수준의 시각화 및 정량 지표를 생성한다.

출력:
  profiling_results/
  ├── fig1_latency_breakdown.png      ← 단계별 레이턴시 (수평 Waterfall)
  ├── fig2_latency_distribution.png   ← Box plot + violin (p50/p95/p99)
  ├── fig3_hardware_timeline.png      ← GPU/CPU/RAM 시계열 (tegrastats)
  ├── fig4_memory_breakdown.png       ← 메모리 구성 파이차트
  ├── fig5_minADE_comparison.png      ← minADE@K / minFDE@K 비교
  └── profiling_report.md             ← 교수 보고용 마크다운 요약

사용법:
  python scripts/profiling/report_generator.py --input_dir profiling_results/
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # 헤드리스 환경 (Thor SSH)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────
# 공통 스타일
# ──────────────────────────────────────────────
COLORS = {
    "vision":   "#4C72B0",
    "prefill":  "#DD8452",
    "decode":   "#55A868",
    "action":   "#C44E52",
    "post":     "#8172B2",
    "gpu":      "#4C72B0",
    "cpu":      "#55A868",
    "ram":      "#DD8452",
    "emc":      "#8172B2",
}
STAGE_LABELS = {
    "vision_encoding": "Vision\nEncoding",
    "llm_prefill":     "LLM\nPrefill",
    "llm_decode":      "LLM Decode\n(CoC 생성)",
    "action_expert":   "Action\nExpert",
    "postprocess":     "Post-\nprocess",
}
STAGE_COLOR_KEYS = {
    "vision_encoding": "vision",
    "llm_prefill":     "prefill",
    "llm_decode":      "decode",
    "action_expert":   "action",
    "postprocess":     "post",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

STAGES = ["vision_encoding", "llm_prefill", "llm_decode", "action_expert", "postprocess"]


# ──────────────────────────────────────────────
# Fig 1: Waterfall 레이턴시 분해
# ──────────────────────────────────────────────
def fig1_latency_breakdown(summary: dict, output_dir: Path):
    t = summary["timing_ms"]
    pct = summary["breakdown_pct"]

    means = [t[s]["mean"] for s in STAGES]
    stds  = [t[s]["std"]  for s in STAGES]
    labels = [STAGE_LABELS[s] for s in STAGES]
    colors = [COLORS[STAGE_COLOR_KEYS[s]] for s in STAGES]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 수평 누적 막대
    left = 0
    bar_h = 0.5
    for i, (m, s, label, color) in enumerate(zip(means, stds, labels, colors)):
        ax.barh(0, m, left=left, height=bar_h, color=color,
                edgecolor="white", linewidth=1.2)
        # 텍스트: 충분히 넓으면 안에 표시
        if m > 3:
            ax.text(left + m / 2, 0, f"{m:.1f}ms\n({pct[STAGES[i]]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        left += m

    # 목표선 (100ms)
    ax.axvline(100, color="red", linestyle="--", linewidth=1.5, label="100ms 목표")

    # 범례
    patches = [mpatches.Patch(color=colors[i], label=labels[i].replace("\n", " "))
               for i in range(len(STAGES))]
    ax.legend(handles=patches + [
        mpatches.Patch(color="red", label="100ms 목표")
    ], loc="upper right", fontsize=9)

    total = t["total_gpu"]["mean"]
    ax.set_xlabel("누적 시간 (ms)")
    ax.set_title(
        f"Alpamayo 1.5 추론 단계별 레이턴시 분해\n"
        f"(총 GPU 시간: {total:.1f}ms | {summary['environment']['device']})",
        pad=12
    )
    ax.set_yticks([])
    ax.set_xlim(0, max(total * 1.15, 110))
    ax.grid(axis="x", alpha=0.3)

    out = output_dir / "fig1_latency_breakdown.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Report] 저장: {out}")


# ──────────────────────────────────────────────
# Fig 2: 분포 (Box + Strip)
# ──────────────────────────────────────────────
def fig2_latency_distribution(summary: dict, output_dir: Path):
    t = summary["timing_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 왼쪽: 단계별 mean ± std 막대
    ax = axes[0]
    means = [t[s]["mean"] for s in STAGES]
    stds  = [t[s]["std"]  for s in STAGES]
    p95s  = [t[s]["p95"]  for s in STAGES]
    colors = [COLORS[STAGE_COLOR_KEYS[s]] for s in STAGES]
    x = np.arange(len(STAGES))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", width=0.6)
    ax.scatter(x, p95s, marker="D", color="black", zorder=5,
               s=40, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGES], fontsize=9)
    ax.set_ylabel("레이턴시 (ms)")
    ax.set_title("단계별 레이턴시 (mean ± std, ◆=p95)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 오른쪽: total GPU 분포 누적분포 (CDF)
    ax2 = axes[1]
    total_mean = t["total_gpu"]["mean"]
    total_std  = t["total_gpu"]["std"]
    total_p50  = t["total_gpu"]["p50"]
    total_p95  = t["total_gpu"]["p95"]
    total_p99  = t["total_gpu"]["p99"]

    # 가우시안 근사로 CDF 시각화
    x_vals = np.linspace(max(0, total_mean - 4*total_std),
                         total_mean + 4*total_std, 300)
    from scipy.special import erf  # noqa
    cdf = 0.5 * (1 + erf((x_vals - total_mean) / (total_std * np.sqrt(2))))
    ax2.plot(x_vals, cdf * 100, color=COLORS["gpu"], linewidth=2)
    ax2.axvline(100, color="red", linestyle="--", label="100ms 목표")
    ax2.axvline(total_p50, color="green", linestyle=":", label=f"p50={total_p50:.1f}ms")
    ax2.axvline(total_p95, color="orange", linestyle=":", label=f"p95={total_p95:.1f}ms")
    ax2.axvline(total_p99, color="purple", linestyle=":", label=f"p99={total_p99:.1f}ms")
    ax2.fill_between(x_vals, cdf * 100, where=(x_vals <= 100),
                     alpha=0.15, color="green", label="목표 달성 구간")
    ax2.set_xlabel("Total GPU 레이턴시 (ms)")
    ax2.set_ylabel("누적 확률 (%)")
    ax2.set_title("Total GPU 레이턴시 누적분포 (CDF)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Alpamayo 1.5 레이턴시 분포 분석 — {summary['environment']['device']}",
                 fontsize=13, y=1.02)
    out = output_dir / "fig2_latency_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Report] 저장: {out}")


# ──────────────────────────────────────────────
# Fig 3: Hardware 시계열 (tegrastats)
# ──────────────────────────────────────────────
def fig3_hardware_timeline(tegrastats_path: Path, output_dir: Path):
    if not tegrastats_path.exists():
        print(f"[Report] tegrastats 데이터 없음, Fig 3 건너뜀")
        return

    with open(tegrastats_path) as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        return

    t0 = records[0]["timestamp"]
    times = [(r["timestamp"] - t0) for r in records]

    gpu_util = [r.get("gpu_util_pct", 0) for r in records]
    cpu_avg  = [r.get("cpu_avg_load_pct", 0) for r in records]
    ram_pct  = [r.get("ram_used_pct", 0) for r in records]
    emc_util = [r.get("emc_util_pct", 0) for r in records]
    power_w  = [r.get("total_power_mw", 0) / 1000 for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # GPU + CPU 이용률
    ax = axes[0]
    ax.plot(times, gpu_util, color=COLORS["gpu"], linewidth=1.5, label="GPU 이용률 (%)")
    ax.plot(times, cpu_avg,  color=COLORS["cpu"], linewidth=1.5, label="CPU 평균 이용률 (%)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("이용률 (%)")
    ax.set_title("GPU / CPU 이용률")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # RAM + EMC
    ax2 = axes[1]
    ax2.plot(times, ram_pct,  color=COLORS["ram"], linewidth=1.5, label="RAM 사용률 (%)")
    ax2.plot(times, emc_util, color=COLORS["emc"], linewidth=1.5,
             linestyle="--", label="메모리 컨트롤러 이용률 (%)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("이용률 (%)")
    ax2.set_title("메모리 사용률")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    # 전력
    ax3 = axes[2]
    ax3.plot(times, power_w, color="#E377C2", linewidth=1.5)
    ax3.fill_between(times, power_w, alpha=0.2, color="#E377C2")
    s = data["summary"]
    ax3.axhline(s["avg_total_power_w"], color="red", linestyle="--",
                label=f"평균 {s['avg_total_power_w']:.1f}W")
    ax3.set_ylabel("소비 전력 (W)")
    ax3.set_xlabel("시간 (초)")
    ax3.set_title("총 소비 전력")
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig.suptitle("Alpamayo 1.5 추론 중 하드웨어 사용 현황 (Jetson AGX Thor)",
                 fontsize=13)
    fig.tight_layout()
    out = output_dir / "fig3_hardware_timeline.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Report] 저장: {out}")


# ──────────────────────────────────────────────
# Fig 4: 메모리 구성 파이
# ──────────────────────────────────────────────
def fig4_memory_breakdown(summary: dict, output_dir: Path):
    mem = summary["memory_mb"]
    param_mb  = mem["param_mem_mb"]
    peak_mb   = mem["peak_gpu_mb"]["mean"]
    kvcache_mb = max(0, peak_mb - param_mb) * 0.6    # 추정
    activ_mb   = max(0, peak_mb - param_mb) * 0.4    # 추정
    total_mb   = 128 * 1024  # Thor 전체 128GB

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: GPU 피크 메모리 구성
    ax = axes[0]
    sizes = [param_mb, kvcache_mb, activ_mb]
    labels = [
        f"모델 파라미터\n{param_mb/1024:.1f} GB",
        f"KV 캐시 (추정)\n{kvcache_mb/1024:.1f} GB",
        f"활성화값 (추정)\n{activ_mb/1024:.1f} GB",
    ]
    colors_pie = [COLORS["vision"], COLORS["prefill"], COLORS["decode"]]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_pie,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title(f"GPU 피크 메모리 구성\n총 {peak_mb/1024:.1f} GB")

    # 오른쪽: Thor 전체 메모리 중 점유 비율
    ax2 = axes[1]
    used_gb   = peak_mb / 1024
    avail_gb  = total_mb / 1024 - used_gb
    bars = ax2.barh(["사용", "여유"], [used_gb, avail_gb],
                    color=[COLORS["prefill"], "#AAAAAA"], height=0.4)
    ax2.set_xlabel("메모리 (GB)")
    ax2.set_title(f"Thor 128GB 통합 메모리 점유\n(추론 피크 기준)")
    ax2.set_xlim(0, total_mb / 1024)
    for bar, val in zip(bars, [used_gb, avail_gb]):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f} GB ({val/total_mb*1024*100:.1f}%)",
                 va="center")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Alpamayo 1.5 메모리 사용 분석", fontsize=13)
    fig.tight_layout()
    out = output_dir / "fig4_memory_breakdown.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Report] 저장: {out}")


# ──────────────────────────────────────────────
# Fig 5: minADE@K / minFDE@K
# ──────────────────────────────────────────────
def fig5_trajectory_metrics(summary: dict, output_dir: Path):
    traj = summary.get("trajectory_metrics")
    if not traj:
        print("[Report] 궤적 지표 없음, Fig 5 건너뜀")
        return

    K_vals = sorted(set(
        int(k.split("@")[1]) for k in traj if "@" in k
    ))
    ade_vals = [traj.get(f"minADE@{k}", None) for k in K_vals]
    fde_vals = [traj.get(f"minFDE@{k}", None) for k in K_vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(K_vals))
    w = 0.35
    b1 = ax.bar(x - w/2, ade_vals, width=w, color=COLORS["gpu"], label="minADE@K")
    b2 = ax.bar(x + w/2, fde_vals, width=w, color=COLORS["action"], label="minFDE@K")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}m", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_vals])
    ax.set_ylabel("오차 (m)")
    ax.set_title("궤적 예측 품질: minADE@K / minFDE@K\n(낮을수록 우수, 검증 데이터 50개 기준)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(filter(None, fde_vals)) * 1.3, 1.0))

    out = output_dir / "fig5_minADE_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Report] 저장: {out}")


# ──────────────────────────────────────────────
# 마크다운 보고서 생성
# ──────────────────────────────────────────────
def generate_markdown_report(summary: dict, output_dir: Path):
    t = summary["timing_ms"]
    m = summary["memory_mb"]
    env = summary["environment"]
    pct = summary["breakdown_pct"]
    traj = summary.get("trajectory_metrics", {})

    today = datetime.now().strftime("%Y-%m-%d")
    report = f"""# Alpamayo 1.5 추론 프로파일링 보고서

**작성일**: {today}
**플랫폼**: {env['device']} ({env['sm_version']})
**프레임워크**: PyTorch {env['torch_version']} / CUDA {env['cuda_version']}
**정밀도**: {env['dtype']} | Attention: {env['attn_impl']}

---

## 1. 실험 환경

| 항목 | 사양 |
|---|---|
| 보드 | NVIDIA Jetson AGX Thor |
| GPU | {env['device']} ({env['sm_version']}) |
| 총 메모리 | {env['total_mem_gb']} GB (CPU+GPU 통합) |
| 입력 형태 | {env['input_shape']} |
| Waypoint 수 | {env['num_waypoints']}개 (6.4초, 10Hz) |
| 측정 횟수 | {summary['num_runs']}회 (웜업 {summary['warmup_runs']}회 제외) |

---

## 2. 레이턴시 분석 (ms)

| 단계 | mean | ±std | p50 | p95 | p99 | 비율 |
|---|---:|---:|---:|---:|---:|---:|
| Vision Encoding | {t['vision_encoding']['mean']:.1f} | {t['vision_encoding']['std']:.1f} | {t['vision_encoding']['p50']:.1f} | {t['vision_encoding']['p95']:.1f} | {t['vision_encoding']['p99']:.1f} | {pct['vision_encoding']:.1f}% |
| LLM Prefill | {t['llm_prefill']['mean']:.1f} | {t['llm_prefill']['std']:.1f} | {t['llm_prefill']['p50']:.1f} | {t['llm_prefill']['p95']:.1f} | {t['llm_prefill']['p99']:.1f} | {pct['llm_prefill']:.1f}% |
| LLM Decode (CoC) | {t['llm_decode']['mean']:.1f} | {t['llm_decode']['std']:.1f} | {t['llm_decode']['p50']:.1f} | {t['llm_decode']['p95']:.1f} | {t['llm_decode']['p99']:.1f} | {pct['llm_decode']:.1f}% |
| Action Expert | {t['action_expert']['mean']:.1f} | {t['action_expert']['std']:.1f} | {t['action_expert']['p50']:.1f} | {t['action_expert']['p95']:.1f} | {t['action_expert']['p99']:.1f} | {pct['action_expert']:.1f}% |
| Postprocess | {t['postprocess']['mean']:.1f} | {t['postprocess']['std']:.1f} | {t['postprocess']['p50']:.1f} | {t['postprocess']['p95']:.1f} | {t['postprocess']['p99']:.1f} | {pct['postprocess']:.1f}% |
| **Total GPU** | **{t['total_gpu']['mean']:.1f}** | **{t['total_gpu']['std']:.1f}** | **{t['total_gpu']['p50']:.1f}** | **{t['total_gpu']['p95']:.1f}** | **{t['total_gpu']['p99']:.1f}** | 100% |
| Total Wall | {t['total_wall']['mean']:.1f} | {t['total_wall']['std']:.1f} | — | — | — | — |

> **100ms 목표 달성률**: {summary['latency_target_met_pct']:.1f}%

---

## 3. 메모리 사용량

| 항목 | 값 |
|---|---|
| 모델 파라미터 메모리 | {m['param_mem_mb']/1024:.1f} GB |
| 추론 피크 GPU 메모리 | {m['peak_gpu_mb']['mean']/1024:.1f} GB (±{m['peak_gpu_mb']['std']/1024:.2f} GB) |
| KV 캐시 추정 | {max(0, m['peak_gpu_mb']['mean'] - m['param_mem_mb']) * 0.6 / 1024:.1f} GB |
| Thor 잔여 메모리 | {128 - m['peak_gpu_mb']['mean']/1024:.1f} GB |

---

## 4. 궤적 예측 품질 (minADE / minFDE)

| 지표 | K=1 | K=5 | K=10 |
|---|---:|---:|---:|
| minADE (m) | {traj.get('minADE@1', 'N/A')} | {traj.get('minADE@5', 'N/A')} | {traj.get('minADE@10', 'N/A')} |
| minFDE (m) | {traj.get('minFDE@1', 'N/A')} | {traj.get('minFDE@5', 'N/A')} | {traj.get('minFDE@10', 'N/A')} |

> minADE@K: K번 샘플링 중 실제 궤적과 가장 가까운 것의 평균 거리
> minFDE@K: K번 샘플링 중 최종 위치 최소 오차

---

## 5. 병목 분석 및 최적화 방향

| 병목 단계 | 원인 | 예상 최적화 |
|---|---|---|
| LLM Prefill ({pct['llm_prefill']:.0f}%) | 카메라 4개×4프레임 → 수천 개 KV 쌍 동시 생성 | KV 캐시 청크 prefill, 이미지 토큰 압축 |
| Vision Encoding ({pct['vision_encoding']:.0f}%) | 고해상도 이미지 → ViT 패치 연산 | 토큰 풀링, 해상도 단계별 감소 |
| LLM Decode ({pct['llm_decode']:.0f}%) | 자기회귀 토큰 생성 (1토큰씩) | Speculative Decoding, CoC 길이 제한 |
| Action Expert ({pct['action_expert']:.0f}%) | Flow Matching ODE 스텝 수 | ODE 스텝 수 감소 (4→2) |

---

## 6. 시각화 파일

- `fig1_latency_breakdown.png` — 단계별 레이턴시 Waterfall
- `fig2_latency_distribution.png` — 레이턴시 분포 (Box + CDF)
- `fig3_hardware_timeline.png` — CPU/GPU/RAM 시계열
- `fig4_memory_breakdown.png` — 메모리 구성 분석
- `fig5_minADE_comparison.png` — 궤적 품질 지표

---

*본 보고서는 report_generator.py에 의해 자동 생성됨.*
"""

    out = output_dir / "profiling_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Report] 마크다운 보고서 저장: {out}")


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Alpamayo 프로파일링 리포트 생성기")
    parser.add_argument("--input_dir", type=str, default="profiling_results",
                        help="profile_alpamayo.py 출력 디렉토리")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary.json"

    if not summary_path.exists():
        print(f"[Report] summary.json 없음: {summary_path}")
        print("  먼저 profile_alpamayo.py를 실행하세요.")
        return

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    print(f"[Report] 데이터 로드 완료: {summary_path}")

    # scipy 없으면 Fig 2 CDF 스킵
    try:
        import scipy  # noqa
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("[Report] scipy 없음, CDF 근사 생략 (pip install scipy)")

    fig1_latency_breakdown(summary, input_dir)
    if has_scipy:
        fig2_latency_distribution(summary, input_dir)
    fig3_hardware_timeline(input_dir / "tegrastats.json", input_dir)
    fig4_memory_breakdown(summary, input_dir)
    fig5_trajectory_metrics(summary, input_dir)
    generate_markdown_report(summary, input_dir)

    print("\n[Report] 모든 그래프 생성 완료")
    print(f"  출력 위치: {input_dir.resolve()}")


if __name__ == "__main__":
    main()
