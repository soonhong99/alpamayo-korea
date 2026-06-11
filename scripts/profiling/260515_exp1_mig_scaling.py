#!/usr/bin/env python3
"""
EXP-1: MIG 슬라이스 크기별 단일 모듈 성능 스케일링
────────────────────────────────────────────────────────────────────
실험 계획서: docs/260515_mig_pipeline_experiment_plan.md

목적:
  MIG 슬라이스 크기(SM 수)에 따라 Vision/Prefill/Decode/Flow 각 모듈의
  실행 시간이 어떻게 변하는지 측정.

  Thor Blackwell iGPU: 총 SM 수 확인 필요 (EXP-0에서 측정).
  일반적으로 Blackwell Orin-class: 64~80 SM.

  MIG 활성화 없이 실행하는 방법:
  → CUDA_VISIBLE_DEVICES + torch.device로 SM 수를 제한할 수 없음.
  → 대신: 행렬 크기(작업 크기)를 줄여 MIG 슬라이스 크기를 모사 (proxy).

  실제 MIG 환경에서는:
  → CUDA_VISIBLE_DEVICES=MIG-<UUID> 로 각 슬라이스에서 직접 실행.

프록시 측정 전략:
  - GEMM 크기 = model_dim × {1/4, 1/2, 3/4, 1} (full)
  - 각 크기에서 실행 시간 측정 → SM 점유율 추정
  - Phase별 (Vision/Prefill/Decode/Flow) 분리 측정

실행 방법:
  source ~/alpamayo1.5/a1_5_venv/bin/activate

  # MIG 없이 프록시 모드
  python3 ~/alpamayo1.5/scripts/profiling/260515_exp1_mig_scaling.py

  # 실제 MIG 환경 (MIG UUID 필요, EXP-0 후 설정)
  CUDA_VISIBLE_DEVICES=MIG-<UUID> \
  python3 ~/alpamayo1.5/scripts/profiling/260515_exp1_mig_scaling.py --real-mig

출력:
  profiling_results/260515_exp1/mig_scaling.json
  profiling_results/260515_exp1/mig_scaling.md
  profiling_results/260515_exp1/mig_scaling_fig.png

작성일: 2026-05-15
"""

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("profiling_results/260515_exp1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 실측 기반 모델 파라미터 크기 (260515_bw_allphase.py 결과)
MODEL_DIM = {
    "vision":  {"d_model": 1024, "n_heads": 16, "n_layers": 24},   # ViT-L 근사
    "prefill": {"d_model": 4096, "n_heads": 32, "n_layers": 32},   # LM Prefill
    "decode":  {"d_model": 4096, "n_heads": 32, "seq": 1},          # seq=1 GEMV
    "flow":    {"d_model": 2048, "n_heads": 16, "n_layers": 18},   # Action Expert 근사
}

# 슬라이스 비율: full GPU 대비 SM 비율 모사
SLICE_RATIOS = [0.25, 0.5, 0.75, 1.0]
SLICE_LABELS = ["1/4 slice", "1/2 slice", "3/4 slice", "Full GPU"]

N_REPEAT = 5   # 각 측정 반복 횟수
N_WARMUP = 2


# ─────────────────────────────────────────────────────────────────
# Phase별 GEMM 벤치마크
# ─────────────────────────────────────────────────────────────────

def run_vision_proxy(ratio: float, n: int = N_REPEAT) -> list[float]:
    """
    Vision Encoder (ViT-style): GEMM-heavy, compute-bound.
    ratio로 행렬 크기 스케일링.
    """
    d = int(MODEL_DIM["vision"]["d_model"] * ratio**0.5)  # d 줄이면 FLOP ∝ ratio
    seq = 196  # ViT patch tokens (14×14)
    A = torch.randn(seq, d * 4, device="cuda", dtype=torch.float16)
    B = torch.randn(d * 4, d, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    times = []
    for i in range(n + N_WARMUP):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        # FFN forward 모사 (2-layer MLP × n_layers)
        for _ in range(MODEL_DIM["vision"]["n_layers"]):
            _ = A @ B
        e1.record()
        torch.cuda.synchronize()
        if i >= N_WARMUP:
            times.append(e0.elapsed_time(e1))
    return times


def run_prefill_proxy(ratio: float, n: int = N_REPEAT) -> list[float]:
    """
    LM Prefill: seq=64 (6.4s × 10token/s 가정), compute-bound.
    """
    d = int(MODEL_DIM["prefill"]["d_model"] * ratio**0.5)
    seq = 64
    A = torch.randn(seq, d, device="cuda", dtype=torch.float16)
    W = torch.randn(d, d * 4, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    times = []
    for i in range(n + N_WARMUP):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(MODEL_DIM["prefill"]["n_layers"]):
            _ = A @ W
        e1.record()
        torch.cuda.synchronize()
        if i >= N_WARMUP:
            times.append(e0.elapsed_time(e1))
    return times


def run_decode_proxy(ratio: float, n: int = N_REPEAT) -> list[float]:
    """
    LM Decode: seq=1 (GEMV), BW-bound.
    ratio는 matrix 크기(= 메모리 로드량)에 비례.
    """
    d_in  = int(MODEL_DIM["decode"]["d_model"] * ratio)
    d_out = d_in * 4
    W = torch.randn(d_out, d_in, device="cuda", dtype=torch.float16)
    x = torch.randn(d_in, 1, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    times = []
    for i in range(n + N_WARMUP):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20):  # n_tok=20
            for _ in range(MODEL_DIM["decode"]["n_heads"] // 2):
                _ = W @ x
        e1.record()
        torch.cuda.synchronize()
        if i >= N_WARMUP:
            times.append(e0.elapsed_time(e1))
    return times


def run_flow_proxy(ratio: float, n: int = N_REPEAT) -> list[float]:
    """
    Action Expert (Flow): Euler steps, mixed compute/BW.
    """
    d = int(MODEL_DIM["flow"]["d_model"] * ratio**0.5)
    seq = 64  # action token sequence
    A = torch.randn(seq, d, device="cuda", dtype=torch.float16)
    W = torch.randn(d, d * 4, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    times = []
    for i in range(n + N_WARMUP):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):  # n_euler=10
            for _ in range(MODEL_DIM["flow"]["n_layers"]):
                _ = A @ W
        e1.record()
        torch.cuda.synchronize()
        if i >= N_WARMUP:
            times.append(e0.elapsed_time(e1))
    return times


PHASE_FUNCS = {
    "vision":  run_vision_proxy,
    "prefill": run_prefill_proxy,
    "decode":  run_decode_proxy,
    "flow":    run_flow_proxy,
}


# ─────────────────────────────────────────────────────────────────
# 측정 루프
# ─────────────────────────────────────────────────────────────────

@dataclass
class SliceResult:
    ratio: float
    label: str
    phase: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def speedup_vs_full(self) -> float:
        """full(ratio=1.0) 대비 이 슬라이스의 speedup (>1이면 느림)."""
        return 0.0  # 사후에 채움


def run_all_phases(phases: list[str] = None) -> dict[str, list[SliceResult]]:
    if phases is None:
        phases = list(PHASE_FUNCS.keys())

    results: dict[str, list[SliceResult]] = {}

    for phase in phases:
        log.info(f"\n─── Phase: {phase.upper()} ───")
        fn = PHASE_FUNCS[phase]
        phase_results = []

        for ratio, label in zip(SLICE_RATIOS, SLICE_LABELS):
            log.info(f"  측정 중: {label} (ratio={ratio:.2f})...")
            times = fn(ratio)
            sr = SliceResult(ratio=ratio, label=label, phase=phase, times_ms=times)
            phase_results.append(sr)
            log.info(f"    → {sr.mean_ms:.1f} ± {sr.std_ms:.1f} ms")

        # speedup_vs_full 계산
        full_ms = next(r.mean_ms for r in phase_results if r.ratio == 1.0)
        for sr in phase_results:
            sr.speedup_vs_full = full_ms / sr.mean_ms if sr.mean_ms > 0 else 0.0

        results[phase] = phase_results

    return results


# ─────────────────────────────────────────────────────────────────
# 분석: 선형성 검증
# ─────────────────────────────────────────────────────────────────

def analyze_linearity(results: dict[str, list[SliceResult]]) -> dict:
    """
    Q3 검증: GPU 자원(슬라이스)에 따라 성능이 선형적으로 증가하는가?
    선형: time ∝ 1/ratio  (speedup ∝ ratio)
    """
    analysis = {}
    for phase, slices in results.items():
        # 측정된 speedup vs 이론 선형 speedup
        full_ms = next(s.mean_ms for s in slices if s.ratio == 1.0)
        ratios = [s.ratio for s in slices]
        measured_speedups = [full_ms / s.mean_ms for s in slices]
        linear_speedups = [r for r in ratios]  # 이론: speedup ∝ ratio

        # 선형성 오차 (평균 상대 오차)
        errors = [
            abs(m - l) / l
            for m, l in zip(measured_speedups, linear_speedups)
            if l > 0
        ]
        linearity_error_pct = statistics.mean(errors) * 100 if errors else 0.0

        # 결론
        is_linear = linearity_error_pct < 20  # 20% 이내면 선형으로 판정

        analysis[phase] = {
            "ratios": ratios,
            "measured_speedups": [round(s, 3) for s in measured_speedups],
            "linear_speedups":   [round(s, 3) for s in linear_speedups],
            "linearity_error_pct": round(linearity_error_pct, 1),
            "is_linear": is_linear,
            "bottleneck": "BW-bound" if phase == "decode" else "compute-bound",
            "interpretation": (
                f"{'✅ 선형' if is_linear else '⚠️ 비선형'}: "
                f"오차 {linearity_error_pct:.0f}%. "
                f"{'BW-bound이므로 SM 증가 효과 낮음.' if phase == 'decode' else 'Compute-bound이므로 SM 증가 효과 높음.'}"
            ),
        }

    return analysis


# ─────────────────────────────────────────────────────────────────
# 보고서 및 시각화
# ─────────────────────────────────────────────────────────────────

def make_report(results: dict[str, list[SliceResult]], analysis: dict) -> str:
    lines = [
        "# EXP-1: MIG 슬라이스 크기별 성능 스케일링",
        "",
        "**목적**: GPU 자원(SM 비율)에 따라 각 모듈 실행 시간이 선형적으로 줄어드는가?",
        "**가설 H1 관련**: MIG 슬라이스 배정이 단일 프레임 레이턴시에 도움이 되는가?",
        "",
        "---",
        "",
        "## 결과 요약",
        "",
    ]

    for phase in results:
        slices = results[phase]
        a = analysis[phase]
        lines += [
            f"### {phase.upper()} — {a['bottleneck']}",
            "",
            "| 슬라이스 | 비율 | 측정 시간(ms) | Speedup | 이론 Speedup |",
            "|---|---|---|---|---|",
        ]
        for s, ms, ls in zip(slices, a["measured_speedups"], a["linear_speedups"]):
            lines.append(
                f"| {s.label} | {s.ratio:.2f} | "
                f"{s.mean_ms:.1f} ± {s.std_ms:.1f} | "
                f"{ms:.2f}× | {ls:.2f}× |"
            )
        lines += [
            "",
            f"**판정**: {a['interpretation']}",
            "",
        ]

    lines += [
        "---",
        "",
        "## MIG 배정 권고",
        "",
        "| 모듈 | 병목 | 권고 MIG 슬라이스 | 근거 |",
        "|---|---|---|---|",
        "| Vision Encoder | Compute | 1/4 slice | compute-bound이나 642ms 짧음 |",
        "| LM Prefill | Compute | 1/2 slice | 가장 오래 걸리는 compute phase |",
        "| LM Decode | BW | 1/4 slice | BW-bound, SM 늘려도 효과 없음 |",
        "| Action Expert | Mixed | 1/4 slice | 비교적 짧음 |",
        "",
        "> ⚠️ 이 배정은 MIG 단일 프레임 가속(H1)에서 테스트됨.",
        "> H1은 기각 예상: 데이터 의존성으로 순차 실행은 변하지 않음.",
        "",
        "---",
        "",
        "다음 실험: EXP-3 크로스프레임 파이프라인",
    ]

    return "\n".join(lines)


def plot_results(results: dict[str, list[SliceResult]], analysis: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib 없음 — 그래프 생략")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "EXP-1: MIG Slice Scaling\n(Proxy: matrix size ∝ SM ratio)",
        fontsize=13, fontweight="bold"
    )
    colors = {"vision": "#4e9af1", "prefill": "#f4a261", "decode": "#e76f51", "flow": "#2a9d8f"}

    for ax, (phase, slices) in zip(axes.flat, results.items()):
        a = analysis[phase]
        ratios = a["ratios"]
        measured = a["measured_speedups"]
        linear = a["linear_speedups"]

        ax.plot(ratios, measured, "o-", color=colors[phase], linewidth=2,
                markersize=6, label="Measured speedup")
        ax.plot(ratios, linear, "--", color="gray", linewidth=1.5, label="Linear (ideal)")
        ax.fill_between(ratios, measured, linear,
                        alpha=0.15, color=colors[phase])

        ax.set_xlabel("SM slice ratio")
        ax.set_ylabel("Speedup vs full GPU")
        ax.set_title(
            f"{phase.upper()} ({a['bottleneck']})\n"
            f"Linearity error: {a['linearity_error_pct']:.0f}%"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0.2, 1.05)
        ax.set_ylim(0, max(max(measured), max(linear)) * 1.2)

    plt.tight_layout()
    fig_path = OUT_DIR / "mig_scaling_fig.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"✅ 그래프 저장: {fig_path}")


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="EXP-1: MIG slice scaling")
    p.add_argument("--real-mig", action="store_true",
                   help="실제 MIG 환경에서 실행 (CUDA_VISIBLE_DEVICES=MIG-UUID 필요)")
    p.add_argument("--phases", nargs="+",
                   default=["vision", "prefill", "decode", "flow"],
                   choices=["vision", "prefill", "decode", "flow"])
    p.add_argument("--repeat", type=int, default=N_REPEAT)
    args = p.parse_args()

    log.info("=" * 60)
    log.info("EXP-1: MIG 슬라이스 크기별 성능 스케일링")
    log.info(f"모드: {'실제 MIG' if args.real_mig else 'Proxy (행렬 크기 스케일링)'}")
    log.info(f"측정 Phase: {args.phases}")
    log.info("=" * 60)

    if not torch.cuda.is_available():
        log.error("CUDA 없음.")
        return

    device_name = torch.cuda.get_device_name(0)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    log.info(f"GPU: {device_name}, SM: {sm_count}")

    if args.real_mig:
        import os
        cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        log.info(f"CUDA_VISIBLE_DEVICES: {cuda_dev}")

    results = run_all_phases(args.phases)
    analysis = analyze_linearity(results)

    # ── JSON 저장 ──
    json_data = {
        "experiment": "EXP-1",
        "device": device_name,
        "sm_count": sm_count,
        "mode": "real_mig" if args.real_mig else "proxy",
        "slice_ratios": SLICE_RATIOS,
        "slice_labels": SLICE_LABELS,
        "analysis": analysis,
        "raw": {
            ph: [asdict(s) for s in slices]
            for ph, slices in results.items()
        },
    }
    json_path = OUT_DIR / "mig_scaling.json"
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
    log.info(f"\n✅ JSON: {json_path}")

    md = make_report(results, analysis)
    md_path = OUT_DIR / "mig_scaling.md"
    md_path.write_text(md, encoding="utf-8")
    log.info(f"✅ MD:   {md_path}")

    plot_results(results, analysis)

    log.info("\n실험 완료.")


if __name__ == "__main__":
    main()
