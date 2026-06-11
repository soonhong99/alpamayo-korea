#!/usr/bin/env python3
"""
EXP-4: GPU 용량 스케일링 (메모리 제한 → 성능 비선형성 검증)
────────────────────────────────────────────────────────────────────
실험 계획서: docs/260515_mig_pipeline_experiment_plan.md
가설 H3: GPU 메모리/컴퓨트 제한 시 성능이 비선형적으로 감소한다
         (가중치가 메모리를 벗어나는 순간 급격한 cliff 발생)

핵심 질문:
  Q3. 가중치(22 GB)가 GPU VRAM 안에 완전히 들어가는 동안은 성능 평탄,
      넘는 순간 paging/swap으로 급격히 느려지는가?

Thor 특성:
  - 통합 메모리: CPU/GPU 동일 LPDDR5X 128 GB
  - 모델 22 GB → 총 메모리의 17% → OOM 없음
  - 하지만 GPU L2 캐시(≈ 수십 MB) + SM 제한은 효과 있음

실험 방법:
  1. 텐서 크기(배치 크기 / 시퀀스 길이)를 변화시켜 GPU 메모리 점유 변화
  2. 각 메모리 점유 수준에서 추론 시간 측정
  3. Phase별 (Decode BW-bound vs Vision compute-bound) 차이 분석

  MIG 슬라이스를 사용하면 메모리 용량 제한 가능:
  → EXP-4-MIG: 각 슬라이스에서 실행, 메모리 용량 = 슬라이스 비율 × 총 메모리

실행:
  python3 ~/alpamayo1.5/scripts/profiling/260515_exp4_capacity_scaling.py

출력:
  profiling_results/260515_exp4/capacity_scaling.json
  profiling_results/260515_exp4/capacity_scaling.md
  profiling_results/260515_exp4/capacity_scaling_fig.png

작성일: 2026-05-15
"""

import argparse
import json
import logging
import statistics
import gc
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("profiling_results/260515_exp4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPEAT = 5
N_WARMUP = 2

# 메모리 점유 수준 (총 GPU 메모리 대비 %)
# Thor 128 GB 통합 메모리에서 의미 있는 변화를 보려면
# 모델+KV cache+활성화 메모리를 포함한 총 점유율 변화 필요
MEMORY_LEVELS = [
    {"label": "1 GB",  "tensor_gb": 1.0},
    {"label": "4 GB",  "tensor_gb": 4.0},
    {"label": "8 GB",  "tensor_gb": 8.0},
    {"label": "16 GB", "tensor_gb": 16.0},
    {"label": "22 GB", "tensor_gb": 22.0},   # 모델 전체 크기
    {"label": "32 GB", "tensor_gb": 32.0},   # 모델 + KV cache
    {"label": "48 GB", "tensor_gb": 48.0},   # 여유 있는 큰 배치
]

# batch_size 변화로 점유 모사
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


@dataclass
class CapacityResult:
    label: str
    tensor_gb: float
    allocated_gb: float = 0.0
    decode_ms: float = 0.0
    decode_std: float = 0.0
    gemm_ms: float = 0.0
    gemm_std: float = 0.0
    oom: bool = False


def allocate_tensor(size_gb: float, device: str = "cuda") -> Optional[torch.Tensor]:
    """size_gb 크기의 텐서를 GPU에 할당. OOM이면 None 반환."""
    try:
        n_elements = int(size_gb * 1e9 / 2)  # float16 = 2 bytes
        t = torch.empty(n_elements, dtype=torch.float16, device=device)
        return t
    except torch.cuda.OutOfMemoryError:
        return None


def try_import_optional():
    """Optional import."""
    try:
        from typing import Optional
        return Optional
    except ImportError:
        return None


# typing.Optional 처리
from typing import Optional


def measure_decode_with_background(bg_gb: float, n: int = N_REPEAT) -> tuple[float, float, bool]:
    """
    bg_gb GB의 텐서가 이미 메모리를 점유한 상태에서 GEMV(Decode 모사) 실행.
    → 메모리 용량이 가득 찼을 때 성능 저하 확인.
    """
    # 백그라운드 텐서 할당 (메모리 점유)
    bg_tensor = None
    actual_gb = 0.0
    try:
        n_el = int(bg_gb * 1e9 / 2)
        bg_tensor = torch.empty(n_el, dtype=torch.float16, device="cuda")
        actual_gb = bg_tensor.element_size() * bg_tensor.nelement() / 1e9
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        log.warning(f"  OOM at {bg_gb:.0f} GB background tensor")
        return 0.0, 0.0, True

    # GEMV (Decode 모사: seq=1, d_model=4096 × ratio)
    try:
        d_model = 4096
        W = torch.randn(d_model * 4, d_model, device="cuda", dtype=torch.float16)
        x = torch.randn(d_model, 1, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()

        times = []
        for i in range(n + N_WARMUP):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(32):  # 32 layers
                _ = W @ x
            e1.record()
            torch.cuda.synchronize()
            if i >= N_WARMUP:
                times.append(e0.elapsed_time(e1))

        del W, x
    except torch.cuda.OutOfMemoryError:
        if bg_tensor is not None:
            del bg_tensor
        return 0.0, 0.0, True

    if bg_tensor is not None:
        del bg_tensor
    torch.cuda.empty_cache()
    gc.collect()

    if not times:
        return 0.0, 0.0, False

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, False


def measure_compute_with_background(bg_gb: float, n: int = N_REPEAT) -> tuple[float, float, bool]:
    """
    bg_gb GB의 텐서가 메모리를 점유한 상태에서 GEMM(Vision/Prefill 모사) 실행.
    → Compute-bound phase는 메모리 점유와 무관하게 성능 일정할 것.
    """
    bg_tensor = None
    try:
        n_el = int(bg_gb * 1e9 / 2)
        bg_tensor = torch.empty(n_el, dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return 0.0, 0.0, True

    try:
        A = torch.randn(512, 4096, device="cuda", dtype=torch.float16)
        B = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()

        times = []
        for i in range(n + N_WARMUP):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(24):  # 24 layers (ViT-L)
                _ = A @ B
            e1.record()
            torch.cuda.synchronize()
            if i >= N_WARMUP:
                times.append(e0.elapsed_time(e1))

        del A, B
    except torch.cuda.OutOfMemoryError:
        if bg_tensor is not None:
            del bg_tensor
        return 0.0, 0.0, True

    if bg_tensor is not None:
        del bg_tensor
    torch.cuda.empty_cache()
    gc.collect()

    if not times:
        return 0.0, 0.0, False

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0, False


def run_capacity_scaling() -> list[CapacityResult]:
    """메모리 점유 수준별 Decode(BW) vs Vision(Compute) 성능 측정."""
    results = []

    for level in MEMORY_LEVELS:
        label = level["label"]
        bg_gb = level["tensor_gb"]
        log.info(f"\n  메모리 점유 {label} ({bg_gb:.0f} GB):")

        cr = CapacityResult(label=label, tensor_gb=bg_gb)

        # GPU 여유 메모리 확인
        free_gb, total_gb = [x / 1e9 for x in torch.cuda.mem_get_info()]
        log.info(f"    GPU 여유: {free_gb:.1f}/{total_gb:.1f} GB")

        if bg_gb > free_gb * 0.9:
            log.warning(f"    {label} → GPU 메모리 부족, 스킵")
            cr.oom = True
            results.append(cr)
            continue

        # Decode (BW-bound)
        d_ms, d_std, oom = measure_decode_with_background(bg_gb)
        if oom:
            cr.oom = True
            results.append(cr)
            continue
        cr.decode_ms = d_ms
        cr.decode_std = d_std
        log.info(f"    Decode (GEMV): {d_ms:.1f} ± {d_std:.1f} ms")

        # Vision/Prefill (Compute-bound)
        g_ms, g_std, oom = measure_compute_with_background(bg_gb)
        if oom:
            cr.oom = True
            results.append(cr)
            continue
        cr.gemm_ms = g_ms
        cr.gemm_std = g_std
        log.info(f"    Vision (GEMM): {g_ms:.1f} ± {g_std:.1f} ms")

        # 실제 점유량 (approximated)
        cr.allocated_gb = bg_gb

        results.append(cr)

    return results


def analyze_cliff(results: list[CapacityResult]) -> dict:
    """
    성능 cliff 검출:
    - 인접 측정값 간 시간 변화율이 일정 임계치 초과하면 cliff로 판정
    """
    valid = [r for r in results if not r.oom and r.decode_ms > 0]
    if len(valid) < 2:
        return {"cliff_detected": False, "note": "측정값 부족"}

    decode_times = [r.decode_ms for r in valid]
    labels = [r.label for r in valid]

    # 연속 비율 계산
    ratios = [
        decode_times[i+1] / decode_times[i]
        for i in range(len(decode_times) - 1)
    ]

    # cliff: 인접 비율 > 2.0 (2× 이상 느려짐)
    cliff_idx = None
    for i, ratio in enumerate(ratios):
        if ratio > 2.0:
            cliff_idx = i + 1
            break

    if cliff_idx is not None:
        cliff_label = labels[cliff_idx]
        cliff_gb = valid[cliff_idx].tensor_gb
        interpretation = (
            f"⚠️ Cliff 감지: {labels[cliff_idx-1]} → {cliff_label} 구간에서 "
            f"{ratios[cliff_idx-1]:.1f}× 급격한 성능 저하. "
            f"GPU 캐시/슬라이스 용량 한계({cliff_gb:.0f} GB) 도달."
        )
    else:
        max_ratio = max(ratios) if ratios else 1.0
        interpretation = (
            f"✅ Cliff 없음: 최대 성능 변동 {max_ratio:.2f}×. "
            "Thor 통합 메모리(128 GB)에서 22 GB 모델은 메모리 cliff 없음."
        )

    return {
        "cliff_detected": cliff_idx is not None,
        "cliff_at_label": labels[cliff_idx] if cliff_idx else None,
        "cliff_at_gb": valid[cliff_idx].tensor_gb if cliff_idx else None,
        "transition_ratios": [round(r, 3) for r in ratios],
        "labels": labels,
        "decode_times_ms": [round(r.decode_ms, 1) for r in valid],
        "gemm_times_ms":   [round(r.gemm_ms, 1) for r in valid],
        "interpretation": interpretation,
        "hypothesis_H3_confirmed": cliff_idx is not None,
    }


def make_report(results: list[CapacityResult], cliff: dict) -> str:
    lines = [
        "# EXP-4: GPU 용량 스케일링 실험 결과",
        "",
        "**목적**: 메모리 점유가 증가할 때 성능이 선형적으로 감소하는가, 아니면 cliff가 있는가?",
        "**가설 H3**: 모델 가중치가 GPU 메모리를 초과하는 순간 비선형적 성능 저하 발생",
        "",
        "---",
        "",
        "## 1. 측정 결과",
        "",
        "| 메모리 점유 | Decode(GEMV, ms) | Vision(GEMM, ms) | OOM |",
        "|---|---|---|---|",
    ]
    for r in results:
        oom_tag = "⚠️ OOM" if r.oom else "✅"
        lines.append(
            f"| {r.label} | "
            f"{r.decode_ms:.1f} ± {r.decode_std:.1f} | "
            f"{r.gemm_ms:.1f} ± {r.gemm_std:.1f} | "
            f"{oom_tag} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Cliff 분석",
        "",
        f"**판정**: {cliff['interpretation']}",
        "",
        "| 구간 | 성능 변화 비율 |",
        "|---|---|",
    ]
    for lbl, ratio in zip(
        [f"{cliff['labels'][i]} → {cliff['labels'][i+1]}" for i in range(len(cliff['labels'])-1)],
        cliff["transition_ratios"]
    ):
        flag = "⚠️" if ratio > 2.0 else "✅"
        lines.append(f"| {lbl} | {ratio:.2f}× {flag} |")

    lines += [
        "",
        "---",
        "",
        "## 3. 해석 및 다음 단계",
        "",
        "- **Thor 통합 메모리** (128 GB): 22 GB 모델은 메모리의 17%만 사용 → cliff 가능성 낮음",
        "- **실제 cliff 테스트**: 실제 Alpamayo 1.5 모델 로드 후 배치 크기 증가 실험 필요",
        "- **MIG 슬라이스에서**: 슬라이스 메모리 = 총 / GI 수 → 작은 슬라이스에서 cliff 가능",
        "",
        "참고: docs/260515_mig_pipeline_experiment_plan.md EXP-4 절",
    ]

    return "\n".join(lines)


def plot_results(results: list[CapacityResult], cliff: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib 없음 — 그래프 생략")
        return

    valid = [r for r in results if not r.oom and r.decode_ms > 0]
    if not valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("EXP-4: GPU Capacity Scaling\n(Background memory occupancy vs inference time)",
                 fontsize=12, fontweight="bold")

    labels = [r.label for r in valid]
    decode_ms = [r.decode_ms for r in valid]
    gemm_ms   = [r.gemm_ms   for r in valid]
    x = range(len(labels))

    # 패널 1: 절대 시간
    ax = axes[0]
    ax.plot(x, decode_ms, "o-", color="#e76f51", linewidth=2, markersize=6, label="Decode (BW-bound, GEMV)")
    ax.plot(x, gemm_ms,   "s-", color="#4e9af1", linewidth=2, markersize=6, label="Vision (compute, GEMM)")
    if cliff.get("cliff_detected") and cliff.get("cliff_at_label") in labels:
        ci = labels.index(cliff["cliff_at_label"])
        ax.axvline(x=ci, color="red", linewidth=2, linestyle="--", alpha=0.7, label="Cliff detected")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Absolute Inference Time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 패널 2: 정규화 (1 GB 기준)
    ax = axes[1]
    if decode_ms[0] > 0 and gemm_ms[0] > 0:
        norm_d = [v / decode_ms[0] for v in decode_ms]
        norm_g = [v / gemm_ms[0]   for v in gemm_ms]
        ax.plot(x, norm_d, "o-", color="#e76f51", linewidth=2, markersize=6, label="Decode (normalized)")
        ax.plot(x, norm_g, "s-", color="#4e9af1", linewidth=2, markersize=6, label="Vision (normalized)")
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
        ax.axhline(y=2.0, color="red",  linestyle="--", linewidth=1, alpha=0.5, label="2× threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Normalized time (vs 1 GB)")
    ax.set_title("Relative Performance Degradation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / "capacity_scaling_fig.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"✅ 그래프 저장: {fig_path}")


def main():
    p = argparse.ArgumentParser(description="EXP-4: GPU capacity scaling")
    p.add_argument("--max-gb", type=float, default=48.0,
                   help="최대 배경 텐서 크기 (GB, 기본: 48)")
    args = p.parse_args()

    log.info("=" * 60)
    log.info("EXP-4: GPU 용량 스케일링")
    log.info("=" * 60)

    if not torch.cuda.is_available():
        log.error("CUDA 없음.")
        return

    device_name = torch.cuda.get_device_name(0)
    free_gb, total_gb = [x / 1e9 for x in torch.cuda.mem_get_info()]
    log.info(f"GPU: {device_name}")
    log.info(f"GPU 메모리: 여유={free_gb:.1f}GB, 총={total_gb:.1f}GB")

    # 최대 테스트 크기 클리핑
    global MEMORY_LEVELS
    MEMORY_LEVELS = [l for l in MEMORY_LEVELS if l["tensor_gb"] <= args.max_gb]

    results = run_capacity_scaling()
    cliff = analyze_cliff(results)

    log.info("\n" + "═" * 60)
    log.info("📊 Cliff 분석")
    log.info("═" * 60)
    log.info(f"  {cliff['interpretation']}")

    # JSON
    json_data = {
        "experiment": "EXP-4",
        "device": device_name,
        "gpu_total_gb": total_gb,
        "cliff_analysis": cliff,
        "results": [asdict(r) for r in results],
    }
    json_path = OUT_DIR / "capacity_scaling.json"
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
    log.info(f"\n✅ JSON: {json_path}")

    md = make_report(results, cliff)
    md_path = OUT_DIR / "capacity_scaling.md"
    md_path.write_text(md, encoding="utf-8")
    log.info(f"✅ MD:   {md_path}")

    plot_results(results, cliff)
    log.info("\n실험 완료.")


if __name__ == "__main__":
    main()
