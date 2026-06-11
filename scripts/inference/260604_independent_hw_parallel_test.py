"""
Thor iGPU — 독립 하드웨어 유닛 병렬 실행 테스트
================================================

테스트 1: DMA Engine ∥ GPU SM
  - cudaMemcpyAsync (DMA copy engine) + matmul (SM) 동시 실행
  - CPU pinned memory → GPU async copy 사용 (가장 직접적인 DMA 엔진 사용법)
  - 이것이 RT-Swap / Demand Layering의 실제 메커니즘

테스트 2: CPU ∥ GPU SM
  - NumPy 행렬 연산 (CPU) + CUDA matmul (GPU) 비동기 동시 실행
  - CUDA 커널은 기본적으로 CPU 비동기 → 둘이 겹치는지 확인

비교 결과 해석:
  speedup > 1.2× → 진짜 병렬 실행 (독립 하드웨어 유닛 효과)
  speedup ≈ 1.0× → 직렬화 (하드웨어가 묶여 있음)
"""

import time
import math
import json
import os
import logging

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda"
N_WARMUP = 3
N_REPEAT = 7


def mean(lst):
    return sum(lst) / len(lst)


def pct(base, new):
    return (base / new - 1) * 100  # 양수 = 빨라짐


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸: 타이밍
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gpu_time(fn, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """GPU 커널 시간 (CUDA event 기반, 정밀)."""
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1))
    return times


def wall_time(fn, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """Wall-clock 시간 (CPU perf_counter 기반)."""
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트 1: DMA Engine ∥ GPU SM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_dma_vs_sm(copy_gb: float, mat_size: int, n_matmul: int, label: str):
    """
    DMA copy(stream_copy) ∥ SM matmul(stream_compute) 병렬 테스트.

    DMA 메커니즘:
      - src_pinned: CPU 고정 메모리 (pin_memory=True)
      - dst_gpu:    GPU 버퍼
      - dst_gpu.copy_(src_pinned, non_blocking=True)
        → CUDA copy engine (DMA) 사용, SM 독립적으로 동작 가능
      - iGPU unified memory라도 copy engine은 SM과 다른 하드웨어 유닛

    RT-Swap / Demand Layering 적용 시나리오:
      - copy_gb ≈ 1 layer 가중치 크기 (22GB/28layers ≈ 785MB)
      - n_matmul ≈ 1 layer 연산량

    기대:
      병렬 → wall ≈ max(dma, sm), speedup > 1.2×
      직렬 → wall ≈ dma + sm,   speedup ≈ 1.0×
    """
    logger.info(f"\n{'─'*60}")
    logger.info(f"  [DMA ∥ SM] {label}")
    logger.info(f"  DMA copy : {copy_gb:.2f} GB  (CPU pinned → GPU)")
    logger.info(f"  SM matmul: [{mat_size}×{mat_size}] × {n_matmul}회")
    logger.info(f"{'─'*60}")

    # 버퍼 준비
    n_elems = int(copy_gb * 1e9 / 2)  # BF16 = 2 bytes
    src_cpu = torch.randn(n_elems, dtype=torch.bfloat16).pin_memory()   # CPU pinned
    dst_gpu = torch.empty(n_elems, dtype=torch.bfloat16, device=DEVICE)  # GPU
    A = torch.randn(mat_size, mat_size, dtype=torch.bfloat16, device=DEVICE)

    stream_copy    = torch.cuda.Stream()
    stream_compute = torch.cuda.Stream()

    def dma_only():
        with torch.cuda.stream(stream_copy):
            dst_gpu.copy_(src_cpu, non_blocking=True)
        torch.cuda.synchronize()

    def sm_only():
        with torch.cuda.stream(stream_compute):
            C = A
            for _ in range(n_matmul):
                C = torch.mm(C, A)
        torch.cuda.synchronize()

    def dma_then_sm():
        with torch.cuda.stream(stream_copy):
            dst_gpu.copy_(src_cpu, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream_copy)
        with torch.cuda.stream(stream_compute):
            C = A
            for _ in range(n_matmul):
                C = torch.mm(C, A)
        torch.cuda.synchronize()

    def dma_parallel_sm():
        # DMA와 SM을 동시에 디스패치
        with torch.cuda.stream(stream_copy):
            dst_gpu.copy_(src_cpu, non_blocking=True)
        with torch.cuda.stream(stream_compute):
            C = A
            for _ in range(n_matmul):
                C = torch.mm(C, A)
        torch.cuda.synchronize()

    t_dma  = wall_time(dma_only)
    t_sm   = wall_time(sm_only)
    t_ser  = wall_time(dma_then_sm)
    t_par  = wall_time(dma_parallel_sm)

    m_dma  = mean(t_dma)
    m_sm   = mean(t_sm)
    m_ser  = mean(t_ser)
    m_par  = mean(t_par)

    theory_ser = m_dma + m_sm
    theory_par = max(m_dma, m_sm)
    speedup    = m_ser / m_par

    logger.info(f"  DMA only          : {m_dma:.1f}ms")
    logger.info(f"  SM only           : {m_sm:.1f}ms")
    logger.info(f"  Serial (DMA→SM)   : {m_ser:.1f}ms  (이론={theory_ser:.0f}ms)")
    logger.info(f"  Parallel (DMA∥SM) : {m_par:.1f}ms  (이론={theory_par:.0f}ms)")
    logger.info(f"")
    logger.info(f"  직렬 overhead: {m_ser/theory_ser:.3f}×  (1.0이면 완전 직렬)")
    logger.info(f"  병렬 speedup : {speedup:.3f}×  (직렬 대비)")
    logger.info(f"  최대 가능    : {theory_ser/theory_par:.2f}×")
    logger.info(f"")

    if speedup > 1.3:
        verdict = f"✅ DMA Engine ∥ SM 병렬 확인! ({speedup:.2f}×)"
    elif speedup > 1.1:
        verdict = f"⚠️  부분 병렬 ({speedup:.2f}×)"
    else:
        verdict = f"❌ 직렬화 ({speedup:.2f}×) — DMA Engine도 SM과 직렬"

    logger.info(f"  → {verdict}")

    del src_cpu, dst_gpu, A
    torch.cuda.empty_cache()

    return {
        "label": label,
        "type": "DMA_vs_SM",
        "ms_dma": round(m_dma, 2),
        "ms_sm": round(m_sm, 2),
        "ms_serial": round(m_ser, 2),
        "ms_parallel": round(m_par, 2),
        "speedup": round(speedup, 3),
        "max_possible": round(theory_ser / theory_par, 2),
        "verdict": verdict,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트 2: CPU ∥ GPU SM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_cpu_vs_gpu(cpu_mat_size: int, gpu_mat_size: int, n_gpu: int, label: str):
    """
    CPU numpy 연산 ∥ GPU CUDA 커널 병렬 테스트.

    CUDA 커널은 launch 후 Python이 즉시 리턴 (비동기).
    따라서 GPU 커널 dispatch 후 CPU 연산을 하면 이론상 항상 겹쳐야 한다.

    Python 관점:
      torch.mm() → GPU에 커널 전달 후 즉시 리턴 (non-blocking)
      이후 CPU 코드 실행 → GPU와 CPU가 동시에 실행

    측정:
      cpu_only    : numpy 연산만
      gpu_only    : matmul만 (synchronize 포함)
      gpu_async + cpu_work + sync:
        → GPU 커널 dispatch → CPU numpy 실행 → GPU sync
        → wall ≈ max(gpu, cpu)이면 병렬 확인
    """
    logger.info(f"\n{'─'*60}")
    logger.info(f"  [CPU ∥ GPU] {label}")
    logger.info(f"  CPU: numpy matmul [{cpu_mat_size}×{cpu_mat_size}]")
    logger.info(f"  GPU: torch.mm [{gpu_mat_size}×{gpu_mat_size}] × {n_gpu}회")
    logger.info(f"{'─'*60}")

    A_gpu = torch.randn(gpu_mat_size, gpu_mat_size, dtype=torch.bfloat16, device=DEVICE)
    A_cpu = np.random.randn(cpu_mat_size, cpu_mat_size).astype(np.float32)

    def cpu_only():
        B = A_cpu @ A_cpu  # numpy CPU matmul
        _ = float(B.sum())  # force evaluate
        return B

    def gpu_only():
        C = A_gpu
        for _ in range(n_gpu):
            C = torch.mm(C, A_gpu)
        torch.cuda.synchronize()
        return C

    def gpu_async_then_cpu_then_sync():
        """
        GPU 커널 dispatch (non-blocking) → CPU 작업 → GPU sync
        Wall time이 max(gpu, cpu)이면 진짜 병렬
        """
        # GPU 커널 dispatch (즉시 리턴, GPU는 백그라운드에서 실행 시작)
        C = A_gpu
        for _ in range(n_gpu):
            C = torch.mm(C, A_gpu)
        # GPU 아직 실행 중 — CPU가 numpy 실행
        B = A_cpu @ A_cpu
        _ = float(B.sum())
        # GPU 완료 대기
        torch.cuda.synchronize()
        return C, B

    # 개별 시간 측정 (wall clock)
    t_cpu_times = []
    for _ in range(N_WARMUP):
        cpu_only()
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        cpu_only()
        t_cpu_times.append((time.perf_counter() - t0) * 1000)

    t_gpu_times = wall_time(gpu_only)

    # 병렬 시간 측정
    for _ in range(N_WARMUP):
        gpu_async_then_cpu_then_sync()
        torch.cuda.synchronize()
    t_par_times = []
    for _ in range(N_REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_async_then_cpu_then_sync()
        torch.cuda.synchronize()
        t_par_times.append((time.perf_counter() - t0) * 1000)

    m_cpu = mean(t_cpu_times)
    m_gpu = mean(t_gpu_times)
    m_par = mean(t_par_times)

    theory_ser = m_cpu + m_gpu
    theory_par = max(m_cpu, m_gpu)
    speedup    = (m_cpu + m_gpu) / m_par   # serial 시간 대비 병렬 시간

    # 더 직관적인 지표: 이론 병렬값(max)과 얼마나 가까운가
    overlap_ratio = (theory_ser - m_par) / (theory_ser - theory_par + 1e-9)
    overlap_ratio = max(0.0, min(1.0, overlap_ratio))

    logger.info(f"  CPU only          : {m_cpu:.1f}ms")
    logger.info(f"  GPU only          : {m_gpu:.1f}ms")
    logger.info(f"  Parallel (wall)   : {m_par:.1f}ms")
    logger.info(f"  이론 직렬         : {theory_ser:.1f}ms")
    logger.info(f"  이론 병렬 (max)   : {theory_par:.1f}ms")
    logger.info(f"")
    logger.info(f"  speedup (vs serial): {speedup:.3f}×")
    logger.info(f"  overlap ratio      : {overlap_ratio:.1%}  (0%=완전직렬, 100%=완전병렬)")
    logger.info(f"")

    if overlap_ratio > 0.8:
        verdict = f"✅ CPU ∥ GPU 병렬 확인! (overlap {overlap_ratio:.0%})"
    elif overlap_ratio > 0.4:
        verdict = f"⚠️  부분 병렬 (overlap {overlap_ratio:.0%})"
    else:
        verdict = f"❌ 직렬화 (overlap {overlap_ratio:.0%})"

    logger.info(f"  → {verdict}")

    del A_gpu, A_cpu
    torch.cuda.empty_cache()

    return {
        "label": label,
        "type": "CPU_vs_GPU",
        "ms_cpu": round(m_cpu, 2),
        "ms_gpu": round(m_gpu, 2),
        "ms_parallel": round(m_par, 2),
        "ms_theory_serial": round(theory_ser, 2),
        "ms_theory_parallel": round(theory_par, 2),
        "speedup": round(speedup, 3),
        "overlap_ratio": round(overlap_ratio, 3),
        "verdict": verdict,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 테스트 3: cudaMemPrefetchAsync ∥ GPU SM (RT-Swap 방식)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_prefetch_vs_sm(prefetch_gb: float, mat_size: int, n_matmul: int, label: str):
    """
    cudaMemPrefetchAsync ∥ GPU SM compute (RT-Swap / Demand Layering 방식).

    RT-Swap 시나리오:
      - 모델 가중치 = unified memory 버퍼 (DRAM)
      - 현재 레이어 연산 (SM) 중 다음 레이어 가중치를 GPU로 prefetch
      - cudaMemPrefetchAsync는 unified memory를 GPU-preferred로 마이그레이션

    unified memory에서 cudaMemPrefetchAsync 효과:
      - 물리 메모리는 동일하지만 page migration hint로 GPU 접근 최적화
      - 또는 L2 cache warm-up 효과 (제한적: L2=32MB << layer size)
    """
    logger.info(f"\n{'─'*60}")
    logger.info(f"  [Prefetch ∥ SM] {label}")
    logger.info(f"  Prefetch : {prefetch_gb:.2f} GB (cudaMemPrefetchAsync)")
    logger.info(f"  SM matmul: [{mat_size}×{mat_size}] × {n_matmul}회")
    logger.info(f"{'─'*60}")

    n_elems    = int(prefetch_gb * 1e9 / 2)
    buf        = torch.empty(n_elems, dtype=torch.bfloat16, device=DEVICE)
    A          = torch.randn(mat_size, mat_size, dtype=torch.bfloat16, device=DEVICE)
    stream_pf  = torch.cuda.Stream()
    stream_sm  = torch.cuda.Stream()

    # cudaMemPrefetchAsync via PyTorch runtime
    try:
        cudart = torch.cuda.cudart()
        has_prefetch = hasattr(cudart, 'cudaMemPrefetchAsync')
    except Exception:
        has_prefetch = False
        logger.warning("  cudaMemPrefetchAsync 사용 불가 — copy_() 로 대체")

    def prefetch_only():
        if has_prefetch:
            with torch.cuda.stream(stream_pf):
                cudart.cudaMemPrefetchAsync(
                    buf.data_ptr(), buf.nbytes,
                    torch.cuda.current_device(), stream_pf.cuda_stream
                )
        else:
            with torch.cuda.stream(stream_pf):
                _ = buf.clone()
        torch.cuda.synchronize()

    def sm_only():
        with torch.cuda.stream(stream_sm):
            C = A
            for _ in range(n_matmul):
                C = torch.mm(C, A)
        torch.cuda.synchronize()

    def serial_pf_then_sm():
        prefetch_only()
        sm_only()

    def parallel_pf_sm():
        if has_prefetch:
            with torch.cuda.stream(stream_pf):
                cudart.cudaMemPrefetchAsync(
                    buf.data_ptr(), buf.nbytes,
                    torch.cuda.current_device(), stream_pf.cuda_stream
                )
        else:
            with torch.cuda.stream(stream_pf):
                _ = buf.clone()
        with torch.cuda.stream(stream_sm):
            C = A
            for _ in range(n_matmul):
                C = torch.mm(C, A)
        torch.cuda.synchronize()

    t_pf  = wall_time(prefetch_only)
    t_sm  = wall_time(sm_only)
    t_ser = wall_time(serial_pf_then_sm)
    t_par = wall_time(parallel_pf_sm)

    m_pf  = mean(t_pf)
    m_sm  = mean(t_sm)
    m_ser = mean(t_ser)
    m_par = mean(t_par)

    theory_ser = m_pf + m_sm
    theory_par = max(m_pf, m_sm)
    speedup    = m_ser / m_par

    logger.info(f"  Prefetch only     : {m_pf:.1f}ms")
    logger.info(f"  SM only           : {m_sm:.1f}ms")
    logger.info(f"  Serial (Pf→SM)    : {m_ser:.1f}ms")
    logger.info(f"  Parallel (Pf∥SM)  : {m_par:.1f}ms  (이론={theory_par:.0f}ms)")
    logger.info(f"  speedup           : {speedup:.3f}×  (최대가능={theory_ser/theory_par:.2f}×)")

    if speedup > 1.3:
        verdict = f"✅ Prefetch ∥ SM 병렬 확인! ({speedup:.2f}×)"
    elif speedup > 1.1:
        verdict = f"⚠️  부분 병렬 ({speedup:.2f}×)"
    else:
        verdict = f"❌ 직렬화 ({speedup:.2f}×)"

    logger.info(f"  → {verdict}")

    del buf, A
    torch.cuda.empty_cache()

    return {
        "label": label,
        "type": "Prefetch_vs_SM",
        "ms_prefetch": round(m_pf, 2),
        "ms_sm": round(m_sm, 2),
        "ms_serial": round(m_ser, 2),
        "ms_parallel": round(m_par, 2),
        "speedup": round(speedup, 3),
        "max_possible": round(theory_ser / theory_par, 2),
        "verdict": verdict,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    logger.info("=" * 60)
    logger.info("  Thor iGPU — 독립 하드웨어 유닛 병렬 실행 테스트")
    logger.info("=" * 60)
    logger.info(f"  CUDA  : {torch.version.cuda}")
    logger.info(f"  SM    : {torch.cuda.get_device_capability()}")
    logger.info(f"  GPU   : {torch.cuda.get_device_name()}")
    logger.info(f"  Mem   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    logger.info(f"  CPU   : {os.cpu_count()} cores")

    results = []

    # ── 테스트 1: DMA Engine ∥ GPU SM ─────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("  TEST BLOCK 1: DMA Engine ∥ GPU SM")
    logger.info(f"{'='*60}")

    # 소형: 빠른 DMA, 빠른 SM → 겹침 창이 작음
    results.append(test_dma_vs_sm(
        copy_gb=1.0, mat_size=2048, n_matmul=5,
        label="Small  (1GB copy, 2K matmul×5)"
    ))

    # 중형: Alpamayo VE ≈ ViT 수준
    results.append(test_dma_vs_sm(
        copy_gb=4.0, mat_size=4096, n_matmul=5,
        label="Medium (4GB copy, 4K matmul×5)"
    ))

    # 실제 수준: 1 LM layer 가중치(~0.8GB) + layer 연산
    # 22GB / 28 layers ≈ 785MB per layer
    results.append(test_dma_vs_sm(
        copy_gb=0.8, mat_size=4096, n_matmul=10,
        label="Layer  (0.8GB copy = 1 LM layer, 4K matmul×10)"
    ))

    # ── 테스트 2: CPU ∥ GPU SM ─────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("  TEST BLOCK 2: CPU ∥ GPU SM")
    logger.info(f"{'='*60}")

    # CPU 가벼움, GPU 무거움
    results.append(test_cpu_vs_gpu(
        cpu_mat_size=1000, gpu_mat_size=4096, n_gpu=10,
        label="CPU light (1K numpy) + GPU heavy (4K×10)"
    ))

    # CPU 무거움, GPU 가벼움
    results.append(test_cpu_vs_gpu(
        cpu_mat_size=3000, gpu_mat_size=2048, n_gpu=5,
        label="CPU heavy (3K numpy) + GPU light (2K×5)"
    ))

    # 균형: CPU ≈ GPU 시간
    results.append(test_cpu_vs_gpu(
        cpu_mat_size=2000, gpu_mat_size=4096, n_gpu=5,
        label="Balanced (2K numpy ≈ 4K GPU×5)"
    ))

    # ── 테스트 3: cudaMemPrefetchAsync ∥ GPU SM ────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("  TEST BLOCK 3: cudaMemPrefetchAsync ∥ GPU SM  (RT-Swap 방식)")
    logger.info(f"{'='*60}")

    results.append(test_prefetch_vs_sm(
        prefetch_gb=0.8, mat_size=4096, n_matmul=10,
        label="1-layer (0.8GB prefetch, 4K matmul×10)"
    ))

    results.append(test_prefetch_vs_sm(
        prefetch_gb=4.0, mat_size=4096, n_matmul=5,
        label="Multi-layer (4GB prefetch, 4K matmul×5)"
    ))

    # ── 최종 요약 ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("  최종 요약")
    logger.info(f"{'='*60}")

    for r in results:
        t = r["type"]
        if t == "DMA_vs_SM":
            logger.info(f"  [{t:15s}] {r['label']:<40} speedup={r['speedup']:.3f}×  {r['verdict'].split('(')[0].strip()}")
        elif t == "CPU_vs_GPU":
            logger.info(f"  [{t:15s}] {r['label']:<40} overlap={r['overlap_ratio']:.0%}    {r['verdict'].split('(')[0].strip()}")
        else:
            logger.info(f"  [{t:15s}] {r['label']:<40} speedup={r['speedup']:.3f}×  {r['verdict'].split('(')[0].strip()}")

    logger.info(f"\n  해석:")
    logger.info(f"    DMA ∥ SM : speedup > 1.2× → layer-level RT-Swap 파이프라인 가능")
    logger.info(f"    CPU ∥ GPU: overlap > 80%  → CPU 전처리 파이프라인 가능")
    logger.info(f"    Prefetch ∥ SM: speedup > 1.2× → cudaMemPrefetchAsync 기반 최적화 가능")

    out_dir  = "profiling_results/260604_independent_hw_parallel_test"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
