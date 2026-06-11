"""
Thor iGPU CUDA Stream Concurrency Test
=======================================
목적: iGPU가 물리적으로 CUDA multi-stream 병렬 실행을 지원하는지 확인.

하드웨어 직렬화 vs PyTorch 소프트웨어 직렬화를 분리하기 위한 최소 단위 테스트.

테스트 구성:
  (A) Compute-bound  : matmul 반복  → Tensor Core 집중 사용
  (B) Memory-bound   : large clone  → DRAM 대역폭 집중 사용

기대 결과:
  병렬 가능 : wall_time ≈ max(compute, memory)   → speedup > 1.3×
  직렬화    : wall_time ≈ compute + memory        → speedup ≈ 1.0×
"""

import time
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda"


def measure_serial(compute_fn, memory_fn, n_warmup=3, n_repeat=5):
    """Compute 후 Memory를 순차 실행 (기준선)."""
    # warmup
    for _ in range(n_warmup):
        compute_fn()
        memory_fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compute_fn()
        memory_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def measure_compute_only(compute_fn, n_warmup=3, n_repeat=5):
    for _ in range(n_warmup):
        compute_fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compute_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def measure_memory_only(memory_fn, n_warmup=3, n_repeat=5):
    for _ in range(n_warmup):
        memory_fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        memory_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def measure_parallel(stream1, stream2, compute_fn, memory_fn, n_warmup=3, n_repeat=5):
    """두 stream에서 동시 실행. allocator 간섭 없는 순수 스트림 테스트."""
    for _ in range(n_warmup):
        with torch.cuda.stream(stream1):
            compute_fn()
        with torch.cuda.stream(stream2):
            memory_fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream1):
            compute_fn()
        with torch.cuda.stream(stream2):
            memory_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def measure_parallel_preallocated(stream1, stream2, compute_fn, memory_fn_preallocated,
                                   n_warmup=3, n_repeat=5):
    """
    사전 할당된 버퍼를 사용하는 병렬 테스트.
    allocator cross-stream sync를 완전히 제거.
    """
    for _ in range(n_warmup):
        with torch.cuda.stream(stream1):
            compute_fn()
        with torch.cuda.stream(stream2):
            memory_fn_preallocated()
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream1):
            compute_fn()
        with torch.cuda.stream(stream2):
            memory_fn_preallocated()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def mean(lst):
    return sum(lst) / len(lst)


def run_test(label, matrix_size, clone_gb, n_iters=10):
    """
    단일 테스트 케이스 실행.

    matrix_size : matmul 행렬 크기 (N×N)
    clone_gb    : clone 텐서 크기 (GB)
    """
    logger.info(f"\n{'─'*60}")
    logger.info(f"  TEST: {label}")
    logger.info(f"  matmul  : [{matrix_size}×{matrix_size}] BF16")
    logger.info(f"  clone   : {clone_gb:.2f} GB BF16")
    logger.info(f"{'─'*60}")

    # ── 텐서 준비 ──────────────────────────────────────────────────────────
    A = torch.randn(matrix_size, matrix_size, device=DEVICE, dtype=torch.bfloat16)

    # clone용 텐서 (clone_gb GB)
    clone_elems = int(clone_gb * 1e9 / 2)   # BF16 = 2 bytes/elem
    B = torch.randn(clone_elems, device=DEVICE, dtype=torch.bfloat16)

    # 사전 할당 버퍼 (allocator interference 제거용)
    B_dst = torch.empty_like(B)

    # ── 함수 정의 ──────────────────────────────────────────────────────────
    def compute_fn():
        # matmul n_iters회 → Tensor Core 집중
        C = A
        for _ in range(n_iters):
            C = torch.mm(C, A)
        return C

    def memory_fn():
        # 대용량 clone → DRAM 대역폭 집중
        return B.clone()

    def memory_fn_preallocated():
        # 사전 할당 버퍼에 copy → allocator 호출 없음
        B_dst.copy_(B)
        return B_dst

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # ── 개별 측정 ──────────────────────────────────────────────────────────
    t_compute = measure_compute_only(compute_fn, n_warmup=3, n_repeat=5)
    t_memory  = measure_memory_only(memory_fn, n_warmup=3, n_repeat=5)
    t_serial  = measure_serial(compute_fn, memory_fn, n_warmup=3, n_repeat=5)
    t_parallel = measure_parallel(stream1, stream2, compute_fn, memory_fn, n_warmup=3, n_repeat=5)
    t_parallel_prealloc = measure_parallel_preallocated(
        stream1, stream2, compute_fn, memory_fn_preallocated, n_warmup=3, n_repeat=5
    )

    mc = mean(t_compute)
    mm = mean(t_memory)
    ms = mean(t_serial)
    mp = mean(t_parallel)
    mpp = mean(t_parallel_prealloc)

    # 이론값
    serial_theory   = mc + mm
    parallel_theory = max(mc, mm)

    logger.info(f"  Compute only      : {mc:.1f}ms")
    logger.info(f"  Memory only       : {mm:.1f}ms")
    logger.info(f"  Serial (C→M)      : {ms:.1f}ms  (theory={serial_theory:.0f}ms)")
    logger.info(f"  Parallel (alloc)  : {mp:.1f}ms  (theory_parallel={parallel_theory:.0f}ms)")
    logger.info(f"  Parallel (prealloc): {mpp:.1f}ms")
    logger.info(f"")
    logger.info(f"  직렬 가속비 (C+M vs serial measured)    : {serial_theory/ms:.3f}×")
    logger.info(f"  병렬 가속비 (serial vs parallel alloc)  : {ms/mp:.3f}×")
    logger.info(f"  병렬 가속비 (serial vs parallel prealloc): {ms/mpp:.3f}×")
    logger.info(f"  최대 가능 가속비 (C+M vs max(C,M))      : {serial_theory/parallel_theory:.2f}×")

    # 판정
    speedup_alloc   = ms / mp
    speedup_prealloc = ms / mpp

    def judge(speedup, threshold=1.2):
        if speedup > threshold:
            return f"✅ 병렬 실행 확인 ({speedup:.2f}×)"
        elif speedup > 1.05:
            return f"⚠️  부분 병렬 ({speedup:.2f}×)"
        else:
            return f"❌ 직렬화 ({speedup:.2f}×)"

    logger.info(f"")
    logger.info(f"  [판정 with alloc]    : {judge(speedup_alloc)}")
    logger.info(f"  [판정 prealloc]      : {judge(speedup_prealloc)}")

    # allocator 간섭 여부
    alloc_diff = mp - mpp
    if abs(alloc_diff) > 20:
        logger.info(f"  → allocator 간섭 의심: {alloc_diff:+.1f}ms "
                    f"(alloc={mp:.1f}ms vs prealloc={mpp:.1f}ms)")
    else:
        logger.info(f"  → allocator 간섭 없음 (차이 {alloc_diff:+.1f}ms)")

    del A, B, B_dst
    torch.cuda.empty_cache()

    return {
        "label": label,
        "ms_compute": round(mc, 1),
        "ms_memory": round(mm, 1),
        "ms_serial": round(ms, 1),
        "ms_parallel_alloc": round(mp, 1),
        "ms_parallel_prealloc": round(mpp, 1),
        "speedup_alloc": round(speedup_alloc, 3),
        "speedup_prealloc": round(speedup_prealloc, 3),
        "max_possible_speedup": round(serial_theory / parallel_theory, 2),
    }


def main():
    logger.info("="*60)
    logger.info("  Thor iGPU CUDA Stream Concurrency Test")
    logger.info("="*60)
    logger.info(f"  CUDA: {torch.version.cuda}")
    logger.info(f"  SM:   {torch.cuda.get_device_capability()}")
    logger.info(f"  GPU:  {torch.cuda.get_device_name()}")
    logger.info(f"  Mem:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    results = []

    # ── Test 1: 소형 (빠름, SM 경합 낮음) ─────────────────────────────────
    r = run_test(
        label="Small  (matmul 2K×2K, clone 1GB)",
        matrix_size=2048,
        clone_gb=1.0,
        n_iters=5,
    )
    results.append(r)

    # ── Test 2: 중형 (Alpamayo VE 수준) ───────────────────────────────────
    r = run_test(
        label="Medium (matmul 4K×4K, clone 4GB)",
        matrix_size=4096,
        clone_gb=4.0,
        n_iters=5,
    )
    results.append(r)

    # ── Test 3: 대형 (Decode 1step ≈ 22GB DRAM) ────────────────────────────
    r = run_test(
        label="Large  (matmul 4K×4K, clone 10GB)",
        matrix_size=4096,
        clone_gb=10.0,
        n_iters=10,
    )
    results.append(r)

    # ── 최종 요약 ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("  최종 요약")
    logger.info(f"{'='*60}")
    logger.info(f"  {'케이스':<38} {'alloc':>8} {'prealloc':>10} {'최대가능':>10}")
    logger.info(f"  {'-'*70}")
    for r in results:
        logger.info(
            f"  {r['label']:<38} "
            f"{r['speedup_alloc']:>7.2f}×  "
            f"{r['speedup_prealloc']:>8.2f}×  "
            f"{r['max_possible_speedup']:>8.2f}×"
        )

    logger.info(f"\n  해석 기준:")
    logger.info(f"    speedup > 1.2× → 병렬 실행 하드웨어 지원 확인")
    logger.info(f"    alloc < prealloc → PyTorch allocator가 병렬화 방해")
    logger.info(f"    둘 다 ≈1.0× → 하드웨어 자체가 직렬화")

    import json, os
    out_dir = "profiling_results/260604_cuda_stream_concurrency_test"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
