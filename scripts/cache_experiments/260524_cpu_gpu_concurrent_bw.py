"""
260524_cpu_gpu_concurrent_bw.py
================================
CPU + GPU 동시 DRAM 대역폭 측정 (Phase 0 - Step 4)

목적:
  async pipeline 설계에서 핵심 질문:
  "CPU가 prefetch 작업을 하는 동안 GPU의 DRAM BW가 줄어드는가?"

  Thor는 CPU+GPU가 같은 DRAM 공유 → 경쟁 발생 가능.

  측정 시나리오:
    A. GPU 단독 BW (기준선)
    B. GPU + CPU 1코어 동시
    C. GPU + CPU 4코어 동시
    D. GPU + CPU 14코어 동시 (최악)

  결과 해석:
    GPU BW 감소 없음 → Memory controller가 GPU 우선 → CPU prefetch 전략 안전
    GPU BW 감소 있음 → 간섭 있음 → CPU prefetch 사용 조심, GPU prefetch stream이 더 나음

실행:
  python3 260524_cpu_gpu_concurrent_bw.py
"""

import threading
import torch
import numpy as np
import time
import json
import os

RESULT_FILE = os.path.expanduser(
    "~/alpamayo1.5/profiling_results/260524_cpu_gpu_concurrent_bw.json"
)

# ── GPU BW 측정 (CUDA Graph 방식) ─────────────────────────────────────────────
def measure_gpu_bw_with_cuda_graph(size_mb: int = 256, n_replays: int = 100) -> float:
    """
    GPU의 DRAM BW를 CUDA Graph로 측정.
    CUDA Graph: 100번 replay 묶음 → kernel launch overhead 제거.
    """
    n_elem = size_mb * 1024 * 1024 // 4  # float32
    x = torch.zeros(n_elem, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    # CUDA Graph 캡처
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_replays):
            y.copy_(x)

    # 워밍업
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()

    # 측정
    times_ms = []
    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)

    for _ in range(10):
        start_e.record()
        g.replay()
        end_e.record()
        torch.cuda.synchronize()
        times_ms.append(start_e.elapsed_time(end_e))

    # n_replays번 복사 → 단위 복사당 시간
    per_copy_ms = np.median(times_ms) / n_replays
    size_bytes  = n_elem * 4
    bw_gbs      = 2 * size_bytes / (per_copy_ms * 1e-3) / 1e9  # read+write

    del x, y
    return bw_gbs

# ── CPU 워커 (백그라운드에서 계속 DRAM 접근) ──────────────────────────────────
def cpu_worker_loop(stop_event: threading.Event, array_size_mb: int = 256):
    """CPU 코어 하나가 계속 DRAM 읽기/쓰기를 반복."""
    n_elem = array_size_mb * 1024 * 1024 // 4
    src = np.ones(n_elem, dtype=np.float32)
    dst = np.empty_like(src)

    while not stop_event.is_set():
        np.copyto(dst, src)

# ── 메인 측정 루프 ────────────────────────────────────────────────────────────
print("=" * 60)
print("CPU + GPU Concurrent DRAM Bandwidth Test")
print("(Async Pipeline 간섭 측정)")
print("=" * 60)

GPU_TENSOR_MB = 256
TEST_CPU_CORES = [0, 1, 2, 4, 8, 14]

results = {}

print(f"\n{'CPU Cores':>10} | {'GPU BW':>12} | {'vs Baseline':>12} | {'판단':>16}")
print("-" * 60)

baseline_bw = None

for n_cpu in TEST_CPU_CORES:
    stop_event = threading.Event()

    # CPU 스레드 시작
    cpu_threads = []
    for _ in range(n_cpu):
        t = threading.Thread(
            target=cpu_worker_loop,
            args=(stop_event, 256),
            daemon=True
        )
        t.start()
        cpu_threads.append(t)

    # CPU 스레드가 안정화되도록 잠시 대기
    if n_cpu > 0:
        time.sleep(0.2)

    # GPU BW 측정
    gpu_bw = measure_gpu_bw_with_cuda_graph(GPU_TENSOR_MB)

    # CPU 스레드 중지
    stop_event.set()
    for t in cpu_threads:
        t.join(timeout=2.0)

    if baseline_bw is None:
        baseline_bw = gpu_bw
        comparison = "  (기준선)"
        判断 = "—"
    else:
        ratio       = gpu_bw / baseline_bw
        comparison  = f"  {ratio:.3f}×"
        if ratio > 0.95:
            判断 = "간섭 없음 ✓"
        elif ratio > 0.85:
            判断 = "소폭 감소 △"
        else:
            判断 = "유의미한 감소 ✗"

    print(
        f"  {n_cpu:>8} CPUs"
        f" | {gpu_bw:>10.1f} GB/s"
        f" | {comparison:>12}"
        f" | {判断}"
    )

    results[f"{n_cpu}_cpu_cores"] = {
        "n_cpu_cores": n_cpu,
        "gpu_bw_gbs":  gpu_bw,
        "ratio":       gpu_bw / baseline_bw if baseline_bw else 1.0,
    }

print("-" * 60)

# ── 결론 ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("결론 (async pipeline 설계 방향)")
print("=" * 60)

r14 = results.get("14_cpu_cores", {}).get("ratio", 1.0)
if r14 > 0.95:
    print("  14코어 동시에도 GPU BW 감소 없음")
    print("  → CPU-side prefetch 전략 안전하게 사용 가능")
    print("  → cudaMemPrefetchAsync + CPU prefetch thread 조합 검토 가능")
elif r14 > 0.85:
    print(f"  14코어 동시 시 GPU BW = {r14:.1%} ({1-r14:.1%} 감소)")
    print("  → 소폭 간섭 있음, CPU prefetch thread 수 제한 필요")
    print("  → 권장: GPU prefetch stream(cudaMemPrefetchAsync) 우선 사용")
else:
    print(f"  14코어 동시 시 GPU BW = {r14:.1%} ({1-r14:.1%} 감소) — 큰 간섭")
    print("  → CPU prefetch 전략 위험, GPU-only prefetch stream 사용 권장")

# 결과 저장
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
with open(RESULT_FILE, "w") as f:
    json.dump({
        "experiment":    "260524_cpu_gpu_concurrent_bw",
        "date":          "2026-05-24",
        "gpu_tensor_mb": GPU_TENSOR_MB,
        "baseline_bw_gbs": baseline_bw,
        "results":       results,
    }, f, indent=2)

print(f"\n결과 저장: {RESULT_FILE}")
print("완료!")
