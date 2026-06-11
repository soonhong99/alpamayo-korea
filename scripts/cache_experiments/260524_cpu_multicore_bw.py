"""
260524_cpu_multicore_bw.py
===========================
CPU 멀티코어 대역폭 측정 (Phase 0 - Step 3)

목적:
  async pipeline에서 CPU가 "prefetch 보조 역할"을 할 수 있는지 판단.
  Thor CPU = 14-core Neoverse V3AE
  단일 코어 BW = ~33 GB/s (기존 측정값)
  14코어 합산 BW = ? (DRAM 포화로 수렴할 것)

결과 해석:
  14코어 BW > 96 MB / t_compute_MLP 이면:
    → CPU가 MLP 가중치를 GPU compute 중에 prefetch 가능
  14코어 BW < 96 MB / t_compute_MLP 이면:
    → CPU prefetch가 병목, GPU prefetch stream이 더 나음

실행:
  python3 260524_cpu_multicore_bw.py
"""

import threading
import time
import numpy as np
import json
import os

RESULT_FILE = os.path.expanduser(
    "~/alpamayo1.5/profiling_results/260524_cpu_multicore_bw.json"
)
ARRAY_SIZE_MB = 256   # 코어당 배열 크기 (L3 초과하여 DRAM 측정)
MEASURE_ITERS = 10    # 반복 횟수

def measure_single_core_bw(array_size_mb: int, n_iters: int) -> float:
    """
    단일 CPU 코어의 memcpy BW (GB/s).
    src → dst 복사, 2× 메모리 접근.
    """
    size  = array_size_mb * 1024 * 1024 // 4  # float32 원소 수
    src   = np.ones(size, dtype=np.float32)
    dst   = np.empty_like(src)

    # 워밍업
    for _ in range(3):
        np.copyto(dst, src)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        np.copyto(dst, src)
    t1 = time.perf_counter()

    elapsed  = t1 - t0
    total_bytes = 2 * size * 4 * n_iters  # read + write
    return total_bytes / elapsed / 1e9

def measure_multicore_bw(n_cores: int, array_size_mb: int, n_iters: int) -> dict:
    """
    n_cores개 코어가 동시에 memcpy할 때의 합산 BW.
    각 코어가 독립적인 배열을 사용 → DRAM BW 측정.
    """
    results = []
    barrier = threading.Barrier(n_cores)  # 동시 시작 보장

    def worker():
        size = array_size_mb * 1024 * 1024 // 4
        src  = np.ones(size, dtype=np.float32)
        dst  = np.empty_like(src)

        # 워밍업
        for _ in range(3):
            np.copyto(dst, src)

        barrier.wait()  # 모든 스레드 준비 완료 후 동시 시작

        t0 = time.perf_counter()
        for _ in range(n_iters):
            np.copyto(dst, src)
        t1 = time.perf_counter()

        elapsed      = t1 - t0
        total_bytes  = 2 * size * 4 * n_iters
        bw_gbs       = total_bytes / elapsed / 1e9
        results.append(bw_gbs)

    threads = [threading.Thread(target=worker) for _ in range(n_cores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {
        "n_cores":      n_cores,
        "total_bw_gbs": sum(results),
        "per_core_gbs": list(results),
        "mean_per_core": float(np.mean(results)),
    }


print("=" * 60)
print("CPU Multi-Core Bandwidth Measurement")
print(f"Array size per core: {ARRAY_SIZE_MB} MB | Iterations: {MEASURE_ITERS}")
print("=" * 60)

TEST_CORES = [1, 2, 4, 7, 10, 14]
all_results = {}

# 비교 기준: async pipeline에서 MLP 가중치(96 MB)를 prefetch하는 데 필요한 BW
# t_compute_MLP (실측 필요) — 여기서는 예시 2ms 가정
MLP_WEIGHT_MB      = 96.0
ASSUMED_COMPUTE_MS = 2.0   # 실제는 layer_compute_profile.py 결과 사용
required_bw_gbs    = MLP_WEIGHT_MB / (ASSUMED_COMPUTE_MS * 1e-3) / 1e3

print(f"\n{'Cores':>6}  {'Total BW':>10}  {'Per Core':>10}  {'MLP prefetch':>14}")
print("-" * 55)

for n in TEST_CORES:
    res = measure_multicore_bw(n, ARRAY_SIZE_MB, MEASURE_ITERS)
    pipeline_ok = "✓ 가능" if res["total_bw_gbs"] >= required_bw_gbs else "✗ 부족"

    print(
        f"  {n:>3} cores"
        f"  {res['total_bw_gbs']:>9.1f} GB/s"
        f"  {res['mean_per_core']:>9.1f} GB/s"
        f"  {pipeline_ok:>13}"
    )
    all_results[f"{n}_cores"] = res

print("-" * 55)
print(f"\n기준: {required_bw_gbs:.1f} GB/s")
print(f"  (96 MB MLP 가중치를 {ASSUMED_COMPUTE_MS} ms 안에 prefetch하려면)")
print(f"  실제 compute time은 260524_layer_compute_profile.py 결과 참조")

# 결과 저장
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
with open(RESULT_FILE, "w") as f:
    json.dump({
        "experiment":       "260524_cpu_multicore_bw",
        "date":             "2026-05-24",
        "array_size_mb":    ARRAY_SIZE_MB,
        "measure_iters":    MEASURE_ITERS,
        "required_bw_gbs":  required_bw_gbs,
        "assumed_compute_ms": ASSUMED_COMPUTE_MS,
        "results":          all_results,
    }, f, indent=2)

print(f"\n결과 저장: {RESULT_FILE}")
print("완료!")
