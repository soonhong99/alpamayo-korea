"""
260526_prefetch_effect_test.py  (v3 — 올바른 BW 측정 방법)
===========================================================
GPU L2 캐시 워밍 효과 측정 (260522 방법론 적용)

[v2에서 수정한 것]
  v2 문제: sum()으로 BW 측정 → L2에 있어도 덧셈 연산이 병목 → BW 측정 불가
  v3 수정: y.copy_(x) + CUDA Graph → 순수 메모리 BW, kernel launch overhead 제거
           (260522_gpu_bw_sweep.py와 동일한 방법론)

[측정 원리]
  y.copy_(x): x 읽기(read) + y 쓰기(write) = 순수 메모리 BW 측정
  working set = 2 × size (x + y)
  L2 cliff: working set = 32 MB → size = 16 MB

  Cold: L2 flush → CUDA Graph 1회 replay → BW
  Hot : CUDA Graph 여러 번 replay (자연스러운 L2 warming) → BW

  기대 결과:
    size ≤ 16 MB  (working set ≤ 32 MB = L2):  Hot ≈ 1126 GB/s (L2 bound)
    size > 16 MB  (working set > 32 MB):         Hot ≈  231 GB/s (DRAM bound)

[Alpamayo 실험 의미]
  KV Cache: 28 MB  → working set 56 MB > L2 → copy_ 방식은 DRAM bound
  Q/O Attn: 32 MB  → working set 64 MB > L2 → DRAM bound
  MLP:      96 MB  → working set 192 MB >> L2 → DRAM bound

  핵심 질문: "96 MB MLP weight을 DRAM에서 읽는 데 얼마나 걸리나?"
  이 숫자가 async pipeline 설계의 기준값이 된다.
"""

import torch
import numpy as np

GRAPH_REPLAY   = 100    # CUDA Graph 1번에 몇 회 replay
MEASURE_ROUNDS = 30     # 측정 반복 (median 취함)
WARMUP_ROUNDS  = 20     # 측정 전 워밍업 횟수 (JIT + clock 안정화)

# ──────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────

def flush_l2_cache():
    """
    L2(32 MB) 전체 eviction.
    작동 원리: L2보다 큰 더미 텐서(128 MB) copy → 기존 L2 내용 모두 밀어냄.
    """
    n = 128 * 1024 * 1024 // 4   # 128 MB, float32
    x = torch.zeros(n, device="cuda")
    y = torch.empty_like(x)
    y.copy_(x)
    torch.cuda.synchronize()
    del x, y


def build_copy_graph(size_bytes: int, replay: int):
    """
    y.copy_(x)를 replay회 묶은 CUDA Graph 반환.
    kernel launch overhead 제거 → 순수 메모리 BW 측정 가능.
    """
    n = size_bytes // 4   # float32 원소 수
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)

    # CUDA Graph 캡처
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(replay):
            y.copy_(x)

    return g, x, y, n


def measure_bw_graph(graph, size_bytes: int, replay: int,
                     n_measure: int, n_warmup: int) -> dict:
    """
    CUDA Graph replay로 BW 측정.
    각 replay = size_bytes × 2 (read + write) × replay 회.
    """
    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)
    times_ms = []

    # 워밍업: JIT 컴파일 / GPU 클럭 안정화
    for _ in range(n_warmup):
        graph.replay()
        torch.cuda.synchronize()

    # 측정
    for _ in range(n_measure):
        start_e.record()
        graph.replay()
        end_e.record()
        torch.cuda.synchronize()
        times_ms.append(start_e.elapsed_time(end_e))

    # BW 계산: 1회 replay당 2 × size_bytes 접근
    per_replay_bytes = 2 * size_bytes           # read + write
    total_bytes      = per_replay_bytes * replay
    median_ms        = float(np.median(times_ms))
    bw_gbs           = total_bytes / (median_ms * 1e-3) / 1e9

    return {
        "bw_gbs":    bw_gbs,
        "median_ms": median_ms,          # GRAPH_REPLAY회 전체 시간
        "per_copy_ms": median_ms / replay,  # copy 1회 시간
        "all_ms":    times_ms,
    }


# ──────────────────────────────────────────────────────────────────
# CUDA 전체 워밍업 (첫 실험 전 반드시 실행)
# ──────────────────────────────────────────────────────────────────
print("CUDA 초기화 워밍업...")
_wn = 16 * 1024 * 1024 // 4
_wx = torch.randn(_wn, device="cuda")
_wy = torch.empty_like(_wx)
for _ in range(50):
    _wy.copy_(_wx)
torch.cuda.synchronize()
del _wx, _wy
print("완료\n")

# ──────────────────────────────────────────────────────────────────
# 실험 A: L2 캐시 cliff 확인
# (260522 방법과 동일 — 이미 측정된 결과 재확인)
# ──────────────────────────────────────────────────────────────────
print("=" * 65)
print("[실험 A] L2 캐시 cliff 확인  (y.copy_(x) + CUDA Graph)")
print(f"  CUDA Graph replay={GRAPH_REPLAY}, warmup={WARMUP_ROUNDS}, measure={MEASURE_ROUNDS}")
print("=" * 65)

CLIFF_SIZES_MB = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 96, 128]
L2_SIZE_MB     = 32
CLIFF_MB       = L2_SIZE_MB // 2   # working set = 2 × size → cliff at 16 MB

print(f"\n  {'Size':>6} | {'BW (GB/s)':>10} | {'vs 231 DRAM':>11} | 상태")
print("  " + "-" * 50)

cliff_results = {}
for size_mb in CLIFF_SIZES_MB:
    size_bytes = size_mb * 1024 * 1024

    g, x, y, _ = build_copy_graph(size_bytes, GRAPH_REPLAY)
    stats = measure_bw_graph(g, size_bytes, GRAPH_REPLAY, MEASURE_ROUNDS, WARMUP_ROUNDS)
    del g, x, y

    ratio = stats["bw_gbs"] / 231.0
    if stats["bw_gbs"] > 600:
        status = "L2 ✓"
    elif stats["bw_gbs"] > 300:
        status = "L2/DRAM 전환"
    else:
        status = "DRAM"

    cliff_marker = " ← cliff" if size_mb == CLIFF_MB else ""
    print(f"  {size_mb:>5}MB | {stats['bw_gbs']:>10.1f} | {ratio:>9.2f}× |"
          f" {status}{cliff_marker}")

    cliff_results[size_mb] = stats["bw_gbs"]

# ──────────────────────────────────────────────────────────────────
# 실험 B: Cold vs Hot (L2 flush 후 vs warming 후)
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("[실험 B] Cold (flush 후) vs Hot (warming 후) BW 비교")
print("  목적: prefetch_stream에서 미리 읽으면 이후 접근이 빨라지는가?")
print("=" * 65)

# 비교 대상: Alpamayo 레이어 실제 크기
COMPARE_SIZES = [
    ("KV Cache",    14,  "28 MB bfloat16, working set 28 MB"),   # 14 MB float32
    ("Q/O Attn",    16,  "32 MB bfloat16, working set 32 MB = L2 경계"),
    ("MLP weight",  48,  "96 MB bfloat16, working set 96 MB"),
]
# Note: bfloat16(2B) → float32(4B) 환산: MB_bf16 / 2 = MB_float32

prefetch_stream = torch.cuda.Stream()

print(f"\n  {'텐서':>12} | {'Cold BW':>10} | {'Hot BW':>10} | {'Speedup':>7} | 판단")
print("  " + "-" * 62)

for name, size_mb_f32, desc in COMPARE_SIZES:
    size_bytes = size_mb_f32 * 1024 * 1024

    g, x, y, _ = build_copy_graph(size_bytes, GRAPH_REPLAY)

    # ── Cold: L2 flush 직후 1회 측정 ─────────────────────────────
    flush_l2_cache()
    cold_times = []
    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        flush_l2_cache()
        start_e.record()
        g.replay()
        end_e.record()
        torch.cuda.synchronize()
        cold_times.append(start_e.elapsed_time(end_e) / GRAPH_REPLAY)
    cold_bw = 2 * size_bytes / (np.median(cold_times) * 1e-3) / 1e9

    # ── Hot: CUDA Graph 반복 실행으로 자연 warming 후 측정 ──────────
    # (prefetch_stream으로 warm_l2 후 바로 측정하는 시나리오)
    flush_l2_cache()
    with torch.cuda.stream(prefetch_stream):
        for _ in range(5):    # 5회 접근으로 L2 warming
            g.replay()
    torch.cuda.current_stream().wait_stream(prefetch_stream)
    torch.cuda.synchronize()

    hot_times = []
    for _ in range(10):
        start_e.record()
        g.replay()
        end_e.record()
        torch.cuda.synchronize()
        hot_times.append(start_e.elapsed_time(end_e) / GRAPH_REPLAY)
    hot_bw = 2 * size_bytes / (np.median(hot_times) * 1e-3) / 1e9

    speedup = hot_bw / cold_bw
    if speedup > 3.0:
        verdict = "L2 히트 ✓"
    elif speedup > 1.3:
        verdict = "부분 히트"
    else:
        verdict = "DRAM bound"

    bfloat_mb = size_mb_f32 * 2   # float32 → bfloat16 환산
    print(f"  {name:>12} | {cold_bw:>9.1f}  | {hot_bw:>9.1f}  |"
          f" {speedup:>5.2f}x  | {verdict}  ({bfloat_mb} MB BF16)")

    del g, x, y

print("  " + "-" * 62)

# ──────────────────────────────────────────────────────────────────
# 실험 C: Alpamayo async pipeline 설계를 위한 핵심 숫자
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("[실험 C] Async Pipeline 설계 기준값 계산")
print("  'MLP 가중치 읽기 시간'과 'compute 시간'을 비교하기 위한 기준")
print("=" * 65)

# DRAM BW 대표값: 96 MB (MLP) 측정
mlp_size_bytes = 96 * 1024 * 1024   # 96 MB bfloat16
g_mlp, x_mlp, y_mlp, _ = build_copy_graph(mlp_size_bytes, GRAPH_REPLAY)
mlp_stats = measure_bw_graph(g_mlp, mlp_size_bytes, GRAPH_REPLAY,
                              MEASURE_ROUNDS, WARMUP_ROUNDS)
del g_mlp, x_mlp, y_mlp

dram_bw     = mlp_stats["bw_gbs"]
mlp_read_ms = (96 * 1024 * 1024) / (dram_bw * 1e9) * 1e3   # read-only (1× BW)

print(f"""
  DRAM BW (실측, 96 MB 기준): {dram_bw:.1f} GB/s
  MLP gate_proj (96 MB) 읽기 시간: {mlp_read_ms:.3f} ms
    계산: 96 MB ÷ {dram_bw:.0f} GB/s = {mlp_read_ms:.3f} ms

  ── async pipeline 가능 조건 ──────────────────────────────────
  t_compute_MLP  > {mlp_read_ms:.3f} ms → 읽기 완전 중첩 가능 ✅ (pipeline 이득 최대)
  t_compute_MLP ≈ {mlp_read_ms:.3f} ms → 일부 중첩
  t_compute_MLP  < {mlp_read_ms:.3f} ms → 읽기가 병목 ❌ (pipeline 이득 없음)

  → 다음 실험: 260524_layer_compute_profile.py
    t_compute_MLP를 측정해서 위 조건에 대입하면
    async pipeline 효과를 실험 전에 예측 가능.
""")
