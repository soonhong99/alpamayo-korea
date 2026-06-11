"""
260531_kv_cache_profiling.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[측정 목적]
  교수님 피드백 #1: "KV 캐시의 정확한 용량, 어디까지 늘어나는지 확실히 조사 필요"
  L2 KV Pinning 실험의 전제 조건: 실제로 L2(32MB)에 올릴 수 있는 대상이 무엇인지 확정

[측정 항목]
  1. KV Cache 텐서 구조 자동 탐색
     - VLM KV: N_layers, N_kv_heads, head_dim, dtype 실측
     - Expert KV: 동일 구조 + Flow 단계에서의 KV 크기 변화

  2. 파이프라인 단계별 KV 크기 (이론값 vs 실측값 교차 검증)
     - Prefill 완료 직후 (seq_len = 3,086)
     - Decode step 0 → 17 step 동안의 step-by-step 증가량
     - Decode 완료 후 최대값
     - Flow 단계: prompt_cache 크기 (crop 전/후)

  3. Layer-by-layer 상세 분석
     - 각 layer의 K, V 텐서 크기 (MB)
     - 전체 KV 중 레이어별 비율

  4. L2 Coverage 분석
     - L2 = 32 MB 기준으로 KV 전체의 몇 %가 들어가는지
     - L2에 올릴 수 있는 최대 토큰 수 (per layer, all layers)
     - Hot block 후보 (text_prefix KV, expert 64 tokens KV)

  5. 대역폭 함의 (Bandwidth Implication)
     - Decode: 매 step KV 읽기량 (MB/step)
     - Flow: 10 ODE steps에서 KV 총 읽기량 (MB)
     - L2 pinning 시 이론적 절감량 계산

  6. GPU 메모리 실측
     - torch.cuda.memory_allocated() 기반 실제 메모리 점유
     - 이론값 대비 fragmentation ratio

[측정 정밀도]
  - GPU Event Timing: CUDA 내부 타임스탬프 (ns 단위)
  - Warmup: 1회 (CUDA JIT, cache cold-start 제거)
  - Measure: 3회 (mean ± std)
  - 이론값: 수식 직접 계산, 실측값과 비교

[출력]
  profiling_results/260531_kv_cache/
    kv_structure.json           - KV 텐서 구조 (자동 탐색)
    kv_sizes_by_stage.json      - 단계별 크기 (이론 vs 실측)
    kv_growth_per_decode_step.json - decode step별 증가량
    kv_l2_analysis.json         - L2 활용 분석 및 pinning 후보
    kv_bandwidth_implication.json - 대역폭 함의
    kv_all_results.json         - 전체 통합 결과

[실행]
  source ~/alpamayo1.5/a1_5_venv/bin/activate && cd ~/alpamayo1.5
  python3 scripts/inference/260531_kv_cache_profiling.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.token_utils import to_special_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────────
CLIP_ID   = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US     = 5_100_000
DEVICE    = "cuda"
NUM_WARMUP  = 1
NUM_MEASURE = 3

# Thor GPU L2 Cache 크기 (SM 11.0 Blackwell)
L2_CACHE_BYTES = 32 * 1024 * 1024   # 32 MB
L2_PERSIST_FRACTION = 0.75           # L2 Persistence API가 사용 가능한 최대 비율
L2_PERSIST_BYTES = int(L2_CACHE_BYTES * L2_PERSIST_FRACTION)  # 24 MB

# Thor DRAM 대역폭 (실측)
DRAM_BW_GB_S = 231.0

# 베이스라인 (비교용)
BASELINE_DECODE_STEPS = 17
BASELINE_FLOW_ODE_STEPS = 10
BASELINE_FLOW_EXPERT_LAYERS = 28  # model.expert 레이어 수

OUT = Path("profiling_results/260531_kv_cache")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1: 측정 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

class GpuTimer:
    """CUDA Event 기반 GPU 타이머."""
    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._s.record()

    def stop(self):
        self._e.record()

    def elapsed_ms(self) -> float:
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *_):
        self.stop()


class WallTimer:
    """Wall-clock 타이머 (GPU sync 포함)."""
    def __init__(self):
        self._t0 = self._t1 = 0.0

    def start(self):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (self._t1 - self._t0) * 1e3

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *_):
        self.stop()


class StatSummary:
    def __init__(self, values: List[float], label: str = ""):
        self.label = label
        self.values = sorted(values)
        self.n = len(values)

    @property
    def mean(self): return mean(self.values) if self.n else float("nan")
    @property
    def std(self): return stdev(self.values) if self.n >= 2 else 0.0
    @property
    def minimum(self): return self.values[0] if self.n else float("nan")
    @property
    def maximum(self): return self.values[-1] if self.n else float("nan")
    @property
    def p95(self):
        if not self.n: return float("nan")
        idx = 0.95 * (self.n - 1)
        lo, hi = int(idx), min(int(idx)+1, self.n-1)
        return self.values[lo]*(1-idx+lo) + self.values[hi]*(idx-lo)

    def to_dict(self):
        return {
            "label": self.label, "n": self.n,
            "mean": round(self.mean, 3), "std": round(self.std, 3),
            "min": round(self.minimum, 3), "max": round(self.maximum, 3),
            "p95": round(self.p95, 3),
            "values": [round(v, 3) for v in self.values],
        }

    def __repr__(self):
        return (f"[{self.label}] mean={self.mean:.2f}ms std={self.std:.2f}ms "
                f"[{self.minimum:.2f}, {self.maximum:.2f}] p95={self.p95:.2f}ms")


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2: DynamicCache 버전 무관 헬퍼 (260528 검증 코드 재사용)
# ══════════════════════════════════════════════════════════════════════════════
# transformers 버전에 따라 DynamicCache 내부 속성명이 다르다:
#   - transformers 4.38-4.46: key_cache / value_cache  (List[Tensor])
#   - transformers 4.47+:     구조 변경 → to_legacy_cache() 로 우회
#   - 구버전 tuple-of-tuples: cache[layer] = (k, v)
# 아래 헬퍼는 이를 통합 처리한다.

def _cache_to_kv_pairs(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """cache 객체에서 레이어별 (key, value) 텐서 쌍 추출. 버전 무관."""
    # 방법 1: key_cache / value_cache 직접 접근 (4.38+)
    kc = getattr(cache, 'key_cache', None)
    vc = getattr(cache, 'value_cache', None)
    if isinstance(kc, list) and len(kc) > 0 and isinstance(kc[0], torch.Tensor):
        return list(zip(kc, vc))

    # 방법 2: _key_cache / _value_cache (private attr)
    kc = getattr(cache, '_key_cache', None)
    vc = getattr(cache, '_value_cache', None)
    if isinstance(kc, list) and len(kc) > 0 and isinstance(kc[0], torch.Tensor):
        return list(zip(kc, vc))

    # 방법 3: to_legacy_cache() → tuple of (k, v) per layer
    if hasattr(cache, 'to_legacy_cache'):
        try:
            legacy = cache.to_legacy_cache()
            if (legacy is not None and len(legacy) > 0
                    and isinstance(legacy[0], (tuple, list))
                    and isinstance(legacy[0][0], torch.Tensor)):
                return [(layer[0], layer[1]) for layer in legacy]
        except Exception:
            pass

    # 방법 4: cache 자체가 tuple of (k, v)
    if (isinstance(cache, (tuple, list)) and len(cache) > 0
            and isinstance(cache[0], (tuple, list)) and len(cache[0]) == 2
            and isinstance(cache[0][0], torch.Tensor)):
        return [(layer[0], layer[1]) for layer in cache]

    # 진단 정보 출력 후 예외
    all_attrs = [a for a in dir(cache) if not a.startswith('__')]
    kv_attrs = [a for a in all_attrs
                if any(x in a.lower() for x in ('key', 'value', 'kv', 'cache', 'layer'))]
    raise AttributeError(
        f"\n[DynamicCache 구조 불명] type={type(cache)}\n"
        f"  KV 관련 attr: {kv_attrs}\n"
        f"  전체 attr:    {all_attrs}"
    )


def _get_cache_seq_len(cache) -> int:
    """DynamicCache에 저장된 시퀀스 길이 반환. 버전 무관."""
    if hasattr(cache, 'get_seq_length') and callable(cache.get_seq_length):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    try:
        pairs = _cache_to_kv_pairs(cache)
        if pairs:
            return int(pairs[0][0].shape[2])
    except Exception:
        pass
    return 0


def _measure_kv_bytes(cache) -> int:
    """KV cache의 총 바이트 수. _cache_to_kv_pairs 기반."""
    pairs = _cache_to_kv_pairs(cache)
    return sum(k.nbytes + v.nbytes for k, v in pairs)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3: KV Cache 분석 함수
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KVLayerInfo:
    layer_idx: int
    k_shape: tuple
    v_shape: tuple
    k_bytes: int
    v_bytes: int
    total_bytes: int
    total_mb: float
    dtype: str
    seq_len: int
    n_kv_heads: int
    head_dim: int


@dataclass
class KVCacheSnapshot:
    """KV Cache의 특정 시점 스냅샷."""
    label: str                          # 측정 시점 레이블
    n_layers: int                       # 실제 레이어 수
    seq_len: int                        # 현재 시퀀스 길이 (토큰 수)
    n_kv_heads: int                     # KV head 수
    head_dim: int                       # Head dimension
    dtype: str                          # 데이터 타입 (e.g., "bfloat16")
    bytes_per_token_per_layer: int      # 토큰 1개, 레이어 1개의 KV 크기 (K+V)
    total_bytes: int                    # 전체 KV 크기 (bytes)
    total_mb: float                     # 전체 KV 크기 (MB)
    # L2 분석
    l2_coverage_pct: float              # L2 32MB 대비 KV 크기 비율 (%)
    tokens_fit_in_l2: int               # L2 Persistence 영역(24MB)에 들어가는 토큰 수
    tokens_fit_in_l2_per_layer: int     # 레이어 1개 기준으로 L2에 들어가는 토큰 수
    # GPU 메모리 (실측)
    gpu_allocated_mb: float             # torch.cuda.memory_allocated() 기준
    # 레이어별 상세
    per_layer_mb: List[float]           # 각 레이어의 KV 크기 (MB)


def inspect_kv_cache(cache, label: str) -> KVCacheSnapshot:
    """
    DynamicCache의 내부 구조를 완전히 분석한다.
    버전 무관 _cache_to_kv_pairs 헬퍼를 사용하므로
    transformers 어느 버전에서도 동작한다.

    KV 텐서 shape: [batch, n_kv_heads, seq_len, head_dim]
    """
    # 버전 무관 KV 추출
    pairs = _cache_to_kv_pairs(cache)
    n_layers = len(pairs)
    assert n_layers > 0, "KV cache is empty"

    # 첫 레이어에서 구조 추출
    k0, v0 = pairs[0]
    batch, n_kv_heads, seq_len, head_dim = k0.shape
    dtype_str = str(k0.dtype).replace("torch.", "")
    bytes_per_element = k0.element_size()  # BF16=2, FP32=4

    # 토큰 1개, 레이어 1개의 KV 크기 (K+V)
    bytes_per_token_per_layer = (
        2           # K + V
        * n_kv_heads
        * head_dim
        * bytes_per_element
    )

    # 전체 KV 크기 (레이어별)
    total_bytes = 0
    per_layer_mb = []
    for k, v in pairs:
        layer_bytes = k.nbytes + v.nbytes
        total_bytes += layer_bytes
        per_layer_mb.append(round(layer_bytes / 1e6, 3))

    total_mb = total_bytes / 1e6

    # L2 분석
    l2_coverage_pct = (total_bytes / L2_CACHE_BYTES) * 100.0
    tokens_fit_in_l2_per_layer = L2_PERSIST_BYTES // bytes_per_token_per_layer
    tokens_fit_in_l2 = L2_PERSIST_BYTES // (bytes_per_token_per_layer * n_layers)

    # GPU 메모리 실측
    gpu_allocated_mb = torch.cuda.memory_allocated(DEVICE) / 1e6

    return KVCacheSnapshot(
        label=label,
        n_layers=n_layers,
        seq_len=seq_len,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtype_str,
        bytes_per_token_per_layer=bytes_per_token_per_layer,
        total_bytes=total_bytes,
        total_mb=round(total_mb, 3),
        l2_coverage_pct=round(l2_coverage_pct, 1),
        tokens_fit_in_l2=tokens_fit_in_l2,
        tokens_fit_in_l2_per_layer=tokens_fit_in_l2_per_layer,
        gpu_allocated_mb=round(gpu_allocated_mb, 1),
        per_layer_mb=per_layer_mb,
    )


def print_snapshot(snap: KVCacheSnapshot):
    """KVCacheSnapshot을 사람이 읽기 좋은 형태로 출력."""
    sep = "─" * 60
    print(f"\n  {sep}")
    print(f"  KV Cache 스냅샷: [{snap.label}]")
    print(f"  {sep}")
    print(f"  구조:  {snap.n_layers} layers × 2(K,V) × {snap.n_kv_heads} heads × "
          f"{snap.seq_len} tokens × {snap.head_dim} head_dim × {snap.dtype}")
    print(f"  크기:  {snap.total_mb:.1f} MB  ({snap.total_bytes:,} bytes)")
    print(f"  토큰당: {snap.bytes_per_token_per_layer/1024:.1f} KB/token/layer, "
          f"{snap.bytes_per_token_per_layer * snap.n_layers / 1024:.1f} KB/token(all layers)")
    print(f"  GPU 메모리: {snap.gpu_allocated_mb:.0f} MB (allocated)")
    print(f"")
    print(f"  ── L2 Cache(32MB) 분석 ──")
    print(f"  KV 전체 / L2:    {snap.total_mb:.1f} MB / 32 MB = {snap.l2_coverage_pct:.1f}× 초과")
    print(f"  L2 Persist(24MB) 기준:")
    print(f"    전체 layer 기준: {snap.tokens_fit_in_l2} tokens 수용 가능")
    print(f"    1 layer 기준:   {snap.tokens_fit_in_l2_per_layer} tokens 수용 가능")

    # Hot block 후보 표시
    text_prefix_bytes = 100 * snap.bytes_per_token_per_layer * snap.n_layers
    expert_64_bytes = 64 * snap.bytes_per_token_per_layer * 28  # expert 28 layers
    print(f"")
    print(f"  ── L2 Pinning 후보 ──")
    print(f"  text_prefix KV (100 tok): {text_prefix_bytes/1e6:.1f} MB  "
          f"→ L2 수용 {'✅' if text_prefix_bytes < L2_PERSIST_BYTES else '❌'}")
    print(f"  expert 64 tok KV (28 L):  {expert_64_bytes/1e6:.1f} MB  "
          f"→ L2 수용 {'✅' if expert_64_bytes < L2_PERSIST_BYTES else '❌'}")
    print(f"  1 layer KV ({snap.seq_len} tok):  {snap.per_layer_mb[0]:.1f} MB  "
          f"→ L2 수용 {'✅' if snap.per_layer_mb[0] * 1e6 < L2_PERSIST_BYTES else '❌'}")
    print(f"  {sep}")


def compute_bandwidth_implications(snap: KVCacheSnapshot) -> dict:
    """
    KV Cache 크기 기반 대역폭 함의 계산.

    Decode: 매 step마다 KV cache 전체를 1회 읽음 (새 query가 모든 KV에 attend)
    Flow:   10 ODE step × 28 layers → 각 레이어의 KV를 10회씩 읽음
    """
    # ── Decode 대역폭 함의 ──────────────────────────────────────────────────
    # 새 토큰 1개가 전체 KV cache에 attend → KV 전체 1회 read
    decode_kv_read_mb_per_step = snap.total_mb
    decode_kv_total_mb = decode_kv_read_mb_per_step * BASELINE_DECODE_STEPS
    decode_kv_time_ms_per_step = decode_kv_read_mb_per_step / (DRAM_BW_GB_S * 1000) * 1e3

    # ── Flow 대역폭 함의 ────────────────────────────────────────────────────
    # 28 expert layers × 각 layer의 KV (seq_len × n_kv_heads × head_dim × 2 × 2bytes)
    # = expert layer 수 / VLM layer 수 비율로 추산
    flow_layers_ratio = BASELINE_FLOW_EXPERT_LAYERS / snap.n_layers
    per_layer_kv_mb = snap.total_mb / snap.n_layers
    flow_kv_per_ode_step_mb = per_layer_kv_mb * BASELINE_FLOW_EXPERT_LAYERS
    flow_kv_total_mb = flow_kv_per_ode_step_mb * BASELINE_FLOW_ODE_STEPS
    flow_kv_time_ms = flow_kv_total_mb / (DRAM_BW_GB_S * 1000) * 1e3

    # ── L2 Pinning으로 절약 가능한 양 (이론값) ─────────────────────────────
    # 2 layers KV가 L2에 고정될 때
    pinnable_layers = L2_PERSIST_BYTES // (int(per_layer_kv_mb * 1e6) + 1)
    pinnable_layers_capped = min(pinnable_layers, BASELINE_FLOW_EXPERT_LAYERS)
    # Flow에서 각 pinned layer의 KV는 10 ODE step에서 재사용
    flow_pinning_saving_mb = pinnable_layers_capped * per_layer_kv_mb * BASELINE_FLOW_ODE_STEPS
    flow_pinning_saving_pct = flow_pinning_saving_mb / (flow_kv_total_mb + 1e-9) * 100

    return {
        "decode": {
            "kv_read_per_step_mb": round(decode_kv_read_mb_per_step, 2),
            "kv_read_total_mb": round(decode_kv_total_mb, 2),
            "kv_time_per_step_ms_theoretical": round(decode_kv_time_ms_per_step, 2),
            "note": "매 decode step마다 KV 전체 1회 read (새 query → 모든 KV attend)",
        },
        "flow": {
            "expert_layers": BASELINE_FLOW_EXPERT_LAYERS,
            "vlm_layers": snap.n_layers,
            "per_layer_kv_mb": round(per_layer_kv_mb, 3),
            "kv_per_ode_step_mb": round(flow_kv_per_ode_step_mb, 2),
            "kv_total_mb_10steps": round(flow_kv_total_mb, 2),
            "kv_theoretical_time_ms": round(flow_kv_time_ms, 2),
            "note": f"10 ODE steps × {BASELINE_FLOW_EXPERT_LAYERS} layers × layer KV",
        },
        "l2_pinning": {
            "pinnable_layers": pinnable_layers_capped,
            "pinnable_kv_mb": round(pinnable_layers_capped * per_layer_kv_mb, 2),
            "flow_savings_mb": round(flow_pinning_saving_mb, 2),
            "flow_savings_pct": round(flow_pinning_saving_pct, 1),
            "verdict": (
                "의미 있는 절약" if flow_pinning_saving_pct > 5
                else "효과 미미 (5% 미만)"
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3: 핵심 측정 — 단계별 KV 크기
# ══════════════════════════════════════════════════════════════════════════════

def run_prefill_and_capture_kv(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
) -> Tuple[DynamicCache, float, int]:
    """
    VE + LM Prefill 실행 후 KV cache 반환.

    Returns:
        kv_cache:    DynamicCache (prefill 완료 상태)
        prefill_ms:  실행 시간 (ms)
        seq_len:     KV에 채워진 토큰 수
    """
    gt = GpuTimer()
    gt.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=input_ids,
            attention_mask=tok_data.get("attention_mask"),
            pixel_values=tok_data.get("pixel_values"),
            image_grid_thw=tok_data.get("image_grid_thw"),
            use_cache=True,
            logits_to_keep=1,
        )
    gt.stop()
    prefill_ms = gt.elapsed_ms()

    kv_cache = out.past_key_values
    seq_len = _get_cache_seq_len(kv_cache)
    return kv_cache, prefill_ms, seq_len


def run_single_decode_step(
    model: Alpamayo1_5,
    next_token: torch.Tensor,
    past_kv: DynamicCache,
    cache_pos: int,
) -> Tuple[torch.Tensor, float]:
    """
    Decode 1 step 실행. KV를 in-place 확장하는 DynamicCache 활용.

    Returns:
        logits:    [1, vocab_size]
        step_ms:   실행 시간 (ms)
    """
    cache_position = torch.tensor([cache_pos], device=DEVICE, dtype=torch.long)
    gt = GpuTimer()
    gt.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=next_token.unsqueeze(0) if next_token.dim() == 0 else next_token,
            past_key_values=past_kv,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
    gt.stop()
    step_ms = gt.elapsed_ms()
    return out.logits[:, -1, :], step_ms


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 4: 메인 프로파일링
# ══════════════════════════════════════════════════════════════════════════════

def profile_kv_cache(model: Alpamayo1_5, input_ids: torch.Tensor, tok_data: dict) -> dict:
    """
    KV Cache 전체 프로파일링 실행.

    측정 순서:
    1. Prefill KV 스냅샷 (3,086 tokens)
    2. Decode step-by-step KV 증가 추적 (최대 25 steps)
    3. Expert KV 구조 분석 (model.expert)
    4. 이론값 vs 실측값 비교
    5. L2 pinning 후보 분석
    """
    eos_id       = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_offset  = model.config.traj_token_start_idx
    traj_vocab   = model.config.traj_vocab_size

    results = {}
    W = 70

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1: Prefill KV 구조 분석
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  Phase 1: Prefill KV 구조 분석")
    print(f"{'═'*W}")

    prefill_ms_list = []
    prefill_snapshot = None

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"

        # GPU 메모리 기준점 기록
        torch.cuda.synchronize()
        mem_before_mb = torch.cuda.memory_allocated(DEVICE) / 1e6

        kv_cache, prefill_ms, seq_len = run_prefill_and_capture_kv(
            model, input_ids, tok_data
        )

        mem_after_mb = torch.cuda.memory_allocated(DEVICE) / 1e6
        kv_mem_delta_mb = mem_after_mb - mem_before_mb

        snap = inspect_kv_cache(kv_cache, f"Prefill({seq_len} tokens)")

        print(f"  [{tag}] Prefill: {prefill_ms:.0f}ms  "
              f"KV: {snap.total_mb:.1f}MB  "
              f"seq_len={seq_len}  "
              f"layers={snap.n_layers}  "
              f"kv_heads={snap.n_kv_heads}  "
              f"head_dim={snap.head_dim}  "
              f"dtype={snap.dtype}")

        if not is_warmup:
            prefill_ms_list.append(prefill_ms)
            if prefill_snapshot is None:
                prefill_snapshot = snap

        del kv_cache
        torch.cuda.empty_cache()

    prefill_stat = StatSummary(prefill_ms_list, "prefill")
    print(f"\n  {prefill_stat}")
    print_snapshot(prefill_snapshot)

    results["prefill"] = {
        "timing_ms": prefill_stat.to_dict(),
        "snapshot": asdict(prefill_snapshot),
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2: Decode Step-by-Step KV 증가 추적
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  Phase 2: Decode Step-by-Step KV 증가 추적 (최대 25 steps)")
    print(f"{'═'*W}")
    print("  step | seq_len | KV total(MB) | delta(KB) | step time(ms)")
    print(f"  {'-'*55}")

    # Prefill 재실행 (1회, decode용)
    kv_for_decode, prefill_ms_d, prefill_seq = run_prefill_and_capture_kv(
        model, input_ids, tok_data
    )

    decode_step_records = []
    MAX_TRACK_STEPS = 25

    # 첫 logit
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=input_ids,
            attention_mask=tok_data.get("attention_mask"),
            pixel_values=tok_data.get("pixel_values"),
            image_grid_thw=tok_data.get("image_grid_thw"),
            use_cache=True,
            logits_to_keep=1,
        )
    prev_kv = out.past_key_values
    prev_seq = _get_cache_seq_len(prev_kv)
    prev_mb = _measure_kv_bytes(prev_kv) / 1e6

    # 첫 decode token
    first_logits = out.logits[:, -1, :].float()
    first_logits[:, traj_offset:traj_offset + traj_vocab] = float("-inf")
    next_tok = first_logits.argmax(dim=-1, keepdim=True)

    for step in range(MAX_TRACK_STEPS):
        cache_position = torch.tensor([prev_seq + step], device=DEVICE, dtype=torch.long)
        gt = GpuTimer()
        gt.start()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            step_out = model.vlm(
                input_ids=next_tok,
                past_key_values=prev_kv,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,
            )
        gt.stop()
        step_ms = gt.elapsed_ms()

        curr_kv = step_out.past_key_values
        curr_seq = _get_cache_seq_len(curr_kv)
        curr_mb = _measure_kv_bytes(curr_kv) / 1e6
        delta_kb = (curr_mb - prev_mb) * 1024.0

        rec = {
            "step": step + 1,
            "seq_len": curr_seq,
            "kv_total_mb": round(curr_mb, 3),
            "delta_kb": round(delta_kb, 2),
            "step_ms": round(step_ms, 2),
        }
        decode_step_records.append(rec)
        print(f"  {step+1:4d} | {curr_seq:7d} | {curr_mb:12.3f} | "
              f"{delta_kb:9.1f} | {step_ms:.1f}ms")

        # EOS 체크
        step_logits = step_out.logits[:, -1, :].float()
        step_logits[:, traj_offset:traj_offset + traj_vocab] = float("-inf")
        next_tok = step_logits.argmax(dim=-1, keepdim=True)
        if (next_tok == eos_id).all():
            print(f"  → EOS at step {step+1}, stopping")
            break

        prev_kv = curr_kv
        prev_mb = curr_mb
        prev_seq = curr_seq

    # 이론값 vs 실측값 비교
    if decode_step_records:
        avg_delta_kb = mean([r["delta_kb"] for r in decode_step_records])
        theoretical_delta_kb = (
            prefill_snapshot.bytes_per_token_per_layer
            * prefill_snapshot.n_layers
            / 1024.0
        )
        print(f"\n  평균 실측 증가량: {avg_delta_kb:.1f} KB/step")
        print(f"  이론 계산값:       {theoretical_delta_kb:.1f} KB/step")
        print(f"  오차: {abs(avg_delta_kb - theoretical_delta_kb):.1f} KB "
              f"({abs(avg_delta_kb - theoretical_delta_kb)/theoretical_delta_kb*100:.1f}%)")

    results["decode_growth"] = {
        "steps": decode_step_records,
        "avg_delta_kb": round(mean([r["delta_kb"] for r in decode_step_records]) if decode_step_records else 0, 2),
        "theoretical_delta_kb": round(theoretical_delta_kb, 2),
    }

    del prev_kv, kv_for_decode
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3: Expert KV 구조 분석 (model.expert용)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  Phase 3: Expert KV 구조 분석 (Flow 단계 model.expert)")
    print(f"{'═'*W}")

    # expert 모듈 구조 탐색
    expert = model.expert
    expert_layers = len(list(expert.children()))

    # expert의 실제 레이어 수는 모델 설정에서
    try:
        n_expert_layers = expert.config.num_hidden_layers
    except AttributeError:
        n_expert_layers = sum(1 for _ in expert.named_modules()
                              if hasattr(_, 'self_attn'))

    print(f"  model.expert: {type(expert).__name__}")
    print(f"  파라미터: {sum(p.numel() for p in expert.parameters())/1e6:.1f}M")

    # expert KV head 수는 VLM과 동일 아키텍처 (Qwen3VL text model)
    try:
        n_expert_kv_heads = expert.config.num_key_value_heads
        n_expert_layers_actual = expert.config.num_hidden_layers
        expert_head_dim = expert.config.hidden_size // expert.config.num_attention_heads
    except AttributeError:
        # fallback: VLM 구조와 동일하다고 가정
        n_expert_kv_heads = prefill_snapshot.n_kv_heads
        n_expert_layers_actual = BASELINE_FLOW_EXPERT_LAYERS
        expert_head_dim = prefill_snapshot.head_dim

    print(f"\n  Expert 구조 (config 기반):")
    print(f"    num_hidden_layers:   {n_expert_layers_actual}")
    print(f"    num_key_value_heads: {n_expert_kv_heads}")
    print(f"    head_dim:            {expert_head_dim}")

    # Expert가 사용하는 KV는 VLM의 prompt_cache (reuse됨)
    # Flow 중 expert는 64 new tokens를 KV에 append → crop
    # 따라서 expert KV 사용량 = VLM KV + 64 임시 tokens

    vlm_kv_mb = prefill_snapshot.total_mb
    bytes_per_token_expert = (
        2 * n_expert_kv_heads * expert_head_dim * 2  # BF16
        * n_expert_layers_actual
    )
    expert_64tok_mb = 64 * bytes_per_token_expert / 1e6

    # Flow 전체에서 KV 읽기량
    flow_kv_per_ode_step_mb = vlm_kv_mb * (n_expert_layers_actual / prefill_snapshot.n_layers)
    flow_kv_total_mb = flow_kv_per_ode_step_mb * BASELINE_FLOW_ODE_STEPS

    print(f"\n  Flow 단계 KV 사용량:")
    print(f"    VLM prompt_cache (fixed):      {vlm_kv_mb:.1f} MB")
    print(f"    Expert 64 token 임시 KV:       {expert_64tok_mb:.1f} MB (crop으로 제거)")
    print(f"    ODE step당 KV 읽기 (28 layers): {flow_kv_per_ode_step_mb:.1f} MB")
    print(f"    10 ODE steps 총 KV 읽기:        {flow_kv_total_mb:.1f} MB")
    print(f"    이론 KV 읽기 시간 (10 ODE):     "
          f"{flow_kv_total_mb/(DRAM_BW_GB_S*1000)*1e3:.1f}ms")

    results["expert_kv"] = {
        "n_layers": n_expert_layers_actual,
        "n_kv_heads": n_expert_kv_heads,
        "head_dim": expert_head_dim,
        "vlm_prompt_cache_mb": round(vlm_kv_mb, 2),
        "expert_64tok_kv_mb": round(expert_64tok_mb, 2),
        "flow_kv_per_ode_step_mb": round(flow_kv_per_ode_step_mb, 2),
        "flow_kv_total_10steps_mb": round(flow_kv_total_mb, 2),
        "flow_kv_theoretical_ms": round(flow_kv_total_mb/(DRAM_BW_GB_S*1000)*1e3, 2),
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase 4: L2 Pinning 후보 종합 분석
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  Phase 4: L2 Pinning 후보 종합 분석")
    print(f"{'═'*W}")

    bw_implications = compute_bandwidth_implications(prefill_snapshot)

    candidates = [
        {
            "name": "VLM KV 전체 (3,086 tok)",
            "mb": prefill_snapshot.total_mb,
            "access_count": BASELINE_DECODE_STEPS + BASELINE_FLOW_ODE_STEPS * BASELINE_FLOW_EXPERT_LAYERS,
            "fits_l2": prefill_snapshot.total_mb * 1e6 < L2_PERSIST_BYTES,
            "note": "decode 17step + flow 280 access",
        },
        {
            "name": f"text_prefix KV (100 tok, {prefill_snapshot.n_layers}L)",
            "mb": 100 * prefill_snapshot.bytes_per_token_per_layer * prefill_snapshot.n_layers / 1e6,
            "access_count": BASELINE_DECODE_STEPS + BASELINE_FLOW_ODE_STEPS * BASELINE_FLOW_EXPERT_LAYERS,
            "fits_l2": 100 * prefill_snapshot.bytes_per_token_per_layer * prefill_snapshot.n_layers < L2_PERSIST_BYTES,
            "note": "프레임 간 불변, 반복 접근",
        },
        {
            "name": f"expert 64tok KV ({n_expert_layers_actual}L)",
            "mb": expert_64tok_mb,
            "access_count": BASELINE_FLOW_ODE_STEPS,
            "fits_l2": expert_64tok_mb * 1e6 < L2_PERSIST_BYTES,
            "note": "ODE step당 append 후 crop",
        },
        {
            "name": f"VLM 1 layer KV (3,086 tok, 1L)",
            "mb": prefill_snapshot.per_layer_mb[0] if prefill_snapshot.per_layer_mb else 0,
            "access_count": BASELINE_DECODE_STEPS + BASELINE_FLOW_ODE_STEPS,
            "fits_l2": (prefill_snapshot.per_layer_mb[0] if prefill_snapshot.per_layer_mb else 999) * 1e6 < L2_PERSIST_BYTES,
            "note": "레이어 1개 단위",
        },
        {
            "name": "action_in_proj + action_out_proj",
            "mb": sum(p.numel() for p in model.action_in_proj.parameters()) * 2 / 1e6,
            "access_count": BASELINE_FLOW_ODE_STEPS,
            "fits_l2": sum(p.numel() for p in model.action_in_proj.parameters()) * 2 < L2_PERSIST_BYTES,
            "note": "10 ODE steps에서 10회 접근, 크기 작음",
        },
    ]

    print(f"  {'후보':40} {'크기':>8} {'접근횟수':>8} {'L2 수용':>8}")
    print(f"  {'-'*68}")
    for c in candidates:
        mark = "✅" if c["fits_l2"] else "❌"
        print(f"  {c['name']:40} {c['mb']:7.1f}MB {c['access_count']:8d}회 {mark:>5} {c['note']}")

    print(f"\n  L2 Pinning 실험 권장 대상:")
    viable = [c for c in candidates if c["fits_l2"] and c["access_count"] >= 5]
    for v in viable:
        print(f"    ★ {v['name']} ({v['mb']:.1f} MB)")

    results["l2_candidates"] = candidates
    results["bandwidth_implications"] = bw_implications
    results["l2_config"] = {
        "l2_total_bytes": L2_CACHE_BYTES,
        "l2_total_mb": L2_CACHE_BYTES / 1e6,
        "l2_persist_bytes": L2_PERSIST_BYTES,
        "l2_persist_mb": L2_PERSIST_BYTES / 1e6,
        "dram_bw_gb_s": DRAM_BW_GB_S,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Phase 5: 종합 요약 출력
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  ★ KV Cache 프로파일링 종합 요약")
    print(f"{'═'*W}")

    snap = prefill_snapshot
    print(f"\n  [KV 구조]")
    print(f"    VLM:    {snap.n_layers} layers × 2 × {snap.n_kv_heads} heads × "
          f"{snap.seq_len} tokens × {snap.head_dim} dim × {snap.dtype}")
    print(f"    Expert: {n_expert_layers_actual} layers × 2 × {n_expert_kv_heads} heads × "
          f"3,087 tokens × {expert_head_dim} dim × bfloat16")

    print(f"\n  [KV 크기]")
    print(f"    Prefill 후(3,086 tok):  {snap.total_mb:.1f} MB")
    print(f"    Decode 완료(3,103 tok): ~{snap.total_mb + BASELINE_DECODE_STEPS * avg_delta_kb / 1024:.1f} MB")
    print(f"    step당 증가:            {avg_delta_kb:.1f} KB/step (이론: {theoretical_delta_kb:.1f} KB)")

    print(f"\n  [L2 분석 (32 MB)]")
    print(f"    전체 KV / L2: {snap.total_mb:.0f} MB / 32 MB = {snap.l2_coverage_pct:.0f}% → {snap.l2_coverage_pct/100:.1f}× 초과")
    print(f"    L2(24MB persist)에 올릴 수 있는 KV: {snap.tokens_fit_in_l2} tokens (all layers)")
    print(f"    1 layer 기준: {snap.tokens_fit_in_l2_per_layer} tokens")

    print(f"\n  [L2 Pinning 효과 예측]")
    bw = bw_implications
    print(f"    Flow KV 총 읽기: {bw['flow']['kv_total_mb_10steps']:.0f} MB "
          f"(이론 {bw['flow']['kv_theoretical_time_ms']:.0f}ms)")
    print(f"    Pinning 절약:    {bw['l2_pinning']['flow_savings_mb']:.0f} MB "
          f"({bw['l2_pinning']['flow_savings_pct']:.1f}%) → {bw['l2_pinning']['verdict']}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 5: main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  KV Cache 정밀 프로파일링")
    print(f"  device={DEVICE}  dtype=BF16  NUM_MEASURE={NUM_MEASURE}")
    print(f"  L2={L2_CACHE_BYTES//1024//1024}MB  DRAM_BW={DRAM_BW_GB_S}GB/s")
    print("=" * 70)

    # 모델 로드
    logger.info("모델 로드 중...")
    model = (
        Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(DEVICE)
        .eval()
    )
    logger.info("모델 로드 완료")

    # 입력 준비
    logger.info(f"입력 준비 중 (clip={CLIP_ID}, t={T0_US})...")
    processor = helper.get_processor(model.tokenizer)
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    raw = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    raw = helper.to_device(raw, DEVICE)
    ego = helper.to_device(
        {"ego_history_xyz": data["ego_history_xyz"],
         "ego_history_rot": data["ego_history_rot"]}, DEVICE
    )
    input_ids_raw = raw.pop("input_ids")
    with torch.no_grad():
        input_ids = model.fuse_traj_tokens(input_ids_raw, ego)
    torch.cuda.synchronize()
    tok_data = raw
    logger.info(f"input_ids: {input_ids.shape} ({input_ids.shape[1]} tokens)")

    # 프로파일링 실행
    all_results = profile_kv_cache(model, input_ids, tok_data)

    # 결과 저장
    all_results["meta"] = {
        "date": "2026-05-31",
        "device": DEVICE,
        "dtype": "bfloat16",
        "input_seq_len": input_ids.shape[1],
        "baseline_decode_steps": BASELINE_DECODE_STEPS,
        "l2_cache_bytes": L2_CACHE_BYTES,
        "dram_bw_gb_s": DRAM_BW_GB_S,
    }

    out_path = OUT / "kv_all_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))

    # 섹션별 저장
    for key in ["prefill", "decode_growth", "expert_kv", "l2_candidates",
                "bandwidth_implications"]:
        if key in all_results:
            p = OUT / f"kv_{key}.json"
            p.write_text(json.dumps(all_results[key], indent=2, default=str))

    logger.info(f"\n결과 저장 완료: {OUT}")
    print(f"\n  모든 결과: {OUT}/kv_all_results.json")


if __name__ == "__main__":
    main()
