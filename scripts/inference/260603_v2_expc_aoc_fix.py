"""
260603_v2_expc_aoc_fix.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[v1 → v2 수정 사항 (3가지 문제 동시 해결)]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
문제 1: AOC suffix 2× overhead (267ms vs DYN 140ms)

원인 확인:
  _k_buf shape = [1, 8, max_seq_len=3182, 128]
  suffix forward 중 AOC.update() 내부:
    k_out = _k_buf[:, :, :3086, :]  ← non-contiguous view
      stride[1] = 3182×128 ≠ 3086×128  → non-contiguous 판정
    k_out = k_out.contiguous()       ← 3086-token full copy 발생 (36 layers)
  이 overhead가 일관된 +115ms 유발

해결:
  suffix forward 전: aoc.force_contiguous = False
    → update()가 non-contiguous view를 그대로 반환
    → .contiguous() copy 없음 → suffix ~140ms (DYN과 동등)
    → FlashAttention이 non-contiguous K/V 처리 가능 (260531 AppendOnlyCache-B 검증됨)
  decode 직전: aoc.force_contiguous = True
    → decode에서만 .contiguous() → 79ms/step 목표

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
문제 2: AOC decode ≈ DYN decode (개선 측정 안됨)

원인 확인:
  v1 trial 순서: FULL → DYN decode → AOC decode (동일 trial 내 back-to-back)
  DYN decode가 L2를 warm하면 AOC decode도 동등하게 빨라짐
  → 두 경로 간 독립적 L2 상태 보장 불가

해결: 독립 trial 구조 도입
  WARMUP 1-5 (mode="both"):
    전체 경로 실행 (FULL + DYN + AOC) → JIT 완전 안정화
  MEASURE 홀수번 (mode="dyn"):
    t0_prefill → slice → suffix_DYN → decode_DYN
    DYN 단독으로 측정 → suffix 직후 L2 상태에서 decode
  MEASURE 짝수번 (mode="aoc"):
    t0_prefill → load_AOC → suffix_AOC(FC=False) → FC=True → decode_AOC
    AOC 단독으로 측정 → 동등한 L2 상태

  결과: DYN과 AOC가 각자 suffix 완료 직후의 동일한 L2 조건에서 decode 측정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
문제 3: EOS rate 67% (N=3 통계 불충분)

원인: N=3에서 1회 MAX_STEPS 도달 = 67%로 표시됨 (통계적 분산)

해결:
  NUM_MEASURE = 5 per path (DYN 5회 + AOC 5회 = 10회 총 측정)
  통계: mean + median + stdev 보고
  EOS rate도 N=5 기반으로 신뢰 구간 제공

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Δt = 100ms 고정 규칙 ★★★]
  Alpamayo = 10Hz (100ms) 추론 주기
  CoC (Chain of Causation) long-tail 인과추론 → 즉각 대응
  0.1초 내 장면이 결정적으로 달라질 수 있음
  → Δt = 100ms 단 하나만 유효. 절대 변경 불가.

[비교 방식]
  FULL: t1 full prefill → DynamicCache decode     (절대 기준선, warmup만)
  DYN : Exp C suffix → DynamicCache decode         (현재 상태)
  AOC : Exp C suffix → AppendOnlyCache-C decode    (통합 목표, FC 동적 전환)

[결과 파일]
  profiling_results/260603_v2_expc_aoc_fix/results.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLIP_ID          = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US            = 5_100_000          # 기준 프레임 (5.1s)
DELTA_T_MS       = 100                # ★ 100ms 고정 — 절대 변경 불가
T1_US            = T0_US + DELTA_T_MS * 1_000   # = 5_200_000 (5.2s)

DEVICE           = "cuda"
MAX_DECODE_STEPS = 80
TEMPERATURE      = 0.6
TOP_P            = 0.98

# ★ v2 변경: 더 많은 warmup + 독립 측정
NUM_WARMUP   = 5   # JIT 완전 안정화 (back-to-back full trial)
NUM_MEASURE  = 5   # per path: DYN 5회 + AOC 5회 = 총 10회 측정

OUT = Path("profiling_results/260603_v2_expc_aoc_fix")
OUT.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUDA 타이머
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CudaTimer:
    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._s.record()

    def stop_ms(self) -> float:
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AppendOnlyCache (260531 검증 완료)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AppendOnlyCache(DynamicCache):
    """
    DynamicCache 상속 + torch.cat → in-place write 교체.

    force_contiguous 동적 전환 (v2 핵심):
      suffix forward: force_contiguous=False
        → non-contiguous view 반환 → .contiguous() copy 없음 → suffix ~140ms
        → FlashAttention이 non-contiguous 처리 가능 (260531 AppendOnlyCache-B 검증)
      decode:         force_contiguous=True
        → .contiguous() copy 반환 → stride-optimal → ~79ms/step
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        force_contiguous: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers         = n_layers
        self.n_kv_heads       = n_kv_heads
        self.head_dim         = head_dim
        self.max_seq_len      = max_seq_len
        self.force_contiguous = force_contiguous
        self._write_pos: list[int] = [0] * n_layers

        self._k_buf: list[torch.Tensor] = []
        self._v_buf: list[torch.Tensor] = []
        for _ in range(n_layers):
            self._k_buf.append(
                torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim,
                            device=device, dtype=dtype)
            )
            self._v_buf.append(
                torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim,
                            device=device, dtype=dtype)
            )

        self.key_cache:   list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self,
        key_states:   torch.Tensor,
        value_states: torch.Tensor,
        layer_idx:    int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_new = key_states.shape[2]
        pos   = self._write_pos[layer_idx]

        if pos + n_new > self.max_seq_len:
            raise RuntimeError(
                f"AppendOnlyCache overflow: layer={layer_idx}, "
                f"pos={pos}, n_new={n_new}, max={self.max_seq_len}"
            )

        # in-place write
        self._k_buf[layer_idx][:, :, pos : pos + n_new, :] = key_states
        self._v_buf[layer_idx][:, :, pos : pos + n_new, :] = value_states
        self._write_pos[layer_idx] += n_new
        cur_len = self._write_pos[layer_idx]

        k_out = self._k_buf[layer_idx][:, :, :cur_len, :]
        v_out = self._v_buf[layer_idx][:, :, :cur_len, :]

        # ★ v2 핵심: force_contiguous 동적 전환 가능
        #   suffix 중 False → non-contiguous view 반환 (copy 없음)
        #   decode 중 True  → .contiguous() copy 반환 (FlashAttn 최적)
        if self.force_contiguous:
            k_out = k_out.contiguous()
            v_out = v_out.contiguous()

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)    # type: ignore[arg-type]
            self.value_cache.append(None)  # type: ignore[arg-type]
        self.key_cache[layer_idx]   = k_out
        self.value_cache[layer_idx] = v_out

        return k_out, v_out

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._write_pos[layer_idx]

    def get_max_length(self) -> int | None:
        return self.max_seq_len

    def reset(self) -> None:
        for i in range(self.n_layers):
            self._write_pos[i] = 0


def build_appendonly_cache(
    text_config: Any,
    prefill_len: int,
    max_decode: int,
    device: str,
    dtype: torch.dtype,
) -> AppendOnlyCache:
    n_layers    = text_config.num_hidden_layers
    n_kv_heads  = text_config.num_key_value_heads
    hidden_size = text_config.hidden_size
    n_q_heads   = text_config.num_attention_heads
    head_dim    = getattr(text_config, "head_dim", hidden_size // n_q_heads)
    max_seq_len = prefill_len + max_decode + 16

    alloc_mb = n_layers * 2 * n_kv_heads * max_seq_len * head_dim * 2 / 1e6
    logger.info(
        f"  [AOC] layers={n_layers}, kv_heads={n_kv_heads}, "
        f"head_dim={head_dim}, max_seq_len={max_seq_len}, alloc={alloc_mb:.1f}MB"
    )

    # ★ v2: force_contiguous=True (suffix 시 동적으로 False로 전환)
    return AppendOnlyCache(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=1,
        device=device,
        dtype=dtype,
        force_contiguous=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DynamicCache 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cache_to_kv_pairs(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if isinstance(kc, list) and kc and isinstance(kc[0], torch.Tensor):
        return list(zip(kc, vc))

    kc = getattr(cache, "_key_cache", None)
    vc = getattr(cache, "_value_cache", None)
    if isinstance(kc, list) and kc and isinstance(kc[0], torch.Tensor):
        return list(zip(kc, vc))

    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            if legacy and isinstance(legacy[0], (tuple, list)) and isinstance(legacy[0][0], torch.Tensor):
                return [(layer[0], layer[1]) for layer in legacy]
        except Exception:
            pass

    if (isinstance(cache, (tuple, list)) and cache
            and isinstance(cache[0], (tuple, list)) and len(cache[0]) == 2
            and isinstance(cache[0][0], torch.Tensor)):
        return [(layer[0], layer[1]) for layer in cache]

    attrs = [a for a in dir(cache) if not a.startswith("__")]
    kv_attrs = [a for a in attrs if any(x in a.lower() for x in ("key", "value", "kv", "cache"))]
    raise AttributeError(
        f"\n[cache 구조 불명] type={type(cache)}\n"
        f"  KV 관련 attr: {kv_attrs}"
    )


def _build_dynamic_cache(
        kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        seen_tokens: int,
) -> DynamicCache:
    new_cache = DynamicCache()
    new_cache._seen_tokens = seen_tokens
    if hasattr(new_cache, "key_cache") and isinstance(new_cache.key_cache, list):
        for k, v in kv_pairs:
            new_cache.key_cache.append(k)
            new_cache.value_cache.append(v)
        return new_cache
    if hasattr(new_cache, "update") and callable(new_cache.update):
        for i, (k, v) in enumerate(kv_pairs):
            new_cache.update(k, v, layer_idx=i)
        return new_cache
    raise RuntimeError(f"DynamicCache 구성 불가: {type(new_cache)}")


def slice_dynamic_cache(cache, end_pos: int) -> DynamicCache:
    pairs = _cache_to_kv_pairs(cache)
    sliced = [
        (k[:, :, :end_pos, :].clone().contiguous(),
         v[:, :, :end_pos, :].clone().contiguous())
        for k, v in pairs
    ]
    return _build_dynamic_cache(sliced, seen_tokens=end_pos)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# load_dyn_into_aoc
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dyn_into_aoc(
        dyn_cache: DynamicCache,
        aoc: AppendOnlyCache,
        end_pos: int,
        label: str = "",
) -> float:
    """
    DynamicCache의 앞 end_pos 토큰 KV를 AOC 버퍼에 로드.
    copy_() 1회로 중간 메모리 할당 없이 직접 복사.

    주의: 이 함수는 force_contiguous 상태에 관계없이 항상 copy_()를 사용.
          _k_buf에 직접 쓰기 때문에 _write_pos를 end_pos로 설정.
    """
    kv_pairs = _cache_to_kv_pairs(dyn_cache)

    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()

    for layer_idx, (k, v) in enumerate(kv_pairs):
        aoc._k_buf[layer_idx][:, :, :end_pos, :].copy_(k[:, :, :end_pos, :])
        aoc._v_buf[layer_idx][:, :, :end_pos, :].copy_(v[:, :, :end_pos, :])
        aoc._write_pos[layer_idx] = end_pos

        while len(aoc.key_cache) <= layer_idx:
            aoc.key_cache.append(None)    # type: ignore[arg-type]
            aoc.value_cache.append(None)  # type: ignore[arg-type]

        k_view = aoc._k_buf[layer_idx][:, :, :end_pos, :]
        v_view = aoc._v_buf[layer_idx][:, :, :end_pos, :]
        # load 시점의 force_contiguous 상태를 따름
        if aoc.force_contiguous:
            aoc.key_cache[layer_idx]   = k_view.contiguous()
            aoc.value_cache[layer_idx] = v_view.contiguous()
        else:
            aoc.key_cache[layer_idx]   = k_view
            aoc.value_cache[layer_idx] = v_view

    aoc._seen_tokens = end_pos

    ms = t.stop_ms()
    total_mb = (
        len(kv_pairs) * 2 * end_pos
        * kv_pairs[0][0].shape[1]
        * kv_pairs[0][0].shape[3]
        * 2 / 1e6
    )
    logger.info(
        f"  [{label}] load_dyn_into_aoc: {ms:.1f}ms  "
        f"({len(kv_pairs)} layers × {end_pos} tok = {total_mb:.0f}MB)"
    )
    return ms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vision 경계 탐지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_vision_regions(model: Any, input_ids: torch.Tensor) -> dict:
    ids   = input_ids[0].tolist()
    total = len(ids)

    img_tok_id = None
    for attr in ("image_token_id", "image_pad_token_id"):
        img_tok_id = getattr(model.vlm.config, attr, None)
        if img_tok_id is not None:
            break
    if img_tok_id is None:
        try:
            img_tok_id = model.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if img_tok_id == model.tokenizer.unk_token_id:
                img_tok_id = None
        except Exception:
            pass

    if img_tok_id is not None and img_tok_id in ids:
        positions = [i for i, t in enumerate(ids) if t == img_tok_id]
        vs = positions[0]
        ve = positions[-1] + 1
        logger.info(
            f"  [vision] image_token_id={img_tok_id}, "
            f"vision=[{vs},{ve}), suffix_len={total-ve}"
        )
        return {"vision_start": vs, "vision_end": ve,
                "suffix_len": total - ve, "total_len": total}

    vs_id = getattr(model.vlm.config, "vision_start_token_id", None)
    ve_id = getattr(model.vlm.config, "vision_end_token_id", None)
    if vs_id is not None and vs_id in ids:
        vs_positions = [i for i, t in enumerate(ids) if t == vs_id]
        ve_positions = [i for i, t in enumerate(ids) if t == ve_id]
        vs = vs_positions[0]
        ve = ve_positions[-1] + 1
        return {"vision_start": vs, "vision_end": ve,
                "suffix_len": total - ve, "total_len": total}

    logger.warning("  [vision] 자동 탐지 실패 → fallback (vision=[29,3011))")
    vs, ve = 29, 3011
    return {"vision_start": vs, "vision_end": ve,
            "suffix_len": total - ve, "total_len": total}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forward 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def full_prefill(
        model:     Any,
        input_ids: torch.Tensor,
        tok_data:  dict,
        label:     str = "",
) -> tuple[DynamicCache, torch.Tensor, float]:
    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=input_ids,
            attention_mask=tok_data.get("attention_mask"),
            pixel_values=tok_data.get("pixel_values"),
            image_grid_thw=tok_data.get("image_grid_thw"),
            use_cache=True,
        )
    ms = t.stop_ms()
    plen = int(input_ids.shape[1])
    logger.info(f"  [{label}] full_prefill: {ms:.0f}ms  ({plen} tokens)")
    return out.past_key_values, out.logits[:, -1, :].float(), ms


def suffix_forward(
        model:      Any,
        suffix_ids: torch.Tensor,
        past_kv:    Any,
        start_pos:  int,
        label:      str = "",
) -> tuple[Any, torch.Tensor, float]:
    """
    suffix_ids: [1, suffix_len] — ego + text_suffix
    past_kv: DynamicCache 또는 AppendOnlyCache (force_contiguous 상태 무관)
    start_pos: vision_end — RoPE 위치 정확성 필수

    ★ AOC with force_contiguous=False:
      update()가 non-contiguous view 반환 → FlashAttention이 처리
      .contiguous() copy 없음 → 140ms (DYN과 동등)
    """
    suffix_len = int(suffix_ids.shape[1])
    cache_pos = torch.arange(
        start_pos, start_pos + suffix_len,
        device=DEVICE, dtype=torch.long,
    )

    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=suffix_ids,
            pixel_values=None,
            past_key_values=past_kv,
            cache_position=cache_pos,
            use_cache=True,
        )
    ms = t.stop_ms()
    logger.info(
        f"  [{label}] suffix_forward: {ms:.0f}ms "
        f"({suffix_len}tok @ pos[{start_pos}..{start_pos+suffix_len-1}])"
    )
    return out.past_key_values, out.logits[:, -1, :].float(), ms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Top-p 샘플링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def top_p_sample(
        logits:      torch.Tensor,
        temperature: float = TEMPERATURE,
        top_p:       float = TOP_P,
) -> torch.Tensor:
    """Returns shape [batch] (1D)."""
    logits = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.zeros_like(logits)
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return torch.multinomial(F.softmax(filtered, dim=-1), 1).squeeze(-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decode Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def decode_loop(
        model:           Any,
        first_logits:    torch.Tensor,
        past_kv:         Any,
        prefill_len:     int,
        eos_id:          int,
        traj_offset:     int,
        traj_vocab_size: int,
        label:           str = "",
) -> dict | None:
    """
    자동회귀 디코딩 루프 (DynamicCache / AppendOnlyCache-C 모두 지원).

    ★ AOC decode 시 force_contiguous=True 보장 필수:
      decode 직전에 aoc.force_contiguous = True 설정 후 이 함수를 호출할 것.
      → update()가 .contiguous() copy 반환 → FlashAttn stride 최적 → ~79ms/step
    """
    lgts = first_logits.clone()
    lgts[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample(lgts)

    eos_found = False
    eos_step  = MAX_DECODE_STEPS
    cur       = next_tok.unsqueeze(1)  # [1, 1]

    if next_tok.item() == eos_id:
        return {"decode_ms": 0.0, "steps": 1, "ms_per_step": 0.0, "eos_ok": True}

    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()

    for step in range(1, MAX_DECODE_STEPS):
        cpos = torch.tensor(
            [prefill_len + step - 1], device=DEVICE, dtype=torch.long
        )
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.vlm(
                    input_ids=cur,
                    pixel_values=None,
                    past_key_values=past_kv,
                    cache_position=cpos,
                    use_cache=True,
                )
        except Exception as e:
            t.stop_ms()
            logger.error(f"  [{label}] step {step} 실패: {e}")
            traceback.print_exc()
            return None

        past_kv  = out.past_key_values
        lgts     = out.logits[:, -1, :].float()
        lgts[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample(lgts)
        cur      = next_tok.unsqueeze(1)

        if next_tok.item() == eos_id:
            eos_found = True
            eos_step  = step
            break

    ms = t.stop_ms()
    steps = eos_step + 1
    ms_per_step = ms / steps if steps > 0 else 0.0
    logger.info(
        f"  [{label}] decode: {ms:.0f}ms  "
        f"{steps}steps × {ms_per_step:.1f}ms/step  "
        f"eos={'✅' if eos_found else '❌ (MAX_STEPS)'}"
    )
    return {
        "decode_ms":   round(ms, 1),
        "steps":       steps,
        "ms_per_step": round(ms_per_step, 2),
        "eos_ok":      eos_found,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 입력 준비
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_inputs(model: Any, processor: Any, data: dict) -> tuple[torch.Tensor, dict]:
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs        = helper.to_device(inputs, DEVICE)
    input_ids_raw = inputs.pop("input_ids")

    ego_data = helper.to_device(
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        DEVICE,
    )
    input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
    return input_ids, inputs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통계 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def safe_mean(lst: list) -> float | None:
    lst = [x for x in lst if x is not None]
    return round(sum(lst) / len(lst), 2) if lst else None

def safe_median(lst: list) -> float | None:
    lst = sorted(x for x in lst if x is not None)
    n = len(lst)
    if n == 0:
        return None
    mid = n // 2
    v = (lst[mid - 1] + lst[mid]) / 2 if n % 2 == 0 else lst[mid]
    return round(v, 2)

def safe_stdev(lst: list) -> float | None:
    lst = [x for x in lst if x is not None]
    if len(lst) < 2:
        return None
    mu = sum(lst) / len(lst)
    variance = sum((x - mu) ** 2 for x in lst) / (len(lst) - 1)
    return round(math.sqrt(variance), 2)

def safe_min(lst: list) -> float | None:
    lst = [x for x in lst if x is not None]
    return round(min(lst), 2) if lst else None

def safe_max(lst: list) -> float | None:
    lst = [x for x in lst if x is not None]
    return round(max(lst), 2) if lst else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 Trial 실행 함수 (mode별)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_trial(
        model:           Any,
        input_ids_t0:    torch.Tensor,
        tok_data_t0:     dict,
        input_ids_t1:    torch.Tensor,
        tok_data_t1:     dict,
        suffix_ids_t1:   torch.Tensor,
        vision_end:      int,
        reuse_prefill_len: int,
        text_config:     Any,
        eos_id:          int,
        traj_offset:     int,
        traj_vocab_size: int,
        tag:             str,
        mode:            str,   # "both" | "dyn" | "aoc"
) -> dict | None:
    """
    단일 trial 실행.

    mode="both" (warmup):
      t0_prefill → t1_full_prefill + FULL_decode → DYN → AOC (전체 JIT 경로 실행)

    mode="dyn" (독립 DYN 측정):
      t0_prefill → slice → suffix_DYN → decode_DYN
      ★ DYN decode 직전 L2 상태: suffix_DYN 완료 직후 (DYN 자신의 계산만)

    mode="aoc" (독립 AOC 측정):
      t0_prefill → load_AOC → suffix_AOC(FC=False) → FC=True → decode_AOC
      ★ AOC decode 직전 L2 상태: suffix_AOC(FC=False) 완료 직후 (AOC 자신의 계산만)
      → DYN decode가 L2를 먼저 warm하는 오염 없음

    Returns:
      {"mode": ..., "dyn": {...} | None, "aoc": {...} | None, "full_prefill_ms": float | None}
    """
    torch.cuda.empty_cache()
    logger.info(f"\n  ─── [{tag}] mode={mode} ────────────────────────────────────")

    result: dict = {"mode": mode, "tag": tag, "dyn": None, "aoc": None, "full_prefill_ms": None}

    # ── Step 1: t0 full prefill (모든 mode 공통) ──────────────────────────
    try:
        kv_t0, _, pf_t0_ms = full_prefill(
            model, input_ids_t0, tok_data_t0,
            label=f"{tag}/t0_full",
        )
    except Exception as e:
        logger.error(f"[{tag}] t0_full_prefill 실패: {e}")
        traceback.print_exc()
        return None

    # ── FULL 기준선 (warmup만) ─────────────────────────────────────────────
    if mode == "both":
        try:
            kv_t1_full, logits_t1_full, pf_t1_ms = full_prefill(
                model, input_ids_t1, tok_data_t1,
                label=f"{tag}/t1_full",
            )
            result["full_prefill_ms"] = round(pf_t1_ms, 1)
        except Exception as e:
            logger.error(f"[{tag}] t1_full_prefill 실패: {e}")
            kv_t1_full = logits_t1_full = None
            result["full_prefill_ms"] = None

        if kv_t1_full is not None:
            dec_full = decode_loop(
                model, logits_t1_full, kv_t1_full, reuse_prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label=f"{tag}/FULL_decode",
            )
            if dec_full:
                logger.info(
                    f"  [{tag}] FULL: prefill={pf_t1_ms:.0f}ms  "
                    f"decode={dec_full['decode_ms']:.0f}ms/{dec_full['steps']}s"
                    f"/{dec_full['ms_per_step']:.1f}ms/step  "
                    f"eos={'✅' if dec_full['eos_ok'] else '❌'}"
                )

    # ── DYN 경로 ──────────────────────────────────────────────────────────
    if mode in ("both", "dyn"):
        # 4a: slice → DynamicCache(3011)
        kv_t0_dyn = slice_dynamic_cache(kv_t0, vision_end)

        # 4b: suffix forward (DYN)
        try:
            kv_dyn, logits_dyn, suf_dyn_ms = suffix_forward(
                model,
                suffix_ids=suffix_ids_t1,
                past_kv=kv_t0_dyn,
                start_pos=vision_end,
                label=f"{tag}/DYN_suffix",
            )
        except Exception as e:
            logger.error(f"[{tag}] DYN suffix_forward 실패: {e}")
            traceback.print_exc()
            kv_dyn = logits_dyn = suf_dyn_ms = None

        # 4c: DYN decode
        dec_dyn = None
        if kv_dyn is not None and logits_dyn is not None:
            dec_dyn = decode_loop(
                model, logits_dyn, kv_dyn, reuse_prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label=f"{tag}/DYN_decode",
            )

        if dec_dyn is not None:
            dyn_total = (suf_dyn_ms or 0) + dec_dyn["decode_ms"]
            logger.info(
                f"  [{tag}] DYN: suffix={suf_dyn_ms:.0f}ms  "
                f"decode={dec_dyn['decode_ms']:.0f}ms/{dec_dyn['steps']}s"
                f"/{dec_dyn['ms_per_step']:.1f}ms/step  "
                f"eos={'✅' if dec_dyn['eos_ok'] else '❌'}  "
                f"total={dyn_total:.0f}ms"
            )
            result["dyn"] = {
                "suf_ms":     round(suf_dyn_ms, 1) if suf_dyn_ms else None,
                "decode_ms":  dec_dyn["decode_ms"],
                "steps":      dec_dyn["steps"],
                "ms_per_step": dec_dyn["ms_per_step"],
                "eos_ok":     dec_dyn["eos_ok"],
                "total_ms":   round(dyn_total, 1),
            }

    # ── AOC 경로 (force_contiguous 동적 전환) ──────────────────────────────
    if mode in ("both", "aoc"):
        # 5a: AOC 빌드
        aoc = None
        try:
            aoc = build_appendonly_cache(
                text_config=text_config,
                prefill_len=reuse_prefill_len,
                max_decode=MAX_DECODE_STEPS,
                device=DEVICE,
                dtype=torch.bfloat16,
            )
        except Exception as e:
            logger.error(f"[{tag}] build_appendonly_cache 실패: {e}")
            traceback.print_exc()

        # 5b: DynCache[:vision_end] → AOC 버퍼에 로드
        #     load 시점: force_contiguous=True (load_dyn_into_aoc가 key_cache 설정)
        conv_ms = None
        if aoc is not None:
            try:
                conv_ms = load_dyn_into_aoc(
                    dyn_cache=kv_t0,
                    aoc=aoc,
                    end_pos=vision_end,
                    label=f"{tag}/AOC_load",
                )
            except Exception as e:
                logger.error(f"[{tag}] load_dyn_into_aoc 실패: {e}")
                traceback.print_exc()
                conv_ms = None

        # ★ [Fix 1] suffix forward 전 force_contiguous=False
        #   → AOC.update()가 non-contiguous view 반환 → copy 없음 → ~140ms
        suf_aoc_ms = None
        logits_aoc = None
        kv_aoc     = None
        if conv_ms is not None and aoc is not None:
            aoc.force_contiguous = False   # ← v2 핵심 수정
            try:
                kv_aoc, logits_aoc, suf_aoc_ms = suffix_forward(
                    model,
                    suffix_ids=suffix_ids_t1,
                    past_kv=aoc,
                    start_pos=vision_end,
                    label=f"{tag}/AOC_suffix(FC=F)",
                )
            except Exception as e:
                logger.error(f"[{tag}] AOC suffix_forward(FC=False) 실패: {e}")
                traceback.print_exc()
            finally:
                # ★ [Fix 1] decode 전 force_contiguous=True 복원
                aoc.force_contiguous = True
                if kv_aoc is not None:
                    # kv_aoc는 aoc와 동일 객체 (in-place 수정)이지만 안전을 위해 명시
                    if hasattr(kv_aoc, "force_contiguous"):
                        kv_aoc.force_contiguous = True

        # 5d: AOC decode (force_contiguous=True 확인됨)
        dec_aoc = None
        if kv_aoc is not None and logits_aoc is not None:
            # 검증: decode 시작 전 force_contiguous=True 확인
            fc_state = getattr(aoc, "force_contiguous", None)
            if fc_state is not True:
                logger.error(f"[{tag}] AOC decode 직전 force_contiguous={fc_state} — True여야 함!")
                aoc.force_contiguous = True

            dec_aoc = decode_loop(
                model, logits_aoc, kv_aoc, reuse_prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label=f"{tag}/AOC_decode(FC=T)",
            )

        if dec_aoc is not None:
            aoc_total = (conv_ms or 0) + (suf_aoc_ms or 0) + dec_aoc["decode_ms"]
            logger.info(
                f"  [{tag}] AOC: conv={conv_ms:.1f}ms  "
                f"suffix={suf_aoc_ms:.0f}ms  "
                f"decode={dec_aoc['decode_ms']:.0f}ms/{dec_aoc['steps']}s"
                f"/{dec_aoc['ms_per_step']:.1f}ms/step  "
                f"eos={'✅' if dec_aoc['eos_ok'] else '❌'}  "
                f"total={aoc_total:.0f}ms"
            )
            result["aoc"] = {
                "conv_ms":    round(conv_ms, 1) if conv_ms else None,
                "suf_ms":     round(suf_aoc_ms, 1) if suf_aoc_ms else None,
                "decode_ms":  dec_aoc["decode_ms"],
                "steps":      dec_aoc["steps"],
                "ms_per_step": dec_aoc["ms_per_step"],
                "eos_ok":     dec_aoc["eos_ok"],
                "total_ms":   round(aoc_total, 1),
            }

        # AOC-only trial에서 DYN과 step 차이 비교 (same trial에서 측정 시만 의미)
        if mode == "both" and result["dyn"] and result["aoc"]:
            dyn_step = result["dyn"]["ms_per_step"]
            aoc_step = result["aoc"]["ms_per_step"]
            diff = aoc_step - dyn_step
            sign = "+" if diff >= 0 else ""
            dyn_suf = result["dyn"]["suf_ms"] or 0
            aoc_suf = result["aoc"]["suf_ms"] or 0
            suf_diff = aoc_suf - dyn_suf
            logger.info(
                f"  [{tag}] suffix diff: AOC-DYN = {suf_diff:+.0f}ms  "
                f"decode diff: {sign}{diff:.1f}ms/step"
            )

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 본체
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(
        model:           Any,
        input_ids_t0:    torch.Tensor,
        tok_data_t0:     dict,
        input_ids_t1:    torch.Tensor,
        tok_data_t1:     dict,
        regions:         dict,
        eos_id:          int,
        traj_offset:     int,
        traj_vocab_size: int,
) -> dict:
    """
    v2 실험 구조:
      WARMUP 1-5      : mode="both" — 전체 경로 JIT 안정화
      MEASURE 홀수번  : mode="dyn"  — DYN 독립 측정 (총 NUM_MEASURE회)
      MEASURE 짝수번  : mode="aoc"  — AOC 독립 측정 (총 NUM_MEASURE회)
      총 측정 = NUM_MEASURE × 2 trials

    각 MEASURE trial은 t0_prefill → 자신의 suffix → 자신의 decode만 실행.
    상대 경로의 decode가 L2를 오염하지 않음 → 공정한 비교.
    """
    vision_end        = regions["vision_end"]
    prefill_len       = int(input_ids_t0.shape[1])
    suffix_ids_t1     = input_ids_t1[:, vision_end:]
    suffix_len        = int(suffix_ids_t1.shape[1])
    reuse_prefill_len = vision_end + suffix_len

    logger.info(
        f"  vision_end={vision_end}, suffix_len={suffix_len}, "
        f"prefill_len={prefill_len}, reuse_prefill_len={reuse_prefill_len}"
    )

    text_config = model.vlm.config.text_config

    trial_kwargs = dict(
        model=model,
        input_ids_t0=input_ids_t0,
        tok_data_t0=tok_data_t0,
        input_ids_t1=input_ids_t1,
        tok_data_t1=tok_data_t1,
        suffix_ids_t1=suffix_ids_t1,
        vision_end=vision_end,
        reuse_prefill_len=reuse_prefill_len,
        text_config=text_config,
        eos_id=eos_id,
        traj_offset=traj_offset,
        traj_vocab_size=traj_vocab_size,
    )

    # ── WARMUP (mode="both") ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  WARMUP ({NUM_WARMUP}회) — JIT 안정화, mode=both")
    print(f"{'='*72}")
    for i in range(NUM_WARMUP):
        run_trial(**trial_kwargs, tag=f"WARMUP {i+1}", mode="both")

    # ── MEASURE (독립 trial 교차) ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  MEASURE ({NUM_MEASURE}회 × 2 = {NUM_MEASURE*2}회 총 측정)")
    print(f"  홀수: mode=dyn (DYN 독립 측정)")
    print(f"  짝수: mode=aoc (AOC 독립 측정)")
    print(f"{'='*72}")

    dyn_runs: list[dict] = []
    aoc_runs: list[dict] = []

    for m in range(NUM_MEASURE * 2):
        mode = "dyn" if m % 2 == 0 else "aoc"
        measure_num = m // 2 + 1
        tag = f"MEASURE-{mode.upper()} {measure_num}"

        trial_result = run_trial(**trial_kwargs, tag=tag, mode=mode)
        if trial_result is None:
            logger.warning(f"[{tag}] trial 실패 — skip")
            continue

        if mode == "dyn" and trial_result.get("dyn"):
            dyn_runs.append(trial_result["dyn"])
            r = trial_result["dyn"]
            print(
                f"  [{tag}] DYN: suffix={r['suf_ms']}ms  "
                f"decode={r['decode_ms']:.0f}ms/{r['steps']}s/{r['ms_per_step']:.1f}ms/step  "
                f"eos={'✅' if r['eos_ok'] else '❌'}  total={r['total_ms']}ms"
            )

        if mode == "aoc" and trial_result.get("aoc"):
            aoc_runs.append(trial_result["aoc"])
            r = trial_result["aoc"]
            print(
                f"  [{tag}] AOC: conv={r['conv_ms']}ms  suffix={r['suf_ms']}ms  "
                f"decode={r['decode_ms']:.0f}ms/{r['steps']}s/{r['ms_per_step']:.1f}ms/step  "
                f"eos={'✅' if r['eos_ok'] else '❌'}  total={r['total_ms']}ms"
            )

    # ── 집계 ──────────────────────────────────────────────────────────────
    def aggregate(runs: list[dict], name: str) -> dict:
        if not runs:
            return {}
        suf        = [r["suf_ms"]      for r in runs]
        dec        = [r["decode_ms"]   for r in runs]
        step       = [r["ms_per_step"] for r in runs]
        total      = [r["total_ms"]    for r in runs]
        eos_count  = sum(1 for r in runs if r.get("eos_ok"))
        eos_rate   = eos_count / len(runs)

        agg = {
            "n":           len(runs),
            "eos_rate":    round(eos_rate, 3),
            "eos_count":   eos_count,
            "suffix": {
                "mean":   safe_mean(suf),
                "median": safe_median(suf),
                "stdev":  safe_stdev(suf),
                "min":    safe_min(suf),
                "max":    safe_max(suf),
            },
            "decode_ms_per_step": {
                "mean":   safe_mean(step),
                "median": safe_median(step),
                "stdev":  safe_stdev(step),
                "min":    safe_min(step),
                "max":    safe_max(step),
            },
            "decode_total_ms": {
                "mean":   safe_mean(dec),
                "median": safe_median(dec),
            },
            "total_ms": {
                "mean":   safe_mean(total),
                "median": safe_median(total),
                "stdev":  safe_stdev(total),
            },
        }
        if name == "aoc":
            conv = [r["conv_ms"] for r in runs]
            agg["conv_ms"] = {
                "mean":   safe_mean(conv),
                "median": safe_median(conv),
                "stdev":  safe_stdev(conv),
            }
        return agg

    dyn_agg = aggregate(dyn_runs, "dyn")
    aoc_agg = aggregate(aoc_runs, "aoc")

    # ── 최종 비교 출력 ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  ★ 최종 결과 (Δt=100ms, DYN {len(dyn_runs)}회 / AOC {len(aoc_runs)}회 독립 측정)")
    print(f"{'='*72}")

    def fmt_stat(d: dict, key: str = "mean") -> str:
        if not d or d.get(key) is None:
            return "N/A"
        return f"{d[key]:.1f}ms"

    if dyn_agg:
        da = dyn_agg
        print(
            f"  DYN : suffix={fmt_stat(da['suffix'])}±{fmt_stat(da['suffix'],'stdev')}  "
            f"decode={fmt_stat(da['decode_ms_per_step'])}±{fmt_stat(da['decode_ms_per_step'],'stdev')}/step  "
            f"(median={fmt_stat(da['decode_ms_per_step'],'median')})  "
            f"total={fmt_stat(da['total_ms'])}  EOS={da['eos_rate']*100:.0f}% ({da['eos_count']}/{da['n']})"
        )

    if aoc_agg:
        aa = aoc_agg
        conv_str = fmt_stat(aa.get("conv_ms", {})) if aa.get("conv_ms") else "N/A"
        print(
            f"  AOC : conv={conv_str}  "
            f"suffix={fmt_stat(aa['suffix'])}±{fmt_stat(aa['suffix'],'stdev')}  "
            f"decode={fmt_stat(aa['decode_ms_per_step'])}±{fmt_stat(aa['decode_ms_per_step'],'stdev')}/step  "
            f"(median={fmt_stat(aa['decode_ms_per_step'],'median')})  "
            f"total={fmt_stat(aa['total_ms'])}  EOS={aa['eos_rate']*100:.0f}% ({aa['eos_count']}/{aa['n']})"
        )

    if dyn_agg and aoc_agg:
        dyn_suf = dyn_agg["suffix"].get("mean") or 0
        aoc_suf = aoc_agg["suffix"].get("mean") or 0
        dyn_step = dyn_agg["decode_ms_per_step"].get("mean") or 0
        aoc_step = aoc_agg["decode_ms_per_step"].get("mean") or 0
        suf_diff = aoc_suf - dyn_suf
        step_diff = dyn_step - aoc_step
        print(f"\n  [suffix  ] AOC - DYN = {suf_diff:+.1f}ms  (목표: ≈0ms, v1: +115ms)")
        print(f"  [decode  ] DYN - AOC = {step_diff:+.1f}ms/step  (양수 = AOC 우세)")
        conv_mean = aoc_agg.get("conv_ms", {}).get("mean") or 0
        print(f"  [conv    ] {conv_mean:.1f}ms (1회성 AOC 로드 비용)")

        dyn_total_mean = dyn_agg["total_ms"].get("mean") or 0
        aoc_total_mean = aoc_agg["total_ms"].get("mean") or 0
        print(
            f"  [total   ] DYN={dyn_total_mean:.0f}ms  AOC={aoc_total_mean:.0f}ms  "
            f"diff={aoc_total_mean - dyn_total_mean:+.0f}ms"
        )
    print(f"{'='*72}")

    return {
        "experiment":  "260603_v2_expc_aoc_fix",
        "delta_t_ms":  DELTA_T_MS,
        "num_warmup":  NUM_WARMUP,
        "num_measure": NUM_MEASURE,
        "dyn_n":       len(dyn_runs),
        "aoc_n":       len(aoc_runs),
        "dyn_runs":    dyn_runs,
        "aoc_runs":    aoc_runs,
        "dyn_agg":     dyn_agg,
        "aoc_agg":     aoc_agg,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 72)
    print("  Exp C + AppendOnlyCache-C 통합 실험 v2 (3문제 수정)")
    print(f"  Δt = {DELTA_T_MS}ms (고정 규칙 — 절대 변경 불가)")
    print(f"  t0 = {T0_US/1e6:.1f}s  →  t1 = {T1_US/1e6:.1f}s")
    print("=" * 72)
    print()
    print("  [v2 수정사항]")
    print("  Fix 1: AOC suffix overhead")
    print("         force_contiguous=False (suffix) → True (decode)")
    print("         기대: AOC suffix ≈ DYN suffix (~140ms)")
    print()
    print("  Fix 2: 공정한 decode 비교")
    print("         DYN-only trial / AOC-only trial 교차 실행")
    print("         각 경로가 자신의 suffix 직후 L2 상태에서 decode 측정")
    print()
    print("  Fix 3: EOS 통계 강건화")
    print(f"         DYN {NUM_MEASURE}회 + AOC {NUM_MEASURE}회 = {NUM_MEASURE*2}회 독립 측정")
    print("         mean + median + stdev 보고")
    print()

    # ── 모델 로드 ─────────────────────────────────────────────────────────
    logger.info("모델 로드 중 (sdpa 기본값, BF16)...")
    model = (
        Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(DEVICE)
        .eval()
    )
    attn_impl = getattr(model.vlm.config, "attn_implementation", "unknown")
    logger.info(f"  → attn_implementation = {attn_impl}")

    eos_id          = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, traj_vocab_size={traj_vocab_size}")

    processor = helper.get_processor(model.tokenizer)

    # ── 데이터 로드 ───────────────────────────────────────────────────────
    logger.info(f"t0 데이터 로드 (T0={T0_US/1e6:.1f}s)...")
    raw_t0 = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    input_ids_t0, tok_data_t0 = prepare_inputs(model, processor, raw_t0)
    logger.info(f"  t0 input_ids: {input_ids_t0.shape}  ({input_ids_t0.shape[1]} tokens)")

    logger.info(f"t1 데이터 로드 (T1={T1_US/1e6:.1f}s, Δt={DELTA_T_MS}ms)...")
    raw_t1 = load_physical_aiavdataset(CLIP_ID, t0_us=T1_US)
    input_ids_t1, tok_data_t1 = prepare_inputs(model, processor, raw_t1)
    logger.info(f"  t1 input_ids: {input_ids_t1.shape}  ({input_ids_t1.shape[1]} tokens)")

    regions = detect_vision_regions(model, input_ids_t0)
    logger.info(
        f"  vision=[{regions['vision_start']},{regions['vision_end']}), "
        f"suffix_len={regions['suffix_len']}"
    )

    # ── 실험 실행 ─────────────────────────────────────────────────────────
    try:
        result = run_experiment(
            model=model,
            input_ids_t0=input_ids_t0,
            tok_data_t0=tok_data_t0,
            input_ids_t1=input_ids_t1,
            tok_data_t1=tok_data_t1,
            regions=regions,
            eos_id=eos_id,
            traj_offset=traj_offset,
            traj_vocab_size=traj_vocab_size,
        )
    except Exception as e:
        logger.error(f"실험 실패: {e}")
        traceback.print_exc()
        return

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"결과 저장: {out_path}")
    print(f"\n  결과 파일: {out_path}")


if __name__ == "__main__":
    main()
