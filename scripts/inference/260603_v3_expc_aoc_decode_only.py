"""
260603_v3_expc_aoc_decode_only.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[v1/v2 근본 원인 확정]

v1/v2에서 AOC suffix overhead(2×)와 EOS 60%는 같은 원인에서 비롯된다:

  AppendOnlyCache._k_buf[layer] = [1, 8, max_seq_len=3182, 128]
  suffix_forward 중 update() 반환값: _k_buf[:,:,:3086,:] (non-contiguous)
    stride[1] = 3182×128  ≠  3086×128  → non-contiguous

  문제 1: FlashAttention이 non-contiguous K/V를 처리할 때
    → 다른 CUDA 커널 경로 → 같은 데이터지만 다른 floating-point 누산 순서
    → 미세한 수치 차이가 36 layers에서 증폭 → 다른 logits
    → EOS rate 60% (DYN 100%와 다름)

  문제 2: non-contiguous K/V로 인한 성능 저하
    → FlashAttention 내부 implicit copy 또는 suboptimal kernel
    → 115ms overhead (FC=True/False 무관하게 동일)
    → force_contiguous 전환 (v2)이 효과 없는 이유

[v3 해결책]

AOC를 suffix_forward에서 완전히 배제한다.

  이전 (v1/v2):
    kv_t0[:3011] → AOC_buf[0:3011] → suffix_AOC → decode_AOC
                   ↑ AOC가 suffix에 개입 → non-contiguous → 2가지 문제 발생

  v3:
    Step 1. t0_prefill → kv_t0 (DynCache, 3086 tokens)
    Step 2. slice_dynamic_cache(kv_t0, 3011) → kv_t0_sliced (DynCache, 3011 tokens)
    Step 3. suffix_forward(kv_t0_sliced) → kv_dyn (DynCache, 3086 tokens), logits_dyn
            [DynCache suffix: 142ms, 정확한 logits, 100% EOS]
    Step 4. load_dyn_into_aoc(kv_dyn, aoc, end_pos=3086)
            [3086 tokens 전체를 AOC buffer에 복사: ~10ms]
            [aoc._write_pos[layer] = 3086]
            [aoc.key_cache[layer] = aoc._k_buf[:,:,:3086,:].contiguous()]
    Step 5a. DYN decode: decode_loop(kv_dyn, logits_dyn)
            [DynamicCache: torch.cat per step]
    Step 5b. AOC decode: decode_loop(aoc, logits_dyn)
            [AppendOnlyCache-C: in-place write per step, force_contiguous=True]

  핵심: DYN과 AOC decode 모두 logits_dyn(동일)에서 시작
    → 동일한 token 분포 → 동일한 decode 경로 → 100% EOS 일치
    → 순수 decode 구현 차이(torch.cat vs in-place+contiguous) 만 비교

[비교 방식]
  FULL : t1 full prefill → DynCache decode           (절대 기준선)
  DYN  : Exp C (DynCache suffix) → DynCache decode   (현재 상태)
  AOC  : Exp C (DynCache suffix) → AOC-C decode      (v3 목표)
         ↑ suffix는 DYN과 완전히 공유 → logits 동일 → EOS 동일

[예상 결과]
  suffix: DYN = AOC = ~142ms (공유, AOC overhead 없음)
  conv:   AOC only = ~10ms (3086 tokens load, 1회)
  decode ms/step: DYN ~75ms, AOC ~75-79ms
  EOS: DYN 100% = AOC 100% (동일 logits에서 시작)
  total: DYN ~1,582ms, AOC ~1,592ms (+10ms conv만 추가)

[Δt = 100ms 고정 규칙 ★★★]
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

CLIP_ID      = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US        = 5_100_000
DELTA_T_MS   = 100          # ★ 절대 고정
T1_US        = T0_US + DELTA_T_MS * 1_000

DEVICE       = "cuda"
MAX_DECODE_STEPS = 80
TEMPERATURE  = 0.6
TOP_P        = 0.98

NUM_WARMUP   = 5    # JIT 안정화
NUM_MEASURE  = 5    # per path (DYN 5 + AOC 5 = 10회 독립 측정)

OUT = Path("profiling_results/260603_v3_expc_aoc_decode_only")
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
# AppendOnlyCache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AppendOnlyCache(DynamicCache):
    """
    DynamicCache 상속 + torch.cat → in-place write 교체.

    v3에서 AOC는 decode ONLY에 사용한다.
    suffix_forward에는 절대 사용하지 않는다.

    load 시점: force_contiguous=True
      → load_dyn_into_aoc()에서 key_cache[layer]를 contiguous copy로 초기화
      → decode 중 update()도 contiguous copy 반환 → FlashAttn 최적

    decode 흐름:
      load_dyn_into_aoc(kv_dyn, aoc, end_pos=3086) → aoc._write_pos = 3086
      decode step k:
        update(k_1, v_1, layer) → _k_buf[layer][3086+k] = k_1 (in-place)
        → _k_buf[:,:,:3087+k,:].contiguous() → new compact tensor
        → FlashAttn uses this compact tensor
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

        self._k_buf[layer_idx][:, :, pos : pos + n_new, :] = key_states
        self._v_buf[layer_idx][:, :, pos : pos + n_new, :] = value_states
        self._write_pos[layer_idx] += n_new
        cur_len = self._write_pos[layer_idx]

        k_out = self._k_buf[layer_idx][:, :, :cur_len, :]
        v_out = self._v_buf[layer_idx][:, :, :cur_len, :]

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

    return AppendOnlyCache(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=1,
        device=device,
        dtype=dtype,
        force_contiguous=True,  # decode에만 사용, 항상 True
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
            if legacy and isinstance(legacy[0], (tuple, list)):
                return [(l[0], l[1]) for l in legacy]
        except Exception:
            pass
    if (isinstance(cache, (tuple, list)) and cache
            and isinstance(cache[0], (tuple, list)) and len(cache[0]) == 2):
        return [(l[0], l[1]) for l in cache]
    raise AttributeError(f"cache 구조 불명: type={type(cache)}")


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
    for i, (k, v) in enumerate(kv_pairs):
        new_cache.update(k, v, layer_idx=i)
    return new_cache


def slice_dynamic_cache(cache, end_pos: int) -> DynamicCache:
    """DynCache 앞 end_pos 토큰만 새 DynCache로 슬라이스 (clone+contiguous)."""
    pairs = _cache_to_kv_pairs(cache)
    sliced = [
        (k[:, :, :end_pos, :].clone().contiguous(),
         v[:, :, :end_pos, :].clone().contiguous())
        for k, v in pairs
    ]
    return _build_dynamic_cache(sliced, seen_tokens=end_pos)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# load_dyn_into_aoc  (v3: end_pos = 3086 — suffix 포함 전체 로드)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dyn_into_aoc(
        dyn_cache: DynamicCache,
        aoc: AppendOnlyCache,
        end_pos: int,
        label: str = "",
) -> float:
    """
    DynamicCache의 앞 end_pos 토큰 KV를 AOC 버퍼에 복사.

    [v3에서의 사용 방법]
    suffix_forward 완료 후 kv_dyn (DynCache, 3086 tokens)을 전달.
    end_pos = reuse_prefill_len = 3086 (vision + suffix 전체)
    → aoc._write_pos[layer] = 3086
    → aoc.key_cache[layer] = contiguous [1,8,3086,128] copy
    → 이후 decode가 position 3086부터 in-place append

    aoc.force_contiguous는 항상 True (decode 전용이므로).
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
        # force_contiguous=True (decode 전용)
        aoc.key_cache[layer_idx]   = k_view.contiguous()
        aoc.value_cache[layer_idx] = v_view.contiguous()

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
        vs, ve = positions[0], positions[-1] + 1
        logger.info(f"  [vision] image_token_id={img_tok_id}, vision=[{vs},{ve}), suffix_len={total-ve}")
        return {"vision_start": vs, "vision_end": ve, "suffix_len": total - ve, "total_len": total}
    vs_id = getattr(model.vlm.config, "vision_start_token_id", None)
    ve_id = getattr(model.vlm.config, "vision_end_token_id", None)
    if vs_id is not None and vs_id in ids:
        vs = [i for i, t in enumerate(ids) if t == vs_id][0]
        ve = [i for i, t in enumerate(ids) if t == ve_id][-1] + 1
        return {"vision_start": vs, "vision_end": ve, "suffix_len": total - ve, "total_len": total}
    logger.warning("  [vision] fallback: vision=[29,3011)")
    return {"vision_start": 29, "vision_end": 3011, "suffix_len": total - 3011, "total_len": total}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forward 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def full_prefill(model, input_ids, tok_data, label=""):
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
    logger.info(f"  [{label}] full_prefill: {ms:.0f}ms  ({input_ids.shape[1]} tokens)")
    return out.past_key_values, out.logits[:, -1, :].float(), ms


def suffix_forward(model, suffix_ids, past_kv, start_pos, label=""):
    """
    suffix_ids: [1, suffix_len]
    past_kv: DynamicCache only (v3에서 AOC는 절대 suffix에 사용 안 함)
    """
    suffix_len = int(suffix_ids.shape[1])
    cache_pos = torch.arange(start_pos, start_pos + suffix_len, device=DEVICE, dtype=torch.long)
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

def top_p_sample(logits, temperature=TEMPERATURE, top_p=TOP_P):
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

def decode_loop(model, first_logits, past_kv, prefill_len, eos_id,
                traj_offset, traj_vocab_size, label=""):
    lgts = first_logits.clone()
    lgts[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample(lgts)

    eos_found = False
    eos_step  = MAX_DECODE_STEPS
    cur       = next_tok.unsqueeze(1)

    if next_tok.item() == eos_id:
        return {"decode_ms": 0.0, "steps": 1, "ms_per_step": 0.0, "eos_ok": True}

    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()

    for step in range(1, MAX_DECODE_STEPS):
        cpos = torch.tensor([prefill_len + step - 1], device=DEVICE, dtype=torch.long)
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
            logger.error(f"  [{label}] step {step}: {e}")
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

def prepare_inputs(model, processor, data):
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    inputs        = helper.to_device(inputs, DEVICE)
    input_ids_raw = inputs.pop("input_ids")
    ego_data = helper.to_device(
        {"ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]},
        DEVICE,
    )
    return model.fuse_traj_tokens(input_ids_raw, ego_data), inputs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통계 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def safe_mean(lst):
    lst = [x for x in lst if x is not None]
    return round(sum(lst) / len(lst), 2) if lst else None

def safe_median(lst):
    lst = sorted(x for x in lst if x is not None)
    n = len(lst)
    if n == 0:
        return None
    mid = n // 2
    v = (lst[mid-1] + lst[mid]) / 2 if n % 2 == 0 else lst[mid]
    return round(v, 2)

def safe_stdev(lst):
    lst = [x for x in lst if x is not None]
    if len(lst) < 2:
        return None
    mu = sum(lst) / len(lst)
    return round(math.sqrt(sum((x-mu)**2 for x in lst) / (len(lst)-1)), 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 본체
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_trial(
        model, input_ids_t0, tok_data_t0, input_ids_t1, tok_data_t1,
        suffix_ids_t1, vision_end, reuse_prefill_len, text_config,
        eos_id, traj_offset, traj_vocab_size, tag, mode,
):
    """
    mode="warmup" : FULL + DYN + AOC 전체 실행 (JIT)
    mode="dyn"    : suffix(DYN) → DYN decode 독립 측정
    mode="aoc"    : suffix(DYN) → load_dyn_into_aoc(3086) → AOC decode 독립 측정

    ★ v3 핵심:
      DYN suffix와 AOC의 suffix는 완전히 동일한 경로(DynCache)를 사용.
      AOC에서 suffix_forward를 실행하지 않음.
      load_dyn_into_aoc(kv_dyn, end_pos=3086)으로 suffix 결과를 AOC에 이식.
      → AOC와 DYN은 동일한 logits에서 decode 시작 → EOS 100% 보장.
    """
    torch.cuda.empty_cache()
    logger.info(f"\n  ─── [{tag}] mode={mode} ────────────────────────────────────")

    result = {"mode": mode, "tag": tag, "full": None, "dyn": None, "aoc": None}

    # ── t0 full prefill (공통) ────────────────────────────────────────────
    try:
        kv_t0, _, pf_t0_ms = full_prefill(model, input_ids_t0, tok_data_t0,
                                           label=f"{tag}/t0_full")
    except Exception as e:
        logger.error(f"[{tag}] t0 prefill 실패: {e}")
        return None

    # ── FULL 기준선 (warmup만) ─────────────────────────────────────────────
    if mode == "warmup":
        try:
            kv_t1_full, logits_t1_full, pf_t1_ms = full_prefill(
                model, input_ids_t1, tok_data_t1, label=f"{tag}/t1_full")
            dec_full = decode_loop(model, logits_t1_full, kv_t1_full, reuse_prefill_len,
                                   eos_id, traj_offset, traj_vocab_size,
                                   label=f"{tag}/FULL_decode")
            if dec_full:
                logger.info(
                    f"  [{tag}] FULL: prefill={pf_t1_ms:.0f}ms  "
                    f"decode={dec_full['ms_per_step']:.1f}ms/step/{dec_full['steps']}s  "
                    f"eos={'✅' if dec_full['eos_ok'] else '❌'}"
                )
        except Exception as e:
            logger.error(f"[{tag}] FULL 실패: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [공통] Exp C suffix forward (DynCache — 항상 DynCache 사용)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # slice: t0 첫 3011 tokens (vision KV)
    kv_t0_sliced = slice_dynamic_cache(kv_t0, vision_end)

    # suffix forward: t1 suffix 75 tokens → kv_dyn(3086), logits_dyn
    kv_dyn = logits_dyn = suf_ms = None
    try:
        kv_dyn, logits_dyn, suf_ms = suffix_forward(
            model, suffix_ids_t1, kv_t0_sliced, vision_end,
            label=f"{tag}/suffix(DYN)",
        )
    except Exception as e:
        logger.error(f"[{tag}] suffix_forward 실패: {e}")
        traceback.print_exc()
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [DYN 경로] DynamicCache decode
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if mode in ("warmup", "dyn"):
        dec_dyn = decode_loop(
            model, logits_dyn, kv_dyn, reuse_prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"{tag}/DYN_decode",
        )
        if dec_dyn:
            dyn_total = (suf_ms or 0) + dec_dyn["decode_ms"]
            logger.info(
                f"  [{tag}] DYN: suffix={suf_ms:.0f}ms  "
                f"decode={dec_dyn['ms_per_step']:.1f}ms/step/{dec_dyn['steps']}s  "
                f"eos={'✅' if dec_dyn['eos_ok'] else '❌'}  total={dyn_total:.0f}ms"
            )
            result["dyn"] = {
                "suf_ms":      round(suf_ms, 1),
                "decode_ms":   dec_dyn["decode_ms"],
                "steps":       dec_dyn["steps"],
                "ms_per_step": dec_dyn["ms_per_step"],
                "eos_ok":      dec_dyn["eos_ok"],
                "total_ms":    round(dyn_total, 1),
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [AOC 경로] suffix DynCache 결과 3086 tokens → AOC → AOC decode
    # ★ v3 핵심: AOC는 decode만 담당, suffix에는 절대 사용 안 함
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if mode in ("warmup", "aoc"):
        # 5a: AOC 빌드
        aoc = None
        try:
            aoc = build_appendonly_cache(
                text_config=text_config,
                prefill_len=reuse_prefill_len,   # 3086
                max_decode=MAX_DECODE_STEPS,
                device=DEVICE,
                dtype=torch.bfloat16,
            )
        except Exception as e:
            logger.error(f"[{tag}] build_appendonly_cache 실패: {e}")

        # 5b: kv_dyn (DynCache, 3086 tokens) 전체를 AOC 버퍼에 복사
        #     ★ end_pos = reuse_prefill_len = 3086 (suffix 포함 전체)
        conv_ms = None
        if aoc is not None and kv_dyn is not None:
            try:
                conv_ms = load_dyn_into_aoc(
                    dyn_cache=kv_dyn,
                    aoc=aoc,
                    end_pos=reuse_prefill_len,  # ★ 3086 (v1/v2: 3011)
                    label=f"{tag}/AOC_load",
                )
            except Exception as e:
                logger.error(f"[{tag}] load_dyn_into_aoc 실패: {e}")
                traceback.print_exc()

        # 5c: AOC decode
        #     logits_dyn = DYN suffix의 last logits → DYN과 동일한 decode 시작
        #     → EOS = DYN과 동일 (100% 예상)
        dec_aoc = None
        if conv_ms is not None and logits_dyn is not None:
            dec_aoc = decode_loop(
                model, logits_dyn, aoc, reuse_prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label=f"{tag}/AOC_decode",
            )
        if dec_aoc is not None:
            # conv_ms는 suffix에서 DYN과 공유하는 비용이 아님
            # suffix는 DYN이 담당 → suf_ms는 DYN suffix 시간
            # AOC 추가 비용 = conv_ms만
            aoc_total = (suf_ms or 0) + (conv_ms or 0) + dec_aoc["decode_ms"]
            logger.info(
                f"  [{tag}] AOC: suffix(DYN)={suf_ms:.0f}ms  conv={conv_ms:.1f}ms  "
                f"decode={dec_aoc['ms_per_step']:.1f}ms/step/{dec_aoc['steps']}s  "
                f"eos={'✅' if dec_aoc['eos_ok'] else '❌'}  total={aoc_total:.0f}ms"
            )
            result["aoc"] = {
                "suf_ms":      round(suf_ms, 1),   # DYN suffix와 동일 (공유)
                "conv_ms":     round(conv_ms, 1),
                "decode_ms":   dec_aoc["decode_ms"],
                "steps":       dec_aoc["steps"],
                "ms_per_step": dec_aoc["ms_per_step"],
                "eos_ok":      dec_aoc["eos_ok"],
                "total_ms":    round(aoc_total, 1),
            }

        # warmup에서 DYN vs AOC 비교 (같은 trial)
        if mode == "warmup" and result["dyn"] and result["aoc"]:
            dyn_step = result["dyn"]["ms_per_step"]
            aoc_step = result["aoc"]["ms_per_step"]
            diff = dyn_step - aoc_step
            sign = "+" if diff >= 0 else ""
            logger.info(
                f"  [{tag}] decode diff: DYN-AOC = {sign}{diff:.1f}ms/step  "
                f"(양수 = AOC 우세)"
            )

    return result


def run_experiment(
        model, input_ids_t0, tok_data_t0, input_ids_t1, tok_data_t1,
        regions, eos_id, traj_offset, traj_vocab_size,
):
    vision_end        = regions["vision_end"]
    prefill_len       = int(input_ids_t0.shape[1])
    suffix_ids_t1     = input_ids_t1[:, vision_end:]
    suffix_len        = int(suffix_ids_t1.shape[1])
    reuse_prefill_len = vision_end + suffix_len

    logger.info(
        f"  vision_end={vision_end}, suffix_len={suffix_len}, "
        f"prefill_len={prefill_len}"
    )

    text_config = model.vlm.config.text_config

    kwargs = dict(
        model=model, input_ids_t0=input_ids_t0, tok_data_t0=tok_data_t0,
        input_ids_t1=input_ids_t1, tok_data_t1=tok_data_t1,
        suffix_ids_t1=suffix_ids_t1, vision_end=vision_end,
        reuse_prefill_len=reuse_prefill_len, text_config=text_config,
        eos_id=eos_id, traj_offset=traj_offset, traj_vocab_size=traj_vocab_size,
    )

    # ── WARMUP ────────────────────────────────────────────────────────────
    print(f"\n{'='*72}\n  WARMUP ({NUM_WARMUP}회) — JIT 안정화\n{'='*72}")
    for i in range(NUM_WARMUP):
        run_trial(**kwargs, tag=f"WARMUP {i+1}", mode="warmup")

    # ── MEASURE (독립 trial 교차) ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  MEASURE ({NUM_MEASURE}×2={NUM_MEASURE*2}회): 홀수=DYN, 짝수=AOC 독립 측정")
    print(f"{'='*72}")

    dyn_runs: list[dict] = []
    aoc_runs: list[dict] = []

    for m in range(NUM_MEASURE * 2):
        mode = "dyn" if m % 2 == 0 else "aoc"
        n = m // 2 + 1
        tag = f"MEASURE-{mode.upper()} {n}"

        trial = run_trial(**kwargs, tag=tag, mode=mode)
        if trial is None:
            logger.warning(f"[{tag}] skip")
            continue

        if mode == "dyn" and trial.get("dyn"):
            r = trial["dyn"]
            dyn_runs.append(r)
            print(
                f"  [{tag}] DYN: suffix={r['suf_ms']}ms  "
                f"decode={r['ms_per_step']:.1f}ms/step/{r['steps']}s  "
                f"eos={'✅' if r['eos_ok'] else '❌'}  total={r['total_ms']}ms"
            )
        if mode == "aoc" and trial.get("aoc"):
            r = trial["aoc"]
            aoc_runs.append(r)
            print(
                f"  [{tag}] AOC: conv={r['conv_ms']}ms  "
                f"decode={r['ms_per_step']:.1f}ms/step/{r['steps']}s  "
                f"eos={'✅' if r['eos_ok'] else '❌'}  total={r['total_ms']}ms"
            )

    # ── 집계 ──────────────────────────────────────────────────────────────
    def agg(runs, name):
        if not runs:
            return {}
        suf   = [r["suf_ms"]      for r in runs]
        dec   = [r["decode_ms"]   for r in runs]
        step  = [r["ms_per_step"] for r in runs]
        total = [r["total_ms"]    for r in runs]
        eos_n = sum(1 for r in runs if r.get("eos_ok"))
        d = {
            "n": len(runs),
            "eos_rate": round(eos_n / len(runs), 3),
            "eos_count": eos_n,
            "suffix":           {"mean": safe_mean(suf), "median": safe_median(suf), "stdev": safe_stdev(suf)},
            "decode_ms_per_step": {"mean": safe_mean(step), "median": safe_median(step), "stdev": safe_stdev(step)},
            "decode_total_ms":  {"mean": safe_mean(dec)},
            "total_ms":         {"mean": safe_mean(total), "median": safe_median(total), "stdev": safe_stdev(total)},
        }
        if name == "aoc":
            conv = [r["conv_ms"] for r in runs]
            d["conv_ms"] = {"mean": safe_mean(conv), "median": safe_median(conv)}
        return d

    dyn_agg = agg(dyn_runs, "dyn")
    aoc_agg = agg(aoc_runs, "aoc")

    # ── 최종 출력 ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  ★ 최종 결과 (Δt=100ms, DYN {len(dyn_runs)}회 / AOC {len(aoc_runs)}회 독립 측정)")
    print(f"{'='*72}")

    def fm(d, k="mean"):
        v = d.get(k) if isinstance(d, dict) else None
        return f"{v:.1f}ms" if v is not None else "N/A"

    if dyn_agg:
        da = dyn_agg
        print(
            f"  DYN : suffix={fm(da['suffix'])}  "
            f"decode={fm(da['decode_ms_per_step'])}±{fm(da['decode_ms_per_step'],'stdev')}/step  "
            f"(median={fm(da['decode_ms_per_step'],'median')})  "
            f"total={fm(da['total_ms'])}  EOS={da['eos_rate']*100:.0f}% ({da['eos_count']}/{da['n']})"
        )
    if aoc_agg:
        aa = aoc_agg
        print(
            f"  AOC : suffix(DYN공유)={fm(aa['suffix'])}  "
            f"conv={fm(aa.get('conv_ms',{}))}  "
            f"decode={fm(aa['decode_ms_per_step'])}±{fm(aa['decode_ms_per_step'],'stdev')}/step  "
            f"(median={fm(aa['decode_ms_per_step'],'median')})  "
            f"total={fm(aa['total_ms'])}  EOS={aa['eos_rate']*100:.0f}% ({aa['eos_count']}/{aa['n']})"
        )

    if dyn_agg and aoc_agg:
        ds = dyn_agg["decode_ms_per_step"].get("mean") or 0
        as_ = aoc_agg["decode_ms_per_step"].get("mean") or 0
        conv = aoc_agg.get("conv_ms", {}).get("mean") or 0
        dt = dyn_agg["total_ms"].get("mean") or 0
        at = aoc_agg["total_ms"].get("mean") or 0
        print(f"\n  [decode ] DYN={ds:.1f}ms/step  AOC={as_:.1f}ms/step  diff={ds-as_:+.1f}ms/step (양수=AOC 우세)")
        print(f"  [conv   ] {conv:.1f}ms (AOC 1회성 로드)")
        print(f"  [total  ] DYN={dt:.0f}ms  AOC={at:.0f}ms  diff={at-dt:+.0f}ms")
        print(f"\n  [EOS 일치 여부]")
        dyn_eos = dyn_agg.get("eos_count", 0)
        aoc_eos = aoc_agg.get("eos_count", 0)
        dyn_n   = dyn_agg.get("n", 0)
        aoc_n   = aoc_agg.get("n", 0)
        print(f"    DYN: {dyn_eos}/{dyn_n}  AOC: {aoc_eos}/{aoc_n}")
        print(f"    ★ DYN/AOC는 동일한 logits에서 시작 → EOS rate 동일 예상")
    print(f"{'='*72}")

    return {
        "experiment":  "260603_v3_expc_aoc_decode_only",
        "delta_t_ms":  DELTA_T_MS,
        "num_warmup":  NUM_WARMUP,
        "num_measure": NUM_MEASURE,
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
    print("  Exp C + AppendOnlyCache-C 통합 실험 v3")
    print(f"  Δt = {DELTA_T_MS}ms (고정 규칙)")
    print(f"  t0 = {T0_US/1e6:.1f}s  →  t1 = {T1_US/1e6:.1f}s")
    print("=" * 72)
    print()
    print("  [v3 근본 원인 수정]")
    print("  AOC를 suffix_forward에서 완전히 배제")
    print("  suffix는 항상 DynCache 사용 → 올바른 logits 보장")
    print("  load_dyn_into_aoc(kv_dyn, end_pos=3086) → AOC에 3086 tokens 로드")
    print("  AOC decode: logits_dyn 공유 → DYN과 동일한 decode 경로 → EOS 100% 예상")
    print()

    logger.info("모델 로드 중...")
    model = (
        Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, local_files_only=True,
        ).to(DEVICE).eval()
    )
    logger.info(f"  → attn_implementation = {getattr(model.vlm.config, 'attn_implementation', 'unknown')}")

    eos_id          = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, traj_vocab_size={traj_vocab_size}")

    processor = helper.get_processor(model.tokenizer)

    logger.info(f"t0 데이터 로드 (T0={T0_US/1e6:.1f}s)...")
    raw_t0 = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    input_ids_t0, tok_data_t0 = prepare_inputs(model, processor, raw_t0)
    logger.info(f"  t0: {input_ids_t0.shape}")

    logger.info(f"t1 데이터 로드 (T1={T1_US/1e6:.1f}s)...")
    raw_t1 = load_physical_aiavdataset(CLIP_ID, t0_us=T1_US)
    input_ids_t1, tok_data_t1 = prepare_inputs(model, processor, raw_t1)
    logger.info(f"  t1: {input_ids_t1.shape}")

    regions = detect_vision_regions(model, input_ids_t0)
    logger.info(f"  vision=[{regions['vision_start']},{regions['vision_end']}), suffix={regions['suffix_len']}")

    try:
        result = run_experiment(
            model=model,
            input_ids_t0=input_ids_t0, tok_data_t0=tok_data_t0,
            input_ids_t1=input_ids_t1, tok_data_t1=tok_data_t1,
            regions=regions,
            eos_id=eos_id, traj_offset=traj_offset, traj_vocab_size=traj_vocab_size,
        )
    except Exception as e:
        logger.error(f"실험 실패: {e}")
        traceback.print_exc()
        return

    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"결과 저장: {out_path}")
    print(f"\n  결과 파일: {out_path}")


if __name__ == "__main__":
    main()
