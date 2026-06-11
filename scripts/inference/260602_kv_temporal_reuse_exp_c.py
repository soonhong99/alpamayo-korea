"""
260602_kv_temporal_reuse_exp_c.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[목적]
  KV Temporal Reuse Experiment C: 인접 프레임 Vision KV 재사용 검증
  "t0의 Vision KV를 t1 추론에 그대로 쓰면 올바른 결과가 나오는가?"

[배경]
  260528 실험 A: 동일 프레임 KV 재사용 → prefill 2,438ms → 107ms (22.8×) ✅
  260528 실험 B: text_prefix KV 재사용 → ~46ms 절약 (3.2%) ✅
  260602 실험 C: t0 vision KV를 t1에 그대로 사용 → suffix만 forward

  핵심 아이디어:
    연속 드라이빙 프레임에서 시각 장면은 100ms~500ms 사이에 거의 변하지 않는다.
    t0의 vision KV(2890 토큰)를 t1에서 재사용하고,
    변경된 부분(ego 82tok + text_suffix 14tok = ~96 tok)만 새로 forward하면
    prefill 비용이 1,423ms → ~44ms 로 줄어든다 (97% 절감, 이론값).

[실험 설계]
  입력 구조 (3,086 토큰):
    [text_prefix: ~29tok] [vision: ~2,890tok] [ego: ~82tok] [text_suffix: ~14tok]
                          ↑                  ↑
                     vision_start        vision_end
                     (이미지 패치)        (suffix 시작)

  t0 full prefill → KV_t0 (vision_end까지 유지)
  t1 suffix forward:
    input_ids = t1의 입력[vision_end:]  (ego+text_suffix, ~96 토큰)
    pixel_values = None                  (이미지 패치 토큰 없음)
    past_key_values = KV_t0[:vision_end] (t0의 text_prefix+vision KV 재사용)
    cache_position = [vision_end, ..., vision_end+96-1]

[다중 Δt 실험]
  Δt = 100ms, 300ms, 500ms, 1000ms 에서 각각 측정:
  → "Δt가 클수록 시각 유사도가 낮아지고 품질이 저하되는가?" 검증
  → 적응형 임계값 설계의 근거 데이터 수집

[측정 지표]
  1. 시간 절약: suffix_prefill vs t1_full_prefill
  2. EOS 정상 생성 여부 (reuse decode가 MAX_DECODE_STEPS에 걸리지 않는가)
  3. Decode step 수 차이 (full vs reuse)
  4. Vision KV 코사인 유사도 (t0 vs t1 KV@layer0, vision region)

[성공 기준]
  ✅ suffix_prefill < 200ms
  ✅ EOS 정상 생성
  ✅ |reuse_steps - full_steps| ≤ 5
  ✅ 절약 > 80%

[알려진 Thor 이슈 및 대응]
  1. DynamicCache._seen_tokens 미초기화: _build_cache_from_kv에서 명시 설정
  2. cache_position 필수: 모든 forward 호출에 명시적으로 전달
  3. torch.autocast: dtype 불일치 방지
  4. NUM_WARMUP=3: JIT 안정화 (260601 실험에서 확인)

[실행]
  source ~/alpamayo1.5/a1_5_venv/bin/activate && cd ~/alpamayo1.5
  python3 scripts/inference/260602_kv_temporal_reuse_exp_c.py [--delta-t 100 300 500 1000]

[결과]
  profiling_results/260602_kv_temporal_reuse_c/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
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

CLIP_ID           = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US             = 5_100_000          # 기준 프레임
DEVICE            = "cuda"
MAX_DECODE_STEPS  = 80                 # EOS 미생성 시 최대 step
EOS_CHECK_EVERY   = 4                  # EOS 체크 주기
TEMPERATURE       = 0.6
TOP_P             = 0.98
NUM_WARMUP        = 3                  # ★ 260601 실험: JIT 안정화에 3회 필요
NUM_MEASURE       = 3

# 기준선 (DynamicCache, sdpa, BF16, 2026-05-28 측정)
BASELINE_VE_MS      = 728.0
BASELINE_PREFILL_MS = 1423.0
BASELINE_DECODE_MS  = 1818.0
BASELINE_FLOW_MS    = 870.0

OUT = Path("profiling_results/260602_kv_temporal_reuse_c")
OUT.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 타이머
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CudaTimer:
    """CUDA 이벤트 기반 정밀 타이머."""

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
# DynamicCache 유틸리티
# (260528 스크립트의 함수를 개선 — _seen_tokens 명시적 설정 추가)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cache_to_kv_pairs(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    DynamicCache에서 레이어별 (K, V) 텐서 쌍 추출.
    transformers 버전 무관 (4종 폴백).
    shape: [batch, heads, seq_len, head_dim]
    """
    # 방법 1: key_cache / value_cache (transformers 4.38+)
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if isinstance(kc, list) and kc and isinstance(kc[0], torch.Tensor):
        return list(zip(kc, vc))

    # 방법 2: private _key_cache / _value_cache
    kc = getattr(cache, "_key_cache", None)
    vc = getattr(cache, "_value_cache", None)
    if isinstance(kc, list) and kc and isinstance(kc[0], torch.Tensor):
        return list(zip(kc, vc))

    # 방법 3: to_legacy_cache()
    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            if legacy and isinstance(legacy[0], (tuple, list)) and isinstance(legacy[0][0], torch.Tensor):
                return [(layer[0], layer[1]) for layer in legacy]
        except Exception:
            pass

    # 방법 4: tuple-of-tuples
    if (isinstance(cache, (tuple, list)) and cache
            and isinstance(cache[0], (tuple, list)) and len(cache[0]) == 2
            and isinstance(cache[0][0], torch.Tensor)):
        return [(layer[0], layer[1]) for layer in cache]

    attrs = [a for a in dir(cache) if not a.startswith("__")]
    kv_attrs = [a for a in attrs if any(x in a.lower() for x in ("key", "value", "kv", "cache"))]
    raise AttributeError(
        f"\n[cache 구조 불명] type={type(cache)}\n"
        f"  KV 관련 attr: {kv_attrs}\n"
        "  → 이 정보를 공유하면 폴백 분기를 추가합니다."
    )


def _build_cache_from_kv(
        kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        seen_tokens: int = 0,
) -> DynamicCache:
    """
    (K, V) 쌍 리스트로 DynamicCache 구성.

    ★ 260602 수정: _seen_tokens를 명시적으로 설정한다.
      이유: Thor의 transformers는 DynamicCache.__init__()에서
            _seen_tokens를 초기화하지 않는다.
            → AttributeError 방지 및 position encoding 정확성 보장.

    seen_tokens: 이 캐시가 표현하는 토큰 수
                 (보통 key.shape[2] = seq_len과 동일)
    """
    new_cache = DynamicCache()

    # _seen_tokens 명시 설정 (Thor transformers 호환)
    new_cache._seen_tokens = seen_tokens

    # 방법 1: key_cache/value_cache 직접 설정
    if hasattr(new_cache, "key_cache") and isinstance(new_cache.key_cache, list):
        for k, v in kv_pairs:
            new_cache.key_cache.append(k)
            new_cache.value_cache.append(v)
        return new_cache

    # 방법 2: update() 메서드
    if hasattr(new_cache, "update") and callable(new_cache.update):
        for i, (k, v) in enumerate(kv_pairs):
            new_cache.update(k, v, layer_idx=i)
        return new_cache

    raise RuntimeError(
        f"DynamicCache 구성 불가: {type(new_cache)}\n"
        f"attrs: {[a for a in dir(new_cache) if not a.startswith('__')]}"
    )


def slice_cache_to(cache, end_pos: int) -> DynamicCache:
    """
    cache에서 앞 end_pos 토큰의 KV만 잘라 새 DynamicCache 반환.
    _seen_tokens = end_pos 로 명시 설정.

    용도: KV_t0[:vision_end] → suffix forward의 past_key_values
    """
    pairs = _cache_to_kv_pairs(cache)
    sliced = [
        (k[:, :, :end_pos, :].clone().contiguous(),
         v[:, :, :end_pos, :].clone().contiguous())
        for k, v in pairs
    ]
    return _build_cache_from_kv(sliced, seen_tokens=end_pos)


def get_cache_seq_len(cache) -> int:
    """캐시의 현재 시퀀스 길이."""
    if hasattr(cache, "get_seq_length") and callable(cache.get_seq_length):
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


def log_cache_info(cache, label: str = "") -> None:
    """캐시 상태 로그 출력."""
    try:
        pairs = _cache_to_kv_pairs(cache)
        if not pairs:
            logger.info(f"  [{label}] cache: empty")
            return
        k0 = pairs[0][0]
        seq_len = k0.shape[2]
        mem_mb = sum(
            (k.element_size() * k.numel() + v.element_size() * v.numel()) / 1e6
            for k, v in pairs
        )
        seen = getattr(cache, "_seen_tokens", "?")
        logger.info(
            f"  [{label}] cache: {len(pairs)}L, seq={seq_len}, "
            f"shape=[{k0.shape[0]},{k0.shape[1]},{seq_len},{k0.shape[3]}], "
            f"mem={mem_mb:.0f}MB, _seen_tokens={seen}"
        )
    except Exception as e:
        logger.warning(f"  [{label}] cache info 출력 실패: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vision 경계 탐지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_vision_regions(model: Any, input_ids: torch.Tensor) -> dict:
    """
    input_ids에서 vision 토큰 구간을 자동 탐지한다.

    Qwen3VL 토큰 구조:
      [text_prefix] [vision_start] [image_pad × N] [vision_end] [ego] [text_suffix]

    Returns: {
      text_prefix_len: vision 시작 전 순수 텍스트 수
      vision_start:    첫 이미지 패치 토큰 위치
      vision_end:      마지막 이미지 패치 토큰 바로 다음
      vision_len:      vision_end - vision_start
      suffix_start:    vision_end (ego+text_suffix 시작)
      suffix_len:      전체 - vision_end
      total_len:       전체 토큰 수
    }
    """
    ids = input_ids[0].tolist()
    total = len(ids)

    # ── 방법 1: image_token_id (이미지 패치 토큰 직접 탐지) ──────────────────
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
            f"  [vision_detect] image_token_id={img_tok_id}, "
            f"vision=[{vs},{ve}), len={ve-vs}, suffix_len={total-ve}"
        )
        return {
            "text_prefix_len": vs,
            "vision_start":    vs,
            "vision_end":      ve,
            "vision_len":      ve - vs,
            "suffix_start":    ve,
            "suffix_len":      total - ve,
            "total_len":       total,
            "n_image_tokens":  len(positions),
            "method":          f"image_token_id={img_tok_id}",
        }

    # ── 방법 2: vision_start/end special token ID ─────────────────────────
    vs_id = getattr(model.vlm.config, "vision_start_token_id", None)
    ve_id = getattr(model.vlm.config, "vision_end_token_id", None)
    if vs_id is not None and vs_id in ids:
        vs_positions = [i for i, t in enumerate(ids) if t == vs_id]
        ve_positions = [i for i, t in enumerate(ids) if t == ve_id]
        vs = vs_positions[0]
        ve = ve_positions[-1] + 1
        logger.info(
            f"  [vision_detect] vision_start_id={vs_id}, "
            f"vision=[{vs},{ve}), suffix_len={total-ve}"
        )
        return {
            "text_prefix_len": vs,
            "vision_start":    vs,
            "vision_end":      ve,
            "vision_len":      ve - vs,
            "suffix_start":    ve,
            "suffix_len":      total - ve,
            "total_len":       total,
            "n_image_tokens":  ve - vs - len(vs_positions) * 2,
            "method":          f"vision_start_token_id={vs_id}",
        }

    # ── 방법 3: 260528 실험 B에서 역산한 추정값 (fallback) ───────────────────
    logger.warning(
        "  [vision_detect] 자동 탐지 실패 → 추정값 사용 "
        "(vision_start=29, vision_len=2880)"
    )
    vs = 29
    ve = 29 + 2880
    return {
        "text_prefix_len": vs,
        "vision_start":    vs,
        "vision_end":      ve,
        "vision_len":      2880,
        "suffix_start":    ve,
        "suffix_len":      total - ve,
        "total_len":       total,
        "n_image_tokens":  2880,
        "method":          "fallback(260528_exp_b_estimate)",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forward 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def full_prefill(
        model: Any,
        input_ids: torch.Tensor,
        tok_data: dict,
        label: str = "",
) -> tuple[Any, torch.Tensor, float, int]:
    """
    표준 full prefill (pixel_values 포함, DynamicCache).
    Returns: (past_kv, last_logits, prefill_ms, prefill_len)
    """
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
    return out.past_key_values, out.logits[:, -1, :].float(), ms, plen


def suffix_prefill(
        model: Any,
        suffix_ids: torch.Tensor,
        past_kv: Any,
        start_pos: int,
        label: str = "",
) -> tuple[Any, torch.Tensor, float]:
    """
    Suffix-only forward: pixel_values=None, cache_position 명시.

    suffix_ids:  [1, suffix_len]  — ego + text_suffix 토큰 (이미지 패치 없음)
    past_kv:     KV_t0[:vision_end] — 이미 있는 t0의 KV
    start_pos:   vision_end — suffix 토큰의 시작 절대 위치

    ★ 핵심: cache_position을 명시적으로 전달해야
            RoPE(회전 위치 인코딩)가 올바른 위치에서 계산됨.
            미전달 시 suffix 토큰이 position 0부터 계산되어 attention 오류 발생.
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
            pixel_values=None,          # ← 이미지 패치 토큰이 없으므로 반드시 None
            past_key_values=past_kv,
            cache_position=cache_pos,   # ← 반드시 명시 (RoPE 위치 정확성)
            use_cache=True,
        )
    ms = t.stop_ms()
    logger.info(
        f"  [{label}] suffix_prefill: {ms:.0f}ms  "
        f"({suffix_len}tok @ pos[{start_pos}..{start_pos+suffix_len-1}])"
    )
    return out.past_key_values, out.logits[:, -1, :].float(), ms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Top-p 샘플링 및 Decode Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def top_p_sample(logits: torch.Tensor,
                 temperature: float = TEMPERATURE,
                 top_p: float = TOP_P) -> torch.Tensor:
    logits = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.zeros_like(logits)
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return torch.multinomial(F.softmax(filtered, dim=-1), 1).squeeze(-1)


def decode_loop(
        model: Any,
        first_logits: torch.Tensor,
        past_kv: Any,
        prefill_len: int,
        eos_id: int,
        traj_offset: int,
        traj_vocab_size: int,
        label: str = "",
) -> dict | None:
    """
    자동회귀 디코딩 루프.

    prefill_len: 현재 KV 캐시에 채워진 토큰 수
                 → 다음 토큰의 cache_position = prefill_len
    """
    lgts = first_logits.clone()
    lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample(lgts)

    eos_found = False
    eos_step = MAX_DECODE_STEPS
    buf = [next_tok.item()]
    cur = next_tok.unsqueeze(1)

    if next_tok.item() == eos_id:
        return {"decode_ms": 0.0, "steps": 1, "ms_per_step": 0.0, "eos_ok": True}

    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()

    for step in range(1, MAX_DECODE_STEPS):
        # ★ cache_position 명시: prefill_len + (현재 step - 1)
        # step=1이면 prefill_len 위치에 쓰는 것이 맞음
        cpos = torch.tensor(
            [prefill_len + step - 1], device=DEVICE, dtype=torch.long
        )
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.vlm(
                    input_ids=cur,
                    past_key_values=past_kv,
                    cache_position=cpos,
                    use_cache=True,
                )
        except Exception as e:
            ms = t.stop_ms()
            logger.error(f"  [{label}] step {step} 실패: {e}")
            return None

        past_kv = out.past_key_values
        lgts = out.logits[:, -1, :].float()
        lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample(lgts)
        buf.append(next_tok.item())
        cur = next_tok.unsqueeze(1)

        if next_tok.item() == eos_id:
            eos_found = True
            eos_step = step
            break

        if step % EOS_CHECK_EVERY == 0 and eos_found:
            break

    ms = t.stop_ms()
    steps = eos_step + 1
    logger.info(
        f"  [{label}] decode: {ms:.0f}ms  "
        f"{steps}steps × {ms/steps:.1f}ms/step  "
        f"eos={'✅' if eos_found else '❌ (MAX_STEPS)'}"
    )
    return {
        "decode_ms":    round(ms, 1),
        "steps":        steps,
        "ms_per_step":  round(ms / steps, 2),
        "eos_ok":       eos_found,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KV 유사도 측정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_kv_similarity(
        kv_t0: Any,
        kv_t1: Any,
        vision_start: int,
        vision_end: int,
        n_layers_to_check: int = 4,
) -> dict:
    """
    두 KV Cache의 vision 구간 코사인 유사도를 계산한다.

    왜 이것이 중요한가:
      - 유사도가 높으면 (>0.95): t0 vision KV를 t1에 재사용해도 품질 유지 예상
      - 유사도가 낮으면 (<0.80): 장면이 크게 바뀌었음 → full prefill 필요

    n_layers_to_check: 계산 비용 절약을 위해 대표 레이어만 샘플링
    """
    try:
        pairs_t0 = _cache_to_kv_pairs(kv_t0)
        pairs_t1 = _cache_to_kv_pairs(kv_t1)
        n_layers = min(len(pairs_t0), len(pairs_t1))

        layer_indices = list(range(0, n_layers,
                                   max(1, n_layers // n_layers_to_check)))[:n_layers_to_check]

        sims_k = []
        sims_v = []

        for li in layer_indices:
            k0 = pairs_t0[li][0][:, :, vision_start:vision_end, :].float()
            k1 = pairs_t1[li][0][:, :, vision_start:vision_end, :].float()
            v0 = pairs_t0[li][1][:, :, vision_start:vision_end, :].float()
            v1 = pairs_t1[li][1][:, :, vision_start:vision_end, :].float()

            # 전체 vision region을 하나의 벡터로 flatten하여 코사인 유사도
            sk = F.cosine_similarity(k0.flatten().unsqueeze(0),
                                     k1.flatten().unsqueeze(0)).item()
            sv = F.cosine_similarity(v0.flatten().unsqueeze(0),
                                     v1.flatten().unsqueeze(0)).item()
            sims_k.append(round(sk, 4))
            sims_v.append(round(sv, 4))

        avg_k = sum(sims_k) / len(sims_k)
        avg_v = sum(sims_v) / len(sims_v)
        avg   = (avg_k + avg_v) / 2.0

        logger.info(
            f"  [kv_sim] layers{layer_indices}: "
            f"K_sim={avg_k:.4f}, V_sim={avg_v:.4f}, avg={avg:.4f}"
        )
        return {
            "k_sim_per_layer": sims_k,
            "v_sim_per_layer": sims_v,
            "k_sim_avg":       round(avg_k, 4),
            "v_sim_avg":       round(avg_v, 4),
            "kv_sim_avg":      round(avg, 4),
            "checked_layers":  layer_indices,
            "vision_region":   [vision_start, vision_end],
        }
    except Exception as e:
        logger.warning(f"  [kv_sim] 계산 실패: {e}")
        return {"error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 Δt 실험
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_one_delta_t(
        model:           Any,
        input_ids_t0:    torch.Tensor,
        tok_data_t0:     dict,
        input_ids_t1:    torch.Tensor,
        tok_data_t1:     dict,
        regions:         dict,
        eos_id:          int,
        traj_offset:     int,
        traj_vocab_size: int,
        delta_t_ms:      int,
) -> dict:
    """
    단일 Δt에 대해 실험 C를 NUM_WARMUP + NUM_MEASURE 회 실행.

    흐름:
      1. t0 full prefill → KV_t0 (t0의 맥락 전체)
      2. t1 full prefill → KV_t1_full (비교 기준선)
      3. KV 유사도 측정 (vision 구간)
      4. KV_t0[:vision_end] 슬라이스 → suffix forward (t1 ego+suffix만)
      5. Decode: t1_full_decode vs reuse_decode 비교
    """
    vision_end  = regions["vision_end"]
    suffix_len  = regions["suffix_len"]
    prefill_len = int(input_ids_t0.shape[1])

    suffix_ids_t1 = input_ids_t1[:, vision_end:]
    actual_suffix_len = int(suffix_ids_t1.shape[1])

    print(f"\n  ── Δt = {delta_t_ms}ms ──────────────────────────────────────────")
    print(f"  vision_end={vision_end}, suffix_len={actual_suffix_len}, "
          f"prefill_len={prefill_len}")
    print(f"  suffix 비율: {actual_suffix_len}/{prefill_len} = "
          f"{actual_suffix_len/prefill_len*100:.1f}%")
    print(f"  이론 절약: ~{(1 - actual_suffix_len/prefill_len)*100:.1f}% = "
          f"~{(1 - actual_suffix_len/prefill_len)*BASELINE_PREFILL_MS:.0f}ms")
    print()

    runs = []

    for trial_idx in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial_idx < NUM_WARMUP
        tag = (f"WARMUP {trial_idx+1}" if is_warmup
               else f"MEASURE {trial_idx - NUM_WARMUP + 1}")

        torch.cuda.empty_cache()
        print(f"  [{tag}]", end=" ", flush=True)

        # ── Step 1: t0 full prefill ───────────────────────────────────────
        try:
            kv_t0, _, pf_ms_t0, _ = full_prefill(
                model, input_ids_t0, tok_data_t0,
                label=f"Δt{delta_t_ms}/{tag}/t0_full",
            )
        except Exception as e:
            print(f"t0_prefill FAIL: {e}")
            traceback.print_exc()
            continue

        # ── Step 2: t1 full prefill (비교 기준) ──────────────────────────
        try:
            kv_t1_full, logits_t1_full, pf_ms_t1_full, _ = full_prefill(
                model, input_ids_t1, tok_data_t1,
                label=f"Δt{delta_t_ms}/{tag}/t1_full",
            )
        except Exception as e:
            print(f"t1_full_prefill FAIL: {e}")
            traceback.print_exc()
            continue

        # ── Step 3: KV 유사도 측정 ────────────────────────────────────────
        kv_sim = compute_kv_similarity(
            kv_t0, kv_t1_full,
            vision_start=regions["vision_start"],
            vision_end=vision_end,
        )

        # ── Step 4: t0 KV 슬라이스 + t1 suffix forward ───────────────────
        # KV_t0[:vision_end]: text_prefix + vision 구간만 (t0의 시각 맥락)
        kv_t0_sliced = slice_cache_to(kv_t0, vision_end)
        log_cache_info(kv_t0_sliced, label=f"Δt{delta_t_ms}/{tag}/KV_t0_sliced")

        try:
            kv_reuse, logits_reuse, pf_ms_suffix = suffix_prefill(
                model,
                suffix_ids=suffix_ids_t1,
                past_kv=kv_t0_sliced,
                start_pos=vision_end,
                label=f"Δt{delta_t_ms}/{tag}/suffix",
            )
        except Exception as e:
            print(f"suffix_prefill FAIL: {e}")
            traceback.print_exc()
            if not is_warmup:
                runs.append({
                    "trial": tag,
                    "error": f"suffix_prefill: {type(e).__name__}: {e}",
                    "pf_t1_full_ms":  round(pf_ms_t1_full, 1),
                    "kv_sim": kv_sim,
                })
            continue

        # ── Step 5a: t1 full baseline decode ─────────────────────────────
        dec_full = decode_loop(
            model, logits_t1_full, kv_t1_full, prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"Δt{delta_t_ms}/{tag}/t1_full_decode",
        )

        # ── Step 5b: reuse decode (t0 vision KV + t1 suffix KV) ──────────
        # reuse_prefill_len: vision_end + actual_suffix_len
        # (KV Cache에는 이 만큼의 토큰이 있음)
        reuse_prefill_len = vision_end + actual_suffix_len
        dec_reuse = decode_loop(
            model, logits_reuse, kv_reuse, reuse_prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"Δt{delta_t_ms}/{tag}/reuse_decode",
        )

        # ── 집계 ─────────────────────────────────────────────────────────
        saving_ms  = pf_ms_t1_full - pf_ms_suffix
        saving_pct = saving_ms / pf_ms_t1_full * 100 if pf_ms_t1_full > 0 else 0.0
        steps_diff = (
            abs(dec_reuse["steps"] - dec_full["steps"])
            if dec_reuse and dec_full else None
        )
        success = (
            dec_reuse is not None
            and dec_reuse["eos_ok"]
            and pf_ms_suffix < 200
            and saving_pct > 80
            and (steps_diff is None or steps_diff <= 5)
        )

        print(
            f"t1_full={pf_ms_t1_full:.0f}ms  "
            f"suffix={pf_ms_suffix:.0f}ms  "
            f"절약={saving_pct:.1f}%  "
            f"kv_sim={kv_sim.get('kv_sim_avg', '?')}  "
            f"t1_steps={dec_full['steps'] if dec_full else '?'}  "
            f"reuse_steps={dec_reuse['steps'] if dec_reuse else '?'}  "
            f"eos={'✅' if dec_reuse and dec_reuse['eos_ok'] else '❌'}  "
            f"{'✅ OK' if success else '⚠️'}"
        )

        if not is_warmup:
            runs.append({
                "trial":              tag,
                "pf_t0_ms":           round(pf_ms_t0, 1),
                "pf_t1_full_ms":      round(pf_ms_t1_full, 1),
                "pf_suffix_ms":       round(pf_ms_suffix, 1),
                "saving_ms":          round(saving_ms, 1),
                "saving_pct":         round(saving_pct, 2),
                "actual_suffix_len":  actual_suffix_len,
                "kv_sim":             kv_sim,
                "decode_full":        dec_full,
                "decode_reuse":       dec_reuse,
                "steps_diff":         steps_diff,
                "success":            success,
            })

    # ── Δt 결과 집계 ─────────────────────────────────────────────────────
    valid = [r for r in runs if "error" not in r]
    summary = {"delta_t_ms": delta_t_ms, "n_valid": len(valid), "runs": runs}

    if valid:
        avg_full    = sum(r["pf_t1_full_ms"] for r in valid) / len(valid)
        avg_suffix  = sum(r["pf_suffix_ms"]  for r in valid) / len(valid)
        avg_saving  = avg_full - avg_suffix
        speedup     = avg_full / avg_suffix if avg_suffix > 0 else float("inf")
        avg_sim     = sum(r["kv_sim"].get("kv_sim_avg", 0) for r in valid) / len(valid)
        eos_ok_rate = sum(1 for r in valid
                          if r.get("decode_reuse") and r["decode_reuse"]["eos_ok"]) / len(valid)
        success_rate = sum(1 for r in valid if r["success"]) / len(valid)

        summary["avg"] = {
            "pf_full_ms":    round(avg_full, 1),
            "pf_suffix_ms":  round(avg_suffix, 1),
            "saving_ms":     round(avg_saving, 1),
            "saving_pct":    round(avg_saving / avg_full * 100, 2),
            "speedup":       round(speedup, 2),
            "kv_sim_avg":    round(avg_sim, 4),
            "eos_ok_rate":   round(eos_ok_rate, 2),
            "success_rate":  round(success_rate, 2),
        }

        print(f"\n  [Δt={delta_t_ms}ms 평균]  "
              f"full={avg_full:.0f}ms  "
              f"suffix={avg_suffix:.0f}ms  "
              f"절약={avg_saving/avg_full*100:.1f}%  "
              f"speedup={speedup:.1f}×  "
              f"kv_sim={avg_sim:.4f}  "
              f"EOS={eos_ok_rate*100:.0f}%  "
              f"success={success_rate*100:.0f}%")
    else:
        summary["avg"] = {"error": "모든 trial 실패"}
        print(f"\n  [Δt={delta_t_ms}ms] ❌ 모든 trial 실패")

    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 입력 준비 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_inputs(model, processor, data) -> tuple[torch.Tensor, dict]:
    """
    데이터셋에서 input_ids와 tok_data(pixel_values 등) 준비.
    fuse_traj_tokens로 ego 토큰을 input_ids에 삽입.
    """
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
    inputs = helper.to_device(inputs, DEVICE)
    input_ids_raw = inputs.pop("input_ids")

    ego_data = helper.to_device(
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        DEVICE,
    )
    input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
    return input_ids, inputs   # inputs = tok_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="KV Temporal Reuse Experiment C: 인접 프레임 vision KV 재사용"
    )
    parser.add_argument(
        "--delta-t",
        nargs="+",
        type=int,
        default=[100, 300, 500, 1000],
        help="테스트할 Δt 값 목록 (ms). 기본: 100 300 500 1000",
    )
    args = parser.parse_args()
    delta_t_list_ms = args.delta_t

    W = 72
    print("=" * W)
    print("  KV Temporal Reuse — Experiment C")
    print(f"  Δt 목록: {delta_t_list_ms} ms")
    print("=" * W)
    print(f"\n  베이스라인 (DynamicCache, sdpa, BF16):")
    print(f"    VE        : {BASELINE_VE_MS:.0f}ms")
    print(f"    LM Prefill: {BASELINE_PREFILL_MS:.0f}ms  ← 이것을 줄이는 것이 목표")
    print(f"    Decode    : {BASELINE_DECODE_MS:.0f}ms")
    print(f"    Flow      : {BASELINE_FLOW_MS:.0f}ms")
    print(f"    합계      : {BASELINE_VE_MS+BASELINE_PREFILL_MS+BASELINE_DECODE_MS+BASELINE_FLOW_MS:.0f}ms")
    print()

    # ── 모델 로드 ─────────────────────────────────────────────────────────
    logger.info("모델 로드 중 (sdpa 기본값, BF16)...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
        # attn_implementation 미지정 → sdpa 기본값 (FlashAttention 유지)
        # ★ eager 사용 금지: seq_len=3086에서 prefill 3,753ms (2.6× 느려짐)
    ).to(DEVICE).eval()

    cfg = model.vlm.config
    attn_impl = getattr(cfg, "_attn_implementation", "unknown")
    logger.info(f"  → attn_implementation = {attn_impl}")

    processor = helper.get_processor(model.tokenizer)

    eos_id = model.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    logger.info(
        f"  eos_id={eos_id}, traj_offset={traj_offset}, "
        f"traj_vocab_size={traj_vocab_size}"
    )

    # ── t0 데이터 로드 + vision 경계 탐지 ────────────────────────────────
    logger.info(f"t0 데이터 로드 (T0={T0_US/1e6:.1f}s)...")
    data_t0 = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    input_ids_t0, tok_data_t0 = prepare_inputs(model, processor, data_t0)
    logger.info(f"  t0 input_ids: {input_ids_t0.shape}  ({input_ids_t0.shape[1]} tokens)")

    regions = detect_vision_regions(model, input_ids_t0)
    logger.info(
        f"  vision 경계: [{regions['vision_start']}, {regions['vision_end']}), "
        f"suffix_len={regions['suffix_len']}, method={regions['method']}"
    )

    # ── 각 Δt에 대해 실험 실행 ────────────────────────────────────────────
    all_results = {
        "attn_implementation": attn_impl,
        "clip_id":             CLIP_ID,
        "t0_us":               T0_US,
        "delta_t_list_ms":     delta_t_list_ms,
        "regions":             {k: v for k, v in regions.items() if k != "method"},
        "regions_method":      regions["method"],
        "baseline":            {
            "ve_ms":       BASELINE_VE_MS,
            "prefill_ms":  BASELINE_PREFILL_MS,
            "decode_ms":   BASELINE_DECODE_MS,
            "flow_ms":     BASELINE_FLOW_MS,
        },
        "experiments": {},
    }

    for dt_ms in delta_t_list_ms:
        dt_us = dt_ms * 1_000
        t1_us = T0_US + dt_us
        logger.info(f"\n{'─'*60}")
        logger.info(f"Δt={dt_ms}ms: t1={t1_us/1e6:.3f}s 데이터 로드...")

        try:
            data_t1 = load_physical_aiavdataset(CLIP_ID, t0_us=t1_us)
            input_ids_t1, tok_data_t1 = prepare_inputs(model, processor, data_t1)
            logger.info(f"  t1 input_ids: {input_ids_t1.shape}  ({input_ids_t1.shape[1]} tokens)")
        except Exception as e:
            logger.error(f"  Δt={dt_ms}ms 데이터 로드 실패: {e} → 스킵")
            all_results["experiments"][f"dt_{dt_ms}ms"] = {"error": str(e)}
            continue

        result = run_one_delta_t(
            model=model,
            input_ids_t0=input_ids_t0,
            tok_data_t0=tok_data_t0,
            input_ids_t1=input_ids_t1,
            tok_data_t1=tok_data_t1,
            regions=regions,
            eos_id=eos_id,
            traj_offset=traj_offset,
            traj_vocab_size=traj_vocab_size,
            delta_t_ms=dt_ms,
        )
        all_results["experiments"][f"dt_{dt_ms}ms"] = result

        # Δt별 중간 저장 (중간에 크래시나도 데이터 보존)
        out_partial = OUT / f"results_dt{dt_ms}ms.json"
        out_partial.write_text(json.dumps(result, indent=2, default=str))
        logger.info(f"  중간 저장: {out_partial}")

    # ── 전체 결과 저장 ────────────────────────────────────────────────────
    out_all = OUT / "results.json"
    out_all.write_text(json.dumps(all_results, indent=2, default=str))

    # ── 최종 요약 테이블 ──────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  ★ Experiment C 종합 결과")
    print(f"{'═'*W}")
    print(f"  {'Δt':>8}  {'KV_sim':>8}  {'full_ms':>8}  {'suffix_ms':>9}  "
          f"{'절약%':>6}  {'speed':>6}  {'EOS%':>5}  {'성공%':>5}")
    print(f"  {'─'*70}")

    for dt_ms in delta_t_list_ms:
        key = f"dt_{dt_ms}ms"
        r = all_results["experiments"].get(key, {})
        avg = r.get("avg", {})
        if "error" in avg:
            print(f"  {dt_ms:>7}ms  {'ERROR':>8}")
            continue
        if not avg:
            print(f"  {dt_ms:>7}ms  {'NO DATA':>8}")
            continue

        # 성공/실패 표시
        ok_sym = "✅" if avg.get("success_rate", 0) >= 0.5 else "⚠️ "

        print(
            f"  {dt_ms:>7}ms  "
            f"{avg.get('kv_sim_avg', 0):>8.4f}  "
            f"{avg.get('pf_full_ms', 0):>8.0f}  "
            f"{avg.get('pf_suffix_ms', 0):>9.0f}  "
            f"{avg.get('saving_pct', 0):>6.1f}%  "
            f"{avg.get('speedup', 0):>5.1f}×  "
            f"{avg.get('eos_ok_rate', 0)*100:>4.0f}%  "
            f"{avg.get('success_rate', 0)*100:>4.0f}%  "
            f"{ok_sym}"
        )

    print(f"{'─'*W}")
    print(f"\n  성공 기준: suffix_prefill < 200ms, 절약 > 80%, EOS ✅, steps diff ≤ 5")
    print(f"\n  해석 가이드:")
    print(f"    kv_sim > 0.99: 시각 장면 거의 동일 → KV 재사용 안전")
    print(f"    kv_sim > 0.95: 소폭 변화 → 재사용 적합")
    print(f"    kv_sim < 0.90: 큰 변화 → full prefill 필요")
    print(f"\n  전체 결과: {out_all}")
    print(f"{'═'*W}")


if __name__ == "__main__":
    main()
