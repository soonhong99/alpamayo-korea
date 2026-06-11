"""
260604_async_ve_pipeline_exp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Async VE Pipeline 구현 및 측정  (v4 — Monkey-Patch 방식)

[목표]
  현재: VE → LM Prefill → Decode  (순차, ~3,213ms)
  목표: Decode(k) 동안 VE(k+1) 미리 실행 → LM-only Prefill(k+1) 진행
        이론 절감: ~640ms/inference (1.25×)

[핵심 설계: Monkey-Patch VE 캐시]
  Qwen3VL (Alpamayo 내부)은 Deepstack 구조를 가진다:
    - VE 실행 시 pooler_output (image_embeds) 외에 deepstack_features도 생성
    - 각 LM 레이어마다:  hidden_states[visual_pos_masks] += deepstack_features[i]
    - pixel_values=None이면 deepstack_features=None → 레이어별 주입 없음 → 출력 오류

  올바른 LM-only Prefill 방법:
    1. VE 실행 → BaseModelOutputWithDeepstackFeatures 캐시 (pooler_output + deepstack_features)
    2. model.vlm.visual.forward를 monkey-patch → 캐시 반환 (VE 계산 없이)
    3. 정상 model.vlm(pixel_values=...) 호출 → VE는 즉시 캐시 반환
       나머지 full forward path (deepstack 포함) 정상 실행
    → Full Prefill과 IDENTICAL 출력 (수치 오차 없음)

[실험 구성]
  Phase 1: Monkey-patch 수치 검증
    - VE 캐시 후 monkey-patch 경로 vs Full Prefill 경로 logit 비교
    - 허용 기준: max_abs_diff < 1e-3 (동일 연산 경로이므로 거의 0)

  Phase 2: LM-only Prefill 시간 측정 (monkey-patch)
    - 기대: ~1,791ms (full_prefill 2,515ms - VE 724ms)

  Phase 3: Async VE Pipeline 실측
    - Decode(k) 중 VE(k+1) 백그라운드 스트림 실행
    - VE 완료 후 monkey-patch LM-only Prefill
    - 순차 대비 실제 절감 측정

  Phase 4: 결과 요약

[Δt = 100ms 고정 규칙 ★★★]
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLIP_ID    = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US      = 5_100_000
DELTA_T_MS = 100        # ★ 절대 고정
DEVICE     = "cuda"

N_WARMUP      = 3    # JIT 워밍업 횟수
N_MEASURE     = 5    # 각 측정 반복 횟수
N_DECODE_STEPS = 15  # Async 테스트용 decode step 수
N_PIPELINE    = 4    # 연속 추론 횟수

TEMPERATURE = 0.6
TOP_P       = 0.98

OUT = Path("profiling_results/260604_async_ve_pipeline_exp")
OUT.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUDA 타이머
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CudaTimer:
    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._s.record()

    def stop_ms(self) -> float:
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KV Cache 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def copy_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    new_cache._seen_tokens = getattr(cache, "_seen_tokens", 0)
    for k, v in zip(getattr(cache, "key_cache", []),
                    getattr(cache, "value_cache", [])):
        new_cache.key_cache.append(k.clone().contiguous())
        new_cache.value_cache.append(v.clone().contiguous())
    return new_cache


def top_p_sample(logits: torch.Tensor, top_p: float = TOP_P) -> torch.Tensor:
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    mask = (cum_probs - sorted_probs) >= top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, idx).squeeze(-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VE 관련 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_ve_dtype(model) -> torch.dtype:
    try:
        return model.vlm.visual.get_dtype()
    except AttributeError:
        return next(model.vlm.visual.parameters()).dtype


def run_ve(model, tok_data: dict, label: str = "") -> tuple[torch.Tensor, float]:
    """VE 단독 실행 (pooler_output만). 반환: (image_embeds tensor, elapsed_ms)
    ★ 시간 측정 전용. LM prefill에는 run_ve_with_cache 사용 필요.
    """
    pv = tok_data["pixel_values"].to(dtype=get_ve_dtype(model))
    grid = tok_data["image_grid_thw"]

    timer = CudaTimer()
    torch.cuda.synchronize()
    timer.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model.vlm.visual(pv, grid_thw=grid)
    ms = timer.stop_ms()

    # 반환 타입 정규화 (버전마다 다름)
    if isinstance(raw, torch.Tensor):
        image_embeds = raw
    elif isinstance(raw, tuple):
        image_embeds = raw[0]
    elif hasattr(raw, "pooler_output") and isinstance(raw.pooler_output, torch.Tensor):
        image_embeds = raw.pooler_output
    elif hasattr(raw, "last_hidden_state"):
        image_embeds = raw.last_hidden_state
    else:
        image_embeds = raw

    if label:
        logger.info(f"  [{label}] VE: {ms:.1f}ms  shape={tuple(image_embeds.shape)}")
    return image_embeds, ms


def run_ve_with_cache(model, tok_data: dict, label: str = "") -> tuple[dict, float]:
    """
    VE 실행 후 monkey-patch용 캐시 구조체 반환.

    [반환 구조]
    ve_cache = {
        'pooler_output': torch.Tensor   ← image_embeds (cloned, 수정 안전)
        'deepstack_features': list      ← 레이어별 시각 특징 (deepstack injection용)
        'last_hidden_state': tensor|None
        'pv_typed': tensor              ← dtype 변환된 pixel_values
        'thw': tensor                   ← image_grid_thw
    }

    [deepstack 중요성]
    Qwen3VL은 각 LM 레이어마다:
        hidden_states[visual_pos_masks] += deepstack_features[i]
    pixel_values=None이면 deepstack_features=None → 레이어 주입 없음 → 출력 WRONG
    → 반드시 이 함수로 캐시 후 monkey-patch 경로를 써야 함
    """
    pv_typed = tok_data["pixel_values"].to(dtype=get_ve_dtype(model))
    thw = tok_data["image_grid_thw"]

    timer = CudaTimer()
    torch.cuda.synchronize()
    timer.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model.vlm.visual(pv_typed, grid_thw=thw, return_dict=True)
    ms = timer.stop_ms()

    # 반환 타입 파싱
    if hasattr(raw, 'pooler_output'):
        pooler_output     = raw.pooler_output
        deepstack_features = getattr(raw, 'deepstack_features', None)
        last_hidden_state  = getattr(raw, 'last_hidden_state', None)
    elif isinstance(raw, (tuple, list)):
        pooler_output     = raw[0]
        deepstack_features = raw[1] if len(raw) > 1 else None
        last_hidden_state  = None
    else:
        pooler_output     = raw
        deepstack_features = None
        last_hidden_state  = None

    # pooler_output 클론: get_image_features 내부에서 .pooler_output을 재할당하므로
    # 원본 tensor를 보존해야 여러 번 재사용 가능
    if isinstance(pooler_output, torch.Tensor):
        pooler_output = pooler_output.clone()

    has_deepstack = (deepstack_features is not None
                     and (isinstance(deepstack_features, (list, tuple))
                          and len(deepstack_features) > 0))

    if label:
        shape = tuple(pooler_output.shape) if isinstance(pooler_output, torch.Tensor) else "?"
        logger.info(f"  [{label}] VE (full): {ms:.1f}ms  pooler={shape}  "
                    f"deepstack={len(deepstack_features) if isinstance(deepstack_features, list) else deepstack_features}")
        if isinstance(deepstack_features, list) and deepstack_features:
            shapes = [tuple(d.shape) for d in deepstack_features[:4]]
            logger.info(f"    deepstack_features[0..3] shapes: {shapes}")

    ve_cache = {
        'pooler_output':     pooler_output,
        'deepstack_features': deepstack_features,
        'last_hidden_state':  last_hidden_state,
        'pv_typed':          pv_typed,
        'thw':               thw,
        'has_deepstack':     has_deepstack,
    }
    return ve_cache, ms


def build_inputs_embeds(
    model,
    input_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """
    text embedding + image embedding을 병합해 inputs_embeds를 구성한다.

    Qwen3VL 내부 forward 로직 재현:
      1. embed_tokens(input_ids) → text_embeds  [B, L, H]
      2. image_mask = (input_ids == image_token_id)  [B, L]
      3. inputs_embeds.masked_scatter(image_mask, image_embeds)

    주의:
      - image_embeds는 [total_img_tokens, H] 형태여야 한다.
      - masked_scatter는 mask가 True인 위치를 순서대로 image_embeds로 채운다.
      - image_token_id는 model.vlm.config.image_token_id 에서 가져온다.
    """
    with torch.no_grad():
        # 1. text embedding
        text_embeds = model.vlm.language_model.embed_tokens(input_ids)  # [B, L, H]

        # 2. image mask
        image_mask = (input_ids == image_token_id)  # [B, L]
        n_img_tokens_expected = image_mask.sum().item()
        n_img_tokens_actual   = image_embeds.shape[0]

        if n_img_tokens_expected != n_img_tokens_actual:
            logger.warning(
                f"  image token count mismatch: "
                f"mask={n_img_tokens_expected}, embeds={n_img_tokens_actual}"
            )

        # 3. merge: image_embeds를 image_mask 위치에 순서대로 삽입
        image_embeds_bf16 = image_embeds.to(
            device=text_embeds.device, dtype=text_embeds.dtype
        )
        expanded_mask = image_mask.unsqueeze(-1).expand_as(text_embeds)
        inputs_embeds = text_embeds.masked_scatter(expanded_mask, image_embeds_bf16)

    return inputs_embeds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prefill 경로들
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_full_prefill(model, input_ids, tok_data, label="") -> tuple[DynamicCache, torch.Tensor, float]:
    """VE 포함 full prefill (기존 방식). 반환: (kv, last_logits, ms)"""
    timer = CudaTimer()
    torch.cuda.synchronize()
    timer.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=input_ids,
            attention_mask=tok_data.get("attention_mask"),
            pixel_values=tok_data.get("pixel_values"),
            image_grid_thw=tok_data.get("image_grid_thw"),
            use_cache=True,
        )
    ms = timer.stop_ms()
    if label:
        logger.info(f"  [{label}] Full Prefill: {ms:.1f}ms")
    return out.past_key_values, out.logits[:, -1, :].float(), ms


def run_lm_prefill_with_embeds(
    model,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    tok_data: dict,
    label: str = "",
) -> tuple[DynamicCache, torch.Tensor, float]:
    """
    pre-computed inputs_embeds를 사용해 LM prefill 실행 (VE 없이).

    [Qwen3VL 소스 분석 기반 설계 v3]

    구조:
      model.vlm      = Qwen3VLForConditionalGeneration  (outer, lm_head 포함)
      model.vlm.model = Qwen3VLModel                    (inner, 실제 forward)

    XOR 체크 (Qwen3VLModel.forward() line 1129):
      if (input_ids is None) ^ (inputs_embeds is not None): raise ValueError
      → input_ids=None, inputs_embeds=tensor → True ^ True = False → 통과 ✅

    get_rope_index 위치:
      model.vlm.get_rope_index        → ❌ AttributeError (outer에 없음)
      model.vlm.model.get_rope_index  → ✅ Qwen3VLModel에 있음

    Qwen3VL 신버전 API:
      get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask)
      mm_token_type_ids: 0=text, 1=image (필수 파라미터)

    rope_deltas 저장:
      compute_3d_position_ids가 past_key_values_length > 0일 때
      model.vlm.model.rope_deltas를 참조 → decode step이 올바른 position offset 사용
      → 우리가 compute_3d_position_ids를 우회하므로 직접 저장 필요

    pixel_values=None: VE 재실행 방지 ★
    image_grid_thw=None: position_ids 이미 계산됐으므로 불필요
    """
    import inspect

    inner = model.vlm.model  # Qwen3VLModel

    # ── 1. mm_token_type_ids 계산 (0=text, 1=image) ───────────────────────
    # Qwen3VL 신버전에서 get_rope_index의 필수 파라미터
    image_token_id: int = getattr(model.vlm.config, 'image_token_id', 151655)
    mm_token_type_ids = (input_ids == image_token_id).to(torch.int32)  # [B, L]

    # ── 2. 3D RoPE position_ids 계산 ──────────────────────────────────────
    # model.vlm.model.get_rope_index 사용 (outer에 없고 inner에 있음)
    get_rope_fn = getattr(inner, 'get_rope_index', None)
    if get_rope_fn is None:
        raise RuntimeError(
            "model.vlm.model.get_rope_index 없음 — "
            "transformers 버전이 Qwen3VL을 지원하지 않을 수 있음. "
            "`pip show transformers` 로 버전 확인."
        )

    with torch.no_grad():
        # API 버전 감지: 신버전은 mm_token_type_ids 필수, 구버전(Qwen2VL-style)은 없음
        sig = inspect.signature(get_rope_fn)
        if 'mm_token_type_ids' in sig.parameters:
            position_ids, rope_deltas = get_rope_fn(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=tok_data.get("image_grid_thw"),
                attention_mask=tok_data.get("attention_mask"),
            )
        else:
            # Qwen2VL-style 구버전 API
            position_ids, rope_deltas = get_rope_fn(
                input_ids,
                image_grid_thw=tok_data.get("image_grid_thw"),
                attention_mask=tok_data.get("attention_mask"),
            )

        # rope_deltas 저장: decode step에서 compute_3d_position_ids가 참조
        if hasattr(inner, 'rope_deltas'):
            inner.rope_deltas = rope_deltas

    # ── 3. LM-only Prefill (inputs_embeds 경로) ───────────────────────────
    timer = CudaTimer()
    torch.cuda.synchronize()
    timer.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=None,              # ★ None — XOR check: True^True=False → OK
            inputs_embeds=inputs_embeds, # 미리 merged된 embeddings (text+image)
            attention_mask=tok_data.get("attention_mask"),
            position_ids=position_ids,   # ★ 3D RoPE positions (pre-computed)
            pixel_values=None,           # VE 재실행 방지 ★
            image_grid_thw=None,         # position_ids 이미 계산됨 → None OK
            use_cache=True,
        )
    ms = timer.stop_ms()
    if label:
        logger.info(f"  [{label}] LM-only Prefill: {ms:.1f}ms")
    return out.past_key_values, out.logits[:, -1, :].float(), ms


def run_lm_prefill_monkey_patched(
    model,
    input_ids: torch.Tensor,
    tok_data: dict,
    ve_cache: dict,
    label: str = "",
) -> tuple[DynamicCache, torch.Tensor, float]:
    """
    Monkey-patch를 이용한 LM-only Prefill (VE 캐시 재사용).

    [원리]
    model.vlm.visual.forward를 일시적으로 교체:
      patched_forward(pv, grid_thw) → (image_embeds, deepstack_image_embeds) 2-tuple

    → model.vlm(input_ids=..., pixel_values=pv, ...) 호출 시:
        get_image_features(pv, thw) 내부:
          image_embeds, deepstack = self.visual(pv, grid_thw)  ← 2-tuple 언패킹
          → 캐시 반환 (~0ms), deepstack injection까지 포함 ✅
    → Full Prefill과 IDENTICAL 수치 결과

    [Thor 버전 확인]
    get_image_features (line 1061):
      image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    → self.visual()은 반드시 iterable한 2-tuple을 반환해야 함
    → SimpleNamespace는 iterable이 아니므로 TypeError 발생
    → 수정: 항상 (pooler_output, deepstack_features) tuple 반환
    """
    pooler_output     = ve_cache['pooler_output']   # cloned tensor (2880, 4096)
    deepstack_features = ve_cache['deepstack_features']  # list of N tensors
    pv_typed          = ve_cache['pv_typed']
    thw               = ve_cache['thw']

    _orig_fwd = model.vlm.visual.forward

    def _patched_forward(hidden_states, grid_thw, return_dict=True, **kwargs):
        # Thor get_image_features:
        #   image_embeds, deepstack = self.visual(pv, grid_thw)  ← 2-tuple 언패킹
        # → tuple 반환 필수 (SimpleNamespace 불가 — iterable 아님)
        return (pooler_output, deepstack_features)

    model.vlm.visual.forward = _patched_forward
    try:
        timer = CudaTimer()
        torch.cuda.synchronize()
        timer.start()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(
                input_ids=input_ids,
                attention_mask=tok_data.get("attention_mask"),
                pixel_values=pv_typed,  # VE 코드 경로 활성화 (패치로 즉시 캐시 반환)
                image_grid_thw=thw,
                use_cache=True,
            )
        ms = timer.stop_ms()
    finally:
        model.vlm.visual.forward = _orig_fwd

    if label:
        logger.info(f"  [{label}] LM-only Prefill (monkey-patch): {ms:.1f}ms")
    return out.past_key_values, out.logits[:, -1, :].float(), ms


def run_decode_steps(
    model,
    kv: DynamicCache,
    first_logits: torch.Tensor,
    prefill_len: int,
    traj_offset: int,
    traj_vocab_size: int,
    eos_id: int,
    n_steps: int,
) -> tuple[float, int]:
    """N steps decode. 반환: (total_ms, actual_steps)"""
    timer = CudaTimer()
    torch.cuda.synchronize()
    timer.start()

    # top_p_sample: [1, vocab] → [1]  (squeeze(-1) keeps batch dim)
    # need [batch=1, seq=1] for input_ids → unsqueeze seq dim only
    cur = top_p_sample(
        first_logits.clone().index_fill_(
            -1, torch.arange(traj_offset, traj_offset + traj_vocab_size, device=DEVICE), float("-inf")
        )
    ).unsqueeze(1)  # [1] → [1, 1]

    steps = 0
    for step in range(n_steps):
        cache_pos = torch.tensor([prefill_len + step], device=DEVICE, dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(
                input_ids=cur,
                pixel_values=None,
                past_key_values=kv,
                cache_position=cache_pos,
                use_cache=True,
            )
        kv = out.past_key_values
        logits = out.logits[:, -1, :].float()
        logits[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample(logits)
        steps += 1
        if next_tok.item() == eos_id:
            break
        cur = next_tok.unsqueeze(1)  # [1] → [1, 1]

    ms = timer.stop_ms()
    return ms, steps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: 모델 로드 + 데이터 준비
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase0_setup():
    logger.info("모델 로딩 중...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(DEVICE).eval()

    eos_id      = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_offset = model.config.traj_token_start_idx
    traj_v_size = model.config.traj_vocab_size

    # image_token_id 확인 (두 가지 경로 시도)
    image_token_id = None
    try:
        image_token_id = model.vlm.config.image_token_id
        logger.info(f"  image_token_id from model.vlm.config: {image_token_id}")
    except AttributeError:
        pass

    if image_token_id is None:
        image_token_id = model.special_token_ids.get("image_pad")
        logger.info(f"  image_token_id from special_token_ids[image_pad]: {image_token_id}")

    if image_token_id is None:
        # 마지막 수단: 토크나이저에서 직접 조회
        image_token_id = model.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        logger.info(f"  image_token_id from tokenizer: {image_token_id}")

    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, image_token_id={image_token_id}")

    # 데이터 로드 (4개 타임스텝)
    processor = helper.get_processor(model.tokenizer)
    tok_list = []
    for i in range(4):
        t_us = T0_US + i * DELTA_T_MS * 1000
        logger.info(f"  데이터 로드 t={i+1} ...")
        data = load_physical_aiavdataset(CLIP_ID, t0_us=t_us)
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        raw = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        input_ids_raw = raw["input_ids"].to(DEVICE)
        ego_data = {
            "ego_history_xyz": data["ego_history_xyz"].to(DEVICE),
            "ego_history_rot": data["ego_history_rot"].to(DEVICE),
        }
        input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
        tok = {
            "input_ids": input_ids,
            "attention_mask": raw.get("attention_mask", torch.ones_like(input_ids)).to(DEVICE),
            "pixel_values": raw["pixel_values"].to(DEVICE),
            "image_grid_thw": raw["image_grid_thw"].to(DEVICE),
        }
        tok_list.append(tok)
        logger.info(f"    input_ids={input_ids.shape}, pixel_values={tok['pixel_values'].shape}")

    ctx = {
        "model": model,
        "tok_list": tok_list,
        "eos_id": eos_id,
        "traj_offset": traj_offset,
        "traj_v_size": traj_v_size,
        "image_token_id": image_token_id,
        "prefill_len": tok_list[0]["input_ids"].shape[1],
    }
    logger.info(f"  GPU: allocated={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return ctx


def phase0_warmup(ctx):
    logger.info("\n  JIT 워밍업...")
    model     = ctx["model"]
    tok_t0    = ctx["tok_list"][0]
    input_ids = tok_t0["input_ids"]

    for i in range(N_WARMUP):
        _, _, _ = run_full_prefill(model, input_ids, tok_t0, label=f"warmup{i}")
    torch.cuda.empty_cache()
    logger.info("  워밍업 완료")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: inputs_embeds 경로 수치 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase1_verify_monkey_patch(ctx) -> dict:
    """
    핵심 검증: Monkey-Patch LM-only Prefill이 Full Prefill과 동일한 logit을 내는가?

    [검증 방법]
    A. Full Prefill (pixel_values 경로)          → logits_full
    B. VE 캐시 (run_ve_with_cache)               → ve_cache (deepstack 포함)
    C. Monkey-patch LM-only Prefill              → logits_monkey
    D. 비교: logits_full vs logits_monkey

    [기대 결과]
    동일한 연산 경로이므로 max_diff ≈ 0 (BF16 정밀도 수준)
    허용 기준: max_diff < 1e-3 (fp32 변환 후 비교)

    [이전 실패 원인 기록]
    - inputs_embeds 주입 방식: deepstack_features 누락 → max_diff=2.3125 (❌)
    - Monkey-patch 방식: full forward path 유지 → diff ≈ 0 (✅ 예상)
    """
    print(f"\n{'='*72}")
    print("  Phase 1: Monkey-Patch 경로 수치 검증")
    print(f"{'='*72}")

    model     = ctx["model"]
    tok_t0    = ctx["tok_list"][0]
    input_ids = tok_t0["input_ids"]

    # ── A. Full Prefill (기준선) ────────────────────────────────────────
    logger.info("  [A] Full Prefill (기준선) 실행...")
    kv_full, logits_full, ms_full = run_full_prefill(
        model, input_ids, tok_t0, label="verify/full"
    )
    logger.info(f"    logits_full shape: {logits_full.shape}")

    # ── B. VE 캐시 획득 ─────────────────────────────────────────────────
    logger.info("  [B] VE 캐시 획득 (deepstack_features 포함)...")
    try:
        ve_cache, ms_ve = run_ve_with_cache(model, tok_t0, label="verify/VE_cache")
    except Exception as e:
        logger.error(f"  VE 캐시 실패: {e}")
        traceback.print_exc()
        return {"status": "FAIL_ve_cache", "detail": str(e)}

    logger.info(f"    deepstack 존재: {ve_cache['has_deepstack']}")

    # ── C. Monkey-Patch LM-only Prefill ────────────────────────────────
    logger.info("  [C] Monkey-Patch LM-only Prefill 실행...")
    try:
        kv_mp, logits_mp, ms_mp = run_lm_prefill_monkey_patched(
            model, input_ids, tok_t0, ve_cache, label="verify/monkey"
        )
    except Exception as e:
        logger.error(f"  Monkey-Patch Prefill 실패: {e}")
        traceback.print_exc()
        return {"status": "FAIL_forward_error", "detail": str(e)}

    # ── D. Logit 비교 ───────────────────────────────────────────────────
    max_diff  = (logits_full - logits_mp).abs().max().item()
    mean_diff = (logits_full - logits_mp).abs().mean().item()
    top1_full = logits_full.argmax(dim=-1).item()
    top1_mp   = logits_mp.argmax(dim=-1).item()
    top1_match = (top1_full == top1_mp)

    logger.info(f"\n  ─── 수치 검증 결과 ───────────────────────────────────")
    logger.info(f"  Full Prefill 시간:           {ms_full:.1f}ms")
    logger.info(f"  VE 캐시 시간:                {ms_ve:.1f}ms")
    logger.info(f"  Monkey-Patch Prefill 시간:   {ms_mp:.1f}ms  (기대: ~{ms_full - ms_ve:.0f}ms)")
    logger.info(f"  logit max_diff:              {max_diff:.8f}  (허용: < 1e-3)")
    logger.info(f"  logit mean_diff:             {mean_diff:.8f}")
    logger.info(f"  top-1 token 일치:            {'✅' if top1_match else '❌'} "
                f"(full={top1_full}, monkey={top1_mp})")

    # Monkey-patch는 동일 연산이므로 매우 엄격한 기준 적용
    PASS = max_diff < 1e-3 and top1_match
    status = "PASS" if PASS else "FAIL_logit_mismatch"
    logger.info(f"\n  → 검증 결과: {'✅ PASS' if PASS else '❌ FAIL'} ({status})")
    if not PASS:
        logger.warning(f"  ★ max_diff={max_diff:.6f} > 1e-3 — deepstack 처리 확인 필요")

    # ve_cache를 ctx에 저장 (Phase 2, 3에서 재사용)
    ctx["ve_cache_t0"] = ve_cache

    return {
        "status": status,
        "ms_full_prefill":   round(ms_full, 2),
        "ms_ve_cache":       round(ms_ve, 2),
        "ms_lm_monkey":      round(ms_mp, 2),
        "lm_only_ratio":     round(ms_mp / ms_full, 3),
        "logit_max_diff":    round(max_diff, 8),
        "logit_mean_diff":   round(mean_diff, 8),
        "top1_match":        top1_match,
        "has_deepstack":     ve_cache['has_deepstack'],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: LM-only Prefill 시간 안정적 측정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase2_lm_only_prefill_timing(ctx, p1_result: dict) -> dict:
    """Monkey-Patch LM-only Prefill 시간 안정적 측정."""
    if "FAIL" in p1_result.get("status", "FAIL"):
        logger.warning("  Phase 1 실패 → Phase 2 건너뜀")
        return {}

    print(f"\n{'='*72}")
    print("  Phase 2: LM-only Prefill 시간 측정 (Monkey-Patch)")
    print(f"{'='*72}")

    import statistics

    model     = ctx["model"]
    tok_t0    = ctx["tok_list"][0]
    input_ids = tok_t0["input_ids"]

    times_full = []
    times_ve   = []
    times_lm   = []

    for i in range(N_MEASURE):
        # full prefill
        _, _, ms_full = run_full_prefill(model, input_ids, tok_t0)
        times_full.append(ms_full)

        # VE 캐시 (매번 새로 측정)
        ve_cache, ms_ve = run_ve_with_cache(model, tok_t0)
        times_ve.append(ms_ve)

        # LM-only prefill (monkey-patch)
        _, _, ms_lm = run_lm_prefill_monkey_patched(model, input_ids, tok_t0, ve_cache)
        times_lm.append(ms_lm)

        logger.info(f"  run {i+1}/{N_MEASURE}: "
                    f"full={ms_full:.0f}ms  ve={ms_ve:.0f}ms  lm_only={ms_lm:.0f}ms  "
                    f"(diff_from_theory={abs(ms_full - ms_ve - ms_lm):.0f}ms)")

    mean_full = statistics.mean(times_full)
    mean_ve   = statistics.mean(times_ve)
    mean_lm   = statistics.mean(times_lm)

    logger.info(f"\n  ─── Phase 2 요약 ───────────────────────────────────────")
    logger.info(f"  Full Prefill (mean):    {mean_full:.1f}ms")
    logger.info(f"  VE alone (mean):        {mean_ve:.1f}ms")
    logger.info(f"  LM-only Prefill (mean): {mean_lm:.1f}ms")
    logger.info(f"  VE + LM:                {mean_ve + mean_lm:.1f}ms  "
                f"(Full과 차이: {abs(mean_full - mean_ve - mean_lm):.1f}ms)")
    logger.info(f"  이론 절감 (1회):         {mean_ve:.1f}ms/inference")

    return {
        "ms_full_prefill": round(mean_full, 2),
        "ms_ve_alone":     round(mean_ve, 2),
        "ms_lm_only":      round(mean_lm, 2),
        "sum_ve_lm":       round(mean_ve + mean_lm, 2),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Async VE Pipeline 실측
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase3_async_pipeline(ctx, p1_result: dict) -> dict:
    """
    Async VE Pipeline 구현 및 시간 측정.

    [파이프라인 설계]

    Inference k의 Decode가 실행되는 동안, k+1의 VE를 백그라운드 stream에서 실행.
    Decode(k) 완료 후, cached image_embeds로 LM-only Prefill(k+1) 실행.

    순차 파이프라인:
      [VE_k][Pf_k][Dec_k] → [VE_{k+1}][Pf_{k+1}][Dec_{k+1}] → ...

    Async 파이프라인:
      [VE_k][Pf_k][Dec_k]     → [Pf_lm_{k+1}][Dec_{k+1}] → ...
                  └─[VE_{k+1}]─┘
                   백그라운드 실행

    [중요 설계 결정]
    - VE는 stream_ve (백그라운드)에서 실행
    - Decode는 default stream에서 실행
    - Phase 2-A에서 확인: 두 작업이 주로 직렬화됨 (GPU SM 포화)
    - 그럼에도 VE가 Decode 완료 직후 또는 일부 겹쳐서 실행됨
    - 핵심 이점: 다음 추론의 VE가 현재 추론의 LM Prefill Critical Path에서 제거됨

    [측정 방법]
    - Sequential N회: VE → Full Prefill → Decode 순차 반복
    - Async N회: Decode 중 VE 백그라운드 실행 → LM-only Prefill
    - 두 방식의 총 시간 비교
    """
    if "FAIL" in p1_result.get("status", "FAIL"):
        logger.warning("  Phase 1 실패 → Phase 3 건너뜀")
        return {}

    print(f"\n{'='*72}")
    print("  Phase 3: Async VE Pipeline 실측")
    print(f"{'='*72}")

    model          = ctx["model"]
    tok_list       = ctx["tok_list"]
    eos_id         = ctx["eos_id"]
    traj_o         = ctx["traj_offset"]
    traj_v         = ctx["traj_v_size"]
    image_token_id = ctx["image_token_id"]
    pf_len         = ctx["prefill_len"]
    N              = min(N_PIPELINE, len(tok_list))

    stream_ve = torch.cuda.Stream()

    # ── A. Sequential 기준선 ─────────────────────────────────────────────
    logger.info("\n  [3-A] Sequential Pipeline (기준선)")
    seq_times = []
    torch.cuda.empty_cache()

    for i in range(N):
        tok = tok_list[i]
        input_ids = tok["input_ids"]

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0.record()

        kv, logits, ms_pf = run_full_prefill(model, input_ids, tok)
        ms_dec, n_steps   = run_decode_steps(model, kv, logits, pf_len, traj_o, traj_v, eos_id, N_DECODE_STEPS)

        t1.record()
        torch.cuda.synchronize()
        total_ms = t0.elapsed_time(t1)
        seq_times.append(total_ms)
        logger.info(f"    inf {i}: pf={ms_pf:.0f}ms  dec={ms_dec:.0f}ms ({n_steps}steps)  total={total_ms:.0f}ms")

    total_seq = sum(seq_times)
    avg_seq = total_seq / N
    logger.info(f"\n  Sequential 총계: {total_seq:.0f}ms  평균: {avg_seq:.0f}ms/inference")

    # ── B. Async VE Pipeline ─────────────────────────────────────────────
    logger.info("\n  [3-B] Async VE Pipeline (Monkey-Patch, Phase 2-B 방식)")
    logger.info("  설계: Decode(k, stream_dec) || VE(k+1, DEFAULT stream)")
    logger.info("  ★ 이전 실험 Phase 2-B 결론: VE→default, Decode→background → VE 21% 빠름")
    logger.info("  ★ 핵심: VE를 먼저 디스패치(~20ms) → Decode(.item()있음) 실행 중 VE가 idle SM 활용")
    logger.info("  ★ .item()은 stream_dec만 sync → default stream(VE) 계속 실행 가능")
    async_times   = []
    ve_times_fg   = []   # default stream(foreground)에서 실행된 VE 시간
    pf_times_lm   = []   # LM-only prefill 시간
    torch.cuda.empty_cache()

    stream_dec = torch.cuda.Stream()   # Decode 전용 background stream

    # 첫 번째 추론은 VE 캐시가 없으므로 full prefill
    tok0 = tok_list[0]
    t_start = torch.cuda.Event(enable_timing=True)
    t_end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t_start.record()

    kv_cur, logits_cur, ms_pf0 = run_full_prefill(model, tok0["input_ids"], tok0)
    logger.info(f"    inf 0 (첫 번째, full prefill): pf={ms_pf0:.0f}ms")

    # 이후 추론: VE 먼저 디스패치 후 Decode 실행 (Phase 2-B 방식)
    for i in range(N):
        tok_next       = tok_list[min(i + 1, len(tok_list) - 1)]
        input_ids_next = tok_next["input_ids"]
        pv_next_typed  = tok_next["pixel_values"].to(dtype=get_ve_dtype(model))
        thw_next       = tok_next["image_grid_thw"]

        # ① STEP 1: VE를 DEFAULT stream에 먼저 빠르게 디스패치 (~20ms Python)
        #   GPU가 VE를 즉시 시작. 이후 Decode의 DRAM 대기 중 유휴 SM을 VE가 활용.
        ve_start_ev = torch.cuda.Event(enable_timing=True)
        ve_end_ev   = torch.cuda.Event(enable_timing=True)
        ve_start_ev.record()  # default stream
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            raw_next = model.vlm.visual(pv_next_typed, grid_thw=thw_next, return_dict=True)
        ve_end_ev.record()  # default stream
        # ~20ms Python 경과. GPU default stream: VE 실행 중.

        # ② STEP 2: Decode를 stream_dec에서 .item() 포함 실행 (~1228ms wall)
        #   .item()은 stream_dec만 sync → default stream(VE)은 방해받지 않고 계속 실행!
        #   VE(572ms)가 Decode(1228ms) 안에 완전히 숨음.
        dec_start_ev = torch.cuda.Event(enable_timing=True)
        dec_end_ev   = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream_dec):
            dec_start_ev.record()
            first_logits_masked = logits_cur.clone().index_fill_(
                -1, torch.arange(traj_o, traj_o + traj_v, device=DEVICE), float("-inf")
            )
            cur_dec = top_p_sample(first_logits_masked).unsqueeze(1)  # [1] → [1,1]
            n_steps_dec = 0
            for step in range(N_DECODE_STEPS):
                cache_pos = torch.tensor([pf_len + step], device=DEVICE, dtype=torch.long)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_dec = model.vlm(
                        input_ids=cur_dec,
                        pixel_values=None,
                        past_key_values=kv_cur,
                        cache_position=cache_pos,
                        use_cache=True,
                    )
                kv_cur = out_dec.past_key_values
                logits_step = out_dec.logits[:, -1, :].float()
                logits_step[:, traj_o:traj_o + traj_v] = float("-inf")
                next_tok_dec = top_p_sample(logits_step)
                n_steps_dec += 1
                # ★ .item()은 stream_dec만 sync → default stream(VE)은 계속 실행!
                if next_tok_dec.item() == eos_id:
                    break
                cur_dec = next_tok_dec.unsqueeze(1)  # [1] → [1,1]
            dec_end_ev.record()
        # Python이 with 블록 탈출: ~1228ms wall 경과 (VE는 ~592ms에 완료)
        # GPU: stream_dec decode 완료, default stream VE도 이미 완료

        # ③ STEP 3: 두 stream 모두 완료 대기
        torch.cuda.current_stream().wait_stream(stream_dec)
        torch.cuda.synchronize()
        ve_ms  = ve_start_ev.elapsed_time(ve_end_ev)
        dec_ms = dec_start_ev.elapsed_time(dec_end_ev)
        ve_times_fg.append(ve_ms)

        # ④ STEP 4: VE 캐시 구성 (sync 완료 후 — VE 결과 사용 가능)
        if hasattr(raw_next, 'pooler_output'):
            pooler_next    = raw_next.pooler_output.clone()
            deepstack_next = getattr(raw_next, 'deepstack_features', None)
            lhs_next       = getattr(raw_next, 'last_hidden_state', None)
        elif isinstance(raw_next, (tuple, list)):
            pooler_next    = raw_next[0].clone()
            deepstack_next = raw_next[1] if len(raw_next) > 1 else None
            lhs_next       = None
        else:
            pooler_next    = raw_next.clone()
            deepstack_next = None
            lhs_next       = None

        # ⑤ STEP 5: monkey-patch LM-only Prefill (다음 inference 준비)
        if i + 1 < len(tok_list):
            ve_cache_next = {
                'pooler_output':      pooler_next,
                'deepstack_features': deepstack_next,
                'last_hidden_state':  lhs_next,
                'pv_typed':           pv_next_typed,
                'thw':                thw_next,
                'has_deepstack':      deepstack_next is not None,
            }
            kv_next, logits_next, ms_pf_lm = run_lm_prefill_monkey_patched(
                model, input_ids_next, tok_next, ve_cache_next
            )
            pf_times_lm.append(ms_pf_lm)
            logger.info(f"    inf {i}: dec={dec_ms:.0f}ms({n_steps_dec}steps)  "
                        f"ve_fg={ve_ms:.0f}ms  lm_pf={ms_pf_lm:.0f}ms")
            kv_cur     = kv_next
            logits_cur = logits_next
        else:
            logger.info(f"    inf {i}: dec={dec_ms:.0f}ms({n_steps_dec}steps)  ve_fg={ve_ms:.0f}ms")

    t_end.record()
    torch.cuda.synchronize()
    total_async = t_start.elapsed_time(t_end)

    avg_async = total_async / N
    logger.info(f"\n  Async 총계: {total_async:.0f}ms  평균: {avg_async:.0f}ms/inference")

    # ── 비교 ────────────────────────────────────────────────────────────
    speedup = total_seq / total_async if total_async > 0 else 0
    saving_ms = total_seq - total_async

    logger.info(f"\n  ─── Phase 3 결과 ───────────────────────────────────────")
    logger.info(f"  Sequential ({N}회):  {total_seq:.0f}ms  (avg {avg_seq:.0f}ms)")
    logger.info(f"  Async      ({N}회):  {total_async:.0f}ms  (avg {avg_async:.0f}ms)")
    logger.info(f"  절감:               {saving_ms:.0f}ms  ({speedup:.2f}×)")
    logger.info(f"  VE foreground 평균:  {sum(ve_times_fg)/len(ve_times_fg):.1f}ms  "
                f"(기대: ~574ms, Phase 2-B 대비 {sum(ve_times_fg)/len(ve_times_fg)/724*100:.0f}%)")
    if pf_times_lm:
        logger.info(f"  LM-only Prefill:    {sum(pf_times_lm)/len(pf_times_lm):.1f}ms")

    import statistics
    return {
        "total_seq_ms":    round(total_seq, 1),
        "total_async_ms":  round(total_async, 1),
        "avg_seq_ms":      round(avg_seq, 1),
        "avg_async_ms":    round(avg_async, 1),
        "speedup":         round(speedup, 3),
        "saving_ms":       round(saving_ms, 1),
        "ve_fg_mean_ms":   round(statistics.mean(ve_times_fg), 1) if ve_times_fg else None,
        "lm_pf_mean_ms":   round(statistics.mean(pf_times_lm), 1) if pf_times_lm else None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase4_summary(p1, p2, p3):
    print(f"\n{'='*72}")
    print("  Phase 4: 최종 요약")
    print(f"{'='*72}")

    p1_ok = "PASS" in p1.get("status", "")

    print(f"\n  [Phase 1] Monkey-Patch 경로 검증: {'✅ PASS' if p1_ok else '❌ FAIL'}")
    if p1_ok:
        print(f"    logit max_diff:      {p1.get('logit_max_diff', '?')}")
        print(f"    top-1 일치:          {p1.get('top1_match', '?')}")
        print(f"    deepstack 존재:      {p1.get('has_deepstack', '?')}")
        print(f"    LM-only prefill:     {p1.get('ms_lm_monkey', '?')}ms  "
              f"(full={p1.get('ms_full_prefill', '?')}ms)")

    if p2:
        print(f"\n  [Phase 2] LM-only Prefill 시간")
        print(f"    Full Prefill:        {p2.get('ms_full_prefill', '?')}ms")
        print(f"    VE alone:            {p2.get('ms_ve_alone', '?')}ms")
        print(f"    LM only:             {p2.get('ms_lm_only', '?')}ms")

    if p3:
        speedup = p3.get("speedup", 0)
        print(f"\n  [Phase 3] Async VE Pipeline 성과")
        print(f"    Sequential ({N_PIPELINE}회):  {p3.get('total_seq_ms', '?')}ms")
        print(f"    Async      ({N_PIPELINE}회):  {p3.get('total_async_ms', '?')}ms")
        print(f"    절감:                {p3.get('saving_ms', '?')}ms")
        print(f"    가속비:              {speedup:.2f}×")
        print(f"    VE foreground:       {p3.get('ve_fg_mean_ms', '?')}ms  (기대 ~574ms, Phase 2-B)")
        if speedup >= 1.15:
            print(f"    → ✅ 유의미한 가속 달성 (Phase 2-B 방식 효과 확인)")
        elif speedup >= 1.05:
            print(f"    → ⚠️ 소폭 가속 (이론값 1.25×에 미달)")
        else:
            print(f"    → ❌ 가속 없음 — VE fg 시간 확인 필요")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 72)
    print("  Async VE Pipeline 검증 및 측정 실험")
    print("  Δt = 100ms (고정 규칙)")
    print("=" * 72)

    results = {}

    try:
        # Phase 0
        print(f"\n{'='*72}")
        print("  Phase 0: 모델 로드 + 준비")
        print(f"{'='*72}")
        ctx = phase0_setup()
        phase0_warmup(ctx)

        # Phase 1: 수치 검증 (가장 중요)
        p1 = phase1_verify_monkey_patch(ctx)
        results["phase1"] = p1

        # Phase 2: LM-only Prefill 시간 측정
        p2 = phase2_lm_only_prefill_timing(ctx, p1)
        results["phase2"] = p2

        # Phase 3: Async Pipeline 실측
        p3 = phase3_async_pipeline(ctx, p1)
        results["phase3"] = p3

        # Phase 4: 요약
        phase4_summary(p1, p2, p3)

    except Exception as e:
        logger.error(f"실험 오류: {e}")
        traceback.print_exc()
        results["error"] = str(e)

    # 결과 저장
    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
