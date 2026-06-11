"""
260604_bandwidth_contention_async_ve_exp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
iGPU (Jetson AGX Thor) 공유 대역폭 경합 측정 + Async VE Pipeline 실험

[실험 배경]
Thor의 CPU와 GPU는 231 GB/s DRAM을 공유한다.
dGPU(RTX PRO 6000)는 GPU 전용 VRAM 1,008 GB/s가 분리되어 있다.

이 구조 차이로 인해 iGPU에서 두 작업을 동시에 실행하면:
  - 두 작업 모두 대역폭 병목이면  → 경합 → 각각 느려짐 (net gain 없음)
  - 한 쪽은 compute-bound, 나머지는 memory-bound → 경합 없음 → 겹치기 효과 full

[핵심 가설]
  Decode: memory-bound (매 step 22GB 가중치 DRAM 로드)
  VE:     compute-bound (2,160 visual token × attention 연산이 지배)
  → 서로 다른 자원 병목 → 경합이 낮을 것 (가설)

[수학적 모델]
  순차 실행당 latency:
    T_seq = T_VE + T_pf + T_dec + T_flow

  Async VE Pipeline (VE_{k+1}을 Decode_k와 겹치기):
    첫 번째 추론: T_VE + T_pf + T_dec_contended
    이후 추론 (n≥2): T_pf + max(T_dec_contended, T_VE_contended)
    단, T_dec_contended = T_dec × c_dec  (경합 인수)
        T_VE_contended = T_VE × c_ve

  N회 추론 총 시간:
    T_seq_total(N) = N × (T_VE + T_pf + T_dec)
    T_async_total(N) = T_VE + N×T_pf + N×T_dec_contended + max(0, T_VE_contended - T_dec_contended)×(N-1)
    단, T_dec_contended > T_VE_contended 이면 VE가 Decode 안에 완전히 숨겨짐

  c_dec, c_ve: 이 실험의 Phase 2에서 실측

[Phase 구성]
  Phase 0: 모델 로드 + 데이터 준비 + JIT Warmup
  Phase 1: 개별 단계 기준선 측정
    1-A: VE 단독 (N_MEASURE 회)
    1-B: Decode step 단독 (N_MEASURE 회)
    1-C: Full Prefill 단독 (N_PREFILL 회)
  Phase 2: 경합 테스트 (핵심 — iGPU 고유)
    2-A: Decode steps while VE runs on side stream
         → contention_ratio_decode = mean(t_step_during_ve) / t_step_alone
    2-B: VE while N decode steps run on side stream
         → contention_ratio_ve = t_ve_with_decode / t_ve_alone
  Phase 3: N-step 연속 추론 파이프라인
    3-A: Sequential (VE→Pf→Dec 순차 반복)
    3-B: Theoretical async estimate (Phase 2 결과 적용)
  Phase 4: 결과 요약 + 결론

[Δt = 100ms 고정 규칙 ★★★]

[실행 방법]
  scp scripts/inference/260604_bandwidth_contention_async_ve_exp.py \\
      ice401@100.95.177.101:~/alpamayo1.5/scripts/inference/
  ssh ice401@100.95.177.101
  cd ~/alpamayo1.5
  source a1_5_venv/bin/activate
  python3 scripts/inference/260604_bandwidth_contention_async_ve_exp.py
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

CLIP_ID    = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US      = 5_100_000
DELTA_T_MS = 100        # ★ 절대 고정
DEVICE     = "cuda"

# Phase 1 설정
N_WARMUP   = 5   # JIT 안정화 (FULL path warmup)
N_MEASURE  = 10  # 단독 측정 반복 횟수 (VE, Decode step, Prefill)

# Phase 2 설정
N_CONTENTION_RUNS   = 5   # 경합 테스트 반복 횟수 (노이즈 제거)
N_DECODE_DURING_VE  = 15  # 경합 테스트 중 실행할 decode step 수
                           # VE ≈ 728ms, decode step ≈ 79ms → 728/79 ≈ 9.2 steps 동안 VE 실행
                           # 15 steps → 앞 ~9 steps 은 VE와 겹침, 나머지 ~6은 VE 완료 후

# Phase 3 설정
N_PIPELINE_STEPS = 4  # 연속 추론 횟수 (sequential vs async 비교)

TEMPERATURE = 0.6
TOP_P       = 0.98
MAX_DECODE  = 30   # 경합 테스트용: 빠른 측정을 위해 30 step으로 제한 (실제 17~19 step)

OUT = Path("profiling_results/260604_bandwidth_contention_async_ve_exp")
OUT.mkdir(parents=True, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUDA 타이머 (CUDA Event 기반 — GPU-side 정밀 측정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CudaTimer:
    """단일 구간 GPU-side 타이머."""
    def __init__(self) -> None:
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._s.record()

    def stop_ms(self) -> float:
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)


def make_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    """(start_event, end_event) 쌍 생성."""
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KV Cache 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def copy_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    """DynamicCache를 완전히 독립된 복사본으로 만든다 (clone, contiguous)."""
    new_cache = DynamicCache()
    new_cache._seen_tokens = getattr(cache, "_seen_tokens", 0)
    kc = getattr(cache, "key_cache", [])
    vc = getattr(cache, "value_cache", [])
    for k, v in zip(kc, vc):
        if isinstance(k, torch.Tensor):
            new_cache.key_cache.append(k.clone().contiguous())
            new_cache.value_cache.append(v.clone().contiguous())
    return new_cache


def get_kv_seq_len(cache: DynamicCache) -> int:
    kc = getattr(cache, "key_cache", [])
    if kc and isinstance(kc[0], torch.Tensor):
        return kc[0].shape[2]
    return getattr(cache, "_seen_tokens", 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통계 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def safe_mean(lst: list) -> float | None:
    lst = [x for x in lst if x is not None and math.isfinite(x)]
    return round(sum(lst) / len(lst), 2) if lst else None

def safe_stdev(lst: list) -> float | None:
    lst = [x for x in lst if x is not None and math.isfinite(x)]
    if len(lst) < 2:
        return None
    mu = sum(lst) / len(lst)
    return round(math.sqrt(sum((x - mu) ** 2 for x in lst) / (len(lst) - 1)), 2)

def safe_median(lst: list) -> float | None:
    lst = sorted(x for x in lst if x is not None and math.isfinite(x))
    n = len(lst)
    if n == 0:
        return None
    mid = n // 2
    return round((lst[mid - 1] + lst[mid]) / 2 if n % 2 == 0 else lst[mid], 2)

def fmt(val, unit="ms") -> str:
    return f"{val:.2f}{unit}" if val is not None else "N/A"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 샘플링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def top_p_sample(logits: torch.Tensor, temperature: float = TEMPERATURE, top_p: float = TOP_P) -> torch.Tensor:
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
# 입력 준비
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_inputs(
    model: Any, processor: Any, data: dict
) -> tuple[torch.Tensor, dict]:
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
    return input_ids, inputs  # inputs contains pixel_values, image_grid_thw, etc.


def detect_vision_end(model: Any, input_ids: torch.Tensor) -> int:
    """input_ids에서 vision 영역 끝 위치 반환."""
    ids = input_ids[0].tolist()
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
        return positions[-1] + 1
    # Fallback
    vs_id = getattr(model.vlm.config, "vision_start_token_id", None)
    ve_id = getattr(model.vlm.config, "vision_end_token_id", None)
    if vs_id is not None and vs_id in ids and ve_id is not None:
        return [i for i, t in enumerate(ids) if t == ve_id][-1] + 1
    logger.warning("vision_end 자동 탐지 실패, fallback=3011")
    return 3011


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 핵심 forward 함수들
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ve(
    model: Any,
    tok_data: dict,
    label: str = "",
) -> tuple[torch.Tensor, float]:
    """
    Vision Encoder(VE) 단독 실행.

    Qwen3VL의 vision tower를 직접 호출한다:
      model.vlm.visual(pixel_values, grid_thw) → image_embeds

    반환:
      image_embeds: (total_image_tokens, hidden_size) — dtype=bfloat16
      elapsed_ms: GPU-side 측정 시간 (ms)

    [왜 직접 호출하는가?]
    model.vlm(pixel_values=pv, ...) 는 VE+LM 전체를 실행한다.
    여기서 VE만 분리해서 호출하면:
      1. VE 단독 latency 측정 가능
      2. 다른 stream에서 VE를 실행하는 경합 테스트 가능
    """
    pixel_values    = tok_data["pixel_values"]
    image_grid_thw  = tok_data["image_grid_thw"]

    # VE의 dtype 확인 (보통 bfloat16)
    try:
        ve_dtype = model.vlm.visual.get_dtype()
    except AttributeError:
        ve_dtype = next(model.vlm.visual.parameters()).dtype

    pv_typed = pixel_values.to(dtype=ve_dtype)

    timer = CudaTimer()
    torch.cuda.synchronize()
    timer.start()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _raw = model.vlm.visual(pv_typed, grid_thw=image_grid_thw)

    # ── 반환 타입 정규화 ──────────────────────────────────────────────────
    # Qwen3VL visual forward 반환 타입이 transformers 버전마다 다르다:
    #   - 구버전: torch.Tensor  (total_tokens, hidden_size)
    #   - 일부버전: tuple       (tensor, ...)  → [0]이 hidden_states
    #   - 신버전: ModelOutput   .last_hidden_state 또는 [0]
    if isinstance(_raw, torch.Tensor):
        image_embeds = _raw
    elif isinstance(_raw, tuple):
        image_embeds = _raw[0]          # 첫 번째 원소 = hidden_states
    elif hasattr(_raw, "last_hidden_state"):
        image_embeds = _raw.last_hidden_state
    else:
        image_embeds = _raw             # fallback: 그대로 사용

    ms = timer.stop_ms()
    if label:
        shape_str = tuple(image_embeds.shape) if isinstance(image_embeds, torch.Tensor) else "?"
        logger.info(f"  [{label}] VE: {ms:.1f}ms  output_shape={shape_str}")
    return image_embeds, ms


def run_full_prefill(
    model: Any,
    input_ids: torch.Tensor,
    tok_data: dict,
    label: str = "",
) -> tuple[DynamicCache, torch.Tensor, float]:
    """
    VE + LM Prefill 전체 (model.vlm 한 번 호출).
    반환: (kv_cache, last_logits, elapsed_ms)
    """
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
        logger.info(f"  [{label}] Full Prefill: {ms:.1f}ms  "
                    f"seq={input_ids.shape[1]} tokens")
    return out.past_key_values, out.logits[:, -1, :].float(), ms


def run_one_decode_step(
    model: Any,
    cur_token: torch.Tensor,           # [1, 1]
    past_kv: DynamicCache,
    prefill_len: int,
    step_offset: int,
    traj_offset: int,
    traj_vocab_size: int,
    label: str = "",
) -> tuple[torch.Tensor, DynamicCache, torch.Tensor, float]:
    """
    Decode 한 step 실행 (GPU-side CUDA event 타이머).

    반환: (next_token [1,1], updated_kv, logits, elapsed_ms)
    """
    cache_pos = torch.tensor(
        [prefill_len + step_offset], device=DEVICE, dtype=torch.long
    )

    ev_s, ev_e = make_event_pair()
    ev_s.record()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=cur_token,
            pixel_values=None,
            past_key_values=past_kv,
            cache_position=cache_pos,
            use_cache=True,
        )

    ev_e.record()
    torch.cuda.synchronize()
    ms = ev_s.elapsed_time(ev_e)

    logits = out.logits[:, -1, :].float()
    logits[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample(logits).unsqueeze(1)  # [1, 1]

    return next_tok, out.past_key_values, logits, ms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: 모델 로드 + 데이터 준비
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase0_setup() -> dict:
    """모델 로드, 데이터 준비, 기본 정보 수집."""
    print("\n" + "=" * 72)
    print("  Phase 0: 모델 로드 + 데이터 준비")
    print("=" * 72)

    logger.info("모델 로딩 중... (~3-4분)")
    model = (
        Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(DEVICE)
        .eval()
    )
    logger.info("  모델 로드 완료")

    eos_id          = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, traj_vocab_size={traj_vocab_size}")

    processor = helper.get_processor(model.tokenizer)

    # N_PIPELINE_STEPS 개 timestep 데이터 로드
    raw_data_list = []
    input_ids_list = []
    tok_data_list = []
    timestamps_us = [T0_US + i * DELTA_T_MS * 1000 for i in range(N_PIPELINE_STEPS)]

    for i, t_us in enumerate(timestamps_us):
        logger.info(f"  데이터 로드 t={t_us/1e6:.1f}s ...")
        raw = load_physical_aiavdataset(CLIP_ID, t0_us=t_us)
        ids, tok = prepare_inputs(model, processor, raw)
        raw_data_list.append(raw)
        input_ids_list.append(ids)
        tok_data_list.append(tok)
        logger.info(f"    t{i}: input_ids={ids.shape}, "
                    f"pixel_values={tok['pixel_values'].shape}, "
                    f"image_grid_thw={tok['image_grid_thw'].shape}")

    vision_end   = detect_vision_end(model, input_ids_list[0])
    prefill_len  = input_ids_list[0].shape[1]
    logger.info(f"  vision_end={vision_end}, prefill_len={prefill_len}")

    # VE dtype 확인
    try:
        ve_dtype = model.vlm.visual.get_dtype()
    except AttributeError:
        ve_dtype = next(model.vlm.visual.parameters()).dtype
    logger.info(f"  VE dtype = {ve_dtype}")

    # GPU 메모리 정보
    mem_alloc_gb = torch.cuda.memory_allocated(DEVICE) / 1e9
    mem_resv_gb  = torch.cuda.memory_reserved(DEVICE) / 1e9
    logger.info(f"  GPU 메모리: allocated={mem_alloc_gb:.1f}GB, reserved={mem_resv_gb:.1f}GB")

    return {
        "model":           model,
        "processor":       processor,
        "input_ids_list":  input_ids_list,
        "tok_data_list":   tok_data_list,
        "eos_id":          eos_id,
        "traj_offset":     traj_offset,
        "traj_vocab_size": traj_vocab_size,
        "vision_end":      vision_end,
        "prefill_len":     prefill_len,
        "ve_dtype":        ve_dtype,
        "timestamps_us":   timestamps_us,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0 Warmup: JIT 안정화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase0_warmup(ctx: dict) -> None:
    """
    JIT 안정화 (N_WARMUP 회).
    VE + Full Prefill + 5 Decode steps 실행.
    """
    print("\n" + "=" * 72)
    print(f"  Phase 0-W: JIT Warmup ({N_WARMUP}회)")
    print("=" * 72)

    model   = ctx["model"]
    ids_t0  = ctx["input_ids_list"][0]
    tok_t0  = ctx["tok_data_list"][0]
    eos_id  = ctx["eos_id"]
    traj_o  = ctx["traj_offset"]
    traj_v  = ctx["traj_vocab_size"]
    pf_len  = ctx["prefill_len"]

    for i in range(N_WARMUP):
        torch.cuda.empty_cache()
        # VE
        _, _ = run_ve(model, tok_t0, label=f"warmup{i}/VE")
        # Full prefill
        kv, logits, _ = run_full_prefill(model, ids_t0, tok_t0, label=f"warmup{i}/Prefill")
        # 5 decode steps
        logits_masked = logits.clone()
        logits_masked[:, traj_o : traj_o + traj_v] = float("-inf")
        cur = top_p_sample(logits_masked).unsqueeze(1)
        for step in range(5):
            cur, kv, _, _ = run_one_decode_step(
                model, cur, kv, pf_len, step, traj_o, traj_v,
                label="" if step > 0 else f"warmup{i}/Decode_step0",
            )
        logger.info(f"  Warmup {i+1}/{N_WARMUP} 완료")

    logger.info("  JIT Warmup 완료 — decode path JIT-compiled")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: 개별 단계 기준선 측정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase1_baselines(ctx: dict) -> dict:
    """
    각 단계를 독립적으로 N_MEASURE 회 측정.

    1-A: VE alone
    1-B: Decode step alone (steady-state, step 5~14 평균)
    1-C: Full Prefill alone
    """
    print("\n" + "=" * 72)
    print(f"  Phase 1: 개별 단계 기준선 측정 ({N_MEASURE}회씩)")
    print("=" * 72)

    model   = ctx["model"]
    ids_t0  = ctx["input_ids_list"][0]
    tok_t0  = ctx["tok_data_list"][0]
    traj_o  = ctx["traj_offset"]
    traj_v  = ctx["traj_vocab_size"]
    pf_len  = ctx["prefill_len"]
    eos_id  = ctx["eos_id"]

    # ── 1-A: VE 단독 ──────────────────────────────────────────────────────
    print("\n  [1-A] VE 단독 측정")
    ve_times: list[float] = []
    for i in range(N_MEASURE):
        torch.cuda.empty_cache()
        _, ms = run_ve(model, tok_t0, label=f"1-A run{i}")
        ve_times.append(ms)
        print(f"    run {i+1:2d}/{N_MEASURE}: {ms:.1f}ms")

    ve_result = {
        "times_ms":  ve_times,
        "mean_ms":   safe_mean(ve_times),
        "stdev_ms":  safe_stdev(ve_times),
        "median_ms": safe_median(ve_times),
    }
    print(f"  → VE mean={fmt(ve_result['mean_ms'])}  "
          f"stdev={fmt(ve_result['stdev_ms'])}  "
          f"median={fmt(ve_result['median_ms'])}")

    # ── 1-B: Decode step 단독 ──────────────────────────────────────────────
    # 단일 step의 steady-state 측정:
    #   full_prefill → 5 steps warmup → N_MEASURE steps 측정
    # 각 측정 전 KV를 prefill 후 상태로 reset (독립성 보장)
    print("\n  [1-B] Decode step 단독 측정 (steady-state)")
    torch.cuda.empty_cache()
    kv_base, logits_base, pf_ms = run_full_prefill(
        model, ids_t0, tok_t0, label="1-B/base_prefill"
    )
    print(f"    기준 prefill: {pf_ms:.1f}ms")

    # 5 warmup steps (JIT 안정화)
    logits_masked = logits_base.clone()
    logits_masked[:, traj_o : traj_o + traj_v] = float("-inf")
    cur = top_p_sample(logits_masked).unsqueeze(1)
    kv_warm = copy_dynamic_cache(kv_base)
    for s in range(5):
        cur, kv_warm, _, _ = run_one_decode_step(
            model, cur, kv_warm, pf_len, s, traj_o, traj_v
        )

    # N_MEASURE 회 각각: KV reset 후 측정
    decode_step_times: list[float] = []
    for i in range(N_MEASURE):
        kv_fresh = copy_dynamic_cache(kv_warm)
        cur_fresh = cur.clone()
        _, _, _, ms = run_one_decode_step(
            model, cur_fresh, kv_fresh, pf_len, 5, traj_o, traj_v,
            label=f"1-B run{i}",
        )
        decode_step_times.append(ms)
        print(f"    run {i+1:2d}/{N_MEASURE}: {ms:.1f}ms/step")

    decode_step_result = {
        "times_ms":  decode_step_times,
        "mean_ms":   safe_mean(decode_step_times),
        "stdev_ms":  safe_stdev(decode_step_times),
        "median_ms": safe_median(decode_step_times),
    }
    print(f"  → Decode step mean={fmt(decode_step_result['mean_ms'])}  "
          f"stdev={fmt(decode_step_result['stdev_ms'])}  "
          f"median={fmt(decode_step_result['median_ms'])}")
    print(f"  [이론 하한: 22GB ÷ 231GB/s = 95.2ms, 현재 {fmt(decode_step_result['mean_ms'])} "
          f"= bandwidth utilization {22/(decode_step_result['mean_ms']/1000)/231*100:.0f}%]")

    # ── 1-C: Full Prefill 단독 ────────────────────────────────────────────
    print("\n  [1-C] Full Prefill 단독 측정")
    prefill_times: list[float] = []
    for i in range(max(N_MEASURE // 2, 3)):  # Prefill은 느리므로 횟수 줄임
        torch.cuda.empty_cache()
        _, _, ms = run_full_prefill(
            model, ids_t0, tok_t0, label=f"1-C run{i}"
        )
        prefill_times.append(ms)
        print(f"    run {i+1}: {ms:.1f}ms")

    prefill_result = {
        "times_ms":  prefill_times,
        "mean_ms":   safe_mean(prefill_times),
        "stdev_ms":  safe_stdev(prefill_times),
        "median_ms": safe_median(prefill_times),
    }
    print(f"  → Prefill mean={fmt(prefill_result['mean_ms'])}  "
          f"stdev={fmt(prefill_result['stdev_ms'])}")

    return {
        "ve":           ve_result,
        "decode_step":  decode_step_result,
        "prefill":      prefill_result,
        "kv_base":      kv_base,       # Phase 2에서 재사용
        "kv_warm":      kv_warm,       # Phase 2 decode 기준 KV
        "cur_warm":     cur,           # Phase 2 decode 기준 token
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2-A: Decode steps while VE runs on side stream
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase2a_decode_vs_ve(ctx: dict, baseline: dict) -> dict:
    """
    [핵심 실험] VE를 side stream에서 실행하면서 Decode step time 변화 측정.

    목적:
      - VE(compute-bound)와 Decode(memory-bound)가 실제로 경합하는지 확인
      - contention_ratio_decode = mean(step_ms_during_ve) / step_ms_alone

    측정 방법:
      1. side stream에 VE를 enqueue (GPU에서 비동기 실행 시작)
      2. default stream에서 N decode steps을 연속 실행
      3. 각 step의 CUDA event timestamp를 기록
      4. VE의 start/end timestamp와 비교하여 overlap 여부 판정
      5. "during VE" steps과 "after VE" steps의 latency를 분리 비교

    [수학적 근거]
    CUDA event는 GPU hardware timer 기반이므로, 서로 다른 stream의
    event 간에도 시간 비교가 가능하다:
      overlap = step_start_time < ve_end_time AND step_end_time > ve_start_time
    """
    print("\n" + "=" * 72)
    print(f"  Phase 2-A: Decode vs VE 경합 측정 ({N_CONTENTION_RUNS}회)")
    print(f"  VE ≈ {baseline['ve']['mean_ms']:.0f}ms, "
          f"Decode step ≈ {baseline['decode_step']['mean_ms']:.0f}ms, "
          f"→ 약 {baseline['ve']['mean_ms']/baseline['decode_step']['mean_ms']:.1f} steps 동안 VE 실행")
    print("=" * 72)

    model      = ctx["model"]
    tok_t0     = ctx["tok_data_list"][0]
    traj_o     = ctx["traj_offset"]
    traj_v     = ctx["traj_vocab_size"]
    pf_len     = ctx["prefill_len"]
    kv_warm    = baseline["kv_warm"]
    cur_warm   = baseline["cur_warm"]

    stream_ve = torch.cuda.Stream()

    all_run_results = []

    for run_idx in range(N_CONTENTION_RUNS):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # GPU idle 확인 후 시작
        kv = copy_dynamic_cache(kv_warm)
        cur = cur_warm.clone()

        # ── VE를 side stream에 enqueue ────────────────────────────────────
        ve_start_ev = torch.cuda.Event(enable_timing=True)
        ve_end_ev   = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream_ve):
            ve_start_ev.record()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    ve_dtype = model.vlm.visual.get_dtype()
                except AttributeError:
                    ve_dtype = next(model.vlm.visual.parameters()).dtype
                pv_typed = tok_t0["pixel_values"].to(dtype=ve_dtype)
                _ = model.vlm.visual(pv_typed, grid_thw=tok_t0["image_grid_thw"])
            ve_end_ev.record()

        # ── default stream에서 N decode steps 실행 ────────────────────────
        # VE enqueue 직후 바로 decode 시작 → GPU가 두 stream을 동시에 실행
        step_start_evs = []
        step_end_evs   = []

        for step in range(N_DECODE_DURING_VE):
            s_ev, e_ev = make_event_pair()
            # default stream에서 record (with torch.cuda.stream 없음 = default stream)
            s_ev.record()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.vlm(
                    input_ids=cur,
                    pixel_values=None,
                    past_key_values=kv,
                    cache_position=torch.tensor(
                        [pf_len + 5 + step], device=DEVICE, dtype=torch.long
                    ),
                    use_cache=True,
                )
            e_ev.record()

            kv = out.past_key_values
            logits = out.logits[:, -1, :].float()
            logits[:, traj_o : traj_o + traj_v] = float("-inf")
            cur = top_p_sample(logits).unsqueeze(1)

            step_start_evs.append(s_ev)
            step_end_evs.append(e_ev)

        # 모든 stream 완료 대기 (default stream이 stream_ve도 기다리도록 barrier)
        # stream_ve의 작업이 완료된 후 default stream이 계속되도록
        torch.cuda.current_stream().wait_stream(stream_ve)
        torch.cuda.synchronize()

        # ── 타이밍 수집 ──────────────────────────────────────────────────
        ve_total_ms = ve_start_ev.elapsed_time(ve_end_ev)

        step_data = []
        for step_i in range(N_DECODE_DURING_VE):
            step_ms = step_start_evs[step_i].elapsed_time(step_end_evs[step_i])
            # VE 기준 상대 시간 계산 (cross-stream elapsed_time 사용)
            step_start_rel = ve_start_ev.elapsed_time(step_start_evs[step_i])  # VE 시작 후 ms
            step_end_rel   = ve_start_ev.elapsed_time(step_end_evs[step_i])
            # overlap 판정:
            #   step_start_rel < ve_total_ms AND step_end_rel > 0
            overlaps_with_ve = (step_start_rel < ve_total_ms) and (step_end_rel > 0)
            fully_after_ve   = step_start_rel >= ve_total_ms
            step_data.append({
                "step":            step_i,
                "step_ms":         round(step_ms, 2),
                "step_start_rel":  round(step_start_rel, 2),
                "step_end_rel":    round(step_end_rel, 2),
                "overlaps_ve":     overlaps_with_ve,
                "fully_after_ve":  fully_after_ve,
            })

        during_ms = [s["step_ms"] for s in step_data if s["overlaps_ve"]]
        after_ms  = [s["step_ms"] for s in step_data if s["fully_after_ve"]]

        run_result = {
            "run":          run_idx,
            "ve_total_ms":  round(ve_total_ms, 2),
            "steps":        step_data,
            "n_during_ve":  len(during_ms),
            "n_after_ve":   len(after_ms),
            "mean_during":  safe_mean(during_ms),
            "mean_after":   safe_mean(after_ms),
        }
        all_run_results.append(run_result)

        print(f"  Run {run_idx+1}: VE={ve_total_ms:.1f}ms  "
              f"during_VE(n={len(during_ms)}): {safe_mean(during_ms):.1f}ms/step  "
              f"after_VE(n={len(after_ms)}): {safe_mean(after_ms):.1f}ms/step")

        for sd in step_data:
            tag = "OVERLAP" if sd["overlaps_ve"] else ("AFTER " if sd["fully_after_ve"] else "PARTIAL")
            logger.debug(f"    step {sd['step']:2d}: {sd['step_ms']:.1f}ms  "
                         f"[{tag}]  start_rel={sd['step_start_rel']:.1f}ms")

    # 집계
    t_alone = baseline["decode_step"]["mean_ms"]
    all_during = [s["step_ms"] for r in all_run_results for s in r["steps"] if s["overlaps_ve"]]
    all_after  = [s["step_ms"] for r in all_run_results for s in r["steps"] if s["fully_after_ve"]]

    mean_during = safe_mean(all_during)
    mean_after  = safe_mean(all_after)
    contention_ratio = round(mean_during / t_alone, 3) if (mean_during and t_alone) else None

    print(f"\n  ─── Phase 2-A 요약 ────────────────────────────────────────")
    print(f"  Decode alone:          {fmt(t_alone)}/step")
    print(f"  During VE  (n={len(all_during)}): {fmt(mean_during)}/step  ← 경합 영향")
    print(f"  After VE   (n={len(all_after)}): {fmt(mean_after)}/step   ← 경합 없음")
    print(f"  contention_ratio = {contention_ratio}  (1.0=경합없음, >1=경합있음)")
    if contention_ratio is not None:
        if contention_ratio < 1.05:
            print(f"  → 경합 없음 ✅ Async VE Pipeline 효과 full")
        elif contention_ratio < 1.2:
            print(f"  → 경합 미미 ✅ Async VE Pipeline 효과 90%+")
        elif contention_ratio < 1.5:
            print(f"  → 경합 중간 ⚠️  Async VE Pipeline 효과 감소")
        else:
            print(f"  → 경합 심함 ❌ VE-during-Flow 대안 설계 필요")

    return {
        "runs":             all_run_results,
        "mean_during_ms":   mean_during,
        "mean_after_ms":    mean_after,
        "contention_ratio": contention_ratio,
        "t_alone_ms":       t_alone,
        "n_overlap_total":  len(all_during),
        "n_after_total":    len(all_after),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2-B: VE while decode runs on side stream
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase2b_ve_vs_decode(ctx: dict, baseline: dict) -> dict:
    """
    Decode steps를 side stream에서 실행하면서 VE time 변화 측정.

    목적:
      - VE가 Decode의 SM/대역폭 사용으로 인해 느려지는지 확인
      - contention_ratio_ve = t_ve_with_decode / t_ve_alone

    참고:
      - 2-A에서 decode가 느려졌다면 → GPU SMs이 VE에 의해 점유됨
      - 2-B에서 VE가 느려졌다면 → VE의 attention 연산이 decode와 SM 충돌
    """
    print("\n" + "=" * 72)
    print(f"  Phase 2-B: VE vs Decode 경합 측정 (반대 방향, {N_CONTENTION_RUNS}회)")
    print("=" * 72)

    model    = ctx["model"]
    tok_t0   = ctx["tok_data_list"][0]
    traj_o   = ctx["traj_offset"]
    traj_v   = ctx["traj_vocab_size"]
    pf_len   = ctx["prefill_len"]
    kv_warm  = baseline["kv_warm"]
    cur_warm = baseline["cur_warm"]

    stream_dec = torch.cuda.Stream()
    ve_times_with_decode: list[float] = []

    for run_idx in range(N_CONTENTION_RUNS):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        kv = copy_dynamic_cache(kv_warm)
        cur = cur_warm.clone()

        # ── decode steps를 side stream에 enqueue ─────────────────────────
        decode_start_ev = torch.cuda.Event(enable_timing=True)
        decode_end_ev   = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream_dec):
            decode_start_ev.record()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for step in range(N_DECODE_DURING_VE):
                    out = model.vlm(
                        input_ids=cur,
                        pixel_values=None,
                        past_key_values=kv,
                        cache_position=torch.tensor(
                            [pf_len + 5 + step], device=DEVICE, dtype=torch.long
                        ),
                        use_cache=True,
                    )
                    kv = out.past_key_values
                    logits = out.logits[:, -1, :].float()
                    logits[:, traj_o : traj_o + traj_v] = float("-inf")
                    cur = top_p_sample(logits).unsqueeze(1)
            decode_end_ev.record()

        # ── VE를 default stream에서 실행 (Decode와 동시 실행 의도) ──────────
        ve_start_ev = torch.cuda.Event(enable_timing=True)
        ve_end_ev   = torch.cuda.Event(enable_timing=True)
        ve_start_ev.record()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                ve_dtype = model.vlm.visual.get_dtype()
            except AttributeError:
                ve_dtype = next(model.vlm.visual.parameters()).dtype
            pv_typed = tok_t0["pixel_values"].to(dtype=ve_dtype)
            _ = model.vlm.visual(pv_typed, grid_thw=tok_t0["image_grid_thw"])
        ve_end_ev.record()

        torch.cuda.current_stream().wait_stream(stream_dec)
        torch.cuda.synchronize()

        ve_ms = ve_start_ev.elapsed_time(ve_end_ev)
        ve_times_with_decode.append(ve_ms)
        print(f"  Run {run_idx+1}: VE_with_decode={ve_ms:.1f}ms")

    t_ve_alone = baseline["ve"]["mean_ms"]
    mean_ve_with_dec = safe_mean(ve_times_with_decode)
    contention_ratio_ve = round(mean_ve_with_dec / t_ve_alone, 3) if (mean_ve_with_dec and t_ve_alone) else None

    print(f"\n  ─── Phase 2-B 요약 ────────────────────────────────────────")
    print(f"  VE alone:           {fmt(t_ve_alone)}")
    print(f"  VE with Decode:     {fmt(mean_ve_with_dec)}")
    print(f"  contention_ratio_ve = {contention_ratio_ve}  (1.0=경합없음)")

    return {
        "times_ms":         ve_times_with_decode,
        "mean_ms":          mean_ve_with_dec,
        "t_alone_ms":       t_ve_alone,
        "contention_ratio": contention_ratio_ve,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: N-step 연속 추론 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_decode_n_steps(
    model: Any,
    kv: DynamicCache,
    first_token: torch.Tensor,
    pf_len: int,
    traj_o: int,
    traj_v: int,
    eos_id: int,
    n_steps: int,
    label: str = "",
) -> tuple[float, int]:
    """
    최대 n_steps 동안 decode loop 실행.
    반환: (total_ms, actual_steps)
    """
    timer = CudaTimer()
    torch.cuda.synchronize()
    timer.start()

    cur = first_token.clone()
    for step in range(n_steps):
        cache_pos = torch.tensor([pf_len + step], device=DEVICE, dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(
                input_ids=cur,
                pixel_values=None,
                past_key_values=kv,
                cache_position=cache_pos,
                use_cache=True,
            )
        logits = out.logits[:, -1, :].float()
        logits[:, traj_o : traj_o + traj_v] = float("-inf")
        next_tok = top_p_sample(logits)
        cur = next_tok.unsqueeze(1)
        kv = out.past_key_values
        if next_tok.item() == eos_id:
            ms = timer.stop_ms()
            return ms, step + 1

    ms = timer.stop_ms()
    return ms, n_steps


def phase3_pipeline_analysis(ctx: dict, phase1: dict, phase2a: dict, phase2b: dict) -> dict:
    """
    Phase 3: N-step 연속 추론 파이프라인 분석.

    3-A: Sequential 실제 측정 (기준선)
    3-B: 이론적 Async VE Pipeline 추정
         (Phase 2 contention 결과를 수학 모델에 대입)
    """
    print("\n" + "=" * 72)
    print(f"  Phase 3: {N_PIPELINE_STEPS}-step 연속 추론 파이프라인 분석")
    print("=" * 72)

    model   = ctx["model"]
    traj_o  = ctx["traj_offset"]
    traj_v  = ctx["traj_vocab_size"]
    eos_id  = ctx["eos_id"]

    # ── 3-A: Sequential 실제 측정 ─────────────────────────────────────────
    print(f"\n  [3-A] Sequential Pipeline ({N_PIPELINE_STEPS}회 연속 추론 실측)")
    seq_inference_times = []

    for inf_idx in range(N_PIPELINE_STEPS):
        torch.cuda.empty_cache()
        ids = ctx["input_ids_list"][inf_idx]
        tok = ctx["tok_data_list"][inf_idx]
        pf_len = ids.shape[1]

        # VE + Prefill
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        t0.record()

        # VE + Prefill (one call)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_pf = model.vlm(
                input_ids=ids,
                attention_mask=tok.get("attention_mask"),
                pixel_values=tok.get("pixel_values"),
                image_grid_thw=tok.get("image_grid_thw"),
                use_cache=True,
            )
        t1.record()

        kv = out_pf.past_key_values
        logits = out_pf.logits[:, -1, :].float()
        logits[:, traj_o : traj_o + traj_v] = float("-inf")
        first_tok = top_p_sample(logits).unsqueeze(1)

        # Decode
        dec_ms, dec_steps = run_decode_n_steps(
            model, kv, first_tok, pf_len, traj_o, traj_v, eos_id,
            n_steps=MAX_DECODE, label=f"3-A inf{inf_idx}",
        )
        t2.record()
        torch.cuda.synchronize()

        pf_ms    = t0.elapsed_time(t1)
        total_ms = pf_ms + dec_ms

        seq_inference_times.append({
            "inf_idx":  inf_idx,
            "pf_ms":    round(pf_ms, 1),
            "dec_ms":   round(dec_ms, 1),
            "dec_steps": dec_steps,
            "total_ms": round(total_ms, 1),
        })
        print(f"    inf {inf_idx}: pf={pf_ms:.0f}ms  dec={dec_ms:.0f}ms ({dec_steps}steps)  "
              f"total={total_ms:.0f}ms")

    seq_total = sum(r["total_ms"] for r in seq_inference_times)
    seq_pf_mean  = safe_mean([r["pf_ms"]  for r in seq_inference_times])
    seq_dec_mean = safe_mean([r["dec_ms"] for r in seq_inference_times])
    print(f"\n  Sequential total ({N_PIPELINE_STEPS} inferences): {seq_total:.0f}ms")
    print(f"  Per-inference: pf={seq_pf_mean:.0f}ms  dec={seq_dec_mean:.0f}ms")

    # ── 3-B: 이론적 Async VE Pipeline 추정 ─────────────────────────────────
    print(f"\n  [3-B] Async VE Pipeline 이론 추정 (Phase 2 contention 적용)")

    T_VE   = phase1["ve"]["mean_ms"] or 728.0
    T_pf   = seq_pf_mean or 2151.0
    T_dec  = seq_dec_mean or 1345.0
    c_dec  = phase2a["contention_ratio"] or 1.0
    c_ve   = phase2b["contention_ratio"] or 1.0

    T_dec_c = T_dec * c_dec
    T_VE_c  = T_VE * c_ve
    N       = N_PIPELINE_STEPS

    # Async 파이프라인 시간 계산
    # VE_k+1이 Decode_k와 겹침. Decode는 경합으로 느려짐.
    # Decode > VE_c 이면 VE가 Decode 안에 완전히 들어옴.
    t_async_first = T_VE + T_pf + T_dec_c                 # 첫 번째 추론 (VE 혼자 실행)
    t_ve_hidden   = max(0.0, T_VE_c - T_dec_c)            # VE가 Decode 이후 남는 시간
    t_async_rest  = T_pf + T_dec_c + t_ve_hidden           # 이후 추론당
    t_async_total = t_async_first + (N - 1) * t_async_rest

    # 경합 없이 이상적인 경우
    t_ideal_first = T_VE + T_pf + T_dec
    t_ideal_rest  = T_pf + T_dec                           # VE 완전히 숨겨짐
    t_ideal_total = t_ideal_first + (N - 1) * t_ideal_rest

    saving_ideal   = seq_total - t_ideal_total
    saving_actual  = seq_total - t_async_total
    speedup_actual = seq_total / t_async_total if t_async_total > 0 else None

    print(f"  파라미터:")
    print(f"    T_VE = {T_VE:.0f}ms,  T_pf = {T_pf:.0f}ms,  T_dec = {T_dec:.0f}ms")
    print(f"    c_dec = {c_dec:.3f} (Decode 경합 인수),  c_ve = {c_ve:.3f} (VE 경합 인수)")
    print(f"    T_dec_contended = {T_dec_c:.0f}ms,  T_VE_contended = {T_VE_c:.0f}ms")
    print(f"    VE hidden in Decode? {T_VE_c:.0f}ms < {T_dec_c:.0f}ms = {'✅ YES' if T_VE_c <= T_dec_c else '❌ NO (VE exceeds Decode)'}")
    print()
    print(f"  Sequential ({N}회):       {seq_total:.0f}ms")
    print(f"  Async ideal (no contention): {t_ideal_total:.0f}ms  (-{saving_ideal:.0f}ms)")
    print(f"  Async actual (contention):   {t_async_total:.0f}ms  (-{saving_actual:.0f}ms)  {speedup_actual:.2f}×")

    return {
        "sequential": {
            "inferences":     seq_inference_times,
            "total_ms":       round(seq_total, 1),
            "mean_pf_ms":     round(seq_pf_mean, 1) if seq_pf_mean else None,
            "mean_dec_ms":    round(seq_dec_mean, 1) if seq_dec_mean else None,
        },
        "async_estimate": {
            "T_VE_ms":              round(T_VE, 1),
            "T_pf_ms":              round(T_pf, 1),
            "T_dec_ms":             round(T_dec, 1),
            "c_dec":                c_dec,
            "c_ve":                 c_ve,
            "T_dec_contended_ms":   round(T_dec_c, 1),
            "T_VE_contended_ms":    round(T_VE_c, 1),
            "ve_hidden_in_decode":  T_VE_c <= T_dec_c,
            "total_ideal_ms":       round(t_ideal_total, 1),
            "total_actual_ms":      round(t_async_total, 1),
            "saving_ideal_ms":      round(saving_ideal, 1),
            "saving_actual_ms":     round(saving_actual, 1),
            "speedup":              round(speedup_actual, 3) if speedup_actual else None,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: 최종 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase4_summary(phase1: dict, phase2a: dict, phase2b: dict, phase3: dict) -> dict:
    print("\n" + "=" * 72)
    print("  Phase 4: 최종 요약 및 결론")
    print("=" * 72)

    t_ve   = phase1["ve"]["mean_ms"]
    t_dec  = phase1["decode_step"]["mean_ms"]
    t_pf   = phase1["prefill"]["mean_ms"]
    c_dec  = phase2a["contention_ratio"]
    c_ve   = phase2b["contention_ratio"]

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  개별 단계 기준선                                                │
  │    VE:           {fmt(t_ve):>10}   (16 images, 4 cameras × 4 frames)   │
  │    Decode step:  {fmt(t_dec):>10}   (steady-state, 1 token/step)       │
  │    Prefill:      {fmt(t_pf):>10}   (3,086 tokens, VE 포함)             │
  │    이론 하한:        ~95.2ms/step (22GB ÷ 231 GB/s DRAM)               │
  ├─────────────────────────────────────────────────────────────────┤
  │  경합 테스트 결과                                                │
  │    Decode during VE: contention_ratio = {c_dec}                │
  │    VE during Decode: contention_ratio = {c_ve}                │
  ├─────────────────────────────────────────────────────────────────┤
  │  Async VE Pipeline 분석                                         │
  │    Sequential total ({N_PIPELINE_STEPS} inferences): {phase3['sequential']['total_ms']:.0f}ms        │
  │    Async ideal:    {phase3['async_estimate']['total_ideal_ms']:.0f}ms  (-{phase3['async_estimate']['saving_ideal_ms']:.0f}ms)  │
  │    Async actual:   {phase3['async_estimate']['total_actual_ms']:.0f}ms  (-{phase3['async_estimate']['saving_actual_ms']:.0f}ms)  {phase3['async_estimate'].get('speedup', 0):.2f}×  │
  └─────────────────────────────────────────────────────────────────┘
    """)

    # 결론
    if c_dec is not None:
        if c_dec < 1.1:
            verdict = "✅ VE와 Decode는 경합 없음 — Async VE Pipeline 즉시 구현 권장"
        elif c_dec < 1.3:
            verdict = "⚠️  경합 미미 — Async VE Pipeline 효과 있음 (약간 손실)"
        else:
            verdict = "❌ 경합 심함 — Flow 단계 중 VE 실행 or DLA 실험으로 전환"
        print(f"  결론: {verdict}")

    summary = {
        "stage_baselines": {
            "ve_ms":           t_ve,
            "decode_step_ms":  t_dec,
            "prefill_ms":      t_pf,
        },
        "contention": {
            "decode_during_ve": {
                "ratio": c_dec,
                "mean_during_ms": phase2a["mean_during_ms"],
                "mean_after_ms":  phase2a["mean_after_ms"],
            },
            "ve_during_decode": {
                "ratio": c_ve,
                "mean_with_decode_ms": phase2b["mean_ms"],
            },
        },
        "pipeline": phase3["async_estimate"],
    }
    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    print("=" * 72)
    print("  Bandwidth Contention + Async VE Pipeline Experiment")
    print(f"  Δt = {DELTA_T_MS}ms (고정 규칙)")
    print(f"  N_MEASURE={N_MEASURE}, N_CONTENTION_RUNS={N_CONTENTION_RUNS}, "
          f"N_PIPELINE_STEPS={N_PIPELINE_STEPS}")
    print(f"  Thor DRAM: 231 GB/s (공유), GPU L2: 32 MB, SM: 20개")
    print("=" * 72)

    results = {}

    try:
        # Phase 0
        ctx = phase0_setup()
        phase0_warmup(ctx)
        results["config"] = {
            "clip_id":       CLIP_ID,
            "delta_t_ms":    DELTA_T_MS,
            "n_measure":     N_MEASURE,
            "n_contention":  N_CONTENTION_RUNS,
            "n_pipeline":    N_PIPELINE_STEPS,
            "vision_end":    ctx["vision_end"],
            "prefill_len":   ctx["prefill_len"],
        }

        # Phase 1
        phase1 = phase1_baselines(ctx)
        results["phase1_baselines"] = {
            "ve":          phase1["ve"],
            "decode_step": phase1["decode_step"],
            "prefill":     phase1["prefill"],
        }

        # Phase 2-A
        phase2a = phase2a_decode_vs_ve(ctx, phase1)
        results["phase2a_contention_decode_vs_ve"] = {
            k: v for k, v in phase2a.items() if k != "runs"
        }
        results["phase2a_runs"] = phase2a["runs"]

        # Phase 2-B
        phase2b = phase2b_ve_vs_decode(ctx, phase1)
        results["phase2b_contention_ve_vs_decode"] = phase2b

        # Phase 3
        phase3 = phase3_pipeline_analysis(ctx, phase1, phase2a, phase2b)
        results["phase3_pipeline"] = phase3

        # Phase 4
        summary = phase4_summary(phase1, phase2a, phase2b, phase3)
        results["summary"] = summary

    except Exception as e:
        logger.error(f"실험 중 오류: {e}")
        traceback.print_exc()
        results["error"] = str(e)

    finally:
        # 결과 저장
        out_path = OUT / "results.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"\n결과 저장: {out_path}")
            print(f"\n  결과 파일: {out_path}")
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")


if __name__ == "__main__":
    main()
