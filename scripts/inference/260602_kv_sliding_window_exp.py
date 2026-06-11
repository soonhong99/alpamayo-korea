"""
260602_kv_sliding_window_exp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[목적]
  KV Temporal Reuse Sliding Window 검증
  Experiment C (통째 재사용) 의 개선 버전: 가장 오래된 temporal frame의 KV만
  교체하고 나머지 3개 frame KV는 t0에서 유지한다.

[배경]
  Experiment C (260602):
    t0 vision KV 전체를 t1에 재사용 → 최신 프레임(t+1) 데이터 없음
    [text_pref | frame_4.8s(stale) | 4.9s | 5.0s | 5.1s] KV + t1 suffix

  Sliding Window (이 실험):
    t0 KV에서 가장 오래된 frame의 KV를 t1 최신 frame의 KV로 교체
    [text_pref | frame_5.2s(fresh) | 4.9s | 5.0s | 5.1s] KV + t1 suffix
                      ↑ KV_t1_full에서 추출한 최신 frame

  실제 운용 파이프라인에서는 슬라이딩 윈도우 방식이 필요:
    T=5.1s: window = [4.8, 4.9, 5.0, 5.1]
    T=5.2s: window = [4.9, 5.0, 5.1, 5.2]  ← 오래된 것 1개 제거, 새것 1개 추가

[구현 핵심 — KV 이식 (KV Transplant)]
  t1 full prefill → KV_t1_full (어차피 baseline용으로 계산)
  KV_t1_full에서 newest temporal frame의 KV를 추출:
    k_new[l] = KV_t1_full[l].K[:, :, newest_frame_start:newest_frame_end, :]

  KV_t0에서 oldest temporal frame의 KV를 교체:
    k_sw[l] = cat([
        KV_t0[l].K[:, :, 0:vision_start],          # text_prefix (29 tok)
        k_new[l],                                     # 최신 frame t+1 (교체)
        KV_t0[l].K[:, :, oldest_frame_end:vision_end] # 유지된 3 frame
    ], dim=2)

  이후 t1 suffix(75 tok) forward는 Experiment C와 동일.

[RoPE 처리에 대한 기술적 근거]
  Alpamayo = Qwen2-VL 기반 → mrope (multimodal RoPE) 사용
  Vision token의 K/V는 sequential position이 아닌 2D patch 좌표(row, col)로 RoPE 인코딩.
  → vision KV를 다른 sequential position에 배치해도 RoPE mismatch 없음.
  → "이 가정이 실제로 성립하는가"를 실험 결과로 검증 (성립하면 SW가 잘 동작, 아니면 품질 저하)

[측정 지표]
  시간:
    - sw_assembly_ms: KV 조립 시간 (추가 비용)
    - sw_suffix_ms: suffix forward 시간 (Exp C와 비교)
  품질:
    - eos_ok: EOS 정상 생성 여부
    - steps_diff: |full_steps - sw_steps|
    - replaced_frame_kv_sim: 교체된 구간(frame 0) t0 vs SW KV 유사도
      → 1.0에 가까워야 함 (fresh t+1 데이터)
    - retained_frame_kv_sim: 유지된 구간(frame 1~3) t0 vs SW KV 유사도
      → t0와 동일하므로 1.0
  비교:
    - Exp C vs SW: 시간 차이, 품질 차이 → SW의 장점 정량화

[성공 기준]
  ✅ sw_suffix_ms < 250ms (Exp C의 142ms + 조립 오버헤드)
  ✅ EOS 정상 생성
  ✅ |full_steps - sw_steps| ≤ 5
  ✅ replaced_frame_kv_sim > 0.95 (t+1 최신 frame 이식 효과)

[알려진 Thor 이슈 및 대응 — Exp C에서 확인된 것]
  1. DynamicCache._seen_tokens 미초기화 → 명시 설정
  2. cache_position 필수 → 모든 suffix/sw forward에 명시 전달
  3. torch.autocast BF16 → dtype 불일치 방지
  4. NUM_WARMUP=3 → JIT 안정화
  5. .contiguous() → L2 재사용 효과

[실행]
  source ~/alpamayo1.5/a1_5_venv/bin/activate && cd ~/alpamayo1.5
  python3 scripts/inference/260602_kv_sliding_window_exp.py [--delta-t 100 300 500 1000]

[결과]
  profiling_results/260602_kv_sliding_window/results.json
  profiling_results/260602_kv_sliding_window/results_dt{N}ms.json  (중간 저장)
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
T0_US             = 5_100_000
DEVICE            = "cuda"
MAX_DECODE_STEPS  = 80
EOS_CHECK_EVERY   = 4
TEMPERATURE       = 0.6
TOP_P             = 0.98
NUM_WARMUP        = 3
NUM_MEASURE       = 3

# 4 cameras × 4 temporal frames = 16 images
N_CAMERAS         = 4
N_TEMPORAL_FRAMES = 4

# Qwen2-VL merge_size (vision token 수 = H*W // merge_size²)
MERGE_SIZE        = 2

# 기준선 (AppendOnlyCache-C, 2026-06-01 실측)
BASELINE_APPENDONLY_TOTAL_MS = 3620.0
BASELINE_PREFILL_MS          = 895.0
BASELINE_EXP_C_SUFFIX_MS     = 142.0   # 260602 실측

OUT = Path("profiling_results/260602_kv_sliding_window")
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
# DynamicCache 유틸리티 (260602 Exp C와 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cache_to_kv_pairs(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """DynamicCache에서 레이어별 (K, V) 텐서 쌍 추출. 4종 폴백 지원."""
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
    raise AttributeError(
        f"[cache 구조 불명] type={type(cache)}, "
        f"attrs={[a for a in dir(cache) if 'cache' in a.lower() or 'key' in a.lower()]}"
    )


def _build_cache_from_kv(
        kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        seen_tokens: int = 0,
) -> DynamicCache:
    """
    (K, V) 쌍 리스트로 DynamicCache 구성.
    ★ _seen_tokens 명시 설정 — Thor transformers는 __init__에서 초기화하지 않음.
    """
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


def slice_cache_range(cache, start: int, end: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    cache의 [start:end] 토큰 구간 KV를 텐서 쌍 리스트로 반환.
    contiguous 복사로 메모리 연속성 보장 (L2 재사용 최적화).
    """
    pairs = _cache_to_kv_pairs(cache)
    return [
        (k[:, :, start:end, :].clone().contiguous(),
         v[:, :, start:end, :].clone().contiguous())
        for k, v in pairs
    ]


def log_cache_info(cache, label: str = "") -> None:
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
            f"  [{label}] {len(pairs)}L, seq={seq_len}, "
            f"shape=[{k0.shape[0]},{k0.shape[1]},{seq_len},{k0.shape[3]}], "
            f"mem={mem_mb:.0f}MB, _seen={seen}"
        )
    except Exception as e:
        logger.warning(f"  [{label}] cache info 실패: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vision 경계 탐지 (260602 Exp C와 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_vision_regions(model: Any, input_ids: torch.Tensor) -> dict:
    """input_ids에서 vision 토큰 구간 자동 탐지."""
    ids = input_ids[0].tolist()
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
        return {
            "text_prefix_len": vs,
            "vision_start": vs,
            "vision_end": ve,
            "vision_len": ve - vs,
            "suffix_start": ve,
            "suffix_len": total - ve,
            "total_len": total,
            "n_image_tokens": len(positions),
            "image_pad_id": img_tok_id,
        }

    logger.warning("vision_detect 실패 → fallback (vision_start=29, n_image=2880)")
    vs, ve = 29, 29 + 2880
    return {
        "text_prefix_len": vs, "vision_start": vs, "vision_end": ve,
        "vision_len": 2880, "suffix_start": ve, "suffix_len": total - ve,
        "total_len": total, "n_image_tokens": 2880, "image_pad_id": None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Temporal Frame 경계 탐지 (신규 — 슬라이딩 윈도우 핵심)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_temporal_frame_boundaries(
        tok_data: dict,
        input_ids: torch.Tensor,
        vision_start: int,
        vision_end: int,
        image_pad_id: int = 151655,
        n_cameras: int = N_CAMERAS,
        n_temporal_frames: int = N_TEMPORAL_FRAMES,
) -> dict:
    """
    image_grid_thw를 사용해 temporal frame별 KV slot 경계를 계산한다.

    Qwen2-VL 구조:
      image_grid_thw: [n_images, 3] tensor/list, 각 row = [T, H, W]
      tokens per image = T * H * W // merge_size²  (merge_size=2)
      총 16 images = 4 cameras × 4 temporal frames
      배열 순서 가정: [cam1_t0, cam2_t0, cam3_t0, cam4_t0, cam1_t1, ...]  (temporal-first)

    Frame slot 정의:
      - frame_slot_start[f] = f번째 temporal frame의 첫 image_pad 위치
      - frame_slot_end[f]   = frame_slot_start[f+1]  (f < n_temporal_frames-1)
                            = vision_end              (f == n_temporal_frames-1)
      - 이 정의로 special token이 다음 frame slot에 포함됨 (일관성 유지)

    반환:
      frame_slots: [(start, end), ...] × n_temporal_frames
        frame_slots[0] = oldest temporal frame slot (교체 대상)
        frame_slots[-1] = newest temporal frame slot (KV_t1_full에서 추출 대상)
      per_image_ranges: [(start, end, n_tok), ...] × n_images
      n_tok_per_image: [int, ...] × n_images
    """
    ids = input_ids[0].tolist()

    # ── 1. per-image token count from image_grid_thw ─────────────────────
    igthw = tok_data.get("image_grid_thw")
    n_images = n_cameras * n_temporal_frames  # 16

    if igthw is not None:
        if isinstance(igthw, torch.Tensor):
            igthw_list = igthw.tolist()
        else:
            igthw_list = list(igthw)

        # tokens per image = T * H * W // (MERGE_SIZE ** 2)
        n_tok_per_image = [
            max(1, int(t * h * w) // (MERGE_SIZE ** 2))
            for t, h, w in igthw_list
        ]
        logger.info(
            f"  [frame_detect] image_grid_thw: {len(igthw_list)} images, "
            f"tok/img={set(n_tok_per_image)}, total={sum(n_tok_per_image)}"
        )
    else:
        # fallback: equal split
        all_img_pos_count = sum(1 for t in ids if t == image_pad_id)
        toks_per_img = all_img_pos_count // n_images
        n_tok_per_image = [toks_per_img] * n_images
        logger.warning(
            f"  [frame_detect] image_grid_thw 없음 → equal split: "
            f"{toks_per_img} tok/img × {n_images} images"
        )

    # ── 2. find all image_pad positions ──────────────────────────────────
    all_img_pos = [i for i, tok in enumerate(ids) if tok == image_pad_id]
    total_img_toks = len(all_img_pos)
    expected_img_toks = sum(n_tok_per_image)

    if total_img_toks != expected_img_toks:
        logger.warning(
            f"  [frame_detect] image_pad count mismatch: "
            f"expected={expected_img_toks}, found={total_img_toks}. "
            f"equal split으로 재계산."
        )
        toks_per_img = total_img_toks // n_images
        n_tok_per_image = [toks_per_img] * n_images
        # Remainder를 마지막 이미지에 추가
        remainder = total_img_toks - toks_per_img * n_images
        if remainder > 0:
            n_tok_per_image[-1] += remainder

    # ── 3. per-image token ranges ─────────────────────────────────────────
    pos = 0
    per_image_ranges = []
    for n_tok in n_tok_per_image:
        if pos + n_tok > total_img_toks:
            n_tok = total_img_toks - pos
        start = all_img_pos[pos]
        end   = all_img_pos[pos + n_tok - 1] + 1
        per_image_ranges.append((start, end, n_tok))
        pos += n_tok

    # ── 4. group into temporal frames ────────────────────────────────────
    #    Temporal frame f = images [f*n_cameras : (f+1)*n_cameras]
    frame_first_img_start = []  # 각 temporal frame의 첫 image_pad 위치
    for f in range(n_temporal_frames):
        frame_first_img_start.append(
            per_image_ranges[f * n_cameras][0]
        )

    # ── 5. frame slot boundaries ─────────────────────────────────────────
    #    Slot f = [frame_first_img_start[f] : frame_first_img_start[f+1]]
    #    Slot last = [frame_first_img_start[-1] : vision_end]
    frame_slots = []
    for f in range(n_temporal_frames):
        slot_start = frame_first_img_start[f]
        if f < n_temporal_frames - 1:
            slot_end = frame_first_img_start[f + 1]
        else:
            slot_end = vision_end
        frame_slots.append((slot_start, slot_end))

    # ── 6. 검증 및 로그 ──────────────────────────────────────────────────
    total_slot_toks = sum(e - s for s, e in frame_slots)
    expected_vision_toks = vision_end - vision_start  # 2982

    # Note: vision_start(29) ~ frame_slots[0][0] 사이에 special token이 있을 수 있음
    # → frame_slots[0][0]이 vision_start와 같으면 OK, 다르면 앞에 special token 있음
    prefix_special_toks = frame_slots[0][0] - vision_start

    logger.info(
        f"  [frame_detect] temporal frame slots:"
    )
    for f, (s, e) in enumerate(frame_slots):
        n_img_toks = sum(
            per_image_ranges[f * n_cameras + c][2]
            for c in range(n_cameras)
        )
        logger.info(
            f"    frame[{f}] (t-{n_temporal_frames-1-f}): "
            f"pos[{s}:{e}], slot_len={e-s}, img_toks={n_img_toks}"
        )
    if prefix_special_toks > 0:
        logger.info(
            f"  [frame_detect] vision_start 앞 special tok: {prefix_special_toks}"
        )

    # 슬라이딩 윈도우를 위한 중요 검증:
    oldest_slot_len = frame_slots[0][1] - frame_slots[0][0]
    newest_slot_len = frame_slots[-1][1] - frame_slots[-1][0]
    if oldest_slot_len != newest_slot_len:
        logger.warning(
            f"  [frame_detect] ⚠️ oldest slot({oldest_slot_len}) != "
            f"newest slot({newest_slot_len}). "
            f"KV 이식 시 크기 불일치 발생할 수 있음."
        )
    else:
        logger.info(
            f"  [frame_detect] ✅ oldest=newest slot_len={oldest_slot_len} → KV 이식 크기 일치"
        )

    return {
        "n_temporal_frames": n_temporal_frames,
        "n_cameras": n_cameras,
        "frame_slots": frame_slots,          # [(start, end), ...] × 4
        "per_image_ranges": per_image_ranges,
        "n_tok_per_image": n_tok_per_image,
        "total_img_tokens": total_img_toks,
        "prefix_special_toks": prefix_special_toks,
        "oldest_frame_slot": frame_slots[0],   # 교체 대상
        "newest_frame_slot": frame_slots[-1],  # t1에서 추출할 대상
        "oldest_slot_len": oldest_slot_len,
        "newest_slot_len": newest_slot_len,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 슬라이딩 윈도우 KV 조립 (핵심)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def assemble_sliding_window_kv(
        kv_t0:              Any,
        kv_t1_full:         Any,
        per_image_ranges_t0: list,  # [(start, end, n_tok), ...] × 16 images (from frame_info_t0)
        per_image_ranges_t1: list,  # [(start, end, n_tok), ...] × 16 images (from frame_info_t1)
        n_cameras:          int,    # 4
        n_frames:           int,    # 4 (temporal frames per camera)
        vision_end:         int,    # 3011
) -> tuple[DynamicCache, float]:
    """
    Per-image KV 이식으로 슬라이딩 윈도우 KV 조립.

    ★ 이미지 순서: camera-first
      create_message()가 image_frames.flatten(0,1) 기준으로 출력:
        images 0-3  = cam0의 4개 temporal frame (f0=oldest, f3=newest)
        images 4-7  = cam1의 4개 temporal frame
        images 8-11 = cam2의 4개 temporal frame
        images 12-15= cam3의 4개 temporal frame

    ★ 교체 대상 (oldest temporal frame, t-3):
        각 카메라의 첫 번째 이미지 = {c * n_frames + 0 for c in range(n_cameras)}
        = image indices {0, 4, 8, 12}

    ★ 이식 소스 (newest temporal frame in t1, t+1):
        각 카메라의 마지막 이미지 in t1 = {c * n_frames + (n_frames-1) for c in range(n_cameras)}
        = image indices {3, 7, 11, 15}

    ★ Transplant: (image 0→3), (4→7), (8→11), (12→15)
        각 교체: 180 image tokens (contiguous, non-overlapping)
        총 교체: 4 × 180 = 720 image tokens out of 2982 vision tokens (~24%)
        텍스트/frame 레이블/유지 frame: t0 KV 그대로 보존

    ★ RoPE 근거 (mrope hypothesis):
        Qwen2-VL vision token의 K/V는 2D patch (row, col) 좌표로 RoPE 인코딩.
        Sequential position과 무관 → 다른 sequential position에 이식해도 올바름.
        이 가정의 성립 여부는 실험 결과(eos_ok, steps_diff)로 검증.

    Returns:
      (kv_sw, assembly_ms): 조립된 DynamicCache + 소요 시간
    """
    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()

    pairs_t0 = _cache_to_kv_pairs(kv_t0)
    pairs_t1 = _cache_to_kv_pairs(kv_t1_full)
    n_layers = len(pairs_t0)

    # Transplant pair: (old_img_idx in t0, new_img_idx in t1)
    # old = cam_c's oldest frame (f=0), new = cam_c's newest frame in t1 (f=n_frames-1)
    transplant_pairs = [
        (c * n_frames,                       # oldest: cam c, frame 0
         c * n_frames + (n_frames - 1))      # newest in t1: cam c, frame n_frames-1
        for c in range(n_cameras)
    ]
    # = [(0,3), (4,7), (8,11), (12,15)]

    sw_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for l in range(n_layers):
        k_t0_l = pairs_t0[l][0]   # [B, H, seq_t0, D]
        v_t0_l = pairs_t0[l][1]
        k_t1_l = pairs_t1[l][0]   # [B, H, seq_t1, D]
        v_t1_l = pairs_t1[l][1]

        # t0의 vision 구간을 그대로 복사 (in-place 교체)
        k_sw = k_t0_l[:, :, :vision_end, :].clone()
        v_sw = v_t0_l[:, :, :vision_end, :].clone()

        # Per-image transplant: scattered, non-contiguous
        for old_idx, new_idx in transplant_pairs:
            old_start, old_end, old_n = per_image_ranges_t0[old_idx]
            new_start, new_end, new_n = per_image_ranges_t1[new_idx]

            if old_n != new_n:
                raise ValueError(
                    f"Per-image token count mismatch: "
                    f"old_img={old_idx}({old_n}tok) vs new_img={new_idx}({new_n}tok). "
                    f"Qwen2-VL는 이미지 해상도가 동일하면 n_tok이 동일해야 함."
                )

            # KV 이식: t0의 cam_c_f0 위치에 t1의 cam_c_f3 KV를 덮어씀
            k_sw[:, :, old_start:old_end, :] = k_t1_l[:, :, new_start:new_end, :]
            v_sw[:, :, old_start:old_end, :] = v_t1_l[:, :, new_start:new_end, :]

        sw_pairs.append((k_sw.contiguous(), v_sw.contiguous()))

    kv_sw = _build_cache_from_kv(sw_pairs, seen_tokens=vision_end)
    assembly_ms = t.stop_ms()

    replaced_tok_total = sum(per_image_ranges_t0[c * n_frames][2] for c in range(n_cameras))
    logger.info(
        f"  [sw_assemble] {n_layers}L, "
        f"transplant_pairs={transplant_pairs}, "
        f"replaced_img_toks={replaced_tok_total} / {vision_end} vision toks "
        f"({replaced_tok_total / vision_end * 100:.1f}%), "
        f"{assembly_ms:.1f}ms"
    )
    log_cache_info(kv_sw, label="KV_sw")
    return kv_sw, assembly_ms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KV 유사도 — 전체 / 구간별
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_kv_similarity_region(
        kv_a: Any,
        kv_b: Any,
        start: int,
        end: int,
        label: str = "",
        n_layers_check: int = 4,
) -> dict:
    """
    두 KV의 [start:end] 구간 코사인 유사도.
    kv_a와 kv_b가 동일한 KV라면 sim≈1.0, 다른 KV라면 < 1.0.
    """
    try:
        pairs_a = _cache_to_kv_pairs(kv_a)
        pairs_b = _cache_to_kv_pairs(kv_b)
        n_layers = min(len(pairs_a), len(pairs_b))
        indices = list(range(0, n_layers, max(1, n_layers // n_layers_check)))[:n_layers_check]

        sims_k, sims_v = [], []
        for li in indices:
            k_a = pairs_a[li][0][:, :, start:end, :].float()
            k_b = pairs_b[li][0][:, :, start:end, :].float()
            v_a = pairs_a[li][1][:, :, start:end, :].float()
            v_b = pairs_b[li][1][:, :, start:end, :].float()

            # 벡터 크기가 0인 경우 예외 처리
            k_a_flat = k_a.flatten()
            k_b_flat = k_b.flatten()
            if k_a_flat.norm() < 1e-8 or k_b_flat.norm() < 1e-8:
                sims_k.append(0.0)
            else:
                sims_k.append(float(
                    F.cosine_similarity(k_a_flat.unsqueeze(0), k_b_flat.unsqueeze(0))
                ))

            v_a_flat = v_a.flatten()
            v_b_flat = v_b.flatten()
            if v_a_flat.norm() < 1e-8 or v_b_flat.norm() < 1e-8:
                sims_v.append(0.0)
            else:
                sims_v.append(float(
                    F.cosine_similarity(v_a_flat.unsqueeze(0), v_b_flat.unsqueeze(0))
                ))

        avg_k  = sum(sims_k) / len(sims_k)
        avg_v  = sum(sims_v) / len(sims_v)
        avg_kv = (avg_k + avg_v) / 2.0

        logger.info(
            f"  [kv_sim/{label}] pos[{start}:{end}] layers{indices}: "
            f"K={avg_k:.4f}, V={avg_v:.4f}, avg={avg_kv:.4f}"
        )
        return {
            "k_sim_avg": round(avg_k, 4),
            "v_sim_avg": round(avg_v, 4),
            "kv_sim_avg": round(avg_kv, 4),
            "k_sim_per_layer": [round(s, 4) for s in sims_k],
            "v_sim_per_layer": [round(s, 4) for s in sims_v],
            "region": [start, end],
            "checked_layers": indices,
        }
    except Exception as e:
        logger.warning(f"  [kv_sim/{label}] 실패: {e}")
        return {"error": str(e), "kv_sim_avg": None}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forward 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def full_prefill(model, input_ids, tok_data, label=""):
    """표준 full prefill."""
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


def suffix_forward(model, suffix_ids, past_kv, start_pos, label=""):
    """
    Suffix-only forward: pixel_values=None, cache_position 명시.
    Exp C의 suffix_prefill과 동일한 구조.
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
# Top-p 샘플링 및 Decode Loop (Exp C와 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def top_p_sample(logits, temperature=TEMPERATURE, top_p=TOP_P):
    logits = logits.float() / temperature
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.zeros_like(logits)
    filtered.scatter_(-1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(filtered, dim=-1), 1).squeeze(-1)


def decode_loop(
        model, first_logits, past_kv, prefill_len,
        eos_id, traj_offset, traj_vocab_size, label="",
):
    """
    자동회귀 디코딩 루프. Exp C와 동일한 패턴.

    핵심:
      ① traj tokens를 logits에서 -inf로 마스킹한 후 CoT 토큰 샘플링
         (clamp가 아닌 masking → EOS 도달 가능)
      ② next_tok.unsqueeze(1) → shape [1, 1] (batch=1, seq=1) ✅
         (double unsqueeze 금지 → [1,1,1]이 되면 embedding 4D → crash)
      ③ cache_position = prefill_len + step - 1 명시 (RoPE 위치 정확성)
    """
    lgts = first_logits.clone()
    lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample(lgts)          # shape [1]

    eos_found = False
    eos_step  = MAX_DECODE_STEPS
    cur       = next_tok.unsqueeze(1)      # shape [1, 1] ← 반드시 unsqueeze(1) 1회

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
                    input_ids=cur,          # shape [1, 1] ✅
                    pixel_values=None,
                    past_key_values=past_kv,
                    cache_position=cpos,
                    use_cache=True,
                )
        except Exception as e:
            ms = t.stop_ms()
            logger.error(f"  [{label}] step {step} 실패: {e}")
            return {"decode_ms": round(ms, 1), "steps": step,
                    "ms_per_step": round(ms / step, 2), "eos_ok": False,
                    "error": str(e)}

        past_kv  = out.past_key_values
        lgts     = out.logits[:, -1, :].float()
        lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample(lgts)
        cur      = next_tok.unsqueeze(1)   # shape [1, 1]

        if next_tok.item() == eos_id:
            eos_found = True
            eos_step  = step
            break

    ms    = t.stop_ms()
    steps = eos_step + 1
    logger.info(
        f"  [{label}] decode: {ms:.0f}ms  "
        f"{steps}steps × {ms/steps:.1f}ms/step  "
        f"eos={'✅' if eos_found else '❌ (MAX_STEPS)'}"
    )
    return {
        "decode_ms":   round(ms, 1),
        "steps":       steps,
        "ms_per_step": round(ms / steps, 2),
        "eos_ok":      eos_found,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 Δt 실험
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_one_delta_t(
        model,
        input_ids_t0, tok_data_t0,
        input_ids_t1, tok_data_t1,
        regions,
        frame_info_t0,
        frame_info_t1,
        eos_id, traj_offset, traj_vocab_size,
        delta_t_ms,
):
    """
    단일 Δt에서 세 가지 방식 비교:
      1. Full prefill (t1 기준선)
      2. Exp C 방식 (t0 vision KV 통째 재사용)
      3. Sliding Window (oldest frame KV → newest frame KV 교체)

    세 방식 모두 동일한 t0 full prefill과 t1 full prefill을 공유.
    """
    vision_start = regions["vision_start"]
    vision_end   = regions["vision_end"]
    suffix_start = regions["suffix_start"]

    # Per-image ranges: [(start, end, n_tok)] × 16 images (camera-first order)
    per_image_ranges_t0 = frame_info_t0["per_image_ranges"]
    per_image_ranges_t1 = frame_info_t1["per_image_ranges"]
    n_imgs = len(per_image_ranges_t0)
    n_frames = frame_info_t0["n_temporal_frames"]    # 4
    n_cameras = frame_info_t0["n_cameras"]           # 4

    # Transplant 대상 (per-image 방식):
    # oldest (t-3) = cam_c's image index c*n_frames+0   → {0,4,8,12}
    # newest (t+1) = cam_c's image index c*n_frames+(n_frames-1) in t1 → {3,7,11,15}
    transplant_pairs_info = [
        {
            "cam": c,
            "old_img_idx": c * n_frames,
            "new_img_idx": c * n_frames + (n_frames - 1),
            "old_range": per_image_ranges_t0[c * n_frames],
            "new_range": per_image_ranges_t1[c * n_frames + (n_frames - 1)],
        }
        for c in range(n_cameras)
    ]

    # suffix_ids: t1의 ego + text_suffix (vision 이후)
    suffix_ids_t1 = input_ids_t1[:, vision_end:]
    actual_suffix_len = int(suffix_ids_t1.shape[1])

    print(f"\n  ── Δt = {delta_t_ms}ms ──────────────────────────────────────────")
    print(f"  images: {n_imgs} total  ({n_cameras} cams × {n_frames} frames)")
    print(f"  transplant pairs (old_img→new_img in t1):", end=" ")
    print([(tp["old_img_idx"], tp["new_img_idx"]) for tp in transplant_pairs_info])
    replaced_tok = sum(tp["old_range"][2] for tp in transplant_pairs_info)
    print(f"  replaced image tokens: {replaced_tok} / {vision_end - vision_start} vision toks"
          f" ({replaced_tok/(vision_end-vision_start)*100:.1f}%)")
    print(f"  suffix_len={actual_suffix_len}, vision_end={vision_end}")
    print(f"  슬라이딩 윈도우 forward tokens: suffix={actual_suffix_len} (Exp C와 동일)")
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
            kv_t0, _, pf_ms_t0 = full_prefill(
                model, input_ids_t0, tok_data_t0,
                label=f"Δt{delta_t_ms}/{tag}/t0_full",
            )
        except Exception as e:
            print(f"t0 prefill FAIL: {e}")
            traceback.print_exc()
            continue

        # ── Step 2: t1 full prefill (기준선) ─────────────────────────────
        try:
            kv_t1_full, logits_t1_full, pf_ms_t1 = full_prefill(
                model, input_ids_t1, tok_data_t1,
                label=f"Δt{delta_t_ms}/{tag}/t1_full",
            )
        except Exception as e:
            print(f"t1 full prefill FAIL: {e}")
            traceback.print_exc()
            continue

        # ── Step 3a: Exp C (t0 vision KV 통째 재사용) ────────────────────
        try:
            # KV_t0[:vision_end]: text_prefix + 전체 vision KV
            pairs_t0_all = _cache_to_kv_pairs(kv_t0)
            kv_expc = _build_cache_from_kv(
                [(k[:, :, :vision_end, :].clone().contiguous(),
                  v[:, :, :vision_end, :].clone().contiguous())
                 for k, v in pairs_t0_all],
                seen_tokens=vision_end,
            )

            kv_after_expc, logits_expc, pf_ms_expc = suffix_forward(
                model, suffix_ids_t1, kv_expc, vision_end,
                label=f"Δt{delta_t_ms}/{tag}/expC_suffix",
            )
        except Exception as e:
            print(f"Exp C suffix FAIL: {e}")
            traceback.print_exc()
            kv_after_expc, logits_expc, pf_ms_expc = None, None, None

        # ── Step 3b: Sliding Window KV 조립 + suffix forward ─────────────
        try:
            kv_sw, assembly_ms = assemble_sliding_window_kv(
                kv_t0, kv_t1_full,
                per_image_ranges_t0=per_image_ranges_t0,
                per_image_ranges_t1=per_image_ranges_t1,
                n_cameras=n_cameras,
                n_frames=n_frames,
                vision_end=vision_end,
            )

            kv_after_sw, logits_sw, pf_ms_sw_suffix = suffix_forward(
                model, suffix_ids_t1, kv_sw, vision_end,
                label=f"Δt{delta_t_ms}/{tag}/sw_suffix",
            )
            pf_ms_sw_total = assembly_ms + pf_ms_sw_suffix
        except Exception as e:
            print(f"Sliding Window FAIL: {e}")
            traceback.print_exc()
            kv_after_sw, logits_sw = None, None
            pf_ms_sw_suffix, assembly_ms, pf_ms_sw_total = None, None, None

        # ── Step 4: KV 유사도 측정 ────────────────────────────────────────
        # 4a. Exp C vs t1_full — 전체 vision 구간
        kv_sim_expC_overall = compute_kv_similarity_region(
            kv_t0, kv_t1_full,
            start=vision_start, end=vision_end,
            label="expC_overall",
        )

        # 4b. SW vs t1_full — 전체 vision 구간 (SW가 t1_full에 얼마나 가까운지)
        kv_sim_sw_vs_t1full = compute_kv_similarity_region(
            kv_sw, kv_t1_full,
            start=vision_start, end=vision_end,
            label="sw_vs_t1full_vision",
        ) if kv_sw is not None else {"error": "sw_failed"}

        # 4c. SW vs t0 — 전체 vision 구간 (이식으로 얼마나 변했는지)
        kv_sim_sw_vs_t0 = compute_kv_similarity_region(
            kv_sw, kv_t0,
            start=vision_start, end=vision_end,
            label="sw_vs_t0_vision",
        ) if kv_sw is not None else {"error": "sw_failed"}

        # 4d. SW: 교체된 이미지별 sim (per-image, 4개 카메라 각각)
        #     SW KV vs t1_full: 교체 위치 → t1의 최신 frame 위치
        #     이론상 sim=1.0 (우리가 직접 복사했으므로)
        #     단, 이를 통해 t1_full의 vision range 전체와 비교 시 어느 위치가 달라지는지 확인
        kv_sim_per_replaced_img = []
        if kv_sw is not None:
            for tp in transplant_pairs_info:
                old_s, old_e, _ = tp["old_range"]
                new_s, new_e, _ = tp["new_range"]
                # SW at old position vs t1_full at new position (should be ~1.0)
                sim_old_vs_new = compute_kv_similarity_region(
                    kv_sw, kv_t1_full,
                    start=old_s, end=old_e,
                    label=f"sw_img{tp['old_img_idx']}_vs_t1img{tp['new_img_idx']}",
                    n_layers_check=2,
                )
                # SW at old position vs t0 at same position (should be ≠1.0 — we changed it)
                sim_old_vs_t0 = compute_kv_similarity_region(
                    kv_sw, kv_t0,
                    start=old_s, end=old_e,
                    label=f"sw_img{tp['old_img_idx']}_vs_t0img{tp['old_img_idx']}",
                    n_layers_check=2,
                )
                kv_sim_per_replaced_img.append({
                    "cam": tp["cam"],
                    "old_img_idx": tp["old_img_idx"],
                    "new_img_idx": tp["new_img_idx"],
                    "old_range": [old_s, old_e],
                    "sim_sw_vs_t1new": sim_old_vs_new.get("kv_sim_avg"),
                    "sim_sw_vs_t0old": sim_old_vs_t0.get("kv_sim_avg"),
                })

        # ── Step 5: Decode ────────────────────────────────────────────────
        prefill_len = int(input_ids_t0.shape[1])
        sw_prefill_len = vision_end + actual_suffix_len  # = total tokens after sw

        dec_full = decode_loop(
            model, logits_t1_full, kv_t1_full, prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"Δt{delta_t_ms}/{tag}/full_decode",
        )

        if logits_expc is not None:
            dec_expc = decode_loop(
                model, logits_expc, kv_after_expc, sw_prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label=f"Δt{delta_t_ms}/{tag}/expC_decode",
            )
        else:
            dec_expc = None

        if logits_sw is not None:
            dec_sw = decode_loop(
                model, logits_sw, kv_after_sw, sw_prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label=f"Δt{delta_t_ms}/{tag}/sw_decode",
            )
        else:
            dec_sw = None

        # ── 집계 ─────────────────────────────────────────────────────────
        saving_expc = ((pf_ms_t1 - pf_ms_expc) / pf_ms_t1 * 100
                       if pf_ms_expc is not None and pf_ms_t1 > 0 else None)
        saving_sw   = ((pf_ms_t1 - pf_ms_sw_total) / pf_ms_t1 * 100
                       if pf_ms_sw_total is not None and pf_ms_t1 > 0 else None)

        steps_diff_expc = (abs(dec_expc["steps"] - dec_full["steps"])
                           if dec_expc else None)
        steps_diff_sw   = (abs(dec_sw["steps"] - dec_full["steps"])
                           if dec_sw else None)

        success_expc = (
            pf_ms_expc is not None
            and pf_ms_expc < 250
            and dec_expc is not None
            and dec_expc["eos_ok"]
            and steps_diff_expc is not None
            and steps_diff_expc <= 5
        )
        success_sw = (
            pf_ms_sw_total is not None
            and pf_ms_sw_total < 350  # assembly + suffix
            and dec_sw is not None
            and dec_sw["eos_ok"]
            and steps_diff_sw is not None
            and steps_diff_sw <= 5
        )

        line_expc = (f"full={pf_ms_t1:.0f}ms  expC={pf_ms_expc:.0f}ms  "
                     f"saving={saving_expc:.1f}%  "
                     f"sim={kv_sim_expC_overall.get('kv_sim_avg', '?'):.4f}  "
                     f"steps_diff={steps_diff_expc}  "
                     f"{'✅ OK' if success_expc else '⚠️'}"
                     if pf_ms_expc is not None else "expC=FAIL")
        line_sw = (f"sw_suffix={pf_ms_sw_suffix:.0f}ms  asm={assembly_ms:.1f}ms  "
                   f"total={pf_ms_sw_total:.0f}ms  saving={saving_sw:.1f}%  "
                   f"sw_vs_t1={kv_sim_sw_vs_t1full.get('kv_sim_avg', '?')}  "
                   f"sw_vs_t0={kv_sim_sw_vs_t0.get('kv_sim_avg', '?')}  "
                   f"steps_diff={steps_diff_sw}  "
                   f"{'✅ OK' if success_sw else '⚠️'}"
                   if pf_ms_sw_total is not None else "SW=FAIL")

        print(f"  [ExpC] {line_expc}")
        print(f"  [SW  ] {line_sw}")

        if not is_warmup:
            runs.append({
                "trial": tag,
                # timing
                "pf_t0_ms": round(pf_ms_t0, 1),
                "pf_t1_full_ms": round(pf_ms_t1, 1),
                "pf_expc_ms": round(pf_ms_expc, 1) if pf_ms_expc is not None else None,
                "pf_sw_suffix_ms": round(pf_ms_sw_suffix, 1) if pf_ms_sw_suffix is not None else None,
                "pf_sw_assembly_ms": round(assembly_ms, 1) if assembly_ms is not None else None,
                "pf_sw_total_ms": round(pf_ms_sw_total, 1) if pf_ms_sw_total is not None else None,
                # savings
                "saving_expc_pct": round(saving_expc, 2) if saving_expc is not None else None,
                "saving_sw_pct": round(saving_sw, 2) if saving_sw is not None else None,
                # kv similarities
                "kv_sim_expC_overall": kv_sim_expC_overall,
                "kv_sim_sw_vs_t1full_vision": kv_sim_sw_vs_t1full,
                "kv_sim_sw_vs_t0_vision": kv_sim_sw_vs_t0,
                "kv_sim_per_replaced_img": kv_sim_per_replaced_img,
                # decode
                "decode_full": dec_full,
                "decode_expc": dec_expc,
                "decode_sw": dec_sw,
                "steps_diff_expc": steps_diff_expc,
                "steps_diff_sw": steps_diff_sw,
                "success_expc": success_expc,
                "success_sw": success_sw,
            })

    if not runs:
        return {"delta_t_ms": delta_t_ms, "n_valid": 0, "error": "all trials failed"}

    # ── 평균 집계 ─────────────────────────────────────────────────────────
    def safe_mean(lst):
        valid = [x for x in lst if x is not None]
        return round(sum(valid) / len(valid), 2) if valid else None

    avg = {
        "pf_full_ms":     safe_mean([r["pf_t1_full_ms"] for r in runs]),
        # Exp C
        "expc_suffix_ms":  safe_mean([r["pf_expc_ms"] for r in runs]),
        "expc_saving_pct": safe_mean([r["saving_expc_pct"] for r in runs]),
        "expc_speedup":    round(safe_mean([r["pf_t1_full_ms"] for r in runs]) /
                                 safe_mean([r["pf_expc_ms"] for r in runs]), 2)
                           if safe_mean([r["pf_expc_ms"] for r in runs]) else None,
        "expc_eos_rate":   sum(1 for r in runs if r.get("decode_expc") and r["decode_expc"]["eos_ok"]) / len(runs),
        "expc_success_rate": sum(1 for r in runs if r.get("success_expc")) / len(runs),
        # Sliding Window
        "sw_suffix_ms":    safe_mean([r["pf_sw_suffix_ms"] for r in runs]),
        "sw_assembly_ms":  safe_mean([r["pf_sw_assembly_ms"] for r in runs]),
        "sw_total_ms":     safe_mean([r["pf_sw_total_ms"] for r in runs]),
        "sw_saving_pct":   safe_mean([r["saving_sw_pct"] for r in runs]),
        "sw_speedup":      round(safe_mean([r["pf_t1_full_ms"] for r in runs]) /
                                 safe_mean([r["pf_sw_total_ms"] for r in runs]), 2)
                           if safe_mean([r["pf_sw_total_ms"] for r in runs]) else None,
        "sw_eos_rate":     sum(1 for r in runs if r.get("decode_sw") and r["decode_sw"]["eos_ok"]) / len(runs),
        "sw_success_rate": sum(1 for r in runs if r.get("success_sw")) / len(runs),
        # KV sim
        "kv_sim_expC_overall": safe_mean([r["kv_sim_expC_overall"].get("kv_sim_avg") for r in runs]),
        # replaced: per-image sim_sw_vs_t1new の平均 (4 cameras × N_MEASURE runs)
        "kv_sim_sw_replaced":  safe_mean([
            img["sim_sw_vs_t1new"]
            for r in runs
            for img in r.get("kv_sim_per_replaced_img", [])
            if img.get("sim_sw_vs_t1new") is not None
        ]),
        # retained: SW overall vs t0 (교체되지 않은 프레임 포함 전체 vision)
        "kv_sim_sw_retained":  safe_mean([r["kv_sim_sw_vs_t0_vision"].get("kv_sim_avg") for r in runs]),
    }

    print(
        f"\n  [Δt={delta_t_ms}ms 평균]\n"
        f"  ExpC: {avg['expc_suffix_ms']}ms ({avg['expc_saving_pct']}% 절약, "
        f"{avg['expc_speedup']}×, EOS={avg['expc_eos_rate']*100:.0f}%, "
        f"overall_sim={avg['kv_sim_expC_overall']})\n"
        f"  SW  : suffix={avg['sw_suffix_ms']}ms + asm={avg['sw_assembly_ms']}ms = "
        f"{avg['sw_total_ms']}ms ({avg['sw_saving_pct']}% 절약, "
        f"{avg['sw_speedup']}×, EOS={avg['sw_eos_rate']*100:.0f}%)\n"
        f"        replaced_sim={avg['kv_sim_sw_replaced']}  "
        f"retained_sim={avg['kv_sim_sw_retained']}"
    )

    result = {
        "delta_t_ms": delta_t_ms,
        "n_valid": len(runs),
        "frame_slots_t0": frame_info_t0["frame_slots"],
        "frame_slots_t1": frame_info_t1["frame_slots"],
        "transplant_pairs": [(tp["old_img_idx"], tp["new_img_idx"]) for tp in transplant_pairs_info],
        "runs": runs,
        "avg": avg,
    }

    # 중간 저장
    mid_path = OUT / f"results_dt{delta_t_ms}ms.json"
    with open(mid_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  중간 저장: {mid_path}")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 입력 준비 (Exp C와 동일한 패턴)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_inputs(model, processor, data) -> tuple[torch.Tensor, dict]:
    """
    데이터셋에서 input_ids와 tok_data(pixel_values 등) 준비.
    fuse_traj_tokens로 ego 토큰을 input_ids에 삽입.
    Exp C (260602_kv_temporal_reuse_exp_c.py)와 동일한 패턴.
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
    return input_ids, inputs   # inputs = tok_data (pixel_values, attention_mask, image_grid_thw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    parser = argparse.ArgumentParser(description="KV Temporal Reuse — Sliding Window")
    parser.add_argument(
        "--delta-t", nargs="+", type=int,
        default=[100, 300, 500, 1000],
        help="측정할 Δt 값(ms) 목록 (default: 100 300 500 1000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    delta_t_list = args.delta_t

    print("=" * 72)
    print("  KV Temporal Reuse — Sliding Window Experiment")
    print(f"  Δt 목록: {delta_t_list} ms")
    print("=" * 72)
    print()
    print("  비교 방식:")
    print("    Full  : t1 full prefill (기준선)")
    print("    Exp C : t0 vision KV 통째 재사용 (260602 실험)")
    print("    SW    : oldest frame KV → newest frame KV 교체 (이 실험)")
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
    logger.info(f"  → attn_implementation = {model.vlm.config._attn_implementation}")

    # processor 초기화 (tokenization에 필요)
    processor = helper.get_processor(model.tokenizer)

    # eos_id, traj 설정 (Exp C와 동일한 패턴)
    eos_id      = model.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab      = model.config.traj_vocab_size
    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, traj_vocab_size={traj_vocab}")

    # ── t0 데이터 로드 ────────────────────────────────────────────────────
    logger.info(f"t0 데이터 로드 (T0={T0_US/1e6:.1f}s)...")
    raw_t0 = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    input_ids_t0, tok_data_t0 = prepare_inputs(model, processor, raw_t0)
    logger.info(f"  t0 input_ids: {input_ids_t0.shape}  ({input_ids_t0.shape[1]} tokens)")

    # vision 경계 탐지 (t0 기준)
    regions = detect_vision_regions(model, input_ids_t0)
    logger.info(
        f"  vision=[{regions['vision_start']},{regions['vision_end']}), "
        f"suffix_len={regions['suffix_len']}"
    )

    # t0 temporal frame 경계 탐지
    logger.info("t0 temporal frame 경계 탐지...")
    frame_info_t0 = detect_temporal_frame_boundaries(
        tok_data_t0, input_ids_t0,
        vision_start=regions["vision_start"],
        vision_end=regions["vision_end"],
        image_pad_id=regions.get("image_pad_id", 151655),
    )

    # ── 각 Δt 실험 ───────────────────────────────────────────────────────
    all_results = {
        "attn_implementation": model.vlm.config._attn_implementation,
        "clip_id": CLIP_ID,
        "t0_us": T0_US,
        "delta_t_list_ms": delta_t_list,
        "regions": {k: v for k, v in regions.items() if k != "image_pad_id"},
        "frame_info_t0": {
            "frame_slots": frame_info_t0["frame_slots"],
            "oldest_slot": frame_info_t0["oldest_frame_slot"],
            "n_tok_per_image": frame_info_t0["n_tok_per_image"][:4],  # first 4만 저장
        },
        "baseline": {
            "appendonly_total_ms": BASELINE_APPENDONLY_TOTAL_MS,
            "appendonly_prefill_ms": BASELINE_PREFILL_MS,
            "expC_suffix_ms": BASELINE_EXP_C_SUFFIX_MS,
        },
        "experiments": {},
    }

    for delta_t_ms in delta_t_list:
        t1_us = T0_US + delta_t_ms * 1000
        logger.info(f"\n{'─'*60}")
        logger.info(f"Δt={delta_t_ms}ms: t1={t1_us/1e6:.3f}s 데이터 로드...")

        raw_t1 = load_physical_aiavdataset(CLIP_ID, t0_us=t1_us)
        input_ids_t1, tok_data_t1 = prepare_inputs(model, processor, raw_t1)
        logger.info(f"  t1 input_ids: {input_ids_t1.shape}  ({input_ids_t1.shape[1]} tokens)")

        # t1 temporal frame 경계 탐지
        frame_info_t1 = detect_temporal_frame_boundaries(
            tok_data_t1, input_ids_t1,
            vision_start=regions["vision_start"],
            vision_end=regions["vision_end"],
            image_pad_id=regions.get("image_pad_id", 151655),
        )

        try:
            result = run_one_delta_t(
                model=model,
                input_ids_t0=input_ids_t0, tok_data_t0=tok_data_t0,
                input_ids_t1=input_ids_t1, tok_data_t1=tok_data_t1,
                regions=regions,
                frame_info_t0=frame_info_t0,
                frame_info_t1=frame_info_t1,
                eos_id=eos_id,
                traj_offset=traj_offset,
                traj_vocab_size=traj_vocab,
                delta_t_ms=delta_t_ms,
            )
            all_results["experiments"][f"dt_{delta_t_ms}ms"] = result
        except Exception as e:
            logger.error(f"Δt={delta_t_ms}ms 실험 실패: {e}")
            traceback.print_exc()
            all_results["experiments"][f"dt_{delta_t_ms}ms"] = {
                "delta_t_ms": delta_t_ms, "error": str(e)
            }

    # ── 최종 저장 ─────────────────────────────────────────────────────────
    final_path = OUT / "results.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n전체 결과: {final_path}")

    # ── 요약 출력 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ★ Sliding Window Experiment 종합 결과")
    print("=" * 72)
    print(f"  {'Δt':>8}  {'ExpC_ms':>8}  {'SW_ms':>8}  {'SW_asm':>7}  "
          f"{'SW_saving':>9}  {'replaced_sim':>12}  {'retained_sim':>12}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  "
          f"{'─'*9}  {'─'*12}  {'─'*12}")

    for dt_ms, res in all_results["experiments"].items():
        if "error" in res and "avg" not in res:
            print(f"  {res.get('delta_t_ms', dt_ms):>7}ms  ERROR: {res['error']}")
            continue
        avg = res.get("avg", {})
        print(
            f"  {res['delta_t_ms']:>7}ms  "
            f"{avg.get('expc_suffix_ms', '?'):>8}  "
            f"{avg.get('sw_suffix_ms', '?'):>8}  "
            f"{avg.get('sw_assembly_ms', '?'):>7}  "
            f"{str(avg.get('sw_saving_pct', '?'))+'%':>9}  "
            f"{str(avg.get('kv_sim_sw_replaced', '?')):>12}  "
            f"{str(avg.get('kv_sim_sw_retained', '?')):>12}  "
            f"EOS={avg.get('sw_eos_rate', '?')}"
        )

    print()
    print("  해석 가이드:")
    print("    replaced_sim ≈ 1.0: t1 newest frame KV 이식 성공 (mrope 가정 성립)")
    print("    replaced_sim < 0.8: 1D RoPE mismatch → SW가 Exp C보다 나쁨")
    print("    SW_ms ≈ ExpC_ms + SW_asm: assembly overhead만큼 Exp C보다 느림이 정상")
    print(f"\n  전체 결과: {final_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
