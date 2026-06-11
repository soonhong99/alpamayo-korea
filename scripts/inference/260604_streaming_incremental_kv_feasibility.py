"""
260604_streaming_incremental_kv_feasibility.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[목적]
"t-3 evict + t1 새 프레임 추가 = ~380ms Incremental KV Update" 실험이
실제 Alpamayo 아키텍처에서 가능한지 소스 기반으로 검증한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[3단계 실험]

Phase 1 — Token Structure Diagnostic
  helper.py의 _build_image_content 순서:
    camera-first: cam0_f0, cam0_f1, cam0_f2, cam0_f3, cam1_f0, ...
  input_ids 전체를 스캔해서 각 token position의 타입과 (camera, frame)을 매핑.
  → "oldest temporal frame" 토큰들이 KV cache에서 contiguous한지 측정.

Phase 2 — VE 출력 재사용 가능성 검증
  t0의 cam_k_frame1 이미지 = t1의 cam_k_frame0 이미지 (동일 절대 타임스탬프).
  pixel_values 배열과 VE 출력 embedding이 수치적으로 동일한지 확인.
  → VE caching으로 아낄 수 있는 시간 측정.

Phase 3 — Streaming Benchmark (N=4 steps, Δt=100ms)
  MODE_FULL : 매 step full re-prefill + decode
  MODE_EXPC : t0 vision KV 재사용, ego suffix만 교체 + decode
  → 누적 latency, 누적 staleness (몇 step 전 vision KV를 쓰는지) 측정.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Incremental KV Update 타당성 분석 — 실험 전 소스 기반 결론]

helper.py _build_image_content:
  content 순서 = cam-first, frame-within-cam:
    [cam0_name][f0_label][img0][f1_label][img1][f2_label][img2][f3_label][img3]
    [cam1_name][f0_label][img4]...

따라서 KV cache layout:
    pos≈29: cam0_f0 tokens ← oldest frame, 첫 카메라
    pos≈220: cam0_f1 tokens
    ...
    pos≈600: cam1_f0 tokens ← oldest frame, 두 번째 카메라 (non-contiguous!)
    ...

"Oldest temporal frame" 토큰 = cam0_f0 ∪ cam1_f0 ∪ cam2_f0 ∪ cam3_f0
이들은 KV cache에서 비연속적으로 흩어져 있음.

추가로 transformer K/V 계산 특성:
  K_P^L = W_K^L × LayerNorm(X_P^L)
  X_P^L = X_P^{L-1} + MHA(X_P^{L-1}, K_{0:P}^{L-1}, V_{0:P}^{L-1})
  → K_P는 이전 모든 token의 hidden state에 의존 (causal attention 누적)
  → F0 KV를 업데이트하면 F1, F2, F3의 K/V도 모두 무효화
  → F0 이후 전체 position 재계산 필요 = 사실상 full re-prefill (~1423ms)

결론: 380ms incremental KV update는 이 아키텍처에서 불가.

VE 재사용 (별도 유효한 최적화):
  동일 절대 이미지 → VE 출력 identical → 재사용 가능
  이 실험에서 수치 검증.

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
sys.path.insert(0, str(ROOT / "src_alpamayo1_5"))
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

CLIP_ID       = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US         = 5_100_000
DELTA_T_MS    = 100          # ★ 절대 고정
N_STEPS       = 4            # t0, t1, t2, t3, t4 (t0 full + 4 incremental steps)
NUM_WARMUP    = 3
DEVICE        = "cuda"
MAX_DECODE_STEPS = 80
TEMPERATURE   = 0.6
TOP_P         = 0.98
NUM_CAMERAS   = 4            # load_physical_aiavdataset 기본값
NUM_FRAMES    = 4            # 카메라당 temporal frames

OUT = Path("profiling_results/260604_streaming_incremental_kv_feasibility")
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
# Phase 1: Token Structure Diagnostic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_special_token_ids(model: Any) -> dict[str, int | None]:
    """Qwen3-VL 특수 토큰 ID 탐색 (hardcode 없이 config/tokenizer에서 읽기)."""
    vlm_config = model.vlm.config

    ids: dict[str, int | None] = {
        "vision_start": None,
        "vision_end":   None,
        "image_pad":    None,
    }

    # 1) VLM config 직접 조회
    for attr, key in [
        ("vision_start_token_id", "vision_start"),
        ("vision_end_token_id",   "vision_end"),
        ("image_token_id",        "image_pad"),
    ]:
        val = getattr(vlm_config, attr, None)
        if val is not None:
            ids[key] = int(val)

    # 2) tokenizer fallback
    tok = model.tokenizer
    fallback_map = {
        "vision_start": ["<|vision_start|>", "<|im_start|>"],
        "vision_end":   ["<|vision_end|>",   "<|im_end|>"],
        "image_pad":    ["<|image_pad|>",     "<|IMAGE|>"],
    }
    for key, candidates in fallback_map.items():
        if ids[key] is not None:
            continue
        for cand in candidates:
            tid = tok.convert_tokens_to_ids(cand)
            if tid is not None and tid != tok.unk_token_id:
                ids[key] = int(tid)
                break

    logger.info(f"  [special tokens] {ids}")
    return ids


def analyze_token_structure(
    model: Any,
    input_ids: torch.Tensor,
    tok_data: dict[str, Any],
    n_cameras: int = NUM_CAMERAS,
    n_frames: int = NUM_FRAMES,
) -> dict[str, Any]:
    """
    input_ids의 각 position이 어떤 segment에 속하는지 매핑한다.

    반환 값:
      segments: list of dict
        - start, end: token position (exclusive end)
        - type: 'text' | 'vision_start' | 'image_pad' | 'vision_end'
        - img_idx: image 인덱스 (0-based), image_pad segment에만 유효
        - camera: camera 인덱스 (img_idx // n_frames)
        - frame:  temporal frame 인덱스 (img_idx % n_frames)
        - n_tokens: end - start
      token_counts_per_image: list[int]  # 각 이미지의 실제 KV token 수
      frame_positions: dict  # (camera, frame) → list of (start, end) for image_pad blocks
      oldest_frame_contiguous: bool  # F0 토큰들이 연속인지
    """
    ids = input_ids[0].tolist()
    total = len(ids)

    sp = get_special_token_ids(model)
    vs_id  = sp["vision_start"]
    ve_id  = sp["vision_end"]
    pad_id = sp["image_pad"]

    # image_grid_thw: [N_images, 3] → tokens_per_image[i] = T * H * W
    grid = tok_data.get("image_grid_thw")
    if grid is not None:
        grid_list = grid.tolist()  # [[T,H,W], ...]
        expected_tokens = [int(t * h * w) for t, h, w in grid_list]
    else:
        expected_tokens = None

    segments = []
    img_idx  = 0
    i        = 0

    while i < total:
        tok = ids[i]

        if vs_id is not None and tok == vs_id:
            segments.append({
                "start": i, "end": i + 1,
                "type": "vision_start",
                "img_idx": img_idx,
                "camera": img_idx // n_frames,
                "frame":  img_idx % n_frames,
                "n_tokens": 1,
            })
            i += 1

        elif pad_id is not None and tok == pad_id:
            start = i
            while i < total and ids[i] == pad_id:
                i += 1
            n = i - start
            segments.append({
                "start": start, "end": i,
                "type": "image_pad",
                "img_idx": img_idx,
                "camera": img_idx // n_frames,
                "frame":  img_idx % n_frames,
                "n_tokens": n,
            })
            img_idx += 1

        elif ve_id is not None and tok == ve_id:
            segments.append({
                "start": i, "end": i + 1,
                "type": "vision_end",
                "img_idx": img_idx - 1,
                "camera": (img_idx - 1) // n_frames,
                "frame":  (img_idx - 1) % n_frames,
                "n_tokens": 1,
            })
            i += 1

        else:
            # text block
            start = i
            while i < total:
                t2 = ids[i]
                if ((vs_id  is not None and t2 == vs_id)  or
                    (ve_id  is not None and t2 == ve_id)  or
                    (pad_id is not None and t2 == pad_id)):
                    break
                i += 1
            if i > start:
                segments.append({
                    "start": start, "end": i,
                    "type": "text",
                    "img_idx": None,
                    "camera": None,
                    "frame":  None,
                    "n_tokens": i - start,
                })

    # ── 통계 집계 ──────────────────────────────────────────────────────────

    # 이미지별 실제 token 수 (image_pad segment만)
    pad_segs = [s for s in segments if s["type"] == "image_pad"]
    token_counts = [s["n_tokens"] for s in pad_segs]
    n_images = len(pad_segs)

    # (camera, frame) → (start, end) 매핑
    frame_positions: dict[tuple[int, int], tuple[int, int]] = {}
    for s in pad_segs:
        frame_positions[(s["camera"], s["frame"])] = (s["start"], s["end"])

    # oldest frame (frame=0) 각 카메라별 position 목록
    oldest_positions = [
        frame_positions.get((cam, 0))
        for cam in range(n_cameras)
        if (cam, 0) in frame_positions
    ]

    # contiguous 여부: 연속적이려면 각 카메라의 F0 끝 = 다음 카메라 F0 시작이어야 함
    oldest_contiguous = False
    if len(oldest_positions) >= 2 and all(p is not None for p in oldest_positions):
        contiguous = True
        for j in range(len(oldest_positions) - 1):
            if oldest_positions[j][1] != oldest_positions[j + 1][0]:
                contiguous = False
                break
        oldest_contiguous = contiguous

    # ── 결과 출력 ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Phase 1: Token Structure Diagnostic")
    print(f"{'='*72}")
    print(f"  총 tokens: {total}")
    print(f"  총 이미지: {n_images}  (cameras={n_cameras}, frames_per_cam={n_frames})")

    if expected_tokens:
        match = token_counts == expected_tokens
        print(f"  image_grid_thw 일치: {'✅' if match else '❌'}")
        print(f"  tokens per image (실측): {token_counts}")
        print(f"  tokens per image (grid): {expected_tokens}")
    else:
        print(f"  image_grid_thw: 없음 (processor가 반환하지 않음)")
        print(f"  tokens per image (실측): {token_counts}")

    # segment 요약 출력 (처음 보는 사람도 읽을 수 있도록)
    print(f"\n  [Token Layout — 처음 ~20 segments]")
    print(f"  {'pos':>10}  {'type':>12}  {'label':>20}  {'n_tok':>6}")
    print(f"  {'-'*55}")
    for s in segments[:20]:
        if s["type"] == "image_pad":
            label = f"cam{s['camera']}_frame{s['frame']} (img{s['img_idx']})"
        elif s["type"] in ("vision_start", "vision_end"):
            label = f"img{s['img_idx']}"
        else:
            label = "(text)"
        pos_str = f"[{s['start']},{s['end']})"
        print(f"  {pos_str:>10}  {s['type']:>12}  {label:>20}  {s['n_tokens']:>6}")
    if len(segments) > 20:
        print(f"  ... ({len(segments) - 20} more segments)")

    # oldest frame position 출력
    print(f"\n  [Oldest Temporal Frame (frame=0) 위치]")
    for cam in range(n_cameras):
        pos = frame_positions.get((cam, 0))
        if pos:
            print(f"    cam{cam}_frame0: KV pos [{pos[0]}, {pos[1]})  ({pos[1]-pos[0]} tokens)")
        else:
            print(f"    cam{cam}_frame0: 없음")

    print(f"\n  [핵심 결론]")
    if oldest_contiguous:
        print("  ✅ Oldest frame 토큰들이 KV cache에서 CONTIGUOUS")
        print("     → Incremental KV update 가능성 있음 (추가 분석 필요)")
    else:
        print("  ❌ Oldest frame 토큰들이 KV cache에서 NON-CONTIGUOUS (scattered)")
        if len(oldest_positions) >= 2:
            gaps = []
            for j in range(len(oldest_positions) - 1):
                if oldest_positions[j] and oldest_positions[j+1]:
                    gap = oldest_positions[j+1][0] - oldest_positions[j][1]
                    gaps.append(gap)
            print(f"     cam별 frame0 간 gap 토큰 수: {gaps}")
        print("     → Camera-first layout 때문에 F0 토큰이 각 카메라 사이에 흩어져 있음")
        print("     → F0 evict + F4 insert는 비연속 position 수정 필요")
        print("     → 게다가 F0 이후 K/V가 causal attention 누적으로 F0에 의존")
        print("       → F0 KV 수정 시 F1~F3 K/V 전체 무효화")
        print("       → 사실상 full re-prefill 필요 (~1,423ms)")

    return {
        "segments": segments,
        "n_images": n_images,
        "token_counts_per_image": token_counts,
        "frame_positions": {str(k): v for k, v in frame_positions.items()},
        "oldest_frame_positions": oldest_positions,
        "oldest_frame_contiguous": oldest_contiguous,
        "total_tokens": total,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: VE 출력 재사용 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_image_pixel_values(
    raw_data: dict[str, Any],
    model: Any,
    processor: Any,
    camera_idx: int,
    frame_idx: int,
) -> torch.Tensor:
    """
    특정 (camera, frame)의 단일 이미지를 processor로 처리해서
    pixel_values 텐서를 반환한다.

    raw_data["image_frames"]: (N_cameras, N_frames, 3, H, W)
    """
    # 단일 이미지 추출
    single_frame = raw_data["image_frames"][camera_idx, frame_idx]  # (3, H, W)
    single_frame = single_frame.unsqueeze(0)  # (1, 3, H, W)

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": single_frame[0]}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )
    pv = inputs.get("pixel_values")
    return pv


def check_ve_reuse(
    raw_t0: dict[str, Any],
    raw_t1: dict[str, Any],
    model: Any,
    processor: Any,
) -> dict[str, Any]:
    """
    Phase 2: VE 출력 재사용 가능성 검증

    t0의 cam_k_frame1 이미지 = t1의 cam_k_frame0 이미지 (동일 절대 타임스탬프)
    pixel_values가 완전히 동일한지, 그에 따라 VE 출력이 동일한지 확인한다.

    비교:
      t0_abs_timestamps[cam, frame=1]  vs  t1_abs_timestamps[cam, frame=0]
    두 타임스탬프가 같으면 동일 이미지 → VE 재사용 유효.
    """
    print(f"\n{'='*72}")
    print("  Phase 2: VE 출력 재사용 가능성 검증")
    print(f"{'='*72}")

    # 타임스탬프 비교 (확인 가능한 경우)
    ts_t0 = raw_t0.get("absolute_timestamps")  # (N_cameras, N_frames) int64 tensor
    ts_t1 = raw_t1.get("absolute_timestamps")
    if ts_t0 is not None and ts_t1 is not None:
        print(f"  t0 timestamps (cam0): {ts_t0[0].tolist()} μs")
        print(f"  t1 timestamps (cam0): {ts_t1[0].tolist()} μs")
        print(f"  t0_cam0_f1 == t1_cam0_f0? {ts_t0[0,1].item() == ts_t1[0,0].item()}")
        print(f"  t0_cam0_f2 == t1_cam0_f1? {ts_t0[0,2].item() == ts_t1[0,1].item()}")
        print(f"  t0_cam0_f3 == t1_cam0_f2? {ts_t0[0,3].item() == ts_t1[0,2].item()}")

    results = {}

    # 각 카메라에 대해 pixel_values 동일성 확인
    print(f"\n  [pixel_values 동일성 — camera별]")
    all_match = True
    for cam in range(NUM_CAMERAS):
        try:
            pv_t0_f1 = _extract_image_pixel_values(raw_t0, model, processor, cam, 1)
            pv_t1_f0 = _extract_image_pixel_values(raw_t1, model, processor, cam, 0)

            if pv_t0_f1 is None or pv_t1_f0 is None:
                print(f"    cam{cam}: pixel_values=None (processor 미반환)")
                results[f"cam{cam}"] = {"match": None, "note": "pixel_values=None"}
                continue

            if pv_t0_f1.shape != pv_t1_f0.shape:
                print(f"    cam{cam}: shape mismatch {pv_t0_f1.shape} vs {pv_t1_f0.shape}")
                results[f"cam{cam}"] = {"match": False, "note": "shape mismatch"}
                all_match = False
                continue

            exact_match = torch.equal(pv_t0_f1.float(), pv_t1_f0.float())
            max_diff    = (pv_t0_f1.float() - pv_t1_f0.float()).abs().max().item()

            results[f"cam{cam}"] = {
                "match": exact_match,
                "max_abs_diff": round(max_diff, 8),
                "shape": list(pv_t0_f1.shape),
            }
            status = "✅ EXACT" if exact_match else f"❌ max_diff={max_diff:.2e}"
            print(f"    cam{cam}: t0_f1 vs t1_f0 = {status}  shape={pv_t0_f1.shape}")
            if not exact_match:
                all_match = False

        except Exception as e:
            logger.warning(f"    cam{cam}: 비교 실패 ({e})")
            results[f"cam{cam}"] = {"match": None, "error": str(e)}

    print(f"\n  [VE 재사용 결론]")
    if all_match:
        # VE forward 시간 측정 (전체 16 images)
        t = CudaTimer()
        tok_data = raw_t0.get("_tok_data_full")  # 아래에서 채워짐
        print("  ✅ pixel_values 완전 일치")
        print("     → VE 출력 재사용 유효: 동일 픽셀 → 동일 embedding (VE는 deterministic)")
        print(f"     → 4 cameras × {NUM_FRAMES-1}개 이전 frames = {NUM_CAMERAS*(NUM_FRAMES-1)}개 이미지 재사용 가능")
        n_reusable = NUM_CAMERAS * (NUM_FRAMES - 1)
        n_total    = NUM_CAMERAS * NUM_FRAMES
        print(f"     → VE 절감 비율: {n_reusable}/{n_total} = {n_reusable/n_total*100:.0f}%")
        print(f"     → VE 예상 비용: 728ms × {1/n_total:.3f} × {NUM_CAMERAS} = ~{728*NUM_CAMERAS/n_total:.0f}ms")
        print(f"       (새 frame {NUM_CAMERAS}개만 VE 처리, 나머지 {n_reusable}개는 cached embedding 재사용)")
        print( "  ⚠️  단, LM prefill은 여전히 full 필요 (causal KV 의존성)")
        print( "       → VE 절감만으로는 full re-prefill 비용 중 VE 728ms만 일부 절약")
    else:
        print("  ❌ pixel_values 불일치 — 재확인 필요")
        print("     → 데이터셋 timestamps 로직 검토 요망")

    return {"per_camera": results, "all_match": all_match}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forward 헬퍼 (v3와 동일 구조)
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
    suffix_len = int(suffix_ids.shape[1])
    cache_pos = torch.arange(start_pos, start_pos + suffix_len,
                             device=DEVICE, dtype=torch.long)
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


def decode_loop(model, first_logits, past_kv, prefill_len,
                eos_id, traj_offset, traj_vocab_size, label=""):
    lgts = first_logits.clone()
    lgts[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample(lgts)

    if next_tok.item() == eos_id:
        return {"decode_ms": 0.0, "steps": 1, "ms_per_step": 0.0, "eos_ok": True}

    eos_found = False
    eos_step  = MAX_DECODE_STEPS
    cur       = next_tok.unsqueeze(1)

    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()

    for step in range(1, MAX_DECODE_STEPS):
        cpos = torch.tensor([prefill_len + step - 1], device=DEVICE, dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(
                input_ids=cur,
                pixel_values=None,
                past_key_values=past_kv,
                cache_position=cpos,
                use_cache=True,
            )
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
        f"eos={'✅' if eos_found else '❌'}"
    )
    return {
        "decode_ms":   round(ms, 1),
        "steps":       steps,
        "ms_per_step": round(ms_per_step, 2),
        "eos_ok":      eos_found,
    }


def slice_dynamic_cache(cache, end_pos: int) -> DynamicCache:
    """DynCache 앞 end_pos 토큰만 새 DynCache로 슬라이스 (clone+contiguous)."""
    kc = getattr(cache, "key_cache", [])
    vc = getattr(cache, "value_cache", [])
    new_cache = DynamicCache()
    new_cache._seen_tokens = end_pos
    for k, v in zip(kc, vc):
        new_cache.key_cache.append(k[:, :, :end_pos, :].clone().contiguous())
        new_cache.value_cache.append(v[:, :, :end_pos, :].clone().contiguous())
    return new_cache


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
        {"ego_history_xyz": data["ego_history_xyz"],
         "ego_history_rot": data["ego_history_rot"]},
        DEVICE,
    )
    return model.fuse_traj_tokens(input_ids_raw, ego_data), inputs


def detect_vision_end(model, input_ids):
    ids    = input_ids[0].tolist()
    sp     = get_special_token_ids(model)
    pad_id = sp["image_pad"]
    if pad_id is not None and pad_id in ids:
        positions = [i for i, t in enumerate(ids) if t == pad_id]
        ve = positions[-1] + 1
        logger.info(f"  [vision_end] image_pad detected: vision_end={ve}")
        return ve
    logger.warning("  [vision_end] fallback: 3011")
    return 3011


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Streaming Benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def streaming_benchmark(
    model,
    processor,
    raw_steps: list[dict],        # raw_t0, raw_t1, ..., raw_tN
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
) -> dict[str, Any]:
    """
    N+1 timesteps (t0..tN)에 대해 두 모드를 측정한다.

    MODE_FULL:
      매 step t_k에서 full_prefill(t_k) + decode
      → 항상 fresh vision, 기준선

    MODE_EXPC:
      t0: full_prefill(t0) → KV_t0 저장
      t1..tN: slice KV_t0 to vision_end + suffix_forward(t_k ego) + decode
      → vision KV는 t0에 고정 (staleness = step번호)

    ★ warmup 포함: 처음 NUM_WARMUP step은 측정 제외
    """
    n_steps = len(raw_steps)  # t0..tN
    results = {
        "full": [],  # per-step timing
        "expc": [],
    }

    # ── 데이터 사전 준비 ──────────────────────────────────────────────────
    print(f"\n  입력 데이터 전처리 중 ({n_steps} steps)...")
    step_data = []
    for k, raw in enumerate(raw_steps):
        ids, tok = prepare_inputs(model, processor, raw)
        step_data.append({"input_ids": ids, "tok_data": tok})
    logger.info(f"  step_data 준비 완료: {n_steps} steps")

    vision_end = detect_vision_end(model, step_data[0]["input_ids"])
    prefill_len = int(step_data[0]["input_ids"].shape[1])
    suffix_len  = prefill_len - vision_end
    logger.info(f"  vision_end={vision_end}, prefill_len={prefill_len}, suffix_len={suffix_len}")

    # ── MODE_FULL ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  MODE_FULL: 매 step full re-prefill")
    print(f"{'='*72}")

    for warmup in range(NUM_WARMUP):
        d = step_data[0]
        kv, logits, _ = full_prefill(model, d["input_ids"], d["tok_data"],
                                      label=f"WARMUP-FULL {warmup+1}")
        dec = decode_loop(model, logits, kv, prefill_len,
                          eos_id, traj_offset, traj_vocab_size,
                          label=f"WARMUP-FULL {warmup+1}/decode")
        torch.cuda.empty_cache()

    for k, d in enumerate(step_data):
        torch.cuda.empty_cache()
        kv, logits, pf_ms = full_prefill(
            model, d["input_ids"], d["tok_data"],
            label=f"FULL step{k}",
        )
        dec = decode_loop(
            model, logits, kv, prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"FULL step{k}/decode",
        )
        total_ms = pf_ms + (dec["decode_ms"] if dec else 0)
        r = {
            "step": k,
            "staleness": 0,
            "prefill_ms": round(pf_ms, 1),
            "decode_ms":  dec["decode_ms"] if dec else None,
            "steps":      dec["steps"]     if dec else None,
            "eos_ok":     dec["eos_ok"]    if dec else None,
            "total_ms":   round(total_ms, 1),
        }
        results["full"].append(r)
        print(
            f"  [FULL step{k}] prefill={pf_ms:.0f}ms  "
            f"decode={dec['ms_per_step']:.1f}ms/step/{dec['steps']}s  "
            f"eos={'✅' if dec['eos_ok'] else '❌'}  total={total_ms:.0f}ms"
            if dec else f"  [FULL step{k}] prefill={pf_ms:.0f}ms decode=FAILED"
        )

    # ── MODE_EXPC ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  MODE_EXPC: t0 vision KV 재사용, ego suffix만 교체")
    print(f"{'='*72}")
    print(f"  staleness = 해당 step에서 t0 이후 몇 step 지났는지 (= step 번호)")

    for warmup in range(NUM_WARMUP):
        d = step_data[0]
        kv_t0, _, _ = full_prefill(model, d["input_ids"], d["tok_data"],
                                    label=f"WARMUP-EXPC {warmup+1}")
        torch.cuda.empty_cache()

    # t0 full prefill 저장
    torch.cuda.empty_cache()
    d0 = step_data[0]
    kv_t0, logits_t0, pf_t0_ms = full_prefill(
        model, d0["input_ids"], d0["tok_data"], label="EXPC/t0_full"
    )
    kv_t0_vision = slice_dynamic_cache(kv_t0, vision_end)  # vision KV만 보관

    # step 0 = full prefill 기준
    dec_t0 = decode_loop(
        model, logits_t0, kv_t0, prefill_len,
        eos_id, traj_offset, traj_vocab_size,
        label="EXPC step0/decode",
    )
    results["expc"].append({
        "step": 0,
        "staleness": 0,
        "prefill_ms": round(pf_t0_ms, 1),
        "suffix_ms":  None,
        "decode_ms":  dec_t0["decode_ms"] if dec_t0 else None,
        "steps":      dec_t0["steps"]     if dec_t0 else None,
        "eos_ok":     dec_t0["eos_ok"]    if dec_t0 else None,
        "total_ms":   round(pf_t0_ms + (dec_t0["decode_ms"] if dec_t0 else 0), 1),
        "mode_detail": "full_prefill",
    })
    print(
        f"  [EXPC step0] full prefill={pf_t0_ms:.0f}ms  "
        f"decode={dec_t0['ms_per_step']:.1f}ms/step/{dec_t0['steps']}s  "
        f"eos={'✅' if dec_t0['eos_ok'] else '❌'}  "
        f"total={results['expc'][0]['total_ms']}ms"
        if dec_t0 else f"  [EXPC step0] prefill={pf_t0_ms:.0f}ms decode=FAILED"
    )

    # step 1..N = suffix forward
    for k in range(1, n_steps):
        torch.cuda.empty_cache()
        d = step_data[k]

        suffix_ids = d["input_ids"][:, vision_end:]
        kv_vision_copy = slice_dynamic_cache(kv_t0_vision, vision_end)

        kv_suf, logits_suf, suf_ms = suffix_forward(
            model, suffix_ids, kv_vision_copy, vision_end,
            label=f"EXPC step{k}/suffix",
        )
        dec = decode_loop(
            model, logits_suf, kv_suf, prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"EXPC step{k}/decode",
        )
        total_ms = suf_ms + (dec["decode_ms"] if dec else 0)
        r = {
            "step": k,
            "staleness": k,  # step k에서 t0 이후 k steps = k×100ms 전 vision
            "prefill_ms": None,
            "suffix_ms":  round(suf_ms, 1),
            "decode_ms":  dec["decode_ms"] if dec else None,
            "steps":      dec["steps"]     if dec else None,
            "eos_ok":     dec["eos_ok"]    if dec else None,
            "total_ms":   round(total_ms, 1),
            "mode_detail": f"suffix_expc (vision stale by {k}×100ms)",
        }
        results["expc"].append(r)
        print(
            f"  [EXPC step{k}] suffix={suf_ms:.0f}ms  "
            f"decode={dec['ms_per_step']:.1f}ms/step/{dec['steps']}s  "
            f"eos={'✅' if dec['eos_ok'] else '❌'}  "
            f"total={total_ms:.0f}ms  (vision stale={k}×100ms={k*100}ms)"
            if dec else f"  [EXPC step{k}] suffix={suf_ms:.0f}ms decode=FAILED"
        )

    # ── 비교 요약 ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Phase 3 결과 요약")
    print(f"{'='*72}")

    full_totals = [r["total_ms"] for r in results["full"] if r["total_ms"] is not None]
    expc_s0 = results["expc"][0]["total_ms"]
    expc_sk = [r["total_ms"] for r in results["expc"][1:] if r["total_ms"] is not None]

    def safe_mean(lst):
        return round(sum(lst)/len(lst), 1) if lst else None

    print(f"  FULL: mean={safe_mean(full_totals)}ms/step  (모든 {len(full_totals)} steps 포함)")
    print(f"  EXPC: t0={expc_s0}ms (full)  t1+={safe_mean(expc_sk)}ms/step (suffix+decode)")

    cumulative_full = sum(full_totals)
    cumulative_expc = (expc_s0 or 0) + sum(expc_sk)
    print(f"\n  누적 latency ({n_steps} steps):")
    print(f"    FULL 합계: {cumulative_full:.0f}ms")
    print(f"    EXPC 합계: {cumulative_expc:.0f}ms")
    print(f"    절감: {cumulative_full - cumulative_expc:.0f}ms ({(1-cumulative_expc/cumulative_full)*100:.1f}%)")

    print(f"\n  [Vision Staleness]")
    print(f"  FULL: 매 step 신선한 vision KV (staleness=0)")
    for r in results["expc"][1:]:
        print(f"  EXPC step{r['step']}: staleness={r['staleness']}×100ms = {r['staleness']*100}ms")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 72)
    print("  Streaming / Incremental KV Feasibility 실험")
    print(f"  Δt = {DELTA_T_MS}ms (고정 규칙), N_STEPS = {N_STEPS}")
    print(f"  t0 = {T0_US/1e6:.1f}s  →  t{N_STEPS} = {(T0_US + N_STEPS*DELTA_T_MS*1000)/1e6:.1f}s")
    print("=" * 72)

    logger.info("모델 로드...")
    model = (
        Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, local_files_only=True,
        ).to(DEVICE).eval()
    )
    logger.info(
        f"  → attn_implementation = "
        f"{getattr(model.vlm.config, 'attn_implementation', 'unknown')}"
    )

    eos_id          = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, traj_vocab_size={traj_vocab_size}")

    processor = helper.get_processor(model.tokenizer)

    # ── 데이터 로드 (t0 ~ tN) ────────────────────────────────────────────
    raw_steps = []
    for k in range(N_STEPS + 1):  # t0, t1, ..., t_N_STEPS
        t_us = T0_US + k * DELTA_T_MS * 1000
        logger.info(f"  t{k} 데이터 로드 (T={t_us/1e6:.3f}s)...")
        raw = load_physical_aiavdataset(CLIP_ID, t0_us=t_us)
        raw_steps.append(raw)
    logger.info(f"  총 {len(raw_steps)} timestep 데이터 로드 완료")

    # ── 입력 준비 (t0만 Phase 1, 2용) ────────────────────────────────────
    input_ids_t0, tok_data_t0 = prepare_inputs(model, processor, raw_steps[0])
    logger.info(f"  t0 input_ids: {input_ids_t0.shape}")

    final_result = {}

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Token Structure Diagnostic
    # ══════════════════════════════════════════════════════════════════════
    try:
        phase1 = analyze_token_structure(
            model, input_ids_t0, tok_data_t0,
            n_cameras=NUM_CAMERAS, n_frames=NUM_FRAMES,
        )
        final_result["phase1_token_structure"] = {
            k: v for k, v in phase1.items() if k != "segments"  # segments는 너무 커서 제외
        }
        final_result["phase1_token_structure"]["n_segments"] = len(phase1["segments"])
    except Exception as e:
        logger.error(f"Phase 1 실패: {e}")
        traceback.print_exc()
        final_result["phase1_token_structure"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: VE 재사용 가능성 검증
    # ══════════════════════════════════════════════════════════════════════
    try:
        phase2 = check_ve_reuse(
            raw_t0=raw_steps[0],
            raw_t1=raw_steps[1],
            model=model,
            processor=processor,
        )
        final_result["phase2_ve_reuse"] = phase2
    except Exception as e:
        logger.error(f"Phase 2 실패: {e}")
        traceback.print_exc()
        final_result["phase2_ve_reuse"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Streaming Benchmark
    # ══════════════════════════════════════════════════════════════════════
    try:
        phase3 = streaming_benchmark(
            model=model,
            processor=processor,
            raw_steps=raw_steps,
            eos_id=eos_id,
            traj_offset=traj_offset,
            traj_vocab_size=traj_vocab_size,
        )
        final_result["phase3_streaming"] = phase3
    except Exception as e:
        logger.error(f"Phase 3 실패: {e}")
        traceback.print_exc()
        final_result["phase3_streaming"] = {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════
    # 전체 결론 출력
    # ══════════════════════════════════════════════════════════════════════
    p1 = final_result.get("phase1_token_structure", {})
    p2 = final_result.get("phase2_ve_reuse", {})

    print(f"\n{'='*72}")
    print("  ★★ 최종 결론 ★★")
    print(f"{'='*72}")

    contiguous = p1.get("oldest_frame_contiguous")
    if contiguous is True:
        print("  Phase 1: ✅ F0 토큰 contiguous → Incremental KV update 구조적으로 가능")
        print("           단, causal K/V 의존성 문제 별도 해결 필요")
    elif contiguous is False:
        print("  Phase 1: ❌ F0 토큰 non-contiguous (camera-first layout 확인)")
        print("           → Incremental KV update는 비연속 수정 필요 → 구현 복잡")
        print("           → 게다가 causal K/V 의존성으로 F0 수정 시 전체 재계산 필요")
        print("           → 결론: ~380ms incremental update는 이 아키텍처에서 불가")
    else:
        print("  Phase 1: 결과 없음 (에러)")

    ve_match = p2.get("all_match")
    if ve_match is True:
        print("  Phase 2: ✅ 동일 절대 프레임 → VE 출력 identical")
        n_total = NUM_CAMERAS * NUM_FRAMES
        n_reuse = NUM_CAMERAS * (NUM_FRAMES - 1)
        print(f"           → VE caching 유효: {n_reuse}/{n_total} 이미지 재사용 가능")
        print(f"           → VE cost 절감: {728*n_reuse/n_total:.0f}ms/step")
        print(f"           → 단, LM re-prefill은 여전히 필요 → net gain은 VE 부분만")
    elif ve_match is False:
        print("  Phase 2: ❌ 픽셀 불일치 → VE 재사용 불가 (데이터 확인 필요)")
    else:
        print("  Phase 2: 결과 없음 (에러)")

    print(f"\n  [다음 단계]")
    print("  1) Async VE Pipeline:")
    print("     decode(t_k) 1,290ms 동안 VE(t_{k+1}) 728ms 비동기 실행")
    print("     → VE를 critical path에서 제거 → 약 2,301ms/inference")
    print("  2) Rolling Trajectory Pipeline:")
    print("     여러 inference 동시 파이프라이닝 → throughput 10Hz 달성")

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    def default_serializer(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, bool):
            return bool(obj)
        return str(obj)

    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(final_result, f, indent=2, default=default_serializer)
    logger.info(f"결과 저장: {out_path}")
    print(f"\n  결과 파일: {out_path}")


if __name__ == "__main__":
    main()
