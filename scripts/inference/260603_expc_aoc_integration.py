"""
260603_expc_aoc_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[목적]
  Exp C (KV Temporal Reuse) + AppendOnlyCache-C 통합 실험

[문제 배경]
  260602 Exp C 스크립트는 suffix forward 후 decode를 DynamicCache로 실행한다.
  DynamicCache.update()는 매 step마다 torch.cat으로 전체 KV를 새 버퍼에 복사한다.
    → ~455 MB × N_steps = 불필요한 BW 낭비

  260531 AppendOnlyCache-C 실험에서 decode를 in-place write로 교체하면
  DynamicCache 107ms/step → AppendOnlyCache-C 79ms/step (-26%) 가 확인됐다.
  단, 그 실험은 Full Prefill + AOC decode 경로만 검증했다.

[통합 목표]
  Exp C suffix forward 결과가 DynamicCache가 아닌 AppendOnlyCache-C 버퍼에 담기도록 한다.
  그러면 decode도 같은 AOC 버퍼에 in-place append → 79ms/step 목표.

[핵심 구현 아이디어]
  기존 Exp C 흐름:
    t0 full prefill → DynamicCache(3086 tok)
    slice_to(vision_end=3011) → DynamicCache(3011 tok)
    suffix forward → DynamicCache(3086 tok)   ← torch.cat
    decode × N     → DynamicCache(3086+N tok) ← torch.cat 매 step

  통합 흐름 (ExpC-AOC):
    t0 full prefill → DynamicCache(3086 tok)
    load_dyn_into_aoc(end_pos=3011) → AppendOnlyCache-C buf[0:3011] (copy, 1회)
    suffix forward → AppendOnlyCache-C buf[3011:3086] (in-place append)
    decode × N     → AppendOnlyCache-C buf[3086:] (in-place append, 매 step)

  핵심 함수: load_dyn_into_aoc()
    DynamicCache의 앞 end_pos 토큰을 AOC의 pre-alloc 버퍼에 .copy_() 한 번으로 로드.
    _write_pos[layer] = end_pos 로 설정하면 이후 suffix forward가 자동으로
    position end_pos 부터 append 한다.

[Δt = 100ms 고정 규칙 ★★★]
  Alpamayo는 10Hz(100ms) 추론 주기를 목표로 설계됨.
  CoC(Chain of Causation) 인과추론 → long-tail 상황 즉각 대응 필요.
  0.1초 내에도 장면이 결정적으로 달라질 수 있음.
  → Δt = 100ms 단 하나만 유효. 300/500/1000ms는 시스템 설계 원칙 위반.

[비교 3방식]
  FULL : t1 full prefill → DynamicCache decode     (절대 기준선)
  DYN  : Exp C suffix forward → DynamicCache decode (현재 상태)
  AOC  : Exp C suffix forward → AppendOnlyCache-C decode (통합 목표)

[측정 지표]
  - suffix_ms: ExpC 경로의 prefix 시간 (DYN, AOC 동일해야 함)
  - convert_ms: DynamicCache → AOC 변환 비용 (1회성)
  - decode_ms/step: 핵심 지표 (DYN vs AOC)
  - total_ms = suffix_ms + [convert_ms] + decode_ms
  - EOS rate, decode steps

[예상 결과]
  suffix_ms (DYN ≈ AOC): ~141ms (동일 - 같은 suffix forward)
  convert_ms (AOC only): ~5ms (449MB copy, 1회)
  decode ms/step: DYN ~75-107ms → AOC ~79ms
  전체 pipeline 절감: Decode 476ms 감소 (17 steps × 28ms/step)

[실행]
  source ~/alpamayo1.5/a1_5_venv/bin/activate && cd ~/alpamayo1.5
  python3 scripts/inference/260603_expc_aoc_integration.py

[결과]
  profiling_results/260603_expc_aoc_integration/results.json
"""

from __future__ import annotations

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

CLIP_ID          = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US            = 5_100_000          # 기준 프레임 (5.1s)
DELTA_T_MS       = 100                # ★ 100ms 고정 — 변경 불가
T1_US            = T0_US + DELTA_T_MS * 1_000   # = 5_200_000 (5.2s)

DEVICE           = "cuda"
MAX_DECODE_STEPS = 80                 # EOS 미생성 시 최대 step
TEMPERATURE      = 0.6
TOP_P            = 0.98
NUM_WARMUP       = 3                  # JIT 안정화 (260531, 260602 경험)
NUM_MEASURE      = 3

OUT = Path("profiling_results/260603_expc_aoc_integration")
OUT.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUDA 타이머
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
# AppendOnlyCache (260531 검증 완료 구현 — 그대로 사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AppendOnlyCache(DynamicCache):
    """
    DynamicCache 상속 + torch.cat → index_copy_(in-place) 교체.

    설계 원리:
      1. DynamicCache 상속 → isinstance(cache, StaticCache) = False
         → FlashAttention 백엔드 유지 (attn_mask = None 경로)
      2. pre-alloc [B, H_kv, MAX_LEN, D] 버퍼 → decode 중 신규 alloc 없음
      3. update() 에서 in-place write → torch.cat(455MB) 완전 제거
      4. force_contiguous=True: .contiguous() 반환 → non-contiguous 오버헤드 제거
         (260531 실측: force_contiguous=True → 79ms/step ← 최선)

    load_from_dynamic() 사용 시:
      - DynamicCache에서 앞 N 토큰을 버퍼에 직접 로드 (copy_ 1회)
      - 이후 suffix forward, decode 모두 in-place append
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
        force_contiguous: bool = True,   # 항상 True (AppendOnlyCache-C)
    ) -> None:
        super().__init__()
        self.n_layers         = n_layers
        self.n_kv_heads       = n_kv_heads
        self.head_dim         = head_dim
        self.max_seq_len      = max_seq_len
        self.force_contiguous = force_contiguous
        self._write_pos: list[int] = [0] * n_layers

        # pre-alloc 버퍼 (전체 sequence 수용 가능)
        self._k_buf: list[torch.Tensor] = []
        self._v_buf: list[torch.Tensor] = []
        for _ in range(n_layers):
            self._k_buf.append(
                torch.zeros(
                    batch_size, n_kv_heads, max_seq_len, head_dim,
                    device=device, dtype=dtype,
                )
            )
            self._v_buf.append(
                torch.zeros(
                    batch_size, n_kv_heads, max_seq_len, head_dim,
                    device=device, dtype=dtype,
                )
            )

        # DynamicCache 호환 리스트 (빈 상태로 시작, update()에서 채워짐)
        self.key_cache:   list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # 핵심 override
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        in-place write into pre-allocated buffer.

        key_states : [B, H_kv, n_new, D]
        returns    : [B, H_kv, cumulative_len, D]
        """
        n_new = key_states.shape[2]
        pos   = self._write_pos[layer_idx]

        if pos + n_new > self.max_seq_len:
            raise RuntimeError(
                f"AppendOnlyCache overflow at layer {layer_idx}: "
                f"pos={pos}, n_new={n_new}, max_seq_len={self.max_seq_len}"
            )

        # ★ in-place write — 새 토큰만 기록, 기존 KV 복사 없음
        self._k_buf[layer_idx][:, :, pos : pos + n_new, :] = key_states
        self._v_buf[layer_idx][:, :, pos : pos + n_new, :] = value_states
        self._write_pos[layer_idx] += n_new

        cur_len = self._write_pos[layer_idx]

        # 유효 구간 반환
        k_out = self._k_buf[layer_idx][:, :, :cur_len, :]
        v_out = self._v_buf[layer_idx][:, :, :cur_len, :]

        if self.force_contiguous:
            # contiguous copy → FlashAttention stride 요구사항 충족
            # 메모리 할당은 없고 기존 버퍼에 쓰기만 함 (pre-alloc 덕분에)
            k_out = k_out.contiguous()
            v_out = v_out.contiguous()

        # DynamicCache 호환 인터페이스 갱신
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)    # type: ignore[arg-type]
            self.value_cache.append(None)  # type: ignore[arg-type]
        self.key_cache[layer_idx]   = k_out
        self.value_cache[layer_idx] = v_out

        return k_out, v_out

    # ------------------------------------------------------------------
    # DynamicCache 인터페이스 유지
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """실제 누적된 시퀀스 길이."""
        return self._write_pos[layer_idx]

    def get_max_length(self) -> int | None:
        return self.max_seq_len

    def reset(self) -> None:
        """다음 inference를 위한 포인터 초기화 (버퍼 zero-fill 불필요)."""
        for i in range(self.n_layers):
            self._write_pos[i] = 0


def build_appendonly_cache(
    text_config: Any,
    prefill_len: int,
    max_decode: int,
    device: str,
    dtype: torch.dtype,
) -> AppendOnlyCache:
    """
    model.vlm.config.text_config 에서 차원 정보를 자동 추출하여 AOC 생성.

    max_seq_len = prefill_len + max_decode + 16 (안전 여유)
    """
    n_layers    = text_config.num_hidden_layers
    n_kv_heads  = text_config.num_key_value_heads
    hidden_size = text_config.hidden_size
    n_q_heads   = text_config.num_attention_heads
    head_dim    = getattr(text_config, "head_dim", hidden_size // n_q_heads)
    max_seq_len = prefill_len + max_decode + 16   # 안전 여유

    alloc_mb = n_layers * 2 * n_kv_heads * max_seq_len * head_dim * 2 / 1e6
    logger.info(
        f"  [AOC] config: layers={n_layers}, kv_heads={n_kv_heads}, "
        f"head_dim={head_dim}, max_seq_len={max_seq_len}, "
        f"alloc={alloc_mb:.1f}MB"
    )

    return AppendOnlyCache(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=1,
        device=device,
        dtype=dtype,
        force_contiguous=True,   # AppendOnlyCache-C
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DynamicCache 유틸리티 (260602 검증 완료)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cache_to_kv_pairs(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    DynamicCache에서 레이어별 (K, V) 텐서 쌍 추출.
    transformers 버전 무관 4종 폴백.
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
        f"  KV 관련 attr: {kv_attrs}"
    )


def _build_dynamic_cache(
        kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        seen_tokens: int,
) -> DynamicCache:
    """(K, V) 쌍 리스트로 DynamicCache 구성. _seen_tokens 명시 설정."""
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
    """
    DynamicCache의 앞 end_pos 토큰만 잘라 새 DynamicCache 반환.
    DYN 경로(기존 Exp C)에서 사용.
    """
    pairs = _cache_to_kv_pairs(cache)
    sliced = [
        (k[:, :, :end_pos, :].clone().contiguous(),
         v[:, :, :end_pos, :].clone().contiguous())
        for k, v in pairs
    ]
    return _build_dynamic_cache(sliced, seen_tokens=end_pos)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ 핵심 신규 함수: DynamicCache → AppendOnlyCache-C 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dyn_into_aoc(
        dyn_cache: DynamicCache,
        aoc: AppendOnlyCache,
        end_pos: int,
        label: str = "",
) -> float:
    """
    DynamicCache의 앞 end_pos 토큰 KV를 AppendOnlyCache-C 버퍼에 로드한다.

    [동작 원리]
      dyn_cache: t0 full prefill 결과 (3086 tok)
      aoc:       비어있는 AppendOnlyCache-C (max_seq_len=3174)
      end_pos:   vision_end (=3011) — vision KV 구간만 로드

      각 레이어에 대해:
        aoc._k_buf[l][:, :, :end_pos, :].copy_(dyn_cache.key_cache[l][:, :, :end_pos, :])
        aoc._v_buf[l][:, :, :end_pos, :].copy_(dyn_cache.value_cache[l][:, :, :end_pos, :])
        aoc._write_pos[l] = end_pos

      copy_()는 non-contiguous 소스도 올바르게 처리하며,
      pre-alloc 버퍼에 직접 쓰므로 중간 메모리 할당 없음.

      이후 suffix forward(75 tok)가 position end_pos 부터 in-place append.
      이후 decode가 position end_pos+75 부터 in-place append.

    Returns: 변환 소요 시간 (ms)
    """
    kv_pairs = _cache_to_kv_pairs(dyn_cache)

    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()

    for layer_idx, (k, v) in enumerate(kv_pairs):
        # k shape: [1, H_kv, full_seq, D]
        # copy_ handles non-contiguous src automatically
        aoc._k_buf[layer_idx][:, :, :end_pos, :].copy_(k[:, :, :end_pos, :])
        aoc._v_buf[layer_idx][:, :, :end_pos, :].copy_(v[:, :, :end_pos, :])
        aoc._write_pos[layer_idx] = end_pos

        # DynamicCache 호환 인터페이스 설정
        while len(aoc.key_cache) <= layer_idx:
            aoc.key_cache.append(None)    # type: ignore[arg-type]
            aoc.value_cache.append(None)  # type: ignore[arg-type]

        k_view = aoc._k_buf[layer_idx][:, :, :end_pos, :]
        v_view = aoc._v_buf[layer_idx][:, :, :end_pos, :]
        if aoc.force_contiguous:
            aoc.key_cache[layer_idx]   = k_view.contiguous()
            aoc.value_cache[layer_idx] = v_view.contiguous()
        else:
            aoc.key_cache[layer_idx]   = k_view
            aoc.value_cache[layer_idx] = v_view

    # DynamicCache 호환: _seen_tokens
    aoc._seen_tokens = end_pos

    ms = t.stop_ms()
    total_mb = (
        len(kv_pairs) * 2 * end_pos
        * kv_pairs[0][0].shape[1]   # n_kv_heads
        * kv_pairs[0][0].shape[3]   # head_dim
        * 2  # BF16 2bytes
        / 1e6
    )
    logger.info(
        f"  [{label}] load_dyn_into_aoc: {ms:.1f}ms  "
        f"({len(kv_pairs)} layers × {end_pos} tok = {total_mb:.0f}MB)"
    )
    return ms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vision 경계 탐지 (260602에서 검증됨)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_vision_regions(model: Any, input_ids: torch.Tensor) -> dict:
    """
    input_ids에서 vision 토큰 구간을 자동 탐지.

    Returns:
      vision_start, vision_end, suffix_len, total_len 등
    """
    ids   = input_ids[0].tolist()
    total = len(ids)

    # 방법 1: image_token_id (이미지 패치 토큰 직접 탐지)
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
        return {
            "vision_start": vs,
            "vision_end":   ve,
            "suffix_len":   total - ve,
            "total_len":    total,
        }

    # 방법 2: vision_start/end special token ID
    vs_id = getattr(model.vlm.config, "vision_start_token_id", None)
    ve_id = getattr(model.vlm.config, "vision_end_token_id", None)
    if vs_id is not None and vs_id in ids:
        vs_positions = [i for i, t in enumerate(ids) if t == vs_id]
        ve_positions = [i for i, t in enumerate(ids) if t == ve_id]
        vs = vs_positions[0]
        ve = ve_positions[-1] + 1
        logger.info(
            f"  [vision] vision_start_id={vs_id}, "
            f"vision=[{vs},{ve}), suffix_len={total-ve}"
        )
        return {
            "vision_start": vs,
            "vision_end":   ve,
            "suffix_len":   total - ve,
            "total_len":    total,
        }

    # 방법 3: fallback (260602 실측 기반)
    logger.warning("  [vision] 자동 탐지 실패 → fallback (vision_start=29, len=2982)")
    vs, ve = 29, 3011
    return {
        "vision_start": vs,
        "vision_end":   ve,
        "suffix_len":   total - ve,
        "total_len":    total,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forward 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def full_prefill(
        model:     Any,
        input_ids: torch.Tensor,
        tok_data:  dict,
        label:     str = "",
) -> tuple[DynamicCache, torch.Tensor, float]:
    """
    표준 full prefill (pixel_values 포함, DynamicCache 출력).
    Returns: (past_kv, last_logits, prefill_ms)
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
    return out.past_key_values, out.logits[:, -1, :].float(), ms


def suffix_forward(
        model:      Any,
        suffix_ids: torch.Tensor,
        past_kv:    Any,       # DynamicCache 또는 AppendOnlyCache-C
        start_pos:  int,
        label:      str = "",
) -> tuple[Any, torch.Tensor, float]:
    """
    Suffix-only forward: pixel_values=None, cache_position 명시.

    suffix_ids: [1, suffix_len] — ego + text_suffix (이미지 패치 없음)
    past_kv:    KV[:vision_end] — text_prefix + vision KV (DYN 또는 AOC)
    start_pos:  vision_end — RoPE 위치 정확성을 위해 필수

    ★ DynamicCache와 AppendOnlyCache-C 모두 동일하게 동작.
      DYN: 75 tok → torch.cat → 새 DynamicCache(3086 tok)
      AOC: 75 tok → in-place write → AppendOnlyCache-C buf[3011:3086]
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
            pixel_values=None,           # 이미지 패치 없음 — 필수
            past_key_values=past_kv,
            cache_position=cache_pos,    # RoPE 위치 정확성 — 필수
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
    """Returns shape [batch] (1D, not 2D)."""
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
# Decode Loop (DynamicCache / AppendOnlyCache-C 모두 동작)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def decode_loop(
        model:          Any,
        first_logits:   torch.Tensor,
        past_kv:        Any,
        prefill_len:    int,
        eos_id:         int,
        traj_offset:    int,
        traj_vocab_size: int,
        label:          str = "",
) -> dict | None:
    """
    자동회귀 디코딩 루프.

    DynamicCache  : torch.cat per step → 455MB alloc → 느림
    AppendOnlyCache-C : in-place write per step → alloc 없음 → 빠름

    두 경우 모두 이 함수 하나로 처리된다.
    past_kv 타입에 따라 모델 내부에서 다른 update() 경로를 타게 됨.

    prefill_len: suffix forward 후 KV에 채워진 토큰 수
                 = vision_end + suffix_len = 3011 + 75 = 3086
    """
    # ── 첫 토큰 샘플링 (suffix forward의 last logits에서) ─────────────
    lgts = first_logits.clone()
    # traj 토큰 마스킹 (EOS=155681은 traj 범위 밖 → 이 마스킹으로 도달 가능)
    lgts[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample(lgts)  # shape [1]

    eos_found = False
    eos_step  = MAX_DECODE_STEPS
    cur       = next_tok.unsqueeze(1)  # shape [1, 1] ← 반드시 unsqueeze(1) 단 1회

    if next_tok.item() == eos_id:
        return {"decode_ms": 0.0, "steps": 1, "ms_per_step": 0.0, "eos_ok": True}

    t = CudaTimer()
    torch.cuda.synchronize()
    t.start()

    for step in range(1, MAX_DECODE_STEPS):
        # cache_position: 이번 토큰이 들어갈 절대 위치
        # step=1 → prefill_len 위치 (0-indexed 기준)
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

        # AOC의 경우 out.past_key_values는 같은 객체 (in-place 수정됨)
        # DynamicCache의 경우도 같은 객체 (torch.cat 결과가 in-place 갱신)
        # 어느 경우든 이 재할당은 안전함
        past_kv  = out.past_key_values
        lgts     = out.logits[:, -1, :].float()
        lgts[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample(lgts)
        cur      = next_tok.unsqueeze(1)  # shape [1, 1]

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
        "decode_ms":    round(ms, 1),
        "steps":        steps,
        "ms_per_step":  round(ms_per_step, 2),
        "eos_ok":       eos_found,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 입력 준비 (260602 검증 완료)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_inputs(model: Any, processor: Any, data: dict) -> tuple[torch.Tensor, dict]:
    """
    데이터셋 → input_ids + tok_data 준비.
    fuse_traj_tokens으로 ego 토큰을 input_ids에 삽입.
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
    inputs     = helper.to_device(inputs, DEVICE)
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
    단일 실험 (Δt=100ms 고정).

    NUM_WARMUP + NUM_MEASURE 회 반복.
    매 trial마다:
      - FULL  : t1 full prefill + DynamicCache decode
      - DYN   : t0 vision KV 재사용 + suffix + DynamicCache decode (현재 Exp C)
      - AOC   : t0 vision KV 재사용 + suffix + AppendOnlyCache-C decode (통합 목표)

    3방식을 같은 trial에서 같은 t0/t1 KV로 실행 → 공정한 비교.
    """
    vision_end       = regions["vision_end"]
    prefill_len      = int(input_ids_t0.shape[1])  # = 3086
    suffix_ids_t1    = input_ids_t1[:, vision_end:]
    suffix_len       = int(suffix_ids_t1.shape[1])  # = 75
    reuse_prefill_len = vision_end + suffix_len      # = 3086 (같음)

    logger.info(f"  vision_end={vision_end}, suffix_len={suffix_len}, "
                f"prefill_len={prefill_len}, reuse_prefill_len={reuse_prefill_len}")

    # text_config 추출 (AOC 생성에 필요)
    text_config = model.vlm.config.text_config

    runs = []

    for trial_idx in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial_idx < NUM_WARMUP
        tag = (f"WARMUP {trial_idx + 1}" if is_warmup
               else f"MEASURE {trial_idx - NUM_WARMUP + 1}")

        torch.cuda.empty_cache()
        logger.info(f"\n  ─── [{tag}] ────────────────────────────────────────")

        # ── Step 1: t0 full prefill (모든 경로 공통) ──────────────────────
        try:
            kv_t0, _, pf_t0_ms = full_prefill(
                model, input_ids_t0, tok_data_t0,
                label=f"{tag}/t0_full",
            )
        except Exception as e:
            logger.error(f"[{tag}] t0_full_prefill 실패: {e}")
            traceback.print_exc()
            continue

        # ── Step 2: t1 full prefill (FULL 경로 + 기준선) ─────────────────
        try:
            kv_t1_full, logits_t1_full, pf_t1_ms = full_prefill(
                model, input_ids_t1, tok_data_t1,
                label=f"{tag}/t1_full",
            )
        except Exception as e:
            logger.error(f"[{tag}] t1_full_prefill 실패: {e}")
            traceback.print_exc()
            continue

        # ── Step 3: FULL decode (DynamicCache) ────────────────────────────
        dec_full = decode_loop(
            model, logits_t1_full, kv_t1_full, prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"{tag}/FULL_decode",
        )

        # ── Step 4: DYN 경로 (현재 Exp C) ────────────────────────────────
        # 4a. t0 KV[:vision_end] 슬라이스 → DynamicCache
        kv_t0_dyn = slice_dynamic_cache(kv_t0, vision_end)

        # 4b. suffix forward with DynamicCache
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
            kv_dyn, logits_dyn, suf_dyn_ms = None, None, None

        # 4c. DYN decode (DynamicCache → torch.cat per step)
        dec_dyn = None
        if kv_dyn is not None and logits_dyn is not None:
            dec_dyn = decode_loop(
                model, logits_dyn, kv_dyn, reuse_prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label=f"{tag}/DYN_decode",
            )

        # ── Step 5: AOC 경로 (통합 목표) ─────────────────────────────────
        # 5a. 비어있는 AppendOnlyCache-C 생성
        #     max_seq_len = reuse_prefill_len + MAX_DECODE_STEPS + 16
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

        # 5b. DynamicCache[:vision_end] → AppendOnlyCache-C 버퍼에 로드
        #     이것이 통합의 핵심: kv_t0의 vision KV가 AOC 버퍼 0:3011에 복사됨
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

        # 5c. suffix forward with AppendOnlyCache-C
        #     in-place write: AOC buf[3011:3086] ← suffix KV
        suf_aoc_ms  = None
        logits_aoc  = None
        kv_aoc      = None
        if conv_ms is not None:
            try:
                kv_aoc, logits_aoc, suf_aoc_ms = suffix_forward(
                    model,
                    suffix_ids=suffix_ids_t1,
                    past_kv=aoc,
                    start_pos=vision_end,
                    label=f"{tag}/AOC_suffix",
                )
            except Exception as e:
                logger.error(f"[{tag}] AOC suffix_forward 실패: {e}")
                traceback.print_exc()

        # 5d. AOC decode (AppendOnlyCache-C → in-place write per step → 빠름)
        dec_aoc = None
        if kv_aoc is not None and logits_aoc is not None:
            dec_aoc = decode_loop(
                model, logits_aoc, kv_aoc, reuse_prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label=f"{tag}/AOC_decode",
            )

        # ── 결과 출력 ─────────────────────────────────────────────────────
        def fmt_dec(d):
            if d is None:
                return "FAIL"
            eos = "✅" if d["eos_ok"] else "❌"
            return f"{d['decode_ms']:.0f}ms/{d['steps']}steps/{d['ms_per_step']:.1f}ms/step {eos}"

        # DYN total = suffix + decode (convert는 DYN에 없음)
        dyn_total = None
        if suf_dyn_ms is not None and dec_dyn is not None:
            dyn_total = suf_dyn_ms + dec_dyn["decode_ms"]

        # AOC total = suffix + convert + decode (or suffix + decode if conv merged)
        aoc_total = None
        if conv_ms is not None and suf_aoc_ms is not None and dec_aoc is not None:
            aoc_total = suf_aoc_ms + conv_ms + dec_aoc["decode_ms"]

        # None-safe 포맷 헬퍼
        def fms(v):
            return f"{v:.0f}ms" if v is not None else "FAIL"
        def fms1(v):
            return f"{v:.1f}ms" if v is not None else "FAIL"

        print(f"\n  [{tag}] 결과 요약:")
        print(f"    FULL : prefill={pf_t1_ms:.0f}ms  decode={fmt_dec(dec_full)}")
        print(f"    DYN  : suffix={fms(suf_dyn_ms)}  decode={fmt_dec(dec_dyn)}"
              f"  total={fms(dyn_total)}")
        print(f"    AOC  : conv={fms1(conv_ms)}  suffix={fms(suf_aoc_ms)}"
              f"  decode={fmt_dec(dec_aoc)}  total={fms(aoc_total)}")

        if dec_dyn is not None and dec_aoc is not None:
            step_diff = dec_aoc["ms_per_step"] - dec_dyn["ms_per_step"]
            sign = "+" if step_diff >= 0 else ""
            print(f"    AOC vs DYN decode: {sign}{step_diff:.1f}ms/step")

        if not is_warmup:
            runs.append({
                "trial": tag,
                # timing
                "pf_t0_ms":       round(pf_t0_ms, 1),
                "pf_t1_full_ms":  round(pf_t1_ms, 1),
                "suf_dyn_ms":     round(suf_dyn_ms, 1) if suf_dyn_ms else None,
                "conv_ms":        round(conv_ms, 1) if conv_ms else None,
                "suf_aoc_ms":     round(suf_aoc_ms, 1) if suf_aoc_ms else None,
                # decode
                "dec_full":  dec_full,
                "dec_dyn":   dec_dyn,
                "dec_aoc":   dec_aoc,
                # totals (suffix + decode, excluding t0 prefill as that is shared)
                "dyn_total_ms": round(dyn_total, 1) if dyn_total else None,
                "aoc_total_ms": round(aoc_total, 1) if aoc_total else None,
                # success flags
                "dyn_ok": (dec_dyn is not None and dec_dyn["eos_ok"] and suf_dyn_ms is not None),
                "aoc_ok": (dec_aoc is not None and dec_aoc["eos_ok"] and suf_aoc_ms is not None),
            })

    # ── 집계 ──────────────────────────────────────────────────────────────
    valid = [r for r in runs if r.get("dyn_ok") or r.get("aoc_ok")]

    def safe_mean(lst):
        lst = [x for x in lst if x is not None]
        return round(sum(lst) / len(lst), 2) if lst else None

    avg = {
        "pf_t1_full_ms":     safe_mean([r["pf_t1_full_ms"] for r in runs]),
        # DYN
        "suf_dyn_ms":        safe_mean([r["suf_dyn_ms"] for r in runs]),
        "dec_dyn_ms":        safe_mean([r["dec_dyn"]["decode_ms"] for r in runs if r.get("dec_dyn")]),
        "dec_dyn_ms_per_step": safe_mean([r["dec_dyn"]["ms_per_step"] for r in runs if r.get("dec_dyn")]),
        "dyn_total_ms":      safe_mean([r["dyn_total_ms"] for r in runs]),
        "dyn_eos_rate":      sum(1 for r in runs if r.get("dec_dyn") and r["dec_dyn"]["eos_ok"]) / max(len(runs), 1),
        # AOC
        "conv_ms":           safe_mean([r["conv_ms"] for r in runs]),
        "suf_aoc_ms":        safe_mean([r["suf_aoc_ms"] for r in runs]),
        "dec_aoc_ms":        safe_mean([r["dec_aoc"]["decode_ms"] for r in runs if r.get("dec_aoc")]),
        "dec_aoc_ms_per_step": safe_mean([r["dec_aoc"]["ms_per_step"] for r in runs if r.get("dec_aoc")]),
        "aoc_total_ms":      safe_mean([r["aoc_total_ms"] for r in runs]),
        "aoc_eos_rate":      sum(1 for r in runs if r.get("dec_aoc") and r["dec_aoc"]["eos_ok"]) / max(len(runs), 1),
    }

    # decode ms/step 개선량
    if avg["dec_dyn_ms_per_step"] and avg["dec_aoc_ms_per_step"]:
        avg["decode_improvement_ms_per_step"] = round(
            avg["dec_dyn_ms_per_step"] - avg["dec_aoc_ms_per_step"], 2
        )
        avg["decode_improvement_pct"] = round(
            avg["decode_improvement_ms_per_step"] / avg["dec_dyn_ms_per_step"] * 100, 1
        )

    print(
        f"\n  ════════════════════════════════════════════════════════"
        f"\n  ★ 통합 실험 (Δt=100ms) 평균 결과 ({NUM_MEASURE}회 측정)"
        f"\n  ════════════════════════════════════════════════════════"
        f"\n  [기준선] FULL prefill : {avg['pf_t1_full_ms']}ms"
        f"\n  [DYN] suffix={avg['suf_dyn_ms']}ms  "
        f"decode={avg['dec_dyn_ms']}ms ({avg['dec_dyn_ms_per_step']}ms/step)  "
        f"total={avg['dyn_total_ms']}ms  EOS={avg['dyn_eos_rate']*100:.0f}%"
        f"\n  [AOC] conv={avg['conv_ms']}ms  suffix={avg['suf_aoc_ms']}ms  "
        f"decode={avg['dec_aoc_ms']}ms ({avg['dec_aoc_ms_per_step']}ms/step)  "
        f"total={avg['aoc_total_ms']}ms  EOS={avg['aoc_eos_rate']*100:.0f}%"
    )
    if avg.get("decode_improvement_ms_per_step") is not None:
        print(
            f"\n  ★ Decode 개선: {avg['decode_improvement_ms_per_step']}ms/step "
            f"({avg['decode_improvement_pct']}%)"
            f"\n  ★ 총 개선 (17steps 기준): "
            f"~{avg['decode_improvement_ms_per_step'] * 17:.0f}ms"
        )
    print("  ════════════════════════════════════════════════════════")

    return {
        "delta_t_ms": DELTA_T_MS,
        "n_runs":     len(runs),
        "n_valid":    len(valid),
        "avg":        avg,
        "runs":       runs,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 72)
    print("  Exp C + AppendOnlyCache-C 통합 실험")
    print(f"  Δt = {DELTA_T_MS}ms (고정 규칙)")
    print(f"  t0 = {T0_US/1e6:.1f}s  →  t1 = {T1_US/1e6:.1f}s")
    print("=" * 72)
    print()
    print("  비교 방식:")
    print("    FULL: t1 full prefill + DynamicCache decode (절대 기준선)")
    print("    DYN : Exp C suffix + DynamicCache decode   (현재 상태)")
    print("    AOC : Exp C suffix + AppendOnlyCache-C decode (통합 목표)")
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

    # ── Vision 경계 탐지 ──────────────────────────────────────────────────
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

    # ── 최종 요약 테이블 ──────────────────────────────────────────────────
    avg = result["avg"]
    print("\n" + "=" * 72)
    print("  ★ 최종 결과 요약 (Δt=100ms, 3회 측정 평균)")
    print("=" * 72)
    print(f"  {'방식':<10} {'suffix':<12} {'conv':<10} {'decode total':<16} "
          f"{'ms/step':<12} {'total':<12} {'EOS':<6}")
    print("  " + "-" * 68)
    print(f"  {'FULL':<10} {'—':<12} {'—':<10} "
          f"{avg.get('pf_t1_full_ms', '?'):<16} {'—':<12} "
          f"{avg.get('pf_t1_full_ms', '?'):<12} {'—':<6}  (prefill만)")
    print(f"  {'DYN':<10} {str(avg.get('suf_dyn_ms','?'))+'ms':<12} {'—':<10} "
          f"{str(avg.get('dec_dyn_ms','?'))+'ms':<16} "
          f"{str(avg.get('dec_dyn_ms_per_step','?'))+'ms':<12} "
          f"{str(avg.get('dyn_total_ms','?'))+'ms':<12} "
          f"{avg.get('dyn_eos_rate',0)*100:.0f}%")
    print(f"  {'AOC':<10} {str(avg.get('suf_aoc_ms','?'))+'ms':<12} "
          f"{str(avg.get('conv_ms','?'))+'ms':<10} "
          f"{str(avg.get('dec_aoc_ms','?'))+'ms':<16} "
          f"{str(avg.get('dec_aoc_ms_per_step','?'))+'ms':<12} "
          f"{str(avg.get('aoc_total_ms','?'))+'ms':<12} "
          f"{avg.get('aoc_eos_rate',0)*100:.0f}%")
    print("  " + "-" + "=" * 68)
    if avg.get("decode_improvement_ms_per_step") is not None:
        print(
            f"\n  decode 개선 (DYN→AOC): "
            f"{avg['decode_improvement_ms_per_step']}ms/step "
            f"(-{avg['decode_improvement_pct']}%)"
        )
        print(
            f"  전체 decode 절감 (17 steps): "
            f"~{avg['decode_improvement_ms_per_step'] * 17:.0f}ms"
        )
    print(
        f"\n  conv overhead: {avg.get('conv_ms','?')}ms "
        f"(1회성, DYN→AOC 변환 비용)"
    )
    print(f"\n  결과 파일: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
