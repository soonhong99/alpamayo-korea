"""
260531_appendonly_cache_exp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적:
  DynamicCache 의 torch.cat BW 낭비 문제를 AppendOnlyCache 로 해결한다.

■ 핵심 분석 (실험 전 이론 정리)
  DynamicCache.update():
      key_cache[i] = torch.cat([key_cache[i], k_new], dim=-2)
    → 매 decode step 마다 기존 KV 전체(≈455 MB) 읽고 → 새 버퍼에 쓰기
    → step당 ≈910 MB 불필요 BW 낭비, 17 step = ≈66 ms 손실 추정

  StaticCache.update():
      key_cache[i][:, :, cache_position] = k_new   (in-place, 복사 없음)
    → 장점: torch.cat 없음, BW 손실 없음
    → 단점: get_mask_sizes() = max_cache_len → float 4D mask → MemEfficientAttention (2×↓)

  AppendOnlyCache (이 실험):
      DynamicCache 상속 → isinstance(cache, StaticCache) = False
      → _update_causal_mask 가 DynamicCache 경로 탐 → None 반환 → FlashAttention 유지
      + pre-alloc [B, H, MAX_LEN, D] + index_copy_(in-place) → torch.cat 제거

■ 실험 구성
  Exp-A : DynamicCache    (baseline, torch.cat, FlashAttn)
  Exp-B : AppendOnlyCache (in-place view, FlashAttn 가설, force_contiguous=False)
  Exp-C : AppendOnlyCache-C (in-place + .contiguous(), FlashAttn 가설)
           → B vs C 비교로 non-contiguous view 오버헤드 측정
  Exp-D : StaticCache     (reference, in-place, MemEfficientAttn 2× 느림)
  Exp-E : BoolMaskStaticCache  (monkey-patch _update_causal_mask → bool mask)
           → StaticCache 에서 FlashAttention 복원 가능한지 확인

■ 측정 항목
  - decode ms/step (메인 지표)
  - GPU 메모리 신규 할당 횟수 (torch.cat = 매 step alloc)
  - SDPA 백엔드 추정 (시간 프록시: <130ms → FlashAttn, >160ms → MemEfficient)
  - prefill ms (각 캐시 초기화 비용 차이)

실행 (Thor):
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5
  python3 scripts/inference/260531_appendonly_cache_exp.py

결과:
  profiling_results/260531_appendonly_cache/results.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import StaticCache
from transformers.cache_utils import DynamicCache

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
T0_US            = 5_100_000
DEVICE           = "cuda"
MAX_DECODE_STEPS = 80
TEMPERATURE      = 0.6
TOP_P            = 0.98
NUM_WARMUP       = 1
NUM_MEASURE      = 3

# 실험 선택 (False 로 끄면 해당 실험 건너뜀)
RUN_EXP_A = True   # DynamicCache baseline
RUN_EXP_B = True   # AppendOnlyCache (non-contiguous view)
RUN_EXP_C = True   # AppendOnlyCache-C (force_contiguous)
RUN_EXP_D = True   # StaticCache reference
RUN_EXP_E = True   # BoolMaskStaticCache

OUT = Path("profiling_results/260531_appendonly_cache")
OUT.mkdir(parents=True, exist_ok=True)

# 비교 기준 (기존 실험 측정값)
BASELINE_DYNAMIC_MS_PER_STEP = 107.0
BASELINE_STATIC_MS_PER_STEP  = 214.0   # sdpa+StaticCache 이전 측정 (MemEfficient)
BANDWIDTH_TOTAL_GB = 231.0              # Thor LPDDR5X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CudaStopwatch:
    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._s.record()

    def stop_ms(self) -> float:
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)


def make_static_cache(
    text_config: Any,
    max_cache_len: int,
    device: str,
    dtype: torch.dtype,
) -> StaticCache:
    """
    transformers 버전마다 StaticCache 생성자 파라미터가 다름:
      ≤4.44: max_batch_size=1
      4.45+: batch_size=1
    두 방법을 순서대로 시도.
    """
    for kwargs in [
        {"batch_size": 1},
        {"max_batch_size": 1},
    ]:
        try:
            return StaticCache(
                config=text_config,
                max_cache_len=max_cache_len,
                device=device,
                dtype=dtype,
                **kwargs,
            )
        except TypeError:
            continue
    raise RuntimeError("StaticCache 생성자 호환 파라미터 없음 (batch_size / max_batch_size 모두 실패)")


def top_p_sample_gpu(
    logits: torch.Tensor,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> torch.Tensor:
    logits = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = (
        cumulative_probs - F.softmax(sorted_logits, dim=-1)
    ) > top_p
    sorted_indices_to_remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
    filtered_logits = torch.zeros_like(logits)
    filtered_logits.scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AppendOnlyCache: 핵심 구현
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AppendOnlyCache(DynamicCache):
    """
    DynamicCache를 상속하면서 torch.cat을 index_copy_(in-place)로 교체.

    설계 원리:
    1. DynamicCache 상속 → isinstance(cache, StaticCache) = False
       → _update_causal_mask 가 DynamicCache 경로로 진입
       → q_len=1 decode step 에서 AttentionMaskConverter._ignore_causal_mask_sdpa() = True
       → attn_mask = None → FlashAttention 백엔드 유지
    2. pre-alloc [B, H_kv, MAX_LEN, D] 버퍼 → decode 중 신규 alloc 없음
    3. update() 에서 slice in-place write → torch.cat(455MB) 제거
    4. view 반환 (non-contiguous 하지만 correct stride)

    force_contiguous=True 시:
       view 대신 .contiguous() 호출 → non-contiguous 오버헤드 측정 비교용
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
        force_contiguous: bool = False,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.force_contiguous = force_contiguous
        self._write_pos: list[int] = [0] * n_layers

        # Pre-allocate: 모든 레이어 버퍼를 한 번에 할당
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

        # DynamicCache 호환: 빈 리스트로 시작, update() 에서 append/assign
        # [None]*n_layers 로 하면 get_seq_length() 가 None.shape[-2] 접근 시 crash
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        In-place write into pre-allocated buffer.

        key_states : [B, H_kv, n_new, D]
        returns    : [B, H_kv, cumulative_len, D]  (view or contiguous copy)
        """
        n_new = key_states.shape[2]
        pos = self._write_pos[layer_idx]

        if pos + n_new > self.max_seq_len:
            raise RuntimeError(
                f"AppendOnlyCache overflow at layer {layer_idx}: "
                f"pos={pos}, n_new={n_new}, max_seq_len={self.max_seq_len}"
            )

        # ★ In-place write — 새 토큰만 기록, 기존 KV 복사 없음
        self._k_buf[layer_idx][:, :, pos : pos + n_new, :] = key_states
        self._v_buf[layer_idx][:, :, pos : pos + n_new, :] = value_states
        self._write_pos[layer_idx] += n_new

        cur_len = self._write_pos[layer_idx]

        # 반환: 유효 구간의 view
        k_out = self._k_buf[layer_idx][:, :, :cur_len, :]
        v_out = self._v_buf[layer_idx][:, :, :cur_len, :]

        if self.force_contiguous:
            # Exp-C 용: non-contiguous 오버헤드 측정 비교
            k_out = k_out.contiguous()
            v_out = v_out.contiguous()

        # DynamicCache 호환 (일부 코드가 .key_cache[i] 직접 접근)
        # 빈 리스트에서 시작하므로 첫 방문 시 append, 이후엔 덮어쓰기
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)   # type: ignore[arg-type]
            self.value_cache.append(None) # type: ignore[arg-type]
        self.key_cache[layer_idx] = k_out
        self.value_cache[layer_idx] = v_out

        return k_out, v_out

    # ------------------------------------------------------------------
    # DynamicCache interface 유지
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """실제 누적된 시퀀스 길이 반환 (DynamicCache 동작과 동일)."""
        return self._write_pos[layer_idx]

    def get_max_length(self) -> int | None:
        return self.max_seq_len

    def reset(self) -> None:
        """다음 inference를 위해 포인터만 초기화 (버퍼 zero-fill 불필요)."""
        for i in range(self.n_layers):
            self._write_pos[i] = 0


def build_appendonly_cache(
    text_config: Any,
    prefill_len: int,
    max_decode: int,
    device: str,
    dtype: torch.dtype,
    force_contiguous: bool = False,
) -> AppendOnlyCache:
    """model.vlm.config.text_config 에서 필요한 차원을 자동 추출."""
    n_layers    = text_config.num_hidden_layers
    n_kv_heads  = text_config.num_key_value_heads
    hidden_size = text_config.hidden_size
    n_q_heads   = text_config.num_attention_heads
    head_dim    = getattr(text_config, "head_dim", hidden_size // n_q_heads)
    max_seq_len = prefill_len + max_decode + 8  # safety margin

    logger.info(
        f"AppendOnlyCache config: layers={n_layers}, kv_heads={n_kv_heads}, "
        f"head_dim={head_dim}, max_seq_len={max_seq_len}, "
        f"alloc={n_layers*2*n_kv_heads*max_seq_len*head_dim*2/1e6:.1f}MB"
    )

    return AppendOnlyCache(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=1,
        device=device,
        dtype=dtype,
        force_contiguous=force_contiguous,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BoolMaskStaticCache: _update_causal_mask 패치 (Exp-E)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BoolMaskStaticCache(StaticCache):
    """
    StaticCache 상속 + _update_causal_mask 를 bool mask 경로로 패치.

    StaticCache 의 문제:
      isinstance(cache, StaticCache) = True
      → target_length = max_cache_len (전체 용량)
      → float(-inf) 마스크 생성 → MemEfficientAttention

    이 클래스:
      monkey-patch로 _update_causal_mask 내부에서 실제 seq_len 으로
      target_length 를 제한하거나, bool mask 를 강제 반환.
    """
    pass   # 아래에서 model 레벨 패치로 처리


def install_bool_mask_patch(model: Any) -> tuple[Any, bool]:
    """
    Qwen3VLTextModel._update_causal_mask 를 monkey-patch.
    StaticCache 일 때도 bool mask(또는 None)를 반환하도록 수정.

    Returns: (original_bound_method_or_None, patched_successfully)
    """
    import types

    text_model = model.vlm.model  # Qwen3VLModel

    # _update_causal_mask 존재 여부 확인 (새 transformers 는 create_causal_mask 사용)
    if not hasattr(text_model, "_update_causal_mask"):
        logger.warning(
            "  [Exp-E] _update_causal_mask 메서드 없음 (새 transformers 버전?). "
            "패치 생략."
        )
        return None, False

    # 원본 저장 (bound method)
    original_bound = text_model._update_causal_mask

    def patched_fn(
        self_inner,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values,
        output_attentions: bool = False,
    ):
        from transformers import StaticCache as _SC

        if not isinstance(past_key_values, _SC):
            # Non-StaticCache: 원본 호출 (original_bound 는 closure 에 캡처됨)
            return original_bound(
                attention_mask, input_tensor, cache_position,
                past_key_values, output_attentions,
            )

        # ── StaticCache 패치 경로 ──────────────────────────────────────
        past_seen = past_key_values.get_seq_length()
        seq_len   = input_tensor.shape[1]
        device    = input_tensor.device

        # decode step (seq_len=1): Q가 모든 과거 K를 볼 수 있으므로 마스크 불필요
        if seq_len == 1:
            attn_impl = getattr(self_inner.config, "_attn_implementation", "sdpa")
            if attn_impl == "sdpa":
                return None  # → FlashAttention

        # prefill: bool causal mask (float mask 대신)
        target_length = past_seen + seq_len

        causal_bool = torch.ones(
            seq_len, target_length, dtype=torch.bool, device=device
        ).tril()                        # [seq_len, target_length]
        causal_bool = causal_bool[None, None]  # [1, 1, seq_len, target_length]

        if attention_mask is not None and attention_mask.dim() == 2:
            pad_mask = attention_mask[:, None, None, :target_length].bool()
            causal_bool = causal_bool & pad_mask

        return causal_bool  # bool → FlashAttention 유지

    # Bind to text_model
    text_model._update_causal_mask = types.MethodType(patched_fn, text_model)
    logger.info("  [Exp-E] _update_causal_mask 패치 설치 완료 (BoolMask + None decode)")
    return original_bound, True


def uninstall_bool_mask_patch(model: Any, original_bound: Any) -> None:
    """원본 bound method 복원."""
    if original_bound is None:
        return
    text_model = model.vlm.model
    text_model._update_causal_mask = original_bound
    logger.info("  [Exp-E] _update_causal_mask 원본 복원")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decode loop (메모리 할당 추적 포함)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_decode_loop(
    model: Any,
    cache: Any,
    first_logits: torch.Tensor,
    prefill_len: int,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
    requires_cache_position: bool = True,
) -> dict:
    """
    Decode loop with per-step memory allocation tracking.

    Returns dict with timing and allocation stats.
    """
    lgts = first_logits.clone()
    lgts[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
    cur = top_p_sample_gpu(lgts).unsqueeze(1)

    # EOS 검사는 CPU sync 없이 최대한 제거하여 캐시 성능 측정에 집중
    step_times_ms: list[float] = []
    alloc_counts: list[int] = []

    # 첫 step 전 stats
    torch.cuda.synchronize()

    exited_at = MAX_DECODE_STEPS
    for step in range(MAX_DECODE_STEPS):
        alloc_before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)

        sw = CudaStopwatch()
        sw.start()

        call_kwargs: dict = dict(
            input_ids=cur,
            past_key_values=cache,
            use_cache=True,
        )
        if requires_cache_position:
            call_kwargs["cache_position"] = torch.tensor(
                [prefill_len + step], device=DEVICE, dtype=torch.long
            )

        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.vlm(**call_kwargs)
        except Exception as e:
            logger.error(f"  Step {step} 실패: {type(e).__name__}: {e}")
            traceback.print_exc()
            return {"error": str(e), "steps": step}

        step_ms = sw.stop_ms()
        alloc_after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)

        step_times_ms.append(step_ms)
        alloc_counts.append(alloc_after - alloc_before)

        lgts = out.logits[:, -1, :].float()
        lgts[:, traj_offset : traj_offset + traj_vocab_size] = float("-inf")
        cur = top_p_sample_gpu(lgts).unsqueeze(1)

        # EOS 체크 (CPU sync 최소화: 4 step 마다)
        if step % 4 == 3 and cur.item() == eos_id:
            exited_at = step + 1
            break

    n_steps = len(step_times_ms)
    if n_steps == 0:
        return {"error": "no steps completed"}

    arr = np.array(step_times_ms)
    return {
        "steps":            n_steps,
        "exited_at":        exited_at,
        "total_decode_ms":  float(arr.sum()),
        "mean_ms_per_step": float(arr.mean()),
        "median_ms_per_step": float(np.median(arr)),
        "min_ms_per_step":  float(arr.min()),
        "max_ms_per_step":  float(arr.max()),
        "step_times_ms":    arr.tolist(),
        "total_new_allocs": int(sum(alloc_counts)),
        "mean_allocs_per_step": float(np.mean(alloc_counts)),
        "alloc_counts":     alloc_counts,
        # BW 추정: 22 GB 모델 weights/step
        "estimated_bw_GBps": round(22.157 / (arr.mean() / 1000), 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Single run: prefill + decode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_once(
    model: Any,
    input_ids: torch.Tensor,
    tok_data: dict,
    prefill_len: int,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
    cache: Any,
    cache_name: str,
    requires_cache_position: bool = True,
) -> dict | None:

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ── Prefill ─────────────────────────────────────────────────────────
    prefill_kwargs: dict = dict(
        input_ids=input_ids,
        attention_mask=tok_data.get("attention_mask"),
        pixel_values=tok_data.get("pixel_values"),
        image_grid_thw=tok_data.get("image_grid_thw"),
        past_key_values=cache,
        use_cache=True,
    )
    if requires_cache_position:
        prefill_kwargs["cache_position"] = torch.arange(
            prefill_len, device=DEVICE, dtype=torch.long
        )

    sw = CudaStopwatch()
    torch.cuda.synchronize()
    sw.start()

    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(**prefill_kwargs)
        prefill_ms = sw.stop_ms()
    except Exception as e:
        logger.error(f"  [{cache_name}] prefill 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    first_logits = out.logits[:, -1, :].float()

    # ── Decode ──────────────────────────────────────────────────────────
    decode_stats = run_decode_loop(
        model=model,
        cache=cache,
        first_logits=first_logits,
        prefill_len=prefill_len,
        eos_id=eos_id,
        traj_offset=traj_offset,
        traj_vocab_size=traj_vocab_size,
        requires_cache_position=requires_cache_position,
    )

    if "error" in decode_stats:
        logger.error(f"  [{cache_name}] decode 실패: {decode_stats['error']}")
        return None

    return {
        "prefill_ms": round(prefill_ms, 1),
        **decode_stats,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run experiment: N runs with given cache factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(
    exp_name: str,
    model: Any,
    input_ids: torch.Tensor,
    tok_data: dict,
    prefill_len: int,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
    cache_factory,           # callable() → new cache instance
    requires_cache_position: bool = True,
    n_warmup: int = NUM_WARMUP,
    n_measure: int = NUM_MEASURE,
) -> dict:

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  실험 [{exp_name}]")
    print(sep)

    warmup_results = []
    measure_results = []

    for i in range(n_warmup + n_measure):
        is_warmup = i < n_warmup
        tag = f"WARMUP {i+1}" if is_warmup else f"RUN {i - n_warmup + 1}"

        cache = cache_factory()

        result = run_once(
            model=model,
            input_ids=input_ids,
            tok_data=tok_data,
            prefill_len=prefill_len,
            eos_id=eos_id,
            traj_offset=traj_offset,
            traj_vocab_size=traj_vocab_size,
            cache=cache,
            cache_name=exp_name,
            requires_cache_position=requires_cache_position,
        )

        if result is None:
            print(f"  [{tag}] ❌ FAILED")
            continue

        steps    = result["steps"]
        ms_step  = result["mean_ms_per_step"]
        dec_ms   = result["total_decode_ms"]
        pre_ms   = result["prefill_ms"]
        allocs   = result["total_new_allocs"]
        bw       = result["estimated_bw_GBps"]

        tag_backend = _guess_backend(ms_step)
        print(
            f"  [{tag}]  prefill={pre_ms:6.0f}ms  "
            f"decode={dec_ms:6.0f}ms ({steps}steps × {ms_step:.1f}ms/step)  "
            f"BW={bw:.0f}GB/s  allocs={allocs}  [{tag_backend}]"
        )

        if is_warmup:
            warmup_results.append(result)
        else:
            measure_results.append(result)

    if not measure_results:
        print(f"  ❌ [{exp_name}] 측정 실패")
        return {"exp_name": exp_name, "success": False}

    # 집계
    def avg(key: str) -> float:
        vals = [r[key] for r in measure_results if isinstance(r.get(key), (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "exp_name":             exp_name,
        "success":              True,
        "n_runs":               len(measure_results),
        "prefill_ms":           round(avg("prefill_ms"), 1),
        "decode_ms":            round(avg("total_decode_ms"), 1),
        "steps":                round(avg("steps"), 1),
        "mean_ms_per_step":     round(avg("mean_ms_per_step"), 2),
        "median_ms_per_step":   round(avg("median_ms_per_step"), 2),
        "total_new_allocs":     round(avg("total_new_allocs"), 0),
        "mean_allocs_per_step": round(avg("mean_allocs_per_step"), 1),
        "estimated_bw_GBps":    round(avg("estimated_bw_GBps"), 1),
        "all_runs":             measure_results,
    }

    backend_guess = _guess_backend(summary["mean_ms_per_step"])
    print(f"\n  ★ [{exp_name}] 평균: {summary['mean_ms_per_step']:.1f}ms/step  "
          f"BW={summary['estimated_bw_GBps']:.0f}GB/s  "
          f"allocs={summary['total_new_allocs']:.0f}  [{backend_guess}]")

    return summary


def _guess_backend(ms_per_step: float) -> str:
    """시간 기반으로 SDPA 백엔드 추정."""
    if ms_per_step < 130:
        return "FlashAttn 추정"
    elif ms_per_step < 170:
        return "불명 (경계)"
    else:
        return "MemEfficient 추정"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    W = 70
    print("=" * W)
    print("  AppendOnlyCache 실험 — torch.cat 제거 + FlashAttention 유지 검증")
    print("=" * W)

    # ── 데이터 로드 ────────────────────────────────────────────────────
    logger.info("데이터 로드...")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    # ── 모델 로드 (sdpa 기본값) ────────────────────────────────────────
    logger.info("모델 로드 (sdpa 기본값)...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(DEVICE).eval()

    actual_attn = getattr(model.vlm.config, "_attn_implementation", "unknown")
    logger.info(f"  attn_implementation = {actual_attn}")

    # ── 토크나이즈 ─────────────────────────────────────────────────────
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = helper.to_device(inputs, DEVICE)

    ego_data = helper.to_device(
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        DEVICE,
    )
    input_ids_raw = inputs.pop("input_ids")
    input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
    tok_data = inputs

    prefill_len = int(input_ids.shape[1])
    logger.info(f"  prefill_len = {prefill_len} tokens")

    eos_id = model.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    traj_offset = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    text_config = model.vlm.config.text_config

    # AppendOnlyCache 차원 정보 출력
    n_layers   = text_config.num_hidden_layers
    n_kv_heads = text_config.num_key_value_heads
    n_q_heads  = text_config.num_attention_heads
    head_dim   = getattr(text_config, "head_dim", text_config.hidden_size // n_q_heads)
    logger.info(
        f"  VLM 텍스트 설정: layers={n_layers}, "
        f"kv_heads={n_kv_heads}, head_dim={head_dim}"
    )

    common_kwargs = dict(
        model=model,
        input_ids=input_ids,
        tok_data=tok_data,
        prefill_len=prefill_len,
        eos_id=eos_id,
        traj_offset=traj_offset,
        traj_vocab_size=traj_vocab_size,
    )

    all_summaries: dict[str, dict] = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Exp-A: DynamicCache baseline
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if RUN_EXP_A:
        def factory_a():
            return DynamicCache()

        all_summaries["A_dynamic"] = run_experiment(
            exp_name="A: DynamicCache (baseline)",
            cache_factory=factory_a,
            requires_cache_position=False,  # DynamicCache는 cache_position 불필요
            **common_kwargs,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Exp-B: AppendOnlyCache (non-contiguous view)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if RUN_EXP_B:
        def factory_b():
            return build_appendonly_cache(
                text_config, prefill_len, MAX_DECODE_STEPS, DEVICE,
                torch.bfloat16, force_contiguous=False
            )

        all_summaries["B_appendonly"] = run_experiment(
            exp_name="B: AppendOnlyCache (non-contiguous view)",
            cache_factory=factory_b,
            requires_cache_position=False,  # DynamicCache 상속 → 불필요
            **common_kwargs,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Exp-C: AppendOnlyCache-C (force_contiguous=True)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if RUN_EXP_C:
        def factory_c():
            return build_appendonly_cache(
                text_config, prefill_len, MAX_DECODE_STEPS, DEVICE,
                torch.bfloat16, force_contiguous=True
            )

        all_summaries["C_appendonly_contiguous"] = run_experiment(
            exp_name="C: AppendOnlyCache-C (force_contiguous)",
            cache_factory=factory_c,
            requires_cache_position=False,
            **common_kwargs,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Exp-D: StaticCache reference (2× 느림 확인)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if RUN_EXP_D:
        def factory_d():
            return make_static_cache(
                text_config,
                max_cache_len=prefill_len + MAX_DECODE_STEPS + 8,
                device=DEVICE,
                dtype=torch.bfloat16,
            )

        all_summaries["D_static"] = run_experiment(
            exp_name="D: StaticCache (reference, MemEfficient 확인)",
            cache_factory=factory_d,
            requires_cache_position=True,  # StaticCache 는 cache_position 필요
            n_warmup=1,
            n_measure=2,   # 시간 절약 (느리므로)
            **common_kwargs,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Exp-E: BoolMaskStaticCache (monkey-patch _update_causal_mask)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if RUN_EXP_E:
        original_bound, patched_ok = install_bool_mask_patch(model)

        if patched_ok:
            def factory_e():
                return make_static_cache(
                    text_config,
                    max_cache_len=prefill_len + MAX_DECODE_STEPS + 8,
                    device=DEVICE,
                    dtype=torch.bfloat16,
                )

            try:
                all_summaries["E_boolmask_static"] = run_experiment(
                    exp_name="E: BoolMaskStaticCache (patched _update_causal_mask)",
                    cache_factory=factory_e,
                    requires_cache_position=True,
                    n_warmup=1,
                    n_measure=2,
                    **common_kwargs,
                )
            except Exception as ex:
                logger.error(f"  [Exp-E] 실험 실패: {ex}")
                all_summaries["E_boolmask_static"] = {"exp_name": "E", "success": False, "error": str(ex)}

            # 패치 반드시 복원
            uninstall_bool_mask_patch(model, original_bound)
        else:
            logger.warning("  [Exp-E] 패치 실패로 건너뜀")
            all_summaries["E_boolmask_static"] = {
                "exp_name": "E: BoolMaskStaticCache",
                "success": False,
                "error": "_update_causal_mask not found",
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 비교 요약 출력
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'═' * W}")
    print("  ★ 최종 비교 요약")
    print(f"{'═' * W}")

    baseline_ms = None
    rows = []
    for key, s in all_summaries.items():
        if not s.get("success"):
            rows.append((key, "FAIL", "--", "--", "--", "--"))
            continue
        ms   = s["mean_ms_per_step"]
        bw   = s["estimated_bw_GBps"]
        alc  = s["total_new_allocs"]
        bkg  = _guess_backend(ms)

        if key == "A_dynamic":
            baseline_ms = ms

        if baseline_ms and key != "A_dynamic":
            chg = (ms - baseline_ms) / baseline_ms * 100
            chg_str = f"{chg:+.1f}%"
        else:
            chg_str = "(기준)"

        rows.append((s["exp_name"], f"{ms:.1f}ms", f"{bw:.0f}GB/s",
                     f"{alc:.0f}", bkg, chg_str))

    name_w = 45
    print(f"  {'실험':<{name_w}} {'ms/step':>8} {'BW':>8} {'allocs':>7} {'backend':>18} {'vs A':>7}")
    print(f"  {'-'*name_w} {'-'*8} {'-'*8} {'-'*7} {'-'*18} {'-'*7}")
    for r in rows:
        print(f"  {r[0]:<{name_w}} {r[1]:>8} {r[2]:>8} {r[3]:>7} {r[4]:>18} {r[5]:>7}")

    # 핵심 결론
    print(f"\n{'─' * W}")
    if "B_appendonly" in all_summaries and all_summaries["B_appendonly"].get("success"):
        b_ms = all_summaries["B_appendonly"]["mean_ms_per_step"]
        b_bkg = _guess_backend(b_ms)
        if baseline_ms:
            diff = baseline_ms - b_ms
            print(f"  ■ AppendOnlyCache (B) vs DynamicCache (A):")
            print(f"    {b_ms:.1f}ms vs {baseline_ms:.1f}ms → 차이: {diff:+.1f}ms/step")
            if diff > 5:
                print(f"    → ✅ AppendOnlyCache 가 DynamicCache 대비 {diff:.0f}ms/step 빠름")
            elif diff > -5:
                print(f"    → ≈ 동등 (torch.cat 제거 효과 ≤ 측정 오차)")
            else:
                print(f"    → ⚠ AppendOnlyCache 가 오히려 느림 ({abs(diff):.0f}ms/step)")
                print(f"       원인: non-contiguous view 에서 .contiguous() 호출 가능성")
        print(f"    백엔드 추정: [{b_bkg}]")

    if "C_appendonly_contiguous" in all_summaries and all_summaries["C_appendonly_contiguous"].get("success"):
        if all_summaries["B_appendonly"].get("success"):
            b_ms = all_summaries["B_appendonly"]["mean_ms_per_step"]
            c_ms = all_summaries["C_appendonly_contiguous"]["mean_ms_per_step"]
            diff_bc = c_ms - b_ms
            print(f"\n  ■ B (non-contiguous) vs C (force_contiguous): {diff_bc:+.1f}ms/step")
            if diff_bc > 3:
                print(f"    → ⚠ .contiguous() 비용 {diff_bc:.1f}ms/step (C가 느림)")
                print(f"       non-contiguous view 를 사용해야 함 (B 방식)")
            elif diff_bc < -3:
                print(f"    → SDPA 내부에서 이미 .contiguous() 호출됨 ({abs(diff_bc):.1f}ms/step)")
                print(f"       B 의 non-contiguous 는 숨겨진 비용 존재")
            else:
                print(f"    → non-contiguous vs contiguous 차이 미미 (≤3ms)")

    print(f"{'═' * W}")

    # 저장
    output = {
        "attn_implementation": actual_attn,
        "prefill_len": prefill_len,
        "model_config": {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
        },
        "baseline_dynamic_ms_per_step": BASELINE_DYNAMIC_MS_PER_STEP,
        "results": all_summaries,
    }
    out_path = OUT / "results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"결과 저장: {out_path}")
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    main()
