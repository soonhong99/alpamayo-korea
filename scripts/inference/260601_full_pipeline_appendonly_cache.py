"""
260601_full_pipeline_appendonly_cache.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[목적]
  260531_appendonly_cache_exp.py 에서 확인된 79ms/step 성능이
  실제 vlm.generate() 생산 경로에서도 재현되는지 검증.

[260531 실험과의 차이]
  이전 실험: vlm_model() 직접 forward() 호출 (bypass 경로)
  이 실험:   vlm.generate() + sample_trajectories_from_data_with_vlm_rollout()
             ← 실제 StopAfterEOS, ExpertLogitsProcessor, Flow 단계 포함

[AppendOnlyCache 추가 구현 사항]
  기존 구현 대비 두 가지 신규 메서드 추가:

  1. crop(max_length)
     Expert (Flow 단계)는 매 ODE step마다 64개 임시 토큰을 cache에 append한 뒤
     prompt_cache.crop(prefill_seq_len) 으로 제거한다.
     DynamicCache.crop()은 key_cache[] slice만 자르고 _write_pos를 갱신하지 않으므로
     → AppendOnlyCache의 _write_pos도 함께 수정해야 다음 ODE step에서 올바른 위치에 쓴다.

  2. batch_repeat_interleave(repeats)
     generate()에서 num_return_sequences > 1이면 transformers 내부적으로
     _expand_inputs_for_generation() → cache.batch_repeat_interleave(repeats) 호출.
     AppendOnlyCache는 _k_buf/_v_buf가 pre-alloc이므로 이 버퍼도 함께 확장해야 함.
     기존 DynamicCache.batch_repeat_interleave()는 key_cache[]만 확장 → _k_buf 불일치 발생.

[측정 구성]
  Phase A: DynamicCache (기준선)
    - generate() 기본 캐시, 수정 없음
    - num_return_sequences = NUM_TRAJ_SAMPLES (생산 설정)

  Phase B: AppendOnlyCache-C (force_contiguous=True)
    - vlm.generate() monkey-patch로 캐시 주입
    - num_return_sequences = NUM_TRAJ_SAMPLES (생산 설정)

  Phase B1: AppendOnlyCache-C, num_return_sequences=1 (단순 검증용)
    - batch_repeat_interleave 미발생 → 결과 해석 단순

[측정 지표]
  - 전체 파이프라인 시간: generate() + Flow (CUDA 이벤트 기반)
  - generate() 내 세부: prefill 추정 + decode 단계 수 / 평균 ms/step
  - trajectories 벡터 유사도: AppendOnly 결과가 DynamicCache 결과와 얼마나 같은가

[실행]
  source ~/alpamayo1.5/a1_5_venv/bin/activate && cd ~/alpamayo1.5
  python3 scripts/inference/260601_full_pipeline_appendonly_cache.py

[결과]
  profiling_results/260601_full_pipeline/results.json
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

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
NUM_TRAJ_SAMPLES = 6    # 생산 설정 (alpamayo1_5.py 기본값)
NUM_WARMUP       = 3    # JIT kernel compile 완료까지 3회 필요 (Phase A 분산 억제)
NUM_MEASURE      = 3

# AppendOnlyCache max_decode 여유: generate()의 max_new_tokens 이상으로 설정
# model.config.tokens_per_future_traj = 64, 여유 16
MAX_DECODE_MARGIN = 80

OUT = Path("profiling_results/260601_full_pipeline")
OUT.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUDA 타이머
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CudaTimer:
    """CUDA 이벤트 기반 타이머. CPU-GPU 동기화 없이 정확한 GPU 시간 측정."""

    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()

    def stop_ms(self) -> float:
        self._end.record()
        torch.cuda.synchronize()
        return self._start.elapsed_time(self._end)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AppendOnlyCache (generate() 경로 완전 대응)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AppendOnlyCache(DynamicCache):
    """
    DynamicCache 상속 + torch.cat 제거 + L2 재사용 최적화.

    generate() 경로에서 추가로 필요한 두 메서드를 오버라이드:

    [crop(max_length)]
      Expert Flow 단계:
        for ode_step in range(10):
            expert.forward(..., past_key_values=prompt_cache)
            prompt_cache.crop(prefill_seq_len)   ← 64 임시 토큰 제거
      DynamicCache.crop()는 key_cache[] 슬라이스만 자름.
      AppendOnlyCache는 _write_pos 도 함께 갱신해야 다음 ODE step에서
      올바른 버퍼 위치에 쓸 수 있음.

    [batch_repeat_interleave(repeats)]
      generate()에서 num_return_sequences > 1이면:
        _expand_inputs_for_generation() → cache.batch_repeat_interleave(repeats)
      DynamicCache는 key_cache[] (view)만 확장.
      AppendOnlyCache는 _k_buf/_v_buf (실제 저장소)도 확장해야 함.
      확장 후 update()에서 _k_buf에 정상적으로 쓰기 위해서.
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
        # ★ 현재 Thor transformers: DynamicCache.__init__()이 _seen_tokens를 초기화하지 않음
        #   → 명시적으로 초기화 필수 (AttributeError 방지)
        self._seen_tokens: int = 0
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self._batch_size = batch_size
        self.force_contiguous = force_contiguous

        # 쓰기 위치 포인터 (레이어별)
        self._write_pos: list[int] = [0] * n_layers

        # Pre-allocated 버퍼 (DRAM에 한 번만 할당)
        self._k_buf: list[torch.Tensor] = [
            torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self._v_buf: list[torch.Tensor] = [
            torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(n_layers)
        ]

        # DynamicCache 호환: 빈 리스트로 시작 (update()에서 동적으로 확장)
        # ⚠️ [None]*n_layers 불가 — get_seq_length()가 None.shape[-2] 접근 시 crash
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    # ──────────────────────────────────────────────────────────────────
    # 핵심: in-place write + (선택적) contiguous copy
    # ──────────────────────────────────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-alloc 버퍼에 in-place 쓰기. torch.cat 없음.

        key_states: [B, H_kv, n_new, D]
        반환:       [B, H_kv, cumulative_len, D]  (view 또는 contiguous copy)
        """
        n_new = key_states.shape[2]
        pos   = self._write_pos[layer_idx]
        B     = key_states.shape[0]

        # 배치 크기 변화 감지 (batch_repeat_interleave 이후 B가 늘어날 수 있음)
        if B != self._k_buf[layer_idx].shape[0]:
            raise RuntimeError(
                f"[AppendOnlyCache] batch size 불일치: "
                f"key_states.B={B}, _k_buf.B={self._k_buf[layer_idx].shape[0]}. "
                f"batch_repeat_interleave()가 올바르게 호출됐는지 확인하세요."
            )

        # 버퍼 오버플로우 방지
        if pos + n_new > self.max_seq_len:
            raise RuntimeError(
                f"[AppendOnlyCache] overflow at layer {layer_idx}: "
                f"pos={pos}, n_new={n_new}, max_seq_len={self.max_seq_len}. "
                f"MAX_DECODE_MARGIN을 늘리거나 max_seq_len을 재확인하세요."
            )

        # ★ In-place write: 새 토큰만 기록, 기존 KV 복사 없음
        self._k_buf[layer_idx][:, :, pos:pos + n_new, :] = key_states
        self._v_buf[layer_idx][:, :, pos:pos + n_new, :] = value_states
        self._write_pos[layer_idx] += n_new

        # DynamicCache._seen_tokens 동기화 (일부 transformers 코드가 참조)
        # layer_idx==0 일 때만 갱신 (중복 카운팅 방지)
        if layer_idx == 0:
            self._seen_tokens += n_new

        cur_len = self._write_pos[layer_idx]

        # 유효 구간의 view 반환
        k_out = self._k_buf[layer_idx][:, :, :cur_len, :]
        v_out = self._v_buf[layer_idx][:, :, :cur_len, :]

        if self.force_contiguous:
            # Exp-C: non-contiguous view → contiguous copy → L2 재사용 효과
            k_out = k_out.contiguous()
            v_out = v_out.contiguous()

        # key_cache / value_cache 동기화
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)   # type: ignore[arg-type]
            self.value_cache.append(None) # type: ignore[arg-type]
        self.key_cache[layer_idx]   = k_out
        self.value_cache[layer_idx] = v_out

        return k_out, v_out

    # ──────────────────────────────────────────────────────────────────
    # DynamicCache interface
    # ──────────────────────────────────────────────────────────────────

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """실제 누적 토큰 수 반환 (DynamicCache 동작과 동일)."""
        if layer_idx >= self.n_layers:
            return 0
        return self._write_pos[layer_idx]

    def get_max_length(self) -> int | None:
        return self.max_seq_len

    # ──────────────────────────────────────────────────────────────────
    # [신규] crop() — Expert Flow 단계 필수
    # ──────────────────────────────────────────────────────────────────

    def crop(self, max_length: int) -> None:
        """
        cache를 max_length 토큰까지 잘라낸다.

        왜 오버라이드가 필요한가:
          DynamicCache.crop()은 key_cache[i] = key_cache[i][..., :max_length, :] 만 수행.
          AppendOnlyCache는 _write_pos도 함께 줄여야 함.
          그렇지 않으면 다음 update() 호출 시 pos = (잘린 후 길이가 아니라) 이전 값이 되어
          _k_buf의 잘못된 위치에 쓰게 됨.

        Expert Flow 호출 패턴:
          expert.forward(..., past_key_values=prompt_cache)  → 64 tokens append
          prompt_cache.crop(prefill_seq_len)                 ← 이 함수
          → 다음 ODE step에서 update() 호출 시 pos = prefill_seq_len 으로 복원
        """
        for i in range(self.n_layers):
            if self._write_pos[i] > max_length:
                self._write_pos[i] = max_length

        # key_cache / value_cache view 갱신
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None and i < self.n_layers:
                cur_len = self._write_pos[i]
                k_view = self._k_buf[i][:, :, :cur_len, :]
                v_view = self._v_buf[i][:, :, :cur_len, :]
                if self.force_contiguous:
                    self.key_cache[i]   = k_view.contiguous()
                    self.value_cache[i] = v_view.contiguous()
                else:
                    self.key_cache[i]   = k_view
                    self.value_cache[i] = v_view

        # _seen_tokens 동기화
        if self._seen_tokens > max_length:
            self._seen_tokens = max_length

    # ──────────────────────────────────────────────────────────────────
    # [신규] batch_repeat_interleave() — num_return_sequences > 1 필수
    # ──────────────────────────────────────────────────────────────────

    def batch_repeat_interleave(self, repeats: int) -> "AppendOnlyCache":
        """
        generate()의 num_return_sequences > 1 처리.

        ★ 호출 타이밍이 두 가지:
          Case A: prefill 이전 (key_cache 비어있음)
            → transformers가 _expand_inputs_for_generation()에서 input_ids를 먼저 확장.
              이 시점에는 key_cache가 비어있으므로 _k_buf 확장 금지.
              우리는 inject_appendonly_cache에서 num_return_sequences를 읽어
              처음부터 batch=N으로 pre-allocate 했으므로 no-op이 맞음.

          Case B: prefill 이후 (key_cache에 데이터 있음, 이론적 경로)
            → _k_buf, _v_buf 모두 확장.

        반환값: self (transformers가 past_key_values = cache.batch_repeat_interleave(n)
                로 재할당하는 경우 대비)
        """
        has_data = (
            len(self.key_cache) > 0
            and any(k is not None for k in self.key_cache)
        )

        if not has_data:
            # Case A: key_cache 비어있음 → pre-allocation이 이미 correct batch.
            # _k_buf를 다시 expand하면 batch가 N² 가 되어 버그 발생.
            logger.info(
                f"[AppendOnlyCache] batch_repeat_interleave({repeats}) called "
                f"before prefill (key_cache empty). Skipping. "
                f"pre-alloc batch={self._batch_size}"
            )
            return self

        # Case B: prefill 이후 확장 (현재 경로에서 발생하지 않을 가능성 높음)
        logger.info(
            f"[AppendOnlyCache] batch_repeat_interleave(repeats={repeats}) "
            f"post-prefill. _k_buf batch {self._batch_size} → {self._batch_size * repeats}"
        )

        cur_lens = list(self._write_pos)

        for i in range(self.n_layers):
            self._k_buf[i] = self._k_buf[i].repeat_interleave(repeats, dim=0)
            self._v_buf[i] = self._v_buf[i].repeat_interleave(repeats, dim=0)

        self._batch_size *= repeats

        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None and i < self.n_layers:
                cur_len = cur_lens[i]
                k_view = self._k_buf[i][:, :, :cur_len, :]
                v_view = self._v_buf[i][:, :, :cur_len, :]
                if self.force_contiguous:
                    self.key_cache[i]   = k_view.contiguous()
                    self.value_cache[i] = v_view.contiguous()
                else:
                    self.key_cache[i]   = k_view
                    self.value_cache[i] = v_view

        logger.info(
            f"[AppendOnlyCache] batch_repeat_interleave 완료. "
            f"_k_buf[0].shape={self._k_buf[0].shape}"
        )
        return self

    def reset(self) -> None:
        """다음 inference를 위해 포인터 초기화 (버퍼 재사용, zero-fill 불필요)."""
        for i in range(self.n_layers):
            self._write_pos[i] = 0
        self._seen_tokens = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AppendOnlyCache 빌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_appendonly_cache(
    text_config: Any,
    prefill_len: int,
    max_decode: int,
    device: str,
    dtype: torch.dtype,
    batch_size: int = 1,
    force_contiguous: bool = True,
) -> AppendOnlyCache:
    """
    model.vlm.config.text_config에서 차원을 자동 추출하여 AppendOnlyCache 생성.

    주의: max_seq_len = prefill_len + max_decode + 8 (안전 여유)
    batch_repeat_interleave(repeats=6) 호출 시 버퍼가 6× 확장되므로
    초기 할당은 batch_size=1로 해도 충분.
    """
    n_layers    = text_config.num_hidden_layers
    n_kv_heads  = text_config.num_key_value_heads
    n_q_heads   = text_config.num_attention_heads
    head_dim    = getattr(text_config, "head_dim", text_config.hidden_size // n_q_heads)
    max_seq_len = prefill_len + max_decode + 8

    alloc_mb = n_layers * 2 * batch_size * n_kv_heads * max_seq_len * head_dim * 2 / 1e6

    logger.info(
        f"  AppendOnlyCache: {n_layers}L × {n_kv_heads}H × {head_dim}D "
        f"max_seq={max_seq_len} B={batch_size} alloc≈{alloc_mb:.0f}MB "
        f"force_contiguous={force_contiguous}"
    )

    return AppendOnlyCache(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        force_contiguous=force_contiguous,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# generate() 주입 컨텍스트 매니저
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@contextlib.contextmanager
def inject_appendonly_cache(
    model: Any,
    text_config: Any,
    max_decode: int,
    force_contiguous: bool = True,
):
    """
    model.vlm.generate()를 monkey-patch하여 AppendOnlyCache-C를 주입.

    사용법:
        with inject_appendonly_cache(model, text_config, max_decode=80):
            result = model.sample_trajectories_from_data_with_vlm_rollout(data)

    동작:
      1. vlm.generate() 호출 시 input_ids.shape[1]로 prefill_len 자동 계산
      2. 매 호출마다 fresh AppendOnlyCache 생성 → past_key_values로 주입
      3. 컨텍스트 종료 시 원본 generate() 복원

    주의:
      - past_key_values가 이미 kwargs에 있으면 override하지 않음
        (generate()가 내부적으로 재호출할 경우 등을 방지)
      - transformers가 _prepare_cache_for_generation()으로 캐시를 교체할 경우
        (cache_implementation 설정 시) 주입이 무시될 수 있음.
        이 실험에서는 generation_config.cache_implementation이 설정되지 않으므로 안전.
    """
    original_generate = model.vlm.generate

    def patched_generate(*args, **kwargs):
        # input_ids 추출 (positional 또는 keyword)
        input_ids = None
        if args:
            input_ids = args[0]
        if input_ids is None:
            input_ids = kwargs.get("input_ids")

        if input_ids is not None and "past_key_values" not in kwargs:
            prefill_len = int(input_ids.shape[1])

            # ★ num_return_sequences 감지: transformers는 _expand_inputs_for_generation()에서
            #   prefill 전에 input_ids를 [1, T] → [N, T]로 먼저 확장한다.
            #   따라서 cache도 처음부터 batch=N으로 pre-allocate해야 함.
            #   generation_config는 sample_trajectories_from_data_with_vlm_rollout()에서
            #   kwargs["generation_config"]로 전달됨.
            num_return_seqs = 1
            gen_cfg = kwargs.get("generation_config", None)
            if gen_cfg is not None:
                num_return_seqs = max(1, getattr(gen_cfg, "num_return_sequences", 1) or 1)

            try:
                cache = build_appendonly_cache(
                    text_config=text_config,
                    prefill_len=prefill_len,
                    max_decode=max_decode,
                    device=str(input_ids.device),
                    dtype=torch.bfloat16,
                    batch_size=num_return_seqs,   # ★ N-배치 pre-allocation
                    force_contiguous=force_contiguous,
                )
                kwargs["past_key_values"] = cache
                logger.debug(
                    f"  [inject] AppendOnlyCache 주입: prefill_len={prefill_len}, "
                    f"max_decode={max_decode}, batch(num_return_seqs)={num_return_seqs}"
                )
            except Exception as e:
                logger.warning(
                    f"  [inject] AppendOnlyCache 생성 실패: {e}. "
                    f"기본 DynamicCache로 폴백."
                )

        return original_generate(*args, **kwargs)

    # Monkey-patch 설치
    model.vlm.generate = patched_generate
    try:
        yield
    finally:
        # 원본 복원
        model.vlm.generate = original_generate
        logger.debug("  [inject] vlm.generate() 원본 복원 완료")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 추론 실행 + 상세 타이밍
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_single_inference(
    model: Any,
    data: dict,
    num_traj_samples: int,
    label: str,
) -> dict:
    """
    sample_trajectories_from_data_with_vlm_rollout() 1회 실행.

    반환 dict:
      total_ms       - 전체 시간 (VE + Prefill + Decode + Flow)
      generate_ms    - vlm.generate() 시간 추정 (Wall clock)
      flow_ms        - Flow 단계 시간
      n_tokens       - 생성된 토큰 수 (전체)
      ms_per_token   - generate_ms / n_tokens (decode 당 ms 추정)
      pred_xyz       - [B, ns, nj, T, 3] numpy
      success        - bool
      error          - 에러 메시지 (실패 시)
    """
    result = {
        "label": label,
        "total_ms": None,
        "generate_ms": None,
        "flow_ms": None,
        "n_tokens": None,
        "ms_per_token": None,
        "pred_xyz": None,
        "success": False,
        "error": None,
    }

    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # ── generate() 타이밍 훅 ──────────────────────────────────────
        # vlm.generate() 호출 구간을 wall-clock으로 측정하기 위해
        # 한 번 더 래핑. 이 시점에 이미 inject_appendonly_cache가 설치된 상태.
        original_generate = model.vlm.generate
        generate_timing = {"start": None, "end": None, "n_tokens": 0}

        def timed_generate(*args, **kwargs):
            torch.cuda.synchronize()
            generate_timing["start"] = time.perf_counter()
            out = original_generate(*args, **kwargs)
            torch.cuda.synchronize()
            generate_timing["end"] = time.perf_counter()
            # 생성된 토큰 수 추출 (sequences 길이 - input 길이)
            # ★ alpamayo1_5.py line 304: vlm.generate(input_ids=input_ids, ...) → kwargs 전달
            #   args가 빈 tuple이면 and args → False → 버그. kwargs도 확인해야 함.
            if hasattr(out, "sequences"):
                input_ids_ref = args[0] if args else kwargs.get("input_ids")
                if input_ids_ref is not None:
                    # out.sequences: [B×num_return_seqs, prefill+decode]
                    # per-sequence new token count = total_len - prefill_len
                    generate_timing["n_tokens"] = (
                        int(out.sequences.shape[1]) - int(input_ids_ref.shape[1])
                    )
            return out

        model.vlm.generate = timed_generate

        # ── 전체 타이머 ──────────────────────────────────────────────
        total_timer = CudaTimer()
        flow_timer  = CudaTimer()

        # Flow 단계를 분리 측정하기 위해 diffusion.sample() 훅
        original_diffusion_sample = model.diffusion.sample
        flow_timing = {"start": None, "end": None}

        def timed_diffusion_sample(*args, **kwargs):
            torch.cuda.synchronize()
            flow_timing["start"] = time.perf_counter()
            out = original_diffusion_sample(*args, **kwargs)
            torch.cuda.synchronize()
            flow_timing["end"] = time.perf_counter()
            return out

        model.diffusion.sample = timed_diffusion_sample

        # ── 실제 추론 ────────────────────────────────────────────────
        total_timer.start()

        # ★ torch.autocast 필수: ego_history_xyz/rot (float32) vs Flow output (bf16)
        #   test_inference.py 참조: "with torch.autocast('cuda', dtype=torch.bfloat16):"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
                data=data,
                num_traj_samples=num_traj_samples,
                num_traj_sets=1,
            )

        total_ms = total_timer.stop_ms()

        # ── 훅 복원 ──────────────────────────────────────────────────
        model.vlm.generate   = original_generate
        model.diffusion.sample = original_diffusion_sample

        # ── 결과 수집 ─────────────────────────────────────────────────
        generate_ms = None
        if generate_timing["start"] and generate_timing["end"]:
            generate_ms = (generate_timing["end"] - generate_timing["start"]) * 1000

        flow_ms = None
        if flow_timing["start"] and flow_timing["end"]:
            flow_ms = (flow_timing["end"] - flow_timing["start"]) * 1000

        n_tokens = generate_timing["n_tokens"]
        ms_per_token = None
        if generate_ms and n_tokens and n_tokens > 1:
            # generate_ms 에는 prefill 도 포함됨.
            # 더 정확한 decode ms/step은 별도 prefill 측정 필요.
            # 여기서는 generate_ms / n_tokens 를 상한 추정치로 제공.
            ms_per_token = generate_ms / n_tokens

        result.update({
            "total_ms": round(total_ms, 2),
            "generate_ms": round(generate_ms, 2) if generate_ms else None,
            "flow_ms": round(flow_ms, 2) if flow_ms else None,
            "n_tokens": n_tokens,
            "ms_per_token": round(ms_per_token, 2) if ms_per_token else None,
            "pred_xyz": pred_xyz.cpu().numpy().tolist(),
            "success": True,
        })

    except Exception as e:
        result["error"] = traceback.format_exc()
        logger.error(f"  [오류] {label}: {e}")
        logger.debug(traceback.format_exc())
        # 훅 복원 (예외 발생 시에도)
        try:
            model.vlm.generate = original_generate
            model.diffusion.sample = original_diffusion_sample
        except Exception:
            pass

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 실행 (warmup + measure)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(
    name: str,
    model: Any,
    data: dict,
    text_config: Any,
    max_decode: int,
    num_traj_samples: int,
    use_appendonly: bool = False,
    force_contiguous: bool = True,
    n_warmup: int = NUM_WARMUP,
    n_measure: int = NUM_MEASURE,
) -> dict:
    """
    Warmup 후 n_measure회 측정. 결과 평균/표준편차 계산.

    use_appendonly=True:  inject_appendonly_cache() 컨텍스트 사용
    use_appendonly=False: 기본 DynamicCache (수정 없음)
    """
    W = 70
    logger.info("=" * W)
    logger.info(f"  실험: {name}")
    logger.info(f"  AppendOnly={use_appendonly}, force_contiguous={force_contiguous}")
    logger.info(f"  num_traj_samples={num_traj_samples}, warmup={n_warmup}, measure={n_measure}")
    logger.info("=" * W)

    all_results = []

    for run_idx in range(n_warmup + n_measure):
        is_warmup = run_idx < n_warmup
        label = f"{'[WARMUP]' if is_warmup else f'[RUN {run_idx - n_warmup + 1}]'} {name}"

        if use_appendonly:
            with inject_appendonly_cache(
                model, text_config, max_decode, force_contiguous
            ):
                r = run_single_inference(model, data, num_traj_samples, label)
        else:
            r = run_single_inference(model, data, num_traj_samples, label)

        status = "✓" if r["success"] else "✗"
        if r["success"]:
            logger.info(
                f"  {status} total={r['total_ms']:.0f}ms  "
                f"generate={r['generate_ms']}ms  "
                f"flow={r['flow_ms']}ms  "
                f"tokens={r['n_tokens']}  "
                f"ms/token≈{r['ms_per_token']}"
            )
        else:
            logger.error(f"  {status} 실패: {r['error'][:200]}")

        if not is_warmup:
            all_results.append(r)

    # 요약 통계
    successful = [r for r in all_results if r["success"]]

    def _avg(key: str) -> float | None:
        vals = [r[key] for r in successful if r[key] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def _std(key: str) -> float | None:
        vals = [r[key] for r in successful if r[key] is not None]
        if len(vals) < 2:
            return None
        mean = sum(vals) / len(vals)
        return round((sum((v - mean)**2 for v in vals) / (len(vals) - 1))**0.5, 2)

    summary = {
        "name": name,
        "use_appendonly": use_appendonly,
        "force_contiguous": force_contiguous,
        "num_traj_samples": num_traj_samples,
        "n_successful": len(successful),
        "n_total": len(all_results),
        "total_ms_mean": _avg("total_ms"),
        "total_ms_std": _std("total_ms"),
        "generate_ms_mean": _avg("generate_ms"),
        "generate_ms_std": _std("generate_ms"),
        "flow_ms_mean": _avg("flow_ms"),
        "flow_ms_std": _std("flow_ms"),
        "n_tokens_mean": _avg("n_tokens"),
        "ms_per_token_mean": _avg("ms_per_token"),
        "raw_results": all_results,
    }

    logger.info(f"\n  ── {name} 요약 ──")
    logger.info(f"  total:    {summary['total_ms_mean']} ± {summary['total_ms_std']} ms")
    logger.info(f"  generate: {summary['generate_ms_mean']} ± {summary['generate_ms_std']} ms")
    logger.info(f"  flow:     {summary['flow_ms_mean']} ± {summary['flow_ms_std']} ms")
    logger.info(f"  tokens:   {summary['n_tokens_mean']}, ms/token≈{summary['ms_per_token_mean']}")

    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 궤적 유사도 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_trajectory_similarity(
    result_baseline: dict,
    result_appendonly: dict,
) -> dict:
    """
    두 결과의 trajectory 평균 L2 거리를 계산.

    AppendOnlyCache가 올바르게 작동하면 DynamicCache와 동일한 분포에서
    샘플링해야 함 (같은 sampling seed 아니므로 완전 일치는 아님).

    반환:
      - mean_l2: 두 결과의 mean trajectory 간 L2 거리
      - max_deviation: 최대 편차
      - note: 해석 가이드
    """
    import numpy as np

    r = {"mean_l2": None, "max_deviation": None, "note": ""}

    try:
        baseline_raw = result_baseline.get("raw_results", [])
        appendonly_raw = result_appendonly.get("raw_results", [])

        # 성공한 마지막 run의 pred_xyz 비교
        b_xyz = next(
            (r["pred_xyz"] for r in reversed(baseline_raw) if r["success"] and r["pred_xyz"]),
            None
        )
        a_xyz = next(
            (r["pred_xyz"] for r in reversed(appendonly_raw) if r["success"] and r["pred_xyz"]),
            None
        )

        if b_xyz is None or a_xyz is None:
            r["note"] = "trajectory 데이터 없음 (실패한 run)"
            return r

        b_arr = np.array(b_xyz)  # [B, ns, nj, T, 3]
        a_arr = np.array(a_xyz)

        # 각 결과의 mean trajectory (num_traj_samples 평균)
        b_mean = b_arr.mean(axis=2)  # [B, ns, T, 3]
        a_mean = a_arr.mean(axis=2)

        if b_mean.shape != a_mean.shape:
            r["note"] = f"shape 불일치: baseline={b_mean.shape}, appendonly={a_mean.shape}"
            return r

        diff = b_mean - a_mean  # [B, ns, T, 3]
        l2_per_step = np.linalg.norm(diff, axis=-1)  # [B, ns, T]
        mean_l2 = float(l2_per_step.mean())
        max_dev = float(l2_per_step.max())

        r["mean_l2"] = round(mean_l2, 4)
        r["max_deviation"] = round(max_dev, 4)
        r["note"] = (
            "주의: sampling이 stochastic이므로 mean_l2 > 0 은 정상. "
            "큰 편차(> 10m)는 cache 동작 이상일 가능성."
        )

    except Exception as e:
        r["note"] = f"계산 실패: {e}"

    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    W = 70
    logger.info("=" * W)
    logger.info("  Full Pipeline: AppendOnlyCache-C → vlm.generate() 통합 검증")
    logger.info(f"  device={DEVICE}  num_traj_samples={NUM_TRAJ_SAMPLES}")
    logger.info(f"  warmup={NUM_WARMUP}  measure={NUM_MEASURE}")
    logger.info("=" * W)

    # ── 1. 모델 로드 ───────────────────────────────────────────────────
    logger.info("\n[1] 모델 로드 중 (nvidia/Alpamayo-1.5-10B)...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    logger.info("  모델 로드 완료")

    # ── 2. 입력 준비 ──────────────────────────────────────────────────
    logger.info(f"\n[2] 데이터 로드 (clip={CLIP_ID}, t0={T0_US}us)...")
    raw_data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=raw_data["image_frames"].flatten(0, 1),
        camera_indices=raw_data["camera_indices"],
    )
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    # to_device로 tokenized_data를 GPU로 이동 (inputs는 CPU BatchEncoding)
    # to_device()는 새 dict를 반환하며 inputs 원본은 수정되지 않음
    inputs_gpu = helper.to_device(inputs, DEVICE)

    # sample_trajectories_from_data_with_vlm_rollout() 가 요구하는 data 구조 구성
    # ★ load_physical_aiavdataset() 이 이미 [1, 1, T, 3] / [1, 1, T, 3, 3] 로 반환함
    #   → unsqueeze(1) 금지! 추가하면 [1, 1, 1, T, 3] (5D) 가 되어
    #     alpamayo1_5.py:247 의 "B, n_traj_group, _, _ = ego_history_xyz.shape" 에서
    #     "too many values to unpack (expected 4)" 오류 발생
    data = {
        "tokenized_data": inputs_gpu,                               # GPU tensors ✓
        "ego_history_xyz": raw_data["ego_history_xyz"].to(DEVICE),  # [1, 1, T, 3]  ✓
        "ego_history_rot": raw_data["ego_history_rot"].to(DEVICE),  # [1, 1, T, 3, 3] ✓
    }
    # pixel_values, image_grid_thw 등은 tokenized_data에 이미 포함됨
    # attention_mask도 포함됨

    input_ids_raw = inputs["input_ids"]
    # fuse_traj_tokens 는 내부에서 호출되므로 여기서는 raw token 길이만 확인
    prefill_len_approx = int(input_ids_raw.shape[1])
    logger.info(f"  input_ids: {input_ids_raw.shape}  (prefill ≈ {prefill_len_approx} tokens)")

    text_config = model.vlm.config.text_config
    max_decode = model.config.tokens_per_future_traj + MAX_DECODE_MARGIN

    logger.info(f"  text_config: {text_config.num_hidden_layers}L "
                f"× {text_config.num_key_value_heads}KV-heads "
                f"× {getattr(text_config, 'head_dim', text_config.hidden_size // text_config.num_attention_heads)}D")
    logger.info(f"  max_decode (margin 포함): {max_decode}")

    all_summaries: dict[str, dict] = {}

    # ── 3. Phase A0: DynamicCache N=1 (B1 공정 비교용 기준선) ───────────
    # JIT를 먼저 워밍업하는 효과도 있어 후속 Phase A(N=6)의 분산을 줄임
    logger.info("\n" + "─" * W)
    logger.info("[3] Phase A0: DynamicCache (N=1, B1 공정 비교 기준선 + JIT warmup)")
    logger.info("─" * W)

    summary_a0 = run_experiment(
        name="A0_DynamicCache_nsamples1",
        model=model,
        data=data,
        text_config=text_config,
        max_decode=max_decode,
        num_traj_samples=1,
        use_appendonly=False,
        n_warmup=NUM_WARMUP,
        n_measure=NUM_MEASURE,
    )
    all_summaries["A0_DynamicCache_nsamples1"] = summary_a0

    # ── 4. Phase A: DynamicCache N=6 baseline ─────────────────────────
    logger.info("\n" + "─" * W)
    logger.info("[4] Phase A: DynamicCache (N=6, 생산 기준선)")
    logger.info("─" * W)

    summary_a = run_experiment(
        name="A_DynamicCache_nsamples6",
        model=model,
        data=data,
        text_config=text_config,
        max_decode=max_decode,
        num_traj_samples=NUM_TRAJ_SAMPLES,
        use_appendonly=False,
        n_warmup=NUM_WARMUP,
        n_measure=NUM_MEASURE,
    )
    all_summaries["A_DynamicCache_nsamples6"] = summary_a

    # ── 5. Phase B1: AppendOnlyCache-C N=1 ────────────────────────────
    logger.info("\n" + "─" * W)
    logger.info("[5] Phase B1: AppendOnlyCache-C (N=1, A0 대비 개선 측정)")
    logger.info("─" * W)

    summary_b1 = run_experiment(
        name="B1_AppendOnlyCache_C_nsamples1",
        model=model,
        data=data,
        text_config=text_config,
        max_decode=max_decode,
        num_traj_samples=1,
        use_appendonly=True,
        force_contiguous=True,
        n_warmup=NUM_WARMUP,
        n_measure=NUM_MEASURE,
    )
    all_summaries["B1_AppendOnly_nsamples1"] = summary_b1

    # ── 6. Phase B2: AppendOnlyCache-C N=6 (생산 설정) ─────────────────
    logger.info("\n" + "─" * W)
    logger.info("[6] Phase B2: AppendOnlyCache-C (N=6, 생산 설정, A 대비 개선 측정)")
    logger.info("─" * W)

    summary_b2 = run_experiment(
        name="B2_AppendOnlyCache_C_nsamples6",
        model=model,
        data=data,
        text_config=text_config,
        max_decode=max_decode,
        num_traj_samples=NUM_TRAJ_SAMPLES,  # = 6
        use_appendonly=True,
        force_contiguous=True,
        n_warmup=NUM_WARMUP,
        n_measure=NUM_MEASURE,
    )
    all_summaries["B2_AppendOnly_nsamples6"] = summary_b2

    # ── 7. 비교 분석 ──────────────────────────────────────────────────
    logger.info("\n" + "=" * W)
    logger.info("  비교 결과")
    logger.info("=" * W)

    # ① A0 vs B1 (N=1 공정 비교)
    a0_total = summary_a0.get("total_ms_mean")
    b1_total = summary_b1.get("total_ms_mean")
    improvement_n1 = None
    if a0_total and b1_total:
        improvement_n1 = round((a0_total - b1_total) / a0_total * 100, 1)

    # ② A vs B2 (N=6 공정 비교, 생산 설정)
    traj_sim = compute_trajectory_similarity(summary_a, summary_b2)
    a_total  = summary_a.get("total_ms_mean")
    b2_total = summary_b2.get("total_ms_mean")
    improvement_n6 = None
    if a_total and b2_total:
        improvement_n6 = round((a_total - b2_total) / a_total * 100, 1)

    comparison = {
        # N=1 비교
        "A0_DynamicCache_N1_total_ms":      a0_total,
        "B1_AppendOnly_N1_total_ms":        b1_total,
        "improvement_N1_pct":               improvement_n1,
        "A0_generate_ms":  summary_a0.get("generate_ms_mean"),
        "B1_generate_ms":  summary_b1.get("generate_ms_mean"),
        "A0_flow_ms":      summary_a0.get("flow_ms_mean"),
        "B1_flow_ms":      summary_b1.get("flow_ms_mean"),
        "A0_ms_per_token": summary_a0.get("ms_per_token_mean"),
        "B1_ms_per_token": summary_b1.get("ms_per_token_mean"),
        # N=6 비교
        "A_DynamicCache_N6_total_ms":       a_total,
        "B2_AppendOnly_N6_total_ms":        b2_total,
        "improvement_N6_pct":               improvement_n6,
        "A_generate_ms":   summary_a.get("generate_ms_mean"),
        "B2_generate_ms":  summary_b2.get("generate_ms_mean"),
        "A_flow_ms":       summary_a.get("flow_ms_mean"),
        "B2_flow_ms":      summary_b2.get("flow_ms_mean"),
        "A_ms_per_token":  summary_a.get("ms_per_token_mean"),
        "B2_ms_per_token": summary_b2.get("ms_per_token_mean"),
        "trajectory_similarity_N6": traj_sim,
    }

    logger.info(f"  ── N=1 비교 (A0 DynamicCache vs B1 AppendOnly-C) ──")
    logger.info(f"    total:    {a0_total} ms  →  {b1_total} ms  ({improvement_n1}%)")
    logger.info(f"    generate: {summary_a0.get('generate_ms_mean')} ms  →  {summary_b1.get('generate_ms_mean')} ms")
    logger.info(f"    flow:     {summary_a0.get('flow_ms_mean')} ms  →  {summary_b1.get('flow_ms_mean')} ms")
    logger.info(f"    ms/token: {summary_a0.get('ms_per_token_mean')} ms  →  {summary_b1.get('ms_per_token_mean')} ms")
    logger.info(f"")
    logger.info(f"  ── N=6 비교 (A DynamicCache vs B2 AppendOnly-C) ──")
    logger.info(f"    total:    {a_total} ms  →  {b2_total} ms  ({improvement_n6}%)")
    logger.info(f"    generate: {summary_a.get('generate_ms_mean')} ms  →  {summary_b2.get('generate_ms_mean')} ms")
    logger.info(f"    flow:     {summary_a.get('flow_ms_mean')} ms  →  {summary_b2.get('flow_ms_mean')} ms")
    logger.info(f"    ms/token: {summary_a.get('ms_per_token_mean')} ms  →  {summary_b2.get('ms_per_token_mean')} ms")
    logger.info(f"  Trajectory 유사도 (mean L2): {traj_sim['mean_l2']} m")
    logger.info(f"    {traj_sim['note']}")

    # ── 7. 결과 저장 ──────────────────────────────────────────────────
    output = {
        "config": {
            "clip_id": CLIP_ID,
            "t0_us": T0_US,
            "num_traj_samples_production": NUM_TRAJ_SAMPLES,
            "num_warmup": NUM_WARMUP,
            "num_measure": NUM_MEASURE,
            "max_decode_margin": MAX_DECODE_MARGIN,
            "model": "nvidia/Alpamayo-1.5-10B",
            "text_config": {
                "num_hidden_layers": text_config.num_hidden_layers,
                "num_key_value_heads": text_config.num_key_value_heads,
                "head_dim": getattr(
                    text_config, "head_dim",
                    text_config.hidden_size // text_config.num_attention_heads
                ),
            },
        },
        "summaries": {
            k: {kk: vv for kk, vv in v.items() if kk != "raw_results"}
            for k, v in all_summaries.items()
        },
        "comparison": comparison,
    }

    out_path = OUT / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n  결과 저장: {out_path}")

    # ── 8. 최종 결론 ─────────────────────────────────────────────────
    logger.info("\n" + "=" * W)
    logger.info("  최종 결론")
    logger.info("=" * W)

    b1_success = summary_b1.get("n_successful", 0) > 0
    b2_success = summary_b2.get("n_successful", 0) > 0

    if b1_success and improvement_n1 is not None:
        logger.info(f"  [N=1]  A0 {a0_total:.0f}ms → B1 {b1_total:.0f}ms  ({improvement_n1:+.1f}%)")
    if b2_success and improvement_n6 is not None:
        logger.info(f"  [N=6]  A  {a_total:.0f}ms → B2 {b2_total:.0f}ms  ({improvement_n6:+.1f}%)")

    if b2_success:
        logger.info(f"  ✅ AppendOnlyCache-C generate() 통합 성공!")
        if improvement_n6 is not None:
            if improvement_n6 > 3:
                logger.info(f"     생산 파이프라인(N=6) {improvement_n6:.1f}% 개선.")
            elif improvement_n6 > 0:
                logger.info(f"     생산 파이프라인(N=6) {improvement_n6:.1f}% 개선 (미미).")
                logger.info("     Prefill이 총 시간의 ~70% → Amdahl 법칙으로 전체 개선 제한.")
            else:
                logger.warning(f"  ⚠ 개선 없음({improvement_n6:.1f}%). 캐시 처리 재확인.")
    else:
        logger.error("  ✗ B2(N=6) 실패.")

    if b1_success and not b2_success:
        logger.info("  ℹ B1(N=1) 성공 → batch_repeat_interleave() 또는 N=6 특이 경로 문제.")


if __name__ == "__main__":
    main()
