"""
260527_kv_cache_bag.py — KV Cache 가방 실험 (GPU-CPU Sync 제거)
==============================================================
동기:
  Nsight 프로파일에서 VLM decode의 매 step 사이에 cudaStreamSynchronize가
  발생하는 것을 관찰했다.  HuggingFace generate()의 StopAfterEOS 기준이
  .all() → Python bool 변환 시 GPU→CPU sync를 유발하기 때문.
  65 decode step = 65번 sync.

  KV Cache 가방 아이디어:
    1) StaticCache: max_decode 만큼 KV cache 사전 할당 → shape 고정
    2) per-step sync 제거: EOS를 GPU에 보관, 마지막에 1회만 CPU 읽기
    3) CUDAGraph 캡처: 단일 decode step 그래프화 → kernel launch 오버헤드 0

실험 구성:
  Phase 0. 베이스라인 + sync 오버헤드 측정
    - StopAfterEOS를 계측해 sync 횟수/시간 확인
    - output_logits=True (baseline 그대로)

  Phase 1. 무료 최적화: output_logits=False
    - logits는 어디서도 사용 안 됨 → 저장 제거

  Phase 2. 고정 스텝 (No-Sync StopCriteria)
    - StopAfterEOS 제거, max_new_tokens=측정된 스텝+5 로 고정
    - per-step sync 완전 제거
    - 마지막에 한 번 EOS 위치 탐색

  Phase 3. StaticCache 실험
    - HuggingFace StaticCache 사용 (KV 공간 사전 할당)
    - Phase 2와 조합

  Phase 4. Full Custom Decode Loop
    - generate() 우회 → 직접 vlm.forward() 호출
    - GPU-resident top-p 샘플링 (sync 없음)
    - 단 1회 CPU sync
    - Action Expert 호환성 검증

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5
  python3 scripts/inference/260527_kv_cache_bag.py

결과:
  profiling_results/260527_kv_cache_bag/results.json
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5, ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
DEVICE = "cuda"
NUM_WARMUP = 2
NUM_MEASURE = 5
NUM_TRAJ_SAMPLES = 1  # KV Cache bag 실험은 1 샘플로 집중
TOP_P = 0.98
TEMPERATURE = 0.6
MAX_NEW_TOKENS_BASELINE = 256  # 현재 baseline 설정

OUT = Path("profiling_results/260527_kv_cache_bag")
OUT.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# Phase 0: 계측 도구
# ──────────────────────────────────────────────────────────────────

class InstrumentedStopAfterEOS(StoppingCriteria):
    """
    StopAfterEOS와 동일한 동작 + 타이밍 계측.
    각 __call__()마다 wall-clock 시간을 측정한다.
    .all() 호출이 GPU→CPU sync를 유발하는지 확인.
    """

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self.eos_found: torch.Tensor | None = None
        # 계측 데이터
        self.call_count = 0
        self.step_times_ms: list[float] = []   # 각 step의 wall-clock time
        self.total_overhead_ms = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        t0 = time.perf_counter()

        batch_size = input_ids.shape[0]
        if self.eos_found is None:
            self.eos_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        if self.eos_found.all():
            elapsed = (time.perf_counter() - t0) * 1000
            self.step_times_ms.append(elapsed)
            self.call_count += 1
            self.total_overhead_ms += elapsed
            return True  # ← GPU→CPU sync (.all() returns Python bool)

        last_tokens = input_ids[:, -1]
        current_has_eos = last_tokens == self.eos_token_id
        self.eos_found = self.eos_found | current_has_eos

        result = bool(self.eos_found.all())  # ← 이 .all()이 sync 유발!
        elapsed = (time.perf_counter() - t0) * 1000
        self.step_times_ms.append(elapsed)
        self.call_count += 1
        self.total_overhead_ms += elapsed
        return result


class NoSyncStopCriteria(StoppingCriteria):
    """
    per-step GPU→CPU sync 없는 stopping criteria.
    항상 False를 반환 → max_new_tokens에 의해서만 종료.
    EOS 위치는 나중에 CPU에서 1회 탐색.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return False  # Python literal — GPU tensor 접근 없음, sync 없음


# ──────────────────────────────────────────────────────────────────
# GPU-resident top-p 샘플러 (sync 없음)
# ──────────────────────────────────────────────────────────────────

def top_p_sample_gpu(
    logits: torch.Tensor,  # [B, vocab_size]
    temperature: float = 0.6,
    top_p: float = 0.98,
) -> torch.Tensor:
    """
    GPU에서만 실행되는 top-p 샘플링.
    CPU sync 없음. torch.multinomial은 GPU 연산.

    Returns: [B] 다음 토큰 ids
    """
    # Temperature scaling
    logits = logits.float() / temperature

    # Sort descending
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 누적 확률 > top_p인 토큰 마스킹 (but keep at least the first token)
    # shift by 1 so first token above threshold is included
    sorted_indices_to_remove = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) > top_p
    # 첫 번째 토큰은 항상 유지
    sorted_indices_to_remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

    # 원래 vocab 순서로 복원
    filtered_logits = torch.zeros_like(logits)
    filtered_logits.scatter_(-1, sorted_indices, sorted_logits)

    # 샘플링 (torch.multinomial = GPU 연산, sync 없음)
    probs = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
    return next_token


# ──────────────────────────────────────────────────────────────────
# GPU-side EOS 위치 추적 (sync 없음)
# ──────────────────────────────────────────────────────────────────

class GPUEosTracker:
    """
    EOS 토큰을 GPU 텐서에 저장하고 CPU sync 없이 위치를 추적.
    루프 종료 후 .get_eos_positions() 한 번만 CPU transfer.
    """

    def __init__(self, batch_size: int, max_steps: int, eos_token_id: int, device: str):
        self.eos_token_id = eos_token_id
        self.device = device
        # EOS 발견된 step index, 미발견이면 max_steps (sentinel)
        self.eos_steps = torch.full(
            (batch_size,), max_steps, dtype=torch.long, device=device
        )
        self.found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def update(self, tokens: torch.Tensor, step: int) -> None:
        """tokens: [B] — GPU에서만 실행, sync 없음."""
        is_eos = (tokens == self.eos_token_id) & ~self.found
        step_tensor = torch.full_like(self.eos_steps, step)
        self.eos_steps = torch.where(is_eos, step_tensor, self.eos_steps)
        self.found = self.found | is_eos

    def get_eos_positions(self) -> list[int]:
        """1회만 CPU transfer (sync 발생)."""
        return self.eos_steps.cpu().tolist()

    def all_found_gpu(self) -> torch.Tensor:
        """CPU sync 없이 GPU tensor로 반환 (bool 검사 안 함)."""
        return self.found


# ──────────────────────────────────────────────────────────────────
# CUDA Event 타이머
# ──────────────────────────────────────────────────────────────────

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


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


# ──────────────────────────────────────────────────────────────────
# Phase 0 + 1: 베이스라인 (계측 포함) + output_logits 최적화
# ──────────────────────────────────────────────────────────────────

def run_phase_baseline(
    model: Alpamayo1_5,
    model_inputs: dict,
    eos_token_id: int,
    output_logits: bool,
    use_no_sync_stop: bool = False,
    fixed_max_tokens: int | None = None,
) -> dict:
    """
    단일 inference 실행 (baseline 또는 no-sync variant).

    Returns: timing + sync profiling dict
    """
    data_copy = copy.deepcopy(model_inputs)
    input_ids = data_copy["tokenized_data"].get("input_ids")
    tokenized_data = data_copy["tokenized_data"]
    _ = tokenized_data.pop("input_ids")

    ego_history_xyz = data_copy["ego_history_xyz"]
    ego_history_rot = data_copy["ego_history_rot"]

    # fuse_traj_tokens
    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

    # Generation config
    gen_cfg = model.vlm.generation_config
    gen_cfg.top_p = TOP_P
    gen_cfg.temperature = TEMPERATURE
    gen_cfg.do_sample = True
    gen_cfg.num_return_sequences = NUM_TRAJ_SAMPLES
    gen_cfg.max_new_tokens = fixed_max_tokens if fixed_max_tokens else MAX_NEW_TOKENS_BASELINE
    gen_cfg.output_logits = output_logits
    gen_cfg.return_dict_in_generate = True
    gen_cfg.pad_token_id = model.tokenizer.pad_token_id

    # Stopping criteria
    if use_no_sync_stop:
        stopping_criteria = StoppingCriteriaList([NoSyncStopCriteria()])
    else:
        stop_eos = InstrumentedStopAfterEOS(eos_token_id=eos_token_id)
        stopping_criteria = StoppingCriteriaList([stop_eos])

    logits_processor = LogitsProcessorList([
        ExpertLogitsProcessor(
            traj_token_offset=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size,
        )
    ])

    # ── 시간 측정 ──
    sw = CudaStopwatch()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    sw.start()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=gen_cfg,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )

    gpu_time_ms = sw.stop_ms()
    wall_time_ms = (time.perf_counter() - wall_start) * 1000

    n_generated = vlm_outputs.sequences.shape[1] - input_ids.shape[1]

    result = {
        "gpu_time_ms": gpu_time_ms,
        "wall_time_ms": wall_time_ms,
        "sync_overhead_ms": wall_time_ms - gpu_time_ms,
        "n_generated_tokens": int(n_generated),
        "output_logits": output_logits,
        "use_no_sync_stop": use_no_sync_stop,
    }

    if not use_no_sync_stop:
        result["stop_eos_calls"] = stop_eos.call_count
        result["stop_eos_overhead_ms"] = stop_eos.total_overhead_ms
        result["stop_eos_per_call_ms"] = (
            stop_eos.total_overhead_ms / max(stop_eos.call_count, 1)
        )
        result["stop_eos_step_times_ms"] = stop_eos.step_times_ms[:10]  # 처음 10개만 저장

    return result


# ──────────────────────────────────────────────────────────────────
# Phase 3: StaticCache 실험
# ──────────────────────────────────────────────────────────────────

def run_phase_static_cache(
    model: Alpamayo1_5,
    model_inputs: dict,
    eos_token_id: int,
    fixed_max_tokens: int,
) -> dict:
    """
    HuggingFace StaticCache를 generate()에 직접 주입.
    pre-allocated KV cache로 동적 allocation 오버헤드 제거.
    """
    try:
        from transformers import StaticCache
    except ImportError:
        return {"error": "StaticCache not available in this transformers version"}

    data_copy = copy.deepcopy(model_inputs)
    input_ids = data_copy["tokenized_data"].pop("input_ids")
    tokenized_data = data_copy["tokenized_data"]

    ego_history_xyz = data_copy["ego_history_xyz"]
    ego_history_rot = data_copy["ego_history_rot"]
    input_ids = model.fuse_traj_tokens(
        input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
    )

    prefill_len = input_ids.shape[1]
    max_cache_len = prefill_len + fixed_max_tokens + 16  # 여유 공간

    # StaticCache 생성
    static_cache = StaticCache(
        config=model.vlm.config.text_config,
        batch_size=NUM_TRAJ_SAMPLES,
        max_cache_len=max_cache_len,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    logger.info(
        f"    StaticCache: batch={NUM_TRAJ_SAMPLES}, "
        f"max_cache_len={max_cache_len}, "
        f"size≈{NUM_TRAJ_SAMPLES * max_cache_len * 144 / 1024:.1f} MB"
        f"  (prefill_len={prefill_len}, decode={fixed_max_tokens})"
    )

    gen_cfg = model.vlm.generation_config
    gen_cfg.top_p = TOP_P
    gen_cfg.temperature = TEMPERATURE
    gen_cfg.do_sample = True
    gen_cfg.num_return_sequences = NUM_TRAJ_SAMPLES
    gen_cfg.max_new_tokens = fixed_max_tokens
    gen_cfg.output_logits = False  # Phase 1 최적화 포함
    gen_cfg.return_dict_in_generate = True
    gen_cfg.pad_token_id = model.tokenizer.pad_token_id

    stopping_criteria = StoppingCriteriaList([NoSyncStopCriteria()])
    logits_processor = LogitsProcessorList([
        ExpertLogitsProcessor(
            traj_token_offset=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size,
        )
    ])

    sw = CudaStopwatch()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    sw.start()

    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = model.vlm.generate(
                input_ids=input_ids,
                generation_config=gen_cfg,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                past_key_values=static_cache,
                **tokenized_data,
            )
        gpu_time_ms = sw.stop_ms()
        wall_time_ms = (time.perf_counter() - wall_start) * 1000
        n_generated = vlm_outputs.sequences.shape[1] - input_ids.shape[1]
        return {
            "gpu_time_ms": gpu_time_ms,
            "wall_time_ms": wall_time_ms,
            "n_generated_tokens": int(n_generated),
            "success": True,
        }
    except Exception as e:
        gpu_time_ms = sw.stop_ms()
        return {
            "error": str(e),
            "gpu_time_ms": gpu_time_ms,
            "success": False,
        }


# ──────────────────────────────────────────────────────────────────
# Phase 4: Full Custom Decode Loop (KV Cache 가방 핵심)
# ──────────────────────────────────────────────────────────────────

def run_phase_custom_loop(
    model: Alpamayo1_5,
    model_inputs: dict,
    eos_token_id: int,
    fixed_max_tokens: int,
    attempt_cuda_graph: bool = True,
) -> dict:
    """
    generate() 완전 우회. GPU-resident decode loop.

    전략:
      1. vlm.forward() 직접 호출로 prefill (with pixel_values)
      2. StaticCache에 prefill K/V 저장
      3. Decode loop: CPU sync 없이 max_new_tokens step 실행
      4. top_p 샘플링 GPU 내부 실행
      5. 루프 종료 후 단 1회 token_buffer.cpu()
      6. (선택) CUDAGraph 캡처 시도

    Action Expert 호환성:
      반환 past_key_values를 alpamayo1_5.sample_trajectories...가
      사용할 수 있는지 검증.
    """
    try:
        from transformers import StaticCache
    except ImportError:
        return {"error": "StaticCache not available"}

    data_copy = copy.deepcopy(model_inputs)
    tokenized_data = data_copy["tokenized_data"]
    input_ids = tokenized_data.pop("input_ids")
    ego_history_xyz = data_copy["ego_history_xyz"]
    ego_history_rot = data_copy["ego_history_rot"]
    input_ids = model.fuse_traj_tokens(
        input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
    )

    device = input_ids.device
    B_orig, prefill_len = input_ids.shape
    B = B_orig * NUM_TRAJ_SAMPLES  # expand for num_return_sequences

    # num_return_sequences 확장
    exp_input_ids = input_ids.repeat_interleave(NUM_TRAJ_SAMPLES, dim=0)  # [B, L]
    exp_attn_mask = tokenized_data.get("attention_mask")
    if exp_attn_mask is not None:
        exp_attn_mask = exp_attn_mask.repeat_interleave(NUM_TRAJ_SAMPLES, dim=0)

    max_cache_len = prefill_len + fixed_max_tokens + 16
    static_cache = StaticCache(
        config=model.vlm.config.text_config,
        batch_size=B,
        max_cache_len=max_cache_len,
        device=device,
        dtype=torch.bfloat16,
    )

    logger.info(
        f"    StaticCache: B={B}, max_cache_len={max_cache_len}, "
        f"size≈{B * max_cache_len * 144 / 1024:.1f} MB"
    )

    # ── Step 1: Prefill ───────────────────────────────────────────
    # pixel_values는 B_orig 배치로 왔으므로, B로 repeat interleave
    pixel_values = tokenized_data.get("pixel_values")
    image_grid_thw = tokenized_data.get("image_grid_thw")
    if pixel_values is not None:
        # pixel_values: [N_total_patches, C, pH, pW] — not batched by B_orig
        # image_grid_thw: [n_images, 3] — grid info per image
        # 각 batch element는 같은 이미지 세트 → repeat interleave
        # HuggingFace internally handles this via batch assignment
        pixel_values = pixel_values.repeat_interleave(NUM_TRAJ_SAMPLES, dim=0)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.repeat_interleave(NUM_TRAJ_SAMPLES, dim=0)

    cache_position_prefill = torch.arange(prefill_len, device=device, dtype=torch.long)

    sw_prefill = CudaStopwatch()
    sw_prefill.start()

    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prefill_out = model.vlm(
                input_ids=exp_input_ids,
                attention_mask=exp_attn_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                past_key_values=static_cache,
                cache_position=cache_position_prefill,
                use_cache=True,
            )
    except Exception as e:
        return {"phase": "prefill", "error": str(e), "success": False}

    prefill_time_ms = sw_prefill.stop_ms()

    # rope_deltas 저장 (위치 인코딩 offset)
    rope_deltas = getattr(model.vlm.model, "rope_deltas", None)
    logger.info(
        f"    Prefill done: {prefill_time_ms:.1f}ms, "
        f"rope_deltas={'set' if rope_deltas is not None else 'None'}"
    )

    # ── Step 2: 첫 decode 토큰 ───────────────────────────────────
    first_logits = prefill_out.logits[:, -1, :].float()  # [B, vocab]

    # ExpertLogitsProcessor 적용 (trajectory 토큰 마스킹)
    traj_offset = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    first_logits[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")

    first_token = top_p_sample_gpu(first_logits, TEMPERATURE, TOP_P)  # [B]

    # ── Step 3: Decode loop (no sync) ────────────────────────────
    token_buffer = torch.zeros(B, fixed_max_tokens, dtype=torch.long, device=device)
    eos_tracker = GPUEosTracker(B, fixed_max_tokens, eos_token_id, device)

    token_buffer[:, 0] = first_token
    eos_tracker.update(first_token, step=0)

    current_token = first_token.unsqueeze(1)  # [B, 1]

    # CUDAGraph 캡처 시도
    cuda_graph = None
    static_input_buf = None
    static_cache_pos = None
    static_logits_out = None  # graph 캡처 성공 시 tensor로 교체됨

    if attempt_cuda_graph:
        logger.info("    Attempting CUDAGraph capture...")
        try:
            # Prefill 이후 첫 decode 위치부터 캡처
            # cache_position=0 으로 설정하면 prefill 데이터 덮어쓰기 → 반드시 prefill_len 사용
            static_input_buf = torch.zeros(B, 1, dtype=torch.long, device=device)
            static_cache_pos = torch.tensor([prefill_len], device=device, dtype=torch.long)

            # Warmup (CUDAGraph 캡처 전 2회 워밍업 필수)
            # prefill_len 위치에 dummy 데이터 작성 — 이후 decode step 0이 덮어씀
            for _ in range(2):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _out = model.vlm(
                        input_ids=static_input_buf,
                        past_key_values=static_cache,
                        cache_position=static_cache_pos,
                        use_cache=True,
                    )
            torch.cuda.synchronize()

            # CUDAGraph 캡처
            # static_input_buf, static_cache_pos는 capture 이후에도 .copy_()로 업데이트 가능
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    graph_out = model.vlm(
                        input_ids=static_input_buf,
                        past_key_values=static_cache,
                        cache_position=static_cache_pos,
                        use_cache=True,
                    )
            # graph_out.logits는 정적 텐서 — replay 시마다 같은 GPU 주소에 결과가 기록됨
            static_logits_out = graph_out.logits[:, -1, :]  # [B, vocab], 정적 텐서
            logger.info("    ✓ CUDAGraph captured successfully")
        except Exception as e:
            cuda_graph = None
            logger.warning(f"    CUDAGraph capture failed: {e}")

    # ── Decode loop 본체 ─────────────────────────────────────────
    sw_decode = CudaStopwatch()
    torch.cuda.synchronize()
    wall_decode_start = time.perf_counter()
    sw_decode.start()

    for step in range(1, fixed_max_tokens):
        cache_pos = torch.tensor([prefill_len + step], device=device, dtype=torch.long)

        if cuda_graph is not None:
            # CUDAGraph replay: CPU→GPU 복사 후 replay
            static_input_buf.copy_(current_token)
            static_cache_pos.copy_(cache_pos)
            cuda_graph.replay()
            logits = static_logits_out.float()
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.vlm(
                    input_ids=current_token,
                    past_key_values=static_cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )
            logits = out.logits[:, -1, :].float()  # [B, vocab]

        # ExpertLogitsProcessor (GPU only, no sync)
        logits[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")

        # top-p 샘플링 (GPU only, no sync)
        next_token = top_p_sample_gpu(logits, TEMPERATURE, TOP_P)

        token_buffer[:, step] = next_token
        eos_tracker.update(next_token, step=step)
        current_token = next_token.unsqueeze(1)

    decode_gpu_time_ms = sw_decode.stop_ms()
    decode_wall_time_ms = (time.perf_counter() - wall_decode_start) * 1000

    # ── Step 4: 단 1회 CPU sync ───────────────────────────────────
    t_cpu_transfer_start = time.perf_counter()
    tokens_cpu = token_buffer.cpu()  # 이 1줄이 유일한 GPU→CPU sync
    eos_positions = eos_tracker.get_eos_positions()  # 여기도 CPU transfer (1회)
    t_cpu_transfer_ms = (time.perf_counter() - t_cpu_transfer_start) * 1000

    # ── Step 5: EOS 위치로 실제 생성 토큰 수 계산 ─────────────────
    actual_steps = [pos + 1 if pos < fixed_max_tokens else fixed_max_tokens
                    for pos in eos_positions]
    mean_steps = float(np.mean(actual_steps))

    logger.info(
        f"    Custom loop done: prefill={prefill_time_ms:.1f}ms, "
        f"decode={decode_gpu_time_ms:.1f}ms, "
        f"cpu_transfer={t_cpu_transfer_ms:.1f}ms, "
        f"eos_pos={eos_positions}, actual_steps≈{mean_steps:.1f}"
    )

    # ── Step 6: Action Expert 호환성 검증 ─────────────────────────
    # vlm_outputs.past_key_values.crop()이 작동하는지 확인
    action_expert_compatible = False
    try:
        seq_len_before = static_cache.get_seq_length()
        static_cache.crop(seq_len_before)  # crop 테스트
        action_expert_compatible = True
    except Exception as e:
        logger.warning(f"    Action Expert compatibility test failed: {e}")

    return {
        "prefill_time_ms": prefill_time_ms,
        "decode_gpu_time_ms": decode_gpu_time_ms,
        "decode_wall_time_ms": decode_wall_time_ms,
        "cpu_transfer_ms": t_cpu_transfer_ms,
        "total_ms": prefill_time_ms + decode_gpu_time_ms + t_cpu_transfer_ms,
        "n_cpu_syncs": 2,  # token_buffer.cpu() + eos_tracker.get_eos_positions()
        "actual_steps_mean": mean_steps,
        "cuda_graph_captured": cuda_graph is not None,
        "action_expert_compatible": action_expert_compatible,
        "success": True,
    }


# ──────────────────────────────────────────────────────────────────
# 메인 실험 루프
# ──────────────────────────────────────────────────────────────────

def run_phase_repeated(phase_fn, phase_name: str, n_warmup: int, n_measure: int,
                        **kwargs) -> dict:
    """phase_fn을 반복 실행해서 통계를 낸다."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase: {phase_name}")
    logger.info(f"{'='*60}")

    times_gpu = []
    times_wall = []

    for run_idx in range(n_warmup + n_measure):
        is_warmup = run_idx < n_warmup
        torch.cuda.empty_cache()

        result = phase_fn(**kwargs)

        if "error" in result:
            logger.error(f"  FAILED: {result['error']}")
            return {"error": result["error"]}

        gpu_t = result.get("gpu_time_ms", result.get("total_ms", 0))
        wall_t = result.get("wall_time_ms", result.get("decode_wall_time_ms", gpu_t))
        n_tok = result.get("n_generated_tokens", result.get("actual_steps_mean", 0))
        tag = f"WARMUP {run_idx+1}" if is_warmup else f"MEASURE {run_idx-n_warmup+1}"

        logger.info(
            f"  [{tag}] GPU={gpu_t:.1f}ms  Wall={wall_t:.1f}ms  "
            f"n_tok={n_tok:.0f}  "
            + (f"sync_ovhd={result.get('sync_overhead_ms',0):.1f}ms" if not is_warmup else "")
        )
        if "stop_eos_calls" in result and not is_warmup:
            logger.info(
                f"          StopAfterEOS: calls={result['stop_eos_calls']}, "
                f"total_overhead={result['stop_eos_overhead_ms']:.2f}ms, "
                f"per_call={result['stop_eos_per_call_ms']:.3f}ms"
            )

        if not is_warmup:
            times_gpu.append(gpu_t)
            times_wall.append(wall_t)

    mu_gpu, std_gpu = mean_std(times_gpu)
    mu_wall, std_wall = mean_std(times_wall)
    logger.info(
        f"\n  SUMMARY: GPU={mu_gpu:.1f}±{std_gpu:.1f}ms  "
        f"Wall={mu_wall:.1f}±{std_wall:.1f}ms"
    )

    return {
        "phase": phase_name,
        "gpu_mean_ms": mu_gpu,
        "gpu_std_ms": std_gpu,
        "wall_mean_ms": mu_wall,
        "wall_std_ms": std_wall,
        "last_result": result,
    }


def main():
    # ── 데이터 + 모델 로드 ────────────────────────────────────────
    logger.info("Loading dataset...")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    logger.info("Loading model...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(DEVICE)
    model.eval()

    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, DEVICE)

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    prefill_len = inputs["input_ids"].shape[1]
    logger.info(f"Prefill length: {prefill_len} tokens, EOS id: {eos_token_id}")

    # KV cache 크기 계산 및 출력
    n_layers = model.vlm.config.text_config.num_hidden_layers
    n_kv_heads = model.vlm.config.text_config.num_key_value_heads
    head_dim = model.vlm.config.text_config.hidden_size // model.vlm.config.text_config.num_attention_heads
    bytes_per_token = n_layers * 2 * n_kv_heads * head_dim * 2  # BF16 = 2 bytes
    logger.info(f"KV cache: {n_layers}L × {n_kv_heads}KV-heads × {head_dim}dim")
    logger.info(f"  Per token: {bytes_per_token/1024:.1f} KB")
    logger.info(f"  Prefill ({prefill_len} tok): {prefill_len*bytes_per_token/1024/1024:.1f} MB")

    torch.cuda.manual_seed_all(42)
    all_results = {}

    # ── Phase 0: 베이스라인 + sync 계측 ──────────────────────────
    r_baseline = run_phase_repeated(
        phase_fn=run_phase_baseline,
        phase_name="Phase0_Baseline_with_sync_profiling",
        n_warmup=NUM_WARMUP, n_measure=NUM_MEASURE,
        model=model, model_inputs=model_inputs,
        eos_token_id=eos_token_id,
        output_logits=True,
        use_no_sync_stop=False,
    )
    all_results["phase0_baseline"] = r_baseline

    # 측정된 실제 토큰 수 파악
    measured_n_tokens = int(r_baseline["last_result"].get("n_generated_tokens", 70))
    fixed_tokens = measured_n_tokens + 5  # 여유 5 토큰
    logger.info(f"\n→ Measured token count: {measured_n_tokens}  "
                f"→ Using fixed_max_tokens={fixed_tokens} for subsequent experiments")

    kv_decode_size_mb = fixed_tokens * bytes_per_token / 1024 / 1024
    logger.info(f"→ Decode-only KV bag size: {kv_decode_size_mb:.1f} MB")

    # ── Phase 1: output_logits=False ─────────────────────────────
    r_phase1 = run_phase_repeated(
        phase_fn=run_phase_baseline,
        phase_name="Phase1_output_logits_False",
        n_warmup=NUM_WARMUP, n_measure=NUM_MEASURE,
        model=model, model_inputs=model_inputs,
        eos_token_id=eos_token_id,
        output_logits=False,
        use_no_sync_stop=False,
    )
    all_results["phase1_no_logits"] = r_phase1

    # ── Phase 2: No-Sync Stopping Criteria ───────────────────────
    r_phase2 = run_phase_repeated(
        phase_fn=run_phase_baseline,
        phase_name="Phase2_NoSync_StopCriteria",
        n_warmup=NUM_WARMUP, n_measure=NUM_MEASURE,
        model=model, model_inputs=model_inputs,
        eos_token_id=eos_token_id,
        output_logits=False,
        use_no_sync_stop=True,
        fixed_max_tokens=fixed_tokens,
    )
    all_results["phase2_no_sync_stop"] = r_phase2

    # ── Phase 3: StaticCache ─────────────────────────────────────
    r_phase3 = run_phase_repeated(
        phase_fn=run_phase_static_cache,
        phase_name="Phase3_StaticCache",
        n_warmup=NUM_WARMUP, n_measure=NUM_MEASURE,
        model=model, model_inputs=model_inputs,
        eos_token_id=eos_token_id,
        fixed_max_tokens=fixed_tokens,
    )
    all_results["phase3_static_cache"] = r_phase3

    # ── Phase 4: Full Custom Loop ────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Phase 4: Full Custom Decode Loop (KV Cache 가방)")
    logger.info(f"{'='*60}")

    times_custom = []
    last_custom_result = None
    for run_idx in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = run_idx < NUM_WARMUP
        torch.cuda.empty_cache()

        result = run_phase_custom_loop(
            model=model,
            model_inputs=model_inputs,
            eos_token_id=eos_token_id,
            fixed_max_tokens=fixed_tokens,
            attempt_cuda_graph=(run_idx == NUM_WARMUP),  # 첫 measure run에만 CUDAGraph 시도
        )

        if not result.get("success", False):
            logger.error(f"  Phase 4 FAILED: {result.get('error')}")
            break

        total_ms = result["total_ms"]
        tag = f"WARMUP {run_idx+1}" if is_warmup else f"MEASURE {run_idx-NUM_WARMUP+1}"
        logger.info(
            f"  [{tag}] total={total_ms:.1f}ms  "
            f"prefill={result['prefill_time_ms']:.1f}ms  "
            f"decode={result['decode_gpu_time_ms']:.1f}ms  "
            f"cpu_xfer={result['cpu_transfer_ms']:.1f}ms  "
            f"n_syncs={result['n_cpu_syncs']}  "
            f"CUDAGraph={result['cuda_graph_captured']}"
        )
        if not is_warmup:
            times_custom.append(total_ms)
            last_custom_result = result

    if times_custom:
        mu, std = mean_std(times_custom)
        logger.info(f"\n  SUMMARY: total={mu:.1f}±{std:.1f}ms")
        all_results["phase4_custom_loop"] = {
            "phase": "Phase4_Custom_Loop",
            "total_mean_ms": mu,
            "total_std_ms": std,
            "n_cpu_syncs": 2,
            "cuda_graph_captured": last_custom_result.get("cuda_graph_captured", False),
            "action_expert_compatible": last_custom_result.get("action_expert_compatible", False),
        }

    # ── 최종 비교 테이블 ──────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("KV CACHE BAG EXPERIMENT — RESULTS SUMMARY")
    logger.info("="*70)

    phases = [
        ("Baseline (output_logits=True, StopAfterEOS)",  "phase0_baseline"),
        ("output_logits=False",                           "phase1_no_logits"),
        ("No-sync StopCriteria + fixed tokens",           "phase2_no_sync_stop"),
        ("StaticCache + no-sync",                         "phase3_static_cache"),
        ("Custom Loop (single sync)",                     "phase4_custom_loop"),
    ]

    baseline_gpu = all_results["phase0_baseline"]["gpu_mean_ms"]
    logger.info(f"{'Phase':<45} | {'GPU(ms)':>9} | {'Wall(ms)':>9} | {'Speedup':>8}")
    logger.info("-"*75)
    for label, key in phases:
        if key not in all_results or "error" in all_results[key]:
            logger.info(f"{label:<45} | {'FAILED':>9} |")
            continue
        r = all_results[key]
        g = r.get("gpu_mean_ms", r.get("total_mean_ms", 0))
        w = r.get("wall_mean_ms", g)
        sp = baseline_gpu / g if g > 0 else 0
        logger.info(f"{label:<45} | {g:>7.1f}ms | {w:>7.1f}ms | {sp:>7.2f}×")

    if "phase0_baseline" in all_results:
        p0 = all_results["phase0_baseline"]["last_result"]
        logger.info(f"\nSync 계측 (Phase 0):")
        logger.info(f"  StopAfterEOS 호출 횟수: {p0.get('stop_eos_calls', 'N/A')}")
        logger.info(f"  StopAfterEOS 총 오버헤드: {p0.get('stop_eos_overhead_ms', 0):.2f}ms")
        logger.info(f"  호출당 평균 시간: {p0.get('stop_eos_per_call_ms', 0):.3f}ms")
        logger.info(
            f"  이론 절약 가능량: "
            f"{p0.get('stop_eos_calls',0) * p0.get('stop_eos_per_call_ms',0):.1f}ms"
        )

    # ── 결과 저장 ─────────────────────────────────────────────────
    out_json = OUT / "results.json"
    # last_result에서 tensor 제거 (JSON 직렬화 불가)
    for k in all_results:
        if "last_result" in all_results[k]:
            lr = all_results[k]["last_result"]
            all_results[k]["last_result"] = {
                kk: vv for kk, vv in lr.items()
                if not isinstance(vv, (torch.Tensor, list))
            }
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {out_json}")


if __name__ == "__main__":
    main()
