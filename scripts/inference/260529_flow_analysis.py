"""
260529_flow_analysis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[실험 목적]
  FlowMatching 소스코드 분석 결과를 바탕으로 올바른 Flow 시간을 측정하고
  새로운 최적화 가능성(Decode Skip, Batch 축소)을 정량화한다.

[소스코드 분석 핵심 발견 (2026-05-29)]
  1. FlowMatching = 10-step Euler ODE
     - x_dims=(64,2), 10번 model.expert forward pass
     - forward() 미구현; sample()로 추론
     - model.diffusion.sample(batch_size, step_fn, device)

  2. step_fn이 VLM KV cache를 크로스 어텐션으로 사용
     - model.expert ← prompt_cache (VLM KV, 3086 토큰, GPU 상주)
     - CPU offload = 아키텍처적으로 불가능 (device mismatch 발생)
     - 매 ODE step마다 prompt_cache.crop(prefill_seq_len) 복원

  3. ForceEarlyEOS(max_coc_tokens=0) 이미 구현됨
     - Decode(1,818ms) 완전 스킵 가능
     - VE(728ms) + Prefill(1,423ms) + Flow만 실행

  4. num_traj_samples=6 기본값 → batch=6으로 60회 expert 호출
     - batch=1로 줄이면 Flow ~870ms → ~200ms 추정

[실험 구성]
  F-A: Flow 단독 시간 측정 (num_traj_samples=6, 10 ODE steps) — 베이스라인 검증
  F-B: Flow 단독 시간 측정 (num_traj_samples=1, 10 ODE steps) — 배치 축소 효과
  F-C: Decode Skip (max_coc_tokens=0, num_traj_samples=6) — 1,818ms 절약
  F-D: Decode Skip + batch=1 (max_coc_tokens=0, num_traj_samples=1) — 최대 latency 축소
  F-Impact: 전체 파이프라인 임팩트 분석 및 비교표

[예상 결과]
  Baseline     : 4,839ms (VE=728 + Prefill=1,423 + Decode=1,818 + Flow=870)
  F-C 예상     : ~3,021ms  (Decode≈0, Flow≈870,  batch=6)  [-38%]
  F-D 예상     : ~2,351ms  (Decode≈0, Flow≈200,  batch=1)  [-51%]
  최대 절약    : 2,488ms (−51%)  [양자화 없이, 모델 구조 수정 없이]

[실행]
  source ~/alpamayo1.5/a1_5_venv/bin/activate && cd ~/alpamayo1.5
  python3 scripts/inference/260529_flow_analysis.py --exp A
  python3 scripts/inference/260529_flow_analysis.py --exp ALL
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
)

# ── 프로젝트 경로 설정 ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import (
    Alpamayo1_5,
    ExpertLogitsProcessor,
    ForceEarlyEOS,
)
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    replace_padding_after_eos,
    to_special_token,
)

# ── 로거 ────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 전역 상수
# ──────────────────────────────────────────────────────────────────────────────
CLIP_ID  = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US    = 5_100_000
DEVICE   = "cuda"

NUM_WARMUP  = 1
NUM_MEASURE = 3

# 2026-05-28 확정 베이스라인
BASELINE_VE_MS      = 728.0
BASELINE_PREFILL_MS = 1_423.0
BASELINE_DECODE_MS  = 1_818.0
BASELINE_FLOW_MS    = 870.0
BASELINE_TOTAL_MS   = 4_839.0

OUT = Path("profiling_results/260529_flow_analysis")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1: 측정 인프라
# ══════════════════════════════════════════════════════════════════════════════

class GpuTimer:
    """CUDA Event 기반 GPU 연산 타이머 (GPU 클럭, ns 단위)."""
    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)

    def start(self, stream=None):
        self._start.record(stream)

    def stop(self, stream=None):
        self._end.record(stream)

    def elapsed_ms(self) -> float:
        return self._start.elapsed_time(self._end)

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *_):
        self.stop()


class WallTimer:
    """CPU wall-clock 타이머. GPU 연산 포함 구간은 cuda.synchronize() 포함."""
    def __init__(self):
        self._t0 = self._t1 = 0.0

    def start(self):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (self._t1 - self._t0) * 1_000.0

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *_):
        self.stop()


class CpuTimer:
    """순수 CPU 타이머 (GPU sync 없음)."""
    def __init__(self):
        self._t0 = self._t1 = 0.0

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (self._t1 - self._t0) * 1_000.0


class StatSummary:
    """기술 통계 (scipy 의존성 없음)."""
    def __init__(self, values: List[float], label: str = ""):
        self.label  = label
        self.values = sorted(values)
        self.n      = len(values)

    @property
    def mean(self) -> float:
        return mean(self.values) if self.n else float("nan")

    @property
    def std(self) -> float:
        return stdev(self.values) if self.n >= 2 else 0.0

    @property
    def minimum(self) -> float:
        return self.values[0] if self.n else float("nan")

    @property
    def maximum(self) -> float:
        return self.values[-1] if self.n else float("nan")

    @property
    def p95(self) -> float:
        if self.n == 0:
            return float("nan")
        idx = 0.95 * (self.n - 1)
        lo, hi = int(idx), min(int(idx) + 1, self.n - 1)
        frac = idx - lo
        return self.values[lo] * (1 - frac) + self.values[hi] * frac

    def to_dict(self) -> dict:
        return {
            "label":  self.label,
            "n":      self.n,
            "mean":   round(self.mean,    2),
            "std":    round(self.std,     2),
            "min":    round(self.minimum, 2),
            "max":    round(self.maximum, 2),
            "p95":    round(self.p95,     2),
            "values": [round(v, 2) for v in self.values],
        }

    def __repr__(self) -> str:
        return (
            f"[{self.label}] "
            f"mean={self.mean:.1f}ms  std={self.std:.1f}ms  "
            f"[{self.minimum:.1f}, {self.maximum:.1f}]  p95={self.p95:.1f}ms"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2: 입력 준비
# ══════════════════════════════════════════════════════════════════════════════

def prepare_inputs(
    model: Alpamayo1_5,
    clip_id: str,
    t_us: int,
    device: str = DEVICE,
) -> Tuple[torch.Tensor, dict]:
    """
    데이터 로드 → 토크나이즈 → fuse_traj_tokens 완료된 input_ids + tok_data 반환.

    반환값:
        input_ids : [1, seq_len] fused input IDs (GPU)
        tok_data  : {attention_mask, pixel_values, image_grid_thw} (GPU)
    """
    processor = helper.get_processor(model.tokenizer)
    data = load_physical_aiavdataset(clip_id, t0_us=t_us)

    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    raw_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    raw_inputs = helper.to_device(raw_inputs, device)
    ego_data   = helper.to_device(
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        device,
    )

    input_ids_raw = raw_inputs.pop("input_ids")
    with torch.no_grad():
        input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
    torch.cuda.synchronize()

    return input_ids, raw_inputs   # raw_inputs = tok_data


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3: VLM Generate 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def run_vlm_generate(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
    num_traj_samples: int = 6,
    max_coc_tokens: Optional[int] = None,
) -> Tuple[Any, float]:
    """
    VLM 생성 단계(VE + Prefill + Decode)를 실행하고 결과 + 소요 시간을 반환한다.

    소스코드 기반 올바른 구현:
      - model.vlm.generate() 직접 호출
      - vlm_outputs.rope_deltas = model.vlm.model.rope_deltas 추가
      - ForceEarlyEOS(max_coc_tokens) 지원: 0이면 Decode 완전 스킵

    인자:
        input_ids        : fused input IDs (from fuse_traj_tokens), shape [1, L]
        tok_data         : attention_mask, pixel_values, image_grid_thw
        num_traj_samples : num_return_sequences → vlm_outputs.sequences shape [N, L']
        max_coc_tokens   : None=정상 Decode, 0=Decode Skip, N=N-token CoC 후 EOS

    반환:
        vlm_outputs : GenerateOutput (sequences, past_key_values, rope_deltas)
        generate_ms : VLM generate 총 소요 시간 (ms, wall-clock)
    """
    eos_token_id = model.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )

    # ── generation_config 설정 (deepcopy로 원본 보호) ────────────────────────
    gen_config = copy.deepcopy(model.vlm.generation_config)
    gen_config.top_p                   = 0.98
    gen_config.temperature             = 0.6
    gen_config.do_sample               = True
    gen_config.num_return_sequences    = num_traj_samples
    gen_config.max_new_tokens          = model.config.tokens_per_future_traj
    gen_config.output_logits           = False   # 타이밍 실험: logits 불필요
    gen_config.return_dict_in_generate = True
    gen_config.pad_token_id            = model.tokenizer.pad_token_id

    # ── 로짓 프로세서 ─────────────────────────────────────────────────────────
    logits_processor = LogitsProcessorList([
        ExpertLogitsProcessor(
            traj_token_offset=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size,
        )
    ])

    if max_coc_tokens is not None:
        logits_processor.append(
            ForceEarlyEOS(eos_token_id=eos_token_id, max_coc_tokens=max_coc_tokens)
        )
        # max_coc_tokens=0: 2 steps (EOS forced + KV update) → 사실상 Decode 없음
        gen_config.max_new_tokens = max_coc_tokens + 4

    # ── stopping criteria ─────────────────────────────────────────────────────
    stopping_criteria = StoppingCriteriaList([
        StopAfterEOS(eos_token_id=eos_token_id)
    ])

    # ── 실행 및 타이밍 ──────────────────────────────────────────────────────
    wt = WallTimer()
    wt.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=gen_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            attention_mask=tok_data.get("attention_mask"),
            pixel_values=tok_data.get("pixel_values"),
            image_grid_thw=tok_data.get("image_grid_thw"),
        )
    wt.stop()
    generate_ms = wt.elapsed_ms()

    # ── rope_deltas 추가 (step_fn 구성에 필수) ────────────────────────────────
    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas

    # ── EOS 이후 패딩 정규화 ─────────────────────────────────────────────────
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )

    return vlm_outputs, generate_ms


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 4: Flow 단독 실행 헬퍼 (핵심)
# ══════════════════════════════════════════════════════════════════════════════

def run_flow_sample(
    model: Alpamayo1_5,
    vlm_outputs: Any,
    tok_data: dict,
    eos_token_id: int,
) -> Tuple[torch.Tensor, float]:
    """
    VLM KV cache를 바탕으로 step_fn closure를 구성하고
    model.diffusion.sample()의 순수 실행 시간을 측정한다.

    소스코드 기반 올바른 구현 (alpamayo1_5.py L346~389):
      - action_in_proj: [B,64,2] → [B,64,hidden_size]  Fourier+MLP
      - model.expert: [B,64,hidden] × VLM KV cache (3086 tok) → [B,64,hidden]
      - action_out_proj: [B,64,hidden] → [B,64,2]  velocity field
      - prompt_cache.crop(prefill_seq_len): 매 ODE step 후 캐시 복원

    인자:
        vlm_outputs  : run_vlm_generate() 의 결과 (sequences, past_key_values, rope_deltas)
        tok_data     : attention_mask (prefix_mask로 사용)
        eos_token_id : traj_future_start 토큰 ID

    반환:
        sampled_action : [B, 64, 2] 최종 궤적 (accel + curvature)
        flow_ms        : Flow ODE 실행 시간 (ms, GPU Event 기반)
    """
    device = DEVICE

    # ── 기본 정보 추출 ────────────────────────────────────────────────────────
    prompt_cache    = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    b_star          = vlm_outputs.sequences.shape[0]           # num_traj_samples
    n_diffusion_tok = model.action_space.get_action_space_dims()[0]  # 64

    logger.info(
        f"    [Flow] b_star={b_star}  n_diffusion_tok={n_diffusion_tok}  "
        f"kv_cache_seq_len={prefill_seq_len}"
    )

    # ── EOS offset 탐색 ───────────────────────────────────────────────────────
    offset = Alpamayo1_5._find_eos_offset(
        sequences=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        device=device,
    )

    # ── prefix_mask (입력 패딩 마스크 반영) ──────────────────────────────────
    prefix_mask = tok_data.get("attention_mask")    # [1, L]
    if prefix_mask is not None:
        # b_star개 시퀀스에 맞게 반복 (HuggingFace generate가 내부적으로 반복한 것과 동일)
        prefix_mask = prefix_mask.repeat_interleave(b_star, dim=0)  # [b_star, L]

    # ── position_ids & 4D attention_mask 구성 ────────────────────────────────
    position_ids, attn_mask_4d = Alpamayo1_5._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=vlm_outputs.rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tok,
        b_star=b_star,
        device=device,
        prefix_mask=prefix_mask,
    )

    # ── expert_non_causal_attention 설정 ─────────────────────────────────────
    forward_kwargs: dict = {}
    if getattr(model.config, "expert_non_causal_attention", False):
        forward_kwargs["is_causal"] = False

    # ── step_fn closure (소스코드 L347~376 기반) ──────────────────────────────
    # 중요: prompt_cache는 클로저가 캡처 → ODE 10 step 동안 동일 캐시를 공유
    # 매 step 후 prompt_cache.crop(prefill_seq_len)으로 복원 (64 expert 토큰 제거)
    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        # ① [B, 64, 2] → [B, 64, hidden_size]  (Fourier + 4-layer MLP)
        future_token_embeds = model.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(bs, n_diffusion_tok, -1)

        # ② model.expert: 64 query tokens × (kv_cache + 64) key-value tokens
        #    VLM KV cache를 크로스 어텐션으로 참조 (CPU offload 불가)
        expert_out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attn_mask_4d,
            use_cache=True,
            **forward_kwargs,
        )

        # ③ 캐시 복원: expert가 추가한 64 토큰 제거 (다음 ODE step을 위해)
        prompt_cache.crop(prefill_seq_len)

        # ④ [B, 64, hidden_size] → [B, 64, 2]  velocity field
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tok:]
        return model.action_out_proj(last_hidden).view(
            -1, *model.action_space.get_action_space_dims()
        )

    # ── Flow ODE 실행 (10 Euler steps × b_star expert calls) ─────────────────
    gt = GpuTimer()
    gt.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sampled_action = model.diffusion.sample(
            batch_size=b_star,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
        )
    gt.stop()
    torch.cuda.synchronize()
    flow_ms = gt.elapsed_ms()

    return sampled_action, flow_ms


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 5: 실험 실행
# ══════════════════════════════════════════════════════════════════════════════

def _run_subexp(
    label: str,
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
    eos_token_id: int,
    num_traj_samples: int,
    max_coc_tokens: Optional[int],
) -> dict:
    """
    단일 (num_traj_samples, max_coc_tokens) 조합에 대해
    generate + flow 시간을 NUM_WARMUP + NUM_MEASURE 회 측정한다.

    반환 dict:
      generate_ms_list, flow_ms_list, total_ms_list (각 StatSummary)
      n_decode_tokens_avg: 평균 생성된 decode 토큰 수 (Decode 스킵 확인용)
    """
    print(f"\n  ─── {label} "
          f"(num_traj_samples={num_traj_samples}, "
          f"max_coc_tokens={'full_decode' if max_coc_tokens is None else max_coc_tokens}) ───")

    gen_ms_list:   List[float] = []
    flow_ms_list:  List[float] = []
    total_ms_list: List[float] = []
    n_decode_tokens_list: List[int] = []

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"

        try:
            # ── VLM Generate (VE + Prefill + Decode) ──────────────────────
            vlm_outputs, gen_ms = run_vlm_generate(
                model, input_ids, tok_data,
                num_traj_samples=num_traj_samples,
                max_coc_tokens=max_coc_tokens,
            )

            # 실제 생성된 decode 토큰 수 (input_ids 이후 새로 생성된 토큰)
            n_new_tokens = vlm_outputs.sequences.shape[1] - input_ids.shape[1]

            # ── Flow ODE 실행 ──────────────────────────────────────────────
            _, flow_ms = run_flow_sample(
                model, vlm_outputs, tok_data, eos_token_id
            )

            total_ms = gen_ms + flow_ms

            print(
                f"  [{tag}]  "
                f"generate={gen_ms:6.0f}ms  "
                f"flow={flow_ms:6.0f}ms  "
                f"total={total_ms:6.0f}ms  "
                f"decode_tokens={n_new_tokens}"
            )

            if not is_warmup:
                gen_ms_list.append(gen_ms)
                flow_ms_list.append(flow_ms)
                total_ms_list.append(total_ms)
                n_decode_tokens_list.append(n_new_tokens)

        except Exception as e:
            logger.error(f"  [{tag}] 실패: {e}")
            traceback.print_exc()

        # KV cache 메모리 해제
        torch.cuda.empty_cache()

    if not flow_ms_list:
        return {"error": f"{label} 모든 trial 실패"}

    gen_stat   = StatSummary(gen_ms_list,   f"{label}_generate")
    flow_stat  = StatSummary(flow_ms_list,  f"{label}_flow")
    total_stat = StatSummary(total_ms_list, f"{label}_total")
    avg_decode_tokens = mean(n_decode_tokens_list) if n_decode_tokens_list else 0

    print(f"\n  {gen_stat}")
    print(f"  {flow_stat}")
    print(f"  {total_stat}")
    print(f"  평균 decode 토큰 수: {avg_decode_tokens:.1f}")

    return {
        "num_traj_samples":    num_traj_samples,
        "max_coc_tokens":      max_coc_tokens,
        "generate_ms":         gen_stat.to_dict(),
        "flow_ms":             flow_stat.to_dict(),
        "total_ms":            total_stat.to_dict(),
        "avg_decode_tokens":   round(avg_decode_tokens, 1),
    }


def run_flow_experiments(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
    eos_token_id: int,
    run_flags: dict,   # {"A": bool, "B": bool, "C": bool, "D": bool}
) -> dict:
    """
    Flow 관련 4가지 실험을 실행하고 전체 결과를 반환한다.
    """
    W = 70
    print(f"\n{'═'*W}")
    print("  Alpamayo Flow 분석 실험 (소스코드 기반 올바른 구현)")
    print(f"{'═'*W}")
    print(f"\n  FlowMatching 구조 (소스코드 확인):")
    print(f"    - ODE: Euler, 10 steps")
    print(f"    - x_dims: (64, 2)  →  64 waypoints × (accel, curvature)")
    print(f"    - step_fn: action_in_proj → model.expert (2279M) → action_out_proj")
    print(f"    - VLM KV cache 크로스 어텐션: {BASELINE_PREFILL_MS:.0f}ms 분량의 컨텍스트")
    print(f"\n  베이스라인: {BASELINE_TOTAL_MS:.0f}ms")
    print(f"    VE={BASELINE_VE_MS:.0f}  "
          f"Prefill={BASELINE_PREFILL_MS:.0f}  "
          f"Decode={BASELINE_DECODE_MS:.0f}  "
          f"Flow={BASELINE_FLOW_MS:.0f}")

    results: dict = {
        "description": "Flow 소스코드 기반 재구현 실험",
        "baseline": {
            "ve_ms":      BASELINE_VE_MS,
            "prefill_ms": BASELINE_PREFILL_MS,
            "decode_ms":  BASELINE_DECODE_MS,
            "flow_ms":    BASELINE_FLOW_MS,
            "total_ms":   BASELINE_TOTAL_MS,
        },
        "sub_experiments": {},
    }

    # ────────────────────────────────────────────────────────────────────────
    # Sub F-A: 베이스라인 (num_traj_samples=6, full Decode)
    # ────────────────────────────────────────────────────────────────────────
    if run_flags.get("A", True):
        print(f"\n{'─'*W}")
        print("  Sub F-A: 베이스라인 Flow 시간 검증 (batch=6, full Decode)")
        print("  목적: 870ms 베이스라인이 올바른지 소스코드 기반으로 재측정")
        r_A = _run_subexp(
            "F-A",
            model, input_ids, tok_data, eos_token_id,
            num_traj_samples=6,
            max_coc_tokens=None,
        )
        results["sub_experiments"]["F_A"] = r_A
        if "flow_ms" in r_A:
            diff = r_A["flow_ms"]["mean"] - BASELINE_FLOW_MS
            print(f"\n  → 측정된 Flow: {r_A['flow_ms']['mean']:.0f}ms  "
                  f"(베이스라인 대비 {diff:+.0f}ms, "
                  f"{diff/BASELINE_FLOW_MS*100:+.1f}%)")

    # ────────────────────────────────────────────────────────────────────────
    # Sub F-B: batch=1 (num_traj_samples=1, full Decode)
    # ────────────────────────────────────────────────────────────────────────
    if run_flags.get("B", True):
        print(f"\n{'─'*W}")
        print("  Sub F-B: batch=1 Flow 시간 (num_traj_samples=1, full Decode)")
        print("  목적: 궤적 샘플 수를 6→1로 줄이면 Flow 시간이 얼마나 단축되는가?")
        print("  이론: expert forward는 메모리 바운드 → batch=1 시 ~870/6=145ms 하한")
        r_B = _run_subexp(
            "F-B",
            model, input_ids, tok_data, eos_token_id,
            num_traj_samples=1,
            max_coc_tokens=None,
        )
        results["sub_experiments"]["F_B"] = r_B
        if "flow_ms" in r_B and "F_A" in results["sub_experiments"] and \
                "flow_ms" in results["sub_experiments"]["F_A"]:
            fa_flow = results["sub_experiments"]["F_A"]["flow_ms"]["mean"]
            fb_flow = r_B["flow_ms"]["mean"]
            ratio   = fa_flow / fb_flow if fb_flow > 0 else float("nan")
            print(f"\n  → batch=6 대비 batch=1: {fa_flow:.0f}ms → {fb_flow:.0f}ms  "
                  f"({ratio:.2f}× 빠름, 이론 상한 6×)")

    # ────────────────────────────────────────────────────────────────────────
    # Sub F-C: Decode Skip (max_coc_tokens=0, batch=6)
    # ────────────────────────────────────────────────────────────────────────
    if run_flags.get("C", True):
        print(f"\n{'─'*W}")
        print("  Sub F-C: Decode Skip (max_coc_tokens=0, batch=6)")
        print("  목적: 1,818ms Decode를 ForceEarlyEOS로 스킵하면 전체 파이프라인이 얼마나 빨라지는가?")
        print("  메커니즘: max_coc_tokens=0 → 즉시 EOS → Prefill KV로만 Flow 조건화")
        r_C = _run_subexp(
            "F-C",
            model, input_ids, tok_data, eos_token_id,
            num_traj_samples=6,
            max_coc_tokens=0,
        )
        results["sub_experiments"]["F_C"] = r_C
        if "total_ms" in r_C:
            saving = BASELINE_TOTAL_MS - r_C["total_ms"]["mean"]
            print(f"\n  → 총 파이프라인: {r_C['total_ms']['mean']:.0f}ms  "
                  f"(베이스라인 대비 {saving:+.0f}ms, "
                  f"{saving/BASELINE_TOTAL_MS*100:.1f}% 절약)")

    # ────────────────────────────────────────────────────────────────────────
    # Sub F-D: Decode Skip + batch=1 (최대 latency 축소)
    # ────────────────────────────────────────────────────────────────────────
    if run_flags.get("D", True):
        print(f"\n{'─'*W}")
        print("  Sub F-D: Decode Skip + batch=1 (max_coc_tokens=0, num_traj_samples=1)")
        print("  목적: 시스템 수준 최적화로 달성 가능한 최소 latency 측정")
        print("  이론: VE(728) + Prefill(1,423) + Decode(≈0) + Flow(~150ms) ≈ 2,301ms")
        r_D = _run_subexp(
            "F-D",
            model, input_ids, tok_data, eos_token_id,
            num_traj_samples=1,
            max_coc_tokens=0,
        )
        results["sub_experiments"]["F_D"] = r_D
        if "total_ms" in r_D:
            saving = BASELINE_TOTAL_MS - r_D["total_ms"]["mean"]
            print(f"\n  → 총 파이프라인: {r_D['total_ms']['mean']:.0f}ms  "
                  f"(베이스라인 대비 {saving:+.0f}ms, "
                  f"{saving/BASELINE_TOTAL_MS*100:.1f}% 절약)")

    # ────────────────────────────────────────────────────────────────────────
    # Impact Analysis: 비교표 + 연구 함의
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  ★ Flow 최적화 실험 종합 결과")
    print(f"{'═'*W}")

    subs = results["sub_experiments"]
    rows = [
        ("Baseline",          4839.0, "full CoC reasoning, 6 trajectories"),
        ("F-A (측정)",        subs.get("F_A", {}).get("total_ms", {}).get("mean", float("nan")),
         "batch=6, full decode (베이스라인 재측정)"),
        ("F-B (batch=1)",     subs.get("F_B", {}).get("total_ms", {}).get("mean", float("nan")),
         "batch=1, full decode"),
        ("F-C (DecodeSkip6)", subs.get("F_C", {}).get("total_ms", {}).get("mean", float("nan")),
         "batch=6, no CoC reasoning"),
        ("F-D (DecodeSkip1)", subs.get("F_D", {}).get("total_ms", {}).get("mean", float("nan")),
         "batch=1, no CoC reasoning"),
    ]

    print(f"\n  {'구성':20}  {'총 시간':>9}  {'절약':>9}  {'절약%':>7}  비고")
    print(f"  {'-'*65}")
    for name, total_ms, note in rows:
        if total_ms != total_ms:   # nan
            print(f"  {name:20}  {'측정 없음':>9}  {'—':>9}  {'—':>7}  {note}")
        else:
            saving = BASELINE_TOTAL_MS - total_ms
            pct    = saving / BASELINE_TOTAL_MS * 100
            marker = " ★" if name.startswith("F-D") else ""
            print(f"  {name:20}  {total_ms:8.0f}ms  {saving:+8.0f}ms  {pct:6.1f}%  {note}{marker}")

    # Flow 단독 시간 비교 (generate 시간 제외)
    print(f"\n  Flow ODE 단독 시간 비교:")
    flow_rows = [
        ("Baseline",  870.0,                                                              "batch=6, 10 ODE steps"),
        ("F-A",       subs.get("F_A", {}).get("flow_ms", {}).get("mean", float("nan")), "batch=6"),
        ("F-B",       subs.get("F_B", {}).get("flow_ms", {}).get("mean", float("nan")), "batch=1"),
        ("F-C",       subs.get("F_C", {}).get("flow_ms", {}).get("mean", float("nan")), "batch=6, no decode KV"),
        ("F-D",       subs.get("F_D", {}).get("flow_ms", {}).get("mean", float("nan")), "batch=1, no decode KV"),
    ]
    print(f"  {'구성':12}  {'Flow 시간':>10}  {'대비 baseline':>14}  비고")
    print(f"  {'-'*55}")
    for name, fms, note in flow_rows:
        if fms != fms:
            print(f"  {name:12}  {'측정 없음':>10}  {'—':>14}  {note}")
        else:
            diff   = fms - 870.0
            ratio  = fms / 870.0
            print(f"  {name:12}  {fms:8.0f}ms    {diff:+8.0f}ms ({ratio:.2f}×)  {note}")

    # 연구 함의
    print(f"\n  ─ 연구 함의 ─")
    fa_flow = subs.get("F_A", {}).get("flow_ms", {}).get("mean", None)
    fb_flow = subs.get("F_B", {}).get("flow_ms", {}).get("mean", None)
    fc_tot  = subs.get("F_C", {}).get("total_ms", {}).get("mean", None)
    fd_tot  = subs.get("F_D", {}).get("total_ms", {}).get("mean", None)

    if fa_flow and fb_flow:
        batch_ratio = fa_flow / fb_flow
        print(f"  1. Flow batch 축소 효과: 6→1로 {batch_ratio:.2f}× 단축")
        print(f"     memory-bound ratio 예상: ~6×,  실측: {batch_ratio:.1f}×")

    if fc_tot:
        fc_save = BASELINE_TOTAL_MS - fc_tot
        print(f"  2. Decode Skip (batch=6): {fc_tot:.0f}ms ({fc_save:+.0f}ms, "
              f"{fc_save/BASELINE_TOTAL_MS*100:.1f}% 절약)")
        print(f"     CoC reasoning 없이 Prefill KV만으로 Flow 조건화")

    if fd_tot:
        fd_save = BASELINE_TOTAL_MS - fd_tot
        print(f"  3. Decode Skip + batch=1: {fd_tot:.0f}ms ({fd_save:+.0f}ms, "
              f"{fd_save/BASELINE_TOTAL_MS*100:.1f}% 절약) ← 최대 절약")
        print(f"     양자화 없이, 모델 구조 수정 없이 달성 가능한 시스템 최적화")

    print(f"\n  4. CPU Flow offload: 아키텍처적으로 불가능")
    print(f"     model.expert가 GPU KV cache(3086 토큰)를 매 ODE step마다 크로스 어텐션")
    print(f"     → expert CPU 이동 시 device mismatch (KV cache는 GPU에 고정)")

    # 이론 파이프라인 임팩트
    impact = {}
    if fd_tot:
        impact["decode_skip_batch1"] = {
            "total_ms":  round(fd_tot, 1),
            "saving_ms": round(BASELINE_TOTAL_MS - fd_tot, 1),
            "saving_pct": round((BASELINE_TOTAL_MS - fd_tot) / BASELINE_TOTAL_MS * 100, 2),
            "description": "Decode Skip (ForceEarlyEOS) + num_traj_samples=1",
        }
    if fc_tot:
        impact["decode_skip_batch6"] = {
            "total_ms":  round(fc_tot, 1),
            "saving_ms": round(BASELINE_TOTAL_MS - fc_tot, 1),
            "saving_pct": round((BASELINE_TOTAL_MS - fc_tot) / BASELINE_TOTAL_MS * 100, 2),
            "description": "Decode Skip (ForceEarlyEOS) only, 궤적 다양성 유지",
        }
    results["impact_analysis"] = impact

    print(f"\n{'═'*W}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 6: main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Alpamayo Flow 분석 실험 (소스코드 기반 올바른 구현)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실험:
  A   : 베이스라인 Flow 시간 (batch=6, full Decode)
  B   : batch=1 Flow 시간 (full Decode)
  C   : Decode Skip, batch=6  (max_coc_tokens=0)
  D   : Decode Skip, batch=1  (max_coc_tokens=0, 최대 최적화)
  ALL : A→B→C→D 전체 실행
        """,
    )
    parser.add_argument("--exp", choices=["A","B","C","D","ALL"], default="ALL")
    args = parser.parse_args()

    W = 70
    print("=" * W)
    print("  Alpamayo Flow 분석 실험 (소스코드 기반)")
    print(f"  exp={args.exp}  device={DEVICE}  dtype=BF16")
    print("=" * W)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
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
    logger.info("모델 로드 완료")

    # 모델 구조 확인
    for mod_name in ["expert", "action_in_proj", "action_out_proj", "diffusion", "action_space"]:
        mod = getattr(model, mod_name, None)
        if mod is not None:
            if isinstance(mod, torch.nn.Module):
                params_M = sum(p.numel() for p in mod.parameters()) / 1e6
                logger.info(f"  model.{mod_name}: {type(mod).__name__}  {params_M:.1f}M params")
            else:
                logger.info(f"  model.{mod_name}: {type(mod).__name__}")

    # action_space dims 확인
    action_dims = model.action_space.get_action_space_dims()
    logger.info(f"  action_space.get_action_space_dims() = {action_dims}  "
                f"(waypoints={action_dims[0]}, action_dim={action_dims[1]})")
    logger.info(f"  diffusion: int_method={model.diffusion.int_method}  "
                f"num_inference_steps={model.diffusion.num_inference_steps}")

    # ── 입력 준비 ─────────────────────────────────────────────────────────────
    logger.info(f"\n입력 데이터 준비 중 (clip={CLIP_ID}, t={T0_US})...")
    input_ids, tok_data = prepare_inputs(model, CLIP_ID, T0_US)
    logger.info(f"  input_ids: {input_ids.shape}  ({input_ids.shape[1]} tokens)")

    eos_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    logger.info(f"  eos_id (traj_future_start) = {eos_id}")

    # ── 실험 플래그 ───────────────────────────────────────────────────────────
    run_flags = {
        "A": args.exp in ("A", "ALL"),
        "B": args.exp in ("B", "ALL"),
        "C": args.exp in ("C", "ALL"),
        "D": args.exp in ("D", "ALL"),
    }

    # ── 실험 실행 ─────────────────────────────────────────────────────────────
    try:
        results = run_flow_experiments(
            model, input_ids, tok_data, eos_id, run_flags
        )
    except Exception as e:
        logger.error(f"실험 실패: {e}")
        traceback.print_exc()
        results = {"error": str(e)}

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    out_path = OUT / f"results_flow_{args.exp}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\n결과 저장: {out_path}")
    print(f"\n  전체 결과: {out_path}")


if __name__ == "__main__":
    main()
