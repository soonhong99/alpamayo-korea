"""
260609_ncu_full_bandwidth.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alpamayo 1.5 추론 4단계 전체 정확한 DRAM 대역폭 측정 스크립트

260607 버전 대비 핵심 수정:
  [수정] Decode: step_010 하나 → DecodeAll (EOS까지 전 step) 로 변경
         - 기존 Phase/Decode_all은 on_generate_end()가 Flow 이후에 호출되어
           Flow 실행 중에도 NVTX가 열려있는 버그가 있었음
         - 수정: DecodeAll을 첫 Flow step 시작 직전에 닫는 방식으로 해결
  [유지] VE     : Phase/Vision_Encoder  (기존 정상 동작)
  [유지] Prefill: Phase/LM_Prefill      (기존 정상 동작)
  [유지] Flow   : FlowODE/FlowStep      (기존 정상 동작)

NVTX 구조 (ncu --nvtx-include 타겟):
  Phase/Vision_Encoder    ← VE 단계
  Phase/LM_Prefill        ← LM Prefill 단계
  DecodeAll               ← Decode 전체 (EOS까지, 수정된 부분)
    └─ Decode/step_001    ← 개별 step (step별 타이밍 + 분석용)
    └─ Decode/step_002
    ...
  FlowODE/FlowStep        ← Flow 각 ODE step

DecodeAll 열고 닫는 방식:
  OPEN : on_vlm_pre에서 seq==1 첫 감지 (post_prefill → decode 전환) 시
  CLOSE: _flow_step_start에서 _flow_step[0]==0 (첫 Flow step) 시
         → on_generate_end() 의존 제거, Flow 시작 전 확실히 닫힘

사용법:
  python3 260609_ncu_full_bandwidth.py --mode timing
  python3 260609_ncu_full_bandwidth.py --mode ncu_single_run
  (ncu 래핑 하에서): sudo -E ncu --nvtx --nvtx-include "DecodeAll" ... python3 ... --mode ncu_single_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.cuda.nvtx as nvtx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════════════

MODEL_ID    = "nvidia/Alpamayo-1.5-10B"
CLIP_ID     = "030c760c-ae38-49aa-9ad8-f5650a545d26"  # 기존 스크립트와 동일
DRAM_PEAK   = 231.0  # GB/s (Thor LPDDR5X)
DEFAULT_RESULTS_DIR = "/home/ice401/alpamayo1.5/profiling_results/260609_ncu_full"


# ═══════════════════════════════════════════════════════════════════════
# CUDA Event 타이머
# ═══════════════════════════════════════════════════════════════════════

class CUDATimer:
    def __init__(self, name: str = ""):
        self.name = name
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)
        self._ms    = None

    def start(self):
        self._start.record()
        self._ms = None

    def stop(self):
        self._end.record()

    def ms(self) -> float:
        if self._ms is None:
            torch.cuda.synchronize()
            self._ms = self._start.elapsed_time(self._end)
        return self._ms

    def reset(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)
        self._ms    = None


# ═══════════════════════════════════════════════════════════════════════
# 모델 / 입력 로드
# ═══════════════════════════════════════════════════════════════════════

def load_model():
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    log.info("모델 로드 시작 (~3-4분)...")
    t0 = time.time()
    # ★ attn_implementation 미지정 (260607 원본과 동일)
    # ★ .cuda() 필수 — 미호출 시 GPU 추론 불가
    model = Alpamayo1_5.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda().eval()
    elapsed = time.time() - t0
    total_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    log.info(f"모델 로드 완료: {elapsed:.1f}s | {total_gb:.3f} GB")
    return model


def load_raw_data():
    """
    실제 데이터셋 입력 로드 — 260607/260513 스크립트와 동일.
    alpamayo1_5.load_physical_aiavdataset + helper 사용.
    """
    log.info("입력 데이터 로드...")
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    data = load_physical_aiavdataset(CLIP_ID, t0_us=5_100_000)
    log.info("  raw data 로드 완료")
    return data, helper


def prepare_model_inputs(model, data, helper) -> dict:
    """raw data → 모델 입력 형식 변환 (260607 원본과 동일)"""
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
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
    model_inputs = helper.to_device(
        {
            "tokenized_data"  : inputs,
            "ego_history_xyz" : data["ego_history_xyz"],
            "ego_history_rot" : data["ego_history_rot"],
        },
        "cuda",
    )
    seq_len = inputs["input_ids"].shape[-1]
    log.info(f"  입력 토큰 길이: {seq_len}")
    return model_inputs


# ═══════════════════════════════════════════════════════════════════════
# 이론값 계산
# ═══════════════════════════════════════════════════════════════════════

def compute_theoretical(model) -> dict:
    """모델 가중치 크기 기반 이론적 DRAM 접근량 계산"""

    def module_gb(mod):
        return sum(p.numel() * p.element_size() for p in mod.parameters()) / 1e9

    result = {}

    # VE
    if hasattr(model, 'vlm') and hasattr(model.vlm, 'visual'):
        result['VE_weights_GB'] = module_gb(model.vlm.visual)
        log.info(f"  VE 가중치:             {result['VE_weights_GB']:.3f} GB")

    # LM
    lm_mod = None
    if hasattr(model, 'vlm'):
        for attr in ('language_model', 'model'):
            cand = getattr(model.vlm, attr, None)
            if cand is not None and hasattr(cand, 'layers'):
                lm_mod = cand
                break
            if cand is not None:
                sub = getattr(cand, 'model', None)
                if sub is not None and hasattr(sub, 'layers'):
                    lm_mod = cand
                    break

    if lm_mod is not None:
        result['LM_weights_GB'] = module_gb(lm_mod)
        n_layers = len(lm_mod.layers) if hasattr(lm_mod, 'layers') else \
                   len(lm_mod.model.layers) if hasattr(getattr(lm_mod, 'model', None), 'layers') else 0
        result['n_lm_layers'] = n_layers
        log.info(f"  LM 가중치 ({n_layers} layers): {result['LM_weights_GB']:.3f} GB")

    # Flow (Action Expert)
    flow_gb = 0.0
    for attr in ('expert', 'action_in_proj', 'action_out_proj'):
        mod = getattr(model, attr, None)
        if mod is not None:
            flow_gb += module_gb(mod)
    result['Flow_weights_GB'] = flow_gb
    log.info(f"  Flow 가중치:           {result['Flow_weights_GB']:.3f} GB")

    # Total
    result['total_weights_GB'] = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1e9
    log.info(f"  전체 모델:             {result['total_weights_GB']:.3f} GB")

    return result


# ═══════════════════════════════════════════════════════════════════════
# PhaseSeparator — NVTX 마킹 + 타이밍
# ═══════════════════════════════════════════════════════════════════════

class PhaseSeparator:
    """
    4단계 분리 + NVTX 마킹.

    NVTX 범위 (ncu --nvtx-include 타겟):
      "Phase/Vision_Encoder" ← VE
      "Phase/LM_Prefill"     ← Prefill
      "DecodeAll"            ← Decode 전체 (EOS까지, ★ 수정된 부분)
        └─ "Decode/step_NNN" ← 개별 step
      "FlowODE/FlowStep"     ← Flow 각 ODE step

    DecodeAll 열고 닫기:
      OPEN : vlm_pre에서 seq==1 첫 감지 시 (post_prefill → decode)
      CLOSE: _flow_step_start에서 _flow_step[0]==0 시 (Flow 시작 직전)
    """

    def __init__(self):
        self.state            = "idle"
        self.decode_step      = 0
        self.decode_all_open  = False   # ★ DecodeAll NVTX 열림 여부 추적
        self.prefill_done     = False
        self.ode_step         = 0       # ★ Flow ODE step 카운터 (reset()에서 초기화)

        self.t_ve             = CUDATimer("VE")
        self.t_prefill        = CUDATimer("LM_Prefill")
        self.t_decode         = CUDATimer("Decode")
        self.t_flow           = CUDATimer("Flow")
        self.t_decode_steps: list[float] = []
        self._t_step          = CUDATimer("decode_step")

    def reset(self):
        self.state           = "idle"
        self.decode_step     = 0
        self.decode_all_open = False
        self.prefill_done    = False
        self.ode_step        = 0       # ★ 매 run마다 리셋 — 이게 없으면 2nd+ run에서 Flow 훅이 안 발동
        for t in (self.t_ve, self.t_prefill, self.t_decode, self.t_flow, self._t_step):
            t.reset()
        self.t_decode_steps.clear()

    def _seq(self, args, kwargs) -> int | None:
        for src in [
            kwargs.get("input_ids"),
            kwargs.get("inputs_embeds"),
            kwargs.get("hidden_states"),
            *(a for a in args if isinstance(a, torch.Tensor)),
        ]:
            if src is None:
                continue
            if isinstance(src, torch.Tensor):
                if src.ndim == 2:
                    return int(src.shape[-1])
                if src.ndim == 3:
                    return int(src.shape[1])
        return None

    # ── VLM hook ──────────────────────────────────────────────────────
    def on_vlm_pre(self, module, args, kwargs):
        seq = self._seq(args, kwargs)
        if seq is None:
            return

        if seq > 1 and self.state == "idle":
            # ── VE 시작 ──
            nvtx.range_push("Phase")
            nvtx.range_push("Vision_Encoder")
            self.t_ve.start()
            self.state = "vision"

        elif seq == 1:
            if self.state == "post_prefill":
                # ── Decode 첫 step 시작 ──
                # ★ ncu --nvtx-include 은 반드시 2레벨 parent/child 필터를 필요로 함
                #   단일 레벨 "DecodeAll" → ncu가 캡처 안 함 (확인된 동작)
                #   → "Phase/DecodeAll" 2레벨로 구성
                #   → close_decode_all()에서 Flow 시작 직전에 Phase+DecodeAll 모두 닫음
                #   → Flow 실행 시 Phase 충돌 없음 (VE/Prefill Phase는 이미 닫혔음)
                nvtx.range_push("Phase")      # 외부 (parent)
                nvtx.range_push("DecodeAll")  # 내부 (child, ncu 필터 타겟)
                self.decode_all_open = True
                self.t_decode.start()
                self.state       = "decode"
                self.decode_step = 1
            elif self.state == "decode":
                self.decode_step += 1

            if self.state == "decode":
                # 개별 step NVTX (타이밍 + 분석용)
                nvtx.range_push("Decode")
                nvtx.range_push(f"step_{self.decode_step:03d}")
                self._t_step.start()

    def on_vlm_post(self, module, args, output):
        if self.state == "decode":
            self._t_step.stop()
            self.t_decode_steps.append(self._t_step.ms())
            self._t_step.reset()
            nvtx.range_pop()  # step_NNN
            nvtx.range_pop()  # Decode
            # ★ DecodeAll은 여기서 닫지 않음
            #   (다음 step이 올 수 있고, Flow 시작 직전에 닫음)

    # ── LM hook (VE/Prefill 분리) ─────────────────────────────────────
    def on_lm_pre(self, module, args, kwargs):
        seq = self._seq(args, kwargs)
        if seq is None:
            return
        if seq > 1 and self.state == "vision":
            nvtx.range_pop()  # Vision_Encoder
            nvtx.range_pop()  # Phase
            self.t_ve.stop()
            nvtx.range_push("Phase")
            nvtx.range_push("LM_Prefill")
            self.t_prefill.start()
            self.state = "lm_prefill"

    def on_lm_post(self, module, args, output):
        if self.state == "lm_prefill":
            nvtx.range_pop()  # LM_Prefill
            nvtx.range_pop()  # Phase
            self.t_prefill.stop()
            self.prefill_done = True
            self.state = "post_prefill"

    def close_decode_all(self):
        """Flow 시작 직전에 호출 — Phase/DecodeAll NVTX 닫기 (2레벨 pop)"""
        if self.decode_all_open:
            self.t_decode.stop()
            nvtx.range_pop()  # DecodeAll (child)
            nvtx.range_pop()  # Phase     (parent) ← 2레벨 구조이므로 Phase도 pop
            self.decode_all_open = False

    def on_flow_pre(self, module, args, kwargs):
        self.t_flow.start()

    def on_flow_post(self, module, args, output):
        self.t_flow.stop()


# ═══════════════════════════════════════════════════════════════════════
# 훅 등록
# ═══════════════════════════════════════════════════════════════════════

class _FlowHookHandle:
    """model.diffusion.sample 래핑 해제용 핸들"""
    def __init__(self, mod, orig_fn):
        self._mod  = mod
        self._orig = orig_fn

    def remove(self):
        self._mod.sample = self._orig


def register_hooks(model, sep: PhaseSeparator) -> list:
    hooks = []

    # ── VLM hook ──────────────────────────────────────────────────────
    if hasattr(model, 'vlm'):
        hooks.append(model.vlm.register_forward_pre_hook(
            lambda m, a, kw: sep.on_vlm_pre(m, a, kw), with_kwargs=True))
        hooks.append(model.vlm.register_forward_hook(
            lambda m, a, o: sep.on_vlm_post(m, a, o)))
        log.info("  훅 등록: model.vlm (VLM pre/post)")

        # LM hook
        lm_mod = None
        for lm_attr in ('language_model', 'model'):
            cand = getattr(model.vlm, lm_attr, None)
            if cand is None:
                continue
            if hasattr(cand, 'layers'):
                lm_mod = cand
                log.info(f"  훅 등록: model.vlm.{lm_attr} (LM pre/post)")
                break
            sub = getattr(cand, 'model', None)
            if sub is not None and hasattr(sub, 'layers'):
                lm_mod = cand
                log.info(f"  훅 등록: model.vlm.{lm_attr}.model (LM pre/post)")
                break

        if lm_mod is not None:
            hooks.append(lm_mod.register_forward_pre_hook(
                lambda m, a, kw: sep.on_lm_pre(m, a, kw), with_kwargs=True))
            hooks.append(lm_mod.register_forward_hook(
                lambda m, a, o: sep.on_lm_post(m, a, o)))
        else:
            log.warning("  LM hook 등록 실패 — VE/Prefill 분리 불가")

    # ── Flow hook ─────────────────────────────────────────────────────
    # ★ 핵심: _flow_step[0]==0 일 때 sep.close_decode_all() 호출
    #   → DecodeAll을 Flow 시작 직전에 확실히 닫음
    #   → 기존 on_generate_end() 의존 제거
    flow_registered = False

    if hasattr(model, 'action_in_proj') and hasattr(model, 'action_out_proj'):

        def _flow_step_start(m, a, kw):
            # ★ sep.ode_step 사용 (클로저 _flow_step 제거)
            # sep.reset()이 ode_step=0으로 리셋하므로 매 run 첫 ODE step에서 발동
            if sep.ode_step == 0:
                sep.close_decode_all()   # DecodeAll NVTX 닫기 (Flow 시작 직전)
                sep.t_flow.start()
            sep.ode_step += 1
            nvtx.range_push("FlowODE")
            nvtx.range_push("FlowStep")

        def _flow_step_end(m, a, o):
            nvtx.range_pop()  # FlowStep
            nvtx.range_pop()  # FlowODE

        hooks.append(model.action_in_proj.register_forward_pre_hook(
            _flow_step_start, with_kwargs=True))
        hooks.append(model.action_out_proj.register_forward_hook(_flow_step_end))
        log.info("  훅 등록: action_in_proj/action_out_proj → FlowODE/FlowStep NVTX")
        log.info("  ★ DecodeAll은 첫 Flow step 시작 직전에 자동으로 닫힘")
        flow_registered = True

    elif hasattr(model, 'expert'):
        # 폴백: expert 단독

        def _expert_start(m, a, kw):
            if sep.ode_step == 0:
                sep.close_decode_all()
                sep.t_flow.start()
            sep.ode_step += 1
            nvtx.range_push("FlowODE")
            nvtx.range_push("FlowStep")

        def _expert_end(m, a, o):
            nvtx.range_pop()
            nvtx.range_pop()

        hooks.append(model.expert.register_forward_pre_hook(_expert_start, with_kwargs=True))
        hooks.append(model.expert.register_forward_hook(_expert_end))
        log.info("  훅 등록: expert (FlowODE/FlowStep NVTX, 폴백)")
        flow_registered = True

    if not flow_registered:
        log.warning("  Flow hook 등록 실패 — Flow DRAM 측정 불가")

    # diffusion.sample 타이밍 래핑 (t_flow.stop 담당)
    # t_flow.start()는 _flow_step_start(ode_step==0)에서 호출되므로
    # start 없이 stop만 호출되는 경우를 방어함
    if hasattr(model, 'diffusion') and hasattr(model.diffusion, 'sample'):
        orig_sample = model.diffusion.sample
        def _timed_sample(*args, **kwargs):
            result = orig_sample(*args, **kwargs)
            if sep.ode_step > 0:  # start가 불린 경우에만 stop
                sep.t_flow.stop()
            return result
        model.diffusion.sample = _timed_sample
        hooks.append(_FlowHookHandle(model.diffusion, orig_sample))
        log.info("  훅 등록: model.diffusion.sample (타이밍 보조)")

    return hooks


# ═══════════════════════════════════════════════════════════════════════
# 추론 실행 (warmup + measure)
# ═══════════════════════════════════════════════════════════════════════

def run_inference(model, model_inputs: dict, sep: PhaseSeparator, label: str = "run"):
    """
    단일 추론 실행. 결과 dict 반환.

    ★ 260607 원본과 동일한 호출 방식:
      - torch.autocast 필수 (action_in_proj BF16, diffusion noise Float32 dtype mismatch 방지)
      - data= 키워드 인자
      - top_p / temperature / return_extra 전달
      - nvtx.range_push(label) 로 Warmup/Measure 외곽 범위 생성
        (ncu가 이 범위 이름으로 warmup/measure 패스 구분)
    """
    sep.reset()
    nvtx.range_push(label)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            return_extra=True,
        )
    torch.cuda.synchronize()
    nvtx.range_pop()  # label
    return collect_timing(sep)


def _safe_ms(timer: CUDATimer, name: str) -> float:
    """타이머 ms 읽기 — 실패 시 0 반환 (타이머별 독립 예외 처리)"""
    try:
        return timer.ms()
    except Exception as e:
        log.warning(f"  {name} 타이밍 실패: {e}")
        return 0.0


def collect_timing(sep: PhaseSeparator) -> dict:
    """PhaseSeparator에서 타이밍 수집. 각 타이머 독립 처리 (한 타이머 실패가 나머지에 영향 없음)"""
    ve_ms      = _safe_ms(sep.t_ve,      "VE")
    prefill_ms = _safe_ms(sep.t_prefill, "LM_Prefill")
    flow_ms    = _safe_ms(sep.t_flow,    "Flow")

    # Decode 전체 및 per-step
    steps = sep.t_decode_steps.copy()
    n     = len(steps)

    try:
        decode_total_ms = sep.t_decode.ms()
    except Exception:
        decode_total_ms = sum(steps) if steps else 0.0

    step_mean = sum(steps) / n if n else 0.0
    step_med  = sorted(steps)[n // 2] if n else 0.0
    ss_steps  = steps[3:] if n > 3 else steps
    step_ss   = sum(ss_steps) / len(ss_steps) if ss_steps else step_mean

    return {
        "VE_ms"              : round(ve_ms, 1),
        "LM_Prefill_ms"      : round(prefill_ms, 1),
        "Decode_total_ms"    : round(decode_total_ms, 1),
        "decode_n_steps"     : n,
        "decode_step_mean_ms": round(step_mean, 2),
        "decode_step_ss_ms"  : round(step_ss, 2),
        "Flow_ms"            : round(flow_ms, 1),
        "wall_ms"            : round(ve_ms + prefill_ms + decode_total_ms + flow_ms, 1),
    }


# ═══════════════════════════════════════════════════════════════════════
# 이론값 출력 (decode per-step DRAM 이론)
# ═══════════════════════════════════════════════════════════════════════

def print_theoretical(theory: dict, timing: dict):
    lm_gb    = theory.get("LM_weights_GB", 0)
    n_layers = theory.get("n_lm_layers", 36)
    n_steps  = timing.get("decode_n_steps", 1)
    step_ms  = timing.get("decode_step_ss_ms", 79.1)

    # KV cache 평균 (step 중간값 기준)
    mid_step   = max(n_steps // 2, 1)
    seq_prefill = 3086  # 실측값 근사
    kv_tokens   = seq_prefill + mid_step
    kv_per_layer = 2 * kv_tokens * 128 * 2 / 1e9  # 2(K+V) * tokens * head_dim * BF16
    kv_total_gb  = kv_per_layer * n_layers
    theory["KV_cache_avg_GB"] = round(kv_total_gb, 3)

    decode_theory_gb = lm_gb + kv_total_gb
    theory["Decode_per_step_theory_GB"] = round(decode_theory_gb, 3)

    log.info("─── Decode 1step DRAM 이론 ───")
    log.info(f"  LM 가중치:       {lm_gb:.3f} GB")
    log.info(f"  KV cache (~mid): {kv_total_gb:.3f} GB")
    log.info(f"  합계/step:       {decode_theory_gb:.3f} GB")
    log.info(f"  이론 BW:         {decode_theory_gb/(step_ms/1000):.1f} GB/s "
             f"({decode_theory_gb/(step_ms/1000)/DRAM_PEAK*100:.0f}% peak)")
    log.info(f"  전체 {n_steps} steps 이론: {decode_theory_gb * n_steps:.3f} GB")


# ═══════════════════════════════════════════════════════════════════════
# 메인 실행 모드
# ═══════════════════════════════════════════════════════════════════════

def mode_timing(model, model_inputs: dict, results_dir: Path):
    """타이밍만 측정 (ncu 없음) — 2회 실행 후 평균"""
    theory = compute_theoretical(model)
    sep    = PhaseSeparator()
    hooks  = register_hooks(model, sep)

    log.info("Warmup 1회...")
    run_inference(model, model_inputs, sep, label="Warmup/run_01")
    steps_w = sep.t_decode_steps.copy()
    log.info(f"  Warmup 완료 - Decode {len(steps_w)} steps: {[round(s,1) for s in steps_w]}")

    log.info("측정 2회...")
    runs = []
    for i in range(2):
        result = run_inference(model, model_inputs, sep, label=f"Measure/run_{i+1:02d}")
        runs.append(result)
        n = result['decode_n_steps']
        log.info(
            f"\n[RUN {i+1}]\n"
            f"  VE              :  {result['VE_ms']:7.0f} ms\n"
            f"  LM Prefill      :  {result['LM_Prefill_ms']:7.0f} ms\n"
            f"  Decode          :  {result['Decode_total_ms']:7.0f} ms  ({n} steps)\n"
            f"    per-step mean :  {result['decode_step_mean_ms']:6.2f} ms\n"
            f"    per-step SS   :  {result['decode_step_ss_ms']:6.2f} ms  (step 4+, steady-state)\n"
            f"  Flow            :  {result['Flow_ms']:7.0f} ms\n"
            f"  Wall total      :  {result['wall_ms']:7.0f} ms"
        )

    print_theoretical(theory, runs[-1])

    out = {"theoretical": theory, "runs": runs}
    out_path = results_dir / "timing_results.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"타이밍 저장: {out_path}")

    for h in hooks:
        h.remove()


def mode_ncu_single_run(model, model_inputs: dict, results_dir: Path):
    """
    ncu --nvtx 래핑 하에서 실행 — 1회 추론만 수행.

    ★ 설계 원칙:
      - 추론 1회만 실행 (warmup 없음)
      - torch.manual_seed(42) 로 토큰 수 결정론적 고정
        → ncu가 캡처할 DecodeAll 범위가 정확히 1개, 스텝 수 확정
      - JIT 워밍업 없이도 OK: Decode 단계는 Prefill(36 layer 순전파)
        이후에 시작되므로, Prefill 중 관련 CUDA 커널들이 이미 JIT 컴파일됨
        (JIT 오버헤드가 DRAM 접근량에 영향 없음 — compute 지연만 있음)

    NVTX 구조 (이 모드):
      Measure/run_01            ← outer label
        Phase/Vision_Encoder    ← VE
        Phase/LM_Prefill        ← Prefill
        DecodeAll               ← Decode 전체 ★
          Decode/step_001 ...
        FlowODE/FlowStep × N    ← Flow N ODE steps
    """
    log.info("═══ Mode: ncu_single_run ═══")
    log.info("※ ncu --nvtx 래핑 하에서 실행 중 (1회 추론, seed=42)")

    compute_theoretical(model)
    sep   = PhaseSeparator()
    hooks = register_hooks(model, sep)

    # 토큰 수 결정론적 고정 — ncu 캡처 범위 확정
    torch.manual_seed(42)

    log.info("추론 실행 중 (ncu 캡처 대상)...")
    r = run_inference(model, model_inputs, sep, label="Measure/run_01")

    n = r['decode_n_steps']
    log.info(
        f"완료 — {r['wall_ms']:.0f}ms\n"
        f"  VE:        {r['VE_ms']:.1f} ms\n"
        f"  Prefill:   {r['LM_Prefill_ms']:.1f} ms\n"
        f"  Decode:    {r['Decode_total_ms']:.1f} ms  ({n} steps × "
        f"{r['decode_step_ss_ms']:.1f}ms/step SS)\n"
        f"  Flow:      {r['Flow_ms']:.1f} ms\n"
        f"  ODE steps: {sep.ode_step}\n"
        f"★ ncu DecodeAll 범위: {n} steps × ~2259 커널/step = ~{n*2259} 커널 예상"
    )

    # ★ ncu 오버헤드 타이밍은 timing_results.json을 덮어쓰지 않는다
    #   (Decode: 33분 등 ncu replay 오버헤드가 포함된 값은 실제 BW 계산에 무효)
    #   → timing_ncu_overhead.json 에 별도 저장
    out = {"seed": 42, "run": r}
    out_path = results_dir / "timing_ncu_overhead.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"ncu 오버헤드 타이밍 저장 (참고용): {out_path}")
    log.info("  ※ timing_results.json은 mode_timing 값을 보존 (덮어쓰기 안 함)")

    for h in hooks:
        h.remove()


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Alpamayo DRAM 대역폭 측정 (Decode 전체 수정판)")
    p.add_argument("--mode", choices=["timing", "ncu_single_run"], default="timing")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--model", default=MODEL_ID)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model               = load_model()
    data, helper        = load_raw_data()
    model_inputs        = prepare_model_inputs(model, data, helper)

    if args.mode == "timing":
        mode_timing(model, model_inputs, results_dir)
    elif args.mode == "ncu_single_run":
        mode_ncu_single_run(model, model_inputs, results_dir)


if __name__ == "__main__":
    main()
