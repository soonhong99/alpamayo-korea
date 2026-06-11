"""
260607_ncu_bandwidth_measurement.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Alpamayo 1.5 on Jetson AGX Thor — 4단계 실제 DRAM 대역폭 정밀 측정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
왜 "가중치크기 / 시간" 으로 계산하면 틀리는가?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  단순 계산이 놓치는 것들:
    1. L2 캐시 히트 (32 MB GPU L2가 자주 쓰이는 텐서 캐싱 가능)
    2. KV 캐시 트래픽 (decode시 step마다 누적 KV를 read/write)
    3. Activation 트래픽 (중간 계산 결과가 DRAM을 경유하는 경우)
    4. DynamicCache vs AppendOnlyCache-C: torch.cat 복사본 존재 여부
    5. Prefill: compute-bound → 가중치 재사용 높음 → 실제 BW << 이론

  예시 (decode 1 step 기준):
    이론: LM 가중치 16 GB + KV cache 0.455 GB / 0.079 s ≈ 209 GB/s
    실제: L2 히트로 KV cache 일부 DRAM 미접근 → 더 낮을 수 있음
    ← ncu가 실제 dram__bytes_read.sum 으로 이를 확인한다

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
다른 논문들의 대역폭 측정 방법
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FlashAttention (Dao et al. 2022):
    → 알고리즘 수준 HBM 접근 횟수를 이론 계산 (O(N·d) vs O(N^0.5·d))
    → A100 ncu로 dram__bytes_read.sum 실측으로 검증
    → 이 스크립트와 동일한 방법론

  LLM in a Flash (Apple, 2024):
    → iOS/MacOS 성능 카운터로 DRAM "cold load" vs "hot load" 구분
    → cold = DRAM에서 직접, hot = on-chip SRAM에서

  DejaVu (Liu et al. 2023):
    → ncu per-operator dram__bytes_read.sum 집계
    → layer type (attn, ffn)별 DRAM 접근 분포 분석

  Orca (Yu et al. 2022):
    → 분석적 모델: bytes/step = 2×(weight_bytes + KV_cache_bytes)
    → 쓰기 포함하여 2× 로 추정

  이 스크립트:
    → ncu dram__bytes_read.sum + dram__bytes_write.sum (ground truth)
    → CUDA Event 타이밍 (별도 실행)
    → 실제 BW = bytes / time, 이론치와 비교

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 방법 (260607_run_ncu_stages.sh 가 자동으로 호출)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # [직접 실행] 타이밍만 측정 (빠름, ncu 불필요)
  python3 260607_ncu_bandwidth_measurement.py --mode timing

  # [ncu 래핑용] 1 warmup + 1 측정 실행 준비
  python3 260607_ncu_bandwidth_measurement.py --mode ncu_single_run

  # ncu 명령 예시 (Shell 스크립트가 자동 실행):
  # ncu --nvtx --nvtx-include "Phase/Decode" --metrics "dram__bytes_read.sum,..." \\
  #     --csv python3 260607_ncu_bandwidth_measurement.py --mode ncu_single_run \\
  #     > ncu_decode.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

# ─────────────────────────────────────────────────────────────────────
# 경로 설정 (기존 프로파일링 스크립트와 동일)
# ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT  = Path("profiling_results/260607_ncu_bandwidth")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# 상수 (260531 실험 기준)
# ─────────────────────────────────────────────────────────────────────
DRAM_BW_GBps   = 231.0   # Thor LPDDR5X peak (CLAUDE.md 기준)
KV_N_LAYERS    = 36
KV_N_HEADS     = 8
KV_HEAD_DIM    = 128
SEQ_LEN_PREFILL = 3086   # 실측 seq_len


# ══════════════════════════════════════════════════════════════════════
# CUDA Event 타이머 (260513_profile_v4.py 와 동일 구조)
# ══════════════════════════════════════════════════════════════════════

class CUDATimer:
    def __init__(self, name: str):
        self.name  = name
        self._s    = torch.cuda.Event(enable_timing=True)
        self._e    = torch.cuda.Event(enable_timing=True)
        self._done = False

    def start(self):
        self._s.record()
        self._done = False

    def stop(self):
        self._e.record()
        self._done = True

    def ms(self) -> float:
        if not self._done:
            return 0.0
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)

    def reset(self):
        self._s    = torch.cuda.Event(enable_timing=True)
        self._e    = torch.cuda.Event(enable_timing=True)
        self._done = False


# ══════════════════════════════════════════════════════════════════════
# 이론적 DRAM 접근 바이트 계산
# ══════════════════════════════════════════════════════════════════════

def compute_theoretical_bytes(model) -> dict:
    """
    단계별 이론적 DRAM 접근 바이트 계산.
    실제 ncu 측정값과 비교하는 기준점.

    Alpamayo 1.5 모듈 구조 (소스코드 확인):
      model.vlm             = Qwen2.5-VL (VLM 전체)
      model.vlm.visual      = VE (Vision Encoder, Qwen2_5_VLVisionTransformer)
      model.vlm.model       = Qwen2_5_VLModel (LM + embedding)
      model.vlm.model.layers= 36 × LM decoder layer
      model.expert          = Action Expert (작은 transformer)
      model.action_in_proj  = Action input projection
      model.action_out_proj = Action output projection
      model.diffusion       = Flow matching ODE 스케줄러 (파라미터 거의 없음)

    Returns
    -------
    dict with keys:
      VE_weights_GB, LM_weights_GB, Flow_weights_GB
      KV_cache_full_GB  (seq=3086+17=3103 토큰 기준)
      Decode_per_step_theory_GB  (LM 가중치 1회 + KV cache 읽기 1회)
    """
    def module_bytes(m):
        return sum(p.numel() * p.element_size() for p in m.parameters())

    result = {}

    # ── VE 파라미터 ────────────────────────────────────────────────────
    # Qwen2.5-VL: model.vlm.visual (ViT 기반 Vision Transformer)
    ve_bytes = 0
    if hasattr(model, 'vlm'):
        # 'visual'이 Qwen2.5-VL의 표준 경로
        ve_candidates = ['visual', 'visual_encoder', 'vision_encoder', 'vision_model']
        for attr in ve_candidates:
            mod = getattr(model.vlm, attr, None)
            if mod is not None:
                ve_bytes = module_bytes(mod)
                log.info(f"  VE 모듈 발견: model.vlm.{attr}")
                break

    # ── LM 파라미터 ────────────────────────────────────────────────────
    # Qwen2.5-VL: model.vlm.model (Qwen2_5_VLModel)
    # vlm.language_model 은 없을 수 있음 (버전 따라 다름)
    lm_bytes = 0
    lm_mod   = None
    if hasattr(model, 'vlm'):
        for lm_attr in ('language_model', 'model'):
            cand = getattr(model.vlm, lm_attr, None)
            if cand is None:
                continue
            # layers 직접 소유하거나, cand.model.layers 형태
            if hasattr(cand, 'layers'):
                lm_mod = cand
                log.info(f"  LM 모듈 발견: model.vlm.{lm_attr} (직접 layers 소유)")
                break
            sub = getattr(cand, 'model', None)
            if sub is not None and hasattr(sub, 'layers'):
                lm_mod = cand   # cand 전체가 LM (embed_tokens 포함)
                log.info(f"  LM 모듈 발견: model.vlm.{lm_attr} (하위 model.layers)")
                break
        if lm_mod is not None:
            lm_bytes = module_bytes(lm_mod)

    # ── Flow 단계 파라미터 ─────────────────────────────────────────────
    # Flow 단계 = step_fn = action_in_proj + expert + action_out_proj
    # model.diffusion 은 ODE 스케줄러 (nn.Parameter 거의 없음)
    flow_bytes = 0
    flow_breakdown = {}
    for attr in ['expert', 'action_in_proj', 'action_out_proj', 'diffusion']:
        mod = getattr(model, attr, None)
        if mod is not None:
            b = module_bytes(mod)
            flow_breakdown[attr] = b / 1e9
            flow_bytes += b
    if flow_breakdown:
        log.info(f"  Flow 모듈 구성: " +
                 ", ".join(f"{k}={v:.3f}GB" for k, v in flow_breakdown.items()))

    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # KV 캐시 크기: 36 layers × 2(K+V) × 8 heads × 128 dim × N_tokens × 2 bytes(BF16)
    n_tokens_total = SEQ_LEN_PREFILL + 17  # prefill + 17 decode steps
    kv_full_bytes = KV_N_LAYERS * 2 * KV_N_HEADS * KV_HEAD_DIM * n_tokens_total * 2

    # Decode 1 step 이론 (Orca 모델 기반):
    # 가중치 1× read + KV cache (step k에서 누적 k tokens) read + new KV write
    # 보수적 추정: 가중치 + KV full read (실제로는 매 step KV가 1 token씩 증가)
    decode_1step_theory_bytes = lm_bytes + kv_full_bytes / 17  # 평균 per-step KV

    result = {
        "VE_weights_GB"             : ve_bytes / 1e9,
        "LM_weights_GB"             : lm_bytes / 1e9,
        "Flow_weights_GB"           : flow_bytes / 1e9,
        "Total_model_GB"            : total_bytes / 1e9,
        "KV_cache_full_GB"          : kv_full_bytes / 1e9,
        "Decode_per_step_theory_GB" : decode_1step_theory_bytes / 1e9,
    }

    log.info("─── 이론적 DRAM 접근 바이트 ───")
    log.info(f"  VE 가중치:            {result['VE_weights_GB']:.3f} GB")
    log.info(f"  LM 가중치:            {result['LM_weights_GB']:.3f} GB")
    log.info(f"  Flow/diffusion 가중치:{result['Flow_weights_GB']:.3f} GB")
    log.info(f"  전체 모델:            {result['Total_model_GB']:.3f} GB")
    log.info(f"  KV cache (전체):      {result['KV_cache_full_GB']:.3f} GB")
    log.info(f"  Decode 1step 이론:    {result['Decode_per_step_theory_GB']:.3f} GB")

    return result


# ══════════════════════════════════════════════════════════════════════
# Layer별 가중치 크기 분석
# ══════════════════════════════════════════════════════════════════════

def _find_lm_layers(model):
    """
    LM decoder layers 리스트 탐색.
    Qwen2.5-VL 경로: model.vlm.model.layers  (표준)
    폴백:           model.vlm.language_model.layers
                    model.vlm.language_model.model.layers
    """
    if not hasattr(model, 'vlm'):
        return None

    candidates = [
        # (속성 경로 튜플)
        ('model', 'layers'),                  # vlm.model.layers (Qwen2.5-VL)
        ('language_model', 'layers'),          # vlm.language_model.layers
        ('model', 'model', 'layers'),          # vlm.model.model.layers
        ('language_model', 'model', 'layers'), # vlm.language_model.model.layers
    ]
    for path in candidates:
        obj = model.vlm
        found = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                found = False
                break
        if found and obj is not None:
            path_str = "model.vlm." + ".".join(path)
            log.info(f"  LM layers 발견: {path_str} ({len(obj)}개)")
            return obj

    log.warning("LM layers 경로를 찾지 못했습니다. "
                "다음 경로 모두 탐색 실패: vlm.model.layers / vlm.language_model.layers 등")
    return None


def analyze_layer_weights(model) -> dict:
    """
    LM 36개 레이어 각각의 가중치 세부 구성 출력.
    ncu 측정값과 대조하여 어느 파트가 DRAM BW를 얼마나 쓰는지 확인.
    """
    layers = _find_lm_layers(model)
    if layers is None:
        return {}

    result = {
        "n_layers": len(layers),
        "per_layer": [],
        "totals": {},
    }

    attn_total_mb = 0.0
    ffn_total_mb  = 0.0
    norm_total_mb = 0.0

    for idx, layer in enumerate(layers):
        layer_info = {"layer": idx}

        # Attention weights (q, k, v, o)
        attn_mb = 0.0
        if hasattr(layer, 'self_attn'):
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(layer.self_attn, proj_name, None)
                if proj is not None and hasattr(proj, 'weight'):
                    mb = proj.weight.numel() * proj.weight.element_size() / 1e6
                    attn_mb += mb
                    layer_info[proj_name] = f"{mb:.2f} MB"

        # FFN weights (gate, up, down)
        ffn_mb = 0.0
        if hasattr(layer, 'mlp'):
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(layer.mlp, proj_name, None)
                if proj is not None and hasattr(proj, 'weight'):
                    mb = proj.weight.numel() * proj.weight.element_size() / 1e6
                    ffn_mb += mb
                    layer_info[proj_name] = f"{mb:.2f} MB"

        # Norms
        norm_mb = 0.0
        for norm_name in ['input_layernorm', 'post_attention_layernorm']:
            n = getattr(layer, norm_name, None)
            if n is not None and hasattr(n, 'weight'):
                norm_mb += n.weight.numel() * n.weight.element_size() / 1e6

        layer_info["attn_total_mb"]  = round(attn_mb, 2)
        layer_info["ffn_total_mb"]   = round(ffn_mb, 2)
        layer_info["norm_mb"]        = round(norm_mb, 3)
        layer_info["layer_total_mb"] = round(attn_mb + ffn_mb + norm_mb, 2)

        result["per_layer"].append(layer_info)
        attn_total_mb += attn_mb
        ffn_total_mb  += ffn_mb
        norm_total_mb += norm_mb

    result["totals"] = {
        "attn_total_MB"     : round(attn_total_mb, 2),
        "ffn_total_MB"      : round(ffn_total_mb, 2),
        "norm_total_MB"     : round(norm_total_mb, 3),
        "lm_total_MB"       : round(attn_total_mb + ffn_total_mb + norm_total_mb, 2),
        "per_layer_avg_MB"  : round((attn_total_mb + ffn_total_mb + norm_total_mb) / len(layers), 2),
    }

    log.info("─── LM Layer별 가중치 구성 ───")
    t = result["totals"]
    log.info(f"  Attention 합계: {t['attn_total_MB']:.1f} MB "
             f"({t['attn_total_MB']/1024:.2f} GB)")
    log.info(f"  FFN 합계:       {t['ffn_total_MB']:.1f} MB "
             f"({t['ffn_total_MB']/1024:.2f} GB)")
    log.info(f"  LayerNorm 합계: {t['norm_total_MB']:.1f} MB")
    log.info(f"  LM 전체:        {t['lm_total_MB']:.1f} MB "
             f"({t['lm_total_MB']/1024:.2f} GB)")
    log.info(f"  레이어당 평균:  {t['per_layer_avg_MB']:.1f} MB")

    # Decode BW breakdown 추정 (1 step)
    decode_time_s = 0.0791  # AppendOnlyCache-C steady state
    log.info("─── Decode 1 step DRAM 이론 BW ───")
    log.info(f"  시간: {decode_time_s*1000:.1f} ms (AppendOnlyCache-C steady)")
    for name, mb in [
        ("Attention (QKV+O)", t['attn_total_MB']),
        ("FFN (gate+up+down)", t['ffn_total_MB']),
        ("KV cache (455 MB avg)", 455.0),
    ]:
        bw = (mb / 1024) / decode_time_s
        pct = bw / DRAM_BW_GBps * 100
        log.info(f"  {name:<28} {mb/1024:.2f} GB → {bw:.1f} GB/s ({pct:.0f}%)")

    return result


# ══════════════════════════════════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════════════════════════════════

def load_model():
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    log.info("모델 로드 시작 (~3-4분)...")
    t0 = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda().eval()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    log.info(f"모델 로드 완료: {elapsed:.1f}s | {total_gb:.3f} GB")
    return model


def load_inputs():
    """실제 데이터셋 입력 로드 (260513_profile_v4.py 와 동일)"""
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    # 기존 프로파일링에서 사용한 동일 clip
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    return data, helper


def prepare_model_inputs(model, data, helper) -> dict:
    """입력 데이터를 모델 입력 형식으로 변환"""
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
    log.info(f"입력 토큰 길이: {seq_len}")
    return model_inputs


# ══════════════════════════════════════════════════════════════════════
# 단계 분리 훅 (260513_profile_v4.py의 PhaseDetectorV4 와 동일 로직)
# NVTX 범위 이름은 ncu --nvtx-include 가 참조할 그대로 유지
# ══════════════════════════════════════════════════════════════════════

class PhaseSeparator:
    """
    4단계 분리 + NVTX 마킹.

    NVTX 범위 (ncu --nvtx-include 타겟):
      "Phase/Vision_Encoder"    ← VE 단계
      "Phase/LM_Prefill"        ← LM Prefill 단계
      "Phase/Decode"            ← Decode 전체
      "Decode/step_010"         ← Decode 개별 step (010 = 10번째)
      "Phase/Flow"              ← Flow 단계
    """

    def __init__(self):
        self.state        = "idle"
        self.decode_step  = 0
        self.hooks        = []

        # CUDA Event 타이머
        self.t_ve         = CUDATimer("VE")
        self.t_prefill    = CUDATimer("LM_Prefill")
        self.t_decode     = CUDATimer("Decode")
        self.t_decode_steps: list[float] = []
        self._t_step      = CUDATimer("decode_step")
        self.t_flow       = CUDATimer("Flow")

    def reset(self):
        self.state       = "idle"
        self.decode_step = 0
        self.t_ve.reset()
        self.t_prefill.reset()
        self.t_decode.reset()
        self.t_decode_steps.clear()
        self._t_step.reset()
        self.t_flow.reset()

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
            # ncu --nvtx-include "Phase/Vision_Encoder" 계층 필터를 위해
            # 이중 push: "Phase"(외부) → "Vision_Encoder"(내부)
            nvtx.range_push("Phase")
            nvtx.range_push("Vision_Encoder")
            self.t_ve.start()
            self.state = "vision"
        elif seq == 1:
            if self.state == "post_prefill":
                # ncu --nvtx-include "Phase/Decode_all" 계층 필터용 이중 push
                nvtx.range_push("Phase")
                nvtx.range_push("Decode_all")
                self.t_decode.start()
                self.state = "decode"
                self.decode_step = 1
            elif self.state == "decode":
                self.decode_step += 1
            if self.state == "decode":
                # ncu --nvtx-include "Decode/step_NNN" 계층 필터를 위해
                # 두 번 중첩 push: "Decode"(외부) → "step_NNN"(내부)
                # 단일 push("Decode/step_NNN")은 ncu가 hierarchy로 인식 못함
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

    # ── LM hook (VE/Prefill 분리) ─────────────────────────────────────
    def on_lm_pre(self, module, args, kwargs):
        seq = self._seq(args, kwargs)
        if seq is None:
            return
        if seq > 1 and self.state == "vision":
            nvtx.range_pop()  # Vision_Encoder
            nvtx.range_pop()  # Phase  (이중 pop: on_vlm_pre에서 이중 push했으므로)
            self.t_ve.stop()
            # ncu --nvtx-include "Phase/LM_Prefill" 계층 필터용 이중 push
            nvtx.range_push("Phase")
            nvtx.range_push("LM_Prefill")
            self.t_prefill.start()
            self.state = "lm_prefill"

    def on_lm_post(self, module, args, output):
        if self.state == "lm_prefill":
            nvtx.range_pop()  # LM_Prefill
            nvtx.range_pop()  # Phase  (이중 pop)
            self.t_prefill.stop()
            self.state = "post_prefill"

    # ── Decode 종료 + Flow 시작 hook ─────────────────────────────────
    def on_generate_end(self):
        if self.state == "decode":
            self.t_decode.stop()
            nvtx.range_pop()  # Decode_all
            nvtx.range_pop()  # Phase  (이중 pop)

    def on_flow_pre(self, module, args, kwargs):
        # NVTX는 action_in_proj/action_out_proj 훅에서 처리 (FlowODE/FlowStep)
        # 여기서는 타이밍만 담당 (diffusion.sample 래핑이 동작할 경우)
        self.t_flow.start()

    def on_flow_post(self, module, args, output):
        self.t_flow.stop()


def register_hooks(model, sep: PhaseSeparator) -> list:
    """
    모델 서브모듈에 훅 등록. 훅 핸들 리스트 반환.

    Alpamayo 1.5 구조 기반:
      VLM hook  : model.vlm (Qwen2.5-VL ForConditionalGeneration)
      LM hook   : model.vlm.model (Qwen2_5_VLModel, VE/LM 분리)
                  또는 model.vlm.language_model (버전 따라)
      Flow hook : model.diffusion (BaseDiffusion, sample 메서드 래핑)
                  → 실제 연산은 step_fn(action_in_proj + expert + action_out_proj)
    """
    hooks = []

    # ── VLM hook ─────────────────────────────────────────────────────
    if hasattr(model, 'vlm'):
        hooks.append(model.vlm.register_forward_pre_hook(
            lambda m, a, kw: sep.on_vlm_pre(m, a, kw),
            with_kwargs=True,
        ))
        hooks.append(model.vlm.register_forward_hook(
            lambda m, a, o: sep.on_vlm_post(m, a, o)
        ))
        log.info("  훅 등록: model.vlm (VLM pre/post)")

        # ── LM hook (VE / Prefill 분리) ────────────────────────────
        # 260513_profile_v4.py 의 patch_lm_forward 와 동일한 탐색 로직
        lm_mod = None
        for lm_attr in ('language_model', 'model'):
            cand = getattr(model.vlm, lm_attr, None)
            if cand is None:
                continue
            if hasattr(cand, 'layers'):
                lm_mod = cand
                log.info(f"  훅 등록: model.vlm.{lm_attr} (LM pre/post, 직접 layers)")
                break
            sub = getattr(cand, 'model', None)
            if sub is not None and hasattr(sub, 'layers'):
                lm_mod = cand
                log.info(f"  훅 등록: model.vlm.{lm_attr} (LM pre/post, sub.model.layers)")
                break

        if lm_mod is not None:
            hooks.append(lm_mod.register_forward_pre_hook(
                lambda m, a, kw: sep.on_lm_pre(m, a, kw),
                with_kwargs=True,
            ))
            hooks.append(lm_mod.register_forward_hook(
                lambda m, a, o: sep.on_lm_post(m, a, o)
            ))
        else:
            log.warning("  LM hook 등록 실패 — VE/Prefill 분리 불가 (폴백: vlm hook만)")

    # ── Flow hook ─────────────────────────────────────────────────────
    # 전략: model.diffusion.sample() 몽키패치는 실제 Flow 실행경로를 놓칠 수 있음.
    #        대신 nn.Module 표준 훅 API를 사용.
    #
    # Flow 1 ODE step 실행 순서:
    #   action_in_proj(noise) → expert(transformed) → action_out_proj(output)
    #
    # action_in_proj pre-hook  → push("FlowODE") + push("FlowStep")  ← step 시작
    # action_out_proj post-hook → pop("FlowStep") + pop("FlowODE")   ← step 종료
    #
    # ★ "Phase"를 쓰지 않는 이유:
    #   Flow 실행 시점에는 Phase(Decode용)/Decode_all이 NVTX 스택에 살아있다.
    #   on_generate_end()가 sample_trajectories... 반환 후에 호출되기 때문이다.
    #   → "Phase"를 또 push하면 stack = Phase(Decode)/Decode_all/Phase(Flow)/Flow
    #   → ncu "Phase/Flow" 필터는 FIRST Phase의 direct child = Decode_all 확인 → 매칭 실패
    #   → 충돌 없는 고유 이름 "FlowODE/FlowStep" 사용
    #
    # 결과: 65 ODE step × 1쌍 = 65개 FlowODE/FlowStep 범위
    # ncu --nvtx-include "FlowODE/FlowStep" 로 65 step 전체 커널 캡처
    # timing: action_in_proj 첫 호출에서 t_flow.start(), diffusion 완료 후 stop()
    #
    # 타이밍을 위해 model.diffusion.sample 래핑은 별도 유지.

    flow_nvtx_registered = False

    # 1) action_in_proj + action_out_proj 으로 NVTX 등록 (우선순위)
    if hasattr(model, 'action_in_proj') and hasattr(model, 'action_out_proj'):
        _flow_step = [0]  # ODE step 카운터

        def _flow_step_start(m, a, kw):
            if _flow_step[0] == 0:
                sep.t_flow.start()  # 첫 step에서만 전체 타이머 시작
            _flow_step[0] += 1
            nvtx.range_push("FlowODE")   # ← Phase 충돌 방지용 고유 이름
            nvtx.range_push("FlowStep")

        def _flow_step_end(m, a, o):
            nvtx.range_pop()  # FlowStep
            nvtx.range_pop()  # FlowODE

        hooks.append(model.action_in_proj.register_forward_pre_hook(
            _flow_step_start, with_kwargs=True
        ))
        hooks.append(model.action_out_proj.register_forward_hook(_flow_step_end))
        log.info("  훅 등록: action_in_proj(pre)+action_out_proj(post) → FlowODE/FlowStep NVTX")
        flow_nvtx_registered = True

    # 2) expert 단독 훅 (action_in/out_proj 없을 때 폴백)
    if not flow_nvtx_registered and hasattr(model, 'expert'):
        def _expert_start(m, a, kw):
            nvtx.range_push("FlowODE")
            nvtx.range_push("FlowStep")
        def _expert_end(m, a, o):
            nvtx.range_pop()  # FlowStep
            nvtx.range_pop()  # FlowODE
        hooks.append(model.expert.register_forward_pre_hook(_expert_start, with_kwargs=True))
        hooks.append(model.expert.register_forward_hook(_expert_end))
        log.info("  훅 등록: model.expert(pre/post) → FlowODE/FlowStep NVTX (폴백)")
        flow_nvtx_registered = True

    # 3) 타이밍용 diffusion.sample 래핑 (NVTX 없이 t_flow 측정만)
    if hasattr(model, 'diffusion') and hasattr(model.diffusion, 'sample'):
        orig_sample = model.diffusion.sample

        def _timed_sample(*args, **kwargs):
            # action_in_proj 훅이 없을 때만 t_flow.start()
            if not flow_nvtx_registered:
                sep.t_flow.start()
            result = orig_sample(*args, **kwargs)
            sep.t_flow.stop()
            return result

        model.diffusion.sample = _timed_sample
        hooks.append(_FlowHookHandle(model.diffusion, orig_sample))
        log.info("  훅 등록: model.diffusion.sample (타이밍 전용 래핑)")

    if not flow_nvtx_registered:
        log.warning("  Flow NVTX 훅 등록 실패 — action_in_proj/expert/action_out_proj 없음")

    return hooks


class _FlowHookHandle:
    """model.diffusion.sample 래핑 해제용 핸들 (remove() 인터페이스)"""
    def __init__(self, diffusion_mod, orig_fn):
        self._mod  = diffusion_mod
        self._orig = orig_fn

    def remove(self):
        self._mod.sample = self._orig


# ══════════════════════════════════════════════════════════════════════
# 추론 실행 공통 함수
# ══════════════════════════════════════════════════════════════════════

def run_inference(model, model_inputs: dict, sep: PhaseSeparator, label: str):
    """1회 추론 실행 + 단계별 CUDA Event 타이밍"""
    sep.reset()
    nvtx.range_push(label)

    t_wall = time.perf_counter()
    # torch.autocast 필수:
    #   model.action_in_proj 가중치 = BF16 (config.keep_same_dtype=True)
    #   diffusion noise x = Float32 (기본값)
    #   → autocast 없으면 dtype mismatch RuntimeError 발생
    #   260513_profile_v4.py 와 동일하게 autocast 적용
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            return_extra=True,
        )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t_wall) * 1000.0

    sep.on_generate_end()
    nvtx.range_pop()  # label

    return wall_ms


# ══════════════════════════════════════════════════════════════════════
# Mode 1: 타이밍 전용 (ncu 없이 빠르게)
# ══════════════════════════════════════════════════════════════════════

def mode_timing(model, model_inputs: dict, args):
    """
    CUDA Event로 4단계 정확한 wall-clock 측정.
    ncu 없이 직접 실행 가능. 결과를 JSON 저장.
    """
    log.info("═══ Mode: timing ═══")

    theoretical = compute_theoretical_bytes(model)
    layer_info  = analyze_layer_weights(model)

    sep   = PhaseSeparator()
    hooks = register_hooks(model, sep)

    # Warmup
    log.info(f"Warmup {args.warmup}회...")
    for i in range(args.warmup):
        run_inference(model, model_inputs, sep, f"Warmup/run_{i+1:02d}")
        log.info(f"  warmup {i+1}/{args.warmup} 완료  "
                 f"VE={sep.t_ve.ms():.0f}ms  "
                 f"Prefill={sep.t_prefill.ms():.0f}ms  "
                 f"Decode={sep.t_decode.ms():.0f}ms  "
                 f"Flow={sep.t_flow.ms():.0f}ms")

    # 측정
    log.info(f"측정 {args.runs}회...")
    records = []
    for i in range(args.runs):
        wall_ms = run_inference(model, model_inputs, sep, f"Measure/run_{i+1:02d}")

        ve_ms      = sep.t_ve.ms()
        prefill_ms = sep.t_prefill.ms()
        decode_ms  = sep.t_decode.ms()
        flow_ms    = sep.t_flow.ms()
        steps      = sep.t_decode_steps.copy()
        n_steps    = len(steps)

        # Decode step별 통계
        steps_arr = np.array(steps) if steps else np.array([0.0])
        step_mean = float(steps_arr.mean())
        step_med  = float(np.median(steps_arr))
        step_ss   = float(steps_arr[3:].mean()) if len(steps_arr) > 3 else step_mean

        # 이론 대역폭 활용률 (decode steady-state 기준)
        lm_gb         = theoretical["LM_weights_GB"]
        kv_gb         = theoretical["KV_cache_full_GB"]
        theory_gb_step = lm_gb + kv_gb / max(n_steps, 1)
        actual_bw_theory = theory_gb_step / (step_ss / 1000.0) if step_ss > 0 else 0.0
        theory_util_pct  = actual_bw_theory / DRAM_BW_GBps * 100.0

        record = {
            "run"          : i + 1,
            "wall_ms"      : round(wall_ms, 1),
            "VE_ms"        : round(ve_ms, 1),
            "LM_Prefill_ms": round(prefill_ms, 1),
            "Decode_ms"    : round(decode_ms, 1),
            "Flow_ms"      : round(flow_ms, 1),
            "decode_n_steps"       : n_steps,
            "decode_step_mean_ms"  : round(step_mean, 2),
            "decode_step_median_ms": round(step_med, 2),
            "decode_step_ss_ms"    : round(step_ss, 2),
            "decode_theory_bw_GBps": round(actual_bw_theory, 1),
            "decode_theory_util_pct": round(theory_util_pct, 1),
        }
        records.append(record)

        print(f"\n[RUN {i+1}]")
        print(f"  VE              : {ve_ms:7.0f} ms")
        print(f"  LM Prefill      : {prefill_ms:7.0f} ms")
        print(f"  Decode          : {decode_ms:7.0f} ms  ({n_steps} steps)")
        print(f"    per-step mean : {step_mean:7.2f} ms")
        print(f"    per-step SS   : {step_ss:7.2f} ms  (step 4+, steady-state)")
        print(f"    Theory BW     : {actual_bw_theory:7.1f} GB/s  "
              f"= {theory_util_pct:.0f}% of {DRAM_BW_GBps} GB/s peak")
        print(f"  Flow            : {flow_ms:7.0f} ms")
        print(f"  Wall total      : {wall_ms:7.0f} ms")
        print(f"  ※ 이 BW는 'LM weights + KV cache' 이론치 기준.")
        print(f"    실제 DRAM bytes는 ncu 실행(Phase 2) 결과와 비교 필요.")

    # 저장
    output = {
        "mode"       : "timing",
        "dram_bw_peak_GBps": DRAM_BW_GBps,
        "theoretical": theoretical,
        "layer_info" : layer_info.get("totals", {}),
        "runs"       : records,
    }
    out_path = OUT / "timing_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"타이밍 결과 저장: {out_path}")

    # 훅 제거
    for h in hooks:
        h.remove()

    return output


# ══════════════════════════════════════════════════════════════════════
# Mode 2: ncu 래핑용 (warmup 1회 + 측정 1회만 실행)
# ══════════════════════════════════════════════════════════════════════

def mode_ncu_single_run(model, model_inputs: dict, args):
    """
    ncu가 DRAM bytes를 캡처하기 위한 최소 실행.
    - warmup 1회 (JIT 컴파일 완료)
    - 측정 1회 (ncu가 NVTX 범위 내 커널을 프로파일)

    ncu 실행 시 이 스크립트를 래핑하면, NVTX 범위 이름으로
    특정 단계만 선택적으로 캡처할 수 있다:

      ncu --nvtx --nvtx-include "Phase/Decode" \\
          --metrics dram__bytes_read.sum,... \\
          --csv python3 이_스크립트.py --mode ncu_single_run

    단계별 NVTX 범위 (이중 push 구조, ncu hierarchy 필터 호환):
      "Phase/Vision_Encoder"  ← VE       push("Phase") + push("Vision_Encoder")
      "Phase/LM_Prefill"      ← Prefill  push("Phase") + push("LM_Prefill")
      "Phase/Decode_all"      ← Decode 전체 (모든 step)
      "Decode/step_010"       ← Decode 10번째 step (steady-state)
      "Phase/Flow"            ← Flow     push("Phase") + push("Flow")

    ncu 명령 예시:
      ncu --nvtx --nvtx-include "Phase/Vision_Encoder" --metrics ... --csv python3 ... --mode ncu_single_run
      ncu --nvtx --nvtx-include "Phase/LM_Prefill"     --metrics ... --csv python3 ... --mode ncu_single_run
      ncu --nvtx --nvtx-include "Decode/step_010"      --metrics ... --csv python3 ... --mode ncu_single_run
      ncu --nvtx --nvtx-include "Phase/Flow"           --metrics ... --csv python3 ... --mode ncu_single_run
    """
    log.info("═══ Mode: ncu_single_run ═══")
    log.info("※ 이 모드는 ncu --nvtx 래핑 하에서 실행되어야 의미가 있습니다.")

    sep   = PhaseSeparator()
    hooks = register_hooks(model, sep)

    # Warmup: JIT 컴파일 트리거, ncu는 이 구간을 캡처하지 않음
    log.info("Warmup 1회 (JIT 트리거)...")
    run_inference(model, model_inputs, sep, "Warmup/run_01")
    log.info(f"  Warmup 완료 - Decode {sep.t_decode.ms():.0f} ms")

    # 실제 측정: ncu가 NVTX 범위 내 커널 캡처
    log.info("ncu 캡처 대상 추론 실행...")
    nvtx.range_push("ncu_capture_run")
    wall_ms = run_inference(model, model_inputs, sep, "Measure/run_01")
    nvtx.range_pop()

    log.info(f"실행 완료 - 총 {wall_ms:.0f} ms")
    log.info("  VE:      " + str(round(sep.t_ve.ms(), 1)) + " ms")
    log.info("  Prefill: " + str(round(sep.t_prefill.ms(), 1)) + " ms")
    log.info("  Decode:  " + str(round(sep.t_decode.ms(), 1)) + " ms")
    log.info("  Flow:    " + str(round(sep.t_flow.ms(), 1)) + " ms")

    for h in hooks:
        h.remove()


# ══════════════════════════════════════════════════════════════════════
# Mode 3: 이론 추정값 빠른 출력 (ncu 없이)
# ══════════════════════════════════════════════════════════════════════

def mode_estimate_only(model):
    """
    모델을 추론 없이 이론 DRAM 접근량과 레이어 구조만 분석.
    ncu 없이 빠른 확인용.
    """
    log.info("═══ Mode: estimate_only ═══")

    theoretical = compute_theoretical_bytes(model)
    layer_info  = analyze_layer_weights(model)

    # 이론 BW 표 출력
    print("\n" + "="*72)
    print("단계별 이론 DRAM 접근량 (실측 시간 + 이론 가중치 기준)")
    print("="*72)
    print(f"{'단계':<16} {'이론 GB':>9} {'측정 시간 ms':>13} {'이론 BW GB/s':>14} {'Peak 대비':>9}")
    print("-"*72)

    # 실측 시간 (AppendOnlyCache-C, CLAUDE.md 기준)
    # Flow = expert + action_in_proj + action_out_proj (model.diffusion 제외)
    reference = {
        "VE"         : (theoretical["VE_weights_GB"],           728.0),
        "LM_Prefill" : (theoretical["LM_weights_GB"],           1423.0),
        "Decode_step": (theoretical["Decode_per_step_theory_GB"], 79.1),
        "Flow"       : (theoretical["Flow_weights_GB"],          870.0),
    }

    for stage, (gb, ms) in reference.items():
        bw   = gb / (ms / 1000.0) if ms > 0 else 0.0
        pct  = bw / DRAM_BW_GBps * 100.0
        print(f"{stage:<16} {gb:>9.2f} {ms:>13.1f} {bw:>14.1f} {pct:>8.0f}%")

    print("="*72)
    print(f"Peak DRAM BW = {DRAM_BW_GBps} GB/s (Thor LPDDR5X)")
    print()
    print("※ Prefill: BW > Peak → compute-bound (가중치 재사용으로 실제 DRAM < 이론)")
    print("※ Decode:  BW ~ Peak → memory-bound  (가중치 1× 접근 per step)")
    print("※ 실제값: ncu dram__bytes_read.sum + dram__bytes_write.sum 으로 확인")

    out_path = OUT / "estimate_only.json"
    with open(out_path, "w") as f:
        json.dump({"theoretical": theoretical, "layer_totals": layer_info.get("totals", {})}, f, indent=2)
    log.info(f"추정값 저장: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Alpamayo DRAM 대역폭 측정")
    p.add_argument(
        "--mode",
        choices=["timing", "ncu_single_run", "estimate_only"],
        default="timing",
        help=(
            "timing: CUDA Event 타이밍만 (빠름, ncu 불필요) | "
            "ncu_single_run: ncu 래핑용 1회 실행 | "
            "estimate_only: 모델 로드 후 이론값만 계산"
        ),
    )
    p.add_argument("--warmup", type=int, default=1, help="Warmup 횟수 (timing 모드)")
    p.add_argument("--runs",   type=int, default=2, help="측정 횟수 (timing 모드)")
    return p.parse_args()


def main():
    args  = parse_args()
    model = load_model()

    if args.mode == "estimate_only":
        mode_estimate_only(model)
        return

    log.info("입력 데이터 로드...")
    data, helper     = load_inputs()
    model_inputs     = prepare_model_inputs(model, data, helper)

    if args.mode == "timing":
        mode_timing(model, model_inputs, args)
    elif args.mode == "ncu_single_run":
        mode_ncu_single_run(model, model_inputs, args)


if __name__ == "__main__":
    main()
