"""
260515_bw_allphase.py  --  All-Phase CUDA Events Bandwidth Profiler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v4 대비 핵심 추가:
  - 모델 컴포넌트별 파라미터 크기 자동 측정 (Vision / LM / Flow)
  - CUDA Events BW를 4개 Phase 전체에 적용

BW 계산 공식 (이론적 근거):
  Vision    BW = vision_gb   × 1 pass  / vision_ms          (ViT 가중치 1회 읽기)
  LM Prefill BW = lm_gb      × 1 pass  / lm_prefill_ms      (Transformer 1 fwd pass)
  Decode     BW = model_gb   × n_tok   / decode_ms           (매 token마다 전체 재로드)
  Flow       BW = flow_gb    × n_euler / flow_ms             (Euler step × flow 가중치)

  ★ Decode는 autoregressive → KV cache 있어도 매 step마다
     transformer weights 전체 스트리밍 → 가장 BW-bound

측정 방법: CUDA Events (microsecond-accurate GPU timer)
의존 없음: tegrastats / ncu / bwmon 불필요

출력:
  profiling_results/260515_bw/allphase_bw.json
  profiling_results/260515_bw/allphase_bw.md
  profiling_results/260515_bw/figures/

Usage:
  python scripts/profiling/260515_bw_allphase.py
  python scripts/profiling/260515_bw_allphase.py --warmup 1 --runs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path("profiling_results/260515_bw")
FIGD = OUT / "figures"
for d in [OUT, FIGD]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DRAM_BW  = 273.0   # GB/s  Jetson AGX Thor LPDDR5X
MODEL_GB = 22.157  # bf16 전체 모델  (파라미터 실측값으로 덮어씀)

# 컴포넌트 크기 (model load 후 measure_component_gb()가 채움)
COMP_GB: dict[str, float] = {
    "total":  MODEL_GB,
    "vision": 0.0,
    "lm":     0.0,
    "flow":   0.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# 컴포넌트 크기 측정
# ══════════════════════════════════════════════════════════════════════════════

def measure_component_gb(model) -> dict[str, float]:
    """
    모델 서브모듈별 파라미터 크기를 GB 단위로 반환.

    Alpamayo 1.5 실측 구조:
      model.vlm       = 17.596 GB  (Vision Encoder + LM Transformer + embeddings)
      model.expert    =  4.558 GB  ← 실제 Flow/Action Expert 가중치
      model.diffusion =  0.000 GB  ← 빈 wrapper (샘플러만, 파라미터 없음)
      model.action_in_proj = 0.003 GB
    """
    def _gb(m) -> float:
        return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e9

    result: dict[str, float] = {}
    result["total"] = _gb(model)

    # ── Flow (Action Expert) ──────────────────────────────────────────────────
    # model.diffusion은 파라미터 없는 wrapper → model.expert가 실제 가중치
    # 우선순위: expert > diffusion(파라미터 있을 때만) > flow_model > action_expert
    for attr in ("expert", "flow_model", "action_expert"):
        if hasattr(model, attr):
            gb = _gb(getattr(model, attr))
            if gb > 0.01:
                result["flow"] = gb
                print(f"  [컴포넌트] flow  = model.{attr}  ({gb:.3f} GB)")
                break
    # diffusion이 wrapper인 경우 fallback: total - vlm - projections
    if result.get("flow", 0.0) < 0.01 and hasattr(model, "diffusion"):
        gb = _gb(getattr(model, "diffusion"))
        if gb > 0.01:
            result["flow"] = gb
            print(f"  [컴포넌트] flow  = model.diffusion  ({gb:.3f} GB)")

    # ── VLM 전체 ────────────────────────────────────────────────────────────
    vlm = getattr(model, "vlm", None)
    if vlm is None:
        result.setdefault("flow", 0.0)
        result["vision"] = 0.0
        result["lm"]     = result["total"] - result.get("flow", 0.0)
        return result

    result["vlm"] = _gb(vlm)

    # ── Vision Encoder ───────────────────────────────────────────────────────
    for attr in ("visual", "vision_encoder", "vision_model",
                 "image_encoder", "encoder", "patch_embed"):
        if hasattr(vlm, attr):
            mod = getattr(vlm, attr)
            if hasattr(mod, "parameters"):
                v_gb = _gb(mod)
                if v_gb > 0.1:
                    result["vision"] = v_gb
                    print(f"  [컴포넌트] vision = vlm.{attr}  ({v_gb:.3f} GB)")
                    break

    # ── Language Model ────────────────────────────────────────────────────────
    for attr in ("language_model", "model", "decoder", "transformer"):
        if hasattr(vlm, attr):
            mod = getattr(vlm, attr)
            if hasattr(mod, "layers"):
                result["lm"] = _gb(mod)
                print(f"  [컴포넌트] lm    = vlm.{attr}  ({result['lm']:.3f} GB)")
                break
            sub = getattr(mod, "model", None)
            if sub is not None and hasattr(sub, "layers"):
                result["lm"] = _gb(sub)
                print(f"  [컴포넌트] lm    = vlm.{attr}.model  ({result['lm']:.3f} GB)")
                break

    # ── 미측정 항목 채우기 ─────────────────────────────────────────────────
    result.setdefault("flow",   0.0)
    result.setdefault("vision", 0.0)
    if "lm" not in result:
        result["lm"] = result.get("vlm", result["total"]) - result["vision"]
        print(f"  [컴포넌트] lm    = vlm - vision (추정: {result['lm']:.3f} GB)")

    # flow가 여전히 0이면 total - vlm - misc로 추정
    if result["flow"] < 0.01 and "vlm" in result:
        result["flow"] = result["total"] - result["vlm"]
        print(f"  [컴포넌트] flow  = total - vlm (추정: {result['flow']:.3f} GB)")

    # ── vlm_gb 저장 (Decode BW 분모로 사용) ──────────────────────────────────
    # Decode 단계는 vlm.generate() 안에서만 실행됨 → model.expert 로드 없음
    # → Decode BW 분모 = vlm_gb (total이 아님!)
    result["vlm_gb"] = result.get("vlm", result["total"] - result["flow"])

    # ── 검증 출력 ────────────────────────────────────────────────────────────
    vlm_gb = result["vlm_gb"]
    print(f"  [컴포넌트 요약]")
    print(f"    total  = {result['total']:.3f} GB")
    print(f"    vlm    = {vlm_gb:.3f} GB  ({vlm_gb/result['total']*100:.1f}%)")
    print(f"      vision = {result['vision']:.3f} GB  "
          f"({result['vision']/result['total']*100:.1f}%)")
    print(f"      lm     = {result['lm']:.3f} GB  "
          f"({result['lm']/result['total']*100:.1f}%)")
    print(f"    flow   = {result['flow']:.3f} GB  "
          f"({result['flow']/result['total']*100:.1f}%)")
    print(f"  [중요] Decode BW 분모 = vlm_gb ({vlm_gb:.3f} GB)")
    print(f"         (Decode 중 model.expert 미로드 → total 22GB 사용하면 과대추정)")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CUDA Event 기반 타이머
# ══════════════════════════════════════════════════════════════════════════════

class CUDATimer:
    def __init__(self, name: str):
        self.name = name
        self._reset_events()
        self._started = False
        self._stopped = False

    def _reset_events(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self):
        torch.cuda.synchronize()
        self._s.record()
        self._started = True
        self._stopped = False

    def stop(self):
        if not self._started:
            return
        self._e.record()
        torch.cuda.synchronize()
        self._stopped = True

    def ms(self) -> float:
        if self._started and self._stopped:
            return self._s.elapsed_time(self._e)
        return 0.0

    def reset(self):
        self._reset_events()
        self._started = False
        self._stopped = False


def gpu_mem_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9

def gpu_peak_gb() -> float:
    return torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1e9


# ══════════════════════════════════════════════════════════════════════════════
# Phase 상태 머신 (v4와 동일)
# ══════════════════════════════════════════════════════════════════════════════

class PhaseDetectorV4:
    IDLE         = "idle"
    VISION       = "vision"
    LM_PREFILL   = "lm_prefill"
    POST_PREFILL = "post_prefill"
    DECODE       = "decode"

    def __init__(self):
        self._state       = self.IDLE
        self._decode_step = 0
        self._calls_vlm   = []
        self._calls_lm    = []

        self.t_vision     = CUDATimer("vision")
        self.t_lm_prefill = CUDATimer("lm_prefill")
        self.t_decode     = CUDATimer("decode")
        self.mem          = {}
        self._lm_patched  = False

    def reset(self):
        self._state       = self.IDLE
        self._decode_step = 0
        self._calls_vlm   = []
        self._calls_lm    = []
        self.t_vision.reset()
        self.t_lm_prefill.reset()
        self.t_decode.reset()
        self.mem = {}

    def _seq(self, args, kwargs) -> int | None:
        for src in [kwargs.get("input_ids"),
                    kwargs.get("hidden_states"),
                    kwargs.get("inputs_embeds"),
                    *(a for a in args if isinstance(a, torch.Tensor))]:
            if src is None:
                continue
            if isinstance(src, torch.Tensor):
                if src.ndim == 2:
                    return int(src.shape[-1])
                if src.ndim == 3:
                    return int(src.shape[1])
        return None

    def on_vlm_before(self, args, kwargs):
        seq = self._seq(args, kwargs)
        if seq is None:
            return
        self._calls_vlm.append(seq)

        if seq > 1:
            if self._state == self.IDLE:
                self.mem["vision_before_gb"] = gpu_mem_gb()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.nvtx.range_push("Phase/Vision_Encoder")
                self.t_vision.start()
                self._state = self.VISION
        elif seq == 1:
            if self._state == self.POST_PREFILL:
                self.mem["decode_before_gb"] = gpu_mem_gb()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.nvtx.range_push("Phase/Decode")
                self.t_decode.start()
                self._state = self.DECODE
                self._decode_step = 1
            elif self._state == self.DECODE:
                self._decode_step += 1
            if self._state == self.DECODE:
                torch.cuda.nvtx.range_push(
                    f"Decode/step_{self._decode_step:03d}")

    def on_vlm_after(self):
        if self._state == self.DECODE:
            torch.cuda.nvtx.range_pop()

    def on_lm_before(self, args, kwargs):
        seq = self._seq(args, kwargs)
        if seq is None:
            return
        self._calls_lm.append(seq)
        self._lm_patched = True

        if seq > 1 and self._state == self.VISION:
            self.t_vision.stop()
            torch.cuda.nvtx.range_pop()
            self.mem["vision_after_gb"]  = gpu_mem_gb()
            self.mem["vision_peak_gb"]   = gpu_peak_gb()
            torch.cuda.reset_peak_memory_stats()

            self.mem["lm_prefill_before_gb"] = gpu_mem_gb()
            torch.cuda.nvtx.range_push("Phase/LM_Prefill")
            self.t_lm_prefill.start()
            self._state = self.LM_PREFILL

    def on_lm_after(self):
        if self._state == self.LM_PREFILL:
            self.t_lm_prefill.stop()
            torch.cuda.nvtx.range_pop()
            self.mem["lm_prefill_after_gb"] = gpu_mem_gb()
            self.mem["lm_prefill_peak_gb"]  = gpu_peak_gb()
            self._state = self.POST_PREFILL

    def end_generate(self):
        if self._state == self.DECODE:
            self.t_decode.stop()
            torch.cuda.nvtx.range_pop()
            self.mem["decode_after_gb"] = gpu_mem_gb()
            self.mem["decode_peak_gb"]  = gpu_peak_gb()
        elif self._state == self.LM_PREFILL:
            self.t_lm_prefill.stop()
            torch.cuda.nvtx.range_pop()
        elif self._state == self.VISION:
            self.t_vision.stop()
            torch.cuda.nvtx.range_pop()
        self._state = self.IDLE

    def diagnostics(self) -> str:
        n_vlm_pf = sum(1 for s in self._calls_vlm if s > 1)
        n_vlm_dc = sum(1 for s in self._calls_vlm if s == 1)
        n_lm_pf  = sum(1 for s in self._calls_lm  if s > 1)
        n_lm_dc  = sum(1 for s in self._calls_lm  if s == 1)
        return (
            f"    vlm.forward: prefill={n_vlm_pf}  decode={n_vlm_dc}\n"
            f"     lm.forward: prefill={n_lm_pf}   decode={n_lm_dc}"
            + ("  [lm patched]" if self._lm_patched else "  [lm NOT patched]")
        )

    @property
    def split_ok(self) -> bool:
        return self.t_vision.ms() > 0 and self.t_lm_prefill.ms() > 0


# ══════════════════════════════════════════════════════════════════════════════
# 패치 함수 (v4와 동일)
# ══════════════════════════════════════════════════════════════════════════════

def patch_vlm_forward(vlm, det: PhaseDetectorV4) -> bool:
    if not hasattr(vlm, "forward"):
        return False
    orig = vlm.forward

    def _patched(*args, **kwargs):
        det.on_vlm_before(args, kwargs)
        result = orig(*args, **kwargs)
        det.on_vlm_after()
        return result

    vlm.forward = _patched
    print("  [패치] vlm.forward")
    return True


def patch_lm_forward(vlm, det: PhaseDetectorV4) -> bool:
    candidates = []
    for attr in ("language_model", "model"):
        mod = getattr(vlm, attr, None)
        if mod is not None and hasattr(mod, "forward"):
            candidates.append((f"vlm.{attr}", mod))
        if mod is not None:
            sub = getattr(mod, "model", None)
            if sub is not None and hasattr(sub, "forward") \
                    and hasattr(sub, "layers"):
                candidates.append((f"vlm.{attr}.model", sub))
            if hasattr(mod, "layers"):
                break

    for path, mod in candidates:
        orig = mod.forward

        def _patched(*args, _orig=orig, **kwargs):
            det.on_lm_before(args, kwargs)
            result = _orig(*args, **kwargs)
            det.on_lm_after()
            return result

        mod.forward = _patched
        print(f"  [패치] {path}.forward")
        return True

    print("  [경고] lm.forward 패치 실패")
    return False


def wrap_generate(vlm, det: PhaseDetectorV4,
                  t_vlm: CUDATimer, tok_n: list[int],
                  input_tok_len: int):
    orig = vlm.generate.__func__

    def _gen(self_v, *args, **kwargs):
        det.reset()
        tok_n[0] = 0
        t_vlm.reset()
        t_vlm.start()
        torch.cuda.nvtx.range_push("Phase/VLM_Generate")
        result = orig(self_v, *args, **kwargs)
        torch.cuda.nvtx.range_pop()
        t_vlm.stop()
        det.end_generate()

        if hasattr(result, "sequences"):
            gl = result.sequences.shape[-1]
        elif isinstance(result, torch.Tensor):
            gl = result.shape[-1]
        else:
            gl = 0
        tok_n[0] = max(0, gl - input_tok_len)
        return result

    vlm.generate = types.MethodType(_gen, vlm)
    print("  [패치] vlm.generate")


def wrap_diffusion(diffusion, t_flow: CUDATimer, euler_n: list[int]):
    orig_euler = diffusion._euler

    def _cnt_euler(self_d, *args, **kwargs):
        n = kwargs.get("inference_step") \
            or getattr(self_d, "num_inference_steps", 1)
        euler_n[0] += int(n)
        torch.cuda.nvtx.range_push(f"Flow/Euler_x{euler_n[0]}")
        result = orig_euler(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return result

    diffusion._euler = types.MethodType(_cnt_euler, diffusion)

    orig_sample = diffusion.sample

    def _timed_sample(*args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.nvtx.range_push("Phase/Flow")
        t_flow.start()
        result = orig_sample(*args, **kwargs)
        t_flow.stop()
        torch.cuda.nvtx.range_pop()
        return result

    diffusion.sample = _timed_sample
    print("  [패치] diffusion.sample + _euler")


# ══════════════════════════════════════════════════════════════════════════════
# 측정 결과 데이터 클래스  (BW 전 Phase 포함)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    label:         str
    is_warmup:     bool

    wall_ms:       float
    vlm_ms:        float
    vision_ms:     float
    lm_prefill_ms: float
    decode_ms:     float
    flow_ms:       float

    n_tok:         int
    n_euler:       int
    split_ok:      bool
    mem:           dict = field(default_factory=dict)

    @property
    def prefill_ms(self) -> float:
        if self.split_ok:
            return self.vision_ms + self.lm_prefill_ms
        return self.vlm_ms - self.decode_ms

    # ── BW (CUDA Events 기반, 컴포넌트 크기 이용) ─────────────────────────

    @property
    def vision_bw_GBps(self) -> float:
        """vision_gb / vision_ms  (ViT 가중치 1회 스트리밍)"""
        if self.vision_ms > 0 and COMP_GB["vision"] > 0.01:
            return COMP_GB["vision"] / (self.vision_ms / 1000.0)
        return 0.0

    @property
    def lm_prefill_bw_GBps(self) -> float:
        """lm_gb / lm_prefill_ms  (LM 가중치 1회 forward pass)"""
        if self.lm_prefill_ms > 0 and COMP_GB["lm"] > 0.01:
            return COMP_GB["lm"] / (self.lm_prefill_ms / 1000.0)
        return 0.0

    @property
    def decode_bw_GBps(self) -> float:
        """(vlm_gb - vision_gb) × n_tok / decode_ms

        Decode step (seq=1)에서 실제 로드되는 가중치:
          - vlm.language_model (15.168 GB)   ← 항상 로드
          - embed + lm_head (~1.275 GB)      ← 항상 로드
          - vlm.visual (1.153 GB)            ← 미로드 (KV cache 사용)

        따라서 분모 = vlm_gb - vision_gb = 16.443 GB
        (vision 포함 시 과대추정 +7%)
        """
        if self.decode_ms > 0 and self.n_tok > 0:
            decode_gb = (COMP_GB.get("vlm_gb", COMP_GB["total"])
                         - COMP_GB.get("vision", 0.0))
            return decode_gb * self.n_tok / (self.decode_ms / 1000.0)
        return 0.0

    @property
    def flow_bw_GBps(self) -> float:
        """flow_gb × n_euler / flow_ms
        flow_gb = model.expert 파라미터 (4.558 GB)
        n_euler = Flow Euler 스텝 수 (실측, 보통 10)
        """
        if self.flow_ms > 0 and COMP_GB["flow"] > 0.01 and self.n_euler > 0:
            return COMP_GB["flow"] * self.n_euler / (self.flow_ms / 1000.0)
        return 0.0

    def mbu(self, bw: float) -> float:
        return bw / DRAM_BW * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 단일 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_one(label: str, is_warmup: bool,
            model, model_inputs,
            det: PhaseDetectorV4,
            t_vlm: CUDATimer, t_flow: CUDATimer,
            tok_n: list[int], euler_n: list[int]) -> RunResult:

    det.reset()
    euler_n[0] = 0
    tok_n[0]   = 0
    t_vlm.reset()
    t_flow.reset()
    torch.cuda.synchronize()

    run_nvtx = f"Warmup/{label}" if is_warmup else f"Measure/{label}"
    torch.cuda.nvtx.range_push(run_nvtx)
    torch.cuda.nvtx.range_push("Inference/Total")

    t_wall = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98, temperature=0.6,
            num_traj_samples=1, return_extra=True,
        )
    wall_ms = (time.perf_counter() - t_wall) * 1000.0

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()

    r = RunResult(
        label=label,
        is_warmup=is_warmup,
        wall_ms=wall_ms,
        vlm_ms=t_vlm.ms(),
        vision_ms=det.t_vision.ms(),
        lm_prefill_ms=det.t_lm_prefill.ms(),
        decode_ms=det.t_decode.ms(),
        flow_ms=t_flow.ms(),
        n_tok=tok_n[0],
        n_euler=euler_n[0],
        split_ok=det.split_ok,
        mem=dict(det.mem),
    )

    tag = "WARMUP" if is_warmup else "  RUN "
    print(f"\n  [{tag} {label}]")
    if r.split_ok:
        vis_bw = r.vision_bw_GBps
        lm_bw  = r.lm_prefill_bw_GBps
        dec_bw = r.decode_bw_GBps
        fl_bw  = r.flow_bw_GBps
        print(f"    Vision    : {r.vision_ms:>7.0f} ms  "
              f"BW={vis_bw:>6.1f} GB/s  ({r.mbu(vis_bw):>4.0f}% MBU)")
        print(f"    LM Prefill: {r.lm_prefill_ms:>7.0f} ms  "
              f"BW={lm_bw:>6.1f} GB/s  ({r.mbu(lm_bw):>4.0f}% MBU)")
        print(f"    Decode    : {r.decode_ms:>7.0f} ms  "
              f"BW={dec_bw:>6.1f} GB/s  ({r.mbu(dec_bw):>4.0f}% MBU)  "
              f"[{r.n_tok} tok]")
        print(f"    Flow      : {r.flow_ms:>7.0f} ms  "
              f"BW={fl_bw:>6.1f} GB/s  ({r.mbu(fl_bw):>4.0f}% MBU)  "
              f"[{r.n_euler} euler]")
    else:
        print(f"    Prefill(통합): {r.prefill_ms:>7.0f} ms  [분리 실패]")
        print(f"    Decode    : {r.decode_ms:>7.0f} ms  "
              f"BW={r.decode_bw_GBps:>6.1f} GB/s  "
              f"({r.mbu(r.decode_bw_GBps):>4.0f}% MBU)  [{r.n_tok} tok]")
        print(f"    Flow      : {r.flow_ms:>7.0f} ms  "
              f"BW={r.flow_bw_GBps:>6.1f} GB/s  [{r.n_euler} euler]")
    print(f"    VLM       : {r.vlm_ms:>7.0f} ms  |  Wall: {r.wall_ms:.0f} ms")
    print(det.diagnostics())

    return r


# ══════════════════════════════════════════════════════════════════════════════
# 메인 프로파일링
# ══════════════════════════════════════════════════════════════════════════════

def run_profiling(warmup: int = 1, runs: int = 2):
    print("\n[1/4] 모델 로드...")
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    t0 = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16,
    ).cuda().eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    # ── 컴포넌트 크기 측정 ──────────────────────────────────────────────────
    print("\n  [컴포넌트 크기 측정]")
    comp = measure_component_gb(model)
    COMP_GB.update(comp)
    # MODEL_GB 전역 갱신
    global MODEL_GB
    MODEL_GB = comp["total"]

    print(f"  로드 완료: {load_s:.1f}s  /  total={comp['total']:.3f} GB")

    print("\n[2/4] 입력 준비...")
    clip_id  = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data     = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    processor    = helper.get_processor(model.tokenizer)
    inputs       = processor.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt",
    )
    model_inputs = helper.to_device({
        "tokenized_data"  : inputs,
        "ego_history_xyz" : data["ego_history_xyz"],
        "ego_history_rot" : data["ego_history_rot"],
    }, "cuda")
    input_tok_len = inputs["input_ids"].shape[-1]
    print(f"  입력 토큰: {input_tok_len}")

    print("\n[3/4] 패치 등록...")
    det     = PhaseDetectorV4()
    t_vlm   = CUDATimer("vlm")
    t_flow  = CUDATimer("flow")
    tok_n   = [0]
    euler_n = [0]

    patch_vlm_forward(model.vlm, det)
    patch_lm_forward(model.vlm, det)
    wrap_generate(model.vlm, det, t_vlm, tok_n, input_tok_len)
    wrap_diffusion(model.diffusion, t_flow, euler_n)

    print("\n[4/4] 프로파일링 시작")
    all_results: list[RunResult] = []

    print(f"\n  ── Warmup ({warmup}회) ──────────────────────────────────────")
    for i in range(warmup):
        r = run_one(f"run_{i+1:02d}", is_warmup=True,
                    model=model, model_inputs=model_inputs,
                    det=det, t_vlm=t_vlm, t_flow=t_flow,
                    tok_n=tok_n, euler_n=euler_n)
        all_results.append(r)

    print(f"\n  ── Measurement ({runs}회) ───────────────────────────────────")
    measure_results: list[RunResult] = []
    for i in range(runs):
        r = run_one(f"run_{i+1:02d}", is_warmup=False,
                    model=model, model_inputs=model_inputs,
                    det=det, t_vlm=t_vlm, t_flow=t_flow,
                    tok_n=tok_n, euler_n=euler_n)
        all_results.append(r)
        measure_results.append(r)

    return all_results, measure_results, comp, input_tok_len


# ══════════════════════════════════════════════════════════════════════════════
# 결과 집계
# ══════════════════════════════════════════════════════════════════════════════

def summarize(measure_results: list[RunResult],
              all_results:     list[RunResult],
              comp:            dict[str, float],
              input_tok_len:   int):

    def M(key):
        return float(np.mean([getattr(r, key) for r in measure_results]))
    def S(key):
        return float(np.std([getattr(r, key)  for r in measure_results]))

    split_ok = measure_results[0].split_ok

    vis_ms   = M("vision_ms");     vis_s  = S("vision_ms")
    lm_ms    = M("lm_prefill_ms"); lm_s   = S("lm_prefill_ms")
    dec_ms   = M("decode_ms");     dec_s  = S("decode_ms")
    fl_ms    = M("flow_ms");       fl_s   = S("flow_ms")
    vlm_ms   = M("vlm_ms");        vlm_s  = S("vlm_ms")
    wall_ms  = M("wall_ms")
    n_tok    = M("n_tok")
    n_euler  = M("n_euler")

    vis_bw  = M("vision_bw_GBps")
    lm_bw   = M("lm_prefill_bw_GBps")
    dec_bw  = M("decode_bw_GBps")
    fl_bw   = M("flow_bw_GBps")

    W   = 96
    sep = "=" * W

    print(f"\n{sep}")
    print(f"  Alpamayo 1.5 on Jetson AGX Thor -- All-Phase CUDA Events BW")
    print(f"  n={len(measure_results)} measure runs  |  bf16  |  torch.autocast")
    print(f"  컴포넌트: total={comp['total']:.3f} GB  vlm={comp.get('vlm_gb',0):.3f} GB  "
          f"(vision={comp['vision']:.3f}  lm={comp['lm']:.3f})  flow={comp['flow']:.3f} GB")
    print(sep)

    if split_ok:
        total_ms = vis_ms + lm_ms + dec_ms + fl_ms
        rows = [
            ("Vision Enc.",  vis_ms,  vis_s,  vis_bw,  comp["vision"], 1),
            ("LM Prefill",   lm_ms,   lm_s,   lm_bw,   comp["lm"],     1),
            # Decode: vlm_gb - vision_gb (vision encoder는 decode 시 미로드)
        ("Decode",       dec_ms,  dec_s,  dec_bw,
         comp.get("vlm_gb", comp["total"]) - comp.get("vision", 0.0), n_tok),
            ("Flow",         fl_ms,   fl_s,   fl_bw,   comp["flow"],   n_euler),
        ]
        hdr = (f"  {'Phase':<14} {'ms':>8} {'+-':>6}  {'share':>6}  "
               f"{'mod_GB':>7}  {'x_pass':>6}  {'BW(GB/s)':>9}  {'MBU%':>5}  판정")
        print(hdr)
        print(f"  {'-'*94}")
        for name, ms, std, bw, gb, xp in rows:
            share = ms / total_ms * 100 if total_ms > 0 else 0
            mbu   = bw / DRAM_BW * 100
            bw_s  = f"{bw:>9.1f}" if bw > 0 else "        --"
            mbu_s = f"{mbu:>5.0f}" if bw > 0 else "   --"
            meth  = "CUDA Events" if bw > 0 else "no data"
            verd  = ("★ BW-BOUND"    if mbu >= 70
                     else "BW-bound"  if mbu >= 40
                     else "compute"   if bw > 0
                     else "?")
            xp_s  = f"{int(xp):>4}tok" if name == "Decode" else (
                     f"{int(xp):>4}eul" if name == "Flow" else "  1pss")
            print(f"  {name:<14} {ms:>8.0f} {std:>6.0f}  {share:>5.1f}%  "
                  f"{gb:>7.3f}  {xp_s}  {bw_s}  {mbu_s}  {verd}")
        print(f"  {'-'*94}")
        print(f"  {'Total':<14} {total_ms:>8.0f}           "
              f"(VLM={vlm_ms:.0f} +/-{vlm_s:.0f} ms  |  wall={wall_ms:.0f} ms)")
    else:
        print("  [경고] Vision/LM 분리 실패 -> Decode + Flow BW만 보고")
        pf_ms = M("prefill_ms")
        pf_s  = S("prefill_ms")
        for name, ms, std, bw in [
            ("Prefill(통합)", pf_ms,  pf_s,  0.0),
            ("Decode",       dec_ms, dec_s, dec_bw),
            ("Flow",         fl_ms,  fl_s,  fl_bw),
        ]:
            mbu   = bw / DRAM_BW * 100
            bw_s  = f"{bw:.1f}" if bw > 0 else "--"
            mbu_s = f"{mbu:.0f}%" if bw > 0 else "--"
            print(f"  {name:<18} {ms:.0f} +/-{std:.0f} ms  BW={bw_s}  MBU={mbu_s}")

    warmup_rs = [r for r in all_results if r.is_warmup]
    if warmup_rs:
        print(f"\n{'─'*W}")
        print("  Warmup vs Measure")
        for label, key in [("VLM total", "vlm_ms"), ("Decode", "decode_ms"),
                            ("Flow",     "flow_ms")]:
            wm = float(np.mean([getattr(r, key) for r in warmup_rs]))
            mm = float(np.mean([getattr(r, key) for r in measure_results]))
            print(f"  {label:<14}  warmup={wm:.0f}ms  measure={mm:.0f}ms  "
                  f"diff={wm-mm:+.0f}ms ({(wm-mm)/mm*100:+.1f}%)"
                  if mm > 0 else f"  {label:<14}  warmup={wm:.0f}ms  measure={mm:.0f}ms")

    print(sep)

    # ── JSON 저장 ──────────────────────────────────────────────────────────
    out_data = {
        "script":          "260515_bw_allphase",
        "dram_bw_peak_GBps": DRAM_BW,
        "comp_gb":         comp,
        "n_warmup":        len(warmup_rs),
        "n_measure":       len(measure_results),
        "input_tok_len":   input_tok_len,
        "split_ok":        split_ok,
        "measure_means": {
            "vision_ms":          vis_ms,   "vision_ms_std":       vis_s,
            "lm_prefill_ms":      lm_ms,    "lm_prefill_ms_std":   lm_s,
            "decode_ms":          dec_ms,   "decode_ms_std":       dec_s,
            "flow_ms":            fl_ms,    "flow_ms_std":         fl_s,
            "vlm_ms":             vlm_ms,   "vlm_ms_std":          vlm_s,
            "wall_ms":            wall_ms,
            "n_tok":              n_tok,
            "n_euler":            n_euler,
            # BW (모두 CUDA Events 기반)
            "vision_bw_GBps":     vis_bw,   "vision_mbu_pct":      vis_bw / DRAM_BW * 100,
            "lm_prefill_bw_GBps": lm_bw,    "lm_prefill_mbu_pct":  lm_bw / DRAM_BW * 100,
            "decode_bw_GBps":     dec_bw,   "decode_mbu_pct":      dec_bw / DRAM_BW * 100,
            "flow_bw_GBps":       fl_bw,    "flow_mbu_pct":        fl_bw / DRAM_BW * 100,
        },
        "runs": [asdict(r) for r in all_results],
    }
    jp = OUT / "allphase_bw.json"
    jp.write_text(json.dumps(out_data, indent=2, ensure_ascii=False, default=float))
    print(f"\n[저장] {jp}")

    _write_md(out_data)
    _plot(out_data, all_results, warmup_rs, measure_results)


# ══════════════════════════════════════════════════════════════════════════════
# 마크다운 저장
# ══════════════════════════════════════════════════════════════════════════════

def _write_md(d: dict):
    m   = d["measure_means"]
    cmp = d["comp_gb"]
    lines = [
        "# Alpamayo 1.5 -- All-Phase DRAM Bandwidth (CUDA Events)",
        f"**보드**: Jetson AGX Thor (LPDDR5X {DRAM_BW} GB/s)",
        f"**측정**: CUDA Events (n={d['n_measure']}회)  |  bf16  |  2026-05-15",
        "",
        "## 모델 컴포넌트 크기",
        "",
        f"| 컴포넌트 | 크기 (GB) | 비중 | 비고 |",
        f"|---|---|---|---|",
        f"| **Total**  | **{cmp['total']:.3f}** | 100% | |",
        f"| VLM (vlm)  | {cmp.get('vlm_gb',0):.3f} | {cmp.get('vlm_gb',0)/cmp['total']*100:.1f}% | Decode 분모 |",
        f"| -- Vision Enc. | {cmp['vision']:.3f} | {cmp['vision']/cmp['total']*100:.1f}% | model.vlm.visual |",
        f"| -- LM Transformer | {cmp['lm']:.3f} | {cmp['lm']/cmp['total']*100:.1f}% | model.vlm.language_model |",
        f"| -- 기타 (embed+head) | {cmp.get('vlm_gb',0)-cmp['vision']-cmp['lm']:.3f} | | embed, lm_head |",
        f"| **Flow (model.expert)** | **{cmp['flow']:.3f}** | **{cmp['flow']/cmp['total']*100:.1f}%** | Action Expert |",
        "",
        "## Phase별 DRAM 대역폭 (실측 -- 추정값 없음)",
        "",
        "| Phase | 시간 (ms) | 모델 GB | x pass | **BW (GB/s)** | **MBU%** | 판정 |",
        "|---|---|---|---|---|---|---|",
    ]

    decode_gb_denom = cmp.get("vlm_gb", cmp["total"]) - cmp.get("vision", 0.0)
    rows = [
        ("Vision Enc.", "vision_ms",    cmp["vision"],        1,
         "vision_bw_GBps",     "vision_mbu_pct",    "compute (weight BW only)"),
        ("LM Prefill",  "lm_prefill_ms", cmp["lm"],           1,
         "lm_prefill_bw_GBps", "lm_prefill_mbu_pct","compute (weight BW only)"),
        ("Decode",      "decode_ms",    decode_gb_denom,      m["n_tok"],
         "decode_bw_GBps",     "decode_mbu_pct",    "BW-bound"),
        ("Flow",        "flow_ms",      cmp["flow"],          m["n_euler"],
         "flow_bw_GBps",       "flow_mbu_pct",      "compute"),
    ]
    for name, ms_k, gb_v, xp, bw_k, mbu_k, verd_note in rows:
        ms_v  = m[ms_k]
        bw_v  = m[bw_k]
        mbu_v = m[mbu_k]
        xp_s  = f"{int(xp)} tok" if "decode" in bw_k else (
                f"{int(xp)} euler" if "flow" in bw_k else "1 pass")
        verd  = "**BW-BOUND**" if mbu_v >= 70 else (
                "BW-bound"     if mbu_v >= 40 else verd_note)
        bw_note = "" if name != "Decode" else " *(vision enc. excluded)*"
        lines.append(
            f"| {name} | {ms_v:.0f} | {gb_v:.3f} | {xp_s} "
            f"| **{bw_v:.1f}**{bw_note} | **{mbu_v:.0f}%** | {verd} |"
        )

    lines += [
        "",
        "## BW 계산 공식",
        "",
        "```",
        "Vision    BW = vision_gb   x 1 pass  / vision_ms",
        "LM Prefill BW = lm_gb     x 1 pass  / lm_prefill_ms",
        "Decode     BW = total_gb  x n_tok   / decode_ms",
        "Flow       BW = flow_gb   x n_euler / flow_ms",
        "```",
        "",
        "> **측정 방법**: CUDA Events  ",
        "> tegrastats / ncu / bwmon 불사용  ",
        "> 모든 값 직접 측정 (추정값 없음)",
    ]
    mp = OUT / "allphase_bw.md"
    mp.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {mp}")


# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Vision Enc.": "#5B9BD5",
    "LM Prefill":  "#70AD47",
    "Decode":      "#C0504D",
    "Flow":        "#9067A7",
}


def _plot(d: dict, all_results, warmup_rs, measure_rs):
    m    = d["measure_means"]
    cmp  = d["comp_gb"]
    split = d["split_ok"]

    phases = ["Vision Enc.", "LM Prefill", "Decode", "Flow"]
    ms_v   = [m["vision_ms"],    m["lm_prefill_ms"], m["decode_ms"], m["flow_ms"]]
    bw_v   = [m["vision_bw_GBps"], m["lm_prefill_bw_GBps"],
              m["decode_bw_GBps"],  m["flow_bw_GBps"]]
    mbu_v  = [m["vision_mbu_pct"], m["lm_prefill_mbu_pct"],
              m["decode_mbu_pct"],  m["flow_mbu_pct"]]
    cols   = [COLORS[p] for p in phases]

    # ── Fig 1: Phase duration + BW side by side ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Alpamayo 1.5 on Jetson AGX Thor -- All-Phase DRAM BW (CUDA Events)\n"
        f"Total={comp_total:.3f} GB  Vision={cmp['vision']:.2f}  "
        f"LM={cmp['lm']:.2f}  Flow={cmp['flow']:.2f} GB",
        fontsize=10, fontweight="bold",
    )

    # (a) 시간
    ax = axes[0]
    ax.set_facecolor("white")
    bars = ax.bar(phases, ms_v, color=cols, alpha=0.88,
                  edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, ms_v):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 20,
                f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("GPU Time (ms)", fontsize=10)
    ax.set_title("(a) Phase Duration (CUDA Events)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")

    # (b) BW + MBU
    ax = axes[1]
    ax.set_facecolor("white")
    bars = ax.bar(phases, bw_v, color=cols, alpha=0.88,
                  edgecolor="white", linewidth=1.5)
    ax.axhline(DRAM_BW, color="#444444", lw=1.2, ls="--",
               label=f"Peak {DRAM_BW:.0f} GB/s")
    ax.axhline(DRAM_BW * 0.70, color="orange", lw=1.0, ls=":",
               label="BW-bound (70%)")
    for bar, bw, mbu in zip(bars, bw_v, mbu_v):
        if bw > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bw + 4,
                    f"{bw:.0f}\n({mbu:.0f}%)",
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, DRAM_BW * 1.25)
    ax.set_ylabel("DRAM Bandwidth (GB/s)", fontsize=10)
    ax.set_title("(b) Measured BW & MBU -- CUDA Events\n(no estimates)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")

    plt.tight_layout(pad=1.8)
    for ext in ("png", "pdf"):
        fp = FIGD / f"fig_allphase_bw.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Fig] {fp}")
    plt.close(fig)

    # ── Fig 2: MBU Radar / Bar ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # (a) MBU 수평 bar
    ax = axes[0]
    ax.set_facecolor("white")
    y_pos = np.arange(len(phases))
    hbars = ax.barh(y_pos, mbu_v, color=cols, alpha=0.88,
                    edgecolor="white", linewidth=1.0)
    ax.axvline(70, color="orange", lw=1.2, ls=":", label="BW-bound 70%")
    ax.axvline(100, color="#444", lw=1.0, ls="--", label="Peak 100%")
    for bar, mbu in zip(hbars, mbu_v):
        if mbu > 0:
            ax.text(mbu + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{mbu:.0f}%",
                    va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(phases, fontsize=10)
    ax.set_xlabel("MBU -- Memory Bandwidth Utilization (%)", fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_title("(a) MBU per Phase", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) timeline Gantt
    ax = axes[1]
    ax.set_facecolor("white")
    total_ms_sum = sum(ms_v)
    starts = [0]
    for v in ms_v[:-1]:
        starts.append(starts[-1] + v)
    for s, w, ph, c in zip(starts, ms_v, phases, cols):
        ax.barh(0, w, left=s, height=0.5, color=c, alpha=0.88,
                edgecolor="white", linewidth=1.5)
        if w > total_ms_sum * 0.04:
            ax.text(s + w / 2, 0,
                    f"{ph}\n{w:.0f}ms",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white")
    ax.set_xlim(0, total_ms_sum * 1.02)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("GPU Time (ms)", fontsize=10)
    ax.set_yticks([])
    ax.set_title("(b) Inference Timeline (Measure, avg)", fontsize=10, fontweight="bold")
    patches = [mpatches.Patch(color=COLORS[p], label=p) for p in phases]
    ax.legend(handles=patches, loc="upper right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, ls="--")

    plt.tight_layout(pad=1.8)
    for ext in ("png", "pdf"):
        fp = FIGD / f"fig_allphase_mbu.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Fig] {fp}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

comp_total = MODEL_GB  # 출력용 (main에서 갱신)


def main():
    global comp_total
    ap = argparse.ArgumentParser(
        description="Alpamayo 1.5 All-Phase DRAM BW (CUDA Events)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Warmup 횟수 (기본 1)")
    ap.add_argument("--runs",   type=int, default=2,
                    help="측정 횟수 (기본 2)")
    args = ap.parse_args()

    all_r, meas_r, comp, input_tok_len = run_profiling(
        warmup=args.warmup, runs=args.runs
    )
    comp_total = comp["total"]
    summarize(meas_r, all_r, comp, input_tok_len)


if __name__ == "__main__":
    main()
