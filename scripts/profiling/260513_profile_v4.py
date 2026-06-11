"""
260513_profile_v4.py  ·  Phase-Separated Profiler v4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[v3 → v4 개선 사항]
  v3: Phase/Prefill (ViT + LM 혼합)  ← Vision과 LM이 섞여 있었음
  v4: 완전 4단계 분리
      Phase/Vision_Encoder  ← Cosmos Vision Tokenizer (Kernel2 ×2)
      Phase/LM_Prefill      ← Transformer 언어모델 prefill 단독
      Phase/Decode          ← Autoregressive 토큰 생성 전체
      Phase/Flow            ← Flow Matching 궤적 생성

  + Warmup/run_N vs Measure/run_N NVTX 명시적 구분
  + 각 Phase 전환점 GPU 메모리 스냅샷
  + Decode 개별 step NVTX (Decode/step_001 ~ Decode/step_NNN)
  + Flow/Euler NVTX

[분리 메커니즘]
  vlm.forward(seq>1) 시작       → Vision_Encoder 시작
  lm.forward(seq>1) 시작        → Vision_Encoder 종료, LM_Prefill 시작
  lm.forward(seq>1) 종료        → LM_Prefill 종료
  첫 vlm.forward(seq=1) 시작    → Decode 시작
  각 vlm.forward(seq=1) 시작/종료→ Decode/step_NNN 마킹
  end_generate()                 → Decode 종료
  diffusion.sample()             → Flow 시작/종료
  diffusion._euler()             → Flow/Euler 마킹

[nsys NVTX 계층 구조]
  Warmup/run_01  (또는 Measure/run_01)
  └── Inference/Total
      ├── Phase/VLM_Generate
      │   ├── Phase/Vision_Encoder    ← Cosmos ViT
      │   ├── Phase/LM_Prefill        ← Transformer prefill
      │   └── Phase/Decode
      │       ├── Decode/step_001
      │       ├── Decode/step_002
      │       └── ...
      └── Phase/Flow
          └── Flow/Euler

[폴백]
  lm.forward 패치 실패 시 → Vision_Encoder/LM_Prefill 분리 불가
                            → Phase/Prefill(통합)로 보고
  모든 패치 실패 시       → VLM 전체 시간만 측정
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
OUT  = Path("profiling_results/260513_v4")
FIGD = OUT / "figures"
for d in [OUT, FIGD]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DRAM_BW  = 273.0   # GB/s (Jetson AGX Thor LPDDR5X)
MODEL_GB = 22.157  # bf16 전체 모델


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
# Phase 상태 머신 v4
# ══════════════════════════════════════════════════════════════════════════════

class PhaseDetectorV4:
    """
    상태 전이:
      idle
        → vision      (vlm.forward seq>1 시작)
        → lm_prefill  (lm.forward seq>1 시작, ViT 종료)
        → post_prefill(lm.forward seq>1 종료, LM prefill 종료)
        → decode      (첫 vlm.forward seq=1)
        → idle        (end_generate)

    폴백 상태:
      lm_forward가 한 번도 호출되지 않으면 ViT 분리 불가
      → vision 상태에서 vlm.forward가 종료되면 prefill_fallback 처리
    """

    # 상태 상수
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

        self.t_vision    = CUDATimer("vision")
        self.t_lm_prefill= CUDATimer("lm_prefill")
        self.t_decode    = CUDATimer("decode")
        self.mem         = {}
        self._lm_patched = False   # lm.forward 패치 성공 여부

    def reset(self):
        self._state       = self.IDLE
        self._decode_step = 0
        self._calls_vlm   = []
        self._calls_lm    = []
        self.t_vision.reset()
        self.t_lm_prefill.reset()
        self.t_decode.reset()
        self.mem = {}

    # ── seq_len 추출 ─────────────────────────────────────────────────────────

    def _seq(self, args, kwargs) -> int | None:
        for src in [kwargs.get("input_ids"),
                    kwargs.get("hidden_states"),
                    kwargs.get("inputs_embeds"),
                    *(a for a in args if isinstance(a, torch.Tensor))]:
            if src is None:
                continue
            if isinstance(src, torch.Tensor):
                if src.ndim == 2:          # [B, T]
                    return int(src.shape[-1])
                if src.ndim == 3:          # [B, T, D]
                    return int(src.shape[1])
        return None

    # ── vlm.forward 훅 ───────────────────────────────────────────────────────

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
                # 첫 번째 decode step
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
                    f"Decode/step_{self._decode_step:03d}"
                )

    def on_vlm_after(self):
        if self._state == self.DECODE:
            torch.cuda.nvtx.range_pop()   # Decode/step_NNN

        elif self._state == self.VISION:
            # lm.forward가 한 번도 안 불렸음 = ViT 분리 실패 폴백
            # Vision_Encoder NVTX를 LM_Prefill로 재명명하는 것은 불가
            # 그냥 Vision_Encoder 마커를 유지하면서 다음 상태 진행
            # (이 경우 Vision_Encoder가 실제로는 ViT+LM 혼합을 의미)
            pass

    # ── lm.forward 훅 ────────────────────────────────────────────────────────

    def on_lm_before(self, args, kwargs):
        seq = self._seq(args, kwargs)
        if seq is None:
            return
        self._calls_lm.append(seq)
        self._lm_patched = True

        if seq > 1 and self._state == self.VISION:
            # ViT 종료 → LM Prefill 시작
            self.t_vision.stop()
            torch.cuda.nvtx.range_pop()         # Phase/Vision_Encoder
            self.mem["vision_after_gb"]    = gpu_mem_gb()
            self.mem["vision_peak_gb"]     = gpu_peak_gb()
            torch.cuda.reset_peak_memory_stats()

            self.mem["lm_prefill_before_gb"] = gpu_mem_gb()
            torch.cuda.nvtx.range_push("Phase/LM_Prefill")
            self.t_lm_prefill.start()
            self._state = self.LM_PREFILL

    def on_lm_after(self):
        if self._state == self.LM_PREFILL:
            # LM Prefill 완료
            self.t_lm_prefill.stop()
            torch.cuda.nvtx.range_pop()         # Phase/LM_Prefill
            self.mem["lm_prefill_after_gb"] = gpu_mem_gb()
            self.mem["lm_prefill_peak_gb"]  = gpu_peak_gb()
            self._state = self.POST_PREFILL

    # ── generate 종료 ─────────────────────────────────────────────────────────

    def end_generate(self):
        if self._state == self.DECODE:
            self.t_decode.stop()
            torch.cuda.nvtx.range_pop()         # Phase/Decode
            self.mem["decode_after_gb"] = gpu_mem_gb()
            self.mem["decode_peak_gb"]  = gpu_peak_gb()

        elif self._state == self.LM_PREFILL:
            self.t_lm_prefill.stop()
            torch.cuda.nvtx.range_pop()         # Phase/LM_Prefill

        elif self._state == self.VISION:
            self.t_vision.stop()
            torch.cuda.nvtx.range_pop()         # Phase/Vision_Encoder

        elif self._state == self.POST_PREFILL:
            pass  # No token was generated

        self._state = self.IDLE

    # ── 진단 출력 ─────────────────────────────────────────────────────────────

    def diagnostics(self) -> str:
        n_vlm_pf = sum(1 for s in self._calls_vlm if s > 1)
        n_vlm_dc = sum(1 for s in self._calls_vlm if s == 1)
        n_lm_pf  = sum(1 for s in self._calls_lm  if s > 1)
        n_lm_dc  = sum(1 for s in self._calls_lm  if s == 1)
        lines = [
            f"    vlm.forward calls : prefill={n_vlm_pf}  decode={n_vlm_dc}",
            f"     lm.forward calls : prefill={n_lm_pf}   decode={n_lm_dc}"
            + ("  ← lm 패치 성공" if self._lm_patched else "  ← lm 패치 미작동"),
        ]
        return "\n".join(lines)

    @property
    def split_ok(self) -> bool:
        """ViT / LM_Prefill 분리 성공 여부."""
        return self.t_vision.ms() > 0 and self.t_lm_prefill.ms() > 0


# ══════════════════════════════════════════════════════════════════════════════
# 패치 함수
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
    print("  [패치] vlm.forward ✓")
    return True


def patch_lm_forward(vlm, det: PhaseDetectorV4) -> bool:
    """
    vlm 내부의 언어 모델 forward 패치.
    Qwen2VL 구조: vlm.model  (Qwen2VLModel)
    추가 후보:    vlm.language_model, vlm.model.model
    """
    candidates: list[tuple[str, object]] = []

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
        print(f"  [패치] {path}.forward ✓")
        return True

    print("  [경고] lm.forward 패치 실패 — ViT/LM 분리 불가")
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

        torch.cuda.nvtx.range_pop()   # Phase/VLM_Generate
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
    print("  [패치] vlm.generate (VLM_Generate wrapper) ✓")


def wrap_diffusion(diffusion, t_flow: CUDATimer, euler_n: list[int]):
    # _euler 카운터 + NVTX
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

    # sample 타이머 + NVTX
    orig_sample = diffusion.sample

    def _timed_sample(*args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.nvtx.range_push("Phase/Flow")
        t_flow.start()
        result = orig_sample(*args, **kwargs)
        t_flow.stop()
        torch.cuda.nvtx.range_pop()   # Phase/Flow
        return result

    diffusion.sample = _timed_sample
    print("  [패치] diffusion.sample + _euler (Flow wrapper) ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 측정 결과 데이터 클래스
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResultV4:
    label:        str     # "warmup1" / "run1"
    is_warmup:    bool

    wall_ms:      float
    vlm_ms:       float

    vision_ms:    float   # 0 = 분리 실패
    lm_prefill_ms:float   # 0 = 분리 실패
    decode_ms:    float
    flow_ms:      float

    n_tok:        int
    n_euler:      int
    split_ok:     bool    # Vision/LM 분리 성공 여부

    mem:          dict    = field(default_factory=dict)

    # ── 유도 속성 ────────────────────────────────────────────────────────────

    @property
    def prefill_ms(self) -> float:
        """ViT + LM Prefill 합산 (v3 호환)."""
        if self.split_ok:
            return self.vision_ms + self.lm_prefill_ms
        return self.vlm_ms - self.decode_ms

    @property
    def decode_bw_GBps(self) -> float:
        if self.decode_ms > 0 and self.n_tok > 0:
            return MODEL_GB * self.n_tok / (self.decode_ms / 1000.0)
        return 0.0

    @property
    def decode_bw_pct(self) -> float:
        return self.decode_bw_GBps / DRAM_BW * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 단일 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_one(label: str, is_warmup: bool,
            model, model_inputs,
            det: PhaseDetectorV4,
            t_vlm: CUDATimer, t_flow: CUDATimer,
            tok_n: list[int], euler_n: list[int]) -> RunResultV4:

    det.reset()
    euler_n[0] = 0
    tok_n[0]   = 0
    t_vlm.reset()
    t_flow.reset()
    torch.cuda.synchronize()

    # ── run 레벨 NVTX (warmup / measure 구분) ────────────────────────────
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

    torch.cuda.nvtx.range_pop()   # Inference/Total
    torch.cuda.nvtx.range_pop()   # Warmup/run_N or Measure/run_N

    # ── 결과 수집 ────────────────────────────────────────────────────────
    r = RunResultV4(
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

    # ── 터미널 출력 ──────────────────────────────────────────────────────
    tag = "WARMUP" if is_warmup else "  RUN "
    print(f"\n  [{tag} {label}]")
    if r.split_ok:
        print(f"    ViT  : {r.vision_ms:>7.0f} ms")
        print(f"    LM↑  : {r.lm_prefill_ms:>7.0f} ms   "
              f"(Prefill={r.prefill_ms:.0f} ms)")
    else:
        print(f"    Prefill(통합): {r.prefill_ms:>7.0f} ms  [ViT/LM 미분리]")
    print(f"    Decode: {r.decode_ms:>7.0f} ms  "
          f"({r.n_tok} tok)  "
          f"BW={r.decode_bw_GBps:.1f} GB/s ({r.decode_bw_pct:.0f}%)")
    print(f"    Flow  : {r.flow_ms:>7.0f} ms  ({r.n_euler} euler steps)")
    print(f"    VLM   : {r.vlm_ms:>7.0f} ms  |  Wall: {r.wall_ms:.0f} ms")
    print(det.diagnostics())

    if det.mem:
        print(f"    [메모리 스냅샷]")
        for k, v in sorted(det.mem.items()):
            print(f"      {k:<28} {v:.3f} GB")

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
    model_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"  로드 완료: {time.perf_counter()-t0:.1f}s  /  {model_gb:.3f} GB")

    print("[2/4] 입력 준비...")
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

    print("[3/4] 패치 등록...")
    det     = PhaseDetectorV4()
    t_vlm   = CUDATimer("vlm")
    t_flow  = CUDATimer("flow")
    tok_n   = [0]
    euler_n = [0]

    patch_vlm_forward(model.vlm, det)
    patch_lm_forward(model.vlm, det)
    wrap_generate(model.vlm, det, t_vlm, tok_n, input_tok_len)
    wrap_diffusion(model.diffusion, t_flow, euler_n)

    print("[4/4] 프로파일링 시작")
    all_results: list[RunResultV4] = []

    # warmup
    print(f"\n  ── Warmup ({warmup}회) ──────────────────────────────────────")
    for i in range(warmup):
        r = run_one(f"run_{i+1:02d}", is_warmup=True,
                    model=model, model_inputs=model_inputs,
                    det=det, t_vlm=t_vlm, t_flow=t_flow,
                    tok_n=tok_n, euler_n=euler_n)
        all_results.append(r)

    # 측정
    print(f"\n  ── Measurement ({runs}회) ───────────────────────────────────")
    measure_results: list[RunResultV4] = []
    for i in range(runs):
        r = run_one(f"run_{i+1:02d}", is_warmup=False,
                    model=model, model_inputs=model_inputs,
                    det=det, t_vlm=t_vlm, t_flow=t_flow,
                    tok_n=tok_n, euler_n=euler_n)
        all_results.append(r)
        measure_results.append(r)

    return all_results, measure_results, model_gb, input_tok_len


# ══════════════════════════════════════════════════════════════════════════════
# 결과 집계 및 출력
# ══════════════════════════════════════════════════════════════════════════════

def summarize(measure_results: list[RunResultV4],
              all_results: list[RunResultV4],
              model_gb: float, input_tok_len: int):

    def M(key):  # mean over measure runs
        return float(np.mean([getattr(r, key) for r in measure_results]))
    def S(key):
        return float(np.std([getattr(r, key) for r in measure_results]))

    split_ok   = measure_results[0].split_ok
    vision_m   = M("vision_ms");     vision_s   = S("vision_ms")
    lmpf_m     = M("lm_prefill_ms"); lmpf_s     = S("lm_prefill_ms")
    prefill_m  = M("prefill_ms");    prefill_s  = S("prefill_ms")
    decode_m   = M("decode_ms");     decode_s   = S("decode_ms")
    flow_m     = M("flow_ms");       flow_s     = S("flow_ms")
    vlm_m      = M("vlm_ms");        vlm_s      = S("vlm_ms")
    wall_m     = M("wall_ms")
    n_tok      = M("n_tok")
    n_euler    = M("n_euler")
    bw_m       = M("decode_bw_GBps")
    bwp_m      = M("decode_bw_pct")

    W = 88
    sep = "═" * W

    print(f"\n{sep}")
    print(f"  Alpamayo 1.5 on Jetson AGX Thor — v4 Phase-Separated 결과")
    print(f"  측정: CUDA Events (n={len(measure_results)}) | bf16 | torch.autocast")
    print(sep)

    if split_ok:
        total_ms = vision_m + lmpf_m + decode_m + flow_m
        rows = [
            ("Vision Encoder",  vision_m, vision_s, 0.0,  0.0 ),
            ("LM Prefill",      lmpf_m,   lmpf_s,   0.0,  0.0 ),
            ("Decode",          decode_m, decode_s, bw_m, bwp_m),
            ("Flow",            flow_m,   flow_s,   0.0,  0.0 ),
        ]
        print(f"  {'Phase':<18} {'GPU Time (ms)':>16}  {'Share':>6}  "
              f"{'BW GB/s':>10}  {'MBU%':>5}  판정")
        print(f"  {'-'*18} {'-'*16}  {'-'*6}  {'-'*10}  {'-'*5}  {'-'*22}")
        for name, ms, std, bw, bwp in rows:
            share = ms / total_ms * 100 if total_ms > 0 else 0
            bw_s  = f"{bw:.1f}" if bw > 0 else "—"
            bwp_s = f"{bwp:.0f}%" if bwp > 0 else "—"
            verd  = ("★ BW-BOUND"          if bwp >= 70
                     else "BW-bound (mod)"  if bwp >= 40
                     else "compute-bound"   if name in ("LM Prefill",)
                     else "overhead-dom."   if name == "Flow"
                     else "compute-bound")
            print(f"  {name:<18} {ms:>8.0f} ±{std:>5.0f} ms  "
                  f"{share:>5.1f}%  {bw_s:>10}  {bwp_s:>5}  {verd}")
        print(f"  {'-'*18} {'-'*16}")
        print(f"  {'Total':<18} {total_ms:>8.0f} ms  (VLM={vlm_m:.0f}±{vlm_s:.0f}ms)")
    else:
        # 폴백 출력
        total_ms = prefill_m + decode_m + flow_m
        print(f"  [주의] ViT/LM_Prefill 분리 실패 → Prefill 통합 보고")
        rows = [
            ("Prefill (ViT+LM)", prefill_m, prefill_s, 0.0,  0.0 ),
            ("Decode",           decode_m,  decode_s,  bw_m, bwp_m),
            ("Flow",             flow_m,    flow_s,    0.0,  0.0 ),
        ]
        for name, ms, std, bw, bwp in rows:
            share = ms / total_ms * 100
            bw_s  = f"{bw:.1f}" if bw > 0 else "—"
            bwp_s = f"{bwp:.0f}%" if bwp > 0 else "—"
            print(f"  {name:<22} {ms:>8.0f} ±{std:>5.0f} ms  "
                  f"{share:>5.1f}%  {bw_s}  {bwp_s}")

    # warmup vs measure 비교
    warmup_rs = [r for r in all_results if r.is_warmup]
    if warmup_rs:
        print(f"\n{'─'*W}")
        print(f"  Warmup vs Measure 비교 (JIT + Allocator 오버헤드)")
        print(f"  {'':18} {'Warmup':>12}  {'Measure':>12}  {'차이':>10}  {'비율':>6}")
        print(f"  {'-'*18} {'-'*12}  {'-'*12}  {'-'*10}  {'-'*6}")
        pairs = [
            ("VLM total",      "vlm_ms"),
            ("Prefill(total)", "prefill_ms"),
            ("Decode",         "decode_ms"),
            ("Flow",           "flow_ms"),
        ]
        for label, key in pairs:
            w_m = float(np.mean([getattr(r, key) for r in warmup_rs]))
            m_m = float(np.mean([getattr(r, key) for r in measure_results]))
            diff = w_m - m_m
            pct  = diff / m_m * 100 if m_m > 0 else 0
            print(f"  {label:<18} {w_m:>10.0f}ms  {m_m:>10.0f}ms  "
                  f"{diff:>+8.0f}ms  {pct:>+5.1f}%")

    print(f"\n  BW_decode = {model_gb:.3f} GB × {n_tok:.0f} tok / "
          f"{decode_m:.0f} ms = {bw_m:.1f} GB/s ({bwp_m:.1f}% of {DRAM_BW} GB/s)")
    print(sep)

    # ── JSON 저장 ───────────────────────────────────────────────────────────
    out_data = {
        "split_ok"       : split_ok,
        "n_warmup"       : len(warmup_rs),
        "n_measure"      : len(measure_results),
        "input_tok_len"  : input_tok_len,
        "model_gb"       : model_gb,
        "dram_bw_GBps"   : DRAM_BW,
        "measure_means"  : {
            "vision_ms"    : vision_m,
            "lm_prefill_ms": lmpf_m,
            "prefill_ms"   : prefill_m,
            "decode_ms"    : decode_m,
            "flow_ms"      : flow_m,
            "vlm_ms"       : vlm_m,
            "wall_ms"      : wall_m,
            "decode_bw_GBps": bw_m,
            "decode_bw_pct" : bwp_m,
            "n_tok"        : n_tok,
            "n_euler"      : n_euler,
        },
        "measure_stds"   : {
            "vision_ms"    : vision_s,
            "lm_prefill_ms": lmpf_s,
            "prefill_ms"   : prefill_s,
            "decode_ms"    : decode_s,
            "flow_ms"      : flow_s,
            "vlm_ms"       : vlm_s,
        },
        "runs": [asdict(r) for r in all_results],
    }
    p = OUT / "phase_v4.json"
    p.write_text(json.dumps(out_data, indent=2, ensure_ascii=False, default=float))
    print(f"\n[저장] {p}")

    _write_md(out_data)
    _plot(out_data, all_results, warmup_rs, measure_results)


# ══════════════════════════════════════════════════════════════════════════════
# 마크다운 저장
# ══════════════════════════════════════════════════════════════════════════════

def _write_md(d: dict):
    m  = d["measure_means"]
    sd = d["measure_stds"]
    lines = [
        "# Alpamayo 1.5 — v4 Phase-Separated Profiling",
        f"**보드**: Jetson AGX Thor | **측정**: CUDA Events (n={d['n_measure']}) | bf16",
        "",
    ]
    if d["split_ok"]:
        total = m["vision_ms"] + m["lm_prefill_ms"] + m["decode_ms"] + m["flow_ms"]
        lines += [
            "## Phase별 GPU 시간 (4단계 완전 분리)",
            "",
            "| Phase | GPU Time (ms) | ±σ | Share | BW (GB/s) | MBU% | 판정 |",
            "|---|---|---|---|---|---|---|",
            f"| Vision Encoder | {m['vision_ms']:.0f} | {sd['vision_ms']:.0f} "
            f"| {m['vision_ms']/total*100:.1f}% | — | — | compute |",
            f"| LM Prefill | {m['lm_prefill_ms']:.0f} | {sd['lm_prefill_ms']:.0f} "
            f"| {m['lm_prefill_ms']/total*100:.1f}% | — | — | compute |",
            f"| Decode | {m['decode_ms']:.0f} | {sd['decode_ms']:.0f} "
            f"| {m['decode_ms']/total*100:.1f}% "
            f"| **{m['decode_bw_GBps']:.1f}** | **{m['decode_bw_pct']:.0f}%** "
            f"| **★ BW-BOUND** |",
            f"| Flow | {m['flow_ms']:.0f} | {sd['flow_ms']:.0f} "
            f"| {m['flow_ms']/total*100:.1f}% | — | — | overhead-dom. |",
            f"| **Total** | **{total:.0f}** | | | | | |",
            "",
            f"**BW_decode** = {d['model_gb']:.3f} GB × {m['n_tok']:.0f} tokens "
            f"/ {m['decode_ms']:.0f} ms = **{m['decode_bw_GBps']:.1f} GB/s** "
            f"({m['decode_bw_pct']:.0f}% of {d['dram_bw_GBps']} GB/s)",
        ]
    else:
        lines += [
            "## Phase별 GPU 시간 (ViT/LM 미분리)",
            "",
            "| Phase | GPU Time (ms) | Share | BW | MBU% |",
            "|---|---|---|---|---|",
            f"| Prefill (ViT+LM) | {m['prefill_ms']:.0f} | — | — | — |",
            f"| Decode | {m['decode_ms']:.0f} | — "
            f"| {m['decode_bw_GBps']:.1f} | {m['decode_bw_pct']:.0f}% |",
            f"| Flow | {m['flow_ms']:.0f} | — | — | — |",
        ]
    p = OUT / "phase_v4.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {p}")


# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Vision Encoder": "#70B0D0",
    "LM Prefill"    : "#6ACC65",
    "Prefill"       : "#5DC85D",
    "Decode"        : "#D65F5F",
    "Flow"          : "#B47CC7",
    "Warmup"        : "#AAAAAA",
    "Measure"       : "#4878CF",
}


def _plot(d: dict, all_results, warmup_rs, measure_rs):
    m     = d["measure_means"]
    split = d["split_ok"]

    # ── Fig 1: Phase breakdown (stacked bar, warmup vs measure) ────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Alpamayo 1.5 on Jetson AGX Thor — v4 Phase Profiling\n"
                 "(CUDA Events, bf16, torch.autocast)",
                 fontsize=11, fontweight="bold")

    # (a) Measure 평균 bar chart
    ax = axes[0]
    ax.set_facecolor("white")
    if split:
        phases = ["Vision Encoder", "LM Prefill", "Decode", "Flow"]
        vals   = [m["vision_ms"], m["lm_prefill_ms"], m["decode_ms"], m["flow_ms"]]
    else:
        phases = ["Prefill", "Decode", "Flow"]
        vals   = [m["prefill_ms"], m["decode_ms"], m["flow_ms"]]

    cols = [COLORS[p] for p in phases]
    bars = ax.bar(phases, vals, color=cols, alpha=0.88, edgecolor="white",
                  linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 20,
                f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("GPU Time (ms)", fontsize=10)
    ax.set_title("(a) Measure runs — Phase duration", fontsize=10,
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")

    # (b) Warmup vs Measure VLM 비교
    ax = axes[1]
    ax.set_facecolor("white")
    w_means = [float(np.mean([r.vlm_ms    for r in warmup_rs])),
               float(np.mean([r.prefill_ms for r in warmup_rs])),
               float(np.mean([r.decode_ms  for r in warmup_rs]))]
    m_means = [float(np.mean([r.vlm_ms    for r in measure_rs])),
               float(np.mean([r.prefill_ms for r in measure_rs])),
               float(np.mean([r.decode_ms  for r in measure_rs]))]
    labels  = ["VLM Total", "Prefill", "Decode"]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, w_means, w, color=COLORS["Warmup"], alpha=0.82,
                label="Warmup", edgecolor="white")
    b2 = ax.bar(x + w/2, m_means, w, color=COLORS["Measure"], alpha=0.82,
                label="Measure", edgecolor="white")
    for bar, v in list(zip(b1, w_means)) + list(zip(b2, m_means)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 30,
                f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("GPU Time (ms)", fontsize=10)
    ax.set_title("(b) Warmup vs Measure\n(JIT + Allocator overhead)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")

    plt.tight_layout(pad=1.5)
    for ext in ("png", "pdf"):
        fp = FIGD / f"fig_v4_phase.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Fig] {fp}")
    plt.close(fig)

    # ── Fig 2: BW 비교 (Vision + LM_Prefill + Decode) ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Memory Bandwidth Analysis — Alpamayo 1.5 on Jetson AGX Thor",
                 fontsize=11, fontweight="bold")

    # (a) 달성 BW bar
    ax = axes[0]
    ax.set_facecolor("white")
    bw_decode = m["decode_bw_GBps"]

    if split and m["vision_ms"] > 0:
        bw_vision = MODEL_GB / (m["vision_ms"] / 1000.0)
        bw_lmpf   = MODEL_GB / (m["lm_prefill_ms"] / 1000.0) \
                    if m["lm_prefill_ms"] > 0 else 0
        cats = ["DRAM Peak", "LM Prefill\n(theory≈BW)", "Decode\n(measured)"]
        bws  = [DRAM_BW, bw_lmpf, bw_decode]
        cols = ["#AAAAAA", COLORS["LM Prefill"], COLORS["Decode"]]
    else:
        cats = ["DRAM Peak", "Decode\n(measured)"]
        bws  = [DRAM_BW, bw_decode]
        cols = ["#AAAAAA", COLORS["Decode"]]

    bars = ax.bar(cats, bws, color=cols, alpha=0.88, edgecolor="white")
    for bar, v in zip(bars, bws):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 4,
                f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(DRAM_BW * 0.70, color="orange", ls=":", lw=2,
               label="BW-bound threshold 70%")
    ax.set_ylim(0, DRAM_BW * 1.2)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=10)
    ax.set_title("(a) Phase별 DRAM bandwidth", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")

    # (b) MBU 도넛 차트
    ax = axes[1]
    ax.set_facecolor("white")
    mbu = m["decode_bw_pct"]
    size  = [mbu, 100 - mbu]
    clrs  = [COLORS["Decode"], "#EEEEEE"]
    wedges, _ = ax.pie(size, colors=clrs, startangle=90,
                       wedgeprops={"width": 0.55, "edgecolor": "white",
                                   "linewidth": 2})
    ax.text(0, 0, f"{mbu:.1f}%\nMBU",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=COLORS["Decode"])
    ax.set_title(f"(b) Decode Memory Bandwidth Utilization\n"
                 f"({m['decode_bw_GBps']:.1f} / {DRAM_BW} GB/s)",
                 fontsize=10, fontweight="bold")

    plt.tight_layout(pad=1.5)
    for ext in ("png", "pdf"):
        fp = FIGD / f"fig_v4_bw.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Fig] {fp}")
    plt.close(fig)

    # ── Fig 3: 4단계 타임라인 (measure runs) ────────────────────────────────
    if split:
        fig, ax = plt.subplots(figsize=(13, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_title("Alpamayo 1.5 — Inference Timeline (Measure runs, avg)",
                     fontsize=11, fontweight="bold")

        starts = [0,
                  m["vision_ms"],
                  m["vision_ms"] + m["lm_prefill_ms"],
                  m["vision_ms"] + m["lm_prefill_ms"] + m["decode_ms"]]
        widths = [m["vision_ms"], m["lm_prefill_ms"],
                  m["decode_ms"],  m["flow_ms"]]
        labels = ["Vision\nEncoder", "LM\nPrefill", "Decode", "Flow"]
        cols   = [COLORS["Vision Encoder"], COLORS["LM Prefill"],
                  COLORS["Decode"],         COLORS["Flow"]]

        total_w = sum(widths)
        for s, w, lb, c in zip(starts, widths, labels, cols):
            ax.barh(0, w, left=s, height=0.5, color=c, alpha=0.88,
                    edgecolor="white", linewidth=1.5)
            if w > total_w * 0.04:
                ax.text(s + w / 2, 0, f"{lb}\n{w:.0f}ms",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")

        ax.set_xlim(0, sum(widths) * 1.02)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_yticks([])
        patches = [mpatches.Patch(color=COLORS[l.replace("\n", " ")], label=l.replace("\n", " "))
                   for l in ["Vision\nEncoder", "LM\nPrefill", "Decode", "Flow"]]
        ax.legend(handles=patches, loc="upper right", fontsize=9,
                  framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", alpha=0.2, ls="--")

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fp = FIGD / f"fig_v4_timeline.{ext}"
            plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"[Fig] {fp}")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Alpamayo 1.5 v4 Phase-Separated Profiler"
    )
    ap.add_argument("--warmup", type=int, default=1,
                    help="Warmup 횟수 (JIT 컴파일 제거용, 기본 1)")
    ap.add_argument("--runs",   type=int, default=2,
                    help="측정 횟수 (기본 2)")
    ap.add_argument("--nsys",   action="store_true",
                    help="nsys 실행 모드 (안내 메시지 억제)")
    args = ap.parse_args()

    if not args.nsys:
        print("=" * 72)
        print("  [nsys 실행 명령]")
        print("  nsys profile \\")
        print("    --trace=cuda,nvtx \\")
        print("    --cuda-memory-usage=true \\")
        print("    --sample=none --cpuctxsw=none \\")
        print(f"    --output={OUT}/nsys_v4 \\")
        print(f"    python {Path(__file__).name} --warmup 1 --runs 2 --nsys")
        print("=" * 72)

    all_r, meas_r, model_gb, input_tok_len = run_profiling(
        warmup=args.warmup, runs=args.runs
    )
    summarize(meas_r, all_r, model_gb, input_tok_len)


if __name__ == "__main__":
    main()
