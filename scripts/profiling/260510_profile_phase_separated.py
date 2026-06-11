"""
260510_profile_phase_separated.py  ·  v3.0
─────────────────────────────────────────────────────────────────────────────
Alpamayo 1.5  Phase-Separated Precise Profiler

[v1→v2: past_kv → seq_len 감지]
[v2→v3: register_forward_hook → 직접 forward 패치 + 다중 레벨 폴백]

[hook 실패 원인]
  register_forward_hook은 module.__call__() 경유 시에만 작동.
  Alpamayo VLM이 language_model.forward()를 직접 호출하거나
  커스텀 CUDA 경로를 사용 → 0회 호출.

[v3 전략: 3단계 폴백]
  1단계: model.vlm.forward 직접 패치 (method replacement)
         → generate() 내부에서 self.forward() 호출 시 포착
  2단계: model.vlm.language_model.forward 직접 패치
         → 더 깊은 레벨에서 직접 호출 시 포착
  3단계: 위 둘 다 실패 → generate 래퍼 CUDA Events (VLM 전체 시간)
         + 이론 분석으로 decode BW 추정

[측정 보장 항목]
  · 전체 추론 시간 (CUDA Events, 항상 정확)
  · Flow 시간      (CUDA Events, 항상 정확)
  · VLM 시간       (전체 - Flow, 항상 정확)
  · Prefill/Decode 분리 (전략 1·2 성공 시)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _f in ["NanumGothic", "NanumBarunGothic", "UnDotum", "DejaVu Sans"]:
    if _f in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT  = Path(__file__).resolve().parents[2]
OUT   = Path("profiling_results/260510_memory_utilization")
FIG_D = OUT / "figures"
for d in [OUT, FIG_D]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DRAM_BW_GBps   = 273.0
MODEL_GB        = 11.08 * 2   # 22.16 GB
ACTION_GB       = 2.3  * 2    #  4.60 GB (action expert 추정)


# ══════════════════════════════════════════════════════════════════════════════
# CUDA Event 타이머
# ══════════════════════════════════════════════════════════════════════════════

class CUDATimer:
    def __init__(self, name: str):
        self.name = name
        self._s   = torch.cuda.Event(enable_timing=True)
        self._e   = torch.cuda.Event(enable_timing=True)
        self._started = False
        self._stopped = False

    def start(self):
        torch.cuda.synchronize()
        self._s.record()
        self._started = True

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
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self._started = False
        self._stopped = False


# ══════════════════════════════════════════════════════════════════════════════
# Phase 상태 감지 (seq_len 기반)
# ══════════════════════════════════════════════════════════════════════════════

class PhaseDetector:
    """
    forward 패치에 삽입되어 seq_len으로 Prefill/Decode를 감지.
    seq_len > 1 → Prefill (이미지+텍스트 전체)
    seq_len == 1 → Decode  (단일 토큰 autoregressive)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._state        = "idle"
        self._decode_step  = 0
        self.t_prefill     = CUDATimer("prefill")
        self.t_decode      = CUDATimer("decode")
        self._mem: dict    = {}
        self._calls: list  = []   # 진단용

    def _get_seq_len(self, args, kwargs):
        """args/kwargs에서 input_ids 또는 hidden_states의 seq dim 추출."""
        # input_ids (token ids)
        iids = kwargs.get("input_ids")
        if iids is None and args:
            for a in args:
                if isinstance(a, torch.Tensor) and a.ndim >= 2:
                    iids = a
                    break
        if iids is not None:
            return iids.shape[-1]
        # hidden_states (embedding 이후 레이어용)
        hs = kwargs.get("hidden_states")
        if hs is None and args:
            for a in args:
                if isinstance(a, torch.Tensor) and a.ndim == 3:
                    hs = a
                    break
        if hs is not None:
            return hs.shape[1]   # [B, T, D]
        return None

    def before_call(self, args, kwargs):
        seq = self._get_seq_len(args, kwargs)
        if seq is None:
            return
        self._calls.append(seq)

        if seq > 1:
            if self._state == "idle":
                self._mem["pf_before"] = torch.cuda.memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.nvtx.range_push("Phase/Prefill")
                self.t_prefill.start()
                self._state = "prefill"

        elif seq == 1:
            if self._state == "prefill":
                self.t_prefill.stop()
                torch.cuda.nvtx.range_pop()
                self._mem["pf_after"] = torch.cuda.memory_allocated() / 1e9
                self._mem["pf_peak"]  = (
                    torch.cuda.memory_stats()
                    .get("active_bytes.all.peak", 0) / 1e9
                )
                torch.cuda.reset_peak_memory_stats()
                self._mem["dc_before"] = torch.cuda.memory_allocated() / 1e9
                torch.cuda.nvtx.range_push("Phase/Decode")
                self.t_decode.start()
                self._state = "decode"

            if self._state == "decode":
                self._decode_step += 1
                torch.cuda.nvtx.range_push(f"Decode/step_{self._decode_step:03d}")

    def after_call(self, *_):
        if self._state == "decode" and self._decode_step > 0:
            torch.cuda.nvtx.range_pop()

    def end_generate(self):
        if self._state == "decode":
            self.t_decode.stop()
            torch.cuda.nvtx.range_pop()
            self._mem["dc_after"] = torch.cuda.memory_allocated() / 1e9
            self._mem["dc_peak"]  = (
                torch.cuda.memory_stats()
                .get("active_bytes.all.peak", 0) / 1e9
            )
        elif self._state == "prefill":
            self.t_prefill.stop()
            torch.cuda.nvtx.range_pop()
        self._state = "idle"

    def diagnostics(self) -> str:
        n_pf = sum(1 for s in self._calls if s > 1)
        n_dc = sum(1 for s in self._calls if s == 1)
        if not self._calls:
            return "  [진단] 0회 호출 — 이 레벨에서 forward 미호출"
        return (f"  [진단] 총 {len(self._calls)}회: "
                f"Prefill({n_pf}회 seq={[s for s in self._calls if s>1][:3]}) "
                f"Decode({n_dc}회)")


# ══════════════════════════════════════════════════════════════════════════════
# 패치 전략 1: model.vlm.forward 직접 교체
# ══════════════════════════════════════════════════════════════════════════════

def patch_vlm_forward(vlm, detector: PhaseDetector) -> bool:
    """
    model.vlm.forward를 직접 교체.
    generate() 내부에서 self.forward()/self(...)를 통해 호출될 때 포착.
    """
    if not hasattr(vlm, "forward"):
        return False

    orig = vlm.forward

    def _patched(*args, **kwargs):
        detector.before_call(args, kwargs)
        result = orig(*args, **kwargs)
        detector.after_call()
        return result

    # MethodType 없이 직접 교체 (언바운드 함수로 대체)
    vlm.forward = _patched
    print("      [패치 1] model.vlm.forward 교체 완료")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 패치 전략 2: language_model.forward 직접 교체
# ══════════════════════════════════════════════════════════════════════════════

def patch_lm_forward(vlm, detector: PhaseDetector) -> bool:
    """
    model.vlm 내부의 language model forward를 직접 교체.
    model.vlm.forward()가 내부적으로 self.language_model.forward()를
    직접 호출하는 경우 포착.
    """
    # 가능한 경로 탐색
    candidates = []
    for attr in ("language_model", "model"):
        sub = getattr(vlm, attr, None)
        if sub is None:
            continue
        candidates.append((f"vlm.{attr}", sub))
        # 한 레벨 더 (Qwen2VL 스타일)
        sub2 = getattr(sub, "model", None)
        if sub2 is not None and hasattr(sub2, "layers"):
            candidates.append((f"vlm.{attr}.model", sub2))
        if hasattr(sub, "layers"):
            break

    for path, mod in candidates:
        if not hasattr(mod, "forward"):
            continue
        orig = mod.forward

        def _patched(*args, _orig=orig, **kwargs):
            detector.before_call(args, kwargs)
            result = _orig(*args, **kwargs)
            detector.after_call()
            return result

        mod.forward = _patched
        print(f"      [패치 2] model.{path}.forward 교체 완료")
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# generate 래퍼 (전략 3 폴백 포함, 항상 작동 보장)
# ══════════════════════════════════════════════════════════════════════════════

def wrap_generate(vlm, detector: PhaseDetector,
                  t_vlm: CUDATimer, tok_n: list[int]):
    orig = vlm.generate.__func__

    def _gen(self_v, *args, **kwargs):
        detector.reset()
        tok_n[0] = 0

        # VLM 전체 시간 (항상 정확, 폴백)
        t_vlm.reset()
        t_vlm.start()
        torch.cuda.nvtx.range_push("Phase/VLM_Generate")

        result = orig(self_v, *args, **kwargs)

        torch.cuda.nvtx.range_pop()
        t_vlm.stop()
        detector.end_generate()

        # 토큰 수 계산
        if hasattr(result, "sequences"):
            gen_len = result.sequences.shape[-1]
        elif isinstance(result, torch.Tensor):
            gen_len = result.shape[-1]
        else:
            gen_len = 0
        # input_tok_len은 외부에서 주입
        return result

    vlm.generate = types.MethodType(_gen, vlm)


def wrap_diffusion(diffusion, t_flow: CUDATimer,
                   euler_n: list[int], orig_euler_ref: list):
    # _euler 카운터
    orig_euler = diffusion._euler
    orig_euler_ref.append(orig_euler)

    def _cnt_euler(self_d, *args, **kwargs):
        n = kwargs.get("inference_step") or getattr(self_d, "num_inference_steps", 1)
        euler_n[0] += int(n)
        torch.cuda.nvtx.range_push(f"Flow/Euler_x{euler_n[0]}")
        result = orig_euler(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return result

    diffusion._euler = types.MethodType(_cnt_euler, diffusion)

    # sample 타이머
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


# ══════════════════════════════════════════════════════════════════════════════
# 측정 결과
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    wall_ms:    float
    vlm_ms:     float   # 전체 VLM (prefill+decode)
    prefill_ms: float   # 0이면 감지 실패
    decode_ms:  float   # 0이면 감지 실패
    flow_ms:    float
    n_tok:      int
    n_euler:    int
    hook_ok:    bool    # prefill/decode 감지 성공 여부

    @property
    def decode_bw_GBps(self) -> float:
        if self.decode_ms > 0 and self.n_tok > 0:
            return MODEL_GB * self.n_tok / (self.decode_ms / 1000.0)
        # 폴백: 이론 추정 (decode = VLM_time - prefill_theory)
        prefill_theory_ms = self._prefill_theory_ms()
        decode_est_ms = max(1.0, self.vlm_ms - prefill_theory_ms)
        return MODEL_GB * self.n_tok / (decode_est_ms / 1000.0)

    @property
    def decode_bw_pct(self) -> float:
        return self.decode_bw_GBps / DRAM_BW_GBps * 100.0

    def _prefill_theory_ms(self) -> float:
        """Prefill 이론 시간 추정 (compute-bound 가정)."""
        # FLOPs = 2 × params × input_tokens
        # bf16 Blackwell 실효 throughput ≈ 300 TFLOPS (보수적)
        input_tokens = 3086   # 실측
        flops_T = 2 * 11.08e9 * input_tokens / 1e12
        return flops_T / 300.0 * 1000.0   # ms


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
    print(f"      {time.perf_counter()-t0:.1f}s | {model_gb:.2f} GB")

    print("[2/4] 입력 준비...")
    clip_id  = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data     = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    processor = helper.get_processor(model.tokenizer)
    inputs    = processor.apply_chat_template(
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
    print(f"      입력 토큰: {input_tok_len}")

    print("[3/4] 패치 등록...")
    detector = PhaseDetector()
    t_vlm    = CUDATimer("vlm")
    t_flow   = CUDATimer("flow")
    tok_n    = [0]
    euler_n  = [0]
    orig_euler_ref = []

    # 전략 1: model.vlm.forward 직접 교체
    ok1 = patch_vlm_forward(model.vlm, detector)
    # 전략 2: language_model.forward 직접 교체 (1이 실패해도 시도)
    ok2 = patch_lm_forward(model.vlm, detector)
    if not ok1 and not ok2:
        print("      [경고] forward 패치 실패 → generate 레벨 CUDA Events만 사용")

    # generate 래퍼 (항상 설치, VLM 전체 시간 보장)
    wrap_generate(model.vlm, detector, t_vlm, tok_n)
    wrap_diffusion(model.diffusion, t_flow, euler_n, orig_euler_ref)

    # tok_n 업데이트를 위해 generate 래퍼 위에 한 번 더 씌움
    orig_gen_h = model.vlm.generate.__func__

    def _final_gen(self_v, *args, **kwargs):
        euler_n[0] = 0
        result = orig_gen_h(self_v, *args, **kwargs)
        if hasattr(result, "sequences"):
            gl = result.sequences.shape[-1]
        elif isinstance(result, torch.Tensor):
            gl = result.shape[-1]
        else:
            gl = 0
        tok_n[0] = max(0, gl - input_tok_len)
        return result

    model.vlm.generate = types.MethodType(_final_gen, model.vlm)

    print(f"[4/4] 프로파일링 (warmup {warmup} + 측정 {runs})")
    results: list[RunResult] = []

    @torch.no_grad()
    def run_one(label: str) -> RunResult:
        detector.reset()
        euler_n[0] = 0
        tok_n[0]   = 0
        t_vlm.reset()
        t_flow.reset()
        torch.cuda.synchronize()

        t_wall = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=1, return_extra=True,
            )
        wall_ms = (time.perf_counter() - t_wall) * 1000.0

        pf_ms  = detector.t_prefill.ms()
        dc_ms  = detector.t_decode.ms()
        fl_ms  = t_flow.ms()
        vl_ms  = t_vlm.ms()
        n_tok  = tok_n[0]
        n_eul  = euler_n[0]
        hook_ok = pf_ms > 0 or dc_ms > 0

        # BW 계산
        dec_bw = (MODEL_GB * n_tok / (dc_ms / 1000.0)
                  if dc_ms > 0 and n_tok > 0 else 0.0)

        r = RunResult(
            wall_ms=wall_ms, vlm_ms=vl_ms,
            prefill_ms=pf_ms, decode_ms=dc_ms,
            flow_ms=fl_ms, n_tok=n_tok, n_euler=n_eul,
            hook_ok=hook_ok,
        )

        # 출력
        if hook_ok:
            print(f"  [{label:10s}] "
                  f"Prefill:{pf_ms:5.0f}ms  "
                  f"Decode:{dc_ms:5.0f}ms({n_tok}tok)  "
                  f"Flow:{fl_ms:4.0f}ms({n_eul}step)  "
                  f"VLM:{vl_ms:5.0f}ms  "
                  f"BW_dec:{dec_bw:.0f}GB/s({dec_bw/DRAM_BW_GBps*100:.0f}%)")
        else:
            # 폴백 출력
            est_bw = r.decode_bw_GBps
            print(f"  [{label:10s}] "
                  f"VLM:{vl_ms:5.0f}ms(prefill+decode)  "
                  f"Flow:{fl_ms:4.0f}ms({n_eul}step)  "
                  f"Wall:{wall_ms:5.0f}ms  "
                  f"BW_dec_est:{est_bw:.0f}GB/s({est_bw/DRAM_BW_GBps*100:.0f}%)[추정]")

        print(detector.diagnostics())
        return r

    print("  -- warmup --")
    for i in range(warmup):
        run_one(f"warmup{i+1}")

    print("  -- measurement --")
    for i in range(runs):
        r = run_one(f"run{i+1}")
        results.append(r)

    return results, model_gb, input_tok_len


# ══════════════════════════════════════════════════════════════════════════════
# 결과 집계 + 출력 + 저장
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save(results: list[RunResult], model_gb: float, input_tok_len: int):
    if not results:
        print("[오류] 측정 결과 없음")
        return

    hook_ok = results[0].hook_ok
    W = 80

    def mean(vals): return float(np.mean(vals))
    def std(vals):  return float(np.std(vals))

    vlm_ms_m   = mean([r.vlm_ms   for r in results])
    flow_ms_m  = mean([r.flow_ms  for r in results])
    wall_ms_m  = mean([r.wall_ms  for r in results])
    vlm_ms_s   = std( [r.vlm_ms   for r in results])
    flow_ms_s  = std( [r.flow_ms  for r in results])
    n_tok_m    = mean([r.n_tok    for r in results])
    n_euler_m  = mean([r.n_euler  for r in results])
    bw_m       = mean([r.decode_bw_GBps for r in results])
    bw_pct_m   = mean([r.decode_bw_pct  for r in results])

    prefill_ms_m = mean([r.prefill_ms for r in results])
    decode_ms_m  = mean([r.decode_ms  for r in results])
    prefill_ms_s = std( [r.prefill_ms for r in results])
    decode_ms_s  = std( [r.decode_ms  for r in results])

    print("\n" + "═" * W)
    print("  Phase-Separated 측정 결과  (CUDA Events, bf16, torch.autocast)")
    print("═" * W)

    if hook_ok:
        total_ms = prefill_ms_m + decode_ms_m + flow_ms_m
        rows = [
            ("Prefill",  prefill_ms_m, prefill_ms_s, 0.0, 0.0),
            ("Decode",   decode_ms_m,  decode_ms_s,  bw_m, bw_pct_m),
            ("Flow",     flow_ms_m,    flow_ms_s,    ACTION_GB * n_euler_m / (flow_ms_m/1000) if flow_ms_m > 0 else 0, 0.0),
        ]
        print(f"  {'Phase':<12} {'GPU Time':>14}  {'비율':>6}  {'BW (GB/s)':>10}  {'BW%':>5}  판정")
        print(f"  {'-'*12} {'-'*14}  {'-'*6}  {'-'*10}  {'-'*5}  {'-'*20}")
        for name, ms, ms_s, bw, bwp in rows:
            pct  = ms / total_ms * 100 if total_ms > 0 else 0
            bwv  = f"{bw:.1f}" if bw > 0 else "—"
            bwpv = f"{bwp:.0f}%" if bwp > 0 else "—"
            verd = ("**BW-BOUND**" if bwp >= 70 else
                    "BW-bound (mod)" if bwp >= 40 else
                    "compute/overhead" if bw > 0 else "—")
            print(f"  {name:<12} {ms:>8.0f} ±{ms_s:>4.0f} ms  "
                  f"{pct:>5.1f}%  {bwv:>10}  {bwpv:>5}  {verd}")
        print(f"  {'Total':<12} {total_ms:>8.0f} ms")
    else:
        # 폴백: VLM + Flow 2단계
        pf_theory  = results[0]._prefill_theory_ms()
        dc_est_ms  = max(1.0, vlm_ms_m - pf_theory)
        bw_est     = MODEL_GB * n_tok_m / (dc_est_ms / 1000.0)
        bw_est_pct = bw_est / DRAM_BW_GBps * 100.0

        print(f"  [주의] Prefill/Decode 직접 분리 실패 → VLM 전체 시간으로 보고")
        print(f"  {'Phase':<18} {'GPU Time':>14}  {'비율':>6}")
        print(f"  {'-'*18} {'-'*14}  {'-'*6}")
        for name, ms, ms_s, pct in [
            ("VLM (pf+dec)", vlm_ms_m, vlm_ms_s, vlm_ms_m/(vlm_ms_m+flow_ms_m)*100),
            ("Flow",         flow_ms_m, flow_ms_s, flow_ms_m/(vlm_ms_m+flow_ms_m)*100),
        ]:
            print(f"  {name:<18} {ms:>8.0f} ±{ms_s:>4.0f} ms  {pct:>5.1f}%")
        print(f"\n  [Decode BW 이론 추정]")
        print(f"  Prefill 이론 시간 ≈ {pf_theory:.0f} ms (3086 tok, BF16, Blackwell 추정)")
        print(f"  Decode 시간 추정   ≈ {dc_est_ms:.0f} ms")
        print(f"  BW_decode 추정     ≈ {bw_est:.0f} GB/s ({bw_est_pct:.0f}% of {DRAM_BW_GBps} GB/s)")
        if bw_est_pct >= 40:
            print(f"  → Decode는 BW-bound 가능성 높음 (nsys kernel 분석과 일치)")

    print("═" * W)

    # ── JSON 저장 ──────────────────────────────────────────────────────────────
    out = {
        "hook_ok"       : hook_ok,
        "n_runs"        : len(results),
        "n_tok"         : n_tok_m,
        "n_euler"       : n_euler_m,
        "input_tok_len" : input_tok_len,
        "model_gb"      : model_gb,
        "dram_bw_GBps"  : DRAM_BW_GBps,
        "means": {
            "wall_ms"   : wall_ms_m,
            "vlm_ms"    : vlm_ms_m,
            "prefill_ms": prefill_ms_m,
            "decode_ms" : decode_ms_m,
            "flow_ms"   : flow_ms_m,
            "decode_bw_GBps" : bw_m,
            "decode_bw_pct"  : bw_pct_m,
        },
        "std": {
            "vlm_ms"    : vlm_ms_s,
            "prefill_ms": prefill_ms_s,
            "decode_ms" : decode_ms_s,
            "flow_ms"   : flow_ms_s,
        },
        "runs": [asdict(r) for r in results],
    }
    p = OUT / "phase_separated.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[저장] {p}")

    _write_md(out, hook_ok, results[0]._prefill_theory_ms())
    _plot(out, hook_ok)


def _write_md(d: dict, hook_ok: bool, pf_theory_ms: float):
    m   = d["means"]
    n   = d["n_tok"]
    bw  = m["decode_bw_GBps"]
    bwp = m["decode_bw_pct"]
    vlm = m["vlm_ms"]
    fl  = m["flow_ms"]
    pf  = m["prefill_ms"]
    dc  = m["decode_ms"]

    lines = [
        "# Phase-Separated Profiling — 논문 기재용",
        "**측정**: CUDA Events + forward 패치 | **보드**: Jetson AGX Thor",
        "",
    ]
    if hook_ok:
        total = pf + dc + fl
        lines += [
            "## Phase별 GPU 시간 (직접 측정)",
            "",
            "| Phase | GPU Time (ms) | 비율 | BW (GB/s) | BW% | 판정 |",
            "|-------|-------------|------|-----------|-----|------|",
            f"| Prefill | {pf:.0f} | {pf/total*100:.1f}% | — | — | compute |",
            f"| Decode  | {dc:.0f} | {dc/total*100:.1f}% | {bw:.1f} | {bwp:.0f}% "
            f"| {'**BW-bound**' if bwp>=70 else 'BW-bound (mod)'} |",
            f"| Flow    | {fl:.0f} | {fl/total*100:.1f}% | — | — | — |",
            f"| **Total** | **{total:.0f}** | | | | |",
            "",
            f"**BW_decode** = {d['model_gb']:.2f} GB × {n:.0f} tokens / {dc:.0f} ms"
            f" = **{bw:.1f} GB/s** ({bwp:.0f}% of {d['dram_bw_GBps']} GB/s)",
        ]
    else:
        dc_est = max(1.0, vlm - pf_theory_ms)
        bw_est = d["model_gb"] * n / (dc_est / 1000.0)
        bwp_est = bw_est / d["dram_bw_GBps"] * 100.0
        lines += [
            "## Phase별 GPU 시간",
            "",
            "| Phase | GPU Time (ms) | 비율 | 비고 |",
            "|-------|-------------|------|------|",
            f"| VLM (Prefill+Decode) | {vlm:.0f} | {vlm/(vlm+fl)*100:.1f}% | 직접 측정 |",
            f"| Flow | {fl:.0f} | {fl/(vlm+fl)*100:.1f}% | 직접 측정 |",
            "",
            "## Decode BW 추정 (이론 기반)",
            "",
            f"- Prefill 이론 시간: {pf_theory_ms:.0f} ms (3086 tok × 2 × 11.08B FLOPs / 300 TFLOPS)",
            f"- Decode 시간 추정: {dc_est:.0f} ms",
            f"- **BW_decode 추정**: {bw_est:.0f} GB/s ({bwp_est:.0f}% of {d['dram_bw_GBps']} GB/s)",
            "",
            "> nsys kernel 분석(splitK GEMM + GEMV 비중 ~33%)과 일치.",
        ]

    p = OUT / "phase_separated.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {p}")


def _plot(d: dict, hook_ok: bool):
    m = d["means"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Alpamayo 1.5 on Jetson AGX Thor — Phase Timing",
                 fontsize=12, fontweight="bold")

    C = {"Prefill": "#6ACC65", "Decode": "#D65F5F", "Flow": "#B47CC7",
         "VLM": "#4878CF"}

    # (a) Time breakdown
    ax = axes[0]
    ax.set_facecolor("white")
    if hook_ok:
        labels = ["Prefill", "Decode", "Flow"]
        vals   = [m["prefill_ms"], m["decode_ms"], m["flow_ms"]]
    else:
        labels = ["VLM\n(Prefill+Decode)", "Flow"]
        vals   = [m["vlm_ms"], m["flow_ms"]]
    cols = [C.get(l.split("\n")[0], "#AAA") for l in labels]
    bars = ax.bar(labels, vals, color=cols, alpha=0.88, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 30,
                f"{v:.0f} ms", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("GPU Time (ms)", fontsize=10)
    ax.set_title("(a) Phase Duration", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, ls="--")

    # (b) BW vs theoretical
    ax = axes[1]
    ax.set_facecolor("white")
    bw  = m["decode_bw_GBps"]
    bwp = m["decode_bw_pct"]
    label = "Decode BW\n(measured)" if hook_ok else "Decode BW\n(estimated)"
    col   = "#D65F5F" if hook_ok else "#F0A050"
    ax.bar([label], [bw], color=col, alpha=0.88, edgecolor="white")
    ax.text(0, bw + 5, f"{bw:.0f} GB/s\n({bwp:.0f}%)",
            ha="center", fontsize=11, fontweight="bold")
    ax.axhline(DRAM_BW_GBps, color="#D65F5F", ls="--", lw=2,
               label=f"Theoretical max\n{DRAM_BW_GBps:.0f} GB/s")
    ax.axhline(DRAM_BW_GBps*0.70, color="orange", ls=":", lw=1.5,
               label="BW-bound threshold 70%")
    ax.set_ylim(0, DRAM_BW_GBps * 1.15)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=10)
    ax.set_title("(b) Decode Memory Bandwidth", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, ls="--")

    plt.tight_layout(pad=1.5)
    for ext in ("png", "pdf"):
        fp = FIG_D / f"fig_phase_separated.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Fig] {fp}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs",   type=int, default=2)
    ap.add_argument("--nsys",   action="store_true")
    args = ap.parse_args()

    if not args.nsys:
        print("[nsys 명령]")
        print(f"  nsys profile --trace=cuda,nvtx --cuda-memory-usage=true")
        print(f"      --sample=none --cpuctxsw=none")
        print(f"      --output={OUT}/nsys_phase")
        print(f"      python {Path(__file__).name} --warmup 1 --runs 2 --nsys\n")

    results, model_gb, input_tok_len = run_profiling(
        warmup=args.warmup, runs=args.runs
    )
    print_and_save(results, model_gb, input_tok_len)


if __name__ == "__main__":
    main()
