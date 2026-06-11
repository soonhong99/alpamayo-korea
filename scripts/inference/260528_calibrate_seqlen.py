"""
260528_calibrate_seqlen.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적:
  "prefill이 병목인가, decode가 병목인가?"를 확정하기 위해
  실제 Alpamayo 추론에서 발생하는 실제 토큰 수와 각 phase 시간을 측정한다.

배경:
  - 260513 timeline 실험: LM Prefill=1,435ms, Decode=1,886ms → decode가 병목처럼 보임
  - 260527-260528 benchmark: prefill_len=3,170 고정 사용 → prefill=4,487ms (병목처럼 보임)
  - 두 실험이 다른 수치를 보임 → 실제 입력 토큰 수가 불명확
  - 이 스크립트가 그 답을 확정한다.

측정 항목:
  [A] 토큰 수
      - input_ids 전체 길이          (processor 출력)
      - LM backbone 실제 입력 seq    (lm.forward hook)
      - decode step 수               (vlm.forward hook)

  [B] Phase 시간 (CUDA Event, ms)
      - Vision Encoder
      - LM Prefill
      - Decode (total + per-step)
      - Flow

  [C] 병목 판정
      - 각 phase의 비율 + BW 분석

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  python3 scripts/inference/260528_calibrate_seqlen.py

결과:
  터미널 출력 + profiling_results/260528_calibrate/results.json
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = Path("profiling_results/260528_calibrate")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE    = "cuda"
MODEL_GB  = 22.157   # BF16 전체 모델 크기
DRAM_BW   = 231.0    # GB/s (실측)

NUM_WARMUP   = 1
NUM_MEASURE  = 3


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Event 타이머
# ─────────────────────────────────────────────────────────────────────────────

class CUDATimer:
    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self._started = False
        self._stopped = False

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
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self._started = False
        self._stopped = False


# ─────────────────────────────────────────────────────────────────────────────
# Phase Detector — 토큰 수 + 타이밍 캡처
# ─────────────────────────────────────────────────────────────────────────────

class PhaseDetector:
    """
    profile_v4의 PhaseDetectorV4와 동일한 로직 +
    LM 실제 prefill seq_len 명시적 기록 추가.
    """
    IDLE, VISION, LM_PREFILL, POST_PREFILL, DECODE = \
        "idle", "vision", "lm_prefill", "post_prefill", "decode"

    def __init__(self):
        self.state           = self.IDLE
        self.t_vision        = CUDATimer()
        self.t_lm_prefill    = CUDATimer()
        self.t_decode        = CUDATimer()
        self.decode_steps    = 0

        # ★ 핵심 측정값
        self.lm_prefill_seqlen : int | None = None  # LM이 실제로 받은 seq 길이
        self.decode_seqlen_log : list[int]  = []    # decode step별 seq 길이 (검증용)

        self._lm_patched = False

    def reset(self):
        self.state           = self.IDLE
        self.t_vision.reset()
        self.t_lm_prefill.reset()
        self.t_decode.reset()
        self.decode_steps    = 0
        self.lm_prefill_seqlen = None
        self.decode_seqlen_log = []

    # ── seq_len 추출 ──────────────────────────────────────────────────────────
    @staticmethod
    def _extract_seq(args, kwargs) -> int | None:
        for src in [
            kwargs.get("input_ids"),
            kwargs.get("hidden_states"),
            kwargs.get("inputs_embeds"),
            *(a for a in args if isinstance(a, torch.Tensor)),
        ]:
            if src is None:
                continue
            if not isinstance(src, torch.Tensor):
                continue
            if src.ndim == 2:   # [B, T]
                return int(src.shape[-1])
            if src.ndim == 3:   # [B, T, D]
                return int(src.shape[1])
        return None

    # ── vlm.forward hooks ─────────────────────────────────────────────────────
    def on_vlm_before(self, args, kwargs):
        seq = self._extract_seq(args, kwargs)
        if seq is None:
            return

        if seq > 1 and self.state == self.IDLE:
            self.t_vision.start()
            self.state = self.VISION

        elif seq == 1:
            if self.state == self.POST_PREFILL:
                self.t_decode.start()
                self.state = self.DECODE
                self.decode_steps = 1
                self.decode_seqlen_log.append(seq)
            elif self.state == self.DECODE:
                self.decode_steps += 1
                self.decode_seqlen_log.append(seq)

    def on_vlm_after(self):
        pass  # decode step 종료 처리는 lm_after에서

    # ── lm.forward hooks ──────────────────────────────────────────────────────
    def on_lm_before(self, args, kwargs):
        seq = self._extract_seq(args, kwargs)
        if seq is None:
            return
        self._lm_patched = True

        if seq > 1 and self.state == self.VISION:
            # Vision Encoder 종료 → LM Prefill 시작
            self.t_vision.stop()
            self.t_lm_prefill.start()
            self.state = self.LM_PREFILL
            # ★ 실제 LM prefill 토큰 수 기록
            self.lm_prefill_seqlen = seq
            print(f"    [hook] LM prefill 실제 seq_len = {seq}")

    def on_lm_after(self):
        if self.state == self.LM_PREFILL:
            self.t_lm_prefill.stop()
            self.state = self.POST_PREFILL

    # ── generate 종료 ─────────────────────────────────────────────────────────
    def end_generate(self):
        if self.state == self.DECODE:
            self.t_decode.stop()
        elif self.state == self.LM_PREFILL:
            self.t_lm_prefill.stop()
        elif self.state == self.VISION:
            self.t_vision.stop()
        self.state = self.IDLE


# ─────────────────────────────────────────────────────────────────────────────
# 모델 패치
# ─────────────────────────────────────────────────────────────────────────────

def patch_vlm_forward(vlm, det: PhaseDetector):
    orig = vlm.forward
    def _patched(*args, **kwargs):
        det.on_vlm_before(args, kwargs)
        result = orig(*args, **kwargs)
        det.on_vlm_after()
        return result
    vlm.forward = _patched
    print("  [패치] vlm.forward ✓")


def patch_lm_forward(vlm, det: PhaseDetector) -> bool:
    """LM backbone (Qwen2 transformer) forward 패치."""
    for attr in ("language_model", "model"):
        mod = getattr(vlm, attr, None)
        if mod is None:
            continue

        # 직접 레이어가 있는 경우
        if hasattr(mod, "forward") and hasattr(mod, "layers"):
            orig = mod.forward
            def _patched(*args, _o=orig, **kwargs):
                det.on_lm_before(args, kwargs)
                result = _o(*args, **kwargs)
                det.on_lm_after()
                return result
            mod.forward = _patched
            print(f"  [패치] vlm.{attr}.forward ✓")
            return True

        # 한 단계 더 내려가기
        sub = getattr(mod, "model", None)
        if sub is not None and hasattr(sub, "forward") and hasattr(sub, "layers"):
            orig = sub.forward
            def _patched(*args, _o=orig, **kwargs):
                det.on_lm_before(args, kwargs)
                result = _o(*args, **kwargs)
                det.on_lm_after()
                return result
            sub.forward = _patched
            print(f"  [패치] vlm.{attr}.model.forward ✓")
            return True

    print("  [경고] lm.forward 패치 실패 — LM/ViT 분리 불가")
    return False


def wrap_generate(vlm, det: PhaseDetector,
                  t_vlm: CUDATimer, t_flow: CUDATimer,
                  tok_counter: list[int], input_tok_len: int):
    orig = vlm.generate.__func__

    def _gen(self_v, *args, **kwargs):
        det.reset()
        tok_counter[0] = 0
        t_vlm.reset()
        t_vlm.start()
        result = orig(self_v, *args, **kwargs)
        t_vlm.stop()
        det.end_generate()

        if hasattr(result, "sequences"):
            gl = result.sequences.shape[-1]
        elif isinstance(result, torch.Tensor):
            gl = result.shape[-1]
        else:
            gl = 0
        tok_counter[0] = max(0, gl - input_tok_len)
        return result

    vlm.generate = types.MethodType(_gen, vlm)
    print("  [패치] vlm.generate ✓")


def wrap_diffusion(diffusion, t_flow: CUDATimer):
    orig = diffusion.sample
    def _timed(*args, **kwargs):
        t_flow.reset()
        t_flow.start()
        result = orig(*args, **kwargs)
        t_flow.stop()
        return result
    diffusion.sample = _timed
    print("  [패치] diffusion.sample ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 단일 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_once(tag: str, model, model_inputs, det: PhaseDetector,
             t_vlm: CUDATimer, t_flow: CUDATimer,
             tok_counter: list[int]) -> dict:

    torch.cuda.synchronize()
    t_wall = time.perf_counter()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98, temperature=0.6,
            num_traj_samples=1, return_extra=True,
        )

    wall_ms = (time.perf_counter() - t_wall) * 1000.0

    vision_ms   = det.t_vision.ms()
    lm_pf_ms    = det.t_lm_prefill.ms()
    decode_ms   = det.t_decode.ms()
    flow_ms     = t_flow.ms()
    vlm_ms      = t_vlm.ms()
    n_decode    = det.decode_steps
    lm_seq      = det.lm_prefill_seqlen
    pf_ms       = vision_ms + lm_pf_ms if (vision_ms > 0 and lm_pf_ms > 0) \
                  else (vlm_ms - decode_ms)
    total_ms    = vision_ms + lm_pf_ms + decode_ms + flow_ms

    # decode BW
    decode_bw   = MODEL_GB * n_decode / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
    per_step_ms = decode_ms / n_decode if n_decode > 0 else 0.0

    print(f"\n  [{tag}]")
    print(f"    Vision Encoder : {vision_ms:7.0f} ms")
    print(f"    LM Prefill     : {lm_pf_ms:7.0f} ms   (LM seq_len = {lm_seq})")
    print(f"    Decode         : {decode_ms:7.0f} ms   "
          f"({n_decode} steps × {per_step_ms:.1f} ms/step, "
          f"BW={decode_bw:.1f} GB/s = {decode_bw/DRAM_BW*100:.0f}% MBU)")
    print(f"    Flow           : {flow_ms:7.0f} ms")
    print(f"    ─────────────────────────────────────")
    print(f"    Total (phases) : {total_ms:7.0f} ms")
    print(f"    VLM only       : {vlm_ms:7.0f} ms   Wall: {wall_ms:.0f} ms")

    if total_ms > 0:
        print(f"\n    병목 분석:")
        print(f"      Vision  {vision_ms/total_ms*100:5.1f}%  "
              f"LM Prefill {lm_pf_ms/total_ms*100:5.1f}%  "
              f"Decode {decode_ms/total_ms*100:5.1f}%  "
              f"Flow {flow_ms/total_ms*100:5.1f}%")

    return {
        "tag":           tag,
        "wall_ms":       round(wall_ms, 1),
        "vlm_ms":        round(vlm_ms, 1),
        "vision_ms":     round(vision_ms, 1),
        "lm_prefill_ms": round(lm_pf_ms, 1),
        "decode_ms":     round(decode_ms, 1),
        "flow_ms":       round(flow_ms, 1),
        "total_phase_ms":round(total_ms, 1),
        "lm_prefill_seqlen": lm_seq,
        "decode_steps":  n_decode,
        "decode_ms_per_step": round(per_step_ms, 2),
        "decode_bw_GBps":round(decode_bw, 1),
        "decode_mbu_pct":round(decode_bw / DRAM_BW * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Alpamayo 실제 seq_len 캘리브레이션 실험")
    print("  목적: 실제 추론에서 LM이 받는 토큰 수 확정")
    print("=" * 70)

    # ── 1. 모델 로드 ──────────────────────────────────────────────────────────
    print("\n[1/4] 모델 로드...")
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    t0 = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
        # attn_implementation 미지정 → sdpa 기본값 사용
        # ★ eager를 쓰면 seq_len=3086에서 LM Prefill이 3,753ms → 1,435ms로 2.6× 느려짐
        # eager는 StaticCache(KV 가방) 커스텀 경로에서만 사용할 것
    ).cuda().eval()
    torch.cuda.synchronize()
    print(f"  완료: {time.perf_counter()-t0:.1f}s")

    # ── 2. 입력 준비 ──────────────────────────────────────────────────────────
    print("\n[2/4] 실제 입력 준비...")
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data    = load_physical_aiavdataset(clip_id, t0_us=5_100_000)

    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    processor = helper.get_processor(model.tokenizer)
    inputs    = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device({
        "tokenized_data":   inputs,
        "ego_history_xyz":  data["ego_history_xyz"],
        "ego_history_rot":  data["ego_history_rot"],
    }, "cuda")

    input_tok_len = int(inputs["input_ids"].shape[-1])

    # ★ 가장 중요한 수치: processor가 만든 input_ids 길이
    print(f"\n  ★ processor input_ids 길이 = {input_tok_len} 토큰")
    print(f"     (260527-260528 benchmark에서 가정한 prefill_len=3170 과 비교)")
    print(f"     image_frames shape: {data['image_frames'].shape}")
    print(f"     camera_indices:     {data['camera_indices']}")

    # ── 3. 패치 등록 ──────────────────────────────────────────────────────────
    print("\n[3/4] 패치 등록...")
    det         = PhaseDetector()
    t_vlm       = CUDATimer()
    t_flow      = CUDATimer()
    tok_counter = [0]

    patch_vlm_forward(model.vlm, det)
    patch_lm_forward(model.vlm, det)
    wrap_generate(model.vlm, det, t_vlm, t_flow, tok_counter, input_tok_len)
    wrap_diffusion(model.diffusion, t_flow)

    # ── 4. 측정 ───────────────────────────────────────────────────────────────
    print(f"\n[4/4] 측정 시작 (warmup={NUM_WARMUP}, measure={NUM_MEASURE})")
    all_results = []

    print(f"\n  ── Warmup ({NUM_WARMUP}회) ──────────────────")
    for i in range(NUM_WARMUP):
        r = run_once(f"warmup_{i+1:02d}", model, model_inputs,
                     det, t_vlm, t_flow, tok_counter)
        all_results.append(r)

    print(f"\n  ── Measure ({NUM_MEASURE}회) ────────────────")
    measure_results = []
    for i in range(NUM_MEASURE):
        r = run_once(f"run_{i+1:02d}", model, model_inputs,
                     det, t_vlm, t_flow, tok_counter)
        all_results.append(r)
        measure_results.append(r)

    # ── 5. 집계 ───────────────────────────────────────────────────────────────
    def avg(key):
        return sum(r[key] for r in measure_results
                   if isinstance(r.get(key), (int, float))) / len(measure_results)

    vision_avg   = avg("vision_ms")
    lmpf_avg     = avg("lm_prefill_ms")
    decode_avg   = avg("decode_ms")
    flow_avg     = avg("flow_ms")
    total_avg    = vision_avg + lmpf_avg + decode_avg + flow_avg
    steps_avg    = avg("decode_steps")
    per_step_avg = avg("decode_ms_per_step")
    lm_seq       = measure_results[0].get("lm_prefill_seqlen")

    W = 70
    print(f"\n{'═'*W}")
    print("  ★ 최종 캘리브레이션 결과 (measure 평균)")
    print(f"{'═'*W}")
    print(f"  processor input_ids 길이  : {input_tok_len} 토큰")
    print(f"  LM 실제 prefill seq_len   : {lm_seq} 토큰")
    print(f"  decode steps              : {steps_avg:.1f} 회")
    print()
    print(f"  {'Phase':<20} {'ms':>8}   {'비율':>6}")
    print(f"  {'-'*20} {'-'*8}   {'-'*6}")
    print(f"  {'Vision Encoder':<20} {vision_avg:>8.0f}ms  "
          f"{vision_avg/total_avg*100:>5.1f}%")
    print(f"  {'LM Prefill':<20} {lmpf_avg:>8.0f}ms  "
          f"{lmpf_avg/total_avg*100:>5.1f}%")
    print(f"  {'Decode':<20} {decode_avg:>8.0f}ms  "
          f"{decode_avg/total_avg*100:>5.1f}%  "
          f"({per_step_avg:.1f}ms/step × {steps_avg:.0f})")
    print(f"  {'Flow':<20} {flow_avg:>8.0f}ms  "
          f"{flow_avg/total_avg*100:>5.1f}%")
    print(f"  {'─'*20} {'─'*8}")
    print(f"  {'Total':<20} {total_avg:>8.0f}ms")

    # 병목 판정
    bottleneck = max(
        [("Vision Encoder", vision_avg),
         ("LM Prefill",     lmpf_avg),
         ("Decode",         decode_avg),
         ("Flow",           flow_avg)],
        key=lambda x: x[1]
    )
    print(f"\n  → 실제 병목: {bottleneck[0]} ({bottleneck[1]:.0f}ms)")

    # benchmark 가정과 비교
    print(f"\n  ── benchmark 가정(3,170 tokens)과의 비교 ──────────────────────")
    print(f"  실제 LM seq_len    = {lm_seq} 토큰")
    print(f"  benchmark 가정     = 3,170 토큰")
    if lm_seq:
        ratio = 3170 / lm_seq
        print(f"  비율               = {ratio:.1f}×  "
              f"({'과장됨' if ratio > 1.2 else '거의 동일'})")
        if ratio > 1.5:
            print(f"  ⚠️  benchmark prefill_len이 실제보다 {ratio:.1f}배 길다.")
            print(f"      실제 prefill ≈ {lmpf_avg / ratio:.0f}ms (추정)")

    print(f"{'═'*W}")

    # ── 6. JSON 저장 ──────────────────────────────────────────────────────────
    output = {
        "input_tok_len":     input_tok_len,
        "lm_prefill_seqlen": lm_seq,
        "benchmark_assumed_seqlen": 3170,
        "measure_avg": {
            "vision_ms":      round(vision_avg, 1),
            "lm_prefill_ms":  round(lmpf_avg, 1),
            "decode_ms":      round(decode_avg, 1),
            "flow_ms":        round(flow_avg, 1),
            "total_ms":       round(total_avg, 1),
            "decode_steps":   round(steps_avg, 1),
            "decode_ms_per_step": round(per_step_avg, 2),
        },
        "all_runs": all_results,
    }
    out_path = OUT / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    main()
