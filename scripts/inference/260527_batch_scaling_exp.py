"""
260527_batch_scaling_exp.py — num_traj_samples 배치 확장 실험
=============================================================
목적:
  num_traj_samples 1 → 8 로 늘리면서 VLM decode의 memory-bound 탈출 여부를
  실측으로 확인한다.

  이론:
    seq=1 (num_traj_samples=1): DRAM BW bound, 가중치 386 MB 읽기 = 2.62ms/layer
    seq=8 (num_traj_samples=8): 가중치 1번 로드로 8 시퀀스 동시 처리
    → 샘플당 유효 BW 8배 향상 → compute-bound 전환 가능

측정 항목:
  - t_total   : 전체 inference (VLM generate + Action Expert)
  - t_vlm     : VLM generate() 단독
  - t_action  : Action Expert (diffusion) 단독
  - n_tokens  : 실제 생성된 decode step 수
  - minADE    : 궤적 품질

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5
  python3 scripts/inference/260527_batch_scaling_exp.py

결과:
  profiling_results/260527_batch_scaling/results.json
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

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

# ──────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
DEVICE = "cuda"
NUM_WARMUP = 2
NUM_MEASURE = 5
SAMPLES_SWEEP = [1, 2, 4, 6, 8]
MAX_GENERATION_LENGTH = 256

OUT = Path("profiling_results/260527_batch_scaling")
OUT.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# 타이밍 유틸리티
# ──────────────────────────────────────────────────────────────────

class CudaStopwatch:
    """CUDA Event 기반 정밀 타이머."""

    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self._started = False

    def start(self):
        self._s.record()
        self._started = True

    def stop(self) -> float:
        """ms 단위로 반환 (synchronize 후 호출할 것)."""
        if not self._started:
            return 0.0
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)  # ms


def make_timing_hooks(model: Alpamayo1_5):
    """
    model.vlm.generate 와 model.diffusion.sample 을 감싸
    각각의 GPU-side 실행 시간을 ms로 기록하는 hook을 설치한다.

    Returns:
        timing_state: dict { 'vlm_ms': float, 'action_ms': float,
                              'n_tokens': int }
        restore_fn: 원본 복원 함수
    """
    timing_state: dict = {"vlm_ms": 0.0, "action_ms": 0.0, "n_tokens": 0}

    _orig_generate = model.vlm.generate
    _orig_sample = model.diffusion.sample

    def _wrapped_generate(*args, **kwargs):
        sw = CudaStopwatch()
        sw.start()
        result = _orig_generate(*args, **kwargs)
        timing_state["vlm_ms"] = sw.stop()

        # 실제 decode step 수 계산:
        # result.sequences: [B*num_return_sequences, prefill_len + new_tokens]
        # input_ids 길이를 kwargs에서 추출
        # NOTE: 'or' 연산자는 tensor의 truth value를 평가 → RuntimeError
        #       반드시 'is None' 체크를 써야 함
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            timing_state["n_tokens"] = (
                result.sequences.shape[1] - input_ids.shape[1]
            )
        return result

    def _wrapped_sample(*args, **kwargs):
        sw = CudaStopwatch()
        sw.start()
        result = _orig_sample(*args, **kwargs)
        timing_state["action_ms"] = sw.stop()
        return result

    model.vlm.generate = _wrapped_generate
    model.diffusion.sample = _wrapped_sample

    def restore():
        model.vlm.generate = _orig_generate
        model.diffusion.sample = _orig_sample

    return timing_state, restore


# ──────────────────────────────────────────────────────────────────
# 단일 inference 실행 및 타이밍 측정
# ──────────────────────────────────────────────────────────────────

def run_single_inference(
    model: Alpamayo1_5,
    model_inputs: dict,
    timing_state: dict,
    num_traj_samples: int,
) -> dict:
    """
    1회 inference 실행. CUDA Event + wall clock 둘 다 측정.

    Returns: dict with keys t_total_ms, t_vlm_ms, t_action_ms,
             n_tokens, pred_xyz, pred_rot
    """
    # model_inputs를 deepcopy (함수 내부에서 pop() 호출하므로)
    data_copy = copy.deepcopy(model_inputs)

    sw_total = CudaStopwatch()
    torch.cuda.synchronize()
    sw_total.start()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
            data=data_copy,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=MAX_GENERATION_LENGTH,
        )

    t_total_ms = sw_total.stop()

    return {
        "t_total_ms": t_total_ms,
        "t_vlm_ms": timing_state["vlm_ms"],
        "t_action_ms": timing_state["action_ms"],
        "n_tokens": timing_state["n_tokens"],
        "pred_xyz": pred_xyz,
        "pred_rot": pred_rot,
    }


# ──────────────────────────────────────────────────────────────────
# minADE 계산
# ──────────────────────────────────────────────────────────────────

def compute_min_ade(pred_xyz: torch.Tensor, gt_xyz_np: np.ndarray) -> float:
    """
    pred_xyz: [B=1, num_traj_sets=1, num_traj_samples, T, 3]
    gt_xyz_np: [T, 3] (numpy, world frame)
    Returns: minADE in meters
    """
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]  # [N, T, 2]
    gt_xy = gt_xyz_np[:, :2]  # [T, 2]
    # ADE per sample
    diff = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)  # [N, T]
    ade = diff.mean(axis=-1)  # [N]
    return float(ade.min())


# ──────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────

def main():
    # ── 데이터 로드 ────────────────────────────────────────────────
    logger.info(f"Loading dataset: clip_id={CLIP_ID}")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)

    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    # ── 모델 로드 ──────────────────────────────────────────────────
    logger.info("Loading Alpamayo1_5 model...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    logger.info("Model loaded.")

    # ── 입력 준비 ──────────────────────────────────────────────────
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

    # GT trajectory for minADE
    gt_xyz_np = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :]  # [T, 3]

    # prefill 길이 로그
    input_ids = inputs["input_ids"]
    logger.info(f"Prefill token count: {input_ids.shape[1]}")

    # ── 타이밍 훅 설치 ────────────────────────────────────────────
    timing_state, restore_hooks = make_timing_hooks(model)

    # ── 실험 루프 ─────────────────────────────────────────────────
    all_results = {}
    torch.cuda.manual_seed_all(42)

    for num_samples in SAMPLES_SWEEP:
        logger.info(f"\n{'='*60}")
        logger.info(f"num_traj_samples = {num_samples}")
        logger.info(f"{'='*60}")

        run_times = []

        for run_idx in range(NUM_WARMUP + NUM_MEASURE):
            is_warmup = run_idx < NUM_WARMUP
            tag = f"WARMUP {run_idx+1}/{NUM_WARMUP}" if is_warmup else \
                  f"MEASURE {run_idx-NUM_WARMUP+1}/{NUM_MEASURE}"

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            result = run_single_inference(
                model, model_inputs, timing_state, num_samples
            )

            t_total = result["t_total_ms"]
            t_vlm = result["t_vlm_ms"]
            t_action = result["t_action_ms"]
            n_tok = result["n_tokens"]

            min_ade = compute_min_ade(result["pred_xyz"], gt_xyz_np)

            logger.info(
                f"[{tag}] total={t_total:.1f}ms  VLM={t_vlm:.1f}ms  "
                f"action={t_action:.1f}ms  tokens={n_tok}  minADE={min_ade:.4f}m"
            )

            if not is_warmup:
                run_times.append({
                    "t_total_ms": t_total,
                    "t_vlm_ms": t_vlm,
                    "t_action_ms": t_action,
                    "n_tokens": n_tok,
                    "min_ade": min_ade,
                })

        # 통계
        t_totals = [r["t_total_ms"] for r in run_times]
        t_vlms = [r["t_vlm_ms"] for r in run_times]
        t_actions = [r["t_action_ms"] for r in run_times]
        n_toks = [r["n_tokens"] for r in run_times]
        ades = [r["min_ade"] for r in run_times]

        summary = {
            "num_traj_samples": num_samples,
            "t_total_mean_ms": float(np.mean(t_totals)),
            "t_total_std_ms": float(np.std(t_totals)),
            "t_vlm_mean_ms": float(np.mean(t_vlms)),
            "t_vlm_std_ms": float(np.std(t_vlms)),
            "t_action_mean_ms": float(np.mean(t_actions)),
            "t_action_std_ms": float(np.std(t_actions)),
            "t_per_sample_mean_ms": float(np.mean(t_totals) / num_samples),
            "n_tokens_mean": float(np.mean(n_toks)),
            "min_ade_mean": float(np.mean(ades)),
            "min_ade_best": float(np.min(ades)),
            "raw": run_times,
        }
        all_results[str(num_samples)] = summary

        logger.info(
            f"  → t_total: {summary['t_total_mean_ms']:.1f} ± {summary['t_total_std_ms']:.1f} ms"
        )
        logger.info(
            f"  → t_per_sample: {summary['t_per_sample_mean_ms']:.1f} ms/sample"
        )
        logger.info(
            f"  → minADE (mean): {summary['min_ade_mean']:.4f} m  "
            f"(best run: {summary['min_ade_best']:.4f} m)"
        )

    restore_hooks()

    # ── 결과 저장 ─────────────────────────────────────────────────
    out_json = OUT / "results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {out_json}")

    # ── 결과 테이블 출력 ──────────────────────────────────────────
    logger.info("\n" + "="*80)
    logger.info("BATCH SCALING RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(
        f"{'Samples':>8} | {'t_total(ms)':>12} | {'t_VLM(ms)':>10} | "
        f"{'t_action(ms)':>13} | {'ms/sample':>10} | {'n_tokens':>9} | "
        f"{'minADE(m)':>10} | {'VLM speedup':>12}"
    )
    logger.info("-"*100)

    baseline_vlm = all_results["1"]["t_vlm_mean_ms"]
    for num_samples in SAMPLES_SWEEP:
        r = all_results[str(num_samples)]
        vlm_speedup = baseline_vlm / r["t_vlm_mean_ms"]
        logger.info(
            f"{num_samples:>8} | {r['t_total_mean_ms']:>10.1f}ms | "
            f"{r['t_vlm_mean_ms']:>8.1f}ms | "
            f"{r['t_action_mean_ms']:>11.1f}ms | "
            f"{r['t_per_sample_mean_ms']:>8.1f}ms | "
            f"{r['n_tokens_mean']:>8.1f}  | "
            f"{r['min_ade_mean']:>9.4f}m | "
            f"{vlm_speedup:>10.2f}x"
        )
    logger.info("="*100)
    logger.info(
        "Note: 'VLM speedup' = speedup of VLM decode relative to num_samples=1.\n"
        "      Ideal (memory-bound): VLM time stays constant → samples × speedup.\n"
        "      Actual: time increases sublinearly → per-sample speedup."
    )


if __name__ == "__main__":
    main()
