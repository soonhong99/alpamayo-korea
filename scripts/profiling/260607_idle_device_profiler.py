"""
260607_idle_device_profiler.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적:
  Alpamayo 1.5 decode path에서 "놀고 있는 device"를 찾는다.

측정 항목:
  ① 블록별 × 연산자별 시간      (torch.profiler + CUDA Event hook)
  ② 커널 간 갭 (inter-kernel idle time)  (CUDA Event 직접 측정)
  ③ SM 활용률 추정              (이론 FLOPs vs 실제 시간 비교)
  ④ DRAM 활용률 추정            (실측 MB ÷ 실측 ms vs 231 GB/s)
  ⑤ Q/K/V 병렬화 잠재력         (q_proj 시간 vs k+v 시간 비교)
  ⑥ gate/up 병렬화 잠재력       (gate_proj 시간 vs up_proj 비교)

핵심 의문:
  - Tensor Core는 몇 % 사용되는가? (이론: ~1%)
  - 커널 사이 Python dispatch 오버헤드는 얼마인가?
  - Q/K/V를 동시에 실행하면 얼마나 빨라지는가?
  - 현재 DRAM fetch와 compute가 직렬인가, 중첩되는가?

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5
  python3 scripts/profiling/260607_idle_device_profiler.py

결과 파일:
  profiling_results/260607_idle_device/
    ├── summary.json          — 핵심 수치 요약
    ├── per_block_ops.json    — 블록별 연산자별 시간
    ├── inter_kernel_gaps.json — 커널 간 갭
    └── chrome_trace.json     — Chrome chrome://tracing 에서 시각화 가능
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_ID      = "nvidia/Alpamayo-1.5-10B"
WARMUP_RUNS   = 2
MEASURE_RUNS  = 3
DECODE_STEPS  = 10          # 전체 17 steps 대신 10만 측정 (시간 절약)

# Thor 하드웨어 상수 (260526_prefetch_effect_test.py 실측)
DRAM_BW_GBs   = 231.0       # GB/s
L2_BW_GBs     = 1126.0      # GB/s
L2_SIZE_MB    = 32.0        # MB

# Qwen3VL 모델 상수 (260524_layer_compute_profile.py 확인됨)
HIDDEN_SIZE       = 4096
NUM_Q_HEADS       = 32
NUM_KV_HEADS      = 8
HEAD_DIM          = 128
FFN_INTERMEDIATE  = 12288   # gate/up/down 각 100.7 MB 실측 기반 ~12288

# 레이어당 가중치 MB (실측값, alignment 포함)
WEIGHT_MB = {
    "q_proj":    33.6,
    "k_proj":     8.4,
    "v_proj":     8.4,
    "o_proj":    33.6,
    "gate_proj": 100.7,
    "up_proj":   100.7,
    "down_proj": 100.7,
}
TOTAL_WEIGHT_MB_PER_BLOCK = sum(WEIGHT_MB.values())  # ≈ 386 MB

OUT = Path("profiling_results/260607_idle_device")
OUT.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logger.info("=" * 65)
logger.info("Alpamayo Idle Device Profiler — 260607")
logger.info("=" * 65)
logger.info(f"[1/5] 모델 로딩: {MODEL_ID}")

try:
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    model = Alpamayo1_5.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",   # sdpa = 실제 운용 방식
        local_files_only=True,
    ).cuda().eval()
    logger.info("  모델 로드 완료")
except Exception as exc:
    logger.error(f"  모델 로드 실패: {exc}")
    traceback.print_exc()
    sys.exit(1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LM 모델 및 레이어 접근
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_lm_model_and_layers(model):
    """확인된 Alpamayo 1.5 구조로 LM 모델 및 레이어 반환."""
    try:
        lm_model = model.vlm.language_model.model
        layers   = list(model.vlm.language_model.model.layers)
        return lm_model, layers
    except AttributeError:
        pass
    # fallback
    for attr in ["vlm.model", "vlm.language_model"]:
        try:
            m = eval(f"model.{attr}")
            if hasattr(m, "layers"):
                return m, list(m.layers)
        except AttributeError:
            continue
    raise RuntimeError("decoder layers 탐색 실패")

lm_model, layers = get_lm_model_and_layers(model)
N_LAYERS = len(layers)
logger.info(f"  LM layers: {N_LAYERS}개 ({type(layers[0]).__name__})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 더미 입력 (decode 모드: seq_len=1, context=3086)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE  = "cuda"
DTYPE   = torch.bfloat16
SEQ_LEN = 3086  # prefill context
DEC_LEN = 1     # decode: 1 token at a time

# prefill용 더미 hidden
dummy_prefill = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
dummy_decode  = torch.randn(1, DEC_LEN, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)

# mrope position_ids
pos_prefill = torch.zeros(3, 1, SEQ_LEN, dtype=torch.long, device=DEVICE)
pos_decode  = torch.zeros(3, 1, DEC_LEN, dtype=torch.long, device=DEVICE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ① 연산자별 CUDA Event 타이밍 (hook 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logger.info("\n[2/5] 연산자별 타이밍 측정 (CUDA Event hook) ...")

# 측정할 서브모듈 이름 목록
OP_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn",        # attention 전체 (QKV + Attn + O)
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp",              # FFN 전체
]

def get_submodule(layer, path: str):
    """'self_attn.q_proj' → layer.self_attn.q_proj"""
    obj = layer
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def measure_op_times(
    lm_model, layers,
    seq_mode: str,          # "prefill" or "decode"
    n_warmup: int = 2,
    n_measure: int = 5,
) -> dict[str, list[float]]:
    """
    각 블록의 연산자별 ms 측정.

    반환: {
        "layer_0_self_attn.q_proj": [ms, ms, ...],
        "layer_0_self_attn": [...],
        "layer_1_self_attn.q_proj": [...],
        ...
        "inter_kernel_gap_layer_0": [...],  # 레이어 시작~종료 사이의 갭 추정
    }
    """
    # 더미 입력 선택
    dummy_h = dummy_prefill if seq_mode == "prefill" else dummy_decode
    pos_ids = pos_prefill   if seq_mode == "prefill" else pos_decode

    # CUDA Event 준비: 레이어 × 연산자
    start_evts: dict[str, list] = {}
    end_evts:   dict[str, list] = {}

    for i, layer in enumerate(layers):
        for op_name in OP_NAMES:
            key = f"layer_{i}_{op_name}"
            start_evts[key] = [torch.cuda.Event(enable_timing=True)
                               for _ in range(n_measure)]
            end_evts[key]   = [torch.cuda.Event(enable_timing=True)
                               for _ in range(n_measure)]

    run_idx = [0]   # mutable counter in closure

    # Hook 등록
    handles = []
    for i, layer in enumerate(layers):
        for op_name in OP_NAMES:
            try:
                submod = get_submodule(layer, op_name)
            except AttributeError:
                continue
            key = f"layer_{i}_{op_name}"

            def make_hooks(k, idx_ref):
                def pre(module, args):
                    if idx_ref[0] < len(start_evts[k]):
                        start_evts[k][idx_ref[0]].record()
                def post(module, args, output):
                    if idx_ref[0] < len(end_evts[k]):
                        end_evts[k][idx_ref[0]].record()
                return pre, post

            pre_fn, post_fn = make_hooks(key, run_idx)
            handles.append(submod.register_forward_pre_hook(pre_fn))
            handles.append(submod.register_forward_hook(post_fn))

    results: dict[str, list[float]] = {}

    try:
        total_runs = n_warmup + n_measure
        for run in range(total_runs):
            is_warmup = (run < n_warmup)
            if not is_warmup:
                run_idx[0] = run - n_warmup

            with torch.no_grad():
                lm_model(
                    inputs_embeds=dummy_h,
                    position_ids=pos_ids,
                    use_cache=False,
                )
            torch.cuda.synchronize()

        # 결과 수집
        for i in range(N_LAYERS):
            for op_name in OP_NAMES:
                key = f"layer_{i}_{op_name}"
                if key not in start_evts:
                    continue
                times = []
                for m in range(n_measure):
                    try:
                        t = start_evts[key][m].elapsed_time(end_evts[key][m])
                        times.append(t)
                    except RuntimeError:
                        pass
                if times:
                    results[key] = times

    finally:
        for h in handles:
            h.remove()

    return results


# prefill 측정
logger.info("  → Prefill (seq=3086) 측정 중 ...")
prefill_times = measure_op_times(lm_model, layers, "prefill",
                                  n_warmup=WARMUP_RUNS, n_measure=MEASURE_RUNS)

# decode 측정
logger.info("  → Decode (seq=1) 측정 중 ...")
decode_times = measure_op_times(lm_model, layers, "decode",
                                 n_warmup=WARMUP_RUNS, n_measure=MEASURE_RUNS)

logger.info(f"  측정 완료. 키 수: prefill={len(prefill_times)}, decode={len(decode_times)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ② torch.profiler — Chrome trace + 커널 breakdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logger.info("\n[3/5] torch.profiler (Chrome trace 생성) ...")

trace_path = str(OUT / "chrome_trace.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
    profile_memory=False,
) as prof:
    # decode 1 step만 프로파일링
    with record_function("decode_step"):
        with torch.no_grad():
            lm_model(
                inputs_embeds=dummy_decode,
                position_ids=pos_decode,
                use_cache=False,
            )
    torch.cuda.synchronize()

prof.export_chrome_trace(trace_path)
logger.info(f"  Chrome trace 저장: {trace_path}")
logger.info("  → Chrome 브라우저에서 chrome://tracing 열고 trace 파일 로드 가능")

# 커널 breakdown (상위 20개)
key_avgs = prof.key_averages()
try:
    table_str = key_avgs.table(sort_by="cuda_time_total", row_limit=20)
    with open(OUT / "profiler_table.txt", "w", encoding="utf-8") as f:
        f.write(table_str)
    logger.info(f"  상위 20개 커널 → profiler_table.txt 저장")
except Exception as e:
    logger.warning(f"  profiler table 출력 실패: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ③ 분석 — "놀고 있는 device" 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logger.info("\n[4/5] 분석: 놀고 있는 device 추정 ...")


def mean_ms(times: list[float]) -> float:
    return float(np.mean(times)) if times else 0.0


def analyze_idle(decode_times: dict, prefill_times: dict) -> dict:
    analysis = {}

    # ── Decode: 레이어별 aggregation ──
    # 블록별 attention 전체 vs FFN 전체 vs 이론 DRAM fetch 시간
    block_decode = []
    for i in range(N_LAYERS):
        attn_ms  = mean_ms(decode_times.get(f"layer_{i}_self_attn", []))
        ffn_ms   = mean_ms(decode_times.get(f"layer_{i}_mlp",       []))
        q_ms     = mean_ms(decode_times.get(f"layer_{i}_self_attn.q_proj", []))
        k_ms     = mean_ms(decode_times.get(f"layer_{i}_self_attn.k_proj", []))
        v_ms     = mean_ms(decode_times.get(f"layer_{i}_self_attn.v_proj", []))
        o_ms     = mean_ms(decode_times.get(f"layer_{i}_self_attn.o_proj", []))
        gate_ms  = mean_ms(decode_times.get(f"layer_{i}_mlp.gate_proj", []))
        up_ms    = mean_ms(decode_times.get(f"layer_{i}_mlp.up_proj",   []))
        down_ms  = mean_ms(decode_times.get(f"layer_{i}_mlp.down_proj", []))

        # 이론 DRAM fetch 시간 (각 연산자 가중치 크기 ÷ 231 GB/s)
        q_dram_ms    = WEIGHT_MB["q_proj"]    / DRAM_BW_GBs * 1e3
        k_dram_ms    = WEIGHT_MB["k_proj"]    / DRAM_BW_GBs * 1e3
        v_dram_ms    = WEIGHT_MB["v_proj"]    / DRAM_BW_GBs * 1e3
        o_dram_ms    = WEIGHT_MB["o_proj"]    / DRAM_BW_GBs * 1e3
        gate_dram_ms = WEIGHT_MB["gate_proj"] / DRAM_BW_GBs * 1e3
        up_dram_ms   = WEIGHT_MB["up_proj"]   / DRAM_BW_GBs * 1e3
        down_dram_ms = WEIGHT_MB["down_proj"] / DRAM_BW_GBs * 1e3

        # SM 활용률 추정: 실제 FLOPs ÷ (실측 시간 × 이론 피크)
        # W_Q: 2 × 4096 × 4096 = 33.6 MFLOPs, Tensor Core peak ≈ 1600 TFLOPS
        TENSOR_CORE_PEAK_TFLOPS = 1600.0
        q_flops = 2.0 * HIDDEN_SIZE * HIDDEN_SIZE           # 33.6 MFLOPs
        gate_flops = 2.0 * HIDDEN_SIZE * FFN_INTERMEDIATE   # 100.7 MFLOPs

        q_sm_util = 0.0
        if q_ms > 0:
            q_actual_tflops = (q_flops / 1e12) / (q_ms / 1e3)
            q_sm_util = q_actual_tflops / TENSOR_CORE_PEAK_TFLOPS * 100

        gate_sm_util = 0.0
        if gate_ms > 0:
            gate_actual_tflops = (gate_flops / 1e12) / (gate_ms / 1e3)
            gate_sm_util = gate_actual_tflops / TENSOR_CORE_PEAK_TFLOPS * 100

        # Q/K/V 병렬화 잠재력
        # 현재: q + k + v (순차)
        # 이상: max(q, k, v) (병렬 스트림)
        qkv_sequential_ms = q_ms + k_ms + v_ms
        qkv_parallel_ms   = max(q_ms, k_ms, v_ms)
        qkv_speedup       = qkv_sequential_ms / qkv_parallel_ms if qkv_parallel_ms > 0 else 1.0

        # gate/up 병렬화 잠재력
        gate_up_seq_ms    = gate_ms + up_ms
        gate_up_par_ms    = max(gate_ms, up_ms)
        gate_up_speedup   = gate_up_seq_ms / gate_up_par_ms if gate_up_par_ms > 0 else 1.0

        # DRAM 활용률 (q_proj 기준)
        q_dram_util = q_dram_ms / q_ms * 100 if q_ms > 0 else 0

        block_decode.append({
            "layer": i,
            "attn_ms":  attn_ms,
            "ffn_ms":   ffn_ms,
            "q_ms":     q_ms,
            "k_ms":     k_ms,
            "v_ms":     v_ms,
            "o_ms":     o_ms,
            "gate_ms":  gate_ms,
            "up_ms":    up_ms,
            "down_ms":  down_ms,
            # 이론 DRAM 시간
            "q_dram_theory_ms":    round(q_dram_ms, 3),
            "gate_dram_theory_ms": round(gate_dram_ms, 3),
            # SM 활용률
            "q_proj_sm_util_pct":   round(q_sm_util, 2),
            "gate_proj_sm_util_pct": round(gate_sm_util, 2),
            # DRAM 활용률
            "q_proj_dram_util_pct": round(q_dram_util, 2),
            # 병렬화 잠재력
            "qkv_seq_ms":     round(qkv_sequential_ms, 3),
            "qkv_par_ms":     round(qkv_parallel_ms, 3),
            "qkv_speedup_x":  round(qkv_speedup, 3),
            "gate_up_seq_ms": round(gate_up_seq_ms, 3),
            "gate_up_par_ms": round(gate_up_par_ms, 3),
            "gate_up_speedup_x": round(gate_up_speedup, 3),
        })

    # ── 전체 평균 (36 레이어) ──
    if block_decode:
        avg_q_ms        = np.mean([b["q_ms"]   for b in block_decode])
        avg_gate_ms     = np.mean([b["gate_ms"] for b in block_decode])
        avg_q_sm        = np.mean([b["q_proj_sm_util_pct"]   for b in block_decode])
        avg_gate_sm     = np.mean([b["gate_proj_sm_util_pct"] for b in block_decode])
        avg_q_dram      = np.mean([b["q_proj_dram_util_pct"] for b in block_decode])
        avg_qkv_speedup = np.mean([b["qkv_speedup_x"]    for b in block_decode])
        avg_gu_speedup  = np.mean([b["gate_up_speedup_x"] for b in block_decode])

        # 한 decode step에서 Q/K/V 병렬화로 절약 가능한 시간
        total_qkv_seq_ms = sum(b["qkv_seq_ms"] for b in block_decode)
        total_qkv_par_ms = sum(b["qkv_par_ms"] for b in block_decode)
        total_gu_seq_ms  = sum(b["gate_up_seq_ms"] for b in block_decode)
        total_gu_par_ms  = sum(b["gate_up_par_ms"] for b in block_decode)

        summary = {
            "mode": "decode (seq=1)",
            "n_layers": N_LAYERS,
            "avg_per_layer": {
                "q_proj_ms":         round(float(avg_q_ms), 3),
                "gate_proj_ms":      round(float(avg_gate_ms), 3),
                "q_dram_theory_ms":  round(WEIGHT_MB["q_proj"] / DRAM_BW_GBs * 1e3, 3),
                "gate_dram_theory_ms": round(WEIGHT_MB["gate_proj"] / DRAM_BW_GBs * 1e3, 3),
            },
            "sm_utilization": {
                "q_proj_sm_pct":   round(float(avg_q_sm), 2),
                "gate_proj_sm_pct": round(float(avg_gate_sm), 2),
                "interpretation":  "<<1% → Tensor Core 거의 놈 (DRAM 대기 중)",
            },
            "dram_utilization": {
                "q_proj_dram_pct": round(float(avg_q_dram), 2),
                "interpretation":  ">80% → DRAM이 진짜 병목",
            },
            "parallelization_potential": {
                "qkv_parallel": {
                    "total_seq_ms":  round(float(total_qkv_seq_ms), 2),
                    "total_par_ms":  round(float(total_qkv_par_ms), 2),
                    "total_save_ms": round(float(total_qkv_seq_ms - total_qkv_par_ms), 2),
                    "avg_speedup_x": round(float(avg_qkv_speedup), 3),
                    "description":   "q/k/v proj를 3개 CUDA stream으로 동시 실행 시 절약",
                },
                "gate_up_parallel": {
                    "total_seq_ms":  round(float(total_gu_seq_ms), 2),
                    "total_par_ms":  round(float(total_gu_par_ms), 2),
                    "total_save_ms": round(float(total_gu_seq_ms - total_gu_par_ms), 2),
                    "avg_speedup_x": round(float(avg_gu_speedup), 3),
                    "description":   "gate/up proj를 2개 CUDA stream으로 동시 실행 시 절약",
                },
            },
        }
        analysis["decode_summary"] = summary
        analysis["decode_per_block"] = block_decode

    # ── Prefill: 레이어별 attention/FFN 비율 ──
    block_prefill = []
    for i in range(N_LAYERS):
        attn_ms = mean_ms(prefill_times.get(f"layer_{i}_self_attn", []))
        ffn_ms  = mean_ms(prefill_times.get(f"layer_{i}_mlp",       []))
        total_ms = attn_ms + ffn_ms
        block_prefill.append({
            "layer":    i,
            "attn_ms":  round(attn_ms, 3),
            "ffn_ms":   round(ffn_ms, 3),
            "total_ms": round(total_ms, 3),
            "attn_ratio_pct": round(attn_ms / total_ms * 100, 1) if total_ms > 0 else 0,
            "ffn_ratio_pct":  round(ffn_ms  / total_ms * 100, 1) if total_ms > 0 else 0,
        })
    analysis["prefill_per_block"] = block_prefill

    if block_prefill:
        avg_attn_ratio = np.mean([b["attn_ratio_pct"] for b in block_prefill])
        avg_ffn_ratio  = np.mean([b["ffn_ratio_pct"]  for b in block_prefill])
        analysis["prefill_summary"] = {
            "mode": "prefill (seq=3086)",
            "avg_attn_ratio_pct": round(float(avg_attn_ratio), 1),
            "avg_ffn_ratio_pct":  round(float(avg_ffn_ratio), 1),
            "interpretation": "FFN 비율이 높으면 FFN이 병목, Attn이 높으면 quadratic attention"
        }

    return analysis


analysis = analyze_idle(decode_times, prefill_times)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 저장 + 출력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logger.info("\n[5/5] 결과 저장 ...")

# 전체 결과 저장
with open(OUT / "per_block_ops.json", "w", encoding="utf-8") as f:
    json.dump({
        "decode": {k: [round(v, 4) for v in vals]
                   for k, vals in decode_times.items()},
        "prefill": {k: [round(v, 4) for v in vals]
                    for k, vals in prefill_times.items()},
    }, f, indent=2, ensure_ascii=False)

with open(OUT / "analysis.json", "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

# 핵심 수치 요약 출력
print("\n" + "=" * 70)
print("  Alpamayo Idle Device 분석 결과")
print("=" * 70)

ds = analysis.get("decode_summary", {})
if ds:
    avg = ds.get("avg_per_layer", {})
    sm  = ds.get("sm_utilization", {})
    dram= ds.get("dram_utilization", {})
    par = ds.get("parallelization_potential", {})

    print(f"\n[Decode — seq=1, 1 token 생성]")
    print(f"  q_proj 평균 실측:     {avg.get('q_proj_ms', '?'):.3f} ms")
    print(f"  q_proj DRAM 이론:     {avg.get('q_dram_theory_ms', '?'):.3f} ms  (32 MB ÷ 231 GB/s)")
    print(f"  gate_proj 평균 실측:  {avg.get('gate_proj_ms', '?'):.3f} ms")
    print(f"  gate_proj DRAM 이론:  {avg.get('gate_dram_theory_ms', '?'):.3f} ms (96 MB ÷ 231 GB/s)")
    print()
    print(f"  Tensor Core 활용률:")
    print(f"    q_proj:    {sm.get('q_proj_sm_pct', '?'):.2f}%")
    print(f"    gate_proj: {sm.get('gate_proj_sm_pct', '?'):.2f}%")
    print(f"    → {sm.get('interpretation', '')}")
    print()
    print(f"  DRAM 활용률:")
    print(f"    q_proj:    {dram.get('q_proj_dram_pct', '?'):.1f}%  (이론 fetch ÷ 실측 시간)")
    print(f"    → {dram.get('interpretation', '')}")
    print()

    qkv = par.get("qkv_parallel", {})
    gu  = par.get("gate_up_parallel", {})
    print(f"  [병렬화 잠재력]")
    print(f"  Q/K/V stream 병렬 (3 streams):")
    print(f"    현재 순차: {qkv.get('total_seq_ms', '?'):.2f} ms (36 layers 합계)")
    print(f"    병렬 시:   {qkv.get('total_par_ms', '?'):.2f} ms")
    print(f"    절약 가능: {qkv.get('total_save_ms', '?'):.2f} ms / step")
    print(f"    평균 가속: {qkv.get('avg_speedup_x', '?'):.2f}×")
    print()
    print(f"  gate/up stream 병렬 (2 streams):")
    print(f"    현재 순차: {gu.get('total_seq_ms', '?'):.2f} ms (36 layers 합계)")
    print(f"    병렬 시:   {gu.get('total_par_ms', '?'):.2f} ms")
    print(f"    절약 가능: {gu.get('total_save_ms', '?'):.2f} ms / step")
    print(f"    평균 가속: {gu.get('avg_speedup_x', '?'):.2f}×")

ps = analysis.get("prefill_summary", {})
if ps:
    print(f"\n[Prefill — seq=3086]")
    print(f"  평균 Attention 비율: {ps.get('avg_attn_ratio_pct', '?'):.1f}%")
    print(f"  평균 FFN 비율:       {ps.get('avg_ffn_ratio_pct', '?'):.1f}%")
    print(f"  → {ps.get('interpretation', '')}")

print(f"\n결과 저장: {OUT}/")
print(f"  - analysis.json        — 블록별 상세 분석")
print(f"  - per_block_ops.json   — 원시 타이밍 데이터")
print(f"  - chrome_trace.json    — Chrome chrome://tracing 에서 시각화")
print(f"  - profiler_table.txt   — 커널 breakdown (상위 20개)")
print("=" * 70)

logger.info("완료.")
