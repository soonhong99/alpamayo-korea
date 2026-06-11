"""
CPU 전처리 파이프라인 실험
===========================

목적:
  1. Alpamayo 전처리 각 단계의 CPU 시간 측정
  2. CPU 코어 수(1/2/4/8)에 따른 전처리 속도 변화 확인
     → 이전 연구: 4코어까지 대역폭 선형 증가, 이후 plateau
  3. GPU inference ∥ CPU preprocessing 파이프라이닝 효과 측정

전처리 단계:
  Step A: load_physical_aiavdataset  (데이터셋 로드, IO+CPU)
  Step B: helper.create_message      (메시지 포맷 구성)
  Step C: processor.apply_chat_template (이미지 패치 추출 + 토크나이징, CPU heavy)
  Step D: .to(DEVICE)                (통합메모리 near-instant)
  Step E: model.fuse_traj_tokens     (GPU op, 분리 불가)

파이프라인 설계:
  [GPU: inference(k)]
       ↕ 동시 실행
  [CPU 4코어: preprocess(k+1) — Step A~D]
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpamayo1_5 import helper as alpa_helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.token_utils import to_special_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLIP_ID    = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US      = 5_100_000
DELTA_T_MS = 100        # ★ 절대 고정
DEVICE     = "cuda"

N_PROFILE   = 4          # 전처리 시간 측정 반복 횟수
N_PIPELINE  = 4          # 파이프라인 inference 횟수
N_DECODE    = 15         # decode steps
N_WARMUP    = 2          # GPU warmup
WORKERS_LIST = [1, 2, 4, 8]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸: 모델 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model():
    """
    올바른 모델 로딩 방법 (MEMORY.md 기준).
    - attn_implementation 파라미터 없음 (Alpamayo1_5 outer wrapper는 sdpa 미지원)
    - 내부 model.vlm (Qwen3VL)은 자동으로 sdpa 사용
    """
    logger.info("모델 로딩 중...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(DEVICE).eval()

    eos_id      = model.tokenizer.convert_tokens_to_ids(
                      to_special_token("traj_future_start"))
    traj_offset = model.config.traj_token_start_idx
    traj_v_size = model.config.traj_vocab_size

    processor = alpa_helper.get_processor(model.tokenizer)

    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, traj_v_size={traj_v_size}")
    logger.info(f"  GPU: allocated={torch.cuda.memory_allocated()/1e9:.1f}GB")

    return model, processor, eos_id, traj_offset, traj_v_size


def preload_tok_list(model, processor, n: int) -> list[dict]:
    """N개 타임스텝 전처리해서 tok_list 반환 (GPU warmup용)."""
    tok_list = []
    for i in range(n):
        t_us = T0_US + i * DELTA_T_MS * 1000
        logger.info(f"  데이터 로드 t={i+1} ...")
        data = load_physical_aiavdataset(CLIP_ID, t0_us=t_us)
        messages = alpa_helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        raw = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        input_ids_raw = raw["input_ids"].to(DEVICE)
        ego_data = {
            "ego_history_xyz": data["ego_history_xyz"].to(DEVICE),
            "ego_history_rot": data["ego_history_rot"].to(DEVICE),
        }
        input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
        tok = {
            "input_ids":      input_ids,
            "attention_mask": raw.get("attention_mask",
                                       torch.ones_like(input_ids)).to(DEVICE),
            "pixel_values":   raw["pixel_values"].to(DEVICE),
            "image_grid_thw": raw["image_grid_thw"].to(DEVICE),
        }
        tok_list.append(tok)
        logger.info(f"    input_ids={input_ids.shape}, "
                    f"pixel_values={tok['pixel_values'].shape}")
    return tok_list


def top_p_sample(logits, top_p=0.9):
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_logits[
        cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
    ] = float("-inf")
    probs = torch.softmax(sorted_logits, dim=-1)
    tok   = sorted_idx.gather(-1, torch.multinomial(probs, 1))
    return tok.squeeze(-1)


def run_inference(model, tok, traj_offset, traj_v_size, eos_id, pf_len):
    """Full prefill + decode. 시간 반환 (ms)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=tok["input_ids"],
            attention_mask=tok.get("attention_mask"),
            pixel_values=tok.get("pixel_values"),
            image_grid_thw=tok.get("image_grid_thw"),
            use_cache=True,
        )
    kv     = out.past_key_values
    logits = out.logits[:, -1, :].float()

    logits_masked = logits.clone().index_fill_(
        -1,
        torch.arange(traj_offset, traj_offset + traj_v_size, device=DEVICE),
        float("-inf"),
    )
    cur = top_p_sample(logits_masked).unsqueeze(1)

    for step in range(N_DECODE):
        cache_pos = torch.tensor([pf_len + step], device=DEVICE, dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(
                input_ids=cur,
                pixel_values=None,
                past_key_values=kv,
                cache_position=cache_pos,
                use_cache=True,
            )
        kv = out.past_key_values
        logits_step = out.logits[:, -1, :].float()
        logits_step[:, traj_offset:traj_offset + traj_v_size] = float("-inf")
        next_tok = top_p_sample(logits_step)
        if next_tok.item() == eos_id:
            break
        cur = next_tok.unsqueeze(1)

    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전처리 함수 (CPU)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_single(t_us: int, processor, n_threads: int = 1) -> dict:
    """
    CPU 전처리 Step A~D 실행 (fuse_traj_tokens 제외).
    n_threads: torch CPU 스레드 수.
    반환: raw dict + 각 단계 시간 + data (ego용)
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    try:
        # Step A: 데이터 로드
        t0 = time.perf_counter()
        data = load_physical_aiavdataset(CLIP_ID, t0_us=t_us)
        ms_A = (time.perf_counter() - t0) * 1000

        # Step B: 메시지 구성
        t0 = time.perf_counter()
        messages = alpa_helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        ms_B = (time.perf_counter() - t0) * 1000

        # Step C: apply_chat_template (핵심 bottleneck)
        t0 = time.perf_counter()
        raw = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        ms_C = (time.perf_counter() - t0) * 1000

        # Step D: .to(device) — unified memory near-instant
        t0 = time.perf_counter()
        _ = raw["input_ids"].to(DEVICE)
        _ = raw["pixel_values"].to(DEVICE)
        _ = raw["image_grid_thw"].to(DEVICE)
        _ = data["ego_history_xyz"].to(DEVICE)
        _ = data["ego_history_rot"].to(DEVICE)
        ms_D = (time.perf_counter() - t0) * 1000

    finally:
        torch.set_num_threads(old_threads)

    return {
        "raw":        raw,
        "data":       data,
        "ms_A":       ms_A,
        "ms_B":       ms_B,
        "ms_C":       ms_C,
        "ms_D":       ms_D,
        "ms_cpu":     ms_A + ms_B + ms_C + ms_D,
        "n_threads":  n_threads,
    }


def finalize_on_gpu(model, result: dict) -> dict:
    """CPU 전처리 결과에서 GPU op(fuse_traj_tokens) 실행 (~수ms)."""
    input_ids_raw = result["raw"]["input_ids"].to(DEVICE)
    ego_data = {
        "ego_history_xyz": result["data"]["ego_history_xyz"].to(DEVICE),
        "ego_history_rot": result["data"]["ego_history_rot"].to(DEVICE),
    }
    input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
    return {
        "input_ids":      input_ids,
        "attention_mask": result["raw"].get(
            "attention_mask",
            torch.ones_like(result["raw"]["input_ids"])
        ).to(DEVICE),
        "pixel_values":   result["raw"]["pixel_values"].to(DEVICE),
        "image_grid_thw": result["raw"]["image_grid_thw"].to(DEVICE),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: 전처리 단계별 시간 측정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase1_profile_steps(processor) -> dict:
    print(f"\n{'='*68}")
    print("  Phase 1: 전처리 단계별 시간 측정 (1코어)")
    print(f"{'='*68}")

    results = []
    for i in range(N_PROFILE):
        t_us = T0_US + i * DELTA_T_MS * 1000
        r = preprocess_single(t_us, processor, n_threads=1)
        results.append(r)
        logger.info(
            f"  run {i+1}/{N_PROFILE}: "
            f"A={r['ms_A']:.0f}ms  B={r['ms_B']:.0f}ms  "
            f"C={r['ms_C']:.0f}ms  D={r['ms_D']:.0f}ms  "
            f"cpu_total={r['ms_cpu']:.0f}ms"
        )

    def avg(key):
        return statistics.mean(r[key] for r in results)

    summary = {
        "ms_A":      round(avg("ms_A"), 1),
        "ms_B":      round(avg("ms_B"), 1),
        "ms_C":      round(avg("ms_C"), 1),
        "ms_D":      round(avg("ms_D"), 1),
        "ms_cpu":    round(avg("ms_cpu"), 1),
    }

    logger.info(f"\n  ─── Phase 1 요약 (평균) ───")
    logger.info(f"  A. 데이터 로드           : {summary['ms_A']:.1f}ms")
    logger.info(f"  B. create_message        : {summary['ms_B']:.1f}ms")
    logger.info(f"  C. apply_chat_template   : {summary['ms_C']:.1f}ms  ← bottleneck?")
    logger.info(f"  D. .to(device)           : {summary['ms_D']:.1f}ms")
    logger.info(f"  CPU 합계 (A+B+C+D)       : {summary['ms_cpu']:.1f}ms")
    logger.info(f"  GPU inference 시간       : ~4,366ms")

    if summary["ms_cpu"] < 4366:
        logger.info(f"  → ✅ CPU 전처리 < GPU inference → 파이프라이닝 시 전처리 비용 숨김 가능")
    else:
        logger.info(f"  → ⚠️  CPU 전처리 > GPU inference → 코어 수 늘려서 줄여야 함")

    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: CPU 코어 수 스케일링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase2_core_scaling(processor) -> dict:
    print(f"\n{'='*68}")
    print("  Phase 2: CPU 코어 수 스케일링 (이전 연구: 4코어까지 선형 증가)")
    print(f"{'='*68}")

    scaling = {}
    for n_w in WORKERS_LIST:
        times = []
        for i in range(N_PROFILE):
            t_us = T0_US + i * DELTA_T_MS * 1000
            r = preprocess_single(t_us, processor, n_threads=n_w)
            times.append(r["ms_cpu"])
        m = statistics.mean(times)
        scaling[n_w] = {
            "mean_ms":   round(m, 1),
            "median_ms": round(statistics.median(times), 1),
            "all_ms":    [round(t, 1) for t in times],
        }
        logger.info(f"  {n_w:2d}코어: {m:.1f}ms  "
                    f"times={[f'{t:.0f}' for t in times]}")

    base = scaling[1]["mean_ms"]
    logger.info(f"\n  ─── 가속비 (1코어 기준) ───")
    best_n = 1
    for n_w in WORKERS_LIST:
        sp = base / scaling[n_w]["mean_ms"]
        fits = "✅ GPU 안에 숨음" if scaling[n_w]["mean_ms"] < 4366 else "❌ 여전히 bottleneck"
        logger.info(f"  {n_w:2d}코어: {sp:.2f}×  ({scaling[n_w]['mean_ms']:.1f}ms)  {fits}")
        if scaling[n_w]["mean_ms"] < 4366 and best_n == 1:
            best_n = n_w  # GPU inference 안에 처음 들어오는 코어 수

    logger.info(f"\n  → 파이프라이닝에 사용할 최적 코어 수: {best_n}")
    scaling["_best_n"] = best_n
    return scaling


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: GPU inference ∥ CPU preprocessing 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AsyncPreprocessor:
    """백그라운드 스레드에서 CPU 전처리 비동기 실행."""
    def __init__(self, processor, n_threads: int = 4):
        self.processor  = processor
        self.n_threads  = n_threads
        self._result    = None
        self._error     = None
        self._done      = threading.Event()

    def start(self, t_us: int):
        self._result = None
        self._error  = None
        self._done.clear()

        def worker():
            try:
                r = preprocess_single(t_us, self.processor,
                                       n_threads=self.n_threads)
                self._result = r
            except Exception as e:
                self._error = e
            finally:
                self._done.set()

        threading.Thread(target=worker, daemon=True).start()

    def wait(self, timeout: float = 60.0) -> dict:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError("CPU 전처리 타임아웃")
        if self._error:
            raise self._error
        return self._result


def phase3_pipeline(model, processor, tok_list,
                    traj_offset, traj_v_size, eos_id,
                    best_n_workers: int = 4) -> dict:
    print(f"\n{'='*68}")
    print(f"  Phase 3: GPU inference ∥ CPU preprocessing ({best_n_workers}코어)")
    print(f"{'='*68}")

    pf_len = tok_list[0]["input_ids"].shape[1]

    # ── 3-A: GPU inference only (전처리 별도, 기존 측정 방식) ───────────
    logger.info("\n  [3-A] GPU only (전처리 미포함)")
    gpu_times = []
    for i in range(N_PIPELINE):
        tok = tok_list[i % len(tok_list)]
        ms  = run_inference(model, tok, traj_offset, traj_v_size, eos_id, pf_len)
        gpu_times.append(ms)
        logger.info(f"    inf {i}: {ms:.0f}ms")
    avg_gpu = statistics.mean(gpu_times)
    logger.info(f"  GPU only avg: {avg_gpu:.0f}ms")

    # ── 3-B: Sequential (전처리 + GPU, 순차) ────────────────────────────
    logger.info(f"\n  [3-B] Sequential: preprocess(1코어) → GPU inference")
    seq_times   = []
    seq_cpu_ms  = []
    for i in range(N_PIPELINE):
        t_us = T0_US + i * DELTA_T_MS * 1000
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # CPU 전처리 (직렬, 1코어)
        prep = preprocess_single(t_us, processor, n_threads=1)
        seq_cpu_ms.append(prep["ms_cpu"])

        # GPU op 마무리 + inference
        tok_cur = finalize_on_gpu(model, prep)
        run_inference(model, tok_cur, traj_offset, traj_v_size, eos_id, pf_len)

        total = (time.perf_counter() - t0) * 1000
        seq_times.append(total)
        logger.info(f"    inf {i}: prep={prep['ms_cpu']:.0f}ms  total={total:.0f}ms")

    avg_seq      = statistics.mean(seq_times)
    avg_seq_prep = statistics.mean(seq_cpu_ms)
    logger.info(f"  Sequential avg: {avg_seq:.0f}ms  "
                f"(전처리占 {avg_seq_prep/avg_seq*100:.0f}%)")

    # ── 3-C: Async 파이프라인 (CPU ∥ GPU) ──────────────────────────────
    logger.info(f"\n  [3-C] Async: CPU({best_n_workers}코어) ∥ GPU inference")

    preprocessor = AsyncPreprocessor(processor, n_threads=best_n_workers)
    async_times  = []
    wait_times   = []
    cpu_ms_list  = []

    # 첫 프레임 cold start
    logger.info(f"    Cold start 전처리...")
    prep_0  = preprocess_single(T0_US, processor, n_threads=best_n_workers)
    tok_cur = finalize_on_gpu(model, prep_0)
    logger.info(f"    Cold start: {prep_0['ms_cpu']:.0f}ms")

    for i in range(N_PIPELINE):
        t_us_next = T0_US + (i + 1) * DELTA_T_MS * 1000

        t0 = time.perf_counter()

        # ① CPU: 다음 프레임 전처리 비동기 시작 (GPU와 동시!)
        preprocessor.start(t_us_next)

        # ② GPU: 현재 프레임 inference
        gpu_ms = run_inference(
            model, tok_cur, traj_offset, traj_v_size, eos_id, pf_len
        )

        # ③ CPU 완료 대기 (이미 끝났으면 즉시 리턴)
        t_wait = time.perf_counter()
        prep_next = preprocessor.wait()
        wait_ms   = (time.perf_counter() - t_wait) * 1000
        cpu_ms_list.append(prep_next["ms_cpu"])

        # ④ GPU op 마무리 (fuse_traj_tokens, 수ms)
        if i + 1 < N_PIPELINE:
            tok_cur = finalize_on_gpu(model, prep_next)

        total_ms = (time.perf_counter() - t0) * 1000
        async_times.append(total_ms)
        wait_times.append(wait_ms)

        hidden = wait_ms < 30
        logger.info(
            f"    inf {i}: gpu={gpu_ms:.0f}ms  "
            f"cpu_prep={prep_next['ms_cpu']:.0f}ms  "
            f"wait={wait_ms:.0f}ms  "
            f"total={total_ms:.0f}ms  "
            f"{'✅ 전처리 숨음' if hidden else '⚠️  대기 발생'}"
        )

    avg_async     = statistics.mean(async_times)
    avg_wait      = statistics.mean(wait_times)
    avg_cpu_async = statistics.mean(cpu_ms_list)

    # ── 결과 요약 ──────────────────────────────────────────────────────
    logger.info(f"\n  ─── Phase 3 결과 ──────────────────────────────")
    logger.info(f"  [A] GPU only:          {avg_gpu:.0f}ms")
    logger.info(f"  [B] Sequential:        {avg_seq:.0f}ms  "
                f"(+{avg_seq-avg_gpu:.0f}ms 전처리 overhead)")
    logger.info(f"  [C] Async({best_n_workers}코어):     {avg_async:.0f}ms  "
                f"(+{avg_async-avg_gpu:.0f}ms vs GPU only)")
    logger.info(f"")
    logger.info(f"  CPU 전처리 ({best_n_workers}코어): {avg_cpu_async:.0f}ms")
    logger.info(f"  GPU 완료 후 대기:      {avg_wait:.0f}ms  "
                f"({'전처리 완전히 숨음 ✅' if avg_wait < 30 else '대기 발생 ⚠️ '})")
    logger.info(f"  Sequential 대비 가속: {avg_seq/avg_async:.3f}×")
    logger.info(f"  절감 시간:            {avg_seq-avg_async:.0f}ms/inference")

    return {
        "avg_gpu_only_ms":     round(avg_gpu, 1),
        "avg_sequential_ms":   round(avg_seq, 1),
        "avg_async_ms":        round(avg_async, 1),
        "avg_cpu_prep_ms":     round(avg_cpu_async, 1),
        "avg_wait_for_cpu_ms": round(avg_wait, 1),
        "speedup_vs_seq":      round(avg_seq / avg_async, 3),
        "saving_per_inf_ms":   round(avg_seq - avg_async, 1),
        "n_workers":           best_n_workers,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 68)
    print("  Alpamayo CPU 전처리 파이프라인 실험")
    print("=" * 68)
    logger.info(f"  CPU cores: {os.cpu_count()}")
    logger.info(f"  torch threads: {torch.get_num_threads()}")
    logger.info(f"  CUDA: {torch.version.cuda}, SM: {torch.cuda.get_device_capability()}")

    # ── 모델 로드 ───────────────────────────────────────────────────────
    model, processor, eos_id, traj_offset, traj_v_size = load_model()

    # ── tok_list 사전 준비 (GPU warmup 용) ─────────────────────────────
    logger.info("\n  사전 데이터 로드 + GPU warmup...")
    tok_list = preload_tok_list(model, processor, n=N_PIPELINE)
    pf_len   = tok_list[0]["input_ids"].shape[1]

    for i in range(N_WARMUP):
        run_inference(model, tok_list[0],
                      traj_offset, traj_v_size, eos_id, pf_len)
        logger.info(f"  warmup {i} done")
    torch.cuda.empty_cache()

    # ── Phase 1 ─────────────────────────────────────────────────────────
    p1 = phase1_profile_steps(processor)

    # ── Phase 2 ─────────────────────────────────────────────────────────
    p2 = phase2_core_scaling(processor)
    best_n = p2.get("_best_n", 4)

    # ── Phase 3 ─────────────────────────────────────────────────────────
    p3 = phase3_pipeline(
        model, processor, tok_list,
        traj_offset, traj_v_size, eos_id,
        best_n_workers=best_n,
    )

    # ── 최종 요약 ──────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  최종 요약")
    print(f"{'='*68}")
    logger.info(f"  전처리 단계별 (1코어):")
    logger.info(f"    A 로드+B 메시지: {p1['ms_A']+p1['ms_B']:.1f}ms")
    logger.info(f"    C apply_chat:   {p1['ms_C']:.1f}ms  ← 핵심 bottleneck")
    logger.info(f"    D to(device):   {p1['ms_D']:.1f}ms")
    logger.info(f"    CPU 합계:       {p1['ms_cpu']:.1f}ms")
    logger.info(f"")
    logger.info(f"  코어 스케일링:")
    base = p2[1]["mean_ms"]
    for n_w in WORKERS_LIST:
        sp = base / p2[n_w]["mean_ms"]
        logger.info(f"    {n_w:2d}코어: {p2[n_w]['mean_ms']:.1f}ms ({sp:.2f}×)")
    logger.info(f"")
    logger.info(f"  파이프라인 ({best_n}코어):")
    logger.info(f"    GPU only:    {p3['avg_gpu_only_ms']:.1f}ms")
    logger.info(f"    Sequential:  {p3['avg_sequential_ms']:.1f}ms")
    logger.info(f"    Async:       {p3['avg_async_ms']:.1f}ms  "
                f"({p3['speedup_vs_seq']:.2f}× vs Sequential)")
    logger.info(f"    절감:        {p3['saving_per_inf_ms']:.1f}ms/inference")

    out_dir = "profiling_results/260604_cpu_preprocess_pipeline_exp"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        # p2의 _best_n 키는 직렬화 전 정수 변환
        p2_save = {str(k): v for k, v in p2.items()}
        json.dump({"phase1": p1, "phase2": p2_save, "phase3": p3},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
