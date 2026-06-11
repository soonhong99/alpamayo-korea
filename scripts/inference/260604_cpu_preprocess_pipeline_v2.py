"""
CPU 전처리 파이프라인 실험 v2 (수정판)
=======================================

[v1의 문제점]
  v1에서 Step A (load_physical_aiavdataset, ~10,000ms)를 전처리에 포함시켰으나
  이것은 연구용 파일 로더이다. 실제 추론 시스템에서는 카메라 데이터가 이미
  메모리에 있으므로 파일 I/O가 없다. A를 포함하면 실제 시스템을 반영하지 못함.

[v2의 올바른 설계]
  Step A (파일 로드): 실험 준비 단계에서 모든 프레임을 미리 메모리에 올림.
                      이후 파이프라인 루프에서는 A를 실행하지 않음.
  실제 CPU 전처리:    B + C + D만 측정 (카메라 입력 → 토큰 준비, ~321ms 예상)
    B: create_message         (0.3ms, 무시)
    C: apply_chat_template    (316ms, ViT patch 추출 + 토크나이징)
    D: .to(device)            (5ms, unified memory near-instant)

[파이프라인 설계]
  현재 (직렬):
    [CPU BCD(k+1)] → [GPU inference(k+1)]  ← 321ms overhead per cycle

  개선 (파이프라인):
    [GPU inference(k)]
          ↕ 동시 실행 (321ms << 4,366ms)
    [CPU BCD(k+1)]
    → GPU가 끝났을 때 CPU는 이미 완료 → wait ≈ 0ms

  VE 관점:
    Decode(k, 1345ms) 중 CPU BCD(k+1, 321ms) 완료
    → inference(k) 끝난 즉시 VE(k+1) 시작 (gap 없음)

[측정 내용]
  Phase 1: B+C+D 시간 측정 (파일 I/O 없이, 실제 전처리 시간)
  Phase 2: CPU 코어 수 스케일링 (C단계가 유일한 torch 연산)
  Phase 3: GPU inference ∥ CPU BCD — wait_ms ≈ 0ms 검증
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

N_FRAMES    = 6          # 미리 메모리에 로드할 프레임 수
N_PROFILE   = 4          # 전처리 시간 측정 반복 횟수
N_PIPELINE  = 4          # 파이프라인 inference 횟수
N_DECODE    = 15         # decode steps
N_WARMUP    = 2          # GPU warmup
WORKERS_LIST = [1, 2, 4, 8]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model():
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
    processor   = alpa_helper.get_processor(model.tokenizer)

    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, traj_v_size={traj_v_size}")
    logger.info(f"  GPU: allocated={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, processor, eos_id, traj_offset, traj_v_size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step A: 사전 로드 (반복 실험에서 제외할 파일 I/O 부분)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preload_raw_data(n: int) -> list[dict]:
    """
    N개 프레임의 raw data를 메모리에 미리 올림.
    이 함수는 파이프라인 루프 밖에서 한 번만 호출됨.
    실제 시스템에서는 카메라 하드웨어가 이 역할을 함.
    """
    logger.info(f"\n[사전 준비] {n}개 프레임 raw data 메모리 로드 (파일 I/O, 한 번만)...")
    raw_data_list = []
    for i in range(n):
        t_us = T0_US + i * DELTA_T_MS * 1000
        t0 = time.perf_counter()
        data = load_physical_aiavdataset(CLIP_ID, t0_us=t_us)
        ms = (time.perf_counter() - t0) * 1000
        raw_data_list.append(data)
        logger.info(f"  frame {i}: {ms:.0f}ms  "
                    f"images={data['image_frames'].shape}")
    logger.info(f"  → 모든 raw data 메모리 상주 완료 (이후 파일 I/O 없음)")
    return raw_data_list


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step B+C+D: 실제 CPU 전처리 (파이프라인 대상)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_bcd(data: dict, processor, n_threads: int = 1) -> dict:
    """
    Step B+C+D: 이미 메모리에 있는 raw data → 토큰 준비.
    파일 I/O 없음. 실제 배포 환경의 CPU 전처리 시간을 반영.

    B: create_message        (~0.3ms)
    C: apply_chat_template   (~316ms)  ← 유일한 실질 작업
    D: .to(device)           (~5ms)
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    try:
        # Step B: 메시지 구성 (이미 메모리에 있는 image_frames 사용)
        t0 = time.perf_counter()
        messages = alpa_helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        ms_B = (time.perf_counter() - t0) * 1000

        # Step C: apply_chat_template (ViT patch 추출 + 토크나이징)
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

        # Step D: .to(device) (unified memory: near-instant)
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
        "raw":       raw,
        "data":      data,
        "ms_B":      ms_B,
        "ms_C":      ms_C,
        "ms_D":      ms_D,
        "ms_bcd":    ms_B + ms_C + ms_D,
        "n_threads": n_threads,
    }


def finalize_on_gpu(model, result: dict) -> dict:
    """CPU 전처리 결과 → GPU tensor dict (fuse_traj_tokens 포함)."""
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
# GPU inference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
# Phase 1: B+C+D 시간 측정 (실제 전처리 시간)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase1_profile_bcd(raw_data_list, processor) -> dict:
    print(f"\n{'='*68}")
    print("  Phase 1: B+C+D 실제 전처리 시간 (파일 I/O 없음, 1코어)")
    print(f"{'='*68}")
    logger.info("  ← 카메라 데이터가 이미 메모리에 있는 상황 시뮬레이션")

    results = []
    for i in range(N_PROFILE):
        data = raw_data_list[i % len(raw_data_list)]
        r = preprocess_bcd(data, processor, n_threads=1)
        results.append(r)
        logger.info(
            f"  run {i+1}/{N_PROFILE}: "
            f"B={r['ms_B']:.1f}ms  C={r['ms_C']:.1f}ms  "
            f"D={r['ms_D']:.1f}ms  BCD={r['ms_bcd']:.1f}ms"
        )

    def avg(key):
        return statistics.mean(r[key] for r in results)

    summary = {
        "ms_B":   round(avg("ms_B"), 1),
        "ms_C":   round(avg("ms_C"), 1),
        "ms_D":   round(avg("ms_D"), 1),
        "ms_bcd": round(avg("ms_bcd"), 1),
    }

    logger.info(f"\n  ─── Phase 1 요약 (평균) ───")
    logger.info(f"  B. create_message        :  {summary['ms_B']:.1f}ms")
    logger.info(f"  C. apply_chat_template   :  {summary['ms_C']:.1f}ms  ← 실질 CPU 작업")
    logger.info(f"  D. .to(device)           :  {summary['ms_D']:.1f}ms")
    logger.info(f"  ──────────────────────────────────────")
    logger.info(f"  BCD 합계 (실제 전처리)   :  {summary['ms_bcd']:.1f}ms")
    logger.info(f"  GPU inference 시간       :  ~4,366ms")
    logger.info(f"  비율 BCD/GPU             :  {summary['ms_bcd']/4366*100:.1f}%")

    if summary["ms_bcd"] < 4366:
        margin = 4366 - summary["ms_bcd"]
        logger.info(f"  → ✅ BCD({summary['ms_bcd']:.0f}ms) < GPU(4366ms) "
                    f"→ {margin:.0f}ms 여유 → 파이프라인 가능")
    else:
        logger.info(f"  → ❌ BCD > GPU — 파이프라인 효과 없음")

    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: CPU 코어 수 스케일링 (C단계만 torch 영향 받음)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase2_core_scaling(raw_data_list, processor) -> dict:
    print(f"\n{'='*68}")
    print("  Phase 2: CPU 코어 수 스케일링 (BCD 전처리)")
    print(f"{'='*68}")
    logger.info("  ← C(apply_chat_template)가 torch 연산 → set_num_threads 영향 받음")

    scaling = {}
    for n_w in WORKERS_LIST:
        times = []
        for i in range(N_PROFILE):
            data = raw_data_list[i % len(raw_data_list)]
            r = preprocess_bcd(data, processor, n_threads=n_w)
            times.append(r["ms_bcd"])
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
    for n_w in WORKERS_LIST:
        sp  = base / scaling[n_w]["mean_ms"]
        fits = "✅ GPU 안에 완전히 숨음" if scaling[n_w]["mean_ms"] < 4366 else "❌ bottleneck"
        logger.info(f"  {n_w:2d}코어: {sp:.2f}×  ({scaling[n_w]['mean_ms']:.1f}ms)  {fits}")

    # 파이프라인에 사용할 최적 코어: BCD가 가장 짧은 것 (단, 1코어도 충분하면 1코어)
    best_n = min(WORKERS_LIST, key=lambda n: scaling[n]["mean_ms"])
    logger.info(f"\n  → BCD 최단 코어 수: {best_n}코어 ({scaling[best_n]['mean_ms']:.1f}ms)")
    scaling["_best_n"] = best_n
    return scaling


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: GPU inference ∥ CPU BCD (파이프라인 검증)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AsyncBCDPreprocessor:
    """백그라운드 스레드에서 BCD(B+C+D) 비동기 실행."""
    def __init__(self, processor, n_threads: int = 1):
        self.processor  = processor
        self.n_threads  = n_threads
        self._result    = None
        self._error     = None
        self._done      = threading.Event()

    def start(self, data: dict):
        """이미 메모리에 있는 data로 BCD 전처리 시작."""
        self._result = None
        self._error  = None
        self._done.clear()

        def worker():
            try:
                r = preprocess_bcd(data, self.processor, n_threads=self.n_threads)
                self._result = r
            except Exception as e:
                self._error = e
            finally:
                self._done.set()

        threading.Thread(target=worker, daemon=True).start()

    def wait(self, timeout: float = 60.0) -> dict:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError("CPU BCD 전처리 타임아웃")
        if self._error:
            raise self._error
        return self._result


def phase3_pipeline(model, processor, raw_data_list, tok_list_warmup,
                    traj_offset, traj_v_size, eos_id,
                    best_n_workers: int = 1) -> dict:
    print(f"\n{'='*68}")
    print(f"  Phase 3: GPU inference ∥ CPU BCD ({best_n_workers}코어)")
    print(f"{'='*68}")
    logger.info("  ← 파일 I/O 없음. 실제 배포 환경과 동일 조건")

    pf_len = tok_list_warmup[0]["input_ids"].shape[1]

    # ── 3-A: GPU inference only ──────────────────────────────────────
    logger.info("\n  [3-A] GPU only (BCD 전처리 없음, 기준선)")
    gpu_times = []
    for i in range(N_PIPELINE):
        tok = tok_list_warmup[i % len(tok_list_warmup)]
        ms  = run_inference(model, tok, traj_offset, traj_v_size, eos_id, pf_len)
        gpu_times.append(ms)
        logger.info(f"    inf {i}: {ms:.0f}ms")
    avg_gpu = statistics.mean(gpu_times)
    logger.info(f"  GPU only avg: {avg_gpu:.0f}ms")

    # ── 3-B: Sequential (CPU BCD → GPU) ──────────────────────────────
    logger.info(f"\n  [3-B] Sequential: CPU BCD(1코어) → GPU inference")
    seq_times  = []
    seq_bcd_ms = []
    for i in range(N_PIPELINE):
        data = raw_data_list[i % len(raw_data_list)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # CPU BCD (직렬)
        prep = preprocess_bcd(data, processor, n_threads=1)
        seq_bcd_ms.append(prep["ms_bcd"])

        # GPU op 마무리 + inference
        tok_cur = finalize_on_gpu(model, prep)
        run_inference(model, tok_cur, traj_offset, traj_v_size, eos_id, pf_len)

        total = (time.perf_counter() - t0) * 1000
        seq_times.append(total)
        logger.info(f"    inf {i}: bcd={prep['ms_bcd']:.0f}ms  total={total:.0f}ms")

    avg_seq     = statistics.mean(seq_times)
    avg_seq_bcd = statistics.mean(seq_bcd_ms)
    logger.info(f"  Sequential avg: {avg_seq:.0f}ms  "
                f"(BCD 占 {avg_seq_bcd/avg_seq*100:.1f}%)")

    # ── 3-C: Async pipeline (CPU BCD ∥ GPU) ──────────────────────────
    logger.info(f"\n  [3-C] Async pipeline: CPU BCD({best_n_workers}코어) ∥ GPU inference")
    logger.info(f"  기대: wait_ms ≈ 0ms (BCD <<  GPU inference)")

    preprocessor = AsyncBCDPreprocessor(processor, n_threads=best_n_workers)
    async_times  = []
    wait_times   = []
    bcd_ms_list  = []

    # 첫 프레임 cold start (pipeline 진입 전)
    logger.info(f"    cold start BCD (파이프라인 진입 전처리)...")
    prep_0  = preprocess_bcd(raw_data_list[0], processor, n_threads=best_n_workers)
    tok_cur = finalize_on_gpu(model, prep_0)
    logger.info(f"    cold BCD: {prep_0['ms_bcd']:.0f}ms")

    for i in range(N_PIPELINE):
        data_next = raw_data_list[(i + 1) % len(raw_data_list)]
        t0 = time.perf_counter()

        # ① CPU: 다음 프레임 BCD 비동기 시작 (GPU와 동시!)
        preprocessor.start(data_next)

        # ② GPU: 현재 프레임 inference
        gpu_ms = run_inference(
            model, tok_cur, traj_offset, traj_v_size, eos_id, pf_len
        )

        # ③ CPU 완료 대기 (BCD <<  GPU → 이미 끝나있어야 함 → wait ≈ 0ms)
        t_wait    = time.perf_counter()
        prep_next = preprocessor.wait()
        wait_ms   = (time.perf_counter() - t_wait) * 1000
        bcd_ms_list.append(prep_next["ms_bcd"])

        # ④ GPU op 마무리
        if i + 1 < N_PIPELINE:
            tok_cur = finalize_on_gpu(model, prep_next)

        total_ms = (time.perf_counter() - t0) * 1000
        async_times.append(total_ms)
        wait_times.append(wait_ms)

        hidden   = wait_ms < 30
        overhead = prep_next["ms_bcd"] - gpu_ms  # 양수면 CPU가 더 오래 걸림
        logger.info(
            f"    inf {i}: gpu={gpu_ms:.0f}ms  "
            f"bcd={prep_next['ms_bcd']:.0f}ms  "
            f"wait={wait_ms:.0f}ms  "
            f"total={total_ms:.0f}ms  "
            f"{'✅ BCD 완전히 숨음' if hidden else f'⚠️  대기 {wait_ms:.0f}ms'}"
        )

    avg_async    = statistics.mean(async_times)
    avg_wait     = statistics.mean(wait_times)
    avg_bcd_async = statistics.mean(bcd_ms_list)

    logger.info(f"\n  ─── Phase 3 결과 ──────────────────────────────")
    logger.info(f"  [A] GPU only:           {avg_gpu:.0f}ms  (기준선)")
    logger.info(f"  [B] Sequential (BCD+GPU): {avg_seq:.0f}ms  (+{avg_seq-avg_gpu:.0f}ms)")
    logger.info(f"  [C] Async ({best_n_workers}코어):     {avg_async:.0f}ms  "
                f"(+{avg_async-avg_gpu:.0f}ms vs GPU only)")
    logger.info(f"")
    logger.info(f"  BCD 시간 ({best_n_workers}코어):   {avg_bcd_async:.0f}ms")
    logger.info(f"  GPU 완료 후 대기:       {avg_wait:.0f}ms  "
                f"({'✅ BCD 완전히 숨음' if avg_wait < 30 else '⚠️  대기 발생'})")
    logger.info(f"  Sequential 대비 가속:   {avg_seq/avg_async:.3f}×")
    logger.info(f"  절감 시간:              {avg_seq-avg_async:.0f}ms/inference")
    logger.info(f"  이론 최대 절감:         {avg_seq_bcd:.0f}ms/inference (BCD 전체 제거)")

    return {
        "avg_gpu_only_ms":     round(avg_gpu, 1),
        "avg_sequential_ms":   round(avg_seq, 1),
        "avg_seq_bcd_ms":      round(avg_seq_bcd, 1),
        "avg_async_ms":        round(avg_async, 1),
        "avg_bcd_async_ms":    round(avg_bcd_async, 1),
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
    print("  Alpamayo CPU 전처리 파이프라인 실험 v2")
    print("  (파일 I/O 제외 — 실제 배포 환경 시뮬레이션)")
    print("=" * 68)
    logger.info(f"  CPU cores: {os.cpu_count()}")
    logger.info(f"  torch threads: {torch.get_num_threads()}")
    logger.info(f"  CUDA: {torch.version.cuda}, SM: {torch.cuda.get_device_capability()}")

    # ── 모델 로드 ──────────────────────────────────────────────────────
    model, processor, eos_id, traj_offset, traj_v_size = load_model()

    # ── Step A: 모든 raw data 미리 메모리에 로드 (한 번만) ──────────────
    raw_data_list = preload_raw_data(N_FRAMES)

    # ── GPU warmup용 tok_list 준비 ─────────────────────────────────────
    logger.info(f"\n[GPU warmup용 tok_list 준비 ({N_PIPELINE}개)...]")
    tok_list_warmup = []
    for i in range(N_PIPELINE):
        data = raw_data_list[i % len(raw_data_list)]
        prep = preprocess_bcd(data, processor, n_threads=1)
        tok  = finalize_on_gpu(model, prep)
        tok_list_warmup.append(tok)
        logger.info(f"  frame {i}: input_ids={tok['input_ids'].shape}")

    pf_len = tok_list_warmup[0]["input_ids"].shape[1]
    for i in range(N_WARMUP):
        run_inference(model, tok_list_warmup[0],
                      traj_offset, traj_v_size, eos_id, pf_len)
        logger.info(f"  warmup {i} done")
    torch.cuda.empty_cache()

    # ── Phase 1 ─────────────────────────────────────────────────────────
    p1 = phase1_profile_bcd(raw_data_list, processor)

    # ── Phase 2 ─────────────────────────────────────────────────────────
    p2 = phase2_core_scaling(raw_data_list, processor)
    best_n = p2.get("_best_n", 1)

    # ── Phase 3 ─────────────────────────────────────────────────────────
    p3 = phase3_pipeline(
        model, processor, raw_data_list, tok_list_warmup,
        traj_offset, traj_v_size, eos_id,
        best_n_workers=best_n,
    )

    # ── 최종 요약 ──────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  최종 요약 (파일 I/O 없음 = 실제 배포 환경)")
    print(f"{'='*68}")
    logger.info(f"  실제 CPU 전처리 (BCD, 1코어):")
    logger.info(f"    B create_message:      {p1['ms_B']:.1f}ms")
    logger.info(f"    C apply_chat_template: {p1['ms_C']:.1f}ms  ← 실질 작업")
    logger.info(f"    D .to(device):         {p1['ms_D']:.1f}ms")
    logger.info(f"    BCD 합계:              {p1['ms_bcd']:.1f}ms")
    logger.info(f"    GPU inference:         ~4,366ms")
    logger.info(f"    BCD/GPU 비율:          {p1['ms_bcd']/4366*100:.1f}%")
    logger.info(f"")
    logger.info(f"  코어 스케일링 (BCD):")
    base = p2[1]["mean_ms"]
    for n_w in WORKERS_LIST:
        sp = base / p2[n_w]["mean_ms"]
        logger.info(f"    {n_w:2d}코어: {p2[n_w]['mean_ms']:.1f}ms ({sp:.2f}×)")
    logger.info(f"")
    logger.info(f"  파이프라인 ({best_n}코어):")
    logger.info(f"    GPU only:      {p3['avg_gpu_only_ms']:.1f}ms")
    logger.info(f"    Sequential:    {p3['avg_sequential_ms']:.1f}ms  (+{p3['avg_seq_bcd_ms']:.0f}ms BCD)")
    logger.info(f"    Async:         {p3['avg_async_ms']:.1f}ms  ({p3['speedup_vs_seq']:.3f}× vs Seq)")
    logger.info(f"    wait_for_cpu:  {p3['avg_wait_for_cpu_ms']:.1f}ms  "
                f"({'✅ 완전히 숨음' if p3['avg_wait_for_cpu_ms'] < 30 else '⚠️  대기 발생'})")
    logger.info(f"    절감:          {p3['saving_per_inf_ms']:.1f}ms/inference")

    if p3["avg_wait_for_cpu_ms"] < 30:
        throughput_gain = p3["avg_seq_bcd_ms"] / p3["avg_sequential_ms"] * 100
        logger.info(f"")
        logger.info(f"  ★ CPU BCD 전처리가 GPU inference window 안에 완전히 숨음")
        logger.info(f"  ★ throughput 향상: ~{throughput_gain:.1f}% (BCD overhead 제거)")

    out_dir = "profiling_results/260604_cpu_preprocess_pipeline_v2"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        p2_save = {str(k): v for k, v in p2.items()}
        json.dump({"phase1": p1, "phase2": p2_save, "phase3": p3},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
