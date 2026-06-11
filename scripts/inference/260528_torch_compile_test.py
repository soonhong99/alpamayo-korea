"""
260528_torch_compile_test.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적:
  torch.compile(model.vlm, mode="reduce-overhead") 적용 시
  decode 속도 개선 여부를 측정한다.

배경:
  - CUDA Graph 직접 캡처: HuggingFace 내부 D2H sync로 실패
  - torch.compile: JIT 컴파일러가 model.vlm 내부(순수 forward pass)만 컴파일
    → decode loop 밖의 D2H sync는 영향 없음 → 성공 가능성 있음
  - 현재 sdpa 기본값 baseline: 4,838ms 전체 / 107ms/step decode
  - SM 11.0 + CUDA 13.0 + aarch64 Triton 지원 여부가 핵심 변수

베이스라인 (260528_calibrate_seqlen.py 측정값):
  Vision Encoder :  728ms (15%)
  LM Prefill     : 1,423ms (29%)
  Decode         : 1,818ms (38%)  ← 17steps × 107ms/step, 90% MBU
  Flow           :  870ms (18%)
  합계           : 4,838ms

측정 항목:
  [A] 비컴파일 baseline (sdpa, 동일 스크립트 내 재측정)
  [B] torch.compile(mode="reduce-overhead") 적용
  [C] torch.compile(mode="default") — reduce-overhead 실패 시 폴백

비교 지표:
  - prefill_ms, decode_ms, decode_ms_per_step
  - 첫 번째 실행 오버헤드 (컴파일 시간)
  - 메모리 사용량 (torch.cuda.memory_allocated)

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5
  python3 scripts/inference/260528_torch_compile_test.py [--mode reduce-overhead|default|none]

결과:
  profiling_results/260528_torch_compile/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.token_utils import to_special_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
CLIP_ID          = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US            = 5_100_000
DEVICE           = "cuda"
MAX_DECODE_STEPS = 80
EOS_CHECK_INTERVAL = 4
TEMPERATURE      = 0.6
TOP_P            = 0.98
NUM_WARMUP       = 2   # 컴파일 경로: 첫 실행에 트레이싱 시간 포함 → warmup 2회
NUM_MEASURE      = 3

OUT = Path("profiling_results/260528_torch_compile")
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def top_p_sample_gpu(logits: torch.Tensor,
                     temperature: float = TEMPERATURE,
                     top_p: float = TOP_P) -> torch.Tensor:
    logits = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = (
        cumulative_probs - F.softmax(sorted_logits, dim=-1)
    ) > top_p
    sorted_indices_to_remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
    filtered_logits = torch.zeros_like(logits)
    filtered_logits.scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class GPUEosTracker:
    def __init__(self, batch_size, max_steps, eos_token_id, device):
        self.eos_token_id = eos_token_id
        self.eos_steps = torch.full(
            (batch_size,), max_steps, dtype=torch.long, device=device
        )
        self.found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def update(self, tokens, step):
        is_eos = (tokens == self.eos_token_id) & ~self.found
        self.eos_steps = torch.where(
            is_eos, torch.full_like(self.eos_steps, step), self.eos_steps
        )
        self.found = self.found | is_eos

    def get_eos_positions(self):
        return self.eos_steps.cpu().tolist()


class CudaStopwatch:
    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._s.record()

    def stop_ms(self) -> float:
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)


# ──────────────────────────────────────────────────────────────────────────────
# 단일 run: prefill + decode (DynamicCache, sdpa 기본값)
# ──────────────────────────────────────────────────────────────────────────────

def run_once(model, input_ids, tok_data,
             eos_id, traj_offset, traj_vocab_size) -> dict | None:
    """
    DynamicCache(기본값) + sdpa 경로로 prefill+decode 실행.
    torch.compile은 model.vlm 자체에 적용되어 있으므로
    이 함수는 compile 여부와 무관하게 동일.
    """
    # ── prefill ───────────────────────────────────────────────────────────────
    sw_pre = CudaStopwatch()
    torch.cuda.synchronize()
    sw_pre.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=input_ids,
            attention_mask=tok_data.get("attention_mask"),
            pixel_values=tok_data.get("pixel_values"),
            image_grid_thw=tok_data.get("image_grid_thw"),
            use_cache=True,
        )
    prefill_ms = sw_pre.stop_ms()

    first_logits = out.logits[:, -1, :].float()
    past_kv = out.past_key_values  # DynamicCache

    # 첫 토큰 샘플링 (traj 범위 마스킹)
    lgts = first_logits.clone()
    lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample_gpu(lgts)

    # ── decode ────────────────────────────────────────────────────────────────
    buf = torch.zeros(1, MAX_DECODE_STEPS, dtype=torch.long, device=DEVICE)
    tracker = GPUEosTracker(1, MAX_DECODE_STEPS, eos_id, DEVICE)
    buf[:, 0] = next_tok
    tracker.update(next_tok, 0)
    cur = next_tok.unsqueeze(1)

    if tracker.found.all().item():
        return {
            "prefill_ms": round(prefill_ms, 1),
            "decode_ms": 0.0,
            "total_ms": round(prefill_ms, 1),
            "decode_steps": 1,
            "decode_ms_per_step": 0.0,
            "decode_bw_GBps": 0.0,
        }

    sw_dec = CudaStopwatch()
    torch.cuda.synchronize()
    sw_dec.start()

    # prefill_len = 현재 KV cache에 채워진 토큰 수 (RoPE 위치 계산에 필요)
    current_len = input_ids.shape[1]  # fuse_traj_tokens 후 실제 prefill 토큰 수

    exited_at = MAX_DECODE_STEPS - 1
    for step in range(1, MAX_DECODE_STEPS):
        already_done = tracker.found.unsqueeze(1)
        eos_fill = torch.full_like(cur, eos_id)
        cur_in = torch.where(already_done, eos_fill, cur)

        # ★ cache_position: 현재 스텝의 절대 위치 (RoPE 정확도에 필수)
        #   없으면 DynamicCache + sdpa 조합에서 위치 인코딩 오류 → EOS 미탐지 가능
        cpos = torch.tensor([current_len + step - 1], device=DEVICE, dtype=torch.long)

        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.vlm(
                    input_ids=cur_in,
                    past_key_values=past_kv,
                    cache_position=cpos,
                    use_cache=True,
                )
        except Exception as e:
            decode_ms = sw_dec.stop_ms()
            logger.error(f"  ❌ decode step {step} 실패: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

        past_kv = o.past_key_values
        lgts = o.logits[:, -1, :].float()
        lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample_gpu(lgts)
        buf[:, step] = next_tok
        tracker.update(next_tok, step)
        cur = next_tok.unsqueeze(1)

        if step % EOS_CHECK_INTERVAL == 0 and tracker.found.all().item():
            exited_at = step
            break

    decode_ms = sw_dec.stop_ms()
    eos_pos = tracker.get_eos_positions()
    steps = eos_pos[0] + 1 if eos_pos[0] < MAX_DECODE_STEPS else MAX_DECODE_STEPS

    return {
        "prefill_ms":        round(prefill_ms, 1),
        "decode_ms":         round(decode_ms, 1),
        "total_ms":          round(prefill_ms + decode_ms, 1),
        "decode_steps":      steps,
        "decode_ms_per_step":round(decode_ms / steps, 2),
        # 22.157 GB = 모델 가중치 BF16 크기 (LM 부분)
        "decode_bw_GBps":    round(22.157 * steps / (decode_ms / 1000), 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["reduce-overhead", "default", "max-autotune", "cudagraphs", "none"],
        default="reduce-overhead",
        help=(
            "torch.compile mode. 'none' = 비컴파일 baseline만 측정.\n"
            "'cudagraphs' = Inductor/Triton 없이 CUDA Graph만 사용 (Triton 오류 우회)."
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"  torch.compile 테스트  mode={args.mode}")
    print("=" * 70)

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    logger.info("데이터 로드 중...")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    # ── 모델 로드 (sdpa 기본값) ───────────────────────────────────────────────
    logger.info("모델 로드 중 (sdpa 기본값)...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
        # ★ attn_implementation 미지정 → sdpa+DynamicCache = 현재 최선
        # StaticCache 사용 금지 (Flash Attention 비활성화로 2× 느려짐)
        # eager 사용 금지 (2.6× 느려짐)
    ).to(DEVICE).eval()

    cfg = model.vlm.config
    actual_attn = getattr(cfg, "_attn_implementation", "unknown")
    logger.info(f"  → attn_implementation = {actual_attn}")

    # ── 입력 준비 ─────────────────────────────────────────────────────────────
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = helper.to_device(inputs, DEVICE)

    ego_data = helper.to_device(
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        DEVICE,
    )

    input_ids_raw = inputs.pop("input_ids")
    input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
    tok_data = inputs

    prefill_len = int(input_ids.shape[1])
    logger.info(f"  prefill_len = {prefill_len} tokens")

    eos_id = model.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    logger.info(f"  eos_id={eos_id}, traj_offset={traj_offset}, "
                f"traj_vocab_size={traj_vocab_size}")

    results_all = {}

    # ── [A] 비컴파일 baseline ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [A] 비컴파일 Baseline (sdpa + DynamicCache)")
    print("─" * 70)
    baseline_runs = []
    for i in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = i < NUM_WARMUP
        tag = f"WARMUP {i+1}" if is_warmup else f"RUN {i - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()
        r = run_once(model, input_ids, tok_data, eos_id, traj_offset, traj_vocab_size)
        if r is None:
            print(f"  [{tag}] FAILED")
            continue
        print(f"  [{tag}]  total={r['total_ms']:7.1f}ms  "
              f"prefill={r['prefill_ms']:6.0f}ms  decode={r['decode_ms']:6.0f}ms  "
              f"({r['decode_steps']}steps × {r['decode_ms_per_step']:.1f}ms/step  "
              f"BW={r['decode_bw_GBps']:.1f}GB/s)")
        if not is_warmup:
            baseline_runs.append(r)

    def avg_runs(runs, key):
        vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    b_pf  = avg_runs(baseline_runs, "prefill_ms")
    b_dc  = avg_runs(baseline_runs, "decode_ms")
    b_tot = avg_runs(baseline_runs, "total_ms")
    b_stp = avg_runs(baseline_runs, "decode_ms_per_step")
    b_bw  = avg_runs(baseline_runs, "decode_bw_GBps")

    print(f"\n  Baseline 평균: prefill={b_pf:.0f}ms  decode={b_dc:.0f}ms  "
          f"total={b_tot:.0f}ms  {b_stp:.1f}ms/step  {b_bw:.1f}GB/s")
    results_all["baseline"] = {
        "prefill_ms": round(b_pf, 1),
        "decode_ms": round(b_dc, 1),
        "total_ms": round(b_tot, 1),
        "decode_ms_per_step": round(b_stp, 2),
        "decode_bw_GBps": round(b_bw, 1),
    }

    if args.mode == "none":
        logger.info("  --mode none → 컴파일 스킵")
        (OUT / "results.json").write_text(
            json.dumps({"mode": "none", **results_all}, indent=2)
        )
        return

    # ── [B] torch.compile 적용 ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"  [B] torch.compile(mode='{args.mode}')")
    print("─" * 70)

    # "cudagraphs"는 mode가 아니라 backend 인자로 전달
    # 그 외는 mode 인자로 전달 (PyTorch가 하이픈 그대로 받음, underscore 불가)
    compile_ok = False
    compile_time_s = None
    try:
        t_compile_start = time.perf_counter()
        if args.mode == "cudagraphs":
            logger.info("  torch.compile 시작 (backend='cudagraphs', Triton 불필요)...")
            # backend="cudagraphs": Inductor 없이 순수 CUDA Graph 캡처만 수행
            # → Triton 의존성 없음 → triton_key ImportError 우회
            # 단, DynamicCache 사용 시 decode step마다 KV shape 변경 → 재캡처 발생
            model.vlm = torch.compile(model.vlm, backend="cudagraphs")
        else:
            logger.info(f"  torch.compile 시작 (mode='{args.mode}')...")
            model.vlm = torch.compile(model.vlm, mode=args.mode)
        compile_time_s = time.perf_counter() - t_compile_start
        logger.info(f"  torch.compile 완료: {compile_time_s:.1f}s")
        compile_ok = True
    except Exception as e:
        logger.error(f"  ❌ torch.compile 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        results_all["compile"] = {
            "error": f"{type(e).__name__}: {e}",
            "mode": args.mode,
        }
        (OUT / "results.json").write_text(
            json.dumps({"mode": args.mode, **results_all}, indent=2)
        )
        return

    # 첫 실행 (트레이싱 + JIT 컴파일)
    print("\n  ── 첫 실행 (트레이싱 시간 포함) ──")
    mem_before = torch.cuda.memory_allocated() / 1e9
    t0 = time.perf_counter()
    r_first = run_once(
        model, input_ids, tok_data, eos_id, traj_offset, traj_vocab_size
    )
    first_wall_s = time.perf_counter() - t0
    mem_after = torch.cuda.memory_allocated() / 1e9

    if r_first is None:
        print("  ❌ 첫 실행 실패 → 컴파일 경로 미호환")
        results_all["compile"] = {
            "error": "first run failed",
            "mode": args.mode,
        }
        (OUT / "results.json").write_text(
            json.dumps({"mode": args.mode, **results_all}, indent=2)
        )
        return

    print(f"  첫 실행: wall={first_wall_s:.1f}s  "
          f"prefill={r_first['prefill_ms']:.0f}ms  decode={r_first['decode_ms']:.0f}ms  "
          f"(트레이싱 오버헤드 포함)")
    print(f"  GPU 메모리: {mem_before:.2f}GB → {mem_after:.2f}GB "
          f"(+{mem_after - mem_before:.2f}GB, compile 캐시)")

    # warmup + 측정
    compile_runs = []
    print(f"\n  ── 반복 측정 (warmup={NUM_WARMUP}, measure={NUM_MEASURE}) ──")
    for i in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = i < NUM_WARMUP
        tag = f"WARMUP {i+1}" if is_warmup else f"RUN {i - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()
        r = run_once(
            model, input_ids, tok_data, eos_id, traj_offset, traj_vocab_size
        )
        if r is None:
            print(f"  [{tag}] FAILED")
            continue
        print(f"  [{tag}]  total={r['total_ms']:7.1f}ms  "
              f"prefill={r['prefill_ms']:6.0f}ms  decode={r['decode_ms']:6.0f}ms  "
              f"({r['decode_steps']}steps × {r['decode_ms_per_step']:.1f}ms/step  "
              f"BW={r['decode_bw_GBps']:.1f}GB/s)")
        if not is_warmup:
            compile_runs.append(r)

    if not compile_runs:
        print("  ❌ 측정 실패")
        return

    c_pf  = avg_runs(compile_runs, "prefill_ms")
    c_dc  = avg_runs(compile_runs, "decode_ms")
    c_tot = avg_runs(compile_runs, "total_ms")
    c_stp = avg_runs(compile_runs, "decode_ms_per_step")
    c_bw  = avg_runs(compile_runs, "decode_bw_GBps")

    # ── 최종 비교 ─────────────────────────────────────────────────────────────
    W = 70
    print(f"\n{'═'*W}")
    print(f"  ★ torch.compile 결과 비교 (mode='{args.mode}')")
    print(f"{'═'*W}")
    print(f"  {'항목':20}  {'baseline':>12}  {'compile':>12}  {'개선':>8}")
    print(f"  {'-'*60}")
    print(f"  {'prefill':20}  {b_pf:>10.0f}ms  {c_pf:>10.0f}ms  "
          f"{b_pf/c_pf if c_pf else 0:>6.2f}×")
    print(f"  {'decode':20}  {b_dc:>10.0f}ms  {c_dc:>10.0f}ms  "
          f"{b_dc/c_dc if c_dc else 0:>6.2f}×")
    print(f"  {'total':20}  {b_tot:>10.0f}ms  {c_tot:>10.0f}ms  "
          f"{b_tot/c_tot if c_tot else 0:>6.2f}×")
    print(f"  {'ms/step':20}  {b_stp:>11.1f}ms  {c_stp:>11.1f}ms  "
          f"{b_stp/c_stp if c_stp else 0:>6.2f}×")
    print(f"  {'BW (GB/s)':20}  {b_bw:>12.1f}  {c_bw:>12.1f}")
    print(f"{'═'*W}")
    print(f"  torch.compile time: {compile_time_s:.1f}s  |  "
          f"first-run wall: {first_wall_s:.1f}s  |  "
          f"extra GPU mem: +{mem_after - mem_before:.2f}GB")

    results_all["compile"] = {
        "mode": args.mode,
        "compile_time_s": round(compile_time_s, 2),
        "first_run_wall_s": round(first_wall_s, 2),
        "extra_gpu_mem_GB": round(mem_after - mem_before, 3),
        "prefill_ms": round(c_pf, 1),
        "decode_ms": round(c_dc, 1),
        "total_ms": round(c_tot, 1),
        "decode_ms_per_step": round(c_stp, 2),
        "decode_bw_GBps": round(c_bw, 1),
        "speedup_prefill": round(b_pf / c_pf, 3) if c_pf else None,
        "speedup_decode": round(b_dc / c_dc, 3) if c_dc else None,
        "speedup_total": round(b_tot / c_tot, 3) if c_tot else None,
    }

    out_path = OUT / "results.json"
    out_path.write_text(json.dumps({"mode": args.mode, **results_all}, indent=2))
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    main()
