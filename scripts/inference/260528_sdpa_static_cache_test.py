"""
260528_sdpa_static_cache_test.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적:
  sdpa(기본값) + StaticCache(KV 가방) 조합이 동작하는지 검증한다.

배경:
  - 2026-05-19: sdpa + StaticCache → ValueError 확인 (eager로 우회)
  - 2026-05-28: eager 제거만으로 LM Prefill 3,753ms → 1,423ms (2.6×)
  - 질문: sdpa + StaticCache 가 이제 동작하는가?
    YES → eager 완전 제거, sdpa + StaticCache = 최선의 조합
    NO  → 커스텀 파이프라인은 eager 유지, 전체 파이프라인만 sdpa 사용

측정 항목:
  [1단계] sdpa + StaticCache 호환성 테스트
    - prefill: model.vlm(input_ids, past_key_values=StaticCache) 성공 여부
    - decode:  model.vlm(cur, past_key_values=StaticCache) 성공 여부
    - EOS early exit 동작 확인

  [2단계] 성공 시 타이밍 비교
    - sdpa + StaticCache  vs  eager + StaticCache (이전 실험값)
    - prefill_ms / decode_ms / steps / tps

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5
  python3 scripts/inference/260528_sdpa_static_cache_test.py

결과:
  profiling_results/260528_sdpa_static_cache/results.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import StaticCache

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
MAX_DECODE_STEPS = 80      # 안전망
EOS_CHECK_INTERVAL = 4     # 매 4스텝 CPU sync
TEMPERATURE      = 0.6
TOP_P            = 0.98
NUM_WARMUP       = 1
NUM_MEASURE      = 3
OUT = Path("profiling_results/260528_sdpa_static_cache")
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# GPU 유틸리티 (260528_shared_prefill.py에서 검증된 코드 그대로)
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
        self.eos_steps = torch.full((batch_size,), max_steps,
                                    dtype=torch.long, device=device)
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

    def stop_ms(self):
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)


# ──────────────────────────────────────────────────────────────────────────────
# KV 가방 유틸리티 (260528_shared_prefill.py와 동일)
# ──────────────────────────────────────────────────────────────────────────────

def _find_kv_names(layer0):
    _MISSING = object()
    candidates = [
        ("keys",       "values"),
        ("key_cache",  "value_cache"),
        ("k_cache",    "v_cache"),
        ("key",        "value"),
    ]
    for k_name, v_name in candidates:
        k = getattr(layer0, k_name, _MISSING)
        v = getattr(layer0, v_name, _MISSING)
        if k is _MISSING or v is _MISSING:
            continue
        if callable(k) or callable(v):
            continue
        k_ok = isinstance(k, torch.Tensor) or k is None
        v_ok = isinstance(v, torch.Tensor) or v is None
        if k_ok and v_ok:
            return k_name, v_name
    return None


def make_bag(vlm_text_config, prefill_len, max_decode, device, batch_size=1):
    return StaticCache(
        config=vlm_text_config,
        batch_size=batch_size,
        max_cache_len=prefill_len + max_decode + 4,
        device=device,
        dtype=torch.bfloat16,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 핵심 테스트: sdpa + StaticCache prefill
# ──────────────────────────────────────────────────────────────────────────────

def test_prefill_sdpa(model, input_ids, tok_data, vlm_text_config, prefill_len):
    """
    sdpa + StaticCache prefill 호환성 테스트.
    ValueError 발생 시 상세 에러와 함께 None 반환.
    """
    bag = make_bag(vlm_text_config, prefill_len, MAX_DECODE_STEPS, DEVICE)
    cache_pos = torch.arange(prefill_len, device=DEVICE, dtype=torch.long)

    sw = CudaStopwatch()
    torch.cuda.synchronize()
    sw.start()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(
                input_ids=input_ids,
                attention_mask=tok_data.get("attention_mask"),
                pixel_values=tok_data.get("pixel_values"),
                image_grid_thw=tok_data.get("image_grid_thw"),
                past_key_values=bag,
                cache_position=cache_pos,
                use_cache=True,
            )
        prefill_ms = sw.stop_ms()
        logger.info(f"  ✅ prefill sdpa + StaticCache 성공: {prefill_ms:.1f}ms")
        return bag, out.logits[:, -1, :].float(), prefill_ms

    except Exception as e:
        logger.error(f"  ❌ prefill 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# decode loop (조기종료)
# ──────────────────────────────────────────────────────────────────────────────

def run_decode_loop(model, bag, first_logits, prefill_len,
                    eos_id, traj_offset, traj_vocab_size):
    lgts = first_logits.clone()
    lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample_gpu(lgts)

    buf = torch.zeros(1, MAX_DECODE_STEPS, dtype=torch.long, device=DEVICE)
    tracker = GPUEosTracker(1, MAX_DECODE_STEPS, eos_id, DEVICE)
    buf[:, 0] = next_tok
    tracker.update(next_tok, 0)
    cur = next_tok.unsqueeze(1)

    if tracker.found.all().item():
        return buf, 0.0, [1]

    sw = CudaStopwatch()
    torch.cuda.synchronize()
    sw.start()

    exited_at = MAX_DECODE_STEPS - 1
    for step in range(1, MAX_DECODE_STEPS):
        already_done = tracker.found.unsqueeze(1)
        eos_fill = torch.full_like(cur, eos_id)
        cur_in = torch.where(already_done, eos_fill, cur)

        cpos = torch.tensor([prefill_len + step], device=DEVICE, dtype=torch.long)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.vlm(
                    input_ids=cur_in,
                    past_key_values=bag,
                    cache_position=cpos,
                    use_cache=True,
                )
        except Exception as e:
            decode_ms = sw.stop_ms()
            logger.error(f"  ❌ decode step {step} 실패: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None, decode_ms, None

        lgts = o.logits[:, -1, :].float()
        lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample_gpu(lgts)
        buf[:, step] = next_tok
        tracker.update(next_tok, step)
        cur = next_tok.unsqueeze(1)

        if step % EOS_CHECK_INTERVAL == 0 and tracker.found.all().item():
            exited_at = step
            break

    decode_ms = sw.stop_ms()
    eos_pos = tracker.get_eos_positions()
    actual = [p + 1 if p < MAX_DECODE_STEPS else MAX_DECODE_STEPS for p in eos_pos]

    logger.info(
        f"  ✅ decode sdpa + StaticCache 성공: {decode_ms:.1f}ms  "
        f"({actual[0]} steps × {decode_ms/actual[0]:.1f}ms/step)"
    )
    return buf, decode_ms, actual


# ──────────────────────────────────────────────────────────────────────────────
# 단일 run
# ──────────────────────────────────────────────────────────────────────────────

def run_once(model, input_ids, tok_data, vlm_text_config,
             prefill_len, eos_id, traj_offset, traj_vocab_size) -> dict | None:

    # prefill 테스트
    bag, first_logits, prefill_ms = test_prefill_sdpa(
        model, input_ids, tok_data, vlm_text_config, prefill_len
    )
    if bag is None:
        return None  # prefill 실패

    # decode 테스트
    _, decode_ms, actual = run_decode_loop(
        model, bag, first_logits, prefill_len,
        eos_id, traj_offset, traj_vocab_size,
    )
    if actual is None:
        return None  # decode 실패

    steps = actual[0]
    total_ms = prefill_ms + decode_ms
    return {
        "prefill_ms":        round(prefill_ms, 1),
        "decode_ms":         round(decode_ms, 1),
        "total_ms":          round(total_ms, 1),
        "decode_steps":      steps,
        "decode_ms_per_step":round(decode_ms / steps, 2),
        "decode_bw_GBps":    round(22.157 * steps / (decode_ms / 1000), 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  sdpa + StaticCache 호환성 테스트")
    print("=" * 70)

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    logger.info("데이터 로드 중...")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    # ── 모델 로드 (sdpa 기본값, eager 없음) ──────────────────────────────────
    logger.info("모델 로드 중 (attn_implementation 미지정 → sdpa)...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
        # ★ attn_implementation="eager" 없음 → sdpa 기본값
    ).to(DEVICE).eval()

    # 실제 적용된 attn 확인
    cfg = model.vlm.config
    actual_attn = getattr(cfg, "_attn_implementation", "unknown")
    logger.info(f"  → 실제 attn_implementation = {actual_attn}")

    # ── 토크나이즈 + 전처리 (260528_shared_prefill.py 와 동일한 경로) ──────────
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

    # ego history 준비 (fuse_traj_tokens에 필요)
    ego_data = helper.to_device(
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        DEVICE,
    )

    # ★ input_ids_raw pop → fuse_traj_tokens → input_ids (원본과 동일)
    input_ids_raw = inputs.pop("input_ids")          # [1, L_raw]
    input_ids     = model.fuse_traj_tokens(input_ids_raw, ego_data)  # [1, L]
    tok_data      = inputs                            # attention_mask, pixel_values, image_grid_thw

    prefill_len = int(input_ids.shape[1])
    logger.info(f"  prefill_len (fuse_traj_tokens 후) = {prefill_len} tokens")

    # ★ eos_id: 궤적 전용 특수 토큰 (원본과 동일)
    eos_id = model.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )

    # ★ traj 설정: model.config 에서 (원본과 동일)
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size

    # ★ vlm_text_config: .text_config 서브설정 사용 (StaticCache 레이어 구조용)
    vlm_text_config = model.vlm.config.text_config

    logger.info(f"  eos_id = {eos_id},  "
                f"traj_offset = {traj_offset},  "
                f"traj_vocab_size = {traj_vocab_size}")

    # ── 단계별 테스트 ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  [단계 1] prefill 단독 호환성 테스트 (1회)")
    print("─" * 70)
    bag_test, logits_test, pf_ms_test = test_prefill_sdpa(
        model, input_ids, tok_data, vlm_text_config, prefill_len
    )
    if bag_test is None:
        print("\n  ❌ prefill 실패 → sdpa + StaticCache 미호환")
        print("  → 커스텀 파이프라인은 eager 유지 필요")
        result = {"compatible": False, "stage": "prefill"}
        (OUT / "results.json").write_text(json.dumps(result, indent=2))
        return

    print("\n" + "─" * 70)
    print("  [단계 2] decode 단독 호환성 테스트 (1회)")
    print("─" * 70)
    _, dc_ms_test, actual_test = run_decode_loop(
        model, bag_test, logits_test, prefill_len,
        eos_id, traj_offset, traj_vocab_size,
    )
    if actual_test is None:
        print("\n  ❌ decode 실패 → sdpa + StaticCache 미호환 (decode 단계)")
        result = {"compatible": False, "stage": "decode"}
        (OUT / "results.json").write_text(json.dumps(result, indent=2))
        return

    print("\n  ✅ 호환성 확인 완료. 반복 측정 시작...\n")

    # ── 반복 측정 ─────────────────────────────────────────────────────────────
    print("─" * 70)
    print(f"  [단계 3] 타이밍 측정 (warmup={NUM_WARMUP}, measure={NUM_MEASURE})")
    print("─" * 70)
    all_results = []
    measure_results = []

    for i in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = i < NUM_WARMUP
        tag = f"WARMUP {i+1}" if is_warmup else f"RUN {i - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()

        r = run_once(model, input_ids, tok_data, vlm_text_config,
                     prefill_len, eos_id, traj_offset, traj_vocab_size)
        if r is None:
            print(f"  [{tag}] FAILED")
            continue

        print(f"  [{tag}]  "
              f"total={r['total_ms']:7.1f}ms  "
              f"prefill={r['prefill_ms']:6.0f}ms  "
              f"decode={r['decode_ms']:6.0f}ms  "
              f"({r['decode_steps']}steps × {r['decode_ms_per_step']:.1f}ms/step  "
              f"BW={r['decode_bw_GBps']:.1f}GB/s)")
        all_results.append(r)
        if not is_warmup:
            measure_results.append(r)

    if not measure_results:
        print("\n  ❌ 측정 실패")
        return

    # ── 집계 ─────────────────────────────────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in measure_results if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0

    pf_avg   = avg("prefill_ms")
    dc_avg   = avg("decode_ms")
    tot_avg  = avg("total_ms")
    stp_avg  = avg("decode_steps")
    pstp_avg = avg("decode_ms_per_step")
    bw_avg   = avg("decode_bw_GBps")

    W = 70
    print(f"\n{'═'*W}")
    print("  ★ sdpa + StaticCache 최종 결과")
    print(f"{'═'*W}")
    print(f"  attn_implementation : {actual_attn}")
    print(f"  prefill_len         : {prefill_len} tokens")
    print()
    print(f"  prefill    : {pf_avg:7.0f}ms")
    print(f"  decode     : {dc_avg:7.0f}ms  "
          f"({stp_avg:.1f}steps × {pstp_avg:.1f}ms/step, "
          f"BW={bw_avg:.1f}GB/s = {bw_avg/231*100:.0f}% MBU)")
    print(f"  total      : {tot_avg:7.0f}ms")
    print()

    # 이전 실험값과 비교 (260528_shared_prefill.py 기준, eager + StaticCache)
    eager_prefill = 4487.0
    eager_decode  = 2067.0
    eager_total   = eager_prefill + eager_decode
    print(f"  ── eager + StaticCache 비교 (260528 실험 기준) ──────────────")
    print(f"  {'':20}  {'sdpa(현재)':>12}  {'eager(이전)':>12}  {'개선':>8}")
    print(f"  {'prefill':20}  {pf_avg:>10.0f}ms  {eager_prefill:>10.0f}ms  "
          f"{eager_prefill/pf_avg:>6.2f}×")
    print(f"  {'decode':20}  {dc_avg:>10.0f}ms  {eager_decode:>10.0f}ms  "
          f"{eager_decode/dc_avg:>6.2f}×")
    print(f"  {'total':20}  {tot_avg:>10.0f}ms  {eager_total:>10.0f}ms  "
          f"{eager_total/tot_avg:>6.2f}×")
    print(f"{'═'*W}")

    output = {
        "compatible": True,
        "attn_implementation": actual_attn,
        "prefill_len": prefill_len,
        "measure_avg": {
            "prefill_ms":        round(pf_avg, 1),
            "decode_ms":         round(dc_avg, 1),
            "total_ms":          round(tot_avg, 1),
            "decode_steps":      round(stp_avg, 1),
            "decode_ms_per_step":round(pstp_avg, 2),
            "decode_bw_GBps":    round(bw_avg, 1),
            "decode_mbu_pct":    round(bw_avg / 231 * 100, 1),
        },
        "vs_eager": {
            "eager_prefill_ms": eager_prefill,
            "eager_decode_ms":  eager_decode,
            "prefill_speedup":  round(eager_prefill / pf_avg, 2),
            "decode_speedup":   round(eager_decode / dc_avg, 2),
            "total_speedup":    round(eager_total / tot_avg, 2),
        },
        "all_runs": all_results,
    }
    out_path = OUT / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  결과 저장: {out_path}")

    if output["compatible"]:
        print("\n  → 결론: sdpa + StaticCache 호환됨.")
        print("    260528_shared_prefill.py 의 attn_implementation='eager'를")
        print("    제거하면 됨.")


if __name__ == "__main__":
    main()
