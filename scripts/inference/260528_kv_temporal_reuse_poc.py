"""
260528_kv_temporal_reuse_poc.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

목적:
  KV Temporal Reuse 메커니즘의 유효성을 단계적으로 검증한다.
  모델 가중치는 일체 변경하지 않는다.
  past_key_values (DynamicCache) 를 외부에서 관리하는 것만으로
  LM Prefill 비용을 줄일 수 있는지 확인한다.

배경 (2026-05-28 기준 베이스라인):
  - sdpa + DynamicCache = 4,838ms 전체
    VE 728ms | LM Prefill 1,423ms | Decode 1,818ms(17step×107ms) | Flow 870ms
  - 입력 토큰 구조: [text_prefix ~100] + [vision ~2,890] + [ego ~82] + [text_suffix ~14]
  - Vision = 전체의 94%. VE는 항상 재실행 (새 프레임 픽셀 처리, 불가피)
  - LM Prefill 1,423ms: 전체 3,086토큰에 대한 attention 연산 — 이것을 줄이는 것이 목표

핵심 아이디어:
  연속 프레임에서 앞부분(text_prefix)의 KV는 변하지 않는다.
  만약 이전 프레임의 KV를 재사용하고 변경된 부분(vision+ego)만 다시 계산하면
  attention 연산량이 줄어 LM Prefill 비용이 감소해야 한다.

실험 설계 (3단계, 각 단계가 다음 단계의 전제조건):

  [실험 A] 동일 프레임 완전 재사용 (oracle 상한) — 베이스라인 검증
    프레임 t0으로 full prefill 1회 → DynamicCache 저장
    동일 프레임 t0으로 again → next_token_only + 저장된 KV 제공
    기대: 두 번째 prefill ≈ 0ms (1토큰 forward), decode 동일
    의미: 메커니즘 자체가 동작하는지 확인. 실패하면 나머지 실험 의미없음.

  [실험 B] Text prefix KV만 재사용 (현실적 하한)
    프레임 t0으로 full prefill → text_prefix 부분의 KV만 추출·저장
    t0 재처리: text_prefix KV (100토큰) 제공 + 나머지(vision+ego+suffix) 새로 prefill
    기대: prefill 절약 ≈ 100/3086 = 3.2% → ~1,377ms (절약 46ms)
    의미: text_prefix 재사용만으로는 이익이 미미함을 수치로 확인

  [실험 C] 인접 프레임 vision+ego KV 부분교체 — 핵심 실험
    t0 full prefill → 전체 KV 저장
    t1 (t0+500ms, 다른 프레임) VE 실행 → 새 vision embedding 획득
    vision 구간의 KV 슬라이스를 t1 값으로 교체 + ego/text_suffix 부분 재계산
    기대: "전체 KV 교체 prefill" 비용 측정 (현재는 불가능, 향후 구현 방향 검증)
    의미: vision KV 교체 시 어떤 에러/시간 변화가 나타나는지 탐색

중요 제약:
  - 모델 가중치 변경 없음 (from_pretrained 구조 그대로)
  - 양자화 없음 (sdpa + BF16 + DynamicCache)
  - attn_implementation 미지정 (sdpa 기본값, FlashAttention 유지)

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5
  python3 scripts/inference/260528_kv_temporal_reuse_poc.py [--exp A|B|C|ALL]

결과:
  profiling_results/260528_kv_temporal_reuse/results_<EXP>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DynamicCache

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
T1_US            = T0_US + 500_000   # t0 + 500ms: 인접 프레임 (실험 C용)
DEVICE           = "cuda"
MAX_DECODE_STEPS = 80
EOS_CHECK_INTERVAL = 4
TEMPERATURE      = 0.6
TOP_P            = 0.98
NUM_WARMUP       = 1
NUM_MEASURE      = 3

# 베이스라인 수치 (260528_calibrate_seqlen.py 측정, 비교용)
BASELINE_VE_MS      = 728.0
BASELINE_PREFILL_MS = 1423.0
BASELINE_DECODE_MS  = 1818.0
BASELINE_FLOW_MS    = 870.0
BASELINE_TOTAL_MS   = BASELINE_VE_MS + BASELINE_PREFILL_MS + BASELINE_DECODE_MS + BASELINE_FLOW_MS  # 4838ms

OUT = Path("profiling_results/260528_kv_temporal_reuse")
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

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

    def all_done(self) -> bool:
        return bool(self.found.all().item())

    def get_eos_positions(self):
        return self.eos_steps.cpu().tolist()


def decode_loop(model, first_logits, past_kv, prefill_len,
                eos_id, traj_offset, traj_vocab_size,
                label: str = "") -> dict | None:
    """
    공통 decode loop. past_kv는 DynamicCache (mutable, 각 step에서 자동 확장).
    prefill_len: RoPE 위치 계산을 위해 현재 KV cache에 채워진 토큰 수.
    """
    lgts = first_logits.clone()
    lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample_gpu(lgts)

    buf = torch.zeros(1, MAX_DECODE_STEPS, dtype=torch.long, device=DEVICE)
    tracker = GPUEosTracker(1, MAX_DECODE_STEPS, eos_id, DEVICE)
    buf[:, 0] = next_tok
    tracker.update(next_tok, 0)
    cur = next_tok.unsqueeze(1)

    if tracker.all_done():
        logger.info(f"  [{label}] decode: EOS at step 1")
        return {"decode_ms": 0.0, "decode_steps": 1, "decode_ms_per_step": 0.0}

    sw = CudaStopwatch()
    torch.cuda.synchronize()
    sw.start()

    for step in range(1, MAX_DECODE_STEPS):
        already_done = tracker.found.unsqueeze(1)
        eos_fill = torch.full_like(cur, eos_id)
        cur_in = torch.where(already_done, eos_fill, cur)

        # ★ cache_position: 절대 위치. 없으면 DynamicCache + sdpa에서 RoPE 오류
        cpos = torch.tensor([prefill_len + step - 1], device=DEVICE, dtype=torch.long)

        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.vlm(
                    input_ids=cur_in,
                    past_key_values=past_kv,
                    cache_position=cpos,
                    use_cache=True,
                )
        except Exception as e:
            decode_ms = sw.stop_ms()
            logger.error(f"  [{label}] decode step {step} 실패: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

        past_kv = o.past_key_values
        lgts = o.logits[:, -1, :].float()
        lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample_gpu(lgts)
        buf[:, step] = next_tok
        tracker.update(next_tok, step)
        cur = next_tok.unsqueeze(1)

        if step % EOS_CHECK_INTERVAL == 0 and tracker.all_done():
            break

    decode_ms = sw.stop_ms()
    eos_pos = tracker.get_eos_positions()
    steps = eos_pos[0] + 1 if eos_pos[0] < MAX_DECODE_STEPS else MAX_DECODE_STEPS

    logger.info(
        f"  [{label}] decode: {decode_ms:.0f}ms  "
        f"({steps}steps × {decode_ms/steps:.1f}ms/step)"
    )
    return {
        "decode_ms":          round(decode_ms, 1),
        "decode_steps":       steps,
        "decode_ms_per_step": round(decode_ms / steps, 2),
    }


def full_prefill(model, input_ids, tok_data, label: str = "") -> tuple:
    """
    표준 full prefill (DynamicCache, sdpa 기본값).
    Returns: (past_kv, last_logits, prefill_ms, prefill_len)
    """
    sw = CudaStopwatch()
    torch.cuda.synchronize()
    sw.start()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=input_ids,
            attention_mask=tok_data.get("attention_mask"),
            pixel_values=tok_data.get("pixel_values"),
            image_grid_thw=tok_data.get("image_grid_thw"),
            use_cache=True,
        )
    prefill_ms = sw.stop_ms()

    prefill_len = int(input_ids.shape[1])
    logger.info(f"  [{label}] full prefill: {prefill_ms:.0f}ms  ({prefill_len} tokens)")
    return out.past_key_values, out.logits[:, -1, :].float(), prefill_ms, prefill_len


# ──────────────────────────────────────────────────────────────────────────────
# DynamicCache 유틸리티 (버전 무관 구현)
# ──────────────────────────────────────────────────────────────────────────────
# transformers 버전에 따라 DynamicCache 내부 속성명이 다를 수 있다.
#   - transformers 4.38-4.46: key_cache / value_cache (List[Tensor])
#   - transformers 4.47+:     구조 변경 가능 — to_legacy_cache() 로 우회
#   - 구버전 tuple-of-tuples: cache[layer] = (k, v)
# 아래 헬퍼는 이를 통합해 처리한다.
# ──────────────────────────────────────────────────────────────────────────────

def _cache_to_kv_pairs(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    임의 버전의 cache 객체에서 레이어별 (key, value) 텐서 쌍을 추출한다.
    key.shape = [batch, heads, seq_len, head_dim]

    지원하는 형식:
      1) cache.key_cache / cache.value_cache  (transformers 4.38+)
      2) cache._key_cache / cache._value_cache (일부 버전 private)
      3) cache.to_legacy_cache()              (transformers Cache 인터페이스)
      4) tuple/list of (k, v) tuples          (구버전 tuple format)
    """
    # 방법 1: key_cache / value_cache 직접 접근
    kc = getattr(cache, 'key_cache', None)
    vc = getattr(cache, 'value_cache', None)
    if (isinstance(kc, list) and len(kc) > 0
            and isinstance(kc[0], torch.Tensor)):
        return list(zip(kc, vc))

    # 방법 2: _key_cache / _value_cache (private attr)
    kc = getattr(cache, '_key_cache', None)
    vc = getattr(cache, '_value_cache', None)
    if (isinstance(kc, list) and len(kc) > 0
            and isinstance(kc[0], torch.Tensor)):
        return list(zip(kc, vc))

    # 방법 3: to_legacy_cache() → tuple of (k, v) per layer
    if hasattr(cache, 'to_legacy_cache'):
        try:
            legacy = cache.to_legacy_cache()
            if (legacy is not None and len(legacy) > 0
                    and isinstance(legacy[0], (tuple, list))
                    and isinstance(legacy[0][0], torch.Tensor)):
                return [(layer[0], layer[1]) for layer in legacy]
        except Exception:
            pass

    # 방법 4: cache 자체가 tuple of (k, v)
    if (isinstance(cache, (tuple, list)) and len(cache) > 0
            and isinstance(cache[0], (tuple, list))
            and len(cache[0]) == 2
            and isinstance(cache[0][0], torch.Tensor)):
        return [(layer[0], layer[1]) for layer in cache]

    # 진단 정보 출력 후 예외
    all_attrs = [a for a in dir(cache) if not a.startswith('__')]
    kv_attrs  = [a for a in all_attrs
                 if any(x in a.lower() for x in ('key', 'value', 'kv', 'cache', 'layer'))]
    raise AttributeError(
        f"\n[DynamicCache 구조 불명] type={type(cache)}\n"
        f"  KV 관련 attr: {kv_attrs}\n"
        f"  전체 attr:    {all_attrs}\n"
        "  → 위 정보를 공유하면 대응 분기를 추가합니다."
    )


def _build_cache_from_kv(
        kv_pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    """
    (key, value) 쌍 리스트로 새 DynamicCache를 구성한다.
    key.shape = [batch, heads, seq_len, head_dim] 전제.
    """
    new_cache = DynamicCache()

    # 방법 1: key_cache/value_cache 직접 설정이 가능한 버전
    if hasattr(new_cache, 'key_cache') and isinstance(new_cache.key_cache, list):
        for k, v in kv_pairs:
            new_cache.key_cache.append(k)
            new_cache.value_cache.append(v)
        return new_cache

    # 방법 2: update() 메서드 사용 — 빈 캐시에서 호출 시 해당 레이어를 assign
    if hasattr(new_cache, 'update') and callable(new_cache.update):
        for i, (k, v) in enumerate(kv_pairs):
            new_cache.update(k, v, layer_idx=i)
        return new_cache

    raise RuntimeError(
        f"Cannot build DynamicCache: {type(new_cache)} — "
        f"attrs: {[a for a in dir(new_cache) if not a.startswith('__')]}"
    )


def clone_dynamic_cache(cache) -> DynamicCache:
    """
    DynamicCache를 깊은 복사한다.
    copy.deepcopy 를 1차 시도, 실패 시 텐서 레벨 clone 수행.
    """
    try:
        return copy.deepcopy(cache)
    except Exception:
        pairs = _cache_to_kv_pairs(cache)
        return _build_cache_from_kv([(k.clone(), v.clone()) for k, v in pairs])


def get_cache_seq_len(cache) -> int:
    """DynamicCache에 현재 저장된 시퀀스 길이 반환."""
    # 방법 1: get_seq_length() 메서드 (transformers Cache 공식 API)
    if hasattr(cache, 'get_seq_length') and callable(cache.get_seq_length):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    # 방법 2: 텐서 shape 직접 확인
    try:
        pairs = _cache_to_kv_pairs(cache)
        if pairs:
            return int(pairs[0][0].shape[2])
    except Exception:
        pass
    return 0


def slice_cache_prefix(cache, prefix_len: int) -> DynamicCache:
    """
    cache에서 앞 prefix_len 토큰의 KV만 잘라 새 DynamicCache 반환.
    실험 B에서 text_prefix KV 재사용 시 사용.
    """
    pairs = _cache_to_kv_pairs(cache)
    sliced = [
        (k[:, :, :prefix_len, :].clone(),
         v[:, :, :prefix_len, :].clone())
        for k, v in pairs
    ]
    return _build_cache_from_kv(sliced)


def print_cache_info(cache, label: str = ""):
    """DynamicCache 내부 정보 출력 (디버그용). 실패해도 예외를 삼킨다."""
    try:
        pairs = _cache_to_kv_pairs(cache)
        if not pairs:
            logger.info(f"  [{label}] cache: empty (0 layers)")
            return
        k0 = pairs[0][0]
        seq_len = k0.shape[2]
        mem_mb = sum(
            (k.element_size() * k.numel() + v.element_size() * v.numel()) / 1e6
            for k, v in pairs
        )
        logger.info(
            f"  [{label}] cache: {len(pairs)} layers, "
            f"seq_len={seq_len}, "
            f"shape=[{k0.shape[0]}, {k0.shape[1]}, {seq_len}, {k0.shape[3]}], "
            f"mem={mem_mb:.0f}MB, type={type(cache).__name__}"
        )
    except AttributeError as e:
        # 구조 불명 — 진단 정보 전체 출력
        logger.warning(f"  [{label}] cache 구조 미지원: {e}")
    except Exception as e:
        logger.warning(f"  [{label}] cache 정보 출력 실패: {type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 실험 A: 동일 프레임 완전 재사용 (oracle)
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment_A(model, input_ids, tok_data,
                     eos_id, traj_offset, traj_vocab_size) -> dict:
    """
    [실험 A] 동일 프레임 KV 완전 재사용

    원리:
      1st pass: 전체 3,086토큰 full prefill → KV 저장 (prefill_ms_1st)
      2nd pass: 저장된 KV (3,086토큰분) + 새 토큰 1개 → forward
                → prefill ≈ single-token decode 속도여야 함 (≈ 107ms)

    측정:
      - prefill_ms_1st: 전체 prefill (baseline 비교)
      - prefill_ms_2nd: KV 재사용 후 1토큰 forward (≈ 0ms 목표)
      - decode_ms_1st / decode_ms_2nd: decode 품질 비교
    """
    print("\n" + "─" * 70)
    print("  [실험 A] 동일 프레임 완전 재사용 (oracle 상한)")
    print("─" * 70)
    print("  원리: t0 full prefill → KV 저장 → t0 동일 프레임 KV 재사용")
    print("  기대: 2nd prefill ≈ 1토큰 forward ≈ 107ms (vs 1,423ms baseline)")
    print()

    results = {"exp": "A", "runs": []}

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = f"WARMUP {trial+1}" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"

        # ── Full prefill ──────────────────────────────────────────────────────
        torch.cuda.empty_cache()
        try:
            kv_full, logits_full, pf_ms_ref, prefill_len = full_prefill(
                model, input_ids, tok_data, label=f"A/{tag}/full"
            )
        except Exception as e:
            logger.error(f"  [A/{tag}/full] prefill 실패: {e}")
            traceback.print_exc()
            continue

        print_cache_info(kv_full, label=f"A/{tag}/KV_full")

        # ★ decode 전에 미리 clone → decode가 원본 KV를 변형해도 안전
        kv_for_2nd = clone_dynamic_cache(kv_full)

        # 1st decode (original KV 사용, decode_loop이 수정해도 무방)
        dec_1st = decode_loop(
            model, logits_full, kv_full, prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"A/{tag}/decode_1st",
        )
        if dec_1st is None:
            logger.warning(f"  [A/{tag}] 1st decode 실패, skip")
            continue

        # ── 2nd pass: 클론된 KV 재사용, 1토큰 forward ────────────────────────
        # full logits에서 첫 토큰 추출 (재사용 시나리오의 "새 입력")
        lgts_ref = logits_full.clone()
        lgts_ref[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
        first_new_token = top_p_sample_gpu(lgts_ref).unsqueeze(1)  # [1, 1]

        # ── 2nd prefill: KV 재사용 + 1토큰 forward ───────────────────────────
        # 이것이 핵심 측정: 3,086토큰의 KV를 재사용하고 1토큰만 계산
        cpos_2nd = torch.tensor([prefill_len], device=DEVICE, dtype=torch.long)

        sw_2nd = CudaStopwatch()
        torch.cuda.synchronize()
        sw_2nd.start()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_2nd = model.vlm(
                    input_ids=first_new_token,   # 1 토큰
                    past_key_values=kv_for_2nd,  # 3,086토큰 KV 재사용
                    cache_position=cpos_2nd,
                    use_cache=True,
                )
            pf_ms_2nd = sw_2nd.stop_ms()
        except Exception as e:
            pf_ms_2nd_raw = sw_2nd.stop_ms()
            logger.error(
                f"  [A/{tag}/2nd] KV 재사용 forward 실패 ({pf_ms_2nd_raw:.0f}ms): "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            continue

        logger.info(f"  [A/{tag}] KV 재사용 1토큰 forward: {pf_ms_2nd:.1f}ms")

        # 2nd decode (재사용 KV에서 이어서)
        kv_2nd_after = out_2nd.past_key_values
        logits_2nd_first = out_2nd.logits[:, -1, :].float()

        dec_2nd = decode_loop(
            model, logits_2nd_first, kv_2nd_after, prefill_len + 1,
            eos_id, traj_offset, traj_vocab_size,
            label=f"A/{tag}/decode_2nd",
        )

        speedup = pf_ms_ref / pf_ms_2nd if pf_ms_2nd > 0 else float("inf")

        r = {
            "trial": tag,
            "prefill_ms_full":    round(pf_ms_ref, 1),    # baseline: full prefill
            "prefill_ms_reuse":   round(pf_ms_2nd, 1),    # KV 재사용: 1토큰 forward
            "prefill_speedup":    round(speedup, 2),
            "decode_1st":         dec_1st,
            "decode_2nd":         dec_2nd,
            "prefill_len":        prefill_len,
        }

        print(
            f"  [{tag}]  "
            f"full_prefill={pf_ms_ref:.0f}ms  "
            f"reuse_prefill={pf_ms_2nd:.1f}ms  "
            f"speedup={speedup:.1f}×  "
            f"decode_1st={dec_1st['decode_ms']:.0f}ms({dec_1st['decode_steps']}steps)  "
            + (f"decode_2nd={dec_2nd['decode_ms']:.0f}ms({dec_2nd['decode_steps']}steps)"
               if dec_2nd else "decode_2nd=FAILED")
        )

        if not is_warmup:
            results["runs"].append(r)

    # 집계
    if results["runs"]:
        avg_full = sum(r["prefill_ms_full"] for r in results["runs"]) / len(results["runs"])
        avg_reuse = sum(r["prefill_ms_reuse"] for r in results["runs"]) / len(results["runs"])
        avg_speedup = avg_full / avg_reuse if avg_reuse > 0 else float("inf")

        print(f"\n  ── [실험 A] 평균 결과 ───────────────────────────────────────────")
        print(f"  full prefill  (baseline) : {avg_full:.0f}ms")
        print(f"  KV 재사용 1토큰 forward  : {avg_reuse:.1f}ms")
        print(f"  speedup                  : {avg_speedup:.1f}×")
        print(f"  절약                     : {avg_full - avg_reuse:.0f}ms")
        print(
            f"  결론: "
            + ("✅ 메커니즘 동작 확인 (다음 단계 진행 가능)" if avg_speedup > 5
               else "⚠️ 예상보다 개선 낮음 — 원인 분석 필요")
        )

        results["summary"] = {
            "avg_prefill_ms_full":  round(avg_full, 1),
            "avg_prefill_ms_reuse": round(avg_reuse, 1),
            "avg_speedup":          round(avg_speedup, 2),
            "avg_saving_ms":        round(avg_full - avg_reuse, 1),
            "mechanism_ok":         avg_speedup > 5,
        }
    else:
        results["summary"] = {"error": "모든 trial 실패"}
        print("  ❌ 모든 trial 실패")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Vision 경계 자동 탐지
# ──────────────────────────────────────────────────────────────────────────────

def detect_vision_regions(model, input_ids) -> dict:
    """
    input_ids에서 vision 토큰 구간을 자동 탐지한다.

    Qwen3VL 토큰 구조:
      [text_prefix] <|vision_start|> [image_pad × N] <|vision_end|> ... [ego] [text_suffix]

    탐지 방법 (우선순위 순):
      1) model.vlm.config.image_token_id 또는 vision_start_token_id 로 직접 탐지
      2) 에러에서 역산 (실험 B 실패 시 참고: 2880-2809=71 → vision_start=100-71=29)

    Returns:
      text_prefix_len  : vision 시작 전 순수 텍스트 토큰 수
      vision_start     : 첫 번째 이미지 패치 토큰 위치
      vision_end       : 마지막 이미지 패치 토큰 바로 다음 위치
      vision_len       : vision_end - vision_start
      suffix_start     : vision_end (ego+text_suffix 시작)
      suffix_len       : 전체 길이 - vision_end
      n_image_tokens   : 이미지 패치 토큰 총 수 (= pixel_values features 수와 일치해야 함)
    """
    ids = input_ids[0].tolist()
    total_len = len(ids)

    # ── 방법 1: image_token_id (이미지 패치 토큰 ID) ──────────────────────────
    image_token_id = None
    for attr in ('image_token_id', 'image_pad_token_id'):
        image_token_id = getattr(model.vlm.config, attr, None)
        if image_token_id is not None:
            break
    if image_token_id is None:
        # tokenizer에서 <|image_pad|> 조회
        try:
            image_token_id = model.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if image_token_id == model.tokenizer.unk_token_id:
                image_token_id = None
        except Exception:
            pass

    if image_token_id is not None and image_token_id in ids:
        positions = [i for i, t in enumerate(ids) if t == image_token_id]
        vision_start = positions[0]
        vision_end   = positions[-1] + 1
        n_img = len(positions)
        logger.info(
            f"  [detect_vision_regions] image_token_id={image_token_id}, "
            f"vision=[{vision_start},{vision_end}), n_image_tokens={n_img}"
        )
        return {
            "text_prefix_len": vision_start,
            "vision_start":    vision_start,
            "vision_end":      vision_end,
            "vision_len":      vision_end - vision_start,
            "suffix_start":    vision_end,
            "suffix_len":      total_len - vision_end,
            "n_image_tokens":  n_img,
            "total_len":       total_len,
            "method":          f"image_token_id={image_token_id}",
        }

    # ── 방법 2: vision_start_token_id ────────────────────────────────────────
    vs_id = getattr(model.vlm.config, 'vision_start_token_id', None)
    ve_id = getattr(model.vlm.config, 'vision_end_token_id', None)
    if vs_id is not None and vs_id in ids:
        vs_positions = [i for i, t in enumerate(ids) if t == vs_id]
        ve_positions = [i for i, t in enumerate(ids) if t == ve_id]
        # vision 구간: 첫 vision_start ~ 마지막 vision_end (inclusive)
        vision_start = vs_positions[0]
        vision_end   = ve_positions[-1] + 1
        # 이미지 패치 토큰 수는 vision 구간에서 special token 2개/카메라 제외
        n_img = sum(1 for t in ids[vision_start:vision_end]
                    if t not in (vs_id, ve_id))
        logger.info(
            f"  [detect_vision_regions] vision_start_id={vs_id}, "
            f"vision=[{vision_start},{vision_end}), n_special_excluded={len(vs_positions)*2}"
        )
        return {
            "text_prefix_len": vision_start,
            "vision_start":    vision_start,
            "vision_end":      vision_end,
            "vision_len":      vision_end - vision_start,
            "suffix_start":    vision_end,
            "suffix_len":      total_len - vision_end,
            "n_image_tokens":  n_img,
            "total_len":       total_len,
            "method":          f"vision_start_token_id={vs_id}",
        }

    # ── 방법 3: 실험 B 에러에서 역산한 추정값 ──────────────────────────────────
    # 에러: tokens=2809, features=2880 → 2880-2809=71개가 앞 100토큰에 있음
    # → vision_start = 100 - 71 = 29
    logger.warning(
        "  [detect_vision_regions] 자동 탐지 실패 → 추정값 사용 "
        "(vision_start≈29, vision_len≈2880, suffix≈177)"
    )
    vision_start = 29
    vision_end   = 29 + 2880
    return {
        "text_prefix_len": vision_start,
        "vision_start":    vision_start,
        "vision_end":      vision_end,
        "vision_len":      2880,
        "suffix_start":    vision_end,
        "suffix_len":      total_len - vision_end,
        "n_image_tokens":  2880,
        "total_len":       total_len,
        "method":          "fallback_estimate",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 실험 B: Text prefix KV만 재사용 (현실적 하한)
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment_B(model, input_ids, tok_data,
                     eos_id, traj_offset, traj_vocab_size,
                     regions: dict | None = None) -> dict:
    """
    [실험 B] 순수 text_prefix KV 재사용 (vision 시작 전 구간만)

    핵심 수정 (실험 B 실패 원인 반영):
      - Qwen3VL 제약: pixel_values 전달 시 input_ids에 전체 이미지 패치 토큰이 있어야 함
      - text_prefix_len=100은 vision 영역 한가운데를 자름 → 이미지 토큰 불일치 → ValueError
      - 올바른 text_prefix_len = vision_start (첫 이미지 패치 토큰 바로 앞 위치)
        에러 역산: features(2880) - tokens(2809) = 71 → vision_start = 100 - 71 = 29

    올바른 구현:
      1. detect_vision_regions() 로 실제 vision_start 탐지
      2. text_prefix_len = vision_start (≈29)
      3. remaining_ids = input_ids[:, vision_start:]
         → 이 구간에 2,880개 이미지 패치 토큰 전부 포함
      4. pixel_values 전달 가능 → get_placeholder_mask() 통과 ✓

    기대:
      text_prefix (≈29토큰) 절약: 29/3086 ≈ 0.94% → ~13ms 절약
      의미: 텍스트만으로는 이익 거의 없음 → vision KV 재사용이 핵심임을 확인
    """
    # regions가 전달되지 않으면 자동 탐지
    if regions is None:
        regions = detect_vision_regions(model, input_ids)

    text_prefix_len = regions["text_prefix_len"]
    vision_end      = regions["vision_end"]
    n_image_tokens  = regions["n_image_tokens"]
    prefill_len_ref = int(input_ids.shape[1])

    print("\n" + "─" * 70)
    print(f"  [실험 B] 순수 text_prefix KV 재사용 (자동 탐지)")
    print("─" * 70)
    print(f"  탐지 결과: vision_start={text_prefix_len}, vision_end={vision_end}, "
          f"n_image_tokens={n_image_tokens}")
    print(f"  text_prefix = {text_prefix_len}토큰 / 전체 {prefill_len_ref}토큰 = "
          f"{text_prefix_len/prefill_len_ref*100:.1f}%")
    print(f"  기대 절약 ≈ {text_prefix_len/prefill_len_ref*100:.1f}% = "
          f"~{BASELINE_PREFILL_MS * text_prefix_len/prefill_len_ref:.0f}ms "
          f"(총 {BASELINE_PREFILL_MS:.0f}ms 중)")
    print()

    results = {
        "exp":              "B",
        "text_prefix_len":  text_prefix_len,
        "vision_end":       vision_end,
        "n_image_tokens":   n_image_tokens,
        "detect_method":    regions["method"],
        "runs":             [],
    }

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = f"WARMUP {trial+1}" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"

        # ── 1단계: Full prefill → text_prefix KV만 추출 ──────────────────────
        torch.cuda.empty_cache()
        try:
            kv_full, logits_full, pf_ms_full, prefill_len = full_prefill(
                model, input_ids, tok_data, label=f"B/{tag}/full"
            )
        except Exception as e:
            logger.error(f"  [B/{tag}] full prefill 실패: {e}")
            traceback.print_exc()
            continue

        # text_prefix만 slice (vision_start 이전 구간 = 이미지 토큰 없음)
        actual_prefix_len = min(text_prefix_len, get_cache_seq_len(kv_full))
        prefix_kv = slice_cache_prefix(kv_full, actual_prefix_len)
        print_cache_info(prefix_kv, label=f"B/{tag}/prefix_KV({actual_prefix_len}tok)")
        del kv_full

        # ── 2단계: Prefix KV + 나머지 전체 forward ───────────────────────────
        # remaining_ids = input_ids[:, vision_start:]
        # → 이미지 패치 토큰 2,880개 전부 포함 → pixel_values 전달 가능
        remaining_ids = input_ids[:, actual_prefix_len:]
        remaining_len = int(remaining_ids.shape[1])
        cache_pos = torch.arange(
            actual_prefix_len, actual_prefix_len + remaining_len,
            device=DEVICE, dtype=torch.long,
        )

        sw_partial = CudaStopwatch()
        torch.cuda.synchronize()
        sw_partial.start()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_partial = model.vlm(
                    input_ids=remaining_ids,
                    pixel_values=tok_data.get("pixel_values"),       # ✓ 이미지 토큰 전부 있음
                    image_grid_thw=tok_data.get("image_grid_thw"),
                    past_key_values=prefix_kv,
                    cache_position=cache_pos,
                    use_cache=True,
                )
            pf_ms_partial = sw_partial.stop_ms()
        except Exception as e:
            pf_ms_partial_raw = sw_partial.stop_ms()
            logger.error(
                f"  [B/{tag}] partial prefill 실패 ({pf_ms_partial_raw:.0f}ms): "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            if not is_warmup:
                results["runs"].append({
                    "trial": tag,
                    "error": f"{type(e).__name__}: {e}",
                    "prefill_ms_full": round(pf_ms_full, 1),
                })
            continue

        logger.info(
            f"  [B/{tag}] partial prefill (prefix {actual_prefix_len}tok 재사용): "
            f"{pf_ms_partial:.0f}ms  (full={pf_ms_full:.0f}ms, "
            f"절약={pf_ms_full - pf_ms_partial:.0f}ms)"
        )

        # decode
        kv_partial = out_partial.past_key_values
        logits_partial = out_partial.logits[:, -1, :].float()
        dec = decode_loop(
            model, logits_partial, kv_partial, prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"B/{tag}/decode",
        )

        saving_ms  = pf_ms_full - pf_ms_partial
        saving_pct = saving_ms / pf_ms_full * 100 if pf_ms_full > 0 else 0.0

        print(
            f"  [{tag}]  "
            f"full={pf_ms_full:.0f}ms  "
            f"partial(prefix={actual_prefix_len}tok)={pf_ms_partial:.0f}ms  "
            f"절약={saving_ms:.0f}ms ({saving_pct:.1f}%)  "
            + (f"decode={dec['decode_ms']:.0f}ms({dec['decode_steps']}steps)"
               if dec else "decode=FAILED")
        )

        if not is_warmup:
            results["runs"].append({
                "trial":                tag,
                "prefill_ms_full":      round(pf_ms_full, 1),
                "prefill_ms_partial":   round(pf_ms_partial, 1),
                "saving_ms":            round(saving_ms, 1),
                "saving_pct":           round(saving_pct, 2),
                "actual_prefix_len":    actual_prefix_len,
                "remaining_len":        remaining_len,
                "decode":               dec,
            })

    # 집계
    valid_runs = [r for r in results["runs"] if "error" not in r]
    if valid_runs:
        avg_full = sum(r["prefill_ms_full"] for r in valid_runs) / len(valid_runs)
        avg_partial = sum(r["prefill_ms_partial"] for r in valid_runs) / len(valid_runs)
        avg_saving = avg_full - avg_partial
        speedup = avg_full / avg_partial if avg_partial > 0 else float("inf")

        print(f"\n  ── [실험 B] 평균 결과 ───────────────────────────────────────────")
        print(f"  full prefill (baseline)              : {avg_full:.0f}ms")
        print(f"  partial prefill (prefix {text_prefix_len}tok 재사용): {avg_partial:.0f}ms")
        print(f"  절약                                 : {avg_saving:.0f}ms ({avg_saving/avg_full*100:.1f}%)")
        print(f"  speedup                              : {speedup:.2f}×")
        print(f"  이론값: {text_prefix_len}/{prefill_len_ref} = {text_prefix_len/prefill_len_ref*100:.1f}% 절약 기대")

        results["summary"] = {
            "text_prefix_len":          text_prefix_len,
            "avg_prefill_ms_full":      round(avg_full, 1),
            "avg_prefill_ms_partial":   round(avg_partial, 1),
            "avg_saving_ms":            round(avg_saving, 1),
            "avg_saving_pct":           round(avg_saving / avg_full * 100, 2),
            "speedup":                  round(speedup, 3),
            "theory_saving_pct":        round(text_prefix_len / prefill_len_ref * 100, 2),
        }
    else:
        results["summary"] = {"error": "모든 trial 실패 (에러 내용 runs 참조)"}
        print("  ❌ 모든 trial 실패")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 실험 C: 인접 프레임 — t0 vision KV 그대로 재사용 + t1 suffix만 새로 계산
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment_C(model,
                     input_ids_t0, tok_data_t0,
                     input_ids_t1, tok_data_t1,
                     eos_id, traj_offset, traj_vocab_size,
                     regions: dict | None = None) -> dict:
    """
    [실험 C] 인접 프레임 — t0의 vision KV 재사용 + t1 suffix만 forward

    ═══ 설계 변경 이유 (실험 B 실패에서 도출) ═══
    Qwen3VL 제약: pixel_values 전달 시 input_ids에 이미지 패치 토큰 전체가 있어야 함.
    이 제약으로 인해 "t1 vision KV로 교체" 방식은 결국 t1을 full prefill 후 분리해야 하므로
    절약이 없다. 진정한 절약은 아래 방식만 가능:

    ── 올바른 KV Temporal Reuse 구조 ──
      Frame t0: full prefill(t0) → KV_t0 저장 (text_prefix + vision + ego/suffix)
      Frame t1:
        1. [있으면 좋음] VE(t1) 실행 (728ms, 불가피)
        2. KV_t0의 vision 부분까지(vision_end) 재사용
           → pixel_values=None (이미지 토큰 없는 suffix만 forward)
        3. t1의 suffix 토큰(ego(t1)+text_suffix) 만 새로 forward
           → input_ids = input_ids_t1[:, vision_end:]  (이미지 토큰 없음 ✓)
           → past_key_values = KV_t0[:vision_end]  (t0의 vision까지 재사용)
        4. decode

    ── 핵심 질문 ──
    t0의 vision KV를 t1 inference에 그대로 사용해도 EOS가 생성되는가?
    t1 full prefill과 decode step 수가 비슷한가?
    → "YES"이면: 인접 프레임 간 visual KV 재사용 가능 (28% 절약)
    → "NO"이면: vision은 반드시 재계산 필요 → 실질 절약 없음

    ── 타이밍 ──
    suffix_prefill: ~suffix_len/prefill_len × 1,423ms ≈ ~82ms (177토큰 기준)
    전체: VE(728) + suffix_prefill(~82) + Decode(~1,818) + Flow(870) ≈ 3,498ms
    절약: 4,838ms - 3,498ms = 1,340ms (28%)
    """
    # regions 자동 탐지
    if regions is None:
        regions = detect_vision_regions(model, input_ids_t0)

    vision_end   = regions["vision_end"]   # suffix 시작 위치
    suffix_len   = regions["suffix_len"]   # ego + text_suffix 토큰 수
    prefill_len  = int(input_ids_t0.shape[1])

    print("\n" + "─" * 70)
    print("  [실험 C] t0 vision KV 재사용 + t1 suffix만 forward")
    print("─" * 70)
    print(f"  t0={T0_US/1e6:.1f}s → t1={T1_US/1e6:.1f}s  (Δt=500ms)")
    print(f"  vision_end={vision_end}, suffix_len={suffix_len}")
    print(f"  suffix forward 비율: {suffix_len}/{prefill_len} = {suffix_len/prefill_len*100:.1f}%")
    print(f"  예상 suffix_prefill: ~{BASELINE_PREFILL_MS * suffix_len/prefill_len:.0f}ms")
    print(f"  예상 전체: VE(728) + suffix({BASELINE_PREFILL_MS * suffix_len/prefill_len:.0f}) "
          f"+ Decode(~1818) + Flow(870) = "
          f"~{728 + BASELINE_PREFILL_MS * suffix_len/prefill_len + 1818 + 870:.0f}ms")
    print()

    results = {
        "exp":          "C",
        "t0_us":        T0_US,
        "t1_us":        T1_US,
        "vision_end":   vision_end,
        "suffix_len":   suffix_len,
        "prefill_len":  prefill_len,
        "runs":         [],
    }

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = f"WARMUP {trial+1}" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"

        torch.cuda.empty_cache()

        # ── Step C1: t0 full prefill → KV 저장 ───────────────────────────────
        try:
            kv_t0, _logits_t0, pf_ms_t0, _ = full_prefill(
                model, input_ids_t0, tok_data_t0, label=f"C/{tag}/t0_full"
            )
        except Exception as e:
            logger.error(f"  [C/{tag}/t0] prefill 실패: {e}")
            traceback.print_exc()
            continue

        # ── Step C2: t1 full prefill (비교 기준) ─────────────────────────────
        try:
            kv_t1_full, logits_t1_full, pf_ms_t1_full, _ = full_prefill(
                model, input_ids_t1, tok_data_t1, label=f"C/{tag}/t1_full"
            )
        except Exception as e:
            logger.error(f"  [C/{tag}/t1_full] prefill 실패: {e}")
            traceback.print_exc()
            continue

        # ── Step C3: t0 KV 재사용 + t1 suffix만 forward ──────────────────────
        # KV_t0에서 [0:vision_end] 구간만 추출 (ego/suffix는 t1 것으로 갱신할 것)
        kv_t0_vision = slice_cache_prefix(kv_t0, vision_end)
        print_cache_info(kv_t0_vision, label=f"C/{tag}/KV_t0_vision({vision_end}tok)")

        # t1의 suffix 토큰 (ego(t1) + text_suffix)
        # ★ 이미지 패치 토큰이 없음 → pixel_values=None 가능 ✓
        suffix_ids_t1 = input_ids_t1[:, vision_end:]
        actual_suffix_len = int(suffix_ids_t1.shape[1])
        cache_pos_suffix = torch.arange(
            vision_end, vision_end + actual_suffix_len,
            device=DEVICE, dtype=torch.long,
        )

        logger.info(
            f"  [C/{tag}] suffix forward: "
            f"input_ids[{vision_end}:{vision_end+actual_suffix_len}] "
            f"({actual_suffix_len}tok), pixel_values=None"
        )

        sw_suffix = CudaStopwatch()
        torch.cuda.synchronize()
        sw_suffix.start()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_suffix = model.vlm(
                    input_ids=suffix_ids_t1,
                    pixel_values=None,          # 이미지 토큰 없음 → pixel_values 불필요
                    past_key_values=kv_t0_vision,
                    cache_position=cache_pos_suffix,
                    use_cache=True,
                )
            pf_ms_suffix = sw_suffix.stop_ms()
        except Exception as e:
            pf_ms_suffix_raw = sw_suffix.stop_ms()
            logger.error(
                f"  [C/{tag}] suffix forward 실패 ({pf_ms_suffix_raw:.0f}ms): "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            if not is_warmup:
                results["runs"].append({
                    "trial":                 tag,
                    "error":                 f"{type(e).__name__}: {e}",
                    "prefill_ms_t0_full":    round(pf_ms_t0, 1),
                    "prefill_ms_t1_full":    round(pf_ms_t1_full, 1),
                    "actual_suffix_len":     actual_suffix_len,
                })
            continue

        logger.info(
            f"  [C/{tag}] suffix forward 완료: {pf_ms_suffix:.0f}ms  "
            f"({actual_suffix_len}tok / {prefill_len}tok = "
            f"{actual_suffix_len/prefill_len*100:.1f}%)"
        )

        # t1 full baseline decode
        dec_t1_full = decode_loop(
            model, logits_t1_full, kv_t1_full, prefill_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"C/{tag}/t1_full_decode",
        )

        # suffix 재사용 decode (t0 vision KV + t1 suffix KV)
        kv_reuse_final = out_suffix.past_key_values
        logits_reuse   = out_suffix.logits[:, -1, :].float()

        dec_reuse = decode_loop(
            model, logits_reuse, kv_reuse_final, vision_end + actual_suffix_len,
            eos_id, traj_offset, traj_vocab_size,
            label=f"C/{tag}/reuse_decode",
        )

        saving_ms  = pf_ms_t1_full - pf_ms_suffix
        saving_pct = saving_ms / pf_ms_t1_full * 100 if pf_ms_t1_full > 0 else 0.0

        print(
            f"  [{tag}]  "
            f"t1_full={pf_ms_t1_full:.0f}ms  "
            f"suffix_prefill={pf_ms_suffix:.0f}ms({actual_suffix_len}tok)  "
            f"절약={saving_ms:.0f}ms ({saving_pct:.1f}%)  "
            f"t1_decode={dec_t1_full['decode_ms']:.0f}ms({dec_t1_full['decode_steps']}s)  "
            + (f"reuse_decode={dec_reuse['decode_ms']:.0f}ms({dec_reuse['decode_steps']}s)"
               if dec_reuse else "reuse_decode=FAILED")
        )

        if not is_warmup:
            results["runs"].append({
                "trial":                tag,
                "prefill_ms_t0":        round(pf_ms_t0, 1),
                "prefill_ms_t1_full":   round(pf_ms_t1_full, 1),
                "prefill_ms_suffix":    round(pf_ms_suffix, 1),
                "saving_ms":            round(saving_ms, 1),
                "saving_pct":           round(saving_pct, 2),
                "actual_suffix_len":    actual_suffix_len,
                "decode_t1_full":       dec_t1_full,
                "decode_reuse":         dec_reuse,
                "prefill_len":          prefill_len,
            })

    # 집계
    valid_runs = [r for r in results["runs"] if "error" not in r]
    if valid_runs:
        avg_t1_full  = sum(r["prefill_ms_t1_full"] for r in valid_runs) / len(valid_runs)
        avg_hybrid   = sum(r["prefill_ms_suffix"] for r in valid_runs) / len(valid_runs)
        avg_saving   = avg_t1_full - avg_hybrid
        speedup      = avg_t1_full / avg_hybrid if avg_hybrid > 0 else float("inf")

        dec_t1_avg   = sum(r["decode_t1_full"]["decode_ms"] for r in valid_runs) / len(valid_runs)
        dec_ruse_avg = (
            sum(r["decode_reuse"]["decode_ms"] for r in valid_runs if r.get("decode_reuse"))
            / max(1, sum(1 for r in valid_runs if r.get("decode_reuse")))
        )

        print(f"\n  ── [실험 C] 평균 결과 ───────────────────────────────────────────")
        print(f"  t1 full prefill   : {avg_t1_full:.0f}ms")
        print(f"  suffix prefill    : {avg_hybrid:.0f}ms  (suffix {valid_runs[0]['actual_suffix_len']}tok만)")
        print(f"  절약              : {avg_saving:.0f}ms ({avg_saving/avg_t1_full*100:.1f}%)")
        print(f"  speedup           : {speedup:.2f}×")
        print(f"  decode t1_full    : {dec_t1_avg:.0f}ms")
        print(f"  decode reuse      : {dec_ruse_avg:.0f}ms")
        print(
            f"  결론: "
            + ("✅ t0 vision KV 재사용 유효 — 다음 단계 구현 진행"
               if speedup > 2 else "⚠️ 이익 미미 — suffix prefill 비용이 지배적")
        )

        results["summary"] = {
            "avg_prefill_ms_t1_full":  round(avg_t1_full, 1),
            "avg_prefill_ms_suffix":   round(avg_hybrid, 1),
            "avg_saving_ms":           round(avg_saving, 1),
            "avg_saving_pct":          round(avg_saving / avg_t1_full * 100, 2),
            "speedup":                 round(speedup, 3),
            "theory_saving_pct":       round((prefill_len - suffix_len) / prefill_len * 100, 2),
            "decode_t1_full_avg_ms":   round(dec_t1_avg, 1),
            "decode_reuse_avg_ms":     round(dec_ruse_avg, 1),
        }
    else:
        results["summary"] = {"error": "모든 trial 실패 (에러 내용 runs 참조)"}
        print("  ❌ 모든 trial 실패")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KV Temporal Reuse PoC")
    parser.add_argument(
        "--exp",
        choices=["A", "B", "C", "ALL"],
        default="ALL",
        help=(
            "실험 선택. "
            "A=동일프레임재사용(oracle), "
            "B=text_prefix재사용(하한), "
            "C=인접프레임vision교체(핵심), "
            "ALL=A→B→C 순서 실행 (A 실패 시 B,C 스킵)"
        ),
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=100,
        help="실험 B, C에서 재사용할 text_prefix 토큰 수 (기본: 100)",
    )
    parser.add_argument(
        "--vision-len",
        type=int,
        default=2890,
        help="실험 C에서 교체할 vision 구간 토큰 수 (기본: 2890)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  KV Temporal Reuse PoC")
    print(f"  exp={args.exp}  (prefix/vision 경계: 자동 탐지)")
    print("=" * 70)
    print(f"\n  베이스라인 (sdpa+DynamicCache, BF16):")
    print(f"    VE       : {BASELINE_VE_MS:.0f}ms")
    print(f"    LM Prefill: {BASELINE_PREFILL_MS:.0f}ms  ← 이것을 줄이는 것이 목표")
    print(f"    Decode   : {BASELINE_DECODE_MS:.0f}ms")
    print(f"    Flow     : {BASELINE_FLOW_MS:.0f}ms")
    print(f"    합계     : {BASELINE_TOTAL_MS:.0f}ms")

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    logger.info("데이터 로드 중 (t0)...")
    data_t0 = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages_t0 = helper.create_message(
        frames=data_t0["image_frames"].flatten(0, 1),
        camera_indices=data_t0["camera_indices"],
    )

    need_t1 = args.exp in ("C", "ALL")
    data_t1 = None
    messages_t1 = None
    if need_t1:
        logger.info(f"데이터 로드 중 (t1={T1_US/1e6:.1f}s)...")
        try:
            data_t1 = load_physical_aiavdataset(CLIP_ID, t0_us=T1_US)
            messages_t1 = helper.create_message(
                frames=data_t1["image_frames"].flatten(0, 1),
                camera_indices=data_t1["camera_indices"],
            )
        except Exception as e:
            logger.warning(f"  t1 데이터 로드 실패 ({e}) → 실험 C 스킵")
            data_t1 = None

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    logger.info("모델 로드 중 (sdpa 기본값, BF16)...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
        # attn_implementation 미지정 → sdpa 기본값 (FlashAttention 활성화 유지)
        # StaticCache, eager 사용 금지
    ).to(DEVICE).eval()

    cfg = model.vlm.config
    actual_attn = getattr(cfg, "_attn_implementation", "unknown")
    logger.info(f"  → attn_implementation = {actual_attn}")

    # ── 입력 준비 (t0) ────────────────────────────────────────────────────────
    processor = helper.get_processor(model.tokenizer)

    def prepare_inputs(messages, ego_xyz, ego_rot):
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
            {"ego_history_xyz": ego_xyz, "ego_history_rot": ego_rot},
            DEVICE,
        )
        input_ids_raw = inputs.pop("input_ids")
        input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
        return input_ids, inputs  # inputs = tok_data (attention_mask, pixel_values, ...)

    input_ids_t0, tok_data_t0 = prepare_inputs(
        messages_t0,
        data_t0["ego_history_xyz"],
        data_t0["ego_history_rot"],
    )
    logger.info(f"  t0 input_ids: {input_ids_t0.shape}  ({input_ids_t0.shape[1]} tokens)")

    input_ids_t1, tok_data_t1 = None, None
    if data_t1 is not None:
        input_ids_t1, tok_data_t1 = prepare_inputs(
            messages_t1,
            data_t1["ego_history_xyz"],
            data_t1["ego_history_rot"],
        )
        logger.info(f"  t1 input_ids: {input_ids_t1.shape}  ({input_ids_t1.shape[1]} tokens)")

    eos_id = model.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    logger.info(
        f"  eos_id={eos_id}, traj_offset={traj_offset}, "
        f"traj_vocab_size={traj_vocab_size}"
    )

    all_results = {"attn_implementation": actual_attn, "baseline": {
        "ve_ms":       BASELINE_VE_MS,
        "prefill_ms":  BASELINE_PREFILL_MS,
        "decode_ms":   BASELINE_DECODE_MS,
        "flow_ms":     BASELINE_FLOW_MS,
        "total_ms":    BASELINE_TOTAL_MS,
    }}

    run_A = args.exp in ("A", "ALL")
    run_B = args.exp in ("B", "ALL")
    run_C = args.exp in ("C", "ALL") and data_t1 is not None

    # ── 실험 A ───────────────────────────────────────────────────────────────
    if run_A:
        res_A = run_experiment_A(
            model, input_ids_t0, tok_data_t0,
            eos_id, traj_offset, traj_vocab_size,
        )
        all_results["experiment_A"] = res_A
        out_a = OUT / "results_A.json"
        out_a.write_text(json.dumps(res_A, indent=2, default=str))
        logger.info(f"  결과 저장: {out_a}")

        if args.exp == "ALL":
            mech_ok = res_A.get("summary", {}).get("mechanism_ok", False)
            if not mech_ok:
                print("\n  ⚠️ 실험 A 실패 또는 개선 미미 → 실험 B, C 스킵")
                print("    근본 원인 분석 후 재실험 권장")
                run_B = False
                run_C = False

    # ── 실험 B ───────────────────────────────────────────────────────────────
    if run_B:
        res_B = run_experiment_B(
            model, input_ids_t0, tok_data_t0,
            eos_id, traj_offset, traj_vocab_size,
        )
        all_results["experiment_B"] = res_B
        out_b = OUT / "results_B.json"
        out_b.write_text(json.dumps(res_B, indent=2, default=str))
        logger.info(f"  결과 저장: {out_b}")

    # ── 실험 C ───────────────────────────────────────────────────────────────
    if run_C:
        res_C = run_experiment_C(
            model,
            input_ids_t0, tok_data_t0,
            input_ids_t1, tok_data_t1,
            eos_id, traj_offset, traj_vocab_size,
        )
        all_results["experiment_C"] = res_C
        out_c = OUT / "results_C.json"
        out_c.write_text(json.dumps(res_C, indent=2, default=str))
        logger.info(f"  결과 저장: {out_c}")

    # ── 종합 결과 ─────────────────────────────────────────────────────────────
    out_all = OUT / "results_ALL.json"
    out_all.write_text(json.dumps(all_results, indent=2, default=str))

    W = 70
    print(f"\n{'═'*W}")
    print("  ★ KV Temporal Reuse PoC 종합 결과")
    print(f"{'═'*W}")
    print(f"  {'실험':12}  {'항목':30}  {'결과':>10}")
    print(f"  {'-'*60}")

    if "experiment_A" in all_results:
        s = all_results["experiment_A"].get("summary", {})
        if "error" not in s:
            print(f"  {'[실험 A]':12}  {'full_prefill (ms)':30}  {s['avg_prefill_ms_full']:>10.0f}")
            print(f"  {'':12}  {'KV재사용 1tok forward':30}  {s['avg_prefill_ms_reuse']:>10.1f}")
            print(f"  {'':12}  {'speedup':30}  {s['avg_speedup']:>10.1f}×")
            print(f"  {'':12}  {'메커니즘 동작':30}  {'✅' if s['mechanism_ok'] else '❌':>10}")

    if "experiment_B" in all_results:
        s = all_results["experiment_B"].get("summary", {})
        if "error" not in s:
            b_ptok = s["text_prefix_len"]
            b_label = f"prefix({b_ptok}tok) 절약"
            print(f"  {'[실험 B]':12}  {b_label:30}  {s['avg_saving_ms']:>10.0f}ms")
            print(f"  {'':12}  {'절약 비율':30}  {s['avg_saving_pct']:>9.1f}%")

    if "experiment_C" in all_results:
        s = all_results["experiment_C"].get("summary", {})
        if "error" not in s:
            print(f"  {'[실험 C]':12}  {'t1 full prefill':30}  {s['avg_prefill_ms_t1_full']:>10.0f}")
            print(f"  {'':12}  {'suffix only prefill (t0 KV 재사용)':30}  {s['avg_prefill_ms_suffix']:>10.0f}")
            print(f"  {'':12}  {'절약 비율':30}  {s['avg_saving_pct']:>9.1f}%")
            print(f"  {'':12}  {'speedup':30}  {s['speedup']:>10.2f}×")

    print(f"{'─'*W}")
    print(f"  전체 결과: {out_all}")
    print(f"{'═'*W}")


if __name__ == "__main__":
    main()
