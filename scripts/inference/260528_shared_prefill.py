"""
260528_shared_prefill.py — KV 가방(StaticCache) 기반 Shared Prefill 실험
=========================================================================
실험 목표:
  동일한 프롬프트로 N개 trajectory를 생성할 때, prefill을 1번만(B=1) 수행하고
  KV 가방(StaticCache)을 B=N으로 복제해 decode를 실행.
  → 이론상 prefill cost = 1×, throughput = N×.

실험 구성:
  Phase A — Baseline:
    B=N으로 input 확장 → B=N prefill + B=N batched decode (N=1,2,4)

  Phase B — Shared Prefill:
    B=1 prefill → KV 가방 B=1→B=N clone → B=N batched decode (N=1,2,4)

비교 지표:
  prefill_ms / clone_ms / decode_ms / total_ms / tokens_per_sec / speedup

KV 가방 K/V 접근 방식:
  StaticCache 내부 구조는 transformers 버전마다 다르다.
  _find_kv_names()가 isinstance(torch.Tensor) 기반으로 속성명을 탐지 → 버전 무관.
  Thor 실측 (transformers ≥ 4.47): cache.layers[i].keys / .values

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  cd ~/alpamayo1.5
  python3 scripts/inference/260528_shared_prefill.py

결과:
  profiling_results/260528_shared_prefill/results.json
"""

from __future__ import annotations

import json
import logging
import sys
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
CLIP_ID  = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US    = 5_100_000
DEVICE   = "cuda"
NUM_WARMUP   = 2
NUM_MEASURE  = 5
MAX_DECODE_STEPS = 80   # hard upper bound (안전망). 실제 종료는 조기종료가 처리.
EOS_CHECK_INTERVAL = 4  # 매 N 스텝마다 1회 CPU sync → 전체 완료 여부 확인 후 조기탈출
N_SWEEP  = [1, 2, 4]
TEMPERATURE  = 0.6
TOP_P    = 0.98
OUT = Path("profiling_results/260528_shared_prefill")
OUT.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# GPU 유틸리티 (260527_kv_cache_bag.py에서 검증된 코드)
# ──────────────────────────────────────────────────────────────────────────────

def top_p_sample_gpu(
    logits: torch.Tensor,           # [B, vocab_size]
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> torch.Tensor:
    """GPU-only top-p 샘플링. CPU sync 없음. Returns [B]."""
    logits = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) > top_p
    sorted_indices_to_remove[:, 0] = False  # 첫 토큰은 항상 유지
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
    filtered_logits = torch.zeros_like(logits)
    filtered_logits.scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]


class GPUEosTracker:
    """EOS 위치 추적 — 루프 내 CPU sync 없음, 종료 후 1회만 transfer."""

    def __init__(self, batch_size: int, max_steps: int, eos_token_id: int, device: str):
        self.eos_token_id = eos_token_id
        self.max_steps = max_steps
        self.eos_steps = torch.full(
            (batch_size,), max_steps, dtype=torch.long, device=device
        )
        self.found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def update(self, tokens: torch.Tensor, step: int) -> None:
        """tokens: [B] — GPU 연산 only, sync 없음."""
        is_eos = (tokens == self.eos_token_id) & ~self.found
        self.eos_steps = torch.where(
            is_eos, torch.full_like(self.eos_steps, step), self.eos_steps
        )
        self.found = self.found | is_eos

    def get_eos_positions(self) -> list[int]:
        """종료 후 1회 CPU transfer."""
        return self.eos_steps.cpu().tolist()


class CudaStopwatch:
    """CUDA Event 기반 정밀 타이머."""

    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._s.record()

    def stop_ms(self) -> float:
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


# ──────────────────────────────────────────────────────────────────────────────
# KV 가방 유틸리티 — 버전 독립적 StaticCache K/V 접근
# ──────────────────────────────────────────────────────────────────────────────

def _find_kv_names(layer0) -> tuple[str, str] | None:
    """
    StaticCache layer 객체에서 K/V 텐서 속성명을 찾는다.

    핵심 설계:
      - isinstance(Tensor) 체크 대신 sentinel으로 "속성이 존재하는가"를 판별
      - Lazy Initialization 패턴(keys=None, values=None, is_initialized=False)도 정상 탐지
      - callable (메서드)는 제외

    탐지 우선순위:
      1. ('keys',      'values')      ← transformers ≥ 4.47, Thor 실측 확인됨
      2. ('key_cache', 'value_cache') ← transformers ≤ 4.46
      3. ('k_cache',   'v_cache')
      4. ('key',       'value')

    Returns: (k_name, v_name) if found, else None
    """
    _MISSING = object()  # getattr의 default와 구분하기 위한 sentinel
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
            continue  # 속성 자체가 없음
        if callable(k) or callable(v):
            continue  # 메서드는 제외
        # 속성이 존재함: Tensor(초기화됨) 또는 None(lazy, update() 호출 시 초기화될 예정)
        k_ok = isinstance(k, torch.Tensor) or k is None
        v_ok = isinstance(v, torch.Tensor) or v is None
        if k_ok and v_ok:
            return k_name, v_name
    return None


def _kv_layers(cache: StaticCache):
    """
    StaticCache의 모든 레이어 (K_tensor, V_tensor) 쌍을 yield.
    transformers 버전 변화에 무관하게 동작.

    지원 패턴:
      D: cache.layers[i].keys / .values (transformers ≥ 4.47, Thor 현재)
      A: cache.key_cache[i]             (transformers ≤ 4.45)
      B: cache.key_cache_i              (transformers 4.46)
    """
    # Pattern D: cache.layers 리스트
    layers_attr = getattr(cache, "layers", None)
    if isinstance(layers_attr, (list, torch.nn.ModuleList)) and len(layers_attr) > 0:
        layer0 = layers_attr[0]
        names = _find_kv_names(layer0)
        if names is None:
            attrs = {a: type(getattr(layer0, a, None)).__name__
                     for a in dir(layer0) if not a.startswith("_")}
            raise AttributeError(
                f"cache.layers[0] K/V 텐서 탐지 실패.\n"
                f"속성 및 타입: {attrs}"
            )
        k_name, v_name = names
        logger.info(f"  [KV detect] cache.layers[i].{k_name} / .{v_name}  "
                    f"({len(layers_attr)} layers)")
        for i, layer in enumerate(layers_attr):
            k = getattr(layer, k_name)
            v = getattr(layer, v_name)
            if not isinstance(k, torch.Tensor):
                raise RuntimeError(
                    f"layers[{i}].{k_name} = {type(k).__name__} — "
                    f"is_initialized={getattr(layer, 'is_initialized', '?')}. "
                    f"prefill이 완료되지 않은 가방에서 _kv_layers()를 호출했습니다."
                )
            yield k, v
        return

    # Pattern A: list
    kc = getattr(cache, "key_cache", None)
    if isinstance(kc, list) and len(kc) > 0 and isinstance(kc[0], torch.Tensor):
        logger.info(f"  [KV detect] cache.key_cache list pattern ({len(kc)} layers)")
        yield from zip(kc, cache.value_cache)
        return

    # Pattern B: numbered attrs
    i = 0
    while True:
        k = getattr(cache, f"key_cache_{i}", None)
        if k is None:
            break
        yield k, getattr(cache, f"value_cache_{i}")
        i += 1
    if i > 0:
        logger.info(f"  [KV detect] key_cache_N pattern ({i} layers)")
        return

    # 탐지 실패 — 진단 정보 출력
    attrs = {a: type(getattr(cache, a, None)).__name__
             for a in dir(cache) if not a.startswith("__")}
    raise AttributeError(
        f"StaticCache K/V 탐지 실패.\n속성 및 타입: {attrs}"
    )


def _init_and_copy_layer(dst_layer, k_src: torch.Tensor, v_src: torch.Tensor,
                         k_name: str, v_name: str) -> None:
    """
    dst_layer에 K/V 데이터를 주입한다.

    Lazy Initialization 대응:
      - `lazy_initialization(dummy)` 호출로 is_initialized=True + 빈 텐서 할당
      - 이후 copy_로 실제 데이터 주입
      - lazy_initialization이 없으면 setattr + is_initialized 수동 설정

    이 방법이 필요한 이유:
      - update()는 `if not self.is_initialized:` 체크 → lazy_initialization을 재실행
      - setattr만 하면 is_initialized=False 그대로 → 첫 decode update()에서 텐서가 zeros로 덮어써짐
      - lazy_initialization을 먼저 호출하여 is_initialized=True로 만들어야 safe
    """
    is_init = getattr(dst_layer, "is_initialized", True)  # 속성 없으면 True로 가정 (이미 초기화됨)
    lazy_fn  = getattr(dst_layer, "lazy_initialization", None)

    if not is_init and callable(lazy_fn):
        # Lazy init 트리거.
        # transformers 버전마다 lazy_initialization 시그니처가 다름:
        #   Thor 실측: lazy_initialization(self, key_states)       → 인자 1개
        #   main 브랜치: lazy_initialization(self, key_states, value_states) → 인자 2개
        #   일부 버전:  lazy_initialization(self)                   → 인자 0개
        # try/except로 모든 버전 대응
        initialized = False
        for _args in [(k_src,), (k_src, v_src), ()]:
            try:
                lazy_fn(*_args)
                initialized = True
                break
            except TypeError:
                continue
        if not initialized:
            # lazy_fn 호출 실패 → is_initialized 수동 설정
            try:
                setattr(dst_layer, "is_initialized", True)
                setattr(dst_layer, "dtype",  k_src.dtype)
                setattr(dst_layer, "device", k_src.device)
            except Exception:
                pass
        # 이제: dst_layer.keys = zeros[B, H, max_cache_len, D], is_initialized=True ✓

    existing_k = getattr(dst_layer, k_name, None)
    if isinstance(existing_k, torch.Tensor):
        # 할당된 버퍼에 in-place copy (GPU DMA)
        existing_k.copy_(k_src)
        getattr(dst_layer, v_name).copy_(v_src)
    else:
        # fallback: 직접 setattr
        setattr(dst_layer, k_name, k_src.contiguous().clone())
        setattr(dst_layer, v_name, v_src.contiguous().clone())
        try:
            setattr(dst_layer, "is_initialized", True)
        except Exception:
            pass


def make_bag(
    vlm_text_config,
    prefill_len: int,
    max_decode: int,
    device: str,
    batch_size: int = 1,
) -> StaticCache:
    """
    최대 크기(prefill_len + max_decode + 4)로 사전 할당된 KV 가방 생성.
    가방의 크기는 고정 — 이후 update()로 내용만 채워짐.
    """
    return StaticCache(
        config=vlm_text_config,
        batch_size=batch_size,
        max_cache_len=prefill_len + max_decode + 4,
        device=device,
        dtype=torch.bfloat16,
    )


def clone_bag(
    src: StaticCache,
    n: int,
    vlm_text_config,
    device: str,
) -> tuple[StaticCache, float]:
    """
    B=1 KV 가방을 B=n으로 복제한다.

    핵심 순서:
      1. dst StaticCache 생성 (layers[i].keys=None, is_initialized=False)
      2. src layers에서 K/V 속성명 탐지
      3. 각 레이어: lazy_initialization → copy_ (K/V 주입) → 위치 카운터 동기화
      4. 타이밍 측정 (CUDA Event 기반)

    Lazy Initialization 대응:
      - dst layers는 처음에 keys=None, is_initialized=False 상태
      - _init_and_copy_layer()가 lazy_initialization(dummy)를 호출 → is_initialized=True
      - 이후 update()가 호출되어도 lazy_initialization을 재실행하지 않음 ✓

    위치 카운터 동기화:
      - _cumulative_length, cumulative_length, _seen_tokens, seen_tokens 등 best-effort
      - 이를 동기화하지 않으면 decode 첫 step의 KV가 position 0에 덮어쓰일 수 있음

    Returns: (dst_cache, clone_time_ms)
    """
    src_layers = getattr(src, "layers", None)
    if src_layers is None or len(src_layers) == 0:
        raise RuntimeError("src StaticCache에 layers 속성이 없거나 비어있습니다.")

    # K/V 속성명 탐지 (src는 prefill 완료 → tensors가 존재해야 함)
    k_name, v_name = _find_kv_names(src_layers[0]) or (None, None)
    if k_name is None:
        attrs = {a: type(getattr(src_layers[0], a, None)).__name__
                 for a in dir(src_layers[0]) if not a.startswith("_")}
        raise AttributeError(f"src.layers[0] K/V 속성명 탐지 실패.\n속성: {attrs}")

    # src의 첫 K 텐서로 dtype/shape 확인 (prefill 완료 후이므로 None이면 안 됨)
    first_k = getattr(src_layers[0], k_name)
    if not isinstance(first_k, torch.Tensor):
        raise RuntimeError(
            f"src.layers[0].{k_name} = {type(first_k).__name__} — "
            f"_run_prefill()가 먼저 호출되었는지 확인하세요."
        )

    dst = StaticCache(
        config=vlm_text_config,
        batch_size=n,
        max_cache_len=src.max_cache_len,
        device=device,
        dtype=first_k.dtype,
    )
    dst_layers = getattr(dst, "layers")

    sw = CudaStopwatch()
    torch.cuda.synchronize()
    sw.start()

    for i in range(len(src_layers)):
        sk = getattr(src_layers[i], k_name)  # [1, H, max_cache_len, D]
        sv = getattr(src_layers[i], v_name)

        # lazy_initialization 트리거 + K/V 복사 (B=1 → B=n via expand)
        _init_and_copy_layer(
            dst_layer=dst_layers[i],
            k_src=sk.expand(n, -1, -1, -1),
            v_src=sv.expand(n, -1, -1, -1),
            k_name=k_name,
            v_name=v_name,
        )

        # 위치 카운터 동기화 (best-effort)
        # _cumulative_length: update() 내부에서 쓰기 위치를 결정하는 텐서 카운터
        for attr in ("_cumulative_length", "cumulative_length",
                     "_seen_tokens", "seen_tokens"):
            src_val = getattr(src_layers[i], attr, None)
            if src_val is None:
                continue
            dst_val = getattr(dst_layers[i], attr, None)
            try:
                if isinstance(dst_val, torch.Tensor):
                    if isinstance(src_val, torch.Tensor):
                        dst_val.copy_(src_val)
                    else:
                        dst_val.fill_(int(src_val))
                elif isinstance(src_val, torch.Tensor):
                    setattr(dst_layers[i], attr, src_val.clone())
                else:
                    setattr(dst_layers[i], attr, src_val)
            except Exception:
                pass

    clone_ms = sw.stop_ms()

    # cache 레벨 카운터도 동기화
    for attr in ("_seen_tokens", "seen_tokens", "_cache_position"):
        val = getattr(src, attr, None)
        if val is not None:
            try:
                setattr(dst, attr, val)
            except Exception:
                pass

    logger.info(
        f"  clone_bag: B=1→{n}, "
        f"K shape {list(sk.shape)} → [{n},{sk.shape[1]},{sk.shape[2]},{sk.shape[3]}], "
        f"clone_ms={clone_ms:.1f}ms"
    )
    return dst, clone_ms


# ──────────────────────────────────────────────────────────────────────────────
# 핵심 실험 함수
# ──────────────────────────────────────────────────────────────────────────────

def _run_prefill(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,    # [1, prefill_len]
    tok_data: dict,             # attention_mask, pixel_values, image_grid_thw (input_ids 없음)
    vlm_text_config,
    prefill_len: int,
) -> tuple[StaticCache, torch.Tensor, float]:
    """
    B=1 prefill을 StaticCache에 실행.

    Returns:
      bag           : 채워진 KV 가방 (B=1)
      first_logits  : [1, vocab_size] 첫 decode 위치 logits
      prefill_ms    : prefill 소요 시간 (ms)
    """
    bag = make_bag(vlm_text_config, prefill_len, MAX_DECODE_STEPS, DEVICE, batch_size=1)
    cache_pos = torch.arange(prefill_len, device=DEVICE, dtype=torch.long)

    sw = CudaStopwatch()
    torch.cuda.synchronize()
    sw.start()

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
    first_logits = out.logits[:, -1, :].float()  # [1, vocab_size]
    return bag, first_logits, prefill_ms


def _run_decode_loop(
    model: Alpamayo1_5,
    bag: StaticCache,
    first_logits: torch.Tensor,  # [B, vocab_size]
    prefill_len: int,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
    batch_size: int,
) -> tuple[torch.Tensor, float, list[int]]:
    """
    B=batch_size decode loop — 조기종료(Early Exit) 지원.

    조기종료 전략 (두 가지 조합):

    1. GPU Mask (0 extra syncs):
       EOS를 만난 시퀀스는 이후 스텝에서 cur = EOS 토큰으로 교체.
       → forward pass는 계속 돌지만 무해한 입력으로 KV cache 오염 최소화.
       → GPUEosTracker.update()의 '& ~self.found' 가드로 재기록 방지.
       → 이 단계만으론 루프를 멈추지 않음.

    2. Periodic CPU Sync (매 EOS_CHECK_INTERVAL 스텝, 총 ~5회):
       tracker.found.all().item() — 모든 배치 항목이 EOS를 만났으면 break.
       → EOS를 놓칠 위험 없음: tracker는 매 스텝 GPU에서 업데이트하므로
         break 시점에 eos_steps에 정확한 위치가 이미 기록되어 있음.

    3. Hard Upper Bound (MAX_DECODE_STEPS):
       EOS가 오지 않는 비정상 출력에 대한 최종 안전망.
       이 값을 줄이는 것은 위험 — 줄이지 말 것.

    Returns:
      token_buffer : [B, MAX_DECODE_STEPS] 생성된 토큰
      decode_ms    : decode loop 소요 시간 (CudaEvent 기반)
      actual_steps : [B] 각 배치 항목의 실제 생성 토큰 수
    """
    # Step 0: 첫 토큰 샘플링 (prefill logits 사용)
    lgts = first_logits.clone()
    lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
    next_tok = top_p_sample_gpu(lgts)  # [B]

    buf = torch.zeros(batch_size, MAX_DECODE_STEPS, dtype=torch.long, device=DEVICE)
    tracker = GPUEosTracker(batch_size, MAX_DECODE_STEPS, eos_id, DEVICE)
    buf[:, 0] = next_tok
    tracker.update(next_tok, step=0)
    cur = next_tok.unsqueeze(1)  # [B, 1]

    # Step 0에서 이미 모든 배치가 EOS (이론상 드물지만 방어)
    if tracker.found.all().item():
        eos_pos = tracker.get_eos_positions()
        actual = [p + 1 if p < MAX_DECODE_STEPS else MAX_DECODE_STEPS for p in eos_pos]
        return buf, 0.0, actual

    # ── Step 1 ~ MAX_DECODE_STEPS-1: decode loop (조기종료) ──────────────────
    sw = CudaStopwatch()
    torch.cuda.synchronize()
    sw.start()

    exited_at = MAX_DECODE_STEPS - 1  # 로깅용: 실제 루프 종료 스텝
    for step in range(1, MAX_DECODE_STEPS):

        # ── 1. GPU Mask: EOS 이미 만난 시퀀스 → EOS 토큰으로 입력 대체 ────────
        # tracker.found: [B] bool (GPU 상주). CPU sync 없음.
        already_done = tracker.found.unsqueeze(1)           # [B, 1]
        eos_fill     = torch.full_like(cur, eos_id)         # [B, 1]
        cur_in       = torch.where(already_done, eos_fill, cur)  # [B, 1]

        cpos = torch.tensor([prefill_len + step], device=DEVICE, dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.vlm(
                input_ids=cur_in,          # masked input (EOS-padded for done seqs)
                past_key_values=bag,
                cache_position=cpos,
                use_cache=True,
            )
        lgts = o.logits[:, -1, :].float()
        lgts[:, traj_offset:traj_offset + traj_vocab_size] = float("-inf")
        next_tok = top_p_sample_gpu(lgts)
        buf[:, step] = next_tok
        tracker.update(next_tok, step)     # found 항목은 '& ~self.found'로 재기록 방지
        cur = next_tok.unsqueeze(1)

        # ── 2. Periodic CPU Sync: 전체 완료 여부 확인 → 조기 탈출 ─────────────
        # EOS_CHECK_INTERVAL 스텝마다 1회 GPU→CPU transfer.
        # found.all()이 True면 모든 배치 항목이 EOS를 만났음 → break.
        # break 시점에 tracker.eos_steps에 정확한 EOS 위치가 이미 기록됨 → 안전.
        if step % EOS_CHECK_INTERVAL == 0 and tracker.found.all().item():
            exited_at = step
            break

    decode_ms = sw.stop_ms()
    logger.info(
        f"  [decode] exited at step {exited_at + 1} / {MAX_DECODE_STEPS}"
        f"  ({decode_ms:.1f}ms,  "
        f"util={(exited_at + 1) / MAX_DECODE_STEPS * 100:.0f}%)"
    )

    # 루프 종료 후 1회 CPU sync (tracker 최종 확정)
    eos_pos = tracker.get_eos_positions()
    actual = [p + 1 if p < MAX_DECODE_STEPS else MAX_DECODE_STEPS for p in eos_pos]
    return buf, decode_ms, actual


def run_phase_A(
    model: Alpamayo1_5,
    input_ids_1: torch.Tensor,  # [1, prefill_len] (fuse_traj_tokens 적용 완료)
    tok_data_1: dict,           # B=1 attention_mask, pixel_values, image_grid_thw
    vlm_text_config,
    prefill_len: int,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
    N: int,
) -> dict:
    """
    Phase A Baseline: B=N prefill + B=N decode.
    input_ids_1, tok_data_1은 B=1 기준 — 내부에서 B=N으로 확장.
    """
    # B=N 확장
    input_ids_N = input_ids_1.repeat(N, 1)  # [N, L]

    attn_mask_1 = tok_data_1.get("attention_mask")
    pv_1        = tok_data_1.get("pixel_values")
    thw_1       = tok_data_1.get("image_grid_thw")

    attn_mask_N = attn_mask_1.repeat(N, 1) if attn_mask_1 is not None else None
    # pixel_values/image_grid_thw: Qwen-VL은 [n_patches, C, pH, pW] 구조
    # repeat_interleave(N, dim=0)가 HF generate()의 expand_inputs_for_generation과 동일
    pv_N  = pv_1.repeat_interleave(N, dim=0)  if pv_1  is not None else None
    thw_N = thw_1.repeat_interleave(N, dim=0) if thw_1 is not None else None

    bag_N     = make_bag(vlm_text_config, prefill_len, MAX_DECODE_STEPS, DEVICE, batch_size=N)
    cache_pos = torch.arange(prefill_len, device=DEVICE, dtype=torch.long)

    # Prefill
    sw_p = CudaStopwatch()
    torch.cuda.synchronize()
    sw_p.start()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm(
            input_ids=input_ids_N,
            attention_mask=attn_mask_N,
            pixel_values=pv_N,
            image_grid_thw=thw_N,
            past_key_values=bag_N,
            cache_position=cache_pos,
            use_cache=True,
        )

    prefill_ms   = sw_p.stop_ms()
    first_logits = out.logits[:, -1, :].float()  # [N, vocab]

    # Decode
    _, decode_ms, actual = _run_decode_loop(
        model, bag_N, first_logits, prefill_len, eos_id,
        traj_offset, traj_vocab_size, N,
    )

    mean_actual = float(np.mean(actual))
    total_ms    = prefill_ms + decode_ms
    tps         = (N * mean_actual) / (total_ms / 1000.0)

    return {
        "N":                 N,
        "prefill_ms":        prefill_ms,
        "clone_ms":          0.0,
        "decode_ms":         decode_ms,
        "total_ms":          total_ms,
        "actual_steps_mean": mean_actual,
        "tokens_per_sec":    tps,
    }


def run_phase_B(
    model: Alpamayo1_5,
    input_ids_1: torch.Tensor,
    tok_data_1: dict,
    vlm_text_config,
    prefill_len: int,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
    N: int,
) -> dict:
    """
    Phase B Shared Prefill: B=1 prefill → clone → B=N decode.
    핵심: prefill은 B=1로 1회만 실행, 그 KV 가방을 N개 복제.
    """
    # B=1 prefill
    bag_1, first_logits_1, prefill_ms = _run_prefill(
        model, input_ids_1, tok_data_1, vlm_text_config, prefill_len
    )
    logger.info(f"  prefill (B=1): {prefill_ms:.1f}ms")

    # B=1 → B=N clone
    bag_N, clone_ms = clone_bag(bag_1, N, vlm_text_config, DEVICE)

    # 첫 logits: [1, vocab] → [N, vocab] view (multinomial이 행별 독립 샘플링)
    first_logits_N = first_logits_1.expand(N, -1)

    # B=N decode
    _, decode_ms, actual = _run_decode_loop(
        model, bag_N, first_logits_N, prefill_len, eos_id,
        traj_offset, traj_vocab_size, N,
    )

    mean_actual = float(np.mean(actual))
    total_ms    = prefill_ms + clone_ms + decode_ms
    tps         = (N * mean_actual) / (total_ms / 1000.0)

    return {
        "N":                 N,
        "prefill_ms":        prefill_ms,
        "clone_ms":          clone_ms,
        "decode_ms":         decode_ms,
        "total_ms":          total_ms,
        "actual_steps_mean": mean_actual,
        "tokens_per_sec":    tps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 반복 실행 래퍼
# ──────────────────────────────────────────────────────────────────────────────

def run_repeated(phase_fn, phase_name: str, **kwargs) -> dict:
    """phase_fn을 NUM_WARMUP + NUM_MEASURE회 실행해 통계를 낸다."""
    logger.info(f"\n{'='*65}")
    logger.info(f"  {phase_name}")
    logger.info(f"{'='*65}")

    totals: list[float] = []
    last: dict | None = None

    for idx in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = idx < NUM_WARMUP
        torch.cuda.empty_cache()

        try:
            r = phase_fn(**kwargs)
        except Exception as e:
            logger.error(f"  FAILED at run {idx}: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

        tag = f"WARMUP {idx + 1}" if is_warmup else f"MEASURE {idx - NUM_WARMUP + 1}"
        clone_str = f"  clone={r.get('clone_ms', 0):5.0f}ms" if r.get("clone_ms", 0) > 0 else ""
        logger.info(
            f"  [{tag}]  total={r['total_ms']:7.1f}ms"
            f"  prefill={r['prefill_ms']:6.0f}ms"
            f"{clone_str}"
            f"  decode={r['decode_ms']:6.0f}ms"
            f"  steps≈{r['actual_steps_mean']:.1f}"
            f"  tps={r['tokens_per_sec']:.1f}"
        )

        if not is_warmup:
            totals.append(r["total_ms"])
            last = r

    mu, std = mean_std(totals)
    logger.info(f"\n  SUMMARY  total={mu:.1f}±{std:.1f}ms")

    return {
        "phase":          phase_name,
        "total_mean_ms":  mu,
        "total_std_ms":   std,
        "last":           last,
    }


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    data     = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(DEVICE)
    model.eval()

    # ── 토크나이즈 ──────────────────────────────────────────────────────────
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

    # fuse_traj_tokens: B=1로 1번만 실행 (이후 expand)
    ego_data = helper.to_device(
        {"ego_history_xyz": data["ego_history_xyz"],
         "ego_history_rot": data["ego_history_rot"]},
        DEVICE,
    )
    input_ids_raw  = inputs.pop("input_ids")       # [1, L_raw]
    input_ids_1    = model.fuse_traj_tokens(input_ids_raw, ego_data)  # [1, L]
    prefill_len    = input_ids_1.shape[1]

    # tok_data_1: input_ids 없이 남은 텐서들 (attention_mask, pixel_values, image_grid_thw)
    tok_data_1     = inputs  # pop 이후 상태

    eos_id          = model.tokenizer.convert_tokens_to_ids(
                          to_special_token("traj_future_start"))
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    vlm_text_config = model.vlm.config.text_config

    # ── 크기 정보 출력 ──────────────────────────────────────────────────────
    n_layers   = vlm_text_config.num_hidden_layers
    n_kv_heads = vlm_text_config.num_key_value_heads
    head_dim   = vlm_text_config.hidden_size // vlm_text_config.num_attention_heads
    bpt        = n_layers * 2 * n_kv_heads * head_dim * 2  # BF16 = 2 bytes per element
    bag_mb     = (prefill_len + MAX_DECODE_STEPS + 4) * bpt / 1024 / 1024

    logger.info(f"\nprefill_len={prefill_len}, eos_id={eos_id}, "
                f"traj_offset={traj_offset}, traj_vocab={traj_vocab_size}")
    logger.info(f"MAX_DECODE_STEPS={MAX_DECODE_STEPS}, N_SWEEP={N_SWEEP}")
    logger.info(f"KV bag B=1: {bag_mb:.1f} MB  "
                f"(layers={n_layers}, kv_heads={n_kv_heads}, head_dim={head_dim})")

    # ── StaticCache K/V 속성명 탐지 사전 확인 ──────────────────────────────────
    # 주의: 빈 가방은 keys/values=None (Lazy Init 패턴) — None이어도 탐지 성공이 정상
    logger.info("\n[Sanity check] StaticCache K/V 속성명 탐지 테스트...")
    _test_bag = make_bag(vlm_text_config, prefill_len, MAX_DECODE_STEPS, DEVICE, batch_size=1)
    _test_layers = getattr(_test_bag, "layers", None)
    if _test_layers and len(_test_layers) > 0:
        _names = _find_kv_names(_test_layers[0])
        if _names is None:
            attrs = {a: type(getattr(_test_layers[0], a, None)).__name__
                     for a in dir(_test_layers[0]) if not a.startswith("_")}
            logger.error(f"  ✗ K/V 속성명 탐지 실패.\n    속성: {attrs}")
            return
        k0_val = getattr(_test_layers[0], _names[0], None)
        is_lazy = not isinstance(k0_val, torch.Tensor)
        logger.info(
            f"  ✓ 속성명=({_names[0]!r}, {_names[1]!r})  "
            f"is_initialized={getattr(_test_layers[0], 'is_initialized', 'N/A')}  "
            f"lazy_init={'있음' if callable(getattr(_test_layers[0], 'lazy_initialization', None)) else '없음'}  "
            f"{'(None=lazy-init, 정상)' if is_lazy else f'K shape={list(k0_val.shape)}'}"
        )
    else:
        kc = getattr(_test_bag, "key_cache", None)
        if not isinstance(kc, (list, torch.Tensor)):
            logger.error("  ✗ StaticCache 구조를 인식할 수 없습니다.")
            return
        logger.info(f"  ✓ key_cache pattern (older transformers)")
    del _test_bag

    torch.cuda.manual_seed_all(42)
    all_results: dict = {}

    # ── Phase A: Baseline ───────────────────────────────────────────────────
    for N in N_SWEEP:
        key = f"phase_A_N{N}"
        r   = run_repeated(
            phase_fn=run_phase_A,
            phase_name=f"Phase A  Baseline  N={N}",
            model=model,
            input_ids_1=input_ids_1,
            tok_data_1=tok_data_1,
            vlm_text_config=vlm_text_config,
            prefill_len=prefill_len,
            eos_id=eos_id,
            traj_offset=traj_offset,
            traj_vocab_size=traj_vocab_size,
            N=N,
        )
        all_results[key] = r

    # ── Phase B: Shared Prefill ─────────────────────────────────────────────
    for N in N_SWEEP:
        key = f"phase_B_N{N}"
        r   = run_repeated(
            phase_fn=run_phase_B,
            phase_name=f"Phase B  Shared Prefill  N={N}",
            model=model,
            input_ids_1=input_ids_1,
            tok_data_1=tok_data_1,
            vlm_text_config=vlm_text_config,
            prefill_len=prefill_len,
            eos_id=eos_id,
            traj_offset=traj_offset,
            traj_vocab_size=traj_vocab_size,
            N=N,
        )
        all_results[key] = r

    # ── 비교 테이블 ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 78)
    logger.info("  SHARED PREFILL EXPERIMENT — RESULTS SUMMARY")
    logger.info("=" * 78)

    hdr = (f"{'Phase':<35}|{'total(ms)':>10}|{'prefill':>8}|"
           f"{'clone':>7}|{'decode':>8}|{'tps':>8}|{'speedup':>8}")
    logger.info(hdr)
    logger.info("-" * 78)

    for N in N_SWEEP:
        for label, key in [
            (f"A (baseline)       N={N}", f"phase_A_N{N}"),
            (f"B (shared prefill) N={N}", f"phase_B_N{N}"),
        ]:
            r = all_results.get(key, {})
            if "error" in r:
                logger.info(f"  {label:<33}| FAILED: {r['error']}")
                continue
            mu   = r.get("total_mean_ms", 0.0)
            last = r.get("last", {}) or {}
            # speedup: Phase A-N 대비 Phase B-N
            ref_key = f"phase_A_N{N}"
            ref_mu  = all_results.get(ref_key, {}).get("total_mean_ms", 0.0)
            sp_str  = f"{ref_mu / mu:.2f}×" if (mu > 0 and "B" in key and ref_mu > 0) else "  1.00×" if "A" in key else "  ?"
            logger.info(
                f"  {label:<33}|{mu:>8.0f}ms|"
                f"{last.get('prefill_ms', 0):>6.0f}ms|"
                f"{last.get('clone_ms',   0):>5.0f}ms|"
                f"{last.get('decode_ms',  0):>6.0f}ms|"
                f"{last.get('tokens_per_sec', 0):>6.1f}t/s|"
                f"{sp_str:>8}"
            )
        logger.info("-" * 78)

    # ── 결과 저장 ───────────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()
                    if not isinstance(v, (torch.Tensor,))}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    out_json = OUT / "results.json"
    with open(out_json, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    logger.info(f"\nResults saved: {out_json}")
    logger.info("[모든 실험 완료]")


if __name__ == "__main__":
    main()
