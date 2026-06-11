"""
260524_async_pipeline.py  (v3 — hook-based pipeline)
======================================================
RT-Swap + Demand Layering 기반 Async Pipeline PoC  (Phase 1)

논문 → 코드 대응:
  RT-Swap (RTAS'24) §4.1:
    "Issue DMA prefetch on separate stream, compute on another stream, overlap them"
    → prefetch_stream + default stream 동시 실행

  Demand Layering (RTSS'22) §2.2:
    "Prefetch layer i+1 while executing layer i"
    → forward_pre_hook: layer i 실행 전에 i+1 prefetch 발행

  Thor iGPU 특화:
    DMA → cudaMemPrefetchAsync  → L2 cache warming (tensor.sum())
    이유: 통합 메모리 환경 → 실제 데이터 이동 없음, L2 hint만 필요

v3 변경 내용 (2026-05-26):
  ● 근본 원인 분석: Qwen3VLTextDecoderLayer 는 model-level rotary_emb 가
    계산한 position_embeddings=(cos,sin) 이 있어야 호출 가능.
    _get_position_embeddings_and_ids 는 layers[0].self_attn.rotary_emb
    (존재하지 않음)을 시도 → 개별 레이어 직접 호출이 전부 실패.
  ● 해결 전략: lm_model.forward() 를 실행 주체로 사용.
    - lm_model (Qwen3VLTextModel) 이 mrope 내부 처리
    - forward_pre_hook 으로 각 레이어 직전 prefetch 발행 (→ prefetch_stream)
    - pre_hook 시작 시 wait_stream(prefetch_stream) 으로 직전 prefetch 완료 보장
    - compute 는 default stream 에서 진행 (lm_model.forward() 내부)
    ⟹ prefetch (prefetch_stream) ∥ compute (default stream) 동시 실행 = RT-Swap

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  python3 260524_async_pipeline.py --mode [baseline|async|both]

결과 해석:
  speedup > 1.1× → 파이프라인 효과 있음
  speedup ≈ 1.0× → compute time이 너무 짧거나 prefetch overhead > 이득
"""

import argparse
import json
import os
import traceback

import numpy as np
import torch

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Alpamayo Async Pipeline PoC")
parser.add_argument("--mode",         default="both",
                    choices=["baseline", "async", "both"])
parser.add_argument("--decode_steps", type=int, default=65)
parser.add_argument("--warmup",       type=int, default=3)
args = parser.parse_args()

RESULT_FILE = os.path.expanduser(
    "~/alpamayo1.5/profiling_results/260524_async_pipeline.json"
)
MODEL_ID = "nvidia/Alpamayo-1.5-10B"


# ── Prefetch 헬퍼 ─────────────────────────────────────────────────────────────
def prefetch_weights_to_l2(tensor: torch.Tensor,
                            stream: torch.cuda.Stream) -> None:
    """
    [Demand Layering §2.2 / RT-Swap §4.1] 텐서를 GPU L2 캐시로 워밍.

    Thor iGPU 환경:
      cudaMemPrefetchAsync 는 cudaMallocManaged 전용 → PyTorch 텐서 불가.
      대신: 별도 stream 에서 tensor 를 읽는 커널 → L2 캐시 라인 채움.
      (tensor.view(-1).sum() 은 전체 원소를 읽으므로 캐시 라인 채움 효과)

    prefetch_stream 에서 실행 → compute 의 default stream 과 동시 실행 가능.
    """
    with torch.cuda.stream(stream):
        _ = tensor.view(-1).sum()


def get_prefetch_targets(layer: torch.nn.Module) -> list:
    """
    레이어의 prefetch 대상 텐서 반환.
    MLP (100.7 MB) 와 Q/O projection (33.6 MB) 을 우선 대상으로 삼음.
    KV proj 는 GQA 로 인해 상대적으로 작음 (8.4 MB) → 선택 포함.
    """
    targets = []
    for name, param in layer.named_parameters():
        if any(k in name for k in ["gate_proj", "up_proj", "down_proj",
                                    "q_proj", "o_proj", "k_proj", "v_proj"]):
            targets.append(param.data)
    return targets


# ── lm_model + layers 탐색 ────────────────────────────────────────────────────
def get_lm_model_and_layers(model):
    """
    (lm_model, layers) 반환.

    lm_model : Transformer 텍스트 백본 (forward(inputs_embeds=...) 가능)
    layers   : List[Qwen3VLTextDecoderLayer]

    Alpamayo 1.5 구조 (260513_profile_v4.py 검증):
      model.vlm.language_model.model       ← lm_model  (Qwen3VLTextModel)
      model.vlm.language_model.model.layers ← decoder layers
    """
    candidates = [
        (lambda m: m.vlm.language_model.model,
         lambda m: list(m.vlm.language_model.model.layers)),
        (lambda m: m.vlm.model,
         lambda m: list(m.vlm.model.layers)),
        (lambda m: m.vlm.language_model,
         lambda m: list(m.vlm.language_model.layers)),
        (lambda m: m.vlm.model.language_model,
         lambda m: list(m.vlm.model.language_model.layers)),
    ]
    for lm_fn, layer_fn in candidates:
        try:
            lm_model = lm_fn(model)
            layers   = layer_fn(model)
            if lm_model is not None and layers:
                return lm_model, layers
        except AttributeError:
            continue
    raise RuntimeError("Transformer decoder layers 탐색 실패")


def find_working_lm_kwargs(lm_model, dummy_hidden) -> dict:
    """
    lm_model.forward() 에서 작동하는 kwargs 탐색.
    반환: 작동하는 kwargs dict (텐서 값 포함)
    """
    device  = dummy_hidden.device
    seq_len = dummy_hidden.shape[1]

    position_ids_3d = torch.zeros(3, 1, seq_len, dtype=torch.long, device=device)
    position_ids_1d = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    cache_pos       = torch.arange(seq_len, dtype=torch.long, device=device)

    candidates = [
        {"inputs_embeds": dummy_hidden,
         "position_ids":  position_ids_3d,
         "use_cache":     False},
        {"inputs_embeds": dummy_hidden,
         "position_ids":  position_ids_3d,
         "use_cache":     False,
         "return_dict":   False},
        {"inputs_embeds": dummy_hidden,
         "position_ids":  position_ids_3d,
         "use_cache":     False,
         "cache_position": cache_pos},
        {"inputs_embeds": dummy_hidden, "use_cache": False},
        {"inputs_embeds": dummy_hidden},
        {"inputs_embeds": dummy_hidden,
         "position_ids":  position_ids_1d,
         "use_cache":     False},
    ]

    errors = []
    for kwargs in candidates:
        try:
            with torch.no_grad():
                lm_model(**kwargs)
            torch.cuda.synchronize()
            print(f"      lm_model forward 성공: {list(kwargs.keys())}")
            return kwargs
        except Exception as exc:
            errors.append(f"  {list(kwargs.keys())}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"lm_model.forward() 모든 조합 실패 "
        f"(type: {type(lm_model).__name__})\n" + "\n".join(errors)
    )


# ── Baseline Decoder ──────────────────────────────────────────────────────────
class BaselineDecoder:
    """
    기존 방식: lm_model.forward() 를 그대로 실행.
    [비교 기준선] — 파이프라인 없이 순차 실행.

    v3: lm_model.forward() 를 사용하므로 mrope 관련 실패 없음.
    """

    def __init__(self, lm_model, working_kwargs: dict):
        self.lm_model      = lm_model
        self.working_kwargs = working_kwargs

    def decode_step(self, dummy_hidden):
        with torch.no_grad():
            self.lm_model(**self.working_kwargs)

    def warmup(self, dummy_hidden, n: int):
        for _ in range(n):
            self.decode_step(dummy_hidden)
            torch.cuda.synchronize()


# ── Async Pipeline Decoder ────────────────────────────────────────────────────
class AsyncPipelineDecoder:
    """
    RT-Swap + Demand Layering 기반 Async Pipeline Decoder.

    구현 방식 (v3 — hook-based):
    ────────────────────────────────────────────────────────────────
    lm_model.forward() 를 실행 주체로 사용하면서,
    각 레이어에 forward_pre_hook 을 걸어 prefetch 를 주입한다.

    Hook 타임라인 (레이어 i 기준):
      [DEFAULT STREAM]  [PREFETCH_STREAM]
      ...
      layer i pre_hook 진입:
        └─ wait_stream(prefetch_stream)        ← i-1 이 요청한 i 의 prefetch 완료 보장
        └─ issue prefetch(layers[i+1])         → prefetch_stream (비동기, 즉시 반환)
      layer i forward:
        └─ compute                             ← default stream
        └─ prefetch for i+1 실행 중            ← prefetch_stream (동시!)
      layer i+1 pre_hook 진입:
        └─ wait_stream(prefetch_stream)        ← i+1 prefetch 완료 보장
        └─ issue prefetch(layers[i+2])
      ...

    RT-Swap §3.1 수식: T_async = Σ max(t_prefetch_{i+1}, t_compute_i)
    Demand Layering §2.2: "Prefetch layer i+1 while computing layer i"
    ────────────────────────────────────────────────────────────────

    Note: forward_pre_hook 은 default stream 에서 발행되므로,
    wait_stream 은 "default stream이 prefetch_stream 완료를 기다림" 을 의미.
    실제 compute 는 wait_stream 이후에 시작 → 가중치 가용 보장.
    """

    def __init__(self, lm_model, layers, working_kwargs: dict):
        self.lm_model       = lm_model
        self.layers         = layers
        self.working_kwargs = working_kwargs
        self.prefetch_stream = torch.cuda.Stream()

    def decode_step(self, dummy_hidden):
        N               = len(self.layers)
        prefetch_stream = self.prefetch_stream

        # ── Hook 등록 ──────────────────────────────────────────────────────────
        # 각 레이어에 pre_hook 설치:
        #   1) wait_stream(prefetch_stream): 이 레이어의 prefetch 완료 대기
        #   2) issue prefetch for 다음 레이어 (비동기)
        handles = []

        def make_pre_hook(idx: int):
            """
            idx 번째 레이어의 pre_hook:
              - wait_stream: idx 레이어 가중치 prefetch 완료 대기
              - issue:       idx+1 레이어 가중치 prefetch 발행 (비동기)
            """
            def pre_hook(module, args):
                # (1) 이 레이어의 prefetch 완료 보장
                torch.cuda.current_stream().wait_stream(prefetch_stream)
                # (2) 다음 레이어 prefetch 비동기 발행
                if idx + 1 < N:
                    for t in get_prefetch_targets(self.layers[idx + 1]):
                        prefetch_weights_to_l2(t, prefetch_stream)
            return pre_hook

        for i, layer in enumerate(self.layers):
            handles.append(layer.register_forward_pre_hook(make_pre_hook(i)))

        # ── Pipeline Priming: Layer 0 prefetch 선행 발행 ───────────────────────
        # Demand Layering §2.1 "Layer miss → Load" 의 iGPU 버전:
        # 첫 레이어는 pre_hook 이전에 이미 prefetch 시작되어야 함.
        for t in get_prefetch_targets(self.layers[0]):
            prefetch_weights_to_l2(t, self.prefetch_stream)

        # ── 실행 ───────────────────────────────────────────────────────────────
        try:
            with torch.no_grad():
                self.lm_model(**self.working_kwargs)
        finally:
            for h in handles:
                h.remove()

    def warmup(self, dummy_hidden, n: int):
        for _ in range(n):
            self.decode_step(dummy_hidden)
            torch.cuda.synchronize()


# ── Latency 측정 헬퍼 ─────────────────────────────────────────────────────────
def measure_decode_latency(decoder, dummy_hidden,
                            n_steps: int, n_warmup: int) -> dict:
    """n_steps 개 decode step 의 latency 를 CUDA Event 로 측정."""
    decoder.warmup(dummy_hidden, n_warmup)

    times_ms = []
    for _ in range(n_steps):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        decoder.decode_step(dummy_hidden)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    arr = np.array(times_ms)
    return {
        "mean_ms":   float(arr.mean()),
        "std_ms":    float(arr.std()),
        "median_ms": float(np.median(arr)),
        "min_ms":    float(arr.min()),
        "total_ms":  float(arr.sum()),
        "all_ms":    times_ms,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Alpamayo Async Pipeline PoC  (v3 — hook-based)")
print(f"Mode: {args.mode}  |  Steps: {args.decode_steps}  |  Warmup: {args.warmup}")
print("=" * 65)

# ── 모델 로드 ────────────────────────────────────────────────────────────────
print(f"\n[1/3] 모델 로딩: {MODEL_ID}")
MODEL_LOADED = False
model        = None
lm_model     = None
layers       = None
working_kwargs = None

try:
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    model = Alpamayo1_5.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    MODEL_LOADED = True
    print("      모델 로드 완료")
except Exception as exc:
    print(f"      모델 로드 실패: {exc}")
    traceback.print_exc()

# ── 더미 입력 생성 ───────────────────────────────────────────────────────────
D_MODEL = 4096
if MODEL_LOADED:
    for path_str, fn in [
        ("vlm.config.text_config.hidden_size",
         lambda m: m.vlm.config.text_config.hidden_size),
        ("vlm.language_model.config.hidden_size",
         lambda m: m.vlm.language_model.config.hidden_size),
    ]:
        try:
            D_MODEL = fn(model)
            print(f"      D_MODEL: {D_MODEL}  ({path_str})")
            break
        except AttributeError:
            continue
    else:
        print(f"      D_MODEL: {D_MODEL}  (기본값)")

dummy_hidden = torch.randn(1, 1, D_MODEL, dtype=torch.bfloat16, device="cuda")
print(f"      더미 입력: {list(dummy_hidden.shape)},  {dummy_hidden.dtype}")

# ── lm_model + working_kwargs 탐색 ──────────────────────────────────────────
if MODEL_LOADED:
    try:
        lm_model, layers = get_lm_model_and_layers(model)
        print(f"      lm_model:    {type(lm_model).__name__}")
        print(f"      레이어 수:   {len(layers)}")
        print(f"      레이어 타입: {type(layers[0]).__name__}")

        working_kwargs = find_working_lm_kwargs(lm_model, dummy_hidden)
    except Exception as exc:
        print(f"      lm_model / forward 탐색 실패: {exc}")
        traceback.print_exc()
        MODEL_LOADED = False

# ── 측정 ─────────────────────────────────────────────────────────────────────
print(f"\n[2/3] Latency 측정")
results = {}

if MODEL_LOADED and lm_model is not None and working_kwargs is not None:

    if args.mode in ("baseline", "both"):
        print("  [Baseline] 측정 중...")
        baseline = BaselineDecoder(lm_model, working_kwargs)
        stats    = measure_decode_latency(
            baseline, dummy_hidden, args.decode_steps, args.warmup
        )
        results["baseline"] = stats
        print(f"  Baseline : {stats['mean_ms']:7.2f} ms/step "
              f"(total {stats['total_ms']:.0f} ms,  {args.decode_steps} steps)")

    if args.mode in ("async", "both"):
        print("  [Async Pipeline] 측정 중...")
        async_dec = AsyncPipelineDecoder(lm_model, layers, working_kwargs)
        stats     = measure_decode_latency(
            async_dec, dummy_hidden, args.decode_steps, args.warmup
        )
        results["async"] = stats
        print(f"  Async    : {stats['mean_ms']:7.2f} ms/step "
              f"(total {stats['total_ms']:.0f} ms,  {args.decode_steps} steps)")

    if "baseline" in results and "async" in results:
        speedup = results["baseline"]["mean_ms"] / results["async"]["mean_ms"]
        results["speedup"] = float(speedup)

        print(f"\n{'─' * 45}")
        print(f"  Speedup: {speedup:.4f}×")
        if speedup > 1.3:
            print("  → 파이프라인 효과 큼! ✅")
            print("    Phase 2 권장: L2 Persistent KV Cache 추가 최적화")
        elif speedup > 1.05:
            print("  → 파이프라인 소폭 효과 있음")
            print("    권장: prefetch 대상 텐서 확대 또는 double buffering")
        elif speedup > 0.95:
            print("  → 파이프라인 효과 미미 (≈1×)")
            print("    layer_compute_profile 결과와 비교:")
            print("    t_compute < t_prefetch_DRAM 이면 compute가 너무 빠름")
            print("    t_compute > t_prefetch_DRAM 이면 sync overhead 확인 필요")
        else:
            print("  → Async 가 Baseline 보다 느림 (hook overhead 가능성)")
            print("    prefetch 대상 줄이거나 stream sync 방식 조정 필요")

        # 65-step 외삽
        b_total = results["baseline"]["mean_ms"] * 65
        a_total = results["async"]["mean_ms"] * 65
        print(f"\n  65-step 외삽:")
        print(f"    Baseline : {b_total:.0f} ms  =  {b_total / 1000:.2f} s")
        print(f"    Async    : {a_total:.0f} ms  =  {a_total / 1000:.2f} s")
        print(f"    절약 시간: {b_total - a_total:.0f} ms")

else:
    print("  모델 로드 / lm_model 탐색 실패 → 측정 skip")
    print("  260524_layer_compute_profile.py 의 diagnostic 출력 확인 후 재시도")

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
print(f"\n[3/3] 결과 저장")
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

# all_ms 는 JSON 비대해짐 → 통계값만 저장
def strip_all_ms(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "all_ms"}

summary = {
    "experiment":   "260524_async_pipeline",
    "version":      "v3-hook-based",
    "date":         "2026-05-26",
    "method":       "forward_pre_hook prefetch injection via lm_model.forward()",
    "mode":         args.mode,
    "decode_steps": args.decode_steps,
    "warmup":       args.warmup,
    "model":        MODEL_ID,
    "results": {
        k: (strip_all_ms(v) if isinstance(v, dict) else v)
        for k, v in results.items()
    },
    "speedup": results.get("speedup"),
}

with open(RESULT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print(f"    저장: {RESULT_FILE}")
print("\n완료!")
