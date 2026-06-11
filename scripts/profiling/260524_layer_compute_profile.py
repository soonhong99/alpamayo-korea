"""
260524_layer_compute_profile.py  (v3 — hook-based timing)
=========================================================
Alpamayo 레이어별 compute time 측정  (Phase 0 – Step 1)

목적:
  async pipeline 판단 기준인 t_compute_per_layer 를 측정한다.
  Demand Layering: BW_required = m_i / t_compute_i
  RT-Swap:         T_async = Σ max(t_prefetch_{i+1}, t_compute_i)
  → t_compute 를 알아야 "pipeline이 가능한지" 판단 가능

v3 변경 내용 (2026-05-26):
  ● 근본 원인 확인: Qwen3VLTextDecoderLayer 는 model-level rotary_emb 가
    사전 계산한 position_embeddings=(cos, sin) 을 필요로 하지만,
    layers[i].self_attn.rotary_emb 는 존재하지 않음.
    → 개별 레이어 직접 호출 8가지 조합이 모두 실패하는 이유.
  ● 해결책: lm_model.forward() 를 execution vehicle 로 사용.
    - lm_model (Qwen3VLTextModel) 이 mrope position_embeddings 를 내부 계산
    - register_forward_pre_hook / register_forward_hook + CUDA Events 로
      각 레이어의 시작·끝 시각을 기록 → per-layer elapsed_time
  ● 개별 레이어 직접 호출 코드 (_precompute_position_embeddings,
    _find_working_forward_kwargs, measure_layer_compute_time) 전부 제거.

실행:
  source ~/alpamayo1.5/a1_5_venv/bin/activate
  python3 260524_layer_compute_profile.py

결과 해석:
  t_compute > 0.444 ms  → MLP prefetch 완전 중첩 가능 ✅  (실측 DRAM 기준)
  t_compute < 0.444 ms  → prefetch가 bottleneck, 부분 중첩만 가능
  (0.444 ms = 100.7 MB / 227 GB/s — 260526_prefetch_effect_test 실측값)
"""

import json
import os
import traceback

import numpy as np
import torch

# ── 설정 ────────────────────────────────────────────────────────────────────
MODEL_ID      = "nvidia/Alpamayo-1.5-10B"
WARMUP_STEPS  = 3
MEASURE_STEPS = 20
RESULT_FILE   = os.path.expanduser(
    "~/alpamayo1.5/profiling_results/260524_layer_compute_profile.json"
)

# BW 실측값 (260526_prefetch_effect_test.py)
L2_BW_GBs        = 1126.0
DRAM_BW_GBs       = 227.0
MLP_WEIGHT_MB     = 100.7   # gate_proj 실측 (Qwen3VLTextDecoderLayer)
ATTN_QO_WEIGHT_MB = 33.6    # Q 또는 O proj
ATTN_KV_WEIGHT_MB = 8.4     # K 또는 V proj (GQA)


def ms_prefetch_l2(mb: float) -> float:
    return (mb * 1e6) / (L2_BW_GBs * 1e9) * 1e3


def ms_prefetch_dram(mb: float) -> float:
    return (mb * 1e6) / (DRAM_BW_GBs * 1e9) * 1e3


# ── 모델 로드 ────────────────────────────────────────────────────────────────
print("=" * 65)
print("Alpamayo Layer Compute Profiler  (v3 — hook-based)")
print("=" * 65)
print(f"\n[1/4] 모델 로딩: {MODEL_ID}")

try:
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5   # MEMORY.md 확인됨

    model = Alpamayo1_5.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",   # flash_attn2 는 nvcc 필요, eager 사용
        local_files_only=True,         # 네트워크 차단 — 캐시에서만 로드
    ).cuda().eval()
    print("      모델 로드 완료")
except Exception as exc:
    print(f"      모델 로드 실패: {exc}")
    traceback.print_exc()
    model = None


# ── 레이어 + lm_model 접근 ────────────────────────────────────────────────────
def get_lm_model_and_layers(model):
    """
    (lm_model, layers) 반환.

    lm_model : Transformer 텍스트 백본 (forward(inputs_embeds=...) 가능한 것)
    layers   : List[Qwen3VLTextDecoderLayer]

    확인된 Alpamayo 1.5 구조 (260513_profile_v4.py 검증):
      model.vlm                         → Qwen3VLForConditionalGeneration
      model.vlm.language_model          → 텍스트 causal LM wrapper
      model.vlm.language_model.model    → Qwen3VLTextModel  ← lm_model
      model.vlm.language_model.model.layers  ← decoder layer list

    아래 candidates 는 우선순위 순으로 탐색:
      1) vlm.language_model.model  (primary — 검증됨)
      2) vlm.model                 (Qwen2VL/구버전 fallback)
      3) vlm.language_model        (layers 가 직접 있는 경우)
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
    last_err = None
    for lm_fn, layer_fn in candidates:
        try:
            lm_model = lm_fn(model)
            layers   = layer_fn(model)
            if lm_model is not None and layers:
                return lm_model, layers
        except AttributeError as e:
            last_err = e
            continue

    # 모두 실패 → 진단 출력
    print("\n[DIAG] vlm 하위 모듈:")
    try:
        for name, mod in model.vlm.named_children():
            print(f"  vlm.{name}: {type(mod).__name__}")
            for n2, m2 in mod.named_children():
                suffix = "  ← layers!" if hasattr(m2, "layers") else ""
                print(f"    vlm.{name}.{n2}: {type(m2).__name__}{suffix}")
    except Exception:
        pass
    raise RuntimeError(f"Transformer decoder layers 탐색 실패: {last_err}")


def get_layer_weight_mb(layer) -> dict:
    """레이어 parameter 이름 → MB 매핑."""
    return {
        name: param.numel() * param.element_size() / 1e6
        for name, param in layer.named_parameters()
    }


# ── 핵심: hook-based per-layer timing ────────────────────────────────────────
def measure_all_layers_via_hooks(lm_model, layers, dummy_hidden,
                                  n_warmup: int = 3, n_measure: int = 20) -> dict:
    """
    lm_model.forward() 를 execution vehicle 로 사용하여
    각 Transformer 레이어의 compute time 을 CUDA Event hook 으로 측정.

    설계 근거:
    ─────────────────────────────────────────────────────────────────
    Qwen3VLTextDecoderLayer.forward() 는 keyword arg 로
    position_embeddings=(cos, sin) 을 받는다. 이 텐서는
    model-level rotary_emb (Qwen3VLRotaryEmbedding) 가 3D mrope
    position_ids [3, batch, seq] 를 받아 계산한다.
    개별 레이어에는 rotary_emb 가 없으므로 직접 호출 시
    position_embeddings=None → TypeError/RuntimeError 불가피.

    lm_model (= vlm.language_model.model, Qwen3VLTextModel) 은:
      1) inputs_embeds 를 받아 hidden_states 계산
      2) self.rotary_emb 로 position_embeddings 계산  ← mrope 전담
      3) 각 self.layers[i] 에 position_embeddings 를 전달

    → lm_model.forward() 가 mrope 를 내부 처리하므로
      레이어를 직접 호출할 필요 없이 hook 만으로 타이밍 측정 가능.
    ─────────────────────────────────────────────────────────────────

    Hook 동작:
      register_forward_pre_hook  → start_events[i].record()
      register_forward_hook      → end_events[i].record()
      torch.cuda.synchronize()   → 이후 elapsed_time() 로 정확한 ms 획득

    인자:
      lm_model     : vlm.language_model.model  (Qwen3VLTextModel)
      layers       : lm_model.layers  (List[Qwen3VLTextDecoderLayer])
      dummy_hidden : [batch, seq, d_model]  bfloat16  cuda
      n_warmup     : 측정 전 워밍업 횟수  (캐시·JIT 안정화)
      n_measure    : 측정 반복 횟수

    반환:
      {
        "layer_times"   : List[List[float]],  # [n_layers][n_measure] ms
        "working_kwargs": dict,               # 실제 사용된 forward kwargs (str repr)
        "n_success"     : int,                # 성공한 측정 횟수
        "error"         : str | None,         # 실패 시 에러 메시지
      }
    """
    device  = dummy_hidden.device
    seq_len = dummy_hidden.shape[1]
    n       = len(layers)

    # ── CUDA Event 생성 ──────────────────────────────────────────────────────
    # 단일 쌍을 run마다 재사용: record() 는 덮어쓰기, synchronize() 후 elapsed_time()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    layer_times  = [[] for _ in range(n)]

    # ── Hook 등록 ────────────────────────────────────────────────────────────
    handles = []

    def make_pre(idx: int):
        def pre_hook(module, args):
            start_events[idx].record()
        return pre_hook

    def make_post(idx: int):
        def post_hook(module, args, output):
            end_events[idx].record()
        return post_hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre(i)))
        handles.append(layer.register_forward_hook(make_post(i)))

    # ── forward kwargs 탐색 ──────────────────────────────────────────────────
    # mrope: position_ids [3, batch, seq] — text token 은 모두 0 (all-zeros OK)
    position_ids_3d = torch.zeros(3, 1, seq_len, dtype=torch.long, device=device)
    # 1D fallback: [batch, seq]
    position_ids_1d = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    # cache_position: [seq]
    cache_pos = torch.arange(seq_len, dtype=torch.long, device=device)

    forward_candidates = [
        # ── mrope 3D position_ids (Qwen3VLTextModel primary) ─────────────────
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
        {"inputs_embeds": dummy_hidden,
         "position_ids":  position_ids_3d,
         "use_cache":     False,
         "output_hidden_states": False,
         "output_attentions":    False},
        # ── position_ids 없음 (모델 내부 자동 생성 시도) ─────────────────────
        {"inputs_embeds": dummy_hidden,
         "use_cache":     False},
        {"inputs_embeds": dummy_hidden},
        # ── 1D position_ids (Qwen3 text-only / Qwen2VL fallback) ─────────────
        {"inputs_embeds": dummy_hidden,
         "position_ids":  position_ids_1d,
         "use_cache":     False},
        {"inputs_embeds": dummy_hidden,
         "position_ids":  position_ids_1d,
         "use_cache":     False,
         "cache_position": cache_pos},
    ]

    working_kwargs  = None
    discovery_errs  = []

    for kwargs in forward_candidates:
        try:
            with torch.no_grad():
                lm_model(**kwargs)
            torch.cuda.synchronize()
            working_kwargs = kwargs
            keys_str = list(kwargs.keys())
            print(f"      lm_model forward 성공: {keys_str}")
            break
        except Exception as exc:
            discovery_errs.append(
                f"  {list(kwargs.keys())}: {type(exc).__name__}: {exc}"
            )
            continue

    if working_kwargs is None:
        for h in handles:
            h.remove()
        err_msg = (
            f"lm_model.forward() 모든 조합 실패 "
            f"(lm_model type: {type(lm_model).__name__})\n"
            + "\n".join(discovery_errs)
        )
        print(f"\n[ERROR] {err_msg}")
        return {
            "layer_times":    [[] for _ in range(n)],
            "working_kwargs": {},
            "n_success":      0,
            "error":          err_msg,
        }

    # ── Warmup + Measurement ─────────────────────────────────────────────────
    n_success = 0
    error_msg = None
    try:
        print(f"      워밍업 중 ({n_warmup}회)...")
        for _ in range(n_warmup):
            with torch.no_grad():
                lm_model(**working_kwargs)
            torch.cuda.synchronize()

        print(f"      측정 중 ({n_measure}회)...")
        for _run in range(n_measure):
            with torch.no_grad():
                lm_model(**working_kwargs)
            torch.cuda.synchronize()

            # synchronize 후 elapsed_time 은 안전
            for i in range(n):
                t_ms = start_events[i].elapsed_time(end_events[i])
                layer_times[i].append(t_ms)
            n_success += 1

    except Exception as exc:
        error_msg = f"측정 중 예외: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"\n[ERROR] {error_msg}")
    finally:
        for h in handles:
            h.remove()

    # working_kwargs 에서 텐서를 str 로 변환 (JSON 직렬화용)
    wk_serialized = {
        k: (str(v.shape) if hasattr(v, "shape") else str(v))
        for k, v in working_kwargs.items()
    }

    return {
        "layer_times":    layer_times,
        "working_kwargs": wk_serialized,
        "n_success":      n_success,
        "error":          error_msg,
    }


# ── 더미 입력 생성 ────────────────────────────────────────────────────────────
print("\n[2/4] 더미 입력 생성")

BATCH_SIZE = 1
SEQ_LEN    = 1   # autoregressive decode step — 1 token 씩 생성

if model is not None:
    # d_model: vlm config 에서 자동 검출
    for path_str, fn in [
        ("vlm.config.text_config.hidden_size",
         lambda m: m.vlm.config.text_config.hidden_size),
        ("vlm.language_model.config.hidden_size",
         lambda m: m.vlm.language_model.config.hidden_size),
        ("vlm.model.config.hidden_size",
         lambda m: m.vlm.model.config.hidden_size),
    ]:
        try:
            D_MODEL = fn(model)
            print(f"      D_MODEL: {D_MODEL}  ({path_str})")
            break
        except AttributeError:
            continue
    else:
        D_MODEL = 4096
        print(f"      D_MODEL: {D_MODEL}  (기본값 — config 탐색 실패)")
else:
    D_MODEL = 4096
    print(f"      D_MODEL: {D_MODEL}  (모델 없음, 기본값)")

dummy_hidden = torch.randn(
    BATCH_SIZE, SEQ_LEN, D_MODEL,
    dtype=torch.bfloat16,
    device="cuda",
)
print(f"      shape: {list(dummy_hidden.shape)},  dtype: {dummy_hidden.dtype}")

# ── 레이어별 측정 ─────────────────────────────────────────────────────────────
print("\n[3/4] 레이어별 compute time 측정 (hook-based)")
print(f"      warmup={WARMUP_STEPS},  measure={MEASURE_STEPS}")

results     = {}
meas        = None

if model is not None:
    # ── 레이어 탐색 ───────────────────────────────────────────────────────────
    lm_model, layers = get_lm_model_and_layers(model)
    n_layers   = len(layers)
    layer_type = type(layers[0]).__name__
    lm_type    = type(lm_model).__name__

    print(f"\n      레이어 수  : {n_layers}")
    print(f"      레이어 타입: {layer_type}")
    print(f"      lm_model   : {lm_type}")

    # 첫 레이어 가중치 크기 점검
    w_sizes  = get_layer_weight_mb(layers[0])
    total_mb = sum(w_sizes.values())
    print(f"      L00 총 가중치: {total_mb:.1f} MB")
    for name, mb in sorted(w_sizes.items(), key=lambda x: -x[1])[:6]:
        print(f"        {name}: {mb:.1f} MB")
    print()

    # 기준값 출력
    mlp_prefetch_ms  = ms_prefetch_dram(MLP_WEIGHT_MB)
    attn_prefetch_ms = ms_prefetch_dram(ATTN_QO_WEIGHT_MB)
    print(f"      기준: MLP  {MLP_WEIGHT_MB:.0f} MB @ {DRAM_BW_GBs} GB/s"
          f"  →  {mlp_prefetch_ms:.3f} ms  (pipeline overlap 필요 t_compute)")
    print(f"            Attn {ATTN_QO_WEIGHT_MB:.0f} MB @ {DRAM_BW_GBs} GB/s"
          f"  →  {attn_prefetch_ms:.3f} ms")
    print()

    # ── hook-based 측정 ───────────────────────────────────────────────────────
    meas = measure_all_layers_via_hooks(
        lm_model, layers, dummy_hidden,
        n_warmup=WARMUP_STEPS, n_measure=MEASURE_STEPS,
    )

    if meas["error"]:
        print(f"\n[ERROR] 측정 실패:\n{meas['error']}")
    else:
        layer_times = meas["layer_times"]

        print(f"\n      사용된 kwargs : {meas['working_kwargs']}")
        print(f"      성공 횟수     : {meas['n_success']}/{MEASURE_STEPS}")
        print()

        # ── 결과 표 출력 ──────────────────────────────────────────────────────
        hdr = (f"{'Layer':>5}  {'mean_ms':>10}  {'std_ms':>7}  {'min_ms':>7}"
               f"  |  {'MLP pipeline':>14}  {'Attn pipeline':>15}")
        print(hdr)
        print("-" * len(hdr))

        for i, times in enumerate(layer_times):
            if not times:
                continue
            arr     = np.array(times)
            mean_ms = float(arr.mean())
            std_ms  = float(arr.std())
            min_ms  = float(arr.min())
            max_ms  = float(arr.max())

            mlp_flag  = "✓ overlap" if mean_ms > mlp_prefetch_ms  else "✗ bottleneck"
            attn_flag = "✓ overlap" if mean_ms > attn_prefetch_ms else "✗ bottleneck"

            print(
                f"  L{i:02d}   "
                f"  {mean_ms:8.3f} ms"
                f"  {std_ms:6.3f}"
                f"  {min_ms:6.3f}"
                f"  |  {mlp_flag:>14}  {attn_flag:>15}"
            )

            results[f"layer_{i:02d}"] = {
                "mean_ms": mean_ms,
                "std_ms":  std_ms,
                "min_ms":  min_ms,
                "max_ms":  max_ms,
                "all_ms":  times,
            }

        if results:
            all_means  = [v["mean_ms"] for v in results.values()]
            total_step = sum(all_means)
            total_65   = total_step * 65

            print("-" * len(hdr))
            print(f"\n  집계 ({len(all_means)} 레이어 합산):")
            print(f"    1 decode step      : {total_step:.2f} ms")
            print(f"    65 decode steps    : {total_65:.1f} ms"
                  f"  =  {total_65 / 1000:.3f} s")
            print()
            n_mlp_ok  = sum(1 for m in all_means if m > mlp_prefetch_ms)
            n_attn_ok = sum(1 for m in all_means if m > attn_prefetch_ms)
            print(f"  Pipeline overlap 가능 레이어:")
            print(f"    MLP  ({mlp_prefetch_ms:.3f} ms 기준): "
                  f"{n_mlp_ok}/{len(all_means)} 레이어")
            print(f"    Attn ({attn_prefetch_ms:.3f} ms 기준): "
                  f"{n_attn_ok}/{len(all_means)} 레이어")
            print()
            if n_mlp_ok == len(all_means):
                print("  → 전 레이어 MLP prefetch 완전 중첩 가능 ✅")
                print("    RT-Swap async pipeline 구현으로 이론 최대 speedup 달성 기대")
            elif n_mlp_ok > 0:
                print(f"  → 부분 중첩 가능 ({n_mlp_ok}/{len(all_means)}) — "
                      "중첩 가능한 레이어만 pipeline 적용 가능")
            else:
                print("  → t_compute < t_prefetch_DRAM — prefetch가 bottleneck")
                print("    L2 BW (1126 GB/s) 기준으로는 overlap 가능할 수 있음")
                l2_ok = sum(1 for m in all_means if m > ms_prefetch_l2(MLP_WEIGHT_MB))
                print(f"    L2 기준 ({ms_prefetch_l2(MLP_WEIGHT_MB):.3f} ms): "
                      f"{l2_ok}/{len(all_means)} 레이어")

else:
    print("  [모델 없음] 측정 건너뜀")


# ── 결과 저장 ─────────────────────────────────────────────────────────────────
print(f"\n[4/4] 결과 저장: {RESULT_FILE}")
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

summary = {
    "experiment":   "260524_layer_compute_profile",
    "version":      "v3-hook-based",
    "date":         "2026-05-26",
    "method":       "CUDA Event hooks via lm_model.forward()",
    "rationale":    (
        "Qwen3VLTextDecoderLayer requires position_embeddings=(cos,sin) "
        "from model-level rotary_emb (mrope). Individual layer calls impossible "
        "without it. lm_model.forward() handles mrope internally; "
        "per-layer hooks give clean per-layer timing."
    ),
    "hardware":     "Jetson AGX Thor (SM 11.0, 128GB unified)",
    "model":        MODEL_ID,
    "batch_size":   BATCH_SIZE,
    "seq_len":      SEQ_LEN,
    "warmup_steps": WARMUP_STEPS,
    "measure_steps": MEASURE_STEPS,
    "reference": {
        "mlp_weight_mb":         MLP_WEIGHT_MB,
        "attn_qo_weight_mb":     ATTN_QO_WEIGHT_MB,
        "mlp_prefetch_ms_dram":  ms_prefetch_dram(MLP_WEIGHT_MB),
        "attn_prefetch_ms_dram": ms_prefetch_dram(ATTN_QO_WEIGHT_MB),
        "mlp_prefetch_ms_l2":    ms_prefetch_l2(MLP_WEIGHT_MB),
        "l2_bw_gbs":             L2_BW_GBs,
        "dram_bw_gbs":           DRAM_BW_GBs,
        "pipeline_threshold_ms": ms_prefetch_dram(MLP_WEIGHT_MB),
    },
    "layers": results,
}

if meas is not None:
    summary["working_kwargs"] = meas.get("working_kwargs", {})
    summary["n_success_runs"] = meas.get("n_success", 0)
    if meas.get("error"):
        summary["error"] = meas["error"]

with open(RESULT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print(f"      저장 완료: {RESULT_FILE}")
print("\n완료!")
print("다음 실험: 260524_async_pipeline.py (pipeline PoC)")
