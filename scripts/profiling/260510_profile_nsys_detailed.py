"""
260510_profile_nsys_detailed.py  ·  v1.0
──────────────────────────────────────────────────────────────────────────────
Alpamayo 1.5  3계층 정밀 메모리 프로파일러

[측정 계층]
  Layer 1 — CUDA Events + memory_stats  (동기식 스냅샷, 항상 정확)
    · μs 단위 phase별 GPU 실행 시간
    · reset_peak_memory_stats() → phase 경계마다 peak 초기화 → 단계별 정확한 peak
    · torch.cuda.memory_stats()['active_bytes.all.peak'] → 각 phase의 실제 peak

  Layer 2 — Effective Bandwidth 계산  (BW-bound 여부 직접 증거)
    · BW_decode = model_weight_bytes × N_decode_tokens / decode_time_s
    · BW_flow   = action_expert_bytes × N_euler_steps / flow_time_s
    · 이론 최대(273 GB/s)와 비교 → BW-bound 확정 기준

  Layer 3 — torch.cuda.memory history  (allocation 이벤트 단위 추적)
    · _record_memory_history() → 모든 cudaMalloc 수준 이벤트 캡처
    · phase마다 memory_snapshot() 저장 → KV cache 할당 타이밍 정확히 추적
    · _dump_snapshot() → pickle → Windows에서 torch 시각화 또는 수동 분석

[nsys 연동]
  torch.cuda.nvtx.range_push/pop → nsys 타임라인에 phase 색상 표시
  이 스크립트를 nsys로 감싸면 kernel 단위 타임라인 + memory 이벤트 overlay 가능

[실행 방법]
  # 1) 단독 실행 (Layer 1 + 2 + 3, ~10분)
  python scripts/profiling/260510_profile_nsys_detailed.py

  # 2) nsys로 감싸기 (Layer 1+2+3 + kernel timeline, ~15분)
  nsys profile \\
      --trace=cuda,nvtx \\
      --cuda-memory-usage=true \\
      --sample=none \\
      --cpuctxsw=none \\
      --output=profiling_results/260510_memory_utilization/nsys_run \\
      python scripts/profiling/260510_profile_nsys_detailed.py --nsys-mode

[출력 파일]
  profiling_results/260510_memory_utilization/
    ├── phase_timing.json             ← phase별 GPU 시간 + memory peak + BW
    ├── memory_snapshots/
    │   ├── snap_before_vision.pickle
    │   ├── snap_after_vision.pickle
    │   ├── snap_after_prefill.pickle
    │   ├── snap_after_decode.pickle
    │   └── snap_after_flow.pickle
    └── figures/
        ├── fig_phase_timing.png      ← phase 시간 + BW 비교
        └── fig_memory_per_phase.png  ← phase별 peak 메모리 + delta
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 한글 폰트 (Linux: NanumGothic, 없으면 DejaVu) ────────────────────────────
for _f in ["NanumGothic", "NanumBarunGothic", "UnDotum", "DejaVu Sans"]:
    if _f in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
OUT     = Path("profiling_results/260510_memory_utilization")
SNAP_D  = OUT / "memory_snapshots"
FIG_D   = OUT / "figures"
for d in [OUT, SNAP_D, FIG_D]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ── Roofline 상수 ─────────────────────────────────────────────────────────────
DRAM_BW_GBps    = 273.0   # LPDDR5X 이론 최대 [공식]
MODEL_PARAMS_B  = 11.08   # 실측 파라미터 (B)
BYTES_PER_PARAM = 2       # bf16
MODEL_GB        = MODEL_PARAMS_B * BYTES_PER_PARAM  # 22.16 GB


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — PhaseProfiler
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    """단일 추론 phase의 정밀 측정 결과."""
    name:              str
    gpu_time_ms:       float   # CUDA Event 기준 GPU 실행 시간
    wall_time_ms:      float   # CPU wall-clock 시간 (참고용)
    mem_before_gb:     float   # phase 시작 시점 allocated bytes (GB)
    mem_peak_gb:       float   # phase 중 peak allocated bytes (GB)
    mem_after_gb:      float   # phase 종료 후 allocated bytes (GB)
    mem_delta_gb:      float   # peak - before (증가량)
    mem_retained_gb:   float   # after - before (phase 종료 후 남은 증가량)
    n_alloc_events:    int     # phase 중 새로운 alloc 횟수
    n_free_events:     int     # phase 중 free 횟수
    # Layer 2 — BW 계산 (해당 phase에서만 의미있는 경우)
    bw_theory_GBps:    float   = 0.0  # 이론 BW (bytes_read / time)
    bw_pct_of_peak:    float   = 0.0  # 이론 최대(273 GB/s) 대비 %
    n_steps:           int     = 0    # decode 토큰 수 또는 euler step 수
    bytes_read_theory: float   = 0.0  # 이론적으로 읽어야 하는 바이트 (GB)


class PhaseProfiler:
    """
    CUDA Events + torch.cuda.memory_stats()로 단계별 정밀 측정.

    사용법:
        prof = PhaseProfiler()
        with prof.phase("vision"):
            model.encode_images(...)
        result = prof.results["vision"]
    """

    def __init__(self, record_history: bool = True):
        self.results: dict[str, list[PhaseResult]] = {}
        self._record_history = record_history
        self._snap_idx = 0   # snapshot 순번

    class _PhaseCtx:
        """contextmanager 역할."""
        def __init__(self, profiler: "PhaseProfiler", name: str):
            self._p   = profiler
            self._name = name
            self._t_wall_start: float = 0.0
            self._ev_start = torch.cuda.Event(enable_timing=True)
            self._ev_end   = torch.cuda.Event(enable_timing=True)
            self._stats_before: dict = {}
            self._mem_before_gb: float = 0.0

        def __enter__(self):
            # memory snapshot (Layer 3) before phase
            self._p._save_snapshot(f"before_{self._name}")

            # NVTX marker (nsys 연동)
            torch.cuda.nvtx.range_push(self._name)

            # memory stat 기준선 + peak 초기화
            torch.cuda.synchronize()
            self._mem_before_gb = torch.cuda.memory_allocated() / 1e9
            stats_before = torch.cuda.memory_stats()
            self._n_alloc_before = stats_before.get("allocation.all.current", 0)
            self._n_free_before  = stats_before.get("inactive_split.all.current", 0)
            torch.cuda.reset_peak_memory_stats()

            # CUDA Event + wall clock 시작
            self._ev_start.record()
            self._t_wall_start = time.perf_counter()
            return self

        def __exit__(self, *_):
            # CUDA Event + wall clock 종료
            self._ev_end.record()
            t_wall_end = time.perf_counter()
            torch.cuda.synchronize()

            gpu_ms  = self._ev_start.elapsed_time(self._ev_end)
            wall_ms = (t_wall_end - self._t_wall_start) * 1000.0

            # memory 통계 수집
            stats = torch.cuda.memory_stats()
            mem_peak_gb  = stats.get("active_bytes.all.peak", 0) / 1e9
            mem_after_gb = torch.cuda.memory_allocated() / 1e9

            n_alloc = (stats.get("allocation.all.current", 0)
                       - self._n_alloc_before)
            n_free  = stats.get("free_retries.all.current", 0)

            # NVTX 종료
            torch.cuda.nvtx.range_pop()

            # memory snapshot after phase
            self._p._save_snapshot(f"after_{self._name}")

            result = PhaseResult(
                name            = self._name,
                gpu_time_ms     = gpu_ms,
                wall_time_ms    = wall_ms,
                mem_before_gb   = self._mem_before_gb,
                mem_peak_gb     = mem_peak_gb,
                mem_after_gb    = mem_after_gb,
                mem_delta_gb    = max(0.0, mem_peak_gb - self._mem_before_gb),
                mem_retained_gb = max(0.0, mem_after_gb - self._mem_before_gb),
                n_alloc_events  = max(0, n_alloc),
                n_free_events   = max(0, n_free),
            )
            self._p.results.setdefault(self._name, []).append(result)

    def phase(self, name: str) -> "_PhaseCtx":
        return self._PhaseCtx(self, name)

    def _save_snapshot(self, tag: str):
        """torch.cuda.memory_snapshot() 저장 (Layer 3)."""
        if not self._record_history:
            return
        try:
            snap = torch.cuda.memory_snapshot()
            idx  = self._snap_idx
            self._snap_idx += 1
            path = SNAP_D / f"{idx:02d}_{tag}.pickle"
            with open(path, "wb") as f:
                pickle.dump(snap, f)
        except Exception:
            pass   # 실패해도 계속 진행

    def add_bw(self, phase: str, bytes_read_gb: float, n_steps: int = 1):
        """
        Layer 2: 해당 phase의 effective BW를 계산해서 마지막 결과에 추가.
        반드시 phase context 종료 후 호출.
        """
        if phase not in self.results or not self.results[phase]:
            return
        r = self.results[phase][-1]
        if r.gpu_time_ms <= 0:
            return
        bw = bytes_read_gb / (r.gpu_time_ms / 1000.0)
        r.bw_theory_GBps    = bw
        r.bw_pct_of_peak    = bw / DRAM_BW_GBps * 100.0
        r.n_steps           = n_steps
        r.bytes_read_theory = bytes_read_gb

    def summary(self) -> dict:
        """전 phase 결과를 dict로 반환 (run별 평균)."""
        out = {}
        for name, runs in self.results.items():
            if not runs:
                continue
            out[name] = {
                "gpu_time_ms":       float(np.mean([r.gpu_time_ms     for r in runs])),
                "gpu_time_std_ms":   float(np.std( [r.gpu_time_ms     for r in runs])),
                "mem_peak_gb":       float(np.mean([r.mem_peak_gb     for r in runs])),
                "mem_delta_gb":      float(np.mean([r.mem_delta_gb    for r in runs])),
                "mem_retained_gb":   float(np.mean([r.mem_retained_gb for r in runs])),
                "bw_GBps":           float(np.mean([r.bw_theory_GBps for r in runs])),
                "bw_pct_of_peak":    float(np.mean([r.bw_pct_of_peak for r in runs])),
                "n_steps":           runs[0].n_steps,
                "n_runs":            len(runs),
            }
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 추론 실행 + 계측
# ══════════════════════════════════════════════════════════════════════════════

def run_profiling(warmup: int = 1, runs: int = 2, nsys_mode: bool = False):

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print("\n[1/4] 모델 로드 중...")
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t_load = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16,
    ).cuda().eval()
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t_load

    model_param_b  = sum(p.numel() for p in model.parameters()) / 1e9
    model_size_gb  = sum(p.numel() * p.element_size()
                         for p in model.parameters()) / 1e9
    model_mem_gb   = torch.cuda.memory_allocated() / 1e9

    print(f"      파라미터: {model_param_b:.3f}B  |  이론 크기: {model_size_gb:.2f} GB")
    print(f"      CUDA allocated: {model_mem_gb:.2f} GB  |  로드: {t_load:.1f}s")

    # ── 입력 준비 ─────────────────────────────────────────────────────────────
    print("[2/4] 입력 준비 중...")
    clip_id  = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data     = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    processor = helper.get_processor(model.tokenizer)
    inputs    = processor.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data"  : inputs,
        "ego_history_xyz" : data["ego_history_xyz"],
        "ego_history_rot" : data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    # ── 토큰 카운터 패치 ──────────────────────────────────────────────────────
    # generate()가 몇 토큰 생성했는지 정확히 세기 위한 훅
    _decode_token_count: list[int] = [0]

    orig_generate = model.vlm.generate.__func__

    def _counted_generate(self_vlm, *args, **kwargs):
        result = orig_generate(self_vlm, *args, **kwargs)
        # result: generated_ids tensor [B, seq_len]
        # 새로 생성된 토큰 수 = total_len - input_len
        if hasattr(result, "sequences"):
            gen_len = result.sequences.shape[-1]
        elif isinstance(result, torch.Tensor):
            gen_len = result.shape[-1]
        else:
            gen_len = 0
        input_len = inputs["input_ids"].shape[-1]
        _decode_token_count[0] += max(0, gen_len - input_len)
        return result

    import types
    model.vlm.generate = types.MethodType(_counted_generate, model.vlm)

    # Euler step 카운터
    # _euler()의 파라미터 목록은 버전마다 다를 수 있으므로 *args/**kwargs로 전달
    _euler_step_count: list[int] = [0]
    orig_euler = model.diffusion._euler  # bound method (self 이미 고정)

    def _counted_euler(self_diff, *args, **kwargs):
        # inference_step 키워드가 있으면 그 값, 없으면 num_inference_steps 사용
        n = kwargs.get("inference_step") or getattr(self_diff, "num_inference_steps", 1)
        _euler_step_count[0] += int(n)
        return orig_euler(*args, **kwargs)  # orig_euler는 이미 bound → self 불필요

    model.diffusion._euler = types.MethodType(_counted_euler, model.diffusion)

    # ── memory history 시작 (Layer 3) ────────────────────────────────────────
    print("[3/4] Memory history recording 시작...")
    if not nsys_mode:
        try:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
            _hist_ok = True
        except Exception as e:
            print(f"      [WARNING] memory history 미지원: {e}")
            _hist_ok = False
    else:
        _hist_ok = False

    # ── 프로파일링 루프 ───────────────────────────────────────────────────────
    print(f"[4/4] 프로파일링 (warmup {warmup}회 + 측정 {runs}회)")
    profiler = PhaseProfiler(record_history=(not nsys_mode))

    @torch.no_grad()
    def run_one(label: str):
        _decode_token_count[0] = 0
        _euler_step_count[0]   = 0

        # [Vision + Prefill + Decode + Flow] 전체 추론
        # torch.autocast 필수: diffusion _euler() 내부에서 torch.randn()으로
        # 생성된 초기 노이즈가 Float32이므로, BFloat16 가중치와 dtype 충돌 방지
        with profiler.phase("vlm_generate"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    return_extra=True,
                )

        n_tok = _decode_token_count[0]
        n_euler = _euler_step_count[0]

        # BW 계산 (Layer 2)
        # VLM generate = prefill(1회) + decode(N 토큰)
        # Decode BW: 모델 가중치를 토큰당 1회씩 읽음
        # Prefill BW: 입력 시퀀스를 한 번에 처리 (compute-heavy)
        # 보수적 추정: 전체 가중치 × (1 + N_decode) / 총 시간
        total_bytes_gb = MODEL_GB * max(1, n_tok)
        profiler.add_bw("vlm_generate",
                        bytes_read_gb=total_bytes_gb,
                        n_steps=n_tok)

        print(f"  [{label:12s}] "
              f"GPU: {profiler.results['vlm_generate'][-1].gpu_time_ms:6.0f} ms  "
              f"| mem_peak: {profiler.results['vlm_generate'][-1].mem_peak_gb:.2f} GB  "
              f"| KV delta: {profiler.results['vlm_generate'][-1].mem_delta_gb*1024:.0f} MB  "
              f"| n_tok: {n_tok}  "
              f"| BW: {profiler.results['vlm_generate'][-1].bw_theory_GBps:.1f} GB/s"
              f"  ({profiler.results['vlm_generate'][-1].bw_pct_of_peak:.0f}% of peak)")

    # warmup
    print("  -- warmup --")
    for i in range(warmup):
        run_one(f"warmup {i+1}")

    # 측정
    print("  -- measurement --")
    for i in range(runs):
        run_one(f"run {i+1}")

    # ── memory history dump ───────────────────────────────────────────────────
    if _hist_ok:
        snap_path = OUT / "memory_history_dump.pickle"
        try:
            torch.cuda.memory._dump_snapshot(str(snap_path))
            print(f"\n[Memory History] 저장: {snap_path}")
            print("  Windows에서 시각화:")
            print(f"    python -c \"import torch; torch.cuda.memory._visualize_memory_history('{snap_path}', 'memory_timeline.html')\"")

        except Exception as e:
            print(f"[Memory History] dump 실패: {e}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    return profiler, model_size_gb, model_param_b


# ══════════════════════════════════════════════════════════════════════════════
# 결과 출력 + 저장 + 그래프
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save(profiler: PhaseProfiler, model_size_gb: float):
    summary = profiler.summary()

    # ── 터미널 출력 ───────────────────────────────────────────────────────────
    W = 72
    print("\n" + "=" * W)
    print("  정밀 프로파일링 결과 요약")
    print("=" * W)
    print(f"  {'Phase':<18} {'GPU 시간(ms)':<14} {'mem_peak(GB)':<14}"
          f"{'KV delta(MB)':<14} {'BW(GB/s)':<12} {'BW%'}")
    print(f"  {'-'*18} {'-'*13} {'-'*13} {'-'*13} {'-'*11} {'-'*6}")
    for name, s in summary.items():
        bw_str  = f"{s['bw_GBps']:.1f}" if s['bw_GBps'] > 0 else "  N/A"
        bw_pstr = f"{s['bw_pct_of_peak']:.0f}%" if s['bw_GBps'] > 0 else ""
        print(f"  {name:<18} "
              f"{s['gpu_time_ms']:8.0f} ±{s['gpu_time_std_ms']:5.0f}   "
              f"{s['mem_peak_gb']:8.2f}       "
              f"{s['mem_delta_gb']*1024:8.0f}       "
              f"{bw_str:>8}     "
              f"{bw_pstr}")

    print()
    print("  [BW 해석]")
    print(f"  DRAM 이론 최대: {DRAM_BW_GBps:.0f} GB/s")
    for name, s in summary.items():
        if s["bw_GBps"] <= 0:
            continue
        bw = s["bw_GBps"]
        pct = s["bw_pct_of_peak"]
        if pct >= 70:
            verdict = "BW-bound 강함 (>=70%)"
        elif pct >= 40:
            verdict = "BW-bound 중간 (40~70%)"
        else:
            verdict = "compute-bound 또는 overhead-dominant (<40%)"
        print(f"  {name}: {bw:.1f} GB/s ({pct:.0f}%) → {verdict}")

    print("=" * W)

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    out_json = OUT / "phase_timing.json"
    with open(out_json, "w", encoding="utf-8") as f:
        # runs별 상세 데이터도 포함
        full = {
            "summary": summary,
            "model_size_gb": model_size_gb,
            "dram_bw_peak_GBps": DRAM_BW_GBps,
            "runs": {
                name: [asdict(r) for r in runs]
                for name, runs in profiler.results.items()
            }
        }
        json.dump(full, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {out_json}")

    _plot_results(summary, model_size_gb)


def _plot_results(summary: dict, model_size_gb: float):
    if not summary:
        return

    phases = list(summary.keys())
    gpu_ms  = [summary[p]["gpu_time_ms"]  for p in phases]
    gpu_std = [summary[p]["gpu_time_std_ms"] for p in phases]
    peaks   = [summary[p]["mem_peak_gb"]  for p in phases]
    deltas  = [summary[p]["mem_delta_gb"] * 1024 for p in phases]  # MB
    bws     = [summary[p]["bw_GBps"]      for p in phases]
    bw_pcts = [summary[p]["bw_pct_of_peak"] for p in phases]

    colors = {"vlm_generate": "#4878CF", "vision": "#4878CF",
              "prefill": "#6ACC65", "decode": "#D65F5F",
              "flow": "#B47CC7"}
    bar_colors = [colors.get(p, "#AAAAAA") for p in phases]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Alpamayo 1.5 — Phase-Level Profiling (Jetson AGX Thor)",
                 fontsize=13, fontweight="bold")

    # ── (a) GPU time ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("white")
    bars = ax.bar(phases, gpu_ms, color=bar_colors, alpha=0.85,
                  edgecolor="white", linewidth=1.3,
                  yerr=gpu_std, capsize=5, error_kw={"linewidth": 1.2})
    for bar, val in zip(bars, gpu_ms):
        ax.text(bar.get_x() + bar.get_width()/2, val + 30,
                f"{val:.0f} ms", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("GPU Time (ms)", fontsize=10)
    ax.set_title("(a) Phase Duration", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── (b) Memory peak + delta ───────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("white")
    x = np.arange(len(phases))
    ax.bar(x - 0.2, peaks, width=0.38, color=bar_colors, alpha=0.85,
           edgecolor="white", label="Peak allocated (GB)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, deltas, width=0.38, color=bar_colors, alpha=0.45,
            edgecolor="white", hatch="//", label="Delta vs before (MB)")
    ax.axhline(model_size_gb, color="gray", ls="--", lw=1.4, label="Model weights")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel("GPU Memory (GB)", fontsize=10)
    ax2.set_ylabel("Delta (MB)", fontsize=9, color="#555")
    ax.set_title("(b) Memory per Phase", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)

    # ── (c) Effective BW ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("white")
    valid = [(p, b, pct) for p, b, pct in zip(phases, bws, bw_pcts) if b > 0]
    if valid:
        vphases, vbws, vbw_pcts = zip(*valid)
        vcols = [colors.get(p, "#AAAAAA") for p in vphases]
        bars = ax.bar(vphases, vbws, color=vcols, alpha=0.85,
                      edgecolor="white", linewidth=1.3)
        for bar, val, pct in zip(bars, vbws, vbw_pcts):
            ax.text(bar.get_x() + bar.get_width()/2, val + 3,
                    f"{val:.0f}\n({pct:.0f}%)", ha="center", va="bottom",
                    fontsize=8.5)
        ax.axhline(DRAM_BW_GBps, color="#D65F5F", ls="--", lw=1.8,
                   label=f"Theoretical max\n{DRAM_BW_GBps:.0f} GB/s")
        # BW-bound 경계선 (70%)
        ax.axhline(DRAM_BW_GBps * 0.7, color="orange", ls=":", lw=1.2,
                   label="BW-bound threshold\n(70% of peak)")
        ax.set_ylabel("Effective BW (GB/s)", fontsize=10)
        ax.set_title("(c) Effective Memory Bandwidth", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "BW data N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_title("(c) Effective Memory Bandwidth", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.2)
    out = FIG_D / "fig_phase_detailed.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Fig] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 메모리 스냅샷 분석 유틸 (Phase 분리 방식 - 고급)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_snapshot(pickle_path: str):
    """
    저장된 memory snapshot pickle을 분석.
    각 segment의 allocated blocks 합산 → 실제 메모리 구성 출력.

    사용법 (Windows or Thor):
      python 260510_profile_nsys_detailed.py --analyze-snapshot memory_snapshots/03_after_prefill.pickle
    """
    path = Path(pickle_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, "rb") as f:
        snap = pickle.load(f)

    total_reserved = 0
    total_allocated = 0
    large_blocks = []

    for seg in snap:
        seg_size = seg.get("total_size", 0)
        total_reserved += seg_size
        for blk in seg.get("blocks", []):
            if blk.get("state") == "active_allocated":
                sz = blk.get("size", 0)
                total_allocated += sz
                if sz > 100 * 1024 * 1024:   # 100 MB 이상만 표시
                    large_blocks.append((sz / 1e9, blk))

    print(f"\n[Snapshot: {path.name}]")
    print(f"  Total reserved : {total_reserved/1e9:.3f} GB")
    print(f"  Total allocated: {total_allocated/1e9:.3f} GB")
    print(f"  Large blocks (>100 MB):")
    for sz_gb, blk in sorted(large_blocks, reverse=True)[:10]:
        frames = blk.get("frames", [])
        top = frames[0] if frames else {}
        print(f"    {sz_gb:.3f} GB  ← {top.get('name', '?')} "
              f"@ {top.get('filename', '?')}:{top.get('line', '?')}")


# ══════════════════════════════════════════════════════════════════════════════
# nsys 실행 가이드 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_nsys_guide():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  nsys 실행 가이드 (Thor에서)                                              ║
╠══════════════════════════════════════════════════════════════════════════╣

  [Step 1] nsys 버전 확인
    nsys --version

  [Step 2] 경량 프로파일링 (kernel timeline + memory, 권장)
    nsys profile \\
        --trace=cuda,nvtx \\
        --cuda-memory-usage=true \\
        --sample=none \\
        --cpuctxsw=none \\
        --output=profiling_results/260510_memory_utilization/nsys_run \\
        python scripts/profiling/260510_profile_nsys_detailed.py \\
            --warmup 0 --runs 1 --nsys-mode

  [Step 3] 결과 Windows로 전송
    # WSL에서:
    scp ice401@100.95.177.101:~/alpamayo1.5/profiling_results/260510_memory_utilization/nsys_run.nsys-rep \\
        /mnt/c/Users/nanay/Desktop/Alphamayo/profiling_results/260510_memory_utilization/

  [Step 4] Nsight Systems GUI에서 열기
    - Windows에 Nsight Systems 설치: developer.nvidia.com/nsight-systems
    - nsys_run.nsys-rep 파일 열기
    - Timeline에서 NVTX 레인(phase 색상)과 CUDA 커널을 함께 확인

╠══════════════════════════════════════════════════════════════════════════╣
║  memory_history.pickle 시각화 (Windows에서)                               ║
╠══════════════════════════════════════════════════════════════════════════╣

  # PyTorch가 설치된 Windows Python에서:
  import pickle, torch
  # torch >= 2.1 필요
  torch.cuda.memory._visualize_memory_history(
      "memory_history_dump.pickle",
      "memory_timeline.html"
  )
  # memory_timeline.html을 브라우저로 열기
  # → 각 allocation의 stack trace + 시간축 시각화

╠══════════════════════════════════════════════════════════════════════════╣
║  snapshot 분석 (Thor에서)                                                 ║
╠══════════════════════════════════════════════════════════════════════════╣

  python scripts/profiling/260510_profile_nsys_detailed.py \\
      --analyze-snapshot \\
      profiling_results/260510_memory_utilization/memory_snapshots/03_after_prefill.pickle

╚══════════════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Alpamayo 1.5 Phase-Level Memory Profiler (nsys-ready)"
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs",   type=int, default=2)
    ap.add_argument("--nsys-mode", action="store_true",
                    help="nsys로 감쌀 때 사용. memory history 비활성화 (nsys가 대신 수집)")
    ap.add_argument("--analyze-snapshot", type=str, default=None,
                    metavar="PICKLE",
                    help="저장된 snapshot pickle 분석 후 종료")
    ap.add_argument("--guide", action="store_true",
                    help="nsys 실행 가이드 출력 후 종료")
    args = ap.parse_args()

    if args.guide:
        print_nsys_guide()
        return

    if args.analyze_snapshot:
        analyze_snapshot(args.analyze_snapshot)
        return

    profiler, model_size_gb, model_param_b = run_profiling(
        warmup=args.warmup,
        runs=args.runs,
        nsys_mode=args.nsys_mode,
    )
    print_and_save(profiler, model_size_gb)


if __name__ == "__main__":
    main()
