"""
profile_alpamayo.py  ·  v4.0
────────────────────────────────────────────────────────────────────────────────
Alpamayo 1.5  논문급 4계층 GPU + CPU 코어별 단계 분리 프로파일러

[v4.0 변경사항 — v3.0 대비]
  - CPUSampler 클래스 신규 추가:
      · psutil.cpu_percent(percpu=True) → 코어별(Core 0~11) 활용률 50ms 샘플링
      · 백그라운드 스레드 → 인퍼런스 시작/종료와 완벽 동기화
      · 단계별 마커(mark()) → VLM/Vision/Prefill/Flow 구간 CPU 활용률 분리
  - CPUProfile 데이터클래스 신규 추가
  - RunResult에 cpu_profile 필드 추가
  - raw_timings.json에 cpu_profile 포함
  - summary.json에 cpu_summary 포함
  - print_summary()에 CPU 코어별 활용률 테이블 추가

[계측 구조 — GPU]
sample_trajectories_from_data_with_vlm_rollout() 내부 (alpamayo1_5.py L214–L400):

    data = copy.deepcopy(data)                    ← L244 GPU 유휴 (~50ms)
    vlm_outputs = self.vlm.generate(...)          ← VLM 전체 (Vision+Prefill+Decode)
    prompt_cache = vlm_outputs.past_key_values
    ...                                           ← VLM→Flow 전환 Python 코드 (~15ms)
    waypoints = self.diffusion.sample(...)        ← Flow Matching ODE

4계층 GPU 패치:
  Layer 1.  model.vlm.generate()  monkey-patch
            → VLM 전체 GPU 시간 (ev_vlm_s ~ ev_vlm_e)
            → NVTX: "vlm_generate"
            → CPU mark: "vlm_start" / "vlm_end"

  Layer 2.  model.vlm  forward_pre/post hook  (pixel_values / past_kv 검사)
            → Prefill: 첫 번째 VLM forward  (ev_pre_s ~ ev_pre_e)
            → Decode : 이후 forward 누적     (ev_vlm_e - ev_pre_e)
            → NVTX: "llm_prefill", "llm_decode"
            → CPU mark: "prefill_start" / "prefill_end"

  Layer 3.  model.vlm.model.visual (또는 vlm.visual) forward hook
            → Vision Encoding (ev_vis_s ~ ev_vis_e)
            → NVTX: "vision_encoding"
            → CPU mark: "vision_start" / "vision_end"

  Layer 4.  model.diffusion.sample()  monkey-patch
            → Flow Matching ODE 직접 측정 (ev_action_s ~ ev_action_e)
            → NVTX: "flow_matching"
            → CPU mark: "flow_start" / "flow_end"

[계측 구조 — CPU]
CPUSampler 동작 원리:
  1. _profile_one_run() 진입 직후 sampler.start() → 샘플링 스레드 시작
  2. 각 GPU 훅/패치가 sampler.mark("phase_start"/"phase_end") 호출
  3. sample_trajectories 완료 후 sampler.stop() → 스레드 종료
  4. 결과: 전체 평균 + 단계별 코어 활용률 (by_phase)

사용법:
  python scripts/profiling/profile_alpamayo.py --warmup 3 --runs 8
  nsys profile ... python scripts/profiling/profile_alpamayo.py --warmup 3 --runs 8

출력:
  profiling_results/
  ├── raw_timings.json       ← GPU 타이밍 + CPU 코어별 활용률 (런별)
  ├── summary.json           ← 통계 (mean/std/p50/p95/p99) + CPU 요약
  ├── stage_breakdown.csv    ← matplotlib 입력용
  └── pytorch_trace.json     ← Chrome trace (--pytorch_profiler 옵션 시)
"""

import argparse
import contextlib
import json
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

# psutil: CPU 코어별 활용률 측정에 필수
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    print("[WARNING] psutil 미설치 → CPU 코어별 활용률 측정 불가. "
          "pip install psutil 후 재실행하세요.", file=sys.stderr)

# ─────────────────────────────────────────────────────────────────────────────
# 글로벌 설정
# ─────────────────────────────────────────────────────────────────────────────
WARMUP_RUNS        = 3
MEASURE_RUNS       = 8
OUTPUT_DIR         = Path("profiling_results")
CPU_SAMPLE_INTERVAL_MS = 50   # CPU 샘플링 주기 (ms). 낮을수록 정밀, 오버헤드 증가.


# ─────────────────────────────────────────────────────────────────────────────
# CPUSampler  —  CPU 코어별 활용률 측정
# ─────────────────────────────────────────────────────────────────────────────
class CPUSampler:
    """
    인퍼런스 실행 구간 내 CPU 코어별 활용률을 고해상도로 샘플링한다.

    설계 원칙:
      - 백그라운드 데몬 스레드 → 메인(추론) 스레드와 독립 실행
      - 인퍼런스 시작/종료와 완벽 동기화 (start() / stop() 호출 위치로 보장)
      - 단계별 마커(mark()) → VLM / Vision / Prefill / Flow 구간별 활용률 분리
      - psutil.cpu_percent(percpu=True) → 코어 0~N 각각의 % 반환

    Thor (Jetson AGX Thor) 코어 구성:
      - 총 12코어: Cortex-X4(고성능) × 4 + Cortex-A720(효율) × 8
      - logical=True 기준 12개 엔트리 반환

    사용 패턴:
        sampler = CPUSampler(interval_ms=50)
        sampler.start()                         # ← 인퍼런스 직전
        # ... 추론 실행 중 mark() 호출들 ...
        result = sampler.stop()                 # ← 인퍼런스 직후
        # result["overall"]["per_core_mean_pct"] → 코어별 평균 활용률
        # result["by_phase"]["vision_encoding"]  → Vision 단계 코어별 활용률
    """

    # 단계별 마커 쌍 정의 (start_name, end_name, phase_key)
    _PHASE_PAIRS = [
        ("vlm_start",     "vlm_end",     "vlm_total"),
        ("vision_start",  "vision_end",  "vision_encoding"),
        ("prefill_start", "prefill_end", "llm_prefill"),
        ("flow_start",    "flow_end",    "flow_matching"),
    ]

    def __init__(self, interval_ms: int = CPU_SAMPLE_INTERVAL_MS):
        if not _PSUTIL_OK:
            raise RuntimeError("psutil 미설치 — pip install psutil 후 재실행")

        self.interval_s    = interval_ms / 1000.0
        self._samples: list = []   # [(perf_counter, [core0%, core1%, ...])]
        self._markers: list = []   # [(perf_counter, "marker_name")]
        self._lock         = threading.Lock()
        self._stop_event   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._n_cores      = psutil.cpu_count(logical=True) or 1

    def start(self):
        """
        인퍼런스 직전 호출. 백그라운드 샘플링 스레드를 시작한다.

        첫 번째 psutil 호출(baseline 초기화)은 버리고,
        실제 샘플링은 두 번째 호출부터 시작한다.
        (psutil은 이전 호출 이후의 델타를 반환하므로 첫 호출은 항상 0 또는 노이즈)
        """
        with self._lock:
            self._samples.clear()
            self._markers.clear()
        self._stop_event.clear()
        # 첫 호출: baseline 초기화 (반환값 버림)
        psutil.cpu_percent(percpu=True)
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="CPUSampler",
            daemon=True,   # 메인 프로세스 종료 시 자동 소멸
        )
        self._thread.start()

    def mark(self, name: str):
        """
        단계 전환점을 기록한다. 훅/패치에서 호출.

        Thread-safe: lock 없이 append (GIL 보장).
        wall-clock 타임스탬프(perf_counter)를 함께 저장해
        나중에 해당 구간의 샘플을 정확히 추출한다.
        """
        self._markers.append((time.perf_counter(), name))

    def stop(self) -> dict:
        """
        인퍼런스 직후 호출. 스레드를 종료하고 분석 결과를 반환한다.

        Returns:
            dict with keys:
                "overall"     : 전체 인퍼런스 구간 통계
                "by_phase"    : 단계별 통계 (vision_encoding, llm_prefill, etc.)
                "raw_samples" : 원시 샘플 리스트 (시각화용)
                "markers"     : 마커 리스트 (시각화용)
                "available"   : True (정상), False (psutil 미설치)
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self._build_result()

    # ── 내부 메서드 ────────────────────────────────────────────────────────

    def _sample_loop(self):
        """백그라운드 스레드: CPU 활용률을 interval_s 마다 샘플링."""
        while not self._stop_event.is_set():
            t     = time.perf_counter()
            cores = psutil.cpu_percent(percpu=True)
            # psutil이 논리 코어 수보다 적게 반환하는 경우 패딩
            if len(cores) < self._n_cores:
                cores = cores + [0.0] * (self._n_cores - len(cores))
            self._samples.append((t, cores))
            self._stop_event.wait(timeout=self.interval_s)

    def _build_result(self) -> dict:
        """수집된 샘플과 마커로 분석 결과 딕셔너리를 생성한다."""
        if not self._samples:
            return {
                "available": False,
                "error": "샘플 없음 — psutil 오류 또는 추론 시간이 너무 짧음",
            }

        timestamps   = [s[0] for s in self._samples]
        core_matrix  = np.array([s[1] for s in self._samples], dtype=float)
        # shape: (N_samples, N_cores)

        t0          = timestamps[0]
        t_last      = timestamps[-1]
        duration_ms = (t_last - t0) * 1000.0

        # ── 전체 통계 ──
        overall = {
            "per_core_mean_pct" : core_matrix.mean(axis=0).tolist(),
            "per_core_max_pct"  : core_matrix.max(axis=0).tolist(),
            "per_core_p95_pct"  : np.percentile(core_matrix, 95, axis=0).tolist(),
            "all_core_mean_pct" : float(core_matrix.mean()),
            "all_core_max_pct"  : float(core_matrix.max()),
            "n_samples"         : len(self._samples),
            "duration_ms"       : round(duration_ms, 1),
            "interval_ms"       : self.interval_s * 1000,
            "n_cores"           : self._n_cores,
        }

        # ── 단계별 통계 ──
        marker_dict: Dict[str, float] = {name: t for t, name in self._markers}
        by_phase: Dict[str, dict] = {}

        for start_name, end_name, phase_key in self._PHASE_PAIRS:
            t_start = marker_dict.get(start_name)
            t_end   = marker_dict.get(end_name)
            if t_start is None or t_end is None:
                continue  # 해당 마커가 없으면 건너뜀

            # 구간 내 샘플 추출
            phase_samples = [
                cores
                for ts, cores in self._samples
                if t_start <= ts <= t_end
            ]

            if len(phase_samples) < 1:
                # 구간이 너무 짧아 샘플이 없을 경우 인접 샘플 1개 사용
                closest = min(
                    self._samples,
                    key=lambda x: abs(x[0] - (t_start + t_end) / 2)
                )
                phase_samples = [closest[1]]

            pm = np.array(phase_samples, dtype=float)
            by_phase[phase_key] = {
                "per_core_mean_pct" : pm.mean(axis=0).tolist(),
                "per_core_max_pct"  : pm.max(axis=0).tolist(),
                "all_core_mean_pct" : float(pm.mean()),
                "n_samples"         : len(phase_samples),
                "duration_ms"       : round((t_end - t_start) * 1000.0, 1),
            }

        # ── 원시 데이터 (시각화용) ──
        raw_samples = [
            {
                "t_ms"  : round((ts - t0) * 1000.0, 1),
                "cores" : [round(c, 1) for c in cores],
            }
            for ts, cores in self._samples
        ]
        markers_out = [
            {
                "t_ms" : round((t - t0) * 1000.0, 1),
                "name" : name,
            }
            for t, name in sorted(self._markers, key=lambda x: x[0])
        ]

        return {
            "available"   : True,
            "overall"     : overall,
            "by_phase"    : by_phase,
            "raw_samples" : raw_samples,
            "markers"     : markers_out,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StageTimings:
    """한 번의 추론에서 각 단계별 GPU 시간 (ms).

    측정 방법 (v4.0):
        vision_encoding      : Layer 3 hook (직접 측정)
        llm_prefill          : Layer 2 hook 첫 forward - vision (직접 측정)
        llm_decode           : t_vlm - t_prefill (유도)
        action_direct        : Layer 4 patch, diffusion.sample() (직접 측정)
        action_overhead      : t_total - t_vlm - action_direct
                               = L244 copy.deepcopy + VLM→Flow 전환 코드
        action_expert        : t_total - t_vlm (유도, backward compat)
        total_gpu            : CUDA Event 전체 타이머 (직접 측정)
        total_wall           : time.perf_counter (CPU 벽시계)
        cpu_overhead         : total_wall - total_gpu
    """
    vision_encoding       : float = 0.0
    llm_prefill           : float = 0.0
    llm_decode            : float = 0.0
    action_direct         : float = 0.0   # diffusion.sample() 직접 측정값
    action_overhead       : float = 0.0   # deepcopy L244 + 전환 코드
    action_expert         : float = 0.0   # t_total - t_vlm (legacy)
    total_gpu             : float = 0.0
    total_wall            : float = 0.0
    cpu_overhead          : float = 0.0
    decode_steps          : int   = 0     # CoC 자기회귀 스텝 수
    # 계측 품질 플래그
    vision_direct         : bool  = False
    prefill_direct        : bool  = False
    action_direct_measured: bool  = False


@dataclass
class CPUProfile:
    """한 번의 추론에서 CPU 코어별 활용률 요약.

    측정 방법:
        - psutil.cpu_percent(percpu=True) 50ms 주기 샘플링
        - 인퍼런스 시작 직전 ~ 직후 구간만 포함
        - per_core_mean_pct: Core 0~N 각각의 평균 활용률 (%)
        - by_phase: 단계별 CPU 활용률 (vision/prefill/decode/flow)
    """
    available            : bool        = False
    all_core_mean_pct    : float       = 0.0
    all_core_max_pct     : float       = 0.0
    per_core_mean_pct    : List[float] = field(default_factory=list)
    per_core_max_pct     : List[float] = field(default_factory=list)
    n_samples            : int         = 0
    n_cores              : int         = 0
    # 단계별 전체 코어 평균 (%)
    cpu_vision_pct       : float       = 0.0
    cpu_prefill_pct      : float       = 0.0
    cpu_flow_pct         : float       = 0.0
    cpu_vlm_pct          : float       = 0.0
    # 원시 데이터 (JSON 저장용)
    raw                  : dict        = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    peak_gpu_mb   : float = 0.0
    param_mem_mb  : float = 0.0
    activation_mb : float = 0.0   # peak - param


@dataclass
class RunResult:
    run_id      : int
    timings     : StageTimings
    memory      : MemorySnapshot
    cpu_profile : CPUProfile


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────
def _find_submodule(root: Any, dot_paths: list) -> Optional[Any]:
    """점(.) 구분 경로 목록에서 첫 번째로 존재하는 서브모듈을 반환."""
    for path in dot_paths:
        obj = root
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def _safe_elapsed(s: torch.cuda.Event, e: torch.cuda.Event) -> float:
    """두 CUDA Event 사이의 경과 시간(ms). 오류 시 0.0 반환."""
    try:
        if s is None or e is None:
            return 0.0
        return float(s.elapsed_time(e))
    except Exception:
        return 0.0


def _new_cuda_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


# ─────────────────────────────────────────────────────────────────────────────
# AlpamayoStagePatch  —  핵심 계측 클래스 (4계층 GPU + CPU 마커)
# ─────────────────────────────────────────────────────────────────────────────
class AlpamayoStagePatch:
    """
    Alpamayo1_5 모델에 4계층 패치를 적용해 단계별 GPU 타이밍을 측정하고,
    CPUSampler와 연동해 단계별 CPU 코어 활용률 마커를 기록한다.

    사용 패턴:
        patch = AlpamayoStagePatch()
        patch.attach(model)              # 워밍업 전 한 번만 호출
        try:
            for i in range(N):
                sampler = CPUSampler()
                patch.prepare_run(sampler)   # 매 추론 전 호출
                sampler.start()
                # ... sample_trajectories_from_data_with_vlm_rollout 호출 ...
                patch._inside_inference = False
                cpu_result = sampler.stop()
                timings = patch.compute_timings(t_total_ms)
        finally:
            patch.detach(model)
    """

    _VISUAL_PATHS = [
        "model.visual",
        "visual",
        "model.vision_model",
        "vision_model",
        "vision_tower",
        "model.vision_encoder",
        "vision_encoder",
    ]

    _DIFFUSION_ATTRS = [
        "diffusion",
        "action_expert",
        "flow_matching",
        "denoiser",
        "diffusion_model",
    ]

    _DIFFUSION_METHODS = ["sample", "forward"]

    def __init__(self):
        self._hooks: list = []
        self._orig_generate = None

        # Layer 4 복원 정보
        self._diffusion_module      = None
        self._diffusion_method_name = None
        self._orig_diffusion_sample = None

        # 런 간 상태 플래그
        self._inside_inference  = False
        self._inside_generate   = False
        self._prefill_done      = False
        self._decode_nvtx_live  = False
        self._decode_step_count = 0

        # CUDA Events (매 런 prepare_run()에서 초기화)
        self.ev_vlm_s    = self.ev_vlm_e    = None
        self.ev_vis_s    = self.ev_vis_e    = None
        self.ev_pre_s    = self.ev_pre_e    = None
        self.ev_action_s = self.ev_action_e = None

        # CPUSampler 참조 (매 런 prepare_run()에서 주입)
        self._sampler: Optional[CPUSampler] = None

        # 계측 품질 플래그 (attach() 결과)
        self.vis_hook_ok       = False
        self.vlm_hook_ok       = False
        self.diffusion_patched = False

    # ── 런별 초기화 ────────────────────────────────────────────────────────
    def prepare_run(self, sampler: Optional[CPUSampler] = None):
        """
        매 추론 직전 호출.

        Args:
            sampler: CPUSampler 인스턴스. 전달 시 단계별 마커를 자동 기록.
                     None이면 GPU 타이밍만 측정 (CPU 마커 없음).
        """
        self._inside_inference  = True
        self._inside_generate   = False
        self._prefill_done      = False
        self._decode_nvtx_live  = False
        self._decode_step_count = 0
        self._sampler           = sampler   # CPUSampler 참조 주입

        # VLM 이벤트 (Layer 1, 2)
        self.ev_vlm_s = _new_cuda_event()
        self.ev_vlm_e = _new_cuda_event()
        self.ev_vis_s = _new_cuda_event()
        self.ev_vis_e = _new_cuda_event()
        self.ev_pre_s = _new_cuda_event()
        self.ev_pre_e = _new_cuda_event()

        # Flow Matching 이벤트 (Layer 4)
        self.ev_action_s = _new_cuda_event()
        self.ev_action_e = _new_cuda_event()

    # ── 패치/훅 등록 ───────────────────────────────────────────────────────
    def attach(self, model) -> dict:
        """model에 4계층 패치와 훅을 등록한다. 워밍업 전 한 번만 호출."""
        info = {}

        if not hasattr(model, "vlm"):
            raise RuntimeError(
                "model.vlm 속성을 찾을 수 없습니다. "
                "Alpamayo1_5 인스턴스를 전달했는지 확인하세요."
            )

        _patch = self

        # ── Layer 1: vlm.generate() monkey-patch ──────────────────────────
        # GPU: ev_vlm_s ~ ev_vlm_e
        # CPU: mark("vlm_start") / mark("vlm_end")
        self._orig_generate = model.vlm.generate

        def _patched_generate(*args, **kwargs):
            _patch._inside_generate = True
            _patch.ev_vlm_s.record()
            nvtx.range_push("vlm_generate")
            # CPU 마커: VLM 시작
            if _patch._sampler is not None:
                _patch._sampler.mark("vlm_start")

            result = _patch._orig_generate(*args, **kwargs)

            # 마지막 decode step NVTX 닫기
            if _patch._decode_nvtx_live:
                nvtx.range_pop()  # "llm_decode"
                _patch._decode_nvtx_live = False

            _patch.ev_vlm_e.record()
            nvtx.range_pop()  # "vlm_generate"
            # CPU 마커: VLM 종료
            if _patch._sampler is not None:
                _patch._sampler.mark("vlm_end")
            _patch._inside_generate = False
            return result

        model.vlm.generate = _patched_generate
        info["generate_patched"] = True

        # ── Layer 3: Vision Encoder forward hook ──────────────────────────
        # GPU: ev_vis_s ~ ev_vis_e
        # CPU: mark("vision_start") / mark("vision_end")
        visual = _find_submodule(model.vlm, self._VISUAL_PATHS)
        if visual is not None:
            def _vis_pre(module, args, kwargs=None):
                if _patch._inside_generate:
                    _patch.ev_vis_s.record()
                    nvtx.range_push("vision_encoding")
                    if _patch._sampler is not None:
                        _patch._sampler.mark("vision_start")

            def _vis_post(module, args, output):
                if _patch._inside_generate:
                    _patch.ev_vis_e.record()
                    nvtx.range_pop()  # "vision_encoding"
                    if _patch._sampler is not None:
                        _patch._sampler.mark("vision_end")

            try:
                self._hooks.append(
                    visual.register_forward_pre_hook(_vis_pre, with_kwargs=True)
                )
            except TypeError:
                self._hooks.append(
                    visual.register_forward_pre_hook(lambda m, a: _vis_pre(m, a, {}))
                )
            self._hooks.append(visual.register_forward_hook(_vis_post))
            self.vis_hook_ok = True
            info["visual_path"] = next(
                p for p in self._VISUAL_PATHS
                if _find_submodule(model.vlm, [p]) is not None
            )
        else:
            info["visual_path"] = None

        # ── Layer 2: model.vlm forward hook (prefill / decode 구분) ──────
        # GPU: ev_pre_s ~ ev_pre_e (Prefill만 직접 측정)
        # CPU: mark("prefill_start") / mark("prefill_end")
        def _vlm_pre(module, args, kwargs=None):
            if not _patch._inside_generate:
                return
            kw           = kwargs or {}
            pixel_values = kw.get("pixel_values", None)
            past_kv      = kw.get("past_key_values", None)
            is_prefill   = (
                pixel_values is not None
                or past_kv is None
                or (hasattr(past_kv, "get_seq_length")
                    and past_kv.get_seq_length() == 0)
            )

            if is_prefill and not _patch._prefill_done:
                _patch.ev_pre_s.record()
                nvtx.range_push("llm_prefill")
                if _patch._sampler is not None:
                    _patch._sampler.mark("prefill_start")
            elif not is_prefill:
                if not _patch._decode_nvtx_live:
                    nvtx.range_push("llm_decode")
                    _patch._decode_nvtx_live = True
                _patch._decode_step_count += 1

        def _vlm_post(module, args, output):
            if not _patch._inside_generate:
                return
            if not _patch._prefill_done:
                _patch.ev_pre_e.record()
                nvtx.range_pop()  # "llm_prefill"
                if _patch._sampler is not None:
                    _patch._sampler.mark("prefill_end")
                _patch._prefill_done = True

        try:
            self._hooks.append(
                model.vlm.register_forward_pre_hook(_vlm_pre, with_kwargs=True)
            )
        except TypeError:
            self._hooks.append(
                model.vlm.register_forward_pre_hook(lambda m, a: _vlm_pre(m, a, {}))
            )
        self._hooks.append(model.vlm.register_forward_hook(_vlm_post))
        self.vlm_hook_ok = True
        info["vlm_hook"] = True

        # ── Layer 4: diffusion.sample() monkey-patch ──────────────────────
        # GPU: ev_action_s ~ ev_action_e
        # CPU: mark("flow_start") / mark("flow_end")
        diffusion_mod    = None
        diffusion_method = None

        for attr in self._DIFFUSION_ATTRS:
            mod = getattr(model, attr, None)
            if mod is not None:
                for mname in self._DIFFUSION_METHODS:
                    if hasattr(mod, mname):
                        diffusion_mod    = mod
                        diffusion_method = mname
                        break
            if diffusion_mod is not None:
                break

        if diffusion_mod is not None and diffusion_method is not None:
            self._diffusion_module      = diffusion_mod
            self._diffusion_method_name = diffusion_method
            self._orig_diffusion_sample = getattr(diffusion_mod, diffusion_method)

            orig_sample = self._orig_diffusion_sample

            def _patched_sample(*args, **kwargs):
                in_action = _patch._inside_inference and not _patch._inside_generate
                if in_action:
                    _patch.ev_action_s.record()
                    nvtx.range_push("flow_matching")
                    if _patch._sampler is not None:
                        _patch._sampler.mark("flow_start")

                result = orig_sample(*args, **kwargs)

                if in_action:
                    _patch.ev_action_e.record()
                    nvtx.range_pop()  # "flow_matching"
                    if _patch._sampler is not None:
                        _patch._sampler.mark("flow_end")
                return result

            setattr(diffusion_mod, diffusion_method, _patched_sample)
            self.diffusion_patched = True
            info["diffusion_patched"] = True
            info["diffusion_path"]    = f"model.{attr}.{diffusion_method}"
        else:
            info["diffusion_patched"] = False
            info["diffusion_path"]    = None

        # ── 등록 완료 리포트 ──────────────────────────────────────────────
        ok = "OK"
        ng = "NG"
        print("[Patch] ──────────────────────────────────────────")
        print(f"[Patch]  Layer 1 (vlm.generate)     : "
              f"{ok + ' 패치됨'    if info['generate_patched'] else ng + ' 실패'}")
        print(f"[Patch]  Layer 2 (vlm forward hook)  : "
              f"{ok + ' 등록됨'    if info['vlm_hook']          else ng + ' 실패'}")
        print(f"[Patch]  Layer 3 (vision encoder)    : "
              f"{ok + ' ' + str(info['visual_path'])  if info['visual_path'] else ng + ' 미발견 (비율 추정 사용)'}")
        print(f"[Patch]  Layer 4 (diffusion.sample)  : "
              f"{ok + ' ' + str(info['diffusion_path']) if info['diffusion_patched'] else ng + ' 미발견 (t_total-t_vlm 사용)'}")
        print(f"[Patch]  CPU 마커 (CPUSampler)        : "
              f"{ok + ' psutil ' + str(psutil.cpu_count(logical=True)) + '코어' if _PSUTIL_OK else ng + ' psutil 미설치'}")
        print("[Patch] ──────────────────────────────────────────")
        return info

    # ── 타이밍 계산 ────────────────────────────────────────────────────────
    def compute_timings(self, t_total_ms: float) -> dict:
        """기록된 CUDA 이벤트로부터 단계별 시간(ms)을 계산한다."""
        torch.cuda.synchronize()

        t_vlm_ms           = _safe_elapsed(self.ev_vlm_s,    self.ev_vlm_e)
        t_vision_ms        = _safe_elapsed(self.ev_vis_s,    self.ev_vis_e)
        t_prefill_ms       = _safe_elapsed(self.ev_pre_s,    self.ev_pre_e)
        t_action_direct_ms = _safe_elapsed(self.ev_action_s, self.ev_action_e)

        vision_direct          = (t_vision_ms        > 0.0)
        prefill_direct         = (t_prefill_ms       > 0.0)
        action_direct_measured = (t_action_direct_ms > 0.0)

        if not prefill_direct and t_vlm_ms > 0.0:
            t_prefill_ms = t_vlm_ms * 0.279
            t_vision_ms  = t_vlm_ms * 0.065

        t_llm_prefill_ms   = max(0.0, t_prefill_ms - t_vision_ms)
        t_decode_ms        = max(0.0, t_vlm_ms - t_prefill_ms)
        t_action_legacy_ms = max(0.0, t_total_ms - t_vlm_ms)

        if action_direct_measured:
            t_action_overhead_ms = max(0.0, t_action_legacy_ms - t_action_direct_ms)
        else:
            t_action_overhead_ms = 0.0
            t_action_direct_ms   = t_action_legacy_ms

        t_sum_stages = (
            t_vision_ms
            + t_llm_prefill_ms
            + t_decode_ms
            + t_action_direct_ms
            + t_action_overhead_ms
        )
        t_residual = t_total_ms - t_sum_stages

        return {
            "vision_ms"              : round(t_vision_ms,           3),
            "prefill_ms"             : round(t_llm_prefill_ms,      3),
            "decode_ms"              : round(t_decode_ms,           3),
            "action_direct_ms"       : round(t_action_direct_ms,    3),
            "action_overhead_ms"     : round(t_action_overhead_ms,  3),
            "action_legacy_ms"       : round(t_action_legacy_ms,    3),
            "vlm_ms"                 : round(t_vlm_ms,              3),
            "total_ms"               : round(t_total_ms,            3),
            "decode_steps"           : self._decode_step_count,
            "vision_direct"          : vision_direct,
            "prefill_direct"         : prefill_direct,
            "action_direct_measured" : action_direct_measured,
            "sum_stages_ms"          : round(t_sum_stages,          3),
            "residual_ms"            : round(t_residual,            3),
        }

    # ── 패치 해제 ──────────────────────────────────────────────────────────
    def detach(self, model):
        """등록된 모든 훅과 패치를 원복한다."""
        if self._orig_generate is not None:
            model.vlm.generate = self._orig_generate
            self._orig_generate = None

        if self._orig_diffusion_sample is not None and self._diffusion_module is not None:
            setattr(
                self._diffusion_module,
                self._diffusion_method_name,
                self._orig_diffusion_sample,
            )
            self._orig_diffusion_sample = None
            self._diffusion_module      = None

        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        self._inside_inference = False
        self._inside_generate  = False
        self._sampler          = None

        print("[Patch] 모든 훅/패치 해제 완료.")


# ─────────────────────────────────────────────────────────────────────────────
# AlpamayoProfiler
# ─────────────────────────────────────────────────────────────────────────────
class AlpamayoProfiler:
    def __init__(self, model_path: str, use_pytorch_profiler: bool = False):
        self.model_path           = model_path
        self.use_pytorch_profiler = use_pytorch_profiler
        self.model                = None
        self._model_inputs        = None
        self._param_mem_mb        = 0.0
        self.results: List[RunResult] = []

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    def load_model(self):
        """모델과 실제 추론 입력을 준비한다."""
        print("[Profiler] 모델 로드 중...")
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        from alpamayo1_5 import helper
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.model = Alpamayo1_5.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
        ).cuda().eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        self._param_mem_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / 1e6
        print(f"[Profiler] 파라미터: {n_params/1e9:.3f}B  "
              f"메모리: {self._param_mem_mb:.0f} MB")

        print("[Profiler] 입력 데이터 준비 중...")
        clip_id  = "030c760c-ae38-49aa-9ad8-f5650a545d26"
        data     = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        processor = helper.get_processor(self.model.tokenizer)
        inputs    = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        self._model_inputs = {
            "tokenized_data"  : inputs,
            "ego_history_xyz" : data["ego_history_xyz"],
            "ego_history_rot" : data["ego_history_rot"],
        }
        self._model_inputs = helper.to_device(self._model_inputs, "cuda")
        print("[Profiler] 입력 준비 완료.")

    # ── 단일 추론 프로파일링 ────────────────────────────────────────────────
    @torch.no_grad()
    def _profile_one_run(
        self,
        run_id : int,
        patch  : AlpamayoStagePatch,
    ) -> RunResult:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # ── CPUSampler 생성 ──
        # psutil이 없으면 None으로 대체 (GPU 측정은 정상 동작)
        sampler: Optional[CPUSampler] = None
        if _PSUTIL_OK:
            sampler = CPUSampler(interval_ms=CPU_SAMPLE_INTERVAL_MS)

        # ── v4.0: sampler를 prepare_run()에 주입 ──
        # patch 내부의 모든 훅/패치가 sampler.mark() 를 호출할 수 있게 됨
        patch.prepare_run(sampler=sampler)

        # 전체 타이머
        ev_run_s = _new_cuda_event()
        ev_run_e = _new_cuda_event()

        # ── CPUSampler 시작: 인퍼런스 직전 ──
        # 주의: sampler.start() → patch.prepare_run() 순서가 아니라
        #       prepare_run() → sampler.start() 순서여야 한다.
        #       prepare_run()이 sampler 참조를 patch에 주입하고,
        #       sampler.start()가 스레드를 시작하므로
        #       스레드 시작 시점이 실제 추론 직전임이 보장됨.
        if sampler is not None:
            sampler.start()

        wall_s = time.perf_counter()
        nvtx.range_push(f"alpamayo_full_inference")
        ev_run_s.record()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = \
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=self._model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    return_extra=True,
                )

        ev_run_e.record()
        patch._inside_inference = False
        torch.cuda.synchronize()
        nvtx.range_pop()  # "alpamayo_full_inference"

        wall_ms   = (time.perf_counter() - wall_s) * 1000.0

        # ── CPUSampler 종료: 인퍼런스 직후 ──
        cpu_raw: dict = {}
        if sampler is not None:
            cpu_raw = sampler.stop()

        total_gpu = _safe_elapsed(ev_run_s, ev_run_e)
        t = patch.compute_timings(total_gpu)

        if abs(t["residual_ms"]) > 10.0:
            print(f"  [경고] run {run_id}: 단계 합산 잔차 {t['residual_ms']:.1f}ms "
                  f"(|residual| > 10ms — 훅 누락 가능성)")

        # ── CPUProfile 구성 ──
        cpu_profile = CPUProfile()
        if cpu_raw.get("available", False):
            ov = cpu_raw["overall"]
            bp = cpu_raw.get("by_phase", {})
            cpu_profile = CPUProfile(
                available         = True,
                all_core_mean_pct = ov["all_core_mean_pct"],
                all_core_max_pct  = ov["all_core_max_pct"],
                per_core_mean_pct = ov["per_core_mean_pct"],
                per_core_max_pct  = ov["per_core_max_pct"],
                n_samples         = ov["n_samples"],
                n_cores           = ov["n_cores"],
                cpu_vision_pct    = bp.get("vision_encoding", {}).get("all_core_mean_pct", 0.0),
                cpu_prefill_pct   = bp.get("llm_prefill",     {}).get("all_core_mean_pct", 0.0),
                cpu_flow_pct      = bp.get("flow_matching",   {}).get("all_core_mean_pct", 0.0),
                cpu_vlm_pct       = bp.get("vlm_total",       {}).get("all_core_mean_pct", 0.0),
                raw               = cpu_raw,
            )

        timings = StageTimings(
            vision_encoding        = t["vision_ms"],
            llm_prefill            = t["prefill_ms"],
            llm_decode             = t["decode_ms"],
            action_direct          = t["action_direct_ms"],
            action_overhead        = t["action_overhead_ms"],
            action_expert          = t["action_legacy_ms"],
            total_gpu              = t["total_ms"],
            total_wall             = wall_ms,
            cpu_overhead           = wall_ms - total_gpu,
            decode_steps           = t["decode_steps"],
            vision_direct          = t["vision_direct"],
            prefill_direct         = t["prefill_direct"],
            action_direct_measured = t["action_direct_measured"],
        )

        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
        memory = MemorySnapshot(
            peak_gpu_mb   = peak_gpu_mb,
            param_mem_mb  = self._param_mem_mb,
            activation_mb = max(0.0, peak_gpu_mb - self._param_mem_mb),
        )

        return RunResult(
            run_id      = run_id,
            timings     = timings,
            memory      = memory,
            cpu_profile = cpu_profile,
        )

    # ── 메인 프로파일링 루프 ────────────────────────────────────────────────
    def run(self) -> dict:
        patch = AlpamayoStagePatch()
        patch.attach(self.model)

        # ── 워밍업 ──
        print(f"\n[Profiler] ── 워밍업 {WARMUP_RUNS}회 ──")
        for i in range(WARMUP_RUNS):
            r = self._profile_one_run(run_id=-(WARMUP_RUNS - i), patch=patch)
            vlm_sum = (r.timings.vision_encoding
                       + r.timings.llm_prefill
                       + r.timings.llm_decode)
            flag = "(직접)" if r.timings.action_direct_measured else "(유도)"
            print(f"  warmup {i+1}/{WARMUP_RUNS}  "
                  f"total={r.timings.total_gpu:.0f}ms  "
                  f"vlm={vlm_sum:.0f}ms  "
                  f"action={r.timings.action_direct:.0f}ms{flag}  "
                  f"cpu={r.cpu_profile.all_core_mean_pct:.1f}%")

        # ── PyTorch Profiler 설정 ──
        prof_ctx = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
            )
            if self.use_pytorch_profiler
            else contextlib.nullcontext()
        )

        # ── 측정 ──
        header = (
            f"  {'Run':>3} {'Total':>8} {'Vision':>8} {'Prefill':>8} "
            f"{'Decode':>8} {'Flow':>8} {'Steps':>6} "
            f"{'CPU(all)':>9} {'CPU(vlm)':>9} {'CPU(flow)':>10}"
        )
        print(f"\n[Profiler] ── 측정 {MEASURE_RUNS}회 ──")
        print(header)
        print("  " + "─" * (len(header) - 2))

        with prof_ctx as prof:
            for i in range(MEASURE_RUNS):
                nvtx.range_push(f"measure_run_{i}")
                r = self._profile_one_run(run_id=i, patch=patch)
                nvtx.range_pop()

                self.results.append(r)
                t  = r.timings
                cp = r.cpu_profile
                v_flag = "" if t.vision_direct  else "~"
                a_flag = "" if t.action_direct_measured else "~"

                cpu_all  = f"{cp.all_core_mean_pct:>7.1f}%" if cp.available else "     N/A"
                cpu_vlm  = f"{cp.cpu_vlm_pct:>7.1f}%"      if cp.available else "     N/A"
                cpu_flow = f"{cp.cpu_flow_pct:>7.1f}%"      if cp.available else "      N/A"

                print(
                    f"  {i+1:>3} "
                    f"{t.total_gpu:>7.0f}ms "
                    f"{v_flag}{t.vision_encoding:>5.0f}ms "
                    f"{v_flag}{t.llm_prefill:>5.0f}ms "
                    f"{t.llm_decode:>7.0f}ms "
                    f"{a_flag}{t.action_direct:>5.0f}ms "
                    f"{t.decode_steps:>6}  "
                    f"{cpu_all}  {cpu_vlm}  {cpu_flow}"
                )

                if self.use_pytorch_profiler and prof is not None:
                    prof.step()

        if self.use_pytorch_profiler and prof is not None:
            trace_path = OUTPUT_DIR / "pytorch_trace.json"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace_path))
            print(f"[Profiler] Chrome trace: {trace_path}")

        patch.detach(self.model)
        return self._compute_summary()

    # ── 통계 계산 ──────────────────────────────────────────────────────────
    def _compute_summary(self) -> dict:
        def _stats(vals: list) -> dict:
            a = np.array(vals, dtype=float)
            return {
                "mean"  : float(np.mean(a)),
                "std"   : float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                "min"   : float(np.min(a)),
                "p50"   : float(np.percentile(a, 50)),
                "p95"   : float(np.percentile(a, 95)),
                "p99"   : float(np.percentile(a, 99)),
                "max"   : float(np.max(a)),
                "cv_pct": float(np.std(a, ddof=1) / np.mean(a) * 100)
                          if (len(a) > 1 and np.mean(a) > 0) else 0.0,
            }

        stage_keys = [
            ("vision_encoding",  "Vision Encoding"),
            ("llm_prefill",      "LLM Prefill"),
            ("llm_decode",       "LLM Decode"),
            ("action_direct",    "Flow Matching (직접)"),
            ("action_overhead",  "Action Overhead (*)"),
            ("action_expert",    "Action Expert (legacy)"),
            ("total_gpu",        "Total GPU"),
            ("total_wall",       "Total Wall"),
            ("cpu_overhead",     "CPU Overhead"),
        ]

        timing_summary = {}
        for key, _ in stage_keys:
            vals = [getattr(r.timings, key) for r in self.results]
            timing_summary[key] = _stats(vals)

        total_mean    = timing_summary["total_gpu"]["mean"]
        breakdown_pct = {
            key: timing_summary[key]["mean"] / total_mean * 100
            for key, _ in stage_keys[:6]
        }

        decode_steps_list  = [r.timings.decode_steps for r in self.results]
        avg_decode_steps   = float(np.mean(decode_steps_list))
        per_step_decode_ms = (
            timing_summary["llm_decode"]["mean"] / avg_decode_steps
            if avg_decode_steps > 0 else 0.0
        )

        n_runs           = len(self.results)
        n_direct_vision  = sum(1 for r in self.results if r.timings.vision_direct)
        n_direct_prefill = sum(1 for r in self.results if r.timings.prefill_direct)
        n_direct_action  = sum(1 for r in self.results if r.timings.action_direct_measured)

        residual_vals = [
            r.timings.total_gpu
            - (r.timings.vision_encoding + r.timings.llm_prefill + r.timings.llm_decode
               + r.timings.action_direct + r.timings.action_overhead)
            for r in self.results
        ]

        # ── CPU 요약 ──
        cpu_results = [r.cpu_profile for r in self.results if r.cpu_profile.available]
        cpu_summary: dict = {}
        if cpu_results:
            n_cores = cpu_results[0].n_cores
            # 코어별 평균 활용률 (런 평균)
            per_core_means = np.array(
                [cp.per_core_mean_pct for cp in cpu_results], dtype=float
            )  # shape: (n_runs, n_cores)
            per_core_maxes = np.array(
                [cp.per_core_max_pct  for cp in cpu_results], dtype=float
            )

            # 단계별 전체 코어 평균
            phase_keys = ["cpu_vision_pct", "cpu_prefill_pct",
                          "cpu_flow_pct",   "cpu_vlm_pct"]
            phase_summary = {}
            for pk in phase_keys:
                vals = [getattr(cp, pk) for cp in cpu_results]
                phase_summary[pk] = _stats(vals)

            cpu_summary = {
                "n_runs"              : len(cpu_results),
                "n_cores"             : n_cores,
                "interval_ms"         : CPU_SAMPLE_INTERVAL_MS,
                "all_core_mean_pct"   : _stats([cp.all_core_mean_pct for cp in cpu_results]),
                "all_core_max_pct"    : _stats([cp.all_core_max_pct  for cp in cpu_results]),
                # 코어별 평균 (런 평균 → 코어별 평균)
                "per_core_mean_pct"   : per_core_means.mean(axis=0).tolist(),
                "per_core_max_pct"    : per_core_maxes.max(axis=0).tolist(),
                "per_core_p95_pct"    : np.percentile(per_core_means, 95, axis=0).tolist(),
                # 단계별
                "by_phase"            : phase_summary,
                "n_samples_per_run"   : _stats([cp.n_samples for cp in cpu_results]),
                "measurement_note"    : (
                    "psutil.cpu_percent(percpu=True) 50ms 샘플링. "
                    "인퍼런스 시작 직전 sampler.start() ~ 직후 sampler.stop() 구간만 포함. "
                    "by_phase는 GPU 훅 mark() 타임스탬프로 구간 분리."
                ),
            }
        else:
            cpu_summary = {
                "available": False,
                "reason": "psutil 미설치 또는 샘플 없음",
            }

        return {
            "environment"       : self._get_env_info(),
            "timing_ms"         : timing_summary,
            "memory_mb"         : {
                "peak_gpu"    : _stats([r.memory.peak_gpu_mb   for r in self.results]),
                "activation"  : _stats([r.memory.activation_mb for r in self.results]),
                "param_mem_mb": self._param_mem_mb,
            },
            "breakdown_pct"     : breakdown_pct,
            "decode_stats"      : {
                "avg_steps"   : avg_decode_steps,
                "per_step_ms" : per_step_decode_ms,
                "steps_list"  : decode_steps_list,
            },
            "cpu_summary"       : cpu_summary,
            "validation"        : {
                "residual_mean_ms" : float(np.mean(residual_vals)),
                "residual_max_ms"  : float(np.max(np.abs(residual_vals))),
                "note": (
                    "잔차(residual) = total_gpu - (vision+prefill+decode+action_direct+action_overhead). "
                    "이상적으로 |잔차| < 2ms."
                ),
            },
            "measurement_quality": {
                "n_runs"              : n_runs,
                "warmup_runs"         : WARMUP_RUNS,
                "direct_vision_pct"   : 100.0 * n_direct_vision  / n_runs,
                "direct_prefill_pct"  : 100.0 * n_direct_prefill / n_runs,
                "direct_action_pct"   : 100.0 * n_direct_action  / n_runs,
            },
            "latency_target"    : {
                "target_ms"      : 100.0,
                "current_mean_ms": total_mean,
                "overshoot_x"    : total_mean / 100.0,
                "met_pct"        : 100.0 * sum(
                    r.timings.total_gpu <= 100.0 for r in self.results
                ) / n_runs,
            },
        }

    # ── 환경 정보 ──────────────────────────────────────────────────────────
    @staticmethod
    def _get_env_info() -> dict:
        props = torch.cuda.get_device_properties(0)
        info = {
            "device"       : torch.cuda.get_device_name(0),
            "sm_version"   : f"SM {props.major}.{props.minor}",
            "total_mem_gb" : round(props.total_memory / 1e9, 1),
            "torch_version": torch.__version__,
            "cuda_version" : torch.version.cuda,
            "python"       : sys.version.split()[0],
            "dtype"        : "bfloat16",
            "attn_impl"    : "eager",
        }
        if _PSUTIL_OK:
            info["cpu_cores_logical"]  = psutil.cpu_count(logical=True)
            info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
            info["cpu_freq_mhz"]       = (
                round(psutil.cpu_freq().current, 0)
                if psutil.cpu_freq() else None
            )
        return info


# ─────────────────────────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────────────────────────
def save_results(profiler: AlpamayoProfiler, summary: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── raw_timings.json (GPU 타이밍 + CPU 코어별 활용률) ──
    raw = []
    for r in profiler.results:
        row = asdict(r.timings)
        row["run_id"]      = r.run_id
        row["peak_gpu_mb"] = r.memory.peak_gpu_mb
        row["activation_mb"] = r.memory.activation_mb

        # CPUProfile: raw_samples는 크기가 커서 별도 파일 저장
        cp = r.cpu_profile
        if cp.available:
            row["cpu"] = {
                "available"         : True,
                "all_core_mean_pct" : round(cp.all_core_mean_pct, 2),
                "all_core_max_pct"  : round(cp.all_core_max_pct,  2),
                "per_core_mean_pct" : [round(v, 2) for v in cp.per_core_mean_pct],
                "per_core_max_pct"  : [round(v, 2) for v in cp.per_core_max_pct],
                "n_samples"         : cp.n_samples,
                "n_cores"           : cp.n_cores,
                "by_phase"          : {
                    "vision_encoding" : round(cp.cpu_vision_pct,  2),
                    "llm_prefill"     : round(cp.cpu_prefill_pct, 2),
                    "flow_matching"   : round(cp.cpu_flow_pct,    2),
                    "vlm_total"       : round(cp.cpu_vlm_pct,     2),
                },
            }
        else:
            row["cpu"] = {"available": False}

        raw.append(row)

    raw_path = OUTPUT_DIR / "raw_timings.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    # ── cpu_raw_samples.json (원시 샘플 — 시각화용 별도 파일) ──
    cpu_raw_all = []
    for r in profiler.results:
        if r.cpu_profile.available and r.cpu_profile.raw:
            cpu_raw_all.append({
                "run_id"     : r.run_id,
                "raw_samples": r.cpu_profile.raw.get("raw_samples", []),
                "markers"    : r.cpu_profile.raw.get("markers", []),
            })
    if cpu_raw_all:
        cpu_raw_path = OUTPUT_DIR / "cpu_raw_samples.json"
        with open(cpu_raw_path, "w", encoding="utf-8") as f:
            json.dump(cpu_raw_all, f, indent=2, ensure_ascii=False)
        print(f"  {cpu_raw_path}  ← CPU 원시 샘플 (시각화용)")

    # ── summary.json ──
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── stage_breakdown.csv ──
    import csv
    csv_path = OUTPUT_DIR / "stage_breakdown.csv"
    stages = [
        ("vision_encoding", "Vision Encoding"),
        ("llm_prefill",     "LLM Prefill"),
        ("llm_decode",      "LLM Decode"),
        ("action_direct",   "Flow Matching (direct)"),
        ("action_overhead", "Action Overhead"),
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "label", "mean_ms", "std_ms",
                         "p50_ms", "p95_ms", "pct"])
        for key, label in stages:
            t   = summary["timing_ms"][key]
            pct = summary["breakdown_pct"].get(key, 0.0)
            writer.writerow([
                key, label,
                f"{t['mean']:.2f}", f"{t['std']:.2f}",
                f"{t['p50']:.2f}",  f"{t['p95']:.2f}",
                f"{pct:.1f}",
            ])

    # ── cpu_core_breakdown.csv (코어별 활용률 — 논문 테이블용) ──
    cpu_s = summary.get("cpu_summary", {})
    if cpu_s.get("per_core_mean_pct"):
        cpu_csv_path = OUTPUT_DIR / "cpu_core_breakdown.csv"
        with open(cpu_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "core_id", "mean_pct", "max_pct", "p95_pct",
                "phase_vision", "phase_prefill", "phase_flow"
            ])
            means = cpu_s["per_core_mean_pct"]
            maxes = cpu_s["per_core_max_pct"]
            p95s  = cpu_s.get("per_core_p95_pct", [0.0] * len(means))
            # 단계별 코어별 데이터는 by_phase에 없으므로 N/A
            for i, (m, x, p) in enumerate(zip(means, maxes, p95s)):
                writer.writerow([
                    f"Core {i:02d}",
                    f"{m:.2f}", f"{x:.2f}", f"{p:.2f}",
                    "N/A", "N/A", "N/A",
                ])
        print(f"  {cpu_csv_path}  ← CPU 코어별 활용률 (논문 테이블용)")

    print(f"\n[Profiler] 저장 완료:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(summary: dict):
    env = summary["environment"]
    t   = summary["timing_ms"]
    q   = summary["measurement_quality"]
    d   = summary["decode_stats"]
    lt  = summary["latency_target"]
    bp  = summary["breakdown_pct"]
    v   = summary["validation"]
    cs  = summary.get("cpu_summary", {})

    SEP = "=" * 72

    print(f"\n{SEP}")
    print("  Alpamayo 1.5  프로파일링 결과  v4.0  (GPU 4계층 + CPU 코어별)")
    print(SEP)
    print(f"  디바이스    : {env['device']} ({env['sm_version']})")
    print(f"  PyTorch     : {env['torch_version']}  CUDA {env['cuda_version']}")
    print(f"  dtype       : {env['dtype']},  attn: {env['attn_impl']}")
    if env.get("cpu_cores_logical"):
        print(f"  CPU 코어    : {env['cpu_cores_physical']}물리 / "
              f"{env['cpu_cores_logical']}논리  "
              f"({env.get('cpu_freq_mhz', 'N/A')} MHz)")
    print(f"  측정 런     : {q['n_runs']}회  (워밍업 {q['warmup_runs']}회 제외)")
    print()

    print("  [계측 품질]")
    ok_ng = lambda b: "OK" if b else "NG"
    print(f"    Vision   직접 측정률 : {q['direct_vision_pct']:.0f}%  "
          f"{ok_ng(q['direct_vision_pct'] == 100)}")
    print(f"    Prefill  직접 측정률 : {q['direct_prefill_pct']:.0f}%  "
          f"{ok_ng(q['direct_prefill_pct'] == 100)}")
    print(f"    Flow M.  직접 측정률 : {q['direct_action_pct']:.0f}%  "
          f"{ok_ng(q['direct_action_pct'] == 100)}")
    print(f"    단계 합산 잔차 (mean): {v['residual_mean_ms']:.2f}ms  "
          f"(max: {v['residual_max_ms']:.2f}ms)  "
          f"{ok_ng(v['residual_max_ms'] < 5)}")
    print(f"    CPU 샘플링          : "
          f"{cs.get('n_runs', 0)}런 / {CPU_SAMPLE_INTERVAL_MS}ms 주기  "
          f"{ok_ng(cs.get('n_runs', 0) > 0)}")
    print()

    # ── GPU 타이밍 테이블 ──
    col_fmt = f"  {{:<26}} {{:>9}} {{:>7}} {{:>8}} {{:>8}} {{:>8}}"
    print(col_fmt.format("단계 [GPU]", "mean(ms)", "+-std", "p50", "p95", "비율(%)"))
    print("  " + "-" * 68)

    for key, label in [
        ("vision_encoding", "Vision Encoding"),
        ("llm_prefill",     "LLM Prefill"),
        ("llm_decode",      "LLM Decode"),
    ]:
        s   = t[key]
        pct = bp.get(key, 0.0)
        print(col_fmt.format(
            label, f"{s['mean']:.1f}", f"{s['std']:.1f}",
            f"{s['p50']:.1f}", f"{s['p95']:.1f}", f"{pct:.1f}%"
        ))

    print("  " + "." * 68)
    for key, label in [
        ("action_direct",   "Flow Matching (직접)"),
        ("action_overhead", "  Action Overhead (*)"),
    ]:
        s   = t[key]
        pct = bp.get(key, 0.0)
        print(col_fmt.format(
            label, f"{s['mean']:.1f}", f"{s['std']:.1f}",
            f"{s['p50']:.1f}", f"{s['p95']:.1f}", f"{pct:.1f}%"
        ))

    print("  " + "-" * 68)
    sum_mean = sum(t[k]["mean"] for k in [
        "vision_encoding", "llm_prefill", "llm_decode",
        "action_direct", "action_overhead"
    ])
    print(col_fmt.format("Stage Sum", f"{sum_mean:.1f}", "",
                         "", "", f"{sum_mean/t['total_gpu']['mean']*100:.1f}%"))

    sg = t["total_gpu"]
    print(col_fmt.format("Total GPU", f"{sg['mean']:.1f}", f"{sg['std']:.1f}",
                         f"{sg['p50']:.1f}", f"{sg['p95']:.1f}", "100.0%"))

    sw = t["total_wall"]
    print(col_fmt.format("Total Wall (CPU포함)", f"{sw['mean']:.1f}", f"{sw['std']:.1f}",
                         "", "", ""))

    # ── CPU 코어별 활용률 테이블 ──
    if cs.get("per_core_mean_pct"):
        print()
        print(f"\n  [CPU 코어별 활용률 — psutil 실측, 인퍼런스 구간만]")
        print(f"  샘플링 주기: {cs['interval_ms']}ms  |  "
              f"코어 수: {cs['n_cores']}  |  "
              f"평균 샘플/런: {cs['n_samples_per_run']['mean']:.0f}개")
        print()
        n_cores   = cs["n_cores"]
        c_means   = cs["per_core_mean_pct"]
        c_maxes   = cs["per_core_max_pct"]
        c_p95s    = cs.get("per_core_p95_pct", [0.0] * n_cores)

        cpu_col = "  {:<10} {:>9} {:>9} {:>9}"
        print(cpu_col.format("Core", "mean(%)", "max(%)", "p95(%)"))
        print("  " + "-" * 40)
        for i in range(n_cores):
            bar_len = max(0, min(20, int(c_means[i] / 5)))
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"  Core {i:02d}    {c_means[i]:>7.1f}   "
                  f"{c_maxes[i]:>7.1f}   {c_p95s[i]:>7.1f}   [{bar}]")

        print()
        # 단계별 CPU 활용률
        by_phase_stats = cs.get("by_phase", {})
        if by_phase_stats:
            print(f"  [단계별 CPU 전체 코어 평균 활용률]")
            print(f"  {'단계':<22} {'mean(%)':>9} {'+-std':>7}")
            print("  " + "-" * 40)
            phase_labels = {
                "cpu_vision_pct"  : "Vision Encoding",
                "cpu_prefill_pct" : "LLM Prefill",
                "cpu_flow_pct"    : "Flow Matching",
                "cpu_vlm_pct"     : "VLM 전체 (V+P+D)",
            }
            for pk, label in phase_labels.items():
                ps = by_phase_stats.get(pk, {})
                if ps:
                    print(f"  {label:<22} {ps['mean']:>8.1f}%  {ps['std']:>6.1f}%")

        # 전체 인퍼런스 평균
        all_s = cs["all_core_mean_pct"]
        print()
        print(f"  전체 인퍼런스 평균: "
              f"{all_s['mean']:.1f}% ± {all_s['std']:.1f}%  "
              f"(p95: {all_s['p95']:.1f}%)")
        print(f"  (Thor 12코어 기준: 전체 사용 시 100%, "
              f"단일 스레드(Python GIL) 이론 최대 = 1/12 = 8.3%)")

    print()
    print(f"  (*) Action Overhead = t_total - t_vlm - t_flow_matching")
    print(f"      = alpamayo1_5.py L244 copy.deepcopy(data) + VLM to Flow 전환 코드")
    print()
    print(f"  Decode steps  : {d['avg_steps']:.1f}개/추론  "
          f"(per-step {d['per_step_ms']:.1f}ms)")
    print(f"  Peak GPU 메모리: {summary['memory_mb']['peak_gpu']['mean']:.0f} MB  "
          f"(활성화+KV: {summary['memory_mb']['activation']['mean']:.0f} MB)")
    print()
    print(f"  100ms 목표 달성: {lt['met_pct']:.0f}%  "
          f"(현재 {lt['current_mean_ms']:.0f}ms -> {lt['overshoot_x']:.1f}x 초과)")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global WARMUP_RUNS, MEASURE_RUNS, CPU_SAMPLE_INTERVAL_MS

    parser = argparse.ArgumentParser(
        description="Alpamayo 1.5 논문급 4계층 단계 분리 프로파일러 v4.0"
    )
    parser.add_argument(
        "--model", default="nvidia/Alpamayo-1.5-10B",
        help="모델 경로 또는 HuggingFace ID",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS,
        help="워밍업 횟수",
    )
    parser.add_argument(
        "--runs", type=int, default=MEASURE_RUNS,
        help="측정 횟수",
    )
    parser.add_argument(
        "--cpu_interval_ms", type=int, default=CPU_SAMPLE_INTERVAL_MS,
        help="CPU 샘플링 주기 (ms). 기본 50ms.",
    )
    parser.add_argument(
        "--pytorch_profiler", action="store_true",
        help="PyTorch Profiler + Chrome trace 활성화",
    )
    args = parser.parse_args()

    WARMUP_RUNS              = args.warmup
    MEASURE_RUNS             = args.runs
    CPU_SAMPLE_INTERVAL_MS   = args.cpu_interval_ms

    profiler = AlpamayoProfiler(
        model_path=args.model,
        use_pytorch_profiler=args.pytorch_profiler,
    )
    profiler.load_model()
    summary = profiler.run()

    print_summary(summary)
    save_results(profiler, summary)


if __name__ == "__main__":
    main()
