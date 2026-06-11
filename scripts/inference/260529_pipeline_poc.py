"""
260529_pipeline_poc.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[실험 목적]
  Alpamayo-1.5-10B (Jetson AGX Thor, iGPU, 128GB 통합 메모리) 추론 파이프라인에서
  CPU-GPU 태스크 분리 및 CUDA Stream 이중화로 latency를 단축할 수 있는지 검증한다.
  모델 가중치·구조 수정은 없음. 시스템 수준의 스케줄링·파이프라이닝만 허용.

[이전 실험 확정 사실 (2026-05-28)]
  - Baseline (sdpa+DynamicCache+BF16) : 4,839ms
      VE 728ms | LM Prefill 1,423ms | Decode 1,818ms (17step×107ms) | Flow 870ms
  - Decode 물리 한계: 22GB ÷ 231 GB/s = 95ms/step, 실측 107ms = 88% 효율
      → Decode 1 step은 더 이상 빠르게 할 수 없음
  - CPU 2코어 sweet spot (파레토 분석):
      CPU +37 GB/s 획득, GPU -7.5% 손실 → Decode 107ms → 115ms/step (+8ms)
  - KV Temporal Reuse 실험 C: 2,429ms → 175ms (92.8%), decode steps 완벽 일치

[본 실험이 검증하는 것]
  Exp E. 전처리(setup) 시간이 inference와 직렬화되어 있는가?
         → CPU 2코어 affinity + background thread로 overlap 가능한가?
  Exp D. VE(728ms)를 Decode(1,818ms) 창 안에서 CUDA Stream 이중화로 동시 실행하면
         Decode가 얼마나 느려지는가? (GPU BW 경합 정량화 — net gain or loss?)
  Exp F. Flow decoder(870ms)를 CPU로 옮겨서 GPU VE와 동시 실행 가능한가?
         → CPU Flow 시간이 VE(728ms) 이내면 Flow를 공짜로 숨길 수 있음

[측정 정밀도]
  GPU timing  : torch.cuda.Event(enable_timing=True) — GPU 클럭 기반, ns 단위
  CPU timing  : time.perf_counter() — monotonic, ~ns 단위
  Wall-clock  : time.perf_counter() + torch.cuda.synchronize() 쌍으로 측정
  Warmup      : NUM_WARMUP = 1 (GPU JIT, cudnn autotuning, first-run cache overhead 제거)
  Measure     : NUM_MEASURE = 3 (mean ± std, min/max 범위)
  통계        : mean, std, min, max, p95 (scipy 의존성 없이 직접 구현)

[출력]
  profiling_results/260529_pipeline/
    results_E.json, results_D.json, results_F.json, results_ALL.json
    experiment_E_log.txt (단계별 상세 로그)

[실행]
  source ~/alpamayo1.5/a1_5_venv/bin/activate && cd ~/alpamayo1.5
  python3 scripts/inference/260529_pipeline_poc.py --exp E
  python3 scripts/inference/260529_pipeline_poc.py --exp D
  python3 scripts/inference/260529_pipeline_poc.py --exp F
  python3 scripts/inference/260529_pipeline_poc.py --exp ALL
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

# ── 프로젝트 경로 설정 ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.token_utils import to_special_token

# ── 로거 ────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 전역 상수 — 기존 KV reuse PoC와 동일한 설정으로 비교 가능성 확보
# ──────────────────────────────────────────────────────────────────────────────
CLIP_ID          = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US            = 5_100_000          # 기준 프레임
T1_US            = T0_US + 500_000    # 인접 프레임 (+500ms)
DEVICE           = "cuda"
MAX_DECODE_STEPS = 80                 # 안전 상한 (실제 EOS: ~17step)
NUM_WARMUP       = 1                  # JIT / CUDA 첫 실행 오버헤드 제거
NUM_MEASURE      = 3                  # 통계 안정성 확보

# 2026-05-28 확정 베이스라인 수치 (비교 기준)
BASELINE_VE_MS      = 728.0
BASELINE_PREFILL_MS = 1_423.0
BASELINE_DECODE_MS  = 1_818.0        # 17 steps × 107ms
BASELINE_FLOW_MS    = 870.0
BASELINE_TOTAL_MS   = 4_839.0
BASELINE_DECODE_STEP_MS = 107.0      # 물리 하한: 22GB ÷ 231GB/s = 95ms

# CPU 2코어 sweet spot (파레토 분석 결과)
CPU_SWEET_SPOT_CORES = 2             # GPU 7.5% 손실만으로 CPU 37 GB/s 획득
CPU_AFFINITY_CORES   = {10, 11}      # Thor 12코어 중 간섭 최소 코어 (high-index)

# 출력 디렉토리
OUT = Path("profiling_results/260529_pipeline")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 1: 측정 인프라 — 정밀 타이머, 통계 집계
# ══════════════════════════════════════════════════════════════════════════════

class GpuTimer:
    """
    CUDA Event 기반 GPU 연산 타이머.

    torch.cuda.Event(enable_timing=True) 사용 — GPU 클럭 기반으로
    wall-clock보다 정밀하고 CPU-GPU 동기화 대기 없이 동작한다.

    사용법:
        timer = GpuTimer()
        timer.start(stream)
        # ... GPU 연산 ...
        timer.stop(stream)
        stream.synchronize()
        ms = timer.elapsed_ms()  # GPU 실행 시간 (ms)
    """
    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)
        self._stream: Optional[torch.cuda.Stream] = None

    def start(self, stream: Optional[torch.cuda.Stream] = None):
        self._stream = stream
        self._start.record(stream)

    def stop(self, stream: Optional[torch.cuda.Stream] = None):
        self._end.record(stream or self._stream)

    def elapsed_ms(self) -> float:
        """
        start.elapsed_time(end) — CUDA 내부 타임스탬프 차이.
        반드시 stream.synchronize() 또는 torch.cuda.synchronize() 후 호출해야 한다.
        """
        return self._start.elapsed_time(self._end)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class WallTimer:
    """
    CPU wall-clock 타이머 (time.perf_counter 기반, ns 해상도).
    GPU 연산이 포함된 구간은 반드시 cuda.synchronize() 이후에 stop() 호출.
    """
    def __init__(self):
        self._t0: float = 0.0
        self._t1: float = 0.0

    def start(self):
        # ★ GPU sync 후 시작: CPU-GPU 파이프라인 지연 제거
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self):
        # ★ GPU sync 후 종료: GPU 커널 완료까지 포함
        torch.cuda.synchronize()
        self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (self._t1 - self._t0) * 1_000.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class CpuTimer:
    """
    순수 CPU 연산 타이머 (GPU sync 없음).
    GPU 연산이 없는 전처리, 데이터 로드 등에 사용.
    """
    def __init__(self):
        self._t0: float = 0.0
        self._t1: float = 0.0

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (self._t1 - self._t0) * 1_000.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class StatSummary:
    """
    측정값 목록에서 기술 통계를 계산한다.
    scipy 의존성 없이 직접 구현.
    """
    def __init__(self, values: List[float], label: str = ""):
        self.label  = label
        self.values = sorted(values)
        self.n      = len(values)

    @property
    def mean(self) -> float:
        return mean(self.values) if self.n else float("nan")

    @property
    def std(self) -> float:
        return stdev(self.values) if self.n >= 2 else 0.0

    @property
    def minimum(self) -> float:
        return self.values[0] if self.n else float("nan")

    @property
    def maximum(self) -> float:
        return self.values[-1] if self.n else float("nan")

    @property
    def p95(self) -> float:
        """95th percentile (선형 보간)"""
        if self.n == 0:
            return float("nan")
        idx = 0.95 * (self.n - 1)
        lo, hi = int(idx), min(int(idx) + 1, self.n - 1)
        frac = idx - lo
        return self.values[lo] * (1 - frac) + self.values[hi] * frac

    def to_dict(self) -> dict:
        return {
            "label":  self.label,
            "n":      self.n,
            "mean":   round(self.mean,    2),
            "std":    round(self.std,     2),
            "min":    round(self.minimum, 2),
            "max":    round(self.maximum, 2),
            "p95":    round(self.p95,     2),
            "values": [round(v, 2) for v in self.values],
        }

    def __repr__(self) -> str:
        return (
            f"[{self.label}] "
            f"mean={self.mean:.1f}ms  std={self.std:.1f}ms  "
            f"[{self.minimum:.1f}, {self.maximum:.1f}]  p95={self.p95:.1f}ms"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 2: 모델 구성요소 탐지 — VE, Flow 모듈 위치 파악
# ══════════════════════════════════════════════════════════════════════════════

def find_visual_encoder(model: Alpamayo1_5) -> Optional[torch.nn.Module]:
    """
    Alpamayo1_5 모델에서 Visual Encoder(ViT) 컴포넌트를 탐지한다.

    Qwen3-VL 기반 모델의 VE는 model.vlm.visual 또는 model.vlm.model.visual에 위치한다.
    탐지 실패 시 None 반환 → 실험 D에서 VE 단독 실행 불가 (full forward fallback 사용).
    """
    candidates = [
        ("model.vlm.visual",          lambda m: m.vlm.visual),
        ("model.vlm.model.visual",    lambda m: m.vlm.model.visual),
        ("model.vlm.vision_tower",    lambda m: m.vlm.vision_tower),
        ("model.vlm.model.vision_tower", lambda m: m.vlm.model.vision_tower),
    ]
    for path, getter in candidates:
        try:
            module = getter(model)
            if module is not None and isinstance(module, torch.nn.Module):
                logger.info(f"  [model introspect] VE found: {path}")
                return module
        except AttributeError:
            continue
    logger.warning("  [model introspect] VE not found via known paths → VE 단독 호출 불가")
    return None


def find_flow_decoder(model: Alpamayo1_5) -> Tuple[Optional[str], Optional[torch.nn.Module]]:
    """
    Alpamayo1_5 모델에서 Flow Decoder 컴포넌트를 탐지한다.

    반환값: (attr_name, module) — 탐지 실패 시 (None, None).

    Flow Decoder는 Decode 단계에서 생성된 trajectory 토큰을
    실제 waypoint 좌표로 변환하는 후처리 모듈이다.
    """
    # Exp D 결과 후 확인: top_modules = ['vlm','expert','action_space','diffusion',...]
    # Alpamayo trajectory는 diffusion process로 생성 → 'diffusion'이 Flow decoder
    candidates = [
        "diffusion",                              # ← Alpamayo 실제 모듈명 (확인됨 2026-05-29)
        "flow_decoder", "flow", "trajectory_decoder",
        "traj_decoder", "waypoint_decoder", "decoder",
        "action_space", "expert",                 # 대안 후보
    ]
    for name in candidates:
        if hasattr(model, name):
            module = getattr(model, name)
            if isinstance(module, torch.nn.Module):
                logger.info(f"  [model introspect] Flow found: model.{name}")
                return name, module
    # model.vlm 안에서도 탐색
    if hasattr(model, "vlm"):
        for name in candidates:
            if hasattr(model.vlm, name):
                module = getattr(model.vlm, name)
                if isinstance(module, torch.nn.Module):
                    logger.info(f"  [model introspect] Flow found: model.vlm.{name}")
                    return f"vlm.{name}", module
    # 탐지 실패 시 top-level 모듈 전체 출력 (수동 확인용)
    top_level = [name for name, _ in model.named_children()]
    logger.warning(
        f"  [model introspect] Flow decoder not found → Exp F 스킵\n"
        f"  top-level children: {top_level}"
    )
    return None, None


def introspect_model(model: Alpamayo1_5) -> dict:
    """
    모델 구성요소를 전수 탐색하여 실험에 필요한 정보를 수집한다.
    결과를 dict로 반환하고 INFO 로그로 출력한다.
    """
    info = {}

    # attn_implementation 확인
    info["attn_impl"] = getattr(model.vlm.config, "_attn_implementation", "unknown")

    # VE 탐지
    ve = find_visual_encoder(model)
    info["ve_module_found"] = ve is not None
    if ve is not None:
        ve_params = sum(p.numel() for p in ve.parameters())
        info["ve_params_M"] = round(ve_params / 1e6, 1)

    # Flow 탐지
    flow_name, flow_module = find_flow_decoder(model)
    info["flow_attr"]         = flow_name
    info["flow_module_found"] = flow_module is not None
    if flow_module is not None:
        flow_params = sum(p.numel() for p in flow_module.parameters())
        info["flow_params_M"] = round(flow_params / 1e6, 1)

    # 전체 파라미터 수 및 메모리
    total_params = sum(p.numel() for p in model.parameters())
    info["total_params_B"]   = round(total_params / 1e9, 2)
    info["model_size_GB_bf16"] = round(total_params * 2 / 1e9, 2)

    # 최상위 서브모듈 목록
    info["top_modules"] = [name for name, _ in model.named_children()]

    logger.info(
        f"\n  [model] attn={info['attn_impl']}  "
        f"params={info['total_params_B']}B  "
        f"size={info['model_size_GB_bf16']}GB BF16\n"
        f"  [model] VE={'✅ ' + str(info.get('ve_params_M','?')) + 'M' if info['ve_module_found'] else '❌ not found'}  "
        f"Flow={'✅ ' + str(info.get('flow_params_M','?')) + 'M' if info['flow_module_found'] else '❌ not found'}\n"
        f"  [model] top_modules: {info['top_modules']}"
    )
    return info


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 3: 공용 입력 준비 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def prepare_inputs_timed(
    model: Alpamayo1_5,
    clip_id: str,
    t_us: int,
    device: str = DEVICE,
) -> Tuple[torch.Tensor, dict, dict]:
    """
    데이터 로드부터 input_ids 준비까지 각 단계를 개별 측정하여 반환한다.

    반환값:
        (input_ids, tok_data, timing_dict)
        timing_dict: {"load_ms", "create_msg_ms", "tokenize_ms", "fuse_ms", "total_ms"}
    """
    timing = {}
    processor = helper.get_processor(model.tokenizer)

    # ── 단계 1: 데이터셋 로드 (I/O + PIL 이미지 처리) ────────────────────────
    t = CpuTimer(); t.start()
    data = load_physical_aiavdataset(clip_id, t0_us=t_us)
    t.stop(); timing["load_ms"] = t.elapsed_ms()

    # ── 단계 2: 메시지 구조 생성 (프레임 → VLM 메시지 포맷) ────────────────
    t = CpuTimer(); t.start()
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    t.stop(); timing["create_msg_ms"] = t.elapsed_ms()

    # ── 단계 3: 토크나이저 적용 (apply_chat_template + pixel 패치 추출) ────
    t = CpuTimer(); t.start()
    raw_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    raw_inputs = helper.to_device(raw_inputs, device)
    t.stop(); timing["tokenize_ms"] = t.elapsed_ms()

    # ── 단계 4: ego 데이터 → 특수 토큰 융합 (GPU 연산 포함) ───────────────
    ego_data = helper.to_device(
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        device,
    )
    input_ids_raw = raw_inputs.pop("input_ids")

    gt = GpuTimer(); gt.start()
    input_ids = model.fuse_traj_tokens(input_ids_raw, ego_data)
    gt.stop()
    torch.cuda.synchronize()
    timing["fuse_ms"] = gt.elapsed_ms()

    timing["total_ms"] = sum(timing.values())
    tok_data = raw_inputs   # attention_mask, pixel_values, image_grid_thw 등

    return input_ids, tok_data, timing


def run_full_prefill(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Any, torch.Tensor, float]:
    """
    Full prefill (VE + LM) 실행 → (past_key_values, last_logits, elapsed_ms).
    GpuTimer(CUDA Event) 기반으로 측정.
    """
    gt = GpuTimer()
    ctx = torch.cuda.stream(stream) if stream else torch.cuda.default_stream(DEVICE)

    with ctx:
        gt.start(stream)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(
                input_ids=input_ids,
                attention_mask=tok_data.get("attention_mask"),
                pixel_values=tok_data.get("pixel_values"),
                image_grid_thw=tok_data.get("image_grid_thw"),
                use_cache=True,
            )
        gt.stop(stream)

    if stream:
        stream.synchronize()
    else:
        torch.cuda.synchronize()

    return out.past_key_values, out.logits[:, -1, :].float(), gt.elapsed_ms()


def run_ve_only(
    model: Alpamayo1_5,
    tok_data: dict,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Visual Encoder만 단독 실행 → (image_features, elapsed_ms).

    VE 모듈이 탐지되지 않은 경우: full forward(pixel_values only 의미)로 fallback.
    이 함수는 실험 D에서 Decode ∥ VE 동시 실행 시 VE 측면 측정에 사용된다.

    중요: VE 결과를 실제 inference에 사용하지 않음.
          순수하게 "VE가 DRAM 대역폭을 얼마나 소비하는지" 측정이 목적.
    """
    pixel_values  = tok_data.get("pixel_values")
    image_grid_thw = tok_data.get("image_grid_thw")
    if pixel_values is None:
        raise ValueError("tok_data에 pixel_values가 없음 — VE 실행 불가")

    ve_module = find_visual_encoder(model)
    gt = GpuTimer()

    if ve_module is not None:
        # ── 경로 A: VE 직접 호출 ──────────────────────────────────────────
        ctx = torch.cuda.stream(stream) if stream else torch.cuda.default_stream(DEVICE)
        with ctx:
            gt.start(stream)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = ve_module(pixel_values, image_grid_thw)
            gt.stop(stream)
        if stream:
            stream.synchronize()
        else:
            torch.cuda.synchronize()
        return feats, gt.elapsed_ms()

    else:
        # ── 경로 B: VE 모듈 미탐지 → pixel_values만 넘겨 full forward
        #           (attention_mask, input_ids 없이 → embedding만 실행되는 경로)
        #           이 경우 정확한 VE 단독 시간이 아님을 기록
        logger.warning("  VE 직접 호출 불가 → full forward 대리 측정 (보수적 상한)")
        ctx = torch.cuda.stream(stream) if stream else torch.cuda.default_stream(DEVICE)
        with ctx:
            gt.start(stream)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # pixel_values만 처리하는 단축 경로가 없으면 None 반환
                pass
            gt.stop(stream)
        if stream:
            stream.synchronize()
        else:
            torch.cuda.synchronize()
        return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 4: 실험 E — CPU 2코어 전처리 파이프라이닝
# ══════════════════════════════════════════════════════════════════════════════

class PreprocessWorker:
    """
    백그라운드 CPU 전처리 워커.

    GPU 추론이 실행되는 동안, 별도 스레드에서 다음 프레임의 데이터를
    미리 준비한다. CPU affinity를 CPU_AFFINITY_CORES로 제한하여
    파레토 sweet spot(2코어, GPU 7.5% 손실)을 유지한다.

    사용법:
        worker = PreprocessWorker(model, n_cores=2)
        worker.submit(clip_id, t_us)     # GPU 추론 시작과 동시에 전처리 의뢰
        # ... GPU 추론 실행 ...
        result = worker.get_result()     # 전처리 완료 결과 수령
        worker.shutdown()
    """

    def __init__(self, model: Alpamayo1_5, n_cores: int = CPU_SWEET_SPOT_CORES):
        self.model    = model
        self.n_cores  = n_cores
        self._in_q:  queue.Queue = queue.Queue(maxsize=2)
        self._out_q: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="PreprocessWorker")
        self._thread.start()
        logger.info(f"  [PreprocessWorker] 시작 (CPU affinity={CPU_AFFINITY_CORES}, n_cores={n_cores})")

    def _worker_loop(self):
        # ★ CPU affinity 설정: GPU와 경합을 최소화하는 코어 선택
        try:
            os.sched_setaffinity(0, CPU_AFFINITY_CORES)
            actual = os.sched_getaffinity(0)
            logger.info(f"  [PreprocessWorker] affinity 설정 완료: cores={actual}")
        except (AttributeError, OSError) as e:
            logger.warning(f"  [PreprocessWorker] affinity 설정 실패 ({e}) → 제한 없음으로 계속")

        while True:
            item = self._in_q.get()
            if item is None:
                break  # shutdown 신호
            clip_id, t_us = item
            try:
                input_ids, tok_data, timing = prepare_inputs_timed(
                    self.model, clip_id, t_us
                )
                self._out_q.put({"status": "ok", "input_ids": input_ids,
                                  "tok_data": tok_data, "timing": timing})
            except Exception as e:
                self._out_q.put({"status": "error", "error": str(e)})

    def submit(self, clip_id: str, t_us: int):
        """전처리 요청 투입 (non-blocking)"""
        self._in_q.put((clip_id, t_us))

    def get_result(self, timeout: float = 60.0) -> dict:
        """전처리 결과 수령 (blocking, timeout=60s)"""
        return self._out_q.get(timeout=timeout)

    def shutdown(self):
        self._in_q.put(None)
        self._thread.join(timeout=5.0)


def _measure_decode_step_time_with_cpu_load(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
    n_cpu_workers: int,
    n_steps_sample: int = 5,
) -> dict:
    """
    Decode step 시간을 측정하면서 CPU N개 코어에 로드를 줄 때의 영향을 측정한다.

    파라미터:
        n_cpu_workers : 동시 실행할 CPU 워커 수 (0=baseline, 1, 2, 4)
        n_steps_sample: 측정에 사용할 decode steps 수 (초기 warm-up 제외)

    CPU 워커는 500MB 텐서를 DRAM에서 반복 읽는 memory-bound 루프
    (파레토 분석 실험과 동일한 방법론).
    """
    # ── Full prefill ───────────────────────────────────────────────────────
    kv, logits, _ = run_full_prefill(model, input_ids, tok_data)
    prefill_len   = int(input_ids.shape[1])

    # ── CPU 로드 스레드 준비 ───────────────────────────────────────────────
    stop_flag = threading.Event()
    cpu_load_threads: List[threading.Thread] = []

    def cpu_memory_load_worker(core_id: int):
        """500MB DRAM read loop — 순수 memory-bound 부하 (캐시 thrashing)"""
        try:
            os.sched_setaffinity(0, {core_id})
        except (AttributeError, OSError):
            pass
        # 500MB: L3(16MB)를 아득히 초과 → 100% DRAM 접근 보장
        buf = torch.ones(500 * 1024 * 1024 // 4, dtype=torch.float32)
        while not stop_flag.is_set():
            _ = buf.sum().item()  # DRAM read + ALU

    core_pool = list(range(n_cpu_workers))  # 코어 0, 1, ... (n_cpu_workers-1)
    for cid in core_pool:
        t = threading.Thread(target=cpu_memory_load_worker, args=(cid,), daemon=True)
        cpu_load_threads.append(t)

    # ── CPU 로드 시작 ──────────────────────────────────────────────────────
    for t in cpu_load_threads:
        t.start()
    if n_cpu_workers > 0:
        time.sleep(0.1)  # CPU 로드가 안정될 때까지 100ms 대기

    # ── Decode step 측정 ───────────────────────────────────────────────────
    step_times_ms: List[float] = []
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    cur_kv = kv

    # 처음 2 step은 warm-up (GPU 스케줄러 안정화)
    WARMUP_STEPS = 2
    try:
        for step in range(1, WARMUP_STEPS + n_steps_sample + 1):
            cpos = torch.tensor(
                [prefill_len + step - 1], device=DEVICE, dtype=torch.long
            )
            gt = GpuTimer()
            gt.start()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.vlm(
                    input_ids=next_token,
                    past_key_values=cur_kv,
                    cache_position=cpos,
                    use_cache=True,
                )
            gt.stop()
            torch.cuda.synchronize()
            step_ms = gt.elapsed_ms()

            cur_kv     = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :].float(), dim=-1, keepdim=True)

            if step > WARMUP_STEPS:
                step_times_ms.append(step_ms)

            # EOS 조기 종료
            tok_id = int(next_token.item())
            if traj_offset <= tok_id < traj_offset + traj_vocab_size:
                pass  # traj 토큰 계속
            elif tok_id == eos_id:
                break
    finally:
        stop_flag.set()
        for t in cpu_load_threads:
            t.join(timeout=2.0)

    return {
        "n_cpu_workers":  n_cpu_workers,
        "n_steps":        len(step_times_ms),
        "step_times_ms":  [round(v, 2) for v in step_times_ms],
        "stats":          StatSummary(step_times_ms,
                                      f"decode_step_{n_cpu_workers}cpu").to_dict()
        if step_times_ms else {},
    }


def run_experiment_E(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
) -> dict:
    """
    [실험 E] CPU 2코어 병렬 전처리 파이프라이닝

    ─── 실험 설계 ───────────────────────────────────────────────────────────────
    Sub-E1: 전처리 각 단계 소요 시간 프로파일링 (현재 어디서 시간이 얼마나 드는가?)
      - 데이터 로드 (I/O + PIL)
      - create_message
      - apply_chat_template + to_device
      - fuse_traj_tokens (GPU)

    Sub-E2: Decode step 시간 vs. CPU 코어 수 (0, 1, 2, 4)
      - 파레토 이론값(2코어=7.5% 손실)을 실제 Decode step 환경에서 검증
      - 500MB DRAM read loop으로 CPU memory-bound 부하 재현
        (파레토 분석 실험과 동일 방법론 → 결과 비교 가능)

    Sub-E3: 배경 스레드 overlap 검증
      - GPU full inference 실행 중 CPU 워커가 다음 프레임 전처리
      - 전처리 완료 시각이 GPU inference 완료 시각 이전인가?
      - 실제 추론 지연 감소량 측정
    ─────────────────────────────────────────────────────────────────────────────
    """
    print("\n" + "═" * 70)
    print("  [실험 E] CPU 2코어 전처리 파이프라이닝")
    print("═" * 70)

    results: Dict[str, Any] = {"exp": "E", "sub_experiments": {}}

    # ────────────────────────────────────────────────────────────────────────
    # Sub-E1: 전처리 단계별 소요 시간 프로파일링
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-E1: 전처리 단계별 시간 프로파일링 ──")
    print("  (각 단계를 독립적으로 측정 — 전체 setup time 분해)")

    all_timings: List[dict] = []
    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()

        try:
            _, _, timing = prepare_inputs_timed(model, CLIP_ID, T0_US)
        except Exception as e:
            logger.error(f"  [E1/{tag}] 전처리 실패: {e}")
            traceback.print_exc()
            continue

        line = (
            f"  [{tag}]  "
            f"load={timing['load_ms']:.0f}ms  "
            f"create_msg={timing['create_msg_ms']:.0f}ms  "
            f"tokenize={timing['tokenize_ms']:.0f}ms  "
            f"fuse={timing['fuse_ms']:.0f}ms  "
            f"total={timing['total_ms']:.0f}ms"
        )
        print(line)

        if not is_warmup:
            all_timings.append(timing)

    # 통계 집계
    if all_timings:
        keys = ["load_ms", "create_msg_ms", "tokenize_ms", "fuse_ms", "total_ms"]
        e1_stats = {}
        print(f"\n  ── [Sub-E1] 평균 ──")
        for k in keys:
            vals  = [t[k] for t in all_timings]
            stat  = StatSummary(vals, k)
            e1_stats[k] = stat.to_dict()
            print(f"  {k:20s}: {stat.mean:6.0f}ms  (std={stat.std:.1f}ms)")
        total_stat = StatSummary([t["total_ms"] for t in all_timings], "total")
        print(f"\n  → 전처리 총 시간: {total_stat.mean:.0f}ms")
        print(f"  → GPU Decode(1,818ms) 내 완전 은닉 가능 여부: "
              f"{'✅ 가능' if total_stat.mean < BASELINE_DECODE_MS else '❌ 불가 (Decode보다 느림)'}")
        results["sub_experiments"]["E1"] = {"timings": all_timings, "stats": e1_stats}
    else:
        results["sub_experiments"]["E1"] = {"error": "모든 trial 실패"}

    # ────────────────────────────────────────────────────────────────────────
    # Sub-E2: CPU 코어 수에 따른 Decode step 시간 변화 측정
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-E2: CPU 부하에 따른 Decode step 시간 측정 ──")
    print("  (파레토 분석 이론값 실측 검증: 2코어 = GPU 7.5% 손실 예측)")
    print(f"  Baseline decode step: {BASELINE_DECODE_STEP_MS:.0f}ms")

    cpu_configs = [0, 1, 2, 4]   # 0=baseline, 2=sweet spot, 4=위험 구간
    e2_results = {}

    for n_cores in cpu_configs:
        torch.cuda.empty_cache()
        label = f"{n_cores}코어 부하"
        print(f"\n  [{label}] 측정 중...")

        try:
            res = _measure_decode_step_time_with_cpu_load(
                model, input_ids, tok_data, eos_id, traj_offset, traj_vocab_size,
                n_cpu_workers=n_cores, n_steps_sample=5,
            )
        except Exception as e:
            logger.error(f"  [E2/{label}] 측정 실패: {e}")
            traceback.print_exc()
            e2_results[str(n_cores)] = {"error": str(e)}
            continue

        stat = StatSummary(res["step_times_ms"], label)
        slowdown_ms  = stat.mean - BASELINE_DECODE_STEP_MS
        slowdown_pct = slowdown_ms / BASELINE_DECODE_STEP_MS * 100

        print(
            f"  [{label}]  "
            f"step={stat.mean:.1f}ms±{stat.std:.1f}ms  "
            f"slowdown=+{slowdown_ms:.1f}ms ({slowdown_pct:+.1f}%)  "
            f"[이론: {'+7.5%' if n_cores == 2 else '?'}]"
        )
        e2_results[str(n_cores)] = res
        e2_results[str(n_cores)]["slowdown_ms"]  = round(slowdown_ms,  2)
        e2_results[str(n_cores)]["slowdown_pct"] = round(slowdown_pct, 2)

    results["sub_experiments"]["E2"] = e2_results

    # Sub-E2 요약 표
    print(f"\n  ── [Sub-E2] 요약 ──")
    print(f"  {'코어수':>6}  {'step 평균':>10}  {'Decode 저하':>12}  {'파레토 이론':>12}")
    theory = {0: "0%", 1: "-3.4%", 2: "-7.5%", 4: "-19%"}
    for n_cores in cpu_configs:
        r = e2_results.get(str(n_cores), {})
        if "error" in r:
            print(f"  {n_cores:>6}  {'ERROR':>10}")
            continue
        st = r.get("stats", {})
        m = st.get("mean", float("nan"))
        sp = r.get("slowdown_pct", float("nan"))
        print(f"  {n_cores:>6}  {m:>8.1f}ms  {sp:>+10.1f}%  {theory.get(n_cores,'?'):>12}")

    # ────────────────────────────────────────────────────────────────────────
    # Sub-E3: 백그라운드 전처리 overlap 검증
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-E3: 백그라운드 전처리 overlap 검증 ──")
    print("  GPU full inference 실행 중 CPU worker가 다음 프레임을 미리 준비하는가?")
    print("  전처리 완료 시각 < GPU inference 완료 시각 이면 완전 hiding 확인")

    e3_results = []
    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()

        # ★ GPU inference와 CPU 전처리를 동시에 시작
        worker = PreprocessWorker(model, n_cores=CPU_SWEET_SPOT_CORES)
        t_wall = WallTimer()
        t_wall.start()

        # CPU 워커에 t1 전처리 의뢰 (non-blocking)
        t_submit = time.perf_counter()
        worker.submit(CLIP_ID, T1_US)

        # GPU: t0 full prefill (VE + LM) 실행 — 이 동안 CPU worker가 t1 준비
        try:
            kv, _, prefill_ms = run_full_prefill(model, input_ids, tok_data)
        except Exception as e:
            logger.error(f"  [E3/{tag}] GPU prefill 실패: {e}")
            worker.shutdown()
            continue

        t_prefill_done = time.perf_counter()

        # CPU 전처리 완료 대기
        try:
            prep_result = worker.get_result(timeout=30.0)
        except queue.Empty:
            logger.error(f"  [E3/{tag}] CPU 전처리 timeout")
            worker.shutdown()
            continue
        worker.shutdown()

        t_prep_done = time.perf_counter()
        t_wall.stop()

        if prep_result["status"] != "ok":
            logger.error(f"  [E3/{tag}] CPU 전처리 오류: {prep_result.get('error')}")
            continue

        cpu_total_ms  = prep_result["timing"]["total_ms"]
        gpu_total_ms  = prefill_ms   # VE + LM prefill
        wall_ms       = t_wall.elapsed_ms()
        hidden_ms     = cpu_total_ms  # GPU inference 내에 숨겨진 시간
        overlap_pct   = min(cpu_total_ms / gpu_total_ms * 100, 100.0)
        is_fully_hidden = cpu_total_ms <= gpu_total_ms

        print(
            f"  [{tag}]  "
            f"GPU_prefill={gpu_total_ms:.0f}ms  "
            f"CPU_setup={cpu_total_ms:.0f}ms  "
            f"wall={wall_ms:.0f}ms  "
            f"overlap={overlap_pct:.1f}%  "
            f"{'✅ 완전 hidden' if is_fully_hidden else '⚠️ GPU가 먼저 끝남'}"
        )

        if not is_warmup:
            e3_results.append({
                "trial":           tag,
                "gpu_prefill_ms":  round(gpu_total_ms, 1),
                "cpu_setup_ms":    round(cpu_total_ms,  1),
                "wall_ms":         round(wall_ms,        1),
                "overlap_pct":     round(overlap_pct,    2),
                "fully_hidden":    is_fully_hidden,
                "timing_detail":   prep_result["timing"],
            })

    if e3_results:
        avg_gpu  = mean([r["gpu_prefill_ms"] for r in e3_results])
        avg_cpu  = mean([r["cpu_setup_ms"]  for r in e3_results])
        avg_wall = mean([r["wall_ms"]        for r in e3_results])
        hidden_all = all(r["fully_hidden"] for r in e3_results)
        print(f"\n  ── [Sub-E3] 평균 결과 ──")
        print(f"  GPU prefill  : {avg_gpu:.0f}ms")
        print(f"  CPU setup    : {avg_cpu:.0f}ms")
        print(f"  Wall clock   : {avg_wall:.0f}ms (이상적 = max({avg_gpu:.0f}, {avg_cpu:.0f}) = {max(avg_gpu, avg_cpu):.0f}ms)")
        print(f"  Overlap 결론 : {'✅ 전처리가 GPU inference 안에 완전 hiding' if hidden_all else '⚠️ 일부 trial에서 CPU가 GPU보다 늦게 완료'}")
        results["sub_experiments"]["E3"] = {"runs": e3_results, "avg_gpu_ms": round(avg_gpu, 1),
                                             "avg_cpu_ms": round(avg_cpu, 1), "fully_hidden_all": hidden_all}
    else:
        results["sub_experiments"]["E3"] = {"error": "모든 trial 실패"}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 5: 실험 D — CUDA Stream 이중화 (Decode ∥ VE 동시 실행 간섭 측정)
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_D(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
) -> dict:
    """
    [실험 D] CUDA Stream 이중화 — Decode(t) ∥ VE(t+1) 동시 실행 간섭 측정

    ─── 실험 설계 ───────────────────────────────────────────────────────────────
    핵심 질문:
      "VE(728ms)를 Decode(1,818ms = 1,818ms 창) 동안 동시에 돌리면
       Decode step이 얼마나 느려지는가? net gain인가 net loss인가?"

    Sub-D1: VE 단독 실행 시간 측정 (baseline)
      - run_ve_only() → CUDA Event 측정
      - 3회 측정 후 통계 (mean, std)

    Sub-D2: Decode step 단독 시간 측정 (baseline, CUDA Event)
      - 초기 2 step warm-up 제외
      - 5 step 측정 → StatSummary
      - GPU BW 이론값(95ms)과 대조

    Sub-D3: Decode ∥ VE 동시 실행 (2 CUDA Streams)
      - Stream A: Decode step 루프
      - Stream B: VE 반복 실행 (실제 t+1 VE 역할 시뮬레이션)
      - CUDA Event로 각각의 GPU-side 시간 측정
      - WallTimer로 실제 wall-clock overlap 측정
      - 측정 지표:
          * decode_step_ms (overlap 시) vs. baseline
          * ve_ms (overlap 시) vs. baseline
          * overlap_efficiency = (ve_ms + decode_ms - wall_ms) / min(ve_ms, decode_ms)
            → 1.0 = 완전 병렬, 0.0 = 완전 직렬

    Sub-D4: 이론 파이프라인 임팩트 계산
      - Decode slowdown이 X% 일 때 net gain 계산
      - VE 728ms를 Decode 창 안에 숨기기 위한 조건 분석
    ─────────────────────────────────────────────────────────────────────────────

    중요 설계 결정:
      VE는 t0의 pixel_values를 재사용 (t1 데이터 불필요).
      목적은 VE 결과를 실제로 사용하는 것이 아니라 "VE가 돌아갈 때 Decode가 얼마나 느려지는가"를
      측정하는 것이다. 과학적으로 정확한 BW contention 측정이 핵심.
    """
    print("\n" + "═" * 70)
    print("  [실험 D] CUDA Stream 이중화 — Decode ∥ VE 동시 실행 간섭 측정")
    print("═" * 70)

    # VE 모듈 탐지 (없으면 Sub-D1/D3 스킵)
    ve_module = find_visual_encoder(model)
    pixel_values   = tok_data.get("pixel_values")
    image_grid_thw = tok_data.get("image_grid_thw")

    if ve_module is None or pixel_values is None:
        msg = "VE 모듈 탐지 실패 또는 pixel_values 없음 → 실험 D 전체 스킵"
        logger.error(f"  {msg}")
        return {"exp": "D", "error": msg}

    results: Dict[str, Any] = {"exp": "D", "sub_experiments": {}}

    stream_a = torch.cuda.Stream(device=DEVICE)  # Decode용
    stream_b = torch.cuda.Stream(device=DEVICE)  # VE용

    # ────────────────────────────────────────────────────────────────────────
    # Sub-D1: VE 단독 실행 시간 (baseline)
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-D1: VE 단독 실행 시간 (baseline) ──")
    ve_standalone_ms: List[float] = []

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()

        gt = GpuTimer()
        try:
            with torch.cuda.stream(stream_b):
                gt.start(stream_b)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = ve_module(pixel_values, image_grid_thw)
                gt.stop(stream_b)
            stream_b.synchronize()
            ve_ms = gt.elapsed_ms()
        except Exception as e:
            logger.error(f"  [D1/{tag}] VE 실행 실패: {e}")
            traceback.print_exc()
            continue

        print(f"  [{tag}]  VE standalone = {ve_ms:.1f}ms")
        if not is_warmup:
            ve_standalone_ms.append(ve_ms)

    d1_stat = StatSummary(ve_standalone_ms, "VE_standalone")
    print(f"\n  {d1_stat}")
    print(f"  [참고] baseline VE = {BASELINE_VE_MS:.0f}ms (full forward 내 VE 포함 측정)")
    results["sub_experiments"]["D1"] = {
        "description": "VE 단독 실행 시간 (CUDA Event 기반)",
        "baseline_ve_ms": BASELINE_VE_MS,
        "stats": d1_stat.to_dict(),
    }

    # ────────────────────────────────────────────────────────────────────────
    # Sub-D2: Decode step 단독 시간 (baseline)
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-D2: Decode step 단독 시간 (baseline) ──")
    print(f"  이론 하한: {22:.0f}GB ÷ 231 GB/s = 95ms/step")
    print(f"  이전 실측: {BASELINE_DECODE_STEP_MS:.0f}ms/step (88% 효율)")

    WARMUP_STEPS   = 2
    MEASURE_STEPS  = 5
    decode_alone_steps: List[float] = []

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()

        kv, logits, _ = run_full_prefill(model, input_ids, tok_data)
        prefill_len   = int(input_ids.shape[1])
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        trial_steps: List[float] = []

        try:
            for step in range(1, WARMUP_STEPS + MEASURE_STEPS + 1):
                cpos = torch.tensor([prefill_len + step - 1], device=DEVICE, dtype=torch.long)
                gt = GpuTimer()
                with torch.cuda.stream(stream_a):
                    gt.start(stream_a)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model.vlm(
                            input_ids=next_tok,
                            past_key_values=kv,
                            cache_position=cpos,
                            use_cache=True,
                        )
                    gt.stop(stream_a)
                stream_a.synchronize()
                step_ms = gt.elapsed_ms()

                kv      = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :].float(), dim=-1, keepdim=True)

                if step > WARMUP_STEPS:
                    trial_steps.append(step_ms)
        except Exception as e:
            logger.error(f"  [D2/{tag}] Decode 실패: {e}")
            traceback.print_exc()
            continue

        stat_trial = StatSummary(trial_steps, f"decode_alone_{tag}")
        print(f"  [{tag}]  {MEASURE_STEPS} steps: {stat_trial}")
        if not is_warmup:
            decode_alone_steps.extend(trial_steps)

    d2_stat = StatSummary(decode_alone_steps, "Decode_step_alone")
    print(f"\n  Decode 단독 (전체):  {d2_stat}")
    results["sub_experiments"]["D2"] = {
        "description": "Decode step 단독 시간 (CUDA Event, Stream A)",
        "theory_lower_bound_ms": 95.0,
        "baseline_ms": BASELINE_DECODE_STEP_MS,
        "stats": d2_stat.to_dict(),
    }

    # ────────────────────────────────────────────────────────────────────────
    # Sub-D3: Decode ∥ VE 동시 실행 (2 CUDA Streams 간섭 측정)
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-D3: Decode ∥ VE 동시 실행 (CUDA Stream 이중화) ──")
    print("  Stream A: Decode steps  |  Stream B: VE 반복 실행")
    print("  측정: Decode slowdown, VE slowdown, overlap efficiency")

    overlap_results: List[dict] = []

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()

        kv, logits, _ = run_full_prefill(model, input_ids, tok_data)
        prefill_len   = int(input_ids.shape[1])
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)

        # 동시 실행 시 각각의 Step 시간 수집
        decode_overlap_steps: List[float] = []
        ve_overlap_times:     List[float] = []
        ve_done_flag = threading.Event()
        ve_exception: List[Exception] = []

        def run_ve_on_stream_b():
            """
            Stream B에서 VE를 반복 실행.
            Decode 루프가 실행되는 동안 VE를 계속 돌려 BW 경합을 지속시킨다.
            WARMUP_STEPS + MEASURE_STEPS 동안 실행.
            """
            ve_gt = GpuTimer()
            try:
                for _ in range(WARMUP_STEPS + MEASURE_STEPS):
                    with torch.cuda.stream(stream_b):
                        ve_gt.start(stream_b)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            _ = ve_module(pixel_values, image_grid_thw)
                        ve_gt.stop(stream_b)
                    stream_b.synchronize()
                    if not is_warmup:
                        ve_overlap_times.append(ve_gt.elapsed_ms())
            except Exception as e:
                ve_exception.append(e)
            finally:
                ve_done_flag.set()

        # ★ VE 스레드 시작 (non-blocking — GPU 커널은 stream_b에 enqueue)
        ve_thread = threading.Thread(target=run_ve_on_stream_b, daemon=True)

        wt = WallTimer()
        wt.start()
        ve_thread.start()

        try:
            for step in range(1, WARMUP_STEPS + MEASURE_STEPS + 1):
                cpos = torch.tensor([prefill_len + step - 1], device=DEVICE, dtype=torch.long)
                gt_d = GpuTimer()
                with torch.cuda.stream(stream_a):
                    gt_d.start(stream_a)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model.vlm(
                            input_ids=next_tok,
                            past_key_values=kv,
                            cache_position=cpos,
                            use_cache=True,
                        )
                    gt_d.stop(stream_a)
                stream_a.synchronize()
                step_ms = gt_d.elapsed_ms()

                kv       = out.past_key_values
                next_tok = torch.argmax(out.logits[:, -1, :].float(), dim=-1, keepdim=True)

                if step > WARMUP_STEPS:
                    decode_overlap_steps.append(step_ms)
        except Exception as e:
            logger.error(f"  [D3/{tag}] Decode 실패: {e}")
            traceback.print_exc()

        ve_done_flag.wait(timeout=30.0)
        wt.stop()

        if ve_exception:
            logger.error(f"  [D3/{tag}] VE 실패: {ve_exception[0]}")
            continue

        # 통계 계산
        dec_stat = StatSummary(decode_overlap_steps, "decode_overlap")
        ve_stat  = StatSummary(ve_overlap_times,      "ve_overlap")

        decode_slowdown_ms  = dec_stat.mean - d2_stat.mean
        decode_slowdown_pct = decode_slowdown_ms / d2_stat.mean * 100 if d2_stat.mean > 0 else 0
        ve_slowdown_ms      = ve_stat.mean - d1_stat.mean
        ve_slowdown_pct     = ve_slowdown_ms / d1_stat.mean * 100 if d1_stat.mean > 0 else 0

        # Overlap efficiency:
        #   = 1 - (wall - max(decode_total, ve_time)) / min(decode_total, ve_time)
        #   1.0 = 완전 병렬, 0.0 = 완전 직렬
        decode_total_ms = dec_stat.mean * MEASURE_STEPS  # 5 step 총합
        ve_total_ms     = ve_stat.mean  * (WARMUP_STEPS + MEASURE_STEPS)
        wall_ms         = wt.elapsed_ms()
        # 직렬 예상시간: decode_total + ve_total
        serial_expected = decode_total_ms + d1_stat.mean * (WARMUP_STEPS + MEASURE_STEPS)
        overlap_efficiency = max(0.0, (serial_expected - wall_ms) / serial_expected) * 100

        print(
            f"  [{tag}]  "
            f"decode={dec_stat.mean:.1f}ms ({decode_slowdown_pct:+.1f}%)  "
            f"ve={ve_stat.mean:.1f}ms ({ve_slowdown_pct:+.1f}%)  "
            f"wall={wall_ms:.0f}ms  "
            f"overlap_efficiency={overlap_efficiency:.1f}%"
        )

        if not is_warmup:
            overlap_results.append({
                "trial":                tag,
                "decode_step_ms":       dec_stat.to_dict(),
                "ve_overlap_ms":        ve_stat.to_dict(),
                "decode_slowdown_ms":   round(decode_slowdown_ms,  2),
                "decode_slowdown_pct":  round(decode_slowdown_pct, 2),
                "ve_slowdown_ms":       round(ve_slowdown_ms,      2),
                "ve_slowdown_pct":      round(ve_slowdown_pct,     2),
                "wall_ms":              round(wall_ms,              2),
                "overlap_efficiency_pct": round(overlap_efficiency, 2),
            })

    # Sub-D4: 이론 파이프라인 임팩트 계산
    print("\n  ── Sub-D4: 파이프라인 임팩트 계산 ──")
    if overlap_results:
        avg_dec_slow = mean([r["decode_slowdown_pct"] for r in overlap_results])
        avg_eff      = mean([r["overlap_efficiency_pct"] for r in overlap_results])
        avg_ve_slow  = mean([r["ve_slowdown_pct"] for r in overlap_results])

        # 파이프라인 효과: Decode 17 step, VE 728ms 숨기기
        baseline_decode_total = BASELINE_DECODE_STEP_MS * 17  # ~1,818ms
        overlap_decode_total  = d2_stat.mean * (1 + avg_dec_slow / 100) * 17
        penalty_ms            = overlap_decode_total - baseline_decode_total
        ve_hidden_ms          = d1_stat.mean  # VE는 Decode 창 안에서 실행
        net_gain_ms           = ve_hidden_ms - penalty_ms

        print(f"  Decode 17step baseline  : {baseline_decode_total:.0f}ms")
        print(f"  Decode 17step (overlap) : {overlap_decode_total:.0f}ms (+{penalty_ms:.0f}ms, {avg_dec_slow:+.1f}%)")
        print(f"  VE 은닉 이득            : {ve_hidden_ms:.0f}ms (Decode 창 안에서 실행)")
        print(f"  Net gain                : {net_gain_ms:+.0f}ms  "
              f"({'✅ 유리' if net_gain_ms > 0 else '❌ 손해 — Decode 저하가 VE 이득보다 큼'})")
        print(f"  Overlap efficiency      : {avg_eff:.1f}%")
        print(f"  VE 자체 저하            : {avg_ve_slow:+.1f}%")

        d4 = {
            "baseline_decode_17step_ms": round(baseline_decode_total, 1),
            "overlap_decode_17step_ms":  round(overlap_decode_total, 1),
            "penalty_ms":                round(penalty_ms, 1),
            "ve_hidden_ms":              round(ve_hidden_ms, 1),
            "net_gain_ms":               round(net_gain_ms, 1),
            "avg_overlap_efficiency_pct": round(avg_eff, 2),
            "verdict": "유리 (VE 은닉 > Decode 저하)" if net_gain_ms > 0 else "불리 (Decode 저하 > VE 이득)",
        }
        results["sub_experiments"]["D4"] = d4
    else:
        results["sub_experiments"]["D4"] = {"error": "Sub-D3 측정값 없음"}

    results["sub_experiments"]["D3"] = {"runs": overlap_results}
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 6: 실험 F — CPU Flow + GPU VE 동시 실행 (태스크 분리)
# ══════════════════════════════════════════════════════════════════════════════

def _discover_flow_method(
    model: Alpamayo1_5,
    flow_module: torch.nn.Module,
    traj_tensor_gpu: torch.Tensor,
    traj_tokens: List[int],
) -> Optional[callable]:
    """
    FlowMatching 모듈의 실제 추론 메서드를 탐색하고 호출 가능한 callable을 반환한다.

    FlowMatching은 forward()가 구현되지 않은 orchestrator다.
    sample/decode/infer/generate 등 고유 메서드로 추론을 수행한다.

    반환:
        callable(tensor) -> tensor  — 성공 시
        None                        — 모든 시도 실패 시
    """
    import inspect as _ins

    print("\n  ─────────────────────────────────────────────────────────")
    print("  [FlowMatching 호출 인터페이스 탐색]")
    print("  ─────────────────────────────────────────────────────────")

    # 1. 소스 파일 위치
    try:
        src_file = _ins.getfile(type(flow_module))
        print(f"  소스 파일: {src_file}")
    except (TypeError, OSError):
        print("  소스 파일: (built-in or C extension)")

    # 2. FlowMatching 고유 메서드 (nn.Module 기본 메서드 제외)
    nn_base = set(dir(torch.nn.Module))
    fm_specific: List[Tuple[str, str]] = []
    for name in sorted(dir(flow_module)):
        if name in nn_base or name.startswith("_"):
            continue
        attr = getattr(flow_module, name, None)
        if callable(attr):
            try:
                sig = str(_ins.signature(attr))
            except (ValueError, TypeError):
                sig = "(?)"
            fm_specific.append((name, sig))

    if fm_specific:
        print(f"\n  FlowMatching 고유 메서드 ({len(fm_specific)}개):")
        for name, sig in fm_specific:
            print(f"    .{name}{sig}")
    else:
        print("\n  FlowMatching 고유 메서드: 없음")

    # 3. model 레벨 관련 메서드 탐색
    model_flow_methods: List[Tuple[str, str]] = []
    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        attr = getattr(model, name, None)
        if attr is None or isinstance(attr, torch.nn.Module):
            continue
        if not callable(attr):
            continue
        if any(k in name.lower() for k in
               ["flow", "traj", "action", "diffus", "sample", "decode_act", "decode_traj"]):
            try:
                sig = str(_ins.signature(attr))
            except (ValueError, TypeError):
                sig = "(?)"
            model_flow_methods.append((name, sig))

    if model_flow_methods:
        print(f"\n  model 레벨 관련 메서드 ({len(model_flow_methods)}개):")
        for name, sig in model_flow_methods:
            print(f"    model.{name}{sig}")

    # 4. 관련 top-level 모듈 파라미터 확인
    print("\n  관련 모듈 파라미터:")
    for mod_name in ["expert", "action_space", "action_in_proj", "action_out_proj", "diffusion"]:
        mod = getattr(model, mod_name, None)
        if mod is not None and isinstance(mod, torch.nn.Module):
            params_M = sum(p.numel() for p in mod.parameters()) / 1e6
            print(f"    model.{mod_name}: {type(mod).__name__}  {params_M:.1f}M params  "
                  f"device={next(mod.parameters(), torch.empty(0)).device}")

    # 5. 호출 시도 순서 구성
    print("\n  [호출 시도]")
    candidate_calls: List[Tuple[str, callable, torch.Tensor]] = []

    # flow_module 직접 메서드
    priority_methods = ["sample", "decode", "infer", "generate", "step", "denoise",
                        "predict", "inference", "forward_pass"]
    # 탐색된 고유 메서드 포함
    for name, _ in fm_specific:
        if name not in priority_methods:
            priority_methods.append(name)

    for method_name in priority_methods:
        method = getattr(flow_module, method_name, None)
        if method is not None and callable(method):
            candidate_calls.append(
                (f"flow_module.{method_name}(x)", method, traj_tensor_gpu)
            )
            candidate_calls.append(
                (f"flow_module.{method_name}(x.float())", method, traj_tensor_gpu.float())
            )

    # model 레벨 메서드
    for mname, _ in model_flow_methods:
        meth = getattr(model, mname, None)
        if meth is not None and callable(meth):
            candidate_calls.append(
                (f"model.{mname}(traj_tokens_tensor)", meth, traj_tensor_gpu)
            )

    for desc, fn, arg in candidate_calls:
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = fn(arg)
            print(f"    ✅ 성공: {desc}")
            captured_fn  = fn
            captured_arg = arg
            # 고정된 callable 반환 (같은 module + method, 인자 dtype 포함)
            return lambda x, _fn=captured_fn: _fn(x)
        except TypeError as e:
            print(f"    ↳ TypeError (인자 불일치): {desc} — {str(e)[:100]}")
        except NotImplementedError as e:
            print(f"    ↳ NotImplementedError: {desc} — {str(e)[:100]}")
        except Exception as e:
            print(f"    ↳ {type(e).__name__}: {desc} — {str(e)[:100]}")

    # 6. 모두 실패: 수동 확인 안내
    print("\n  ❌ 자동 탐색 실패. 수동 확인 명령:")
    print("  ──────────────────────────────────────────────")
    print("  python3 -c \"")
    print("  import sys; sys.path.insert(0,'src'); import inspect, torch")
    print("  from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5")
    print("  m = Alpamayo1_5.from_pretrained('nvidia/Alpamayo-1.5-10B',")
    print("      dtype=torch.bfloat16, local_files_only=True)")
    print("  print(inspect.getsource(type(m.diffusion)))")
    print("  \"")
    print("  ──────────────────────────────────────────────")
    return None

def run_experiment_F(
    model: Alpamayo1_5,
    input_ids: torch.Tensor,
    tok_data: dict,
    eos_id: int,
    traj_offset: int,
    traj_vocab_size: int,
) -> dict:
    """
    [실험 F] CPU Flow + GPU VE 동시 실행 — 태스크 분리 가능성 검증

    ─── 실험 설계 ───────────────────────────────────────────────────────────────
    핵심 질문:
      "Decode 후 Flow decoder(870ms)를 CPU로 옮기고,
       GPU는 즉시 다음 프레임 VE를 시작할 수 있는가?
       CPU Flow 시간이 VE(728ms) 이내면 Flow를 공짜로 숨길 수 있다."

    Sub-F1: Flow decoder 탐지 및 GPU 실행 시간 측정
      - model inspection으로 flow_decoder 위치 파악
      - GPU에서 실행 시 정확한 시간 측정 (CUDA Event)
      - 베이스라인: 870ms (전체 파이프라인 측정값)

    Sub-F2: Flow decoder CPU 이식 및 시간 측정
      - flow_decoder.cpu() → 동일 데이터로 CPU에서 실행
      - 시간 측정 (CpuTimer, ms 단위)
      - iGPU 통합 메모리: CPU ↔ GPU 전송 = zero-copy (물리적으로 동일 DRAM)
        → .cpu() 호출은 logical device label 변경만. 실제 데이터 이동 없음.

    Sub-F3: CPU Flow ∥ GPU VE 동시 실행
      - 타임라인:
          [GPU Decode 완료] → [CPU thread: Flow 시작] + [GPU: VE 시작]
          → 둘 중 오래 걸리는 쪽이 병목
      - 측정:
          * CPU Flow 완료 시각 vs. GPU VE 완료 시각
          * Wall-clock: max(cpu_flow, gpu_ve) vs. 직렬(cpu_flow + gpu_ve)
          * 실질 hiding: min(cpu_flow, gpu_ve)

    Sub-F4: 이론 파이프라인 임팩트
      - 절약 가능한 시간 = min(flow_cpu_ms, ve_ms)
      - 전체 파이프라인에서 절약 비율 계산
    ─────────────────────────────────────────────────────────────────────────────
    """
    print("\n" + "═" * 70)
    print("  [실험 F] CPU Flow + GPU VE 동시 실행 (태스크 분리)")
    print("═" * 70)

    flow_attr, flow_module = find_flow_decoder(model)
    ve_module   = find_visual_encoder(model)
    pixel_values   = tok_data.get("pixel_values")
    image_grid_thw = tok_data.get("image_grid_thw")

    results: Dict[str, Any] = {"exp": "F", "sub_experiments": {}}

    if flow_module is None:
        # Flow 미탐지: model 최상위 모듈 목록 + 시간 분해 정보 기록
        print("\n  ⚠️  Flow decoder 자동 탐지 실패.")
        print("  → 모델 최상위 모듈:")
        for name, mod in model.named_children():
            n_params = sum(p.numel() for p in mod.parameters()) / 1e6
            print(f"      model.{name:25s}  ({n_params:.1f}M params)")
        print("\n  → model.vlm 하위 모듈:")
        try:
            for name, mod in model.vlm.named_children():
                n_params = sum(p.numel() for p in mod.parameters()) / 1e6
                print(f"      model.vlm.{name:20s}  ({n_params:.1f}M params)")
        except Exception:
            pass
        print("\n  위 정보를 이용해 flow_decoder 경로를 확인 후")
        print("  find_flow_decoder() 함수의 candidates 목록에 추가하세요.")
        results["sub_experiments"]["F_skip"] = {
            "reason": "Flow decoder not found",
            "top_modules": [name for name, _ in model.named_children()],
        }
        return results

    import inspect as _inspect
    print(f"  Flow decoder 탐지: model.{flow_attr}")
    flow_params = sum(p.numel() for p in flow_module.parameters())
    print(f"  Flow params: {flow_params/1e6:.1f}M ({flow_params*2/1e6:.1f}MB BF16)")
    # forward 시그니처 출력 — 호출 규약 파악용
    try:
        sig = _inspect.signature(flow_module.forward)
        print(f"  forward signature: {sig}")
    except Exception:
        print(f"  forward signature: (검사 불가 — C extension 또는 __call__ override)")

    stream_ve = torch.cuda.Stream(device=DEVICE)

    # ── Decode를 1회 실행해서 실제 trajectory 토큰 획득 ───────────────────
    # Sub-F1, F2, F3에서 동일한 traj_tokens를 입력으로 사용
    kv, logits, _ = run_full_prefill(model, input_ids, tok_data)
    prefill_len = int(input_ids.shape[1])
    traj_tokens: List[int] = []
    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
    cur_kv = kv

    for step in range(1, MAX_DECODE_STEPS + 1):
        cpos = torch.tensor([prefill_len + step - 1], device=DEVICE, dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.vlm(input_ids=next_tok, past_key_values=cur_kv,
                            cache_position=cpos, use_cache=True)
        cur_kv   = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :].float(), dim=-1, keepdim=True)
        tok_id   = int(next_tok.item())
        traj_tokens.append(tok_id)
        if tok_id == eos_id:
            break
    torch.cuda.synchronize()

    traj_tensor_gpu = torch.tensor(traj_tokens, device=DEVICE, dtype=torch.long).unsqueeze(0)
    traj_tensor_cpu = traj_tensor_gpu.cpu()
    print(f"  Decode 완료: {len(traj_tokens)} trajectory tokens 수집")

    # ── FlowMatching 호출 인터페이스 자동 탐색 ────────────────────────────────
    flow_fn = _discover_flow_method(model, flow_module, traj_tensor_gpu, traj_tokens)
    if flow_fn is None:
        results["sub_experiments"]["F_skip"] = {
            "reason": "FlowMatching calling convention unknown — see diagnostic output above",
            "flow_module_type": type(flow_module).__name__,
            "flow_params_M": sum(p.numel() for p in flow_module.parameters()) / 1e6,
        }
        return results

    # ────────────────────────────────────────────────────────────────────────
    # Sub-F1: Flow decoder GPU 실행 시간 측정
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-F1: Flow decoder GPU 실행 시간 (baseline) ──")
    flow_gpu_ms_list: List[float] = []

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"
        torch.cuda.empty_cache()

        gt = GpuTimer()
        try:
            gt.start()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = flow_fn(traj_tensor_gpu)
            gt.stop()
            torch.cuda.synchronize()
            flow_gpu_ms = gt.elapsed_ms()
        except Exception as e:
            logger.error(f"  [F1/{tag}] Flow GPU 실패: {e}")
            traceback.print_exc()
            continue

        print(f"  [{tag}]  Flow GPU = {flow_gpu_ms:.1f}ms")
        if not is_warmup:
            flow_gpu_ms_list.append(flow_gpu_ms)

    f1_stat = StatSummary(flow_gpu_ms_list, "Flow_GPU")
    print(f"\n  {f1_stat}")
    print(f"  [참고] baseline Flow = {BASELINE_FLOW_MS:.0f}ms (전체 파이프라인 측정값)")
    results["sub_experiments"]["F1"] = {
        "description": "Flow decoder GPU 실행 시간",
        "baseline_flow_ms": BASELINE_FLOW_MS,
        "stats": f1_stat.to_dict(),
    }

    # ────────────────────────────────────────────────────────────────────────
    # Sub-F2: Flow decoder CPU 이식 및 실행 시간 측정
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-F2: Flow decoder CPU 실행 시간 ──")
    print("  (iGPU 통합 메모리: .cpu() = logical label 변경만, 실제 DMA 없음)")
    print("  CPU affinity: 2코어 고정 (sweet spot — GPU 7.5% 손실 수준)")

    # Flow 모듈을 CPU로 이동 (가중치도 CPU로)
    try:
        flow_module_cpu = flow_module.cpu()
    except Exception as e:
        logger.error(f"  Flow decoder CPU 이식 실패: {e}")
        results["sub_experiments"]["F2"] = {"error": str(e)}
        flow_module.to(DEVICE)  # GPU 복귀
        return results

    # CPU 모듈용 callable 재탐색 (device=cpu로 이동했으므로 재프로브)
    traj_tensor_cpu_probe = traj_tensor_gpu.cpu()
    flow_fn_cpu = _discover_flow_method(model, flow_module_cpu, traj_tensor_cpu_probe, traj_tokens)
    if flow_fn_cpu is None:
        print("  ⚠️  CPU FlowMatching 호출 방법 미탐지 → Sub-F2 스킵")
        results["sub_experiments"]["F2"] = {"error": "CPU FlowMatching calling convention unknown"}
        flow_module.to(DEVICE)
        return results

    flow_cpu_ms_list: List[float] = []
    orig_affinity = None

    try:
        orig_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, CPU_AFFINITY_CORES)
    except (AttributeError, OSError) as e:
        logger.warning(f"  CPU affinity 설정 실패: {e}")

    for trial in range(NUM_WARMUP + NUM_MEASURE):
        is_warmup = trial < NUM_WARMUP
        tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"

        ct = CpuTimer()
        try:
            ct.start()
            with torch.no_grad():
                _ = flow_fn_cpu(traj_tensor_cpu)
            ct.stop()
            flow_cpu_ms = ct.elapsed_ms()
        except Exception as e:
            logger.error(f"  [F2/{tag}] Flow CPU 실패: {e}")
            traceback.print_exc()
            continue

        print(f"  [{tag}]  Flow CPU = {flow_cpu_ms:.1f}ms")
        if not is_warmup:
            flow_cpu_ms_list.append(flow_cpu_ms)

    if orig_affinity:
        try:
            os.sched_setaffinity(0, orig_affinity)
        except (AttributeError, OSError):
            pass

    f2_stat = StatSummary(flow_cpu_ms_list, "Flow_CPU")
    flow_cpu_gpu_ratio = f2_stat.mean / f1_stat.mean if f1_stat.mean > 0 else float("nan")
    print(f"\n  {f2_stat}")
    print(f"  GPU 대비 CPU 비율: {flow_cpu_gpu_ratio:.2f}×  "
          f"({'✅ CPU가 빠름' if flow_cpu_gpu_ratio < 1 else f'CPU가 {flow_cpu_gpu_ratio:.1f}× 느림'})")
    results["sub_experiments"]["F2"] = {
        "description": "Flow decoder CPU 실행 시간 (2코어 affinity)",
        "stats": f2_stat.to_dict(),
        "cpu_vs_gpu_ratio": round(flow_cpu_gpu_ratio, 3),
    }

    # Flow 모듈 GPU 복귀
    flow_module.to(DEVICE)
    torch.cuda.synchronize()

    # ────────────────────────────────────────────────────────────────────────
    # Sub-F3: CPU Flow ∥ GPU VE 동시 실행
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-F3: CPU Flow ∥ GPU VE 동시 실행 ──")
    if ve_module is None or pixel_values is None:
        print("  ⚠️  VE 모듈 미탐지 → Sub-F3 스킵")
        results["sub_experiments"]["F3"] = {"error": "VE module not found"}
    else:
        print("  타임라인: [CPU: Flow(t)] + [GPU Stream B: VE(t+1)] 동시 시작")
        print(f"  VE baseline: {BASELINE_VE_MS:.1f}ms")
        print(f"  Flow CPU:    {f2_stat.mean:.1f}ms")

        # Sub-F2에서 사용한 CPU 모듈 + callable 재준비
        flow_module_cpu_f3 = flow_module.cpu()
        # flow_fn_cpu는 이미 위에서 탐색됨 — closure 내 traj_tensor_cpu 참조를 위해 재사용

        parallel_results: List[dict] = []
        for trial in range(NUM_WARMUP + NUM_MEASURE):
            is_warmup = trial < NUM_WARMUP
            tag = "WARMUP" if is_warmup else f"MEASURE {trial - NUM_WARMUP + 1}"
            torch.cuda.empty_cache()

            cpu_flow_ms_list: List[float] = []
            gpu_ve_ms_list:   List[float] = []
            cpu_exception:    List[Exception] = []
            cpu_done_flag = threading.Event()

            def run_flow_cpu():
                try:
                    orig = None
                    try:
                        orig = os.sched_getaffinity(0)
                        os.sched_setaffinity(0, CPU_AFFINITY_CORES)
                    except (AttributeError, OSError):
                        pass
                    ct = CpuTimer()
                    ct.start()
                    with torch.no_grad():
                        _ = flow_fn_cpu(traj_tensor_cpu)
                    ct.stop()
                    cpu_flow_ms_list.append(ct.elapsed_ms())
                    if orig:
                        os.sched_setaffinity(0, orig)
                except Exception as e:
                    cpu_exception.append(e)
                finally:
                    cpu_done_flag.set()

            # ★ 두 태스크를 동시에 시작
            wt_f3 = WallTimer()
            wt_f3.start()

            cpu_thread = threading.Thread(target=run_flow_cpu, daemon=True)
            cpu_thread.start()

            # GPU: VE 실행 (Stream B)
            gt_ve = GpuTimer()
            try:
                with torch.cuda.stream(stream_ve):
                    gt_ve.start(stream_ve)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = ve_module(pixel_values, image_grid_thw)
                    gt_ve.stop(stream_ve)
                stream_ve.synchronize()
                gpu_ve_ms = gt_ve.elapsed_ms()
            except Exception as e:
                logger.error(f"  [F3/{tag}] GPU VE 실패: {e}")
                cpu_done_flag.wait(timeout=10.0)
                continue

            cpu_done_flag.wait(timeout=30.0)
            wt_f3.stop()

            if cpu_exception:
                logger.error(f"  [F3/{tag}] CPU Flow 실패: {cpu_exception[0]}")
                continue

            cpu_flow_ms = cpu_flow_ms_list[0] if cpu_flow_ms_list else float("nan")
            wall_ms     = wt_f3.elapsed_ms()
            serial_ms   = cpu_flow_ms + gpu_ve_ms
            hiding_ms   = serial_ms - wall_ms   # 실제 hiding된 시간

            print(
                f"  [{tag}]  "
                f"Flow_CPU={cpu_flow_ms:.0f}ms  "
                f"VE_GPU={gpu_ve_ms:.0f}ms  "
                f"wall={wall_ms:.0f}ms  "
                f"serial={serial_ms:.0f}ms  "
                f"hiding={hiding_ms:.0f}ms ({hiding_ms/serial_ms*100:.1f}%)"
            )

            if not is_warmup:
                parallel_results.append({
                    "trial":          tag,
                    "flow_cpu_ms":    round(cpu_flow_ms, 1),
                    "ve_gpu_ms":      round(gpu_ve_ms,   1),
                    "wall_ms":        round(wall_ms,      1),
                    "serial_ms":      round(serial_ms,    1),
                    "hiding_ms":      round(hiding_ms,    1),
                    "hiding_pct":     round(hiding_ms / serial_ms * 100, 2) if serial_ms > 0 else 0.0,
                    "bottleneck":     "flow_cpu" if cpu_flow_ms > gpu_ve_ms else "ve_gpu",
                })

        flow_module.to(DEVICE)  # GPU 복귀

        if parallel_results:
            avg_hiding = mean([r["hiding_ms"] for r in parallel_results])
            avg_wall   = mean([r["wall_ms"]   for r in parallel_results])
            print(f"\n  ── [Sub-F3] 평균 결과 ──")
            print(f"  실제 wall-clock : {avg_wall:.0f}ms")
            print(f"  평균 hiding     : {avg_hiding:.0f}ms")
            bottlenecks = [r["bottleneck"] for r in parallel_results]
            dominant = max(set(bottlenecks), key=bottlenecks.count)
            print(f"  병목            : {dominant}")
            results["sub_experiments"]["F3"] = {
                "runs": parallel_results,
                "avg_hiding_ms": round(avg_hiding, 1),
                "avg_wall_ms":   round(avg_wall, 1),
                "dominant_bottleneck": dominant,
            }
        else:
            results["sub_experiments"]["F3"] = {"error": "모든 trial 실패"}

    # ────────────────────────────────────────────────────────────────────────
    # Sub-F4: 이론 파이프라인 임팩트
    # ────────────────────────────────────────────────────────────────────────
    print("\n  ── Sub-F4: 이론 파이프라인 임팩트 ──")
    flow_cpu_avg = f2_stat.mean if f2_stat.values else BASELINE_FLOW_MS
    ve_avg       = BASELINE_VE_MS   # Exp F는 VE 단독 측정 없음 → 베이스라인 사용

    hiding = min(flow_cpu_avg, ve_avg)
    new_pipeline_ms = BASELINE_TOTAL_MS - hiding

    print(f"  현재 전체 파이프라인  : {BASELINE_TOTAL_MS:.0f}ms")
    print(f"  Flow CPU 시간         : {flow_cpu_avg:.0f}ms")
    print(f"  VE GPU 시간           : {ve_avg:.0f}ms")
    print(f"  숨길 수 있는 시간     : min({flow_cpu_avg:.0f}, {ve_avg:.0f}) = {hiding:.0f}ms")
    print(f"  예상 파이프라인       : {new_pipeline_ms:.0f}ms (절약 {hiding:.0f}ms, {hiding/BASELINE_TOTAL_MS*100:.1f}%)")

    results["sub_experiments"]["F4"] = {
        "flow_cpu_avg_ms":      round(flow_cpu_avg, 1),
        "ve_avg_ms":            round(ve_avg,       1),
        "hiding_ms":            round(hiding,        1),
        "new_pipeline_est_ms":  round(new_pipeline_ms, 1),
        "saving_pct":           round(hiding / BASELINE_TOTAL_MS * 100, 2),
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 섹션 7: main — 인자 파싱, 모델 로드, 실험 실행, 결과 저장
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CPU-GPU 비동기 파이프라이닝 실험 PoC (Exp E, D, F)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실험 설명:
  E  : CPU 2코어 전처리 파이프라이닝 (setup time 측정 + background overlap)
  D  : CUDA Stream 이중화 (Decode ∥ VE 동시 실행 GPU BW 경합 측정)
  F  : CPU Flow + GPU VE 동시 실행 (태스크 분리 가능성)
  ALL: E → D → F 순서 전체 실행
        """,
    )
    parser.add_argument(
        "--exp",
        choices=["E", "D", "F", "ALL"],
        default="ALL",
        help="실험 선택 (기본: ALL)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  CPU-GPU 비동기 파이프라이닝 실험 PoC")
    print(f"  exp={args.exp}  device={DEVICE}  dtype=BF16")
    print("=" * 70)
    print(f"\n  베이스라인 (sdpa+DynamicCache, BF16):")
    print(f"    VE        : {BASELINE_VE_MS:.0f}ms")
    print(f"    LM Prefill: {BASELINE_PREFILL_MS:.0f}ms")
    print(f"    Decode    : {BASELINE_DECODE_MS:.0f}ms  ({BASELINE_DECODE_STEP_MS:.0f}ms/step × 17)")
    print(f"    Flow      : {BASELINE_FLOW_MS:.0f}ms")
    print(f"    합계      : {BASELINE_TOTAL_MS:.0f}ms")
    print(f"\n  CPU sweet spot: 2코어 → +37 GB/s CPU, GPU -7.5% (파레토 분석)")
    print(f"  Decode 물리 하한: 22GB ÷ 231 GB/s = 95ms/step (현재 107ms = 88%)")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    logger.info("\n모델 로드 중 (sdpa 기본값, BF16)...")
    model = (
        Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(DEVICE)
        .eval()
    )
    model_info = introspect_model(model)

    # ── 입력 데이터 준비 ──────────────────────────────────────────────────────
    logger.info("\n입력 데이터 준비 중 (t0)...")
    input_ids, tok_data, setup_timing = prepare_inputs_timed(model, CLIP_ID, T0_US)
    logger.info(
        f"  t0 input_ids: {input_ids.shape}  ({input_ids.shape[1]} tokens)\n"
        f"  setup time: {setup_timing['total_ms']:.0f}ms "
        f"(load={setup_timing['load_ms']:.0f}  "
        f"tokenize={setup_timing['tokenize_ms']:.0f}  "
        f"fuse={setup_timing['fuse_ms']:.0f})"
    )

    eos_id          = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_offset     = model.config.traj_token_start_idx
    traj_vocab_size = model.config.traj_vocab_size
    logger.info(
        f"  eos_id={eos_id}  traj_offset={traj_offset}  "
        f"traj_vocab_size={traj_vocab_size}"
    )

    all_results: Dict[str, Any] = {
        "model_info":        model_info,
        "baseline":          {
            "ve_ms":       BASELINE_VE_MS,
            "prefill_ms":  BASELINE_PREFILL_MS,
            "decode_ms":   BASELINE_DECODE_MS,
            "flow_ms":     BASELINE_FLOW_MS,
            "total_ms":    BASELINE_TOTAL_MS,
            "step_ms":     BASELINE_DECODE_STEP_MS,
        },
        "initial_setup_timing": setup_timing,
    }

    run_E = args.exp in ("E", "ALL")
    run_D = args.exp in ("D", "ALL")
    run_F = args.exp in ("F", "ALL")

    # ── 실험 E ───────────────────────────────────────────────────────────────
    if run_E:
        try:
            res_E = run_experiment_E(
                model, input_ids, tok_data, eos_id, traj_offset, traj_vocab_size
            )
        except Exception as e:
            logger.error(f"실험 E 전체 실패: {e}")
            traceback.print_exc()
            res_E = {"exp": "E", "error": str(e)}
        all_results["experiment_E"] = res_E
        out_e = OUT / "results_E.json"
        out_e.write_text(json.dumps(res_E, indent=2, default=str))
        logger.info(f"  결과 저장: {out_e}")

    # ── 실험 D ───────────────────────────────────────────────────────────────
    if run_D:
        try:
            res_D = run_experiment_D(
                model, input_ids, tok_data, eos_id, traj_offset, traj_vocab_size
            )
        except Exception as e:
            logger.error(f"실험 D 전체 실패: {e}")
            traceback.print_exc()
            res_D = {"exp": "D", "error": str(e)}
        all_results["experiment_D"] = res_D
        out_d = OUT / "results_D.json"
        out_d.write_text(json.dumps(res_D, indent=2, default=str))
        logger.info(f"  결과 저장: {out_d}")

    # ── 실험 F ───────────────────────────────────────────────────────────────
    if run_F:
        try:
            res_F = run_experiment_F(
                model, input_ids, tok_data, eos_id, traj_offset, traj_vocab_size
            )
        except Exception as e:
            logger.error(f"실험 F 전체 실패: {e}")
            traceback.print_exc()
            res_F = {"exp": "F", "error": str(e)}
        all_results["experiment_F"] = res_F
        out_f = OUT / "results_F.json"
        out_f.write_text(json.dumps(res_F, indent=2, default=str))
        logger.info(f"  결과 저장: {out_f}")

    # ── 종합 결과 출력 ────────────────────────────────────────────────────────
    out_all = OUT / "results_ALL.json"
    out_all.write_text(json.dumps(all_results, indent=2, default=str))

    W = 70
    print(f"\n{'═'*W}")
    print("  ★ CPU-GPU 파이프라이닝 실험 종합 결과")
    print(f"{'═'*W}")
    print(f"  {'실험':6}  {'항목':35}  {'결과':>12}")
    print(f"  {'-'*60}")

    if "experiment_E" in all_results:
        s = all_results["experiment_E"].get("sub_experiments", {})
        e1 = s.get("E1", {}).get("stats", {}).get("total_ms", {})
        e3 = s.get("E3", {})
        if e1:
            print(f"  {'[Exp E]':6}  {'전처리 총 시간 (mean)':35}  {e1.get('mean', '?'):>10.0f}ms")
        if isinstance(e3, dict) and "fully_hidden_all" in e3:
            hidden = "✅ 완전 hiding" if e3["fully_hidden_all"] else "⚠️ 부분"
            print(f"  {'':6}  {'GPU inference 내 hiding 여부':35}  {hidden:>12}")

    if "experiment_D" in all_results:
        d4 = all_results["experiment_D"].get("sub_experiments", {}).get("D4", {})
        if "net_gain_ms" in d4:
            verdict = "✅ 유리" if d4["net_gain_ms"] > 0 else "❌ 손해"
            print(f"  {'[Exp D]':6}  {'Net gain (VE 은닉 - Decode 저하)':35}  {d4['net_gain_ms']:>+10.0f}ms")
            print(f"  {'':6}  {'판정':35}  {verdict:>12}")
            print(f"  {'':6}  {'Overlap efficiency':35}  {d4.get('avg_overlap_efficiency_pct',0):>9.1f}%")

    if "experiment_F" in all_results:
        f4 = all_results["experiment_F"].get("sub_experiments", {}).get("F4", {})
        if "hiding_ms" in f4:
            print(f"  {'[Exp F]':6}  {'Flow CPU 시간':35}  {f4.get('flow_cpu_avg_ms','?'):>10.0f}ms")
            print(f"  {'':6}  {'hiding 가능 시간':35}  {f4['hiding_ms']:>10.0f}ms")
            print(f"  {'':6}  {'파이프라인 절약 비율':35}  {f4.get('saving_pct',0):>10.1f}%")

    print(f"{'─'*W}")
    print(f"  전체 결과 저장: {out_all}")
    print(f"{'═'*W}")


if __name__ == "__main__":
    main()
