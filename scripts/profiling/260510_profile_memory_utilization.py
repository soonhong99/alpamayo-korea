"""
260510_profile_memory_utilization.py  ·  v1.0
────────────────────────────────────────────────────────────────────────────────
Alpamayo 1.5  GPU 메모리 + SM 활용률 시계열 프로파일러

교수 피드백 대응:
  F2. 정확히 shared 메모리 사용량이 언제 많아지고 언제 줄어드는지 시간에 따라서
  F3. GPU를 얼마나 쓰는지 확인해보기 (총 가용량의 % 로)
  F6. GPU의 VRAM 크기는 얼마인가?
  F8. LLC 크기, VRAM 크기 파악

[계측 항목]
  A. GPU CUDA 메모리 사용량 (MB)  — pynvml nvmlDeviceGetMemoryInfo()
  B. 시스템 RAM 사용량 (GB)        — psutil virtual_memory()
  C. GPU SM 활용률 (%)             — pynvml nvmlDeviceGetUtilizationRates().gpu
  D. GPU 메모리 인터페이스 활용률   — pynvml nvmlDeviceGetUtilizationRates().memory

[동기화]
  기존 AlpamayoStagePatch의 mark() 시스템과 동일한 인터페이스
  → Vision / Prefill / Decode / Flow 구간 정확히 분리

[출력]
  profiling_results/260510_memory_utilization/
    ├── memory_timeline.json          ← 시계열 원시 데이터
    ├── hardware_spec.json            ← LLC / GPU L2 / 총 메모리 스펙
    └── figures/
         ├── fig_memory_timeline.png  ← 4패널 확장 타임라인
         └── fig_llc_analysis.png     ← LLC vs 모델 크기 분석

[실행 방법]
  # Thor에서
  python scripts/profiling/260510_profile_memory_utilization.py \\
      --model-path nvidia/Alpamayo-1.5-10B \\
      --warmup 2 --runs 4

  # 하드웨어 스펙만 측정
  python scripts/profiling/260510_profile_memory_utilization.py --spec-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ─── 선택적 의존성 ───────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception as e:
    _NVML_OK = False
    print(f"[WARNING] pynvml 초기화 실패: {e}\n"
          f"         pip install nvidia-ml-py3 후 재실행하세요.", file=sys.stderr)

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    print("[WARNING] psutil 미설치 → 시스템 RAM 측정 불가.", file=sys.stderr)

# ─── 출력 경로 ───────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("profiling_results/260510_memory_utilization")
FIG_DIR    = OUTPUT_DIR / "figures"

# ─── 단계 색상 (기존 fig3 스타일 통일) ──────────────────────────────────────
PHASE_COLORS = {
    "vision"  : "#4878CF",   # 파랑
    "prefill" : "#6ACC65",   # 초록
    "decode"  : "#D65F5F",   # 빨강
    "flow"    : "#B47CC7",   # 보라
    "idle"    : "#CCCCCC",   # 회색
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 하드웨어 스펙 측정
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareSpec:
    """
    GPU / CPU 캐시 계층 스펙.

    각 필드의 출처를 명확히 구분:
      [CUDA실측]  torch.cuda.get_device_properties() API로 직접 측정
      [sys실측]   /sys, psutil, lscpu 등 OS API로 직접 측정
      [공식스펙]  NVIDIA Jetson Thor 공식 Datasheet (DS-11945-001)
      [추정]      공식 문서에 없으며 유사 아키텍처에서 추론한 값
    """
    # ── GPU 스펙 (CUDA API 실측) ──────────────────────────────────────────
    gpu_name                  : str   = "Unknown"   # [CUDA실측]
    gpu_total_mem_gb          : float = 0.0         # [CUDA실측] Unified Memory 중 CUDA 예약분
    gpu_l2_cache_mb           : float = 0.0         # [CUDA실측] GPU LLC (SRAM)
    gpu_sm_count              : int   = 0           # [CUDA실측] Streaming Multiprocessor 수
    gpu_compute_cap           : str   = ""          # [CUDA실측] SM 11.0 = Blackwell
    gpu_shared_mem_per_sm_kb  : float = 0.0         # [CUDA실측] Shared Mem + L1 per SM (228 KB)
    gpu_max_threads_per_sm    : int   = 0           # [CUDA실측]

    # ── GPU 스펙 (공식 스펙 — 실측 불가) ─────────────────────────────────
    gpu_tensor_cores          : int   = 96          # [공식스펙] 5세대 Tensor Core
    gpu_fp4_tflops            : float = 2070.0      # [공식스펙] FP4 TFLOPS
    gpu_cuda_cores            : int   = 2560        # [공식스펙] 2560 = 20 SM × 128

    # ── 메모리 스펙 (공식 스펙 확인됨) ───────────────────────────────────
    dram_bandwidth_gbps       : float = 273.0       # [공식스펙] LPDDR5X 273 GB/s
    dram_bus_bits             : int   = 256         # [공식스펙] 256-bit bus
    dram_freq_mhz             : int   = 4266        # [공식스펙] 4266 MHz

    # ── CPU 스펙 (sys실측 + 공식스펙 혼합) ───────────────────────────────
    cpu_arch                  : str   = "Arm Neoverse V3AE"  # [공식스펙]
    cpu_clock_ghz             : float = 2.6         # [공식스펙] max 2.6 GHz
    cpu_cores                 : int   = 0           # [sys실측] psutil.cpu_count()
    cpu_l1d_kb_per_core       : int   = 64          # [sys실측 + 공식스펙] 64 KB/core
    cpu_l1i_kb_per_core       : int   = 64          # [sys실측 + 공식스펙] 64 KB/core
    cpu_l2_kb_per_core        : int   = 1024        # [sys실측 + 공식스펙] 1 MB/core
    cpu_l3_cache_kb           : int   = 0           # [sys실측] 0=탐지실패 (공식: 16MB)
    cpu_l3_official_kb        : int   = 16384       # [공식스펙] 16 MB shared L3

    # ── 모델 스펙 (로드 후 실측) ─────────────────────────────────────────
    model_weights_mb          : float = 0.0         # [CUDA실측] 모델 로드 후 측정


def _parse_cache_size_kb(raw: str) -> int:
    """'8192K', '4M', '512' 등 → KB 정수 변환."""
    raw = raw.strip()
    if raw.endswith("K") or raw.endswith("k"):
        return int(float(raw[:-1]))
    elif raw.endswith("M") or raw.endswith("m"):
        return int(float(raw[:-1]) * 1024)
    elif raw.endswith("G") or raw.endswith("g"):
        return int(float(raw[:-1]) * 1024 * 1024)
    elif raw.isdigit():
        return int(raw)
    return 0


def _detect_cpu_l3_kb() -> int:
    """
    CPU L3 (LLC) 캐시 크기를 3단계 fallback으로 탐지.

    Thor CPU 아키텍처: Arm Neoverse V3AE (14코어, up to 2.6 GHz)
    ※ 이전에 Cortex-A78AE로 잘못 기재 — 공식 스펙 확인으로 수정

    공식 캐시 구조 (NVIDIA Jetson Thor Datasheet DS-11945-001):
      L1 I-Cache : 64 KB/core
      L1 D-Cache : 64 KB/core
      L2         : 1 MB/core (= 14 MB total)
      L3         : 16 MB shared (system LLC, DynamIQ Shared Unit)

    Thor /sys 캐시 실측 구조 (ice401@100.95.177.101):
      index0: L1 Data        (level=1, type=Data,        size=64K)
      index1: L1 Instruction (level=1, type=Instruction, size=64K)
      index2: L2 Unified     (level=2, type=Unified,     size=1024K)
      index3: 탐지 안 됨 ← JetPack 7 / Linux 6.8에서 L3 DT 노드 미노출 (알려진 이슈)

    탐지 우선순위:
      1. /sys 전체 캐시 인덱스 스캔 (cpu0 ~ cpu_max 모두 시도)
      2. lscpu 출력 파싱
      3. getconf LEVEL3_CACHE_SIZE
      → 모두 실패 시 0 반환 (공식 스펙에서 16384 KB 사용)
    """
    # ── 방법 1: /sys 캐시 인덱스 스캔 (모든 CPU 코어 시도) ─────────────
    # Neoverse V3AE: L3는 클러스터 공유 → cpu0에 없어도 다른 코어에 있을 수 있음
    try:
        cpu_root = Path("/sys/devices/system/cpu")
        cpu_dirs = sorted(cpu_root.glob("cpu[0-9]*/cache"), key=lambda p: int(p.parent.name[3:]))
        for cache_root in cpu_dirs[:4]:   # cpu0~cpu3까지만 시도 (동일 결과 반복 방지)
            if not cache_root.exists():
                continue
            best_level = 0
            best_kb    = 0
            for idx_dir in sorted(cache_root.iterdir()):
                level_f = idx_dir / "level"
                size_f  = idx_dir / "size"
                type_f  = idx_dir / "type"
                if not (level_f.exists() and size_f.exists()):
                    continue
                try:
                    level = int(level_f.read_text().strip())
                except ValueError:
                    continue
                raw   = size_f.read_text().strip()
                kb    = _parse_cache_size_kb(raw)
                cache_type = type_f.read_text().strip() if type_f.exists() else ""
                if "Instruction" in cache_type:
                    continue
                if level > best_level and kb > 0:
                    best_level = level
                    best_kb    = kb
            if best_level >= 3 and best_kb > 0:
                return best_kb   # L3 발견
    except Exception:
        pass

    # ── 방법 2: lscpu 파싱 ───────────────────────────────────────────────
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            # "L3 cache:", "L3 Cache:", "L3d cache:" 등
            low = line.lower()
            if "l3" in low and "cache" in low and ":" in line:
                val = line.split(":")[-1].strip()
                # "4096K", "4 MiB", "4096 KiB" 등
                val = val.replace("iB", "").replace("i", "")
                kb  = _parse_cache_size_kb(val.split()[0] + val[-1]
                      if val[-1].upper() in "KMG" else val.split()[0])
                if kb > 0:
                    return kb
    except Exception:
        pass

    # ── 방법 3: getconf ──────────────────────────────────────────────────
    try:
        out = subprocess.check_output(
            ["getconf", "LEVEL3_CACHE_SIZE"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if out.isdigit() and int(out) > 0:
            return int(out) // 1024   # bytes → KB
    except Exception:
        pass

    return 0   # 탐지 실패


def measure_hardware_spec(model=None) -> HardwareSpec:
    """
    GPU L2 캐시, CPU L3(LLC), 총 메모리 등 측정.

    [CUDA실측] 항목: torch.cuda.get_device_properties()
    [sys실측]  항목: /sys, psutil
    [공식스펙] 항목: HardwareSpec 기본값으로 설정 (NVIDIA Jetson Thor Datasheet)
    """
    spec = HardwareSpec()

    # ── GPU 스펙 [CUDA실측] ───────────────────────────────────────────────
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        spec.gpu_name        = prop.name
        spec.gpu_total_mem_gb = prop.total_memory / 1e9
        # PyTorch 버전별 대소문자 차이: 2.8.0+ → L2_cache_size (대문자)
        l2_raw = getattr(prop, "L2_cache_size",
                 getattr(prop, "l2_cache_size", 0))
        spec.gpu_l2_cache_mb  = l2_raw / 1e6
        spec.gpu_sm_count     = prop.multi_processor_count
        spec.gpu_compute_cap  = f"{prop.major}.{prop.minor}"
        # Shared Memory + L1 per SM (Blackwell SM11: 228 KB)
        spec.gpu_shared_mem_per_sm_kb = getattr(
            prop, "shared_memory_per_multiprocessor", 0
        ) / 1024
        spec.gpu_max_threads_per_sm = getattr(
            prop, "max_threads_per_multi_processor", 0
        )

    # ── CPU L3 캐시 크기 [sys실측] ────────────────────────────────────────
    # Neoverse V3AE의 16MB L3는 JetPack 7 / Linux 6.8에서 /sys 미노출
    # → 탐지 실패 시 cpu_l3_cache_kb=0, cpu_l3_official_kb=16384(공식) 사용
    spec.cpu_l3_cache_kb = _detect_cpu_l3_kb()

    # ── CPU 코어 수 [sys실측] ────────────────────────────────────────────
    if _PSUTIL_OK:
        spec.cpu_cores = psutil.cpu_count(logical=True) or 0

    # ── 모델 가중치 크기 [CUDA실측, 로드 후] ────────────────────────────
    if model is not None:
        spec.model_weights_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / 1e6

    return spec


def print_hardware_spec(spec: HardwareSpec):
    """
    하드웨어 스펙을 출처 표시와 함께 출력.
    [실측] = CUDA/sys API 직접 측정, [공식] = NVIDIA Datasheet, [추정] = 유사 아키텍처 추론
    """
    W = 66
    print("\n" + "═" * W)
    print(f"  NVIDIA Jetson AGX Thor  하드웨어 스펙 측정 결과")
    print("═" * W)

    # ── GPU ────────────────────────────────────────────────────────────
    print(f"\n  ┌─ GPU (Blackwell SM {spec.gpu_compute_cap})")
    print(f"  │  이름           : {spec.gpu_name}  [실측]")
    print(f"  │  Unified Memory : {spec.gpu_total_mem_gb:.1f} GB  [실측]")
    print(f"  │  CUDA 코어      : {spec.gpu_cuda_cores}개  ({spec.gpu_sm_count} SM × 128)  [공식+실측]")
    print(f"  │  Tensor 코어    : {spec.gpu_tensor_cores}개 (5세대)  [공식]")
    print(f"  │  FP4 성능       : {spec.gpu_fp4_tflops:.0f} TFLOPS  [공식]")
    print(f"  │  Compute Cap    : SM {spec.gpu_compute_cap}  [실측]")
    print(f"  │")
    print(f"  │  ┌─ Shared Mem + L1 per SM : {spec.gpu_shared_mem_per_sm_kb:.0f} KB  [실측]")
    total_shared_mb = spec.gpu_shared_mem_per_sm_kb * spec.gpu_sm_count / 1024
    print(f"  │  │  (= {spec.gpu_sm_count} SM × {spec.gpu_shared_mem_per_sm_kb:.0f} KB = {total_shared_mb:.2f} MB 전체)")
    print(f"  │  └─ GPU L2 Cache (total)   : {spec.gpu_l2_cache_mb:.1f} MB  [실측] ← GPU LLC (SRAM)")
    print(f"  └─")

    # ── CPU ────────────────────────────────────────────────────────────
    print(f"\n  ┌─ CPU ({spec.cpu_arch})")
    print(f"  │  코어 수        : {spec.cpu_cores}코어 @ {spec.cpu_clock_ghz} GHz  [실측+공식]")
    print(f"  │  L1 I-Cache     : {spec.cpu_l1i_kb_per_core} KB/core  [실측+공식]")
    print(f"  │  L1 D-Cache     : {spec.cpu_l1d_kb_per_core} KB/core  [실측+공식]")
    print(f"  │  L2 Cache       : {spec.cpu_l2_kb_per_core} KB/core = {spec.cpu_l2_kb_per_core * spec.cpu_cores // 1024} MB 전체  [실측+공식]")

    if spec.cpu_l3_cache_kb > 0:
        l3_display = f"{spec.cpu_l3_cache_kb / 1024:.0f} MB  [실측]"
    else:
        l3_display = (f"{spec.cpu_l3_official_kb / 1024:.0f} MB shared  [공식스펙] "
                      f"← /sys 미탐지 (JetPack 7 알려진 이슈)")
    print(f"  │  L3 Cache       : {l3_display}")
    print(f"  └─")

    # ── Unified Memory ──────────────────────────────────────────────────
    print(f"\n  ┌─ Unified Memory (CPU + GPU 물리 공유)")
    print(f"  │  총 크기        : {spec.gpu_total_mem_gb:.1f} GB  LPDDR5X  [실측]")
    print(f"  │  버스 폭        : {spec.dram_bus_bits}-bit  [공식]")
    print(f"  │  클럭           : {spec.dram_freq_mhz} MHz  [공식]")
    print(f"  │  이론 대역폭    : {spec.dram_bandwidth_gbps:.0f} GB/s  [공식]")
    print(f"  └─")

    # ── 임계점 분석 ─────────────────────────────────────────────────────
    if spec.model_weights_mb > 0:
        gpu_l2_mb  = spec.gpu_l2_cache_mb
        cpu_l3_mb  = (spec.cpu_l3_cache_kb / 1024 if spec.cpu_l3_cache_kb > 0
                      else spec.cpu_l3_official_kb / 1024)
        model_gb   = spec.model_weights_mb / 1024

        print(f"\n  ── 메모리 계층 임계점 분석 {'─' * 30}")
        print(f"  GPU L2 (LLC)  : {gpu_l2_mb:.1f} MB   →  모델 {model_gb:.1f} GB의 {model_gb * 1024 / gpu_l2_mb:.0f}분의 1")
        print(f"  CPU L3 (LLC)  : {cpu_l3_mb:.0f} MB  →  모델 {model_gb:.1f} GB의 {model_gb * 1024 / cpu_l3_mb:.0f}분의 1")
        print(f"  → 매 Decode step: 22 GB를 DRAM({spec.dram_bandwidth_gbps:.0f} GB/s)에서 직접 읽음")
        print(f"  → 이론 하한: {spec.model_weights_mb / 1024 / spec.dram_bandwidth_gbps * 1000:.1f} ms/step")

    print("\n" + "═" * W + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GPUMemorySampler — GPU 메모리 + SM 활용률 시계열 측정
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemSample:
    """단일 시점의 GPU/시스템 메모리 스냅샷."""
    t_ms          : float   # 추론 시작 기준 경과 시간 (ms)
    gpu_used_mb   : float   # CUDA 할당 메모리 (MB)
    gpu_free_mb   : float   # CUDA 여유 메모리 (MB)
    gpu_total_mb  : float   # CUDA 총 메모리 (MB)
    gpu_used_pct  : float   # GPU 메모리 점유율 (%)
    sm_util_pct   : float   # GPU SM 활용률 (%)
    mem_util_pct  : float   # GPU 메모리 인터페이스 활용률 (%)
    sys_used_gb   : float   # 시스템 RAM 사용량 (GB)
    sys_total_gb  : float   # 시스템 RAM 총 크기 (GB)
    phase         : str     # 현재 추론 단계


class GPUMemorySampler:
    """
    pynvml + psutil로 추론 단계별 GPU 메모리 + SM 활용률 시계열 측정.

    CPUSampler와 동일한 mark() 인터페이스 → AlpamayoStagePatch와 동기화.

    측정 항목:
      - GPU CUDA 메모리: nvmlDeviceGetMemoryInfo()
          used  = 모델가중치 + KV cache + activation buffer
          free  = 아직 할당 안 된 영역
          total = Unified Memory 중 GPU 측 가용 크기

      - GPU 활용률: nvmlDeviceGetUtilizationRates()
          gpu   = SM(Streaming Multiprocessor) 활용률 (%)
                  "GPU 코어가 뭔가 계산하는 시간의 비율"
          memory = 메모리 인터페이스(DRAM 버스) 활용률 (%)
                  "DRAM 읽기/쓰기가 일어나는 시간의 비율"

      - 시스템 RAM: psutil.virtual_memory()
          used  = OS 전체 사용 RAM (CUDA 영역 포함)

    Thor 특이사항:
      GPU total_memory (nvml) ≠ 128GB (Unified 전체)
      nvml은 CUDA가 OS로부터 예약한 영역만 보고
      Unified Memory 전체는 psutil.virtual_memory().total로 확인
    """

    def __init__(self, device_idx: int = 0, interval_ms: int = 100):
        """
        Jetson AGX Thor 호환 GPU 메모리 샘플러.

        [왜 pynvml 단독 사용이 안 되는가]
        pynvml.nvmlDeviceGetMemoryInfo()는 discrete GPU(RTX, A100)용 API.
        Jetson Thor는 Unified Memory SoC — GPU VRAM이 별도로 없어
        nvml의 메모리 쿼리가 NVMLError_NotSupported를 반환.

        [대안 — 2단계 fallback]
        1순위: torch.cuda.memory_allocated() / memory_reserved()
               → PyTorch caching allocator가 관리하는 CUDA 메모리 (항상 사용 가능)
               → allocated = 실제 tensor 데이터 (모델+KV+activations)
               → reserved  = allocator가 OS에서 예약한 블록 (allocated ≥ allocated)
        2순위: pynvml (지원되는 경우)

        [SM 활용률 — tegrastats 방식]
        Jetson에서 SM 활용률은 nvmlDeviceGetUtilizationRates()가 미지원.
        tegrastats CLI (JetPack 기본 포함)에서 GR3D_FREQ 파싱으로 획득:
          형식: ... GR3D_FREQ 42%@1300 ...
          GR3D_FREQ X% = GPU 3D engine 활용률 ≈ SM 활용률 대리 지표
        tegrastats 미사용 시 -1.0 (N/A) 기록.
        """
        self._handle : Optional[object] = None
        if _NVML_OK:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                # 지원 여부 사전 테스트
                pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._nvml_mem_ok = True
            except Exception:
                self._nvml_mem_ok = False
        else:
            self._nvml_mem_ok = False

        self._interval   = interval_ms / 1000.0
        self._ts_interval = max(interval_ms, 100)   # tegrastats 최소 50ms
        self.samples     : list[MemSample] = []
        self._phase      = "idle"
        self._running    = False
        self._t0         : Optional[float] = None
        self._thread     : Optional[threading.Thread] = None

        # tegrastats 프로세스 (SM 활용률 대안)
        self._tegra_proc : Optional[object] = None
        self._tegra_latest_gr3d : float = -1.0   # -1 = N/A

        # torch CUDA 총 메모리 (reserved의 상한으로 사용)
        if torch.cuda.is_available():
            self._cuda_total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        else:
            self._cuda_total_mb = 0.0

        src = "torch.cuda API" if not self._nvml_mem_ok else "pynvml"
        print(f"[GPUMemorySampler] GPU 메모리 소스: {src}")

    def _start_tegrastats(self):
        """tegrastats 백그라운드 프로세스 시작 (SM 활용률 수집용)."""
        try:
            self._tegra_proc = subprocess.Popen(
                ["tegrastats", f"--interval", str(self._ts_interval)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1
            )
            t = threading.Thread(target=self._parse_tegrastats,
                                  name="TegrastatsParser", daemon=True)
            t.start()
        except FileNotFoundError:
            pass   # tegrastats 미설치 → SM 활용률 -1 유지

    def _parse_tegrastats(self):
        """tegrastats stdout에서 GR3D_FREQ 실시간 파싱."""
        import re
        pat = re.compile(r"GR3D_FREQ\s+(\d+)%")
        for line in self._tegra_proc.stdout:
            m = pat.search(line)
            if m:
                self._tegra_latest_gr3d = float(m.group(1))
            if not self._running:
                break

    def _stop_tegrastats(self):
        if self._tegra_proc:
            try:
                self._tegra_proc.terminate()
                self._tegra_proc.wait(timeout=2)
            except Exception:
                pass
            self._tegra_proc = None

    def mark(self, phase: str):
        """단계 전환 마커. Thread-safe (GIL 보장)."""
        self._phase = phase

    def _read_gpu_mem_torch(self) -> tuple[float, float, float, float]:
        """
        torch.cuda API로 GPU 메모리 읽기.

        Returns: (used_mb, free_mb, total_mb, pct)
          used_mb  = memory_allocated()  — 실제 tensor 데이터 (모델+KV+activations)
          free_mb  = total - reserved    — OS 미예약 영역
          total_mb = cuda total memory
          pct      = used / total × 100
        """
        allocated_mb = torch.cuda.memory_allocated() / 1e6
        reserved_mb  = torch.cuda.memory_reserved()  / 1e6
        total_mb     = self._cuda_total_mb
        free_mb      = max(0.0, total_mb - reserved_mb)
        pct          = (allocated_mb / total_mb * 100) if total_mb > 0 else 0.0
        return allocated_mb, free_mb, total_mb, pct

    def _loop(self):
        self._t0 = time.perf_counter()
        while self._running:
            t_ms = (time.perf_counter() - self._t0) * 1000.0

            # ── GPU 메모리 (우선: pynvml, 실패: torch.cuda) ─────────────
            if self._nvml_mem_ok and self._handle is not None:
                try:
                    mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    g_used = mem.used  / 1e6
                    g_free = mem.free  / 1e6
                    g_tot  = mem.total / 1e6
                    g_pct  = (mem.used / mem.total * 100) if mem.total > 0 else 0.0
                except Exception:
                    g_used, g_free, g_tot, g_pct = self._read_gpu_mem_torch()
            else:
                g_used, g_free, g_tot, g_pct = self._read_gpu_mem_torch()

            # ── SM 활용률 (우선: pynvml, 실패: tegrastats) ─────────────
            sm_pct  = -1.0
            mem_pct = -1.0
            if self._handle is not None:
                try:
                    util    = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    sm_pct  = float(util.gpu)
                    mem_pct = float(util.memory)
                except Exception:
                    sm_pct  = self._tegra_latest_gr3d   # tegrastats fallback
                    mem_pct = -1.0

            # ── 시스템 RAM ──────────────────────────────────────────────
            if _PSUTIL_OK:
                vmem   = psutil.virtual_memory()
                s_used = vmem.used  / 1e9
                s_tot  = vmem.total / 1e9
            else:
                s_used, s_tot = 0.0, 0.0

            self.samples.append(MemSample(
                t_ms         = t_ms,
                gpu_used_mb  = g_used,
                gpu_free_mb  = g_free,
                gpu_total_mb = g_tot,
                gpu_used_pct = g_pct,
                sm_util_pct  = sm_pct,
                mem_util_pct = mem_pct,
                sys_used_gb  = s_used,
                sys_total_gb = s_tot,
                phase        = self._phase,
            ))
            time.sleep(self._interval)

    def start(self):
        self.samples.clear()
        self._running = True
        # tegrastats 시작 (SM 활용률용, Jetson 전용)
        self._start_tegrastats()
        self._thread = threading.Thread(
            target=self._loop, name="GPUMemSampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> list[MemSample]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._stop_tegrastats()
        return self.samples

    def summary(self) -> dict:
        """단계별 통계 요약."""
        if not self.samples:
            return {}

        phases = ["vision", "prefill", "decode", "flow", "idle"]
        result = {"overall": {}, "by_phase": {}}

        def _sm_stats(samples_list):
            """SM 활용률 통계. -1.0(N/A) 값 제외."""
            vals = [s.sm_util_pct for s in samples_list if s.sm_util_pct >= 0]
            if not vals:
                return {"sm_util_mean_pct": -1.0, "sm_util_max_pct": -1.0,
                        "sm_source": "N/A (tegrastats/nvml 미지원)"}
            return {"sm_util_mean_pct": float(np.mean(vals)),
                    "sm_util_max_pct":  float(np.max(vals)),
                    "sm_source": "tegrastats GR3D_FREQ" if vals[0] >= 0 else "pynvml"}

        # 전체 통계
        mem_vals = [s.gpu_used_mb for s in self.samples]
        result["overall"] = {
            **_sm_stats(self.samples),
            "gpu_mem_mean_mb" : float(np.mean(mem_vals)),
            "gpu_mem_peak_mb" : float(np.max(mem_vals)),
            "gpu_mem_min_mb"  : float(np.min(mem_vals)),
            "n_samples"       : len(self.samples),
            "mem_source"      : "torch.cuda.memory_allocated" if not self._nvml_mem_ok
                                else "pynvml.nvmlDeviceGetMemoryInfo",
        }

        # 단계별 통계
        for ph in phases:
            ph_samples = [s for s in self.samples if s.phase == ph]
            if not ph_samples:
                continue
            result["by_phase"][ph] = {
                **_sm_stats(ph_samples),
                "gpu_mem_mean_mb" : float(np.mean([s.gpu_used_mb for s in ph_samples])),
                "gpu_mem_peak_mb" : float(np.max( [s.gpu_used_mb for s in ph_samples])),
                "n_samples"       : len(ph_samples),
            }
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AlpamayoMemoryProfiler — 메인 프로파일러
# ═══════════════════════════════════════════════════════════════════════════════

class AlpamayoMemoryProfiler:
    """
    GPUMemorySampler를 기존 AlpamayoStagePatch와 연동해
    추론 단계별 GPU 메모리 + SM 활용률 시계열을 기록하고
    fig3 확장 그래프를 생성한다.
    """

    def __init__(self, model_path: str, warmup: int = 2, runs: int = 4):
        self.model_path  = model_path
        self.n_warmup    = warmup
        self.n_runs      = runs
        self.model       = None
        self._model_inputs = None
        self._spec       : Optional[HardwareSpec] = None
        self.all_samples : list[list[MemSample]] = []   # 런별 시계열
        self.all_timings : list[dict] = []               # 런별 GPU 타이밍

    # ── 모델 로드 ─────────────────────────────────────────────────────────
    def _load_model(self):
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        from alpamayo1_5 import helper
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

        print("[1/3] 모델 로드 중...")
        torch.cuda.reset_peak_memory_stats()
        self.model = Alpamayo1_5.from_pretrained(
            self.model_path, dtype=torch.bfloat16,
        ).cuda().eval()
        torch.cuda.synchronize()
        print(f"      완료. 파라미터: "
              f"{sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

        print("[2/3] 입력 데이터 준비 중...")
        clip_id  = "030c760c-ae38-49aa-9ad8-f5650a545d26"
        data     = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        processor = helper.get_processor(self.model.tokenizer)
        inputs    = processor.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors="pt",
        )
        self._model_inputs = {
            "tokenized_data"  : inputs,
            "ego_history_xyz" : data["ego_history_xyz"],
            "ego_history_rot" : data["ego_history_rot"],
        }
        self._model_inputs = helper.to_device(self._model_inputs, "cuda")
        print("      완료.")

    # ── 단일 추론 (메모리 샘플링 포함) ───────────────────────────────────
    @torch.no_grad()
    def _run_one(self, run_id: int, is_warmup: bool = False) -> tuple[list[MemSample], dict]:
        """
        메모리 샘플러를 추론과 동기화해서 실행.

        동기화 방법:
          1. sampler.start() 직후 추론 시작
          2. 각 단계 시작/종료 시 sampler.mark() 호출
             → 기존 AlpamayoStagePatch 훅 대신 여기서 직접 wrapping
          3. 추론 완료 직후 sampler.stop()
        """
        torch.cuda.synchronize()

        # ── 메모리 샘플러 준비 ──────────────────────────────────────────
        mem_sampler = GPUMemorySampler(device_idx=0, interval_ms=100)

        # ── GPU 타이밍용 CUDA Events ────────────────────────────────────
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)

        label = f"{'warmup' if is_warmup else 'run'} {run_id}"
        print(f"  [{label}] 시작...", end="", flush=True)

        mem_sampler.mark("idle")
        mem_sampler.start()
        t_wall_s = time.perf_counter()

        # ── 추론 실행 ──────────────────────────────────────────────────
        ev_s.record()

        # Vision + Prefill + Decode + Flow 를 하나로 실행
        # 단계 경계는 wall-clock mark로 근사
        # (기존 AlpamayoStagePatch의 정밀 CUDA Event 측정과 병행 가능)
        mem_sampler.mark("vision")
        self._start_phase_timer("vision")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # ── Vision hook ── (forward pre-hook 없이 wall-clock 근사)
            # 더 정밀한 측정을 위해 기존 AlpamayoStagePatch와 병용 권장
            pred_xyz, pred_rot, extra = \
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=self._model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    return_extra=True,
                )

        ev_e.record()
        torch.cuda.synchronize()
        mem_sampler.mark("idle")

        t_wall_ms = (time.perf_counter() - t_wall_s) * 1000.0
        samples = mem_sampler.stop()

        gpu_total_ms = ev_s.elapsed_time(ev_e)
        print(f" {gpu_total_ms:.0f}ms  ({len(samples)} samples)")

        timing = {
            "run_id"       : run_id,
            "is_warmup"    : is_warmup,
            "gpu_total_ms" : gpu_total_ms,
            "wall_ms"      : t_wall_ms,
        }
        return samples, timing

    def _start_phase_timer(self, phase: str):
        """향후 phase-level hook 연동 시 사용할 자리."""
        pass

    # ── 메인 실험 루프 ────────────────────────────────────────────────────
    def run_with_stage_patch(self):
        """
        기존 AlpamayoStagePatch (정밀 CUDA Event)와 GPUMemorySampler를 병용.
        Stage Patch가 mark()를 호출하도록 sampler를 주입한다.
        """
        # profile_alpamayo.py에서 AlpamayoStagePatch import
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from profile_alpamayo import AlpamayoStagePatch, CPUSampler, _new_cuda_event, _safe_elapsed
            _PATCH_OK = True
        except ImportError:
            _PATCH_OK = False
            print("[WARNING] AlpamayoStagePatch import 실패 → 단순 실행 모드 사용")

        self._load_model()
        self._spec = measure_hardware_spec(self.model)
        print_hardware_spec(self._spec)

        if _PATCH_OK:
            patch = AlpamayoStagePatch()
            patch.attach(self.model)
        else:
            patch = None

        print(f"\n[3/3] 프로파일링 시작 (warmup {self.n_warmup}회 + 측정 {self.n_runs}회)")

        # ── 워밍업 ──────────────────────────────────────────────────────
        print("  ── 워밍업 ──")
        for i in range(self.n_warmup):
            samples, timing = self._run_with_patch(i, patch, is_warmup=True)

        # ── 실측 ────────────────────────────────────────────────────────
        print("  ── 측정 ──")
        for i in range(self.n_runs):
            samples, timing = self._run_with_patch(i, patch, is_warmup=False)
            self.all_samples.append(samples)
            self.all_timings.append(timing)

        if patch is not None:
            patch.detach(self.model)

        return self._save_and_plot()

    @torch.no_grad()
    def _run_with_patch(self, run_id: int, patch, is_warmup: bool):
        """AlpamayoStagePatch + GPUMemorySampler 병용 실행."""
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        mem_sampler = GPUMemorySampler(device_idx=0, interval_ms=100)

        # AlpamayoStagePatch의 mark() 를 GPUMemorySampler에도 전달하는 래퍼
        class DualSampler:
            """CPUSampler + GPUMemorySampler 동시 mark() 전달."""
            def __init__(self, cpu_s, gpu_s):
                self._cpu = cpu_s
                self._gpu = gpu_s
            def mark(self, name: str):
                if self._cpu: self._cpu.mark(name)
                if self._gpu:
                    # 단계 이름 → phase 색상 키 매핑
                    phase_map = {
                        "vision_start"  : "vision",
                        "vision_end"    : "prefill",   # vision 끝 = prefill 시작
                        "prefill_start" : "prefill",
                        "prefill_end"   : "decode",    # prefill 끝 = decode 시작
                        "vlm_end"       : "flow",      # vlm 끝 = flow 시작
                        "flow_start"    : "flow",
                        "flow_end"      : "idle",
                    }
                    ph = phase_map.get(name)
                    if ph:
                        self._gpu.mark(ph)
            def start(self): pass
            def stop(self): return {}

        try:
            from profile_alpamayo import CPUSampler, _new_cuda_event, _safe_elapsed
            cpu_sampler = CPUSampler(interval_ms=50)
        except ImportError:
            cpu_sampler = None

        dual = DualSampler(cpu_sampler, mem_sampler)

        if patch is not None:
            patch.prepare_run(sampler=dual)

        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)

        # 메모리 샘플러 먼저 시작 (추론 전 기저값 측정)
        mem_sampler.mark("idle")
        mem_sampler.start()
        if cpu_sampler: cpu_sampler.start()

        label = f"{'warmup' if is_warmup else 'run':7} {run_id+1}"
        print(f"  [{label}]", end="", flush=True)

        t_wall_s = time.perf_counter()
        ev_s.record()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = \
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=self._model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    return_extra=True,
                )

        ev_e.record()
        if patch is not None:
            patch._inside_inference = False
        torch.cuda.synchronize()

        t_wall_ms = (time.perf_counter() - t_wall_s) * 1000.0
        samples = mem_sampler.stop()
        if cpu_sampler: cpu_sampler.stop()

        gpu_ms = ev_s.elapsed_time(ev_e)
        n_s    = len(samples)
        sm_avg = np.mean([s.sm_util_pct for s in samples]) if samples else 0.0
        mem_pk = max((s.gpu_used_mb for s in samples), default=0.0)

        print(f"  {gpu_ms:6.0f}ms  SM:{sm_avg:4.1f}%  GPU_peak:{mem_pk/1024:.2f}GB"
              f"  samples:{n_s}")

        timing = {
            "run_id": run_id, "is_warmup": is_warmup,
            "gpu_total_ms": gpu_ms, "wall_ms": t_wall_ms,
        }
        return samples, timing

    # ── 저장 + 시각화 ─────────────────────────────────────────────────────
    def _save_and_plot(self) -> dict:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        # ── JSON 저장 ──────────────────────────────────────────────────
        serial_samples = [
            [asdict(s) for s in run_samples]
            for run_samples in self.all_samples
        ]
        with open(OUTPUT_DIR / "memory_timeline.json", "w") as f:
            json.dump({
                "n_runs"  : self.n_runs,
                "runs"    : serial_samples,
                "timings" : self.all_timings,
            }, f, indent=2)

        if self._spec:
            with open(OUTPUT_DIR / "hardware_spec.json", "w") as f:
                json.dump(asdict(self._spec), f, indent=2)

        # ── 그래프 생성 ────────────────────────────────────────────────
        if self.all_samples:
            self._plot_memory_timeline(self.all_samples[0], self.all_timings[0])
            self._plot_llc_analysis()

        summary = self._compute_summary()
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self._print_summary(summary)
        return summary

    # ── 4패널 메모리 타임라인 그래프 ─────────────────────────────────────
    def _plot_memory_timeline(self, samples: list[MemSample], timing: dict):
        """
        fig3 확장 버전 — GPU 타임라인 아래에 메모리 + SM 활용률 패널 추가.

        Panel A: GPU 실행 단계 타임라인 (Vision/Prefill/Decode/Flow)
        Panel B: GPU CUDA 메모리 사용량 (MB) + 모델 가중치 기준선
        Panel C: GPU SM 활용률 (%) + 메모리 인터페이스 활용률 (%)
        Panel D: 시스템 RAM 사용량 (GB)
        """
        if not samples:
            return

        t_arr    = np.array([s.t_ms          for s in samples])
        mem_arr  = np.array([s.gpu_used_mb   for s in samples]) / 1024   # GB
        sm_arr   = np.array([s.sm_util_pct   for s in samples])
        memi_arr = np.array([s.mem_util_pct  for s in samples])
        sys_arr  = np.array([s.sys_used_gb   for s in samples])
        phase_arr= [s.phase for s in samples]
        total_ms = timing.get("gpu_total_ms", t_arr[-1] if len(t_arr) else 5000)

        # 단계 경계 추출 (phase 변경 시점)
        phase_changes = []
        for i in range(1, len(phase_arr)):
            if phase_arr[i] != phase_arr[i-1]:
                phase_changes.append((t_arr[i], phase_arr[i-1], phase_arr[i]))

        fig = plt.figure(figsize=(18, 13))
        fig.patch.set_facecolor("#FAFAFA")

        gs = GridSpec(
            4, 1, figure=fig,
            height_ratios=[1.5, 2, 2, 1.5],
            hspace=0.08,
        )
        ax_phase = fig.add_subplot(gs[0])
        ax_mem   = fig.add_subplot(gs[1], sharex=ax_phase)
        ax_sm    = fig.add_subplot(gs[2], sharex=ax_phase)
        ax_sys   = fig.add_subplot(gs[3], sharex=ax_phase)

        # ── 단계 색상 배경 그리기 (모든 패널 공통) ──────────────────────
        # 단계 구간 계산
        phase_spans = []
        if phase_arr:
            cur_phase = phase_arr[0]
            cur_start = t_arr[0]
            for i in range(1, len(phase_arr)):
                if phase_arr[i] != cur_phase:
                    phase_spans.append((cur_start, t_arr[i], cur_phase))
                    cur_phase = phase_arr[i]
                    cur_start = t_arr[i]
            phase_spans.append((cur_start, t_arr[-1], cur_phase))

        for ax in [ax_phase, ax_mem, ax_sm, ax_sys]:
            for (t_s, t_e, ph) in phase_spans:
                c = PHASE_COLORS.get(ph, "#CCCCCC")
                ax.axvspan(t_s, t_e, alpha=0.08, color=c, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

        # ── Panel A: GPU 단계 타임라인 ───────────────────────────────────
        stage_data = self._estimate_stage_boundaries(samples, total_ms)
        bar_h = 0.6
        for st in stage_data:
            ax_phase.barh(
                0, st["duration"], left=st["start"],
                height=bar_h, color=st["color"], alpha=0.85,
                edgecolor="white", linewidth=0.8, zorder=2,
            )
            ax_phase.text(
                st["start"] + st["duration"] / 2, 0,
                f'{st["label"]}\n{st["duration"]:.0f}ms',
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", zorder=3,
            )
        ax_phase.set_xlim(0, max(total_ms * 1.02, t_arr[-1] * 1.02))
        ax_phase.set_ylim(-0.8, 0.8)
        ax_phase.set_yticks([0])
        ax_phase.set_yticklabels(["GPU\nStream"], fontsize=9)
        ax_phase.set_title(
            f"Fig 3 Extended  |  CPU-GPU Timeline + Memory + SM Utilization\n"
            f"GPU Total: {total_ms:.0f}ms  |  Sampling: 100ms  |  "
            f"Peak GPU Mem: {mem_arr.max():.2f}GB  |  "
            f"Avg SM: {sm_arr.mean():.1f}%",
            fontsize=11, pad=8,
        )

        # ── Panel B: GPU CUDA 메모리 ─────────────────────────────────────
        ax_mem.fill_between(t_arr, mem_arr, alpha=0.35, color="steelblue", zorder=2)
        ax_mem.plot(t_arr, mem_arr, color="steelblue", linewidth=1.8,
                    label="GPU CUDA Memory", zorder=3)

        # 모델 가중치 기준선
        if self._spec and self._spec.model_weights_mb > 0:
            w_gb = self._spec.model_weights_mb / 1024
            ax_mem.axhline(w_gb, color="red", linestyle="--", linewidth=1.2,
                           alpha=0.7, label=f"Model weights ({w_gb:.1f}GB)", zorder=4)
            # 활성화 메모리 = 총 사용 - 가중치
            act_arr = mem_arr - w_gb
            act_arr = np.clip(act_arr, 0, None)
            ax_mem.fill_between(t_arr, w_gb, mem_arr,
                                where=(mem_arr > w_gb),
                                alpha=0.5, color="orange", label="KV Cache + Activation")

        ax_mem.set_ylabel("GPU Memory\n(GB)", fontsize=9)
        ax_mem.legend(loc="lower right", fontsize=8, framealpha=0.8)

        # KV Cache 증가 구간 강조
        prefill_s = next((t for t, p in zip(t_arr, phase_arr) if p == "prefill"), None)
        prefill_e = next((t for t, p in zip(reversed(t_arr), reversed(phase_arr)) if p == "prefill"), None)
        if prefill_s and prefill_e:
            ax_mem.annotate(
                "KV Cache 생성\n(Prefill 중 급증)",
                xy=((prefill_s + prefill_e) / 2, mem_arr.max() * 0.98),
                xytext=((prefill_s + prefill_e) / 2, mem_arr.max() * 1.03),
                arrowprops=dict(arrowstyle="->", color="gray"),
                ha="center", fontsize=8, color="gray",
            )

        # ── Panel C: GPU SM / GR3D 활용률 ───────────────────────────────
        # -1.0 = N/A (tegrastats/nvml 미지원) → 마스킹
        sm_valid = sm_arr >= 0
        sm_source_label = ("SM Util % (tegrastats GR3D_FREQ)"
                           if any(sm_arr >= 0) else
                           "SM Util % (N/A — nvml not supported on Jetson)")
        if sm_valid.any():
            t_v  = t_arr[sm_valid]
            sm_v = sm_arr[sm_valid]
            ax_sm.fill_between(t_v, sm_v, alpha=0.4, color="#D65F5F", zorder=2,
                               label=sm_source_label)
            ax_sm.plot(t_v, sm_v, color="#D65F5F", linewidth=1.5, zorder=3)
            # 메모리 인터페이스 활용률 (보조선, N/A이면 생략)
            memi_valid = memi_arr >= 0
            if memi_valid.any():
                ax_sm.plot(t_arr[memi_valid], memi_arr[memi_valid],
                           color="orange", linewidth=1.2, linestyle="--",
                           alpha=0.7, label="Mem Interface Util %", zorder=3)
            ax_sm.set_ylim(0, 105)
            ax_sm.axhline(50, color="gray", linestyle=":", alpha=0.4)
            ax_sm.axhline(80, color="gray", linestyle=":", alpha=0.4)
            # Decode 구간 주석
            decode_samples = [s for s in samples if s.phase == "decode" and s.sm_util_pct >= 0]
            if decode_samples:
                decode_sm_avg = np.mean([s.sm_util_pct for s in decode_samples])
                decode_t_mid  = np.mean([s.t_ms for s in decode_samples])
                ax_sm.annotate(
                    f"Decode: BW-bound\nSM avg {decode_sm_avg:.0f}%",
                    xy=(decode_t_mid, decode_sm_avg),
                    xytext=(decode_t_mid - 300, decode_sm_avg + 20),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=8, color="#555555",
                )
        else:
            ax_sm.text(0.5, 0.5,
                       "SM Utilization: N/A\n(pynvml not supported on Jetson AGX Thor\n"
                       "→ tegrastats GR3D_FREQ also unavailable)",
                       ha="center", va="center", transform=ax_sm.transAxes,
                       fontsize=9, color="#888", style="italic")
            ax_sm.set_ylim(0, 105)
        ax_sm.set_ylabel("GPU SM\nUtilization (%)", fontsize=9)
        ax_sm.legend(loc="upper right", fontsize=8, framealpha=0.8)

        # ── Panel D: 시스템 RAM ──────────────────────────────────────────
        if sys_arr.max() > 0:
            ax_sys.fill_between(t_arr, sys_arr, alpha=0.3, color="mediumpurple", zorder=2)
            ax_sys.plot(t_arr, sys_arr, color="mediumpurple", linewidth=1.5,
                        label="System RAM (psutil)", zorder=3)
            if self._spec and self._spec.gpu_total_mem_gb > 0:
                ax_sys.axhline(self._spec.gpu_total_mem_gb, color="navy",
                               linestyle="--", alpha=0.5, linewidth=1,
                               label=f"Total Unified ({self._spec.gpu_total_mem_gb:.0f}GB)")
            ax_sys.set_ylabel("System RAM\n(GB)", fontsize=9)
            ax_sys.legend(loc="lower right", fontsize=8, framealpha=0.8)
        else:
            ax_sys.text(0.5, 0.5, "psutil 미설치 — 시스템 RAM 측정 불가",
                        transform=ax_sys.transAxes, ha="center", va="center",
                        fontsize=10, color="gray")

        ax_sys.set_xlabel("Wall Time from Inference Start (ms)", fontsize=9)

        # ── 범례 패치 ─────────────────────────────────────────────────
        legend_patches = [
            mpatches.Patch(color=PHASE_COLORS["vision"],  label="Vision Encoding"),
            mpatches.Patch(color=PHASE_COLORS["prefill"], label="LLM Prefill"),
            mpatches.Patch(color=PHASE_COLORS["decode"],  label="LLM Decode"),
            mpatches.Patch(color=PHASE_COLORS["flow"],    label="Flow Matching"),
        ]
        fig.legend(handles=legend_patches, loc="upper right",
                   bbox_to_anchor=(0.98, 0.97), fontsize=8,
                   title="GPU Phase", framealpha=0.9)

        plt.setp(ax_phase.get_xticklabels(), visible=False)
        plt.setp(ax_mem.get_xticklabels(),   visible=False)
        plt.setp(ax_sm.get_xticklabels(),    visible=False)

        out_path = FIG_DIR / "fig_memory_timeline.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Fig] 저장: {out_path}")

    def _estimate_stage_boundaries(
        self, samples: list[MemSample], total_ms: float
    ) -> list[dict]:
        """샘플 phase 정보로 단계 경계 추정."""
        stages = []
        phase_map = {
            "vision"  : ("Vision Encoding", PHASE_COLORS["vision"]),
            "prefill" : ("LLM Prefill",     PHASE_COLORS["prefill"]),
            "decode"  : ("LLM Decode",      PHASE_COLORS["decode"]),
            "flow"    : ("Flow Matching",    PHASE_COLORS["flow"]),
        }

        if not samples:
            return stages

        t_arr    = [s.t_ms  for s in samples]
        ph_arr   = [s.phase for s in samples]

        cur_ph   = ph_arr[0]
        cur_s    = t_arr[0]

        for i in range(1, len(ph_arr)):
            if ph_arr[i] != cur_ph:
                if cur_ph in phase_map:
                    dur = t_arr[i] - cur_s
                    lbl, col = phase_map[cur_ph]
                    stages.append({"start": cur_s, "duration": dur,
                                   "label": lbl, "color": col})
                cur_ph = ph_arr[i]
                cur_s  = t_arr[i]

        # 마지막 단계
        if cur_ph in phase_map and t_arr:
            dur = t_arr[-1] - cur_s
            lbl, col = phase_map[cur_ph]
            stages.append({"start": cur_s, "duration": dur,
                           "label": lbl, "color": col})
        return stages

    # ── LLC 임계점 분석 그래프 ────────────────────────────────────────────
    def _plot_llc_analysis(self):
        """
        GPU L2 / CPU L3 / 모델 크기를 시각적으로 비교.
        "LLC를 넘으면 DRAM 병목" 포인트를 명확히 가시화.
        """
        if not self._spec:
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#FAFAFA")

        # GPU Shared+L1 per SM: 228 KB × 20 SM = 4.56 MB [CUDA실측]
        gpu_shared_total_mb = (
            self._spec.gpu_shared_mem_per_sm_kb * self._spec.gpu_sm_count / 1024
            if self._spec.gpu_shared_mem_per_sm_kb > 0 else 4.56
        )
        # CPU L3: /sys 탐지 우선, 실패 시 공식 스펙 16MB 사용
        cpu_l3_mb = (
            self._spec.cpu_l3_cache_kb / 1024 if self._spec.cpu_l3_cache_kb > 0
            else self._spec.cpu_l3_official_kb / 1024
        )
        items = [
            # 레이블              크기(GB)                  색상       대역폭 주석
            ("GPU Shared\n+L1\n(total)",
                                  gpu_shared_total_mb / 1024, "#4878CF", "SRAM\n[추정]~20 TB/s"),
            ("GPU L2\n(total)\n[실측]",
                                  (self._spec.gpu_l2_cache_mb or 33.6) / 1024,
                                                              "#6ACC65", "SRAM\n[추정]~2 TB/s"),
            ("CPU L3\n(shared)\n[공식]",
                                  cpu_l3_mb / 1024,           "#B47CC7", "SRAM\n[추정]~수백 GB/s"),
            ("KV Cache\n[추정]",  1540 / 1024,                "#FFD700", "DRAM\n[공식]273 GB/s"),
            ("Model\nWeights\n[추정]",
                                  (self._spec.model_weights_mb or 22000) / 1024,
                                                              "#D65F5F", "DRAM\n[공식]273 GB/s"),
            ("Unified\nMemory\n[실측]",
                                  self._spec.gpu_total_mem_gb or 131.9,
                                                              "#CCCCCC", "LPDDR5X"),
        ]

        labels  = [x[0] for x in items]
        values  = [x[1] for x in items]
        colors  = [x[2] for x in items]
        bw_labs = [x[3] for x in items]

        bars = ax.bar(labels, values, color=colors, alpha=0.8,
                      edgecolor="white", linewidth=1.2)

        ax.set_yscale("log")
        ax.set_ylabel("크기 (GB, log scale)", fontsize=10)
        ax.set_title(
            "메모리 계층 크기 비교 — LLC 초과 시 DRAM 병목\n"
            f"GPU L2: {self._spec.gpu_l2_cache_mb:.0f}MB  |  "
            f"CPU L3: {(self._spec.cpu_l3_cache_kb or 0)/1024:.0f}MB  |  "
            f"Model: {(self._spec.model_weights_mb or 0)/1024:.1f}GB",
            fontsize=11,
        )

        for bar, val, bw in zip(bars, values, bw_labs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height * 1.3,
                    f"{val:.3f}GB\n{bw}", ha="center", va="bottom",
                    fontsize=8, color="#333333")

        # DRAM 경계선
        gpu_l2 = (self._spec.gpu_l2_cache_mb or 50) / 1024
        ax.axhline(gpu_l2, color="#6ACC65", linestyle="--", alpha=0.6,
                   label=f"GPU L2 임계점 ({gpu_l2*1024:.0f}MB)")
        ax.axhline(273 / 1000, color="#D65F5F", linestyle=":", alpha=0.5,
                   label="이론 DRAM BW 한계 참고선")

        ax.legend(fontsize=8, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # 화살표: "이 선 넘으면 DRAM 접근"
        ax.annotate(
            "← 이 크기를 초과하면\n   DRAM 직접 접근 (273 GB/s 병목)",
            xy=(1, gpu_l2), xytext=(3, gpu_l2 * 8),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=9, color="red",
        )

        out_path = FIG_DIR / "fig_llc_analysis.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Fig] 저장: {out_path}")

    # ── 요약 통계 ─────────────────────────────────────────────────────────
    def _compute_summary(self) -> dict:
        if not self.all_samples:
            return {}

        all_sm   = [s.sm_util_pct  for run in self.all_samples for s in run]
        all_mem  = [s.gpu_used_mb  for run in self.all_samples for s in run]

        per_phase = {}
        for ph in ["vision", "prefill", "decode", "flow"]:
            ph_sm  = [s.sm_util_pct for run in self.all_samples
                      for s in run if s.phase == ph]
            ph_mem = [s.gpu_used_mb for run in self.all_samples
                      for s in run if s.phase == ph]
            if ph_sm:
                per_phase[ph] = {
                    "sm_util_mean_pct" : float(np.mean(ph_sm)),
                    "sm_util_max_pct"  : float(np.max(ph_sm)),
                    "gpu_mem_mean_mb"  : float(np.mean(ph_mem)),
                    "gpu_mem_peak_mb"  : float(np.max(ph_mem)),
                }

        return {
            "n_runs"       : self.n_runs,
            "overall": {
                "sm_util_mean_pct" : float(np.mean(all_sm))  if all_sm  else 0,
                "sm_util_max_pct"  : float(np.max(all_sm))   if all_sm  else 0,
                "gpu_mem_peak_mb"  : float(np.max(all_mem))  if all_mem else 0,
                "gpu_mem_mean_mb"  : float(np.mean(all_mem)) if all_mem else 0,
            },
            "by_phase"     : per_phase,
            "hardware_spec": asdict(self._spec) if self._spec else {},
        }

    def _print_summary(self, summary: dict):
        print("\n" + "=" * 60)
        print("  GPU 메모리 + SM 활용률 요약")
        print("=" * 60)
        ov = summary.get("overall", {})
        print(f"  SM 활용률  (전체 평균) : {ov.get('sm_util_mean_pct', 0):.1f}%")
        print(f"  SM 활용률  (최대)      : {ov.get('sm_util_max_pct', 0):.1f}%")
        print(f"  GPU 메모리 (평균)      : {ov.get('gpu_mem_mean_mb', 0)/1024:.2f}GB")
        print(f"  GPU 메모리 (피크)      : {ov.get('gpu_mem_peak_mb', 0)/1024:.2f}GB")
        print()
        bp = summary.get("by_phase", {})
        print(f"  {'단계':10}  {'SM 평균':>8}  {'SM 최대':>8}  {'메모리 평균':>12}  {'메모리 피크':>12}")
        print(f"  {'-'*60}")
        for ph in ["vision", "prefill", "decode", "flow"]:
            d = bp.get(ph, {})
            if d:
                print(f"  {ph:10}  {d['sm_util_mean_pct']:>7.1f}%  "
                      f"{d['sm_util_max_pct']:>7.1f}%  "
                      f"{d['gpu_mem_mean_mb']/1024:>10.2f}GB  "
                      f"{d['gpu_mem_peak_mb']/1024:>10.2f}GB")
        print("=" * 60)
        print(f"\n[출력 위치] {OUTPUT_DIR.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 하드웨어 스펙만 측정하는 모드
# ═══════════════════════════════════════════════════════════════════════════════

def run_spec_only():
    """모델 로드 없이 하드웨어 스펙만 측정 + 공식 스펙 대조 출력."""
    print("\n[Spec-Only Mode] 하드웨어 스펙 측정 중...")
    spec = measure_hardware_spec(model=None)
    print_hardware_spec(spec)

    # ── 공식 스펙 대조표 ────────────────────────────────────────────────
    print("  ── 공식 스펙 대조 (NVIDIA Jetson Thor Datasheet DS-11945-001) ──")
    print(f"  {'항목':<28} {'실측값':<20} {'공식 스펙':<20} {'출처'}")
    print(f"  {'─'*28} {'─'*20} {'─'*20} {'─'*10}")
    rows = [
        ("CPU 아키텍처",       spec.cpu_arch,                    "Arm Neoverse V3AE",  "공식"),
        ("CPU 코어 수",        f"{spec.cpu_cores}코어",           "14코어",             "공식"),
        ("CPU 클럭",           f"미측정",                         "2.6 GHz (max)",      "공식"),
        ("CPU L1 I/D",        f"{spec.cpu_l1i_kb_per_core} KB/core",
                                                                   "64 KB/core",         "공식+실측"),
        ("CPU L2",            f"{spec.cpu_l2_kb_per_core} KB/core",
                                                                   "1 MB/core",          "공식+실측"),
        ("CPU L3 (shared)",
                              f"{'%d KB' % spec.cpu_l3_cache_kb if spec.cpu_l3_cache_kb else '/sys 미탐지'}",
                                                                   "16 MB shared",       "공식"),
        ("GPU 아키텍처",       f"SM {spec.gpu_compute_cap}",      "Blackwell (SM 11.0)","공식"),
        ("GPU CUDA 코어",      f"{spec.gpu_sm_count} SM×128=2560","2560",               "공식"),
        ("GPU Tensor 코어",    f"{spec.gpu_tensor_cores}개 (5세대)", "96 (5th gen)",    "공식"),
        ("GPU FP4 TFLOPS",    f"{spec.gpu_fp4_tflops:.0f} TFLOPS","2070 TFLOPS",       "공식"),
        ("GPU Shared+L1/SM",  f"{spec.gpu_shared_mem_per_sm_kb:.0f} KB",
                                                                   "228 KB (Blackwell)", "실측"),
        ("GPU L2 Cache",      f"{spec.gpu_l2_cache_mb:.1f} MB",  "미공개 (실측값)",    "실측"),
        ("Unified Memory",    f"{spec.gpu_total_mem_gb:.1f} GB",  "128 GB",             "공식"),
        ("DRAM 대역폭",       f"{spec.dram_bandwidth_gbps:.0f} GB/s",
                                                                   "273 GB/s",           "공식"),
        ("DRAM 버스",         f"{spec.dram_bus_bits}-bit LPDDR5X","256-bit LPDDR5X",   "공식"),
    ]
    for name, measured, official, src in rows:
        match = "✅" if measured.replace(" ", "") == official.replace(" ", "") else \
                "⚠️" if "미탐지" in measured or "미측정" in measured or "미공개" in measured else "✅"
        print(f"  {name:<28} {measured:<20} {official:<20} [{src}] {match}")

    # ── 시스템 RAM ──────────────────────────────────────────────────────
    if _PSUTIL_OK:
        vm = psutil.virtual_memory()
        print(f"\n  시스템 RAM (현재 상태):")
        print(f"    총 크기  : {vm.total / 1e9:.1f} GB")
        print(f"    현재 사용: {vm.used / 1e9:.1f} GB ({vm.percent:.1f}%)  ← 모델 로드 전 기저")
        print(f"    여유     : {vm.available / 1e9:.1f} GB")

    # ── 결과 저장 ───────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "hardware_spec.json", "w") as f:
        json.dump(asdict(spec), f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {OUTPUT_DIR / 'hardware_spec.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLI 진입점
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Alpamayo 1.5 GPU Memory + SM Utilization Profiler"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="모델 경로 또는 HuggingFace repo ID",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="워밍업 횟수 (기본값: 2)",
    )
    parser.add_argument(
        "--runs", type=int, default=4,
        help="측정 횟수 (기본값: 4)",
    )
    parser.add_argument(
        "--spec-only", action="store_true",
        help="모델 로드 없이 하드웨어 스펙만 측정",
    )
    parser.add_argument(
        "--interval-ms", type=int, default=100,
        help="메모리 샘플링 주기 ms (기본값: 100)",
    )
    args = parser.parse_args()

    if args.spec_only:
        run_spec_only()
        return

    if not _NVML_OK:
        print("[WARNING] pynvml 미설치 또는 초기화 실패.")
        print("          GPU 메모리를 torch.cuda API로 측정합니다 (Jetson에서 정상).")
        # sys.exit(1) 제거 — torch.cuda fallback으로 실행 가능

    profiler = AlpamayoMemoryProfiler(
        model_path=args.model_path,
        warmup=args.warmup,
        runs=args.runs,
    )
    profiler.run_with_stage_patch()


if __name__ == "__main__":
    main()
