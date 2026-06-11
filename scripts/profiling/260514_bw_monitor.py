"""
260514_bw_monitor.py  ·  DRAM Memory Bandwidth Profiler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[목적]
  Alpamayo 1.5 inference 전 구간에 걸친 DRAM 대역폭 시계열을 측정하고
  Phase별 (Vision / LM Prefill / Decode / Flow) BW 프로파일을 분석한다.

  핵심 질문:
    Q1. Decode는 273 GB/s 피크에 얼마나 근접하는가?
    Q2. CPU는 inference 중 얼마나 대역폭을 사용하는가?
    Q3. 대역폭이 Phase 전환점에서 급변하는가 (spike / drop)?
    Q4. Decode 구간 내부에서 BW가 선형적인가, 아니면 파동치는가?

[측정 아키텍처]
  ┌─ Layer A: tegrastats (100ms 폴링) ──────────────────────────────────────┐
  │  • EMC_FREQ %   → 총 DRAM 대역폭 (CPU + GPU + ISP + NVENC …)           │
  │  • GR3D_FREQ %  → GPU 코어 utilization (BW와 구분!)                    │
  │  • CPU [%@freq] → per-core CPU 사용률                                   │
  │  • RAM used/total → 메모리 점유량 추이                                   │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ Layer B: /sys/kernel/debug/bwmon (10ms 폴링, root 필요) ─────────────┐
  │  • per-client BW: GPU 단독 vs CPU 단독 vs 기타                         │
  │  • 더 세밀한 시간 해상도                                                │
  │  • 없으면 자동 스킵                                                     │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ Layer C: CUDA Events (v4 재활용) ─────────────────────────────────────┐
  │  • GPU Phase 타이밍 (ms 정밀도)                                         │
  │  • Decode BW = model_GB × n_tok / decode_ms (이미 검증된 값)           │
  │  • Wall clock과 동기화하여 Layer A/B와 시간축 통합                      │
  └─────────────────────────────────────────────────────────────────────────┘

[출력]
  profiling_results/260514_bw/
  ├── bw_timeseries.json   ← 원시 시계열 (샘플 단위)
  ├── bw_analysis.json     ← Phase별 통계 분석
  ├── figures/
  │   ├── fig_bw_timeline.png   ← ★ 메인: BW vs time + phase 오버레이
  │   ├── fig_bw_phase_box.png  ← Phase별 BW 분포 (박스플롯)
  │   ├── fig_bw_cpu_gpu.png    ← CPU vs GPU 대역폭 split
  │   └── fig_bw_decode_zoom.png← Decode 구간 확대 (선형성 검증)
  └── bw_report.md

[실행]
  # 기본 (tegrastats만 사용)
  python 260514_bw_monitor.py

  # bwmon 포함 (sudo 필요, per-client 분리 가능)
  sudo python 260514_bw_monitor.py --bwmon

  # nsys 동시 캡처
  nsys profile --trace=cuda,nvtx --output=profiling_results/260514_bw/nsys_bw \\
      python 260514_bw_monitor.py --nsys
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import types
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch

# ──────────────────────────────────────────────────────────────────────────────
# 경로 / 상수
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path("profiling_results/260514_bw")
FIGD = OUT / "figures"
for _d in [OUT, FIGD]:
    _d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DRAM_BW_PEAK = 273.0   # GB/s  Jetson AGX Thor LPDDR5X
MODEL_GB      = 22.157  # bf16 전체 모델
BW_THRESHOLD  = 0.70   # MBU 70% 이상 → BW-bound 판정

# Phase 색상 (v4와 통일)
PHASE_COLORS = {
    "vision"    : "#70B0D0",
    "lm_prefill": "#6ACC65",
    "decode"    : "#D65F5F",
    "flow"      : "#B47CC7",
    "warmup"    : "#CCCCCC",
    "gap"       : "#F0F0F0",
}

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BwSample:
    """tegrastats + nvidia-smi 합성 샘플.

    Thor JetPack 7 tegrastats 포맷:
      RAM X/YMB  CPU [X%@freq,...]  VDD_GPU XmW/XmW  VDD_CPU_SOC_MSS XmW/XmW  ...
      ← EMC_FREQ / GR3D_FREQ 필드 없음

    EMC 대역폭 측정 전략:
      • emc_GBps_derived: Decode 구간은 CUDA Events (정확),
                          나머지는 GR3D% 기반 보정 추정
      • vdd_gpu_mW:  GPU 전력 → compute vs BW-bound 판별 proxy
                    compute-bound (prefill): ~30-80W
                    BW-bound (decode): ~5-20W (메모리 액세스 위주)
    """
    t_wall_s:         float   # time.perf_counter() 시각
    emc_pct:          float   # EMC 대역폭 % (직접 측정 시; 없으면 0)
    emc_GBps:         float   # 직접 측정 EMC GB/s (없으면 0)
    gr3d_pct:         float   # GPU 코어 utilization % (nvidia-smi)
    cpu_pct:          float   # CPU 평균 사용률 %
    ram_GB:           float   # 현재 RAM 사용량 (GB)
    vdd_gpu_mW:       float = 0.0   # GPU 즉시 전력 (mW) — BW vs compute 판별
    vdd_cpu_soc_mW:   float = 0.0   # CPU+SOC+메모리서브시스템 전력 (mW)
    vin_mW:           float = 0.0   # 전체 보드 전력 (mW)
    emc_GBps_derived: float = 0.0   # CUDA Events 또는 추정값 GB/s
    gpu_client_GBps:  float = 0.0   # bwmon per-client (root 있을 때)
    cpu_client_GBps:  float = 0.0   # bwmon per-client (root 있을 때)


@dataclass
class PhaseEvent:
    """Phase 전환 이벤트."""
    t_wall_s: float
    name:     str       # "vision_start", "decode_end" 등


@dataclass
class RunRecord:
    """단일 inference run 전체 기록."""
    label:    str
    is_warmup:bool
    phase_events: list[PhaseEvent]  = field(default_factory=list)
    bw_samples:   list[BwSample]    = field(default_factory=list)
    # CUDA Events 타이밍 (ms)
    t_wall_start_s: float = 0.0
    vision_ms:      float = 0.0
    lm_prefill_ms:  float = 0.0
    decode_ms:      float = 0.0
    flow_ms:        float = 0.0
    n_tok:          int   = 0


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Event 타이머
# ──────────────────────────────────────────────────────────────────────────────

class CUDATimer:
    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self._started = False
        self._stopped = False

    def start(self):
        torch.cuda.synchronize()
        self._s.record()
        self._started = True
        self._stopped = False

    def stop(self):
        if not self._started:
            return
        self._e.record()
        torch.cuda.synchronize()
        self._stopped = True

    def ms(self) -> float:
        if self._started and self._stopped:
            return self._s.elapsed_time(self._e)
        return 0.0

    def reset(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self._started = False
        self._stopped = False


# ──────────────────────────────────────────────────────────────────────────────
# tegrastats 파서
# ──────────────────────────────────────────────────────────────────────────────

# ── tegrastats 정규식 (Thor JetPack 7 포맷 기준) ─────────────────────────────
#
# 실제 출력 예:
#   05-14-2026 21:36:35 RAM 26063/125772MB (lfb 35x4MB)
#   CPU [2%@972,...] VDD_GPU -392mW/-392mW VDD_CPU_SOC_MSS 3924mW/3835mW
#   VIN_SYS_5V0 5064mW/5055mW VIN 15004mW/15096mW
#
# Thor에서는 EMC_FREQ / GR3D_FREQ 없음 → VDD_GPU / power 필드로 대체
_EMC_RE    = re.compile(r'EMC_FREQ\s+(\d+)%(?:@(\d+))?')         # 없을 수 있음
_GR3D_RE   = re.compile(r'GR3D_FREQ\s+(\d+)%(?:@(\d+))?')        # 없을 수 있음
_CPU_RE    = re.compile(r'CPU\s+\[([^\]]+)\]')
_RAM_RE    = re.compile(r'RAM\s+(\d+)/(\d+)MB')
_VDDGPU_RE = re.compile(r'VDD_GPU\s+(-?\d+)mW/(-?\d+)mW')         # Thor: GPU 전력
_VDDCPU_RE = re.compile(r'VDD_CPU_SOC_MSS\s+(-?\d+)mW/(-?\d+)mW') # Thor: CPU+SOC+메모리
_VIN_RE    = re.compile(r'\bVIN\s+(-?\d+)mW/(-?\d+)mW')            # Thor: 전체 입력 전력


def parse_tegrastats_line(line: str) -> Optional[BwSample]:
    """
    tegrastats 출력 한 줄을 BwSample로 변환.

    Thor JetPack 7: EMC_FREQ / GR3D_FREQ 없음.
    RAM 또는 CPU 필드가 있으면 파싱 성공으로 간주.
    없는 필드는 0으로 채움.
    """
    ram_m    = _RAM_RE.search(line)
    cpu_m    = _CPU_RE.search(line)

    # RAM 또는 CPU 중 하나도 없으면 의미 없는 줄
    if not ram_m and not cpu_m:
        return None

    # EMC (있으면 사용, Thor JetPack 7에서는 없음)
    emc_m    = _EMC_RE.search(line)
    emc_pct  = float(emc_m.group(1)) if emc_m else 0.0
    emc_GBps = emc_pct / 100.0 * DRAM_BW_PEAK

    # GR3D (있으면 사용)
    gr3d_m   = _GR3D_RE.search(line)
    gr3d_pct = float(gr3d_m.group(1)) if gr3d_m else 0.0

    # CPU 평균 사용률
    cpu_pct = 0.0
    if cpu_m:
        vals = [float(m.group(1))
                for p in cpu_m.group(1).split(',')
                if (m := re.match(r'(\d+)%', p.strip()))]
        cpu_pct = float(np.mean(vals)) if vals else 0.0

    # RAM
    ram_GB = float(ram_m.group(1)) / 1024.0 if ram_m else 0.0

    # 전력 (Thor 전용 필드)
    vdd_gpu_m  = _VDDGPU_RE.search(line)
    vdd_cpu_m  = _VDDCPU_RE.search(line)
    vin_m      = _VIN_RE.search(line)

    vdd_gpu_mW     = float(vdd_gpu_m.group(1))  if vdd_gpu_m  else 0.0
    vdd_cpu_soc_mW = float(vdd_cpu_m.group(1))  if vdd_cpu_m  else 0.0
    vin_mW         = float(vin_m.group(1))       if vin_m      else 0.0

    return BwSample(
        t_wall_s       = time.perf_counter(),
        emc_pct        = emc_pct,
        emc_GBps       = emc_GBps,
        gr3d_pct       = gr3d_pct,
        cpu_pct        = cpu_pct,
        ram_GB         = ram_GB,
        vdd_gpu_mW     = vdd_gpu_mW,
        vdd_cpu_soc_mW = vdd_cpu_soc_mW,
        vin_mW         = vin_mW,
    )


# ──────────────────────────────────────────────────────────────────────────────
# BW 모니터 (백그라운드 스레드)
# ──────────────────────────────────────────────────────────────────────────────

class SysfsPoller:
    """
    tegrastats 없이 /proc + nvidia-smi 기반으로 BwSample을 생성.

    EMC 대역폭 추정 우선순위:
      1. /sys/kernel/debug/tegra_actmon 또는 /sys/devices/.../actmon (활동률 %)
      2. nvidia-smi 메모리 사용률 (간접 추정)
      3. /sys/class/devfreq/<emc>/cur_freq vs max_freq 비율

    CPU 사용률: /proc/stat delta
    RAM:        /proc/meminfo
    GPU util:   nvidia-smi --query-gpu=utilization.gpu
    """

    def __init__(self):
        self._prev_cpu_stat = self._read_cpu_stat()
        self._emc_path  = self._find_emc_actmon()
        self._emc_freq_paths = self._find_emc_devfreq()
        self._has_smi   = self._probe_nvidiasmi()
        print(f"  [SysfsPoller] emc_actmon={'있음' if self._emc_path else '없음'}  "
              f"emc_devfreq={'있음' if self._emc_freq_paths else '없음'}  "
              f"nvidia-smi={'있음' if self._has_smi else '없음'}")

    # ── 경로 탐색 ────────────────────────────────────────────────────────────

    @staticmethod
    def _find_emc_actmon() -> Optional[Path]:
        """Jetson EMC 활동 모니터 sysfs 경로 탐색."""
        import glob as _glob
        candidates = (
            _glob.glob("/sys/kernel/debug/tegra_actmon/emc/*/avg_activity")
            + _glob.glob("/sys/devices/platform/tegra-actmon/emc/*/avg_activity")
            + _glob.glob("/sys/class/devfreq/*/actmon_avg_activity")
            + _glob.glob("/sys/bus/platform/devices/*/actmon/emc/avg_activity")
        )
        for c in candidates:
            try:
                Path(c).read_text()
                return Path(c)
            except Exception:
                continue
        return None

    @staticmethod
    def _find_emc_devfreq() -> Optional[tuple[Path, Path]]:
        """cur_freq / max_freq 경로 탐색 (EMC devfreq governor)."""
        import glob as _glob
        for pat in ["/sys/class/devfreq/*emc*", "/sys/class/devfreq/*EMC*"]:
            for d in _glob.glob(pat):
                cur = Path(d) / "cur_freq"
                mx  = Path(d) / "max_freq"
                if cur.exists() and mx.exists():
                    try:
                        float(cur.read_text())
                        float(mx.read_text())
                        return (cur, mx)
                    except Exception:
                        continue
        return None

    @staticmethod
    def _probe_nvidiasmi() -> bool:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            return r.returncode == 0
        except Exception:
            return False

    # ── /proc 읽기 ───────────────────────────────────────────────────────────

    @staticmethod
    def _read_cpu_stat() -> Optional[list[int]]:
        try:
            line = Path("/proc/stat").read_text().splitlines()[0]
            return [int(x) for x in line.split()[1:]]
        except Exception:
            return None

    def _cpu_pct(self) -> float:
        new = self._read_cpu_stat()
        if new is None or self._prev_cpu_stat is None:
            return 0.0
        d = [n - o for n, o in zip(new, self._prev_cpu_stat)]
        self._prev_cpu_stat = new
        total = sum(d)
        idle  = d[3] + (d[4] if len(d) > 4 else 0)
        return (1.0 - idle / total) * 100.0 if total > 0 else 0.0

    @staticmethod
    def _ram_gb() -> float:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
            return (total_kb - avail_kb) / 1024.0 / 1024.0
        except Exception:
            return 0.0

    def _emc_pct(self) -> float:
        """EMC 활동률 0-100 추정."""
        # 방법 1: actmon (가장 정확)
        if self._emc_path:
            try:
                val = float(self._emc_path.read_text().strip())
                # actmon은 보통 0-1000 permille 또는 0-100 %
                return val / 10.0 if val > 100 else val
            except Exception:
                pass
        # 방법 2: cur_freq / max_freq
        if self._emc_freq_paths:
            try:
                cur = float(self._emc_freq_paths[0].read_text())
                mx  = float(self._emc_freq_paths[1].read_text())
                return (cur / mx * 100.0) if mx > 0 else 0.0
            except Exception:
                pass
        return 0.0

    def _gr3d_pct(self) -> float:
        if not self._has_smi:
            return 0.0
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                return float(r.stdout.strip().replace(" %", "").replace("%", ""))
        except Exception:
            pass
        return 0.0

    def sample(self) -> BwSample:
        emc_pct = self._emc_pct()
        return BwSample(
            t_wall_s  = time.perf_counter(),
            emc_pct   = emc_pct,
            emc_GBps  = emc_pct / 100.0 * DRAM_BW_PEAK,
            gr3d_pct  = self._gr3d_pct(),
            cpu_pct   = self._cpu_pct(),
            ram_GB    = self._ram_gb(),
        )


class BandwidthMonitor:
    """
    tegrastats (--logfile 방식) 또는 SysfsPoller를 백그라운드에서 샘플링.

    tegrastats --logfile 방식을 우선 사용 — stdout 파이프 버퍼링 문제 회피.
    실패 시 SysfsPoller (nvidia-smi + /proc) 폴백.

    사용:
        mon = BandwidthMonitor()
        mon.start()
        ...  inference  ...
        mon.stop()
        samples = mon.drain()  # list[BwSample]
    """

    def __init__(self, interval_ms: int = 100, use_bwmon: bool = False):
        self._interval_ms  = max(100, interval_ms)
        self._use_bwmon    = use_bwmon and self._probe_bwmon()
        self._samples: deque[BwSample] = deque()
        self._lock     = threading.Lock()
        self._stop_ev  = threading.Event()
        self._thread   = None
        self._proc     = None          # tegrastats subprocess
        self._logfile  = Path("/tmp/_tegrastats_bw_monitor.log")
        self._bwmon_paths: dict[str, Path] = {}
        self._mode     = "unknown"     # "tegrastats_logfile" | "sysfs"
        self._sysfs    = None          # SysfsPoller instance

    # ── bwmon 탐색 ──────────────────────────────────────────────────────────

    def _probe_bwmon(self) -> bool:
        base = Path("/sys/kernel/debug/bwmon")
        if not base.exists():
            return False
        found = {}
        for entry in sorted(base.iterdir()):
            for fname in ("bw", "avg_bw", "bw_count"):
                fp = entry / fname
                if fp.exists():
                    try:
                        fp.read_text()
                        found[entry.name] = fp
                        break
                    except PermissionError:
                        return False
        if found:
            self._bwmon_paths = found
            print(f"  [BandwidthMonitor] bwmon clients: {list(found.keys())}")
            return True
        return False

    def _augment_bwmon(self, s: BwSample):
        gpu_kb = cpu_kb = 0.0
        for name, fpath in self._bwmon_paths.items():
            try:
                val = float(fpath.read_text().strip())
                val_GBps = val / 1e9 if val > 1e9 else val / 1e6
                nl = name.lower()
                if any(k in nl for k in ("gpu", "gr3d", "ga10b")):
                    gpu_kb += val_GBps
                elif any(k in nl for k in ("cpu", "aon", "bpmp")):
                    cpu_kb += val_GBps
            except Exception:
                continue
        s.gpu_client_GBps = gpu_kb
        s.cpu_client_GBps = cpu_kb

    # ── tegrastats --logfile 방식 ────────────────────────────────────────────

    def _launch_tegrastats_logfile(self) -> bool:
        """
        tegrastats --logfile /tmp/... 로 파일에 기록.
        stdout 파이프 버퍼링 문제를 완전히 우회.
        """
        self._logfile.unlink(missing_ok=True)

        # 탐색 경로 + sudo 유무
        teg_cmds = []
        for binary in ("tegrastats",
                        "/usr/bin/tegrastats",
                        "/usr/local/bin/tegrastats"):
            teg_cmds.append([binary, "--interval", str(self._interval_ms),
                             "--logfile", str(self._logfile)])
            teg_cmds.append(["sudo", binary, "--interval", str(self._interval_ms),
                             "--logfile", str(self._logfile)])

        for cmd in teg_cmds:
            try:
                p = subprocess.Popen(cmd,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE,
                                     text=True)
                # 최대 2초 대기하며 로그 파일 생성 확인
                for _ in range(20):
                    time.sleep(0.1)
                    if self._logfile.exists() and self._logfile.stat().st_size > 10:
                        content = self._logfile.read_text()
                        # Thor JetPack 7: EMC_FREQ 없음 → RAM 또는 CPU로 확인
                        if "RAM" in content or "CPU" in content or "EMC" in content:
                            # 파서 동작 검증
                            test_line = [l for l in content.splitlines() if l.strip()]
                            test_ok   = any(parse_tegrastats_line(l) is not None
                                            for l in test_line[-3:])
                            if test_ok:
                                self._proc = p
                                has_emc = "EMC_FREQ" in content
                                print(f"  [BandwidthMonitor] tegrastats --logfile OK: {cmd[0]}"
                                      f"  (EMC_FREQ: {'있음' if has_emc else '없음 → VDD_GPU 모드'})")
                                return True
                            else:
                                print(f"  [BandwidthMonitor] tegrastats 파싱 실패, 내용 앞부분: "
                                      f"{content[:120]!r}")
                # 실패 — 오류 메시지 출력 후 kill
                err = p.stderr.read(200) if p.stderr else ""
                if err.strip():
                    print(f"  [BandwidthMonitor] tegrastats 오류: {err.strip()[:120]}")
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except Exception:
                    pass
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"  [BandwidthMonitor] tegrastats launch 예외: {e}")
                continue

        print("  [BandwidthMonitor] tegrastats --logfile 실패 → SysfsPoller 폴백")
        return False

    # ── 백그라운드 루프 ─────────────────────────────────────────────────────

    @staticmethod
    def _query_gr3d_smi() -> float:
        """
        nvidia-smi로 GR3D util% 조회.
        Thor JetPack 7에서 tegrastats에 GR3D_FREQ 없을 때 보완.
        """
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1,
            )
            if r.returncode == 0:
                val = r.stdout.strip().replace(" %", "").replace("%", "")
                return float(val)
        except Exception:
            pass
        return 0.0

    def _loop_logfile(self):
        """
        로그 파일 tail 방식 폴링.
        tegrastats에 GR3D_FREQ 없으면 nvidia-smi로 보완.
        """
        last_line  = ""
        interval   = self._interval_ms / 1000.0
        need_gr3d  = True   # GR3D_FREQ 필드 없을 때 nvidia-smi 사용
        smi_ok     = True   # nvidia-smi 사용 가능 여부

        while not self._stop_ev.is_set():
            try:
                content = self._logfile.read_text()
                lines   = [l for l in content.splitlines() if l.strip()]
                if lines:
                    line = lines[-1]
                    if line != last_line:
                        s = parse_tegrastats_line(line)
                        if s:
                            # GR3D_FREQ 없으면 nvidia-smi로 보완
                            if need_gr3d and s.gr3d_pct == 0.0 and smi_ok:
                                try:
                                    s.gr3d_pct = self._query_gr3d_smi()
                                except Exception:
                                    smi_ok = False
                            elif s.gr3d_pct > 0:
                                need_gr3d = False  # tegrastats에서 직접 얻음
                            if self._use_bwmon:
                                self._augment_bwmon(s)
                            with self._lock:
                                self._samples.append(s)
                            last_line = line
            except Exception:
                pass
            time.sleep(interval)

    def _loop_sysfs(self):
        """SysfsPoller 폴링 루프."""
        interval = self._interval_ms / 1000.0
        while not self._stop_ev.is_set():
            try:
                s = self._sysfs.sample()
                with self._lock:
                    self._samples.append(s)
            except Exception:
                pass
            time.sleep(interval)

    # ── 공개 API ────────────────────────────────────────────────────────────

    def start(self):
        self._stop_ev.clear()
        ok = self._launch_tegrastats_logfile()
        if ok:
            self._mode   = "tegrastats_logfile"
            target       = self._loop_logfile
        else:
            self._mode   = "sysfs"
            self._sysfs  = SysfsPoller()
            target       = self._loop_sysfs

        self._thread = threading.Thread(target=target, daemon=True, name="bw-monitor")
        self._thread.start()
        print(f"  [BandwidthMonitor] 샘플링 시작 (mode={self._mode}, "
              f"interval={self._interval_ms}ms)")

    def stop(self):
        self._stop_ev.set()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread:
            self._thread.join(timeout=5)
        # 로그 파일 정리
        try:
            self._logfile.unlink(missing_ok=True)
        except Exception:
            pass
        n = len(self._samples)
        src = "tegrastats" if self._mode == "tegrastats_logfile" else "sysfs(/proc)"
        print(f"  [BandwidthMonitor] 중지. 수집 샘플: {n}개 ({src})")
        if n == 0:
            print("  [경고] BW 샘플 0개 — CUDA Events 결과만으로 분석합니다.")

    def drain(self) -> list[BwSample]:
        with self._lock:
            out = list(self._samples)
            self._samples.clear()
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Phase 탐지기 (BW 버전 — CUDA Events + wall clock 동시 기록)
# ──────────────────────────────────────────────────────────────────────────────

class PhaseDetectorBW:
    """
    v4 PhaseDetectorV4와 동일한 상태 머신.
    여기서는 CUDA Events 타이밍과 함께
    wall clock 타임스탬프를 phase_events 리스트에 기록한다.
    """

    IDLE         = "idle"
    VISION       = "vision"
    LM_PREFILL   = "lm_prefill"
    POST_PREFILL = "post_prefill"
    DECODE       = "decode"

    def __init__(self):
        self._state       = self.IDLE
        self._decode_step = 0
        self.phase_events: list[PhaseEvent] = []
        self.t_vision     = CUDATimer()
        self.t_lm_prefill = CUDATimer()
        self.t_decode     = CUDATimer()
        self._lm_patched  = False

    def reset(self):
        self._state       = self.IDLE
        self._decode_step = 0
        self.phase_events = []
        self.t_vision.reset()
        self.t_lm_prefill.reset()
        self.t_decode.reset()

    def _log(self, name: str):
        self.phase_events.append(PhaseEvent(t_wall_s=time.perf_counter(), name=name))

    def _seq(self, args, kwargs) -> Optional[int]:
        for src in [kwargs.get("input_ids"),
                    kwargs.get("hidden_states"),
                    kwargs.get("inputs_embeds"),
                    *(a for a in args if isinstance(a, torch.Tensor))]:
            if src is None:
                continue
            if isinstance(src, torch.Tensor):
                if src.ndim == 2: return int(src.shape[-1])
                if src.ndim == 3: return int(src.shape[1])
        return None

    def on_vlm_before(self, args, kwargs):
        seq = self._seq(args, kwargs)
        if seq is None:
            return
        if seq > 1 and self._state == self.IDLE:
            torch.cuda.nvtx.range_push("Phase/Vision_Encoder")
            self.t_vision.start()
            self._log("vision_start")
            self._state = self.VISION
        elif seq == 1 and self._state == self.POST_PREFILL:
            torch.cuda.nvtx.range_push("Phase/Decode")
            self.t_decode.start()
            self._log("decode_start")
            self._state = self.DECODE
            self._decode_step = 1
        elif seq == 1 and self._state == self.DECODE:
            self._decode_step += 1
        if self._state == self.DECODE:
            torch.cuda.nvtx.range_push(f"Decode/step_{self._decode_step:03d}")

    def on_vlm_after(self):
        if self._state == self.DECODE:
            torch.cuda.nvtx.range_pop()

    def on_lm_before(self, args, kwargs):
        seq = self._seq(args, kwargs)
        if seq is None:
            return
        self._lm_patched = True
        if seq > 1 and self._state == self.VISION:
            self.t_vision.stop()
            torch.cuda.nvtx.range_pop()
            self._log("vision_end")
            torch.cuda.nvtx.range_push("Phase/LM_Prefill")
            self.t_lm_prefill.start()
            self._log("lm_prefill_start")
            self._state = self.LM_PREFILL

    def on_lm_after(self):
        if self._state == self.LM_PREFILL:
            self.t_lm_prefill.stop()
            torch.cuda.nvtx.range_pop()
            self._log("lm_prefill_end")
            self._state = self.POST_PREFILL

    def end_generate(self):
        if self._state == self.DECODE:
            self.t_decode.stop()
            torch.cuda.nvtx.range_pop()
            self._log("decode_end")
        elif self._state in (self.LM_PREFILL, self.VISION):
            # 비정상 종료
            if self._state == self.LM_PREFILL:
                self.t_lm_prefill.stop()
                torch.cuda.nvtx.range_pop()
            else:
                self.t_vision.stop()
                torch.cuda.nvtx.range_pop()
        self._state = self.IDLE

    @property
    def split_ok(self) -> bool:
        return self.t_vision.ms() > 0 and self.t_lm_prefill.ms() > 0


# ──────────────────────────────────────────────────────────────────────────────
# 패치 유틸
# ──────────────────────────────────────────────────────────────────────────────

def patch_vlm_forward(vlm, det: PhaseDetectorBW) -> bool:
    if not hasattr(vlm, "forward"):
        return False
    orig = vlm.forward
    def _p(*args, **kwargs):
        det.on_vlm_before(args, kwargs)
        r = orig(*args, **kwargs)
        det.on_vlm_after()
        return r
    vlm.forward = _p
    return True


def patch_lm_forward(vlm, det: PhaseDetectorBW) -> bool:
    for attr in ("language_model", "model"):
        mod = getattr(vlm, attr, None)
        if mod is None:
            continue
        if hasattr(mod, "forward"):
            sub = getattr(mod, "model", None)
            if sub is not None and hasattr(sub, "layers"):
                orig = sub.forward
                def _p(*a, _o=orig, **kw):
                    det.on_lm_before(a, kw)
                    r = _o(*a, **kw)
                    det.on_lm_after()
                    return r
                sub.forward = _p
                print(f"  [패치] vlm.{attr}.model.forward ✓")
                return True
        if hasattr(mod, "forward") and hasattr(mod, "layers"):
            orig = mod.forward
            def _p(*a, _o=orig, **kw):
                det.on_lm_before(a, kw)
                r = _o(*a, **kw)
                det.on_lm_after()
                return r
            mod.forward = _p
            print(f"  [패치] vlm.{attr}.forward ✓")
            return True
    print("  [경고] lm.forward 패치 실패")
    return False


def wrap_generate(vlm, det: PhaseDetectorBW,
                  t_vlm: CUDATimer, tok_n: list[int],
                  input_tok_len: int):
    orig = vlm.generate.__func__
    def _g(self_v, *args, **kwargs):
        det.reset()
        tok_n[0] = 0
        t_vlm.reset()
        t_vlm.start()
        torch.cuda.nvtx.range_push("Phase/VLM_Generate")
        result = orig(self_v, *args, **kwargs)
        torch.cuda.nvtx.range_pop()
        t_vlm.stop()
        det.end_generate()
        if hasattr(result, "sequences"):
            gl = result.sequences.shape[-1]
        elif isinstance(result, torch.Tensor):
            gl = result.shape[-1]
        else:
            gl = 0
        tok_n[0] = max(0, gl - input_tok_len)
        return result
    vlm.generate = types.MethodType(_g, vlm)


def wrap_diffusion(diffusion, t_flow: CUDATimer, euler_n: list[int]):
    orig_euler  = diffusion._euler
    orig_sample = diffusion.sample

    def _euler(self_d, *a, **kw):
        n = kw.get("inference_step") or getattr(self_d, "num_inference_steps", 1)
        euler_n[0] += int(n)
        torch.cuda.nvtx.range_push(f"Flow/Euler_x{euler_n[0]}")
        r = orig_euler(*a, **kw)
        torch.cuda.nvtx.range_pop()
        return r

    def _sample(*a, **kw):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("Phase/Flow")
        t_flow.start()
        r = orig_sample(*a, **kw)
        t_flow.stop()
        torch.cuda.nvtx.range_pop()
        return r

    diffusion._euler  = types.MethodType(_euler, diffusion)
    diffusion.sample  = _sample


# ──────────────────────────────────────────────────────────────────────────────
# 단일 run 실행
# ──────────────────────────────────────────────────────────────────────────────

def run_one_bw(label: str, is_warmup: bool,
               model, model_inputs: dict,
               det: PhaseDetectorBW, monitor: BandwidthMonitor,
               t_vlm: CUDATimer, t_flow: CUDATimer,
               tok_n: list[int], euler_n: list[int]) -> RunRecord:

    det.reset()
    euler_n[0] = 0
    tok_n[0]   = 0
    t_vlm.reset()
    t_flow.reset()
    torch.cuda.synchronize()

    run_nvtx = f"Warmup/{label}" if is_warmup else f"Measure/{label}"
    torch.cuda.nvtx.range_push(run_nvtx)
    torch.cuda.nvtx.range_push("Inference/Total")

    rec = RunRecord(label=label, is_warmup=is_warmup)
    rec.t_wall_start_s = time.perf_counter()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98, temperature=0.6,
            num_traj_samples=1, return_extra=True,
        )

    torch.cuda.nvtx.range_pop()  # Inference/Total
    torch.cuda.nvtx.range_pop()  # Warmup/Measure

    rec.phase_events  = list(det.phase_events)
    rec.vision_ms     = det.t_vision.ms()
    rec.lm_prefill_ms = det.t_lm_prefill.ms()
    rec.decode_ms     = det.t_decode.ms()
    rec.flow_ms       = t_flow.ms()
    rec.n_tok         = tok_n[0]

    tag = "WARMUP" if is_warmup else "  RUN "
    bw_gpu = MODEL_GB * rec.n_tok / (rec.decode_ms / 1000.0) if rec.decode_ms > 0 and rec.n_tok > 0 else 0
    print(f"\n  [{tag} {label}]")
    print(f"    Vision={rec.vision_ms:.0f}ms  LM Prefill={rec.lm_prefill_ms:.0f}ms  "
          f"Decode={rec.decode_ms:.0f}ms  Flow={rec.flow_ms:.0f}ms")
    print(f"    Decode BW = {bw_gpu:.1f} GB/s ({bw_gpu/DRAM_BW_PEAK*100:.1f}% MBU)")
    print(f"    Phase events: {len(rec.phase_events)}개")

    return rec


# ──────────────────────────────────────────────────────────────────────────────
# 대역폭 분석
# ──────────────────────────────────────────────────────────────────────────────

def assign_phases_to_samples(
    rec: RunRecord,
    samples: list[BwSample],
) -> dict[str, list[BwSample]]:
    """
    BwSample을 Phase별로 분류.
    phase_events 중 *_start / *_end 쌍을 사용한다.
    """
    def _get(name_substr: str) -> Optional[PhaseEvent]:
        for e in rec.phase_events:
            if name_substr in e.name:
                return e
        return None

    vision_s    = _get("vision_start")
    vision_e    = _get("vision_end")
    lmpf_s      = _get("lm_prefill_start")
    lmpf_e      = _get("lm_prefill_end")
    decode_s    = _get("decode_start")
    decode_e    = _get("decode_end")

    # flow는 이 rec에 포함 안 됨 (별도 wall clock 필요)
    # → t_wall_start + VLM_ms + flow_ms 로 추정
    flow_s_t = rec.t_wall_start_s + (rec.vision_ms + rec.lm_prefill_ms + rec.decode_ms) / 1000.0
    flow_e_t = flow_s_t + rec.flow_ms / 1000.0

    # 각 Phase의 (start, end) wall clock 구간
    ranges = {}
    if vision_s and vision_e:
        ranges["vision"]     = (vision_s.t_wall_s, vision_e.t_wall_s)
    if lmpf_s and lmpf_e:
        ranges["lm_prefill"] = (lmpf_s.t_wall_s, lmpf_e.t_wall_s)
    if decode_s and decode_e:
        ranges["decode"]     = (decode_s.t_wall_s, decode_e.t_wall_s)
    if rec.flow_ms > 0:
        ranges["flow"]       = (flow_s_t, flow_e_t)

    phase_samples: dict[str, list[BwSample]] = {k: [] for k in ranges}
    phase_samples["other"] = []

    for s in samples:
        assigned = False
        for phase, (ts, te) in ranges.items():
            if ts <= s.t_wall_s <= te:
                phase_samples[phase].append(s)
                assigned = True
                break
        if not assigned:
            phase_samples["other"].append(s)

    return phase_samples, ranges


def phase_stats(slist: list[BwSample]) -> dict:
    """BwSample 리스트에서 통계 계산."""
    if not slist:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0,
                "p95": 0, "cv": 0, "peak_pct": 0}
    vals = np.array([s.emc_GBps for s in slist])
    return {
        "n"       : len(vals),
        "mean"    : float(np.mean(vals)),
        "std"     : float(np.std(vals)),
        "min"     : float(np.min(vals)),
        "max"     : float(np.max(vals)),
        "p95"     : float(np.percentile(vals, 95)),
        "cv"      : float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0,
        "peak_pct": float(np.mean(vals) / DRAM_BW_PEAK * 100),
    }


def detect_spikes(samples: list[BwSample],
                  z_thresh: float = 2.5) -> list[dict]:
    """
    Z-score 기반 대역폭 급변 구간 탐지.
    Z > z_thresh 인 구간을 spike로 반환.
    """
    if len(samples) < 5:
        return []
    vals = np.array([s.emc_GBps for s in samples])
    ts   = np.array([s.t_wall_s for s in samples])
    mu, sig = np.mean(vals), np.std(vals)
    if sig < 1e-6:
        return []
    z = (vals - mu) / sig
    spikes = []
    for i, (zi, vi, ti) in enumerate(zip(z, vals, ts)):
        if abs(zi) > z_thresh:
            direction = "↑ spike" if zi > 0 else "↓ drop"
            spikes.append({
                "idx": i, "t_wall_s": float(ti),
                "bw_GBps": float(vi), "z_score": float(zi),
                "direction": direction,
            })
    return spikes


def decode_linearity(phase_samples: dict[str, list[BwSample]]) -> dict:
    """
    Decode 구간 BW의 선형성 검증.
    slope ≈ 0 → 평탄 (순수 BW-bound streaming)
    slope > 0 → KV cache 증가에 따른 BW 증가
    CV < 0.05 → 거의 선형
    """
    dec = phase_samples.get("decode", [])
    if len(dec) < 3:
        return {"sufficient_data": False}
    vals = np.array([s.emc_GBps for s in dec])
    t    = np.arange(len(vals))
    # 선형 회귀
    slope, intercept = np.polyfit(t, vals, 1)
    residuals = vals - (slope * t + intercept)
    r2 = 1 - np.var(residuals) / np.var(vals) if np.var(vals) > 0 else 0
    return {
        "sufficient_data": True,
        "n_samples"      : len(vals),
        "slope_GBps_per_step": float(slope),
        "r2"             : float(r2),
        "cv"             : float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0,
        "verdict"        : ("flat (BW-bound streaming)"      if abs(slope) < 1 and np.std(vals) < 10
                            else "slightly increasing"        if slope > 1
                            else "oscillating"                if np.std(vals) > 15
                            else "noisy"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────

def _phase_label(name: str) -> str:
    return {
        "vision"    : "Vision Enc",
        "lm_prefill": "LM Prefill",
        "decode"    : "Decode",
        "flow"      : "Flow",
    }.get(name, name)


def _add_phase_overlay(ax, measure_recs, t0, y_top, fontsize=7):
    """Phase 색상 배경 + 레이블 공통 유틸."""
    if not measure_recs:
        return
    rec = measure_recs[0]
    _, ranges = assign_phases_to_samples(rec, [])
    for phase, (ts_s, te_s) in ranges.items():
        ts_ms_s = (ts_s - t0) * 1000.0
        te_ms_s = (te_s - t0) * 1000.0
        ax.axvspan(ts_ms_s, te_ms_s,
                   color=PHASE_COLORS.get(phase, "#888888"), alpha=0.15)
        mid = (ts_ms_s + te_ms_s) / 2
        ax.text(mid, y_top, _phase_label(phase),
                ha="center", va="bottom", fontsize=fontsize,
                color=PHASE_COLORS.get(phase, "#333333"), fontweight="bold")


def plot_timeline(all_samples: list[BwSample],
                  measure_recs: list[RunRecord],
                  warmup_recs:  list[RunRecord],
                  phase_samples_list: list[tuple],
                  out_dir: Path):
    """
    Fig 1: 전체 시계열 (Thor JetPack 7 적응형)

    EMC_FREQ 없는 경우 (Thor) → 4개 서브플롯:
      (a) GPU 전력 (VDD_GPU mW) — compute vs BW-bound 판별
      (b) GPU Core 사용률 % (nvidia-smi GR3D)
      (c) CPU 평균 사용률 %
      (d) RAM 사용량 (GB)

    EMC_FREQ 있는 경우 (일반 Jetson) → (a)를 EMC BW (GB/s)로 대체
    """
    if not all_samples:
        print("  [시각화] 샘플 없음 — timeline 스킵")
        return

    ts_arr   = np.array([s.t_wall_s      for s in all_samples])
    bw_arr   = np.array([s.emc_GBps      for s in all_samples])
    gr3d_arr = np.array([s.gr3d_pct      for s in all_samples])
    cpu_arr  = np.array([s.cpu_pct       for s in all_samples])
    ram_arr  = np.array([s.ram_GB        for s in all_samples])
    gpu_pw   = np.array([s.vdd_gpu_mW    for s in all_samples])
    cpu_pw   = np.array([s.vdd_cpu_soc_mW for s in all_samples])
    vin_pw   = np.array([s.vin_mW        for s in all_samples])

    has_emc   = bw_arr.max() > 1.0          # EMC 직접 측정 여부
    has_power = gpu_pw.max() > 100.0        # 전력 데이터 여부 (idle ~-392mW → offset)
    # GPU 전력 baseline 보정 (idle 음수 오프셋 제거)
    gpu_pw_offset = gpu_pw.min()
    gpu_pw_corr   = gpu_pw - gpu_pw_offset  # 0 기준으로 보정

    t0    = ts_arr[0]
    ts_ms = (ts_arr - t0) * 1000.0

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("white")
    gs   = gridspec.GridSpec(4, 1, hspace=0.45, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    for ax in axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.18, ls="--")

    emc_note = "" if has_emc else " [EMC_FREQ 없음 — VDD_GPU 전력 표시]"
    fig.suptitle(
        f"Alpamayo 1.5 — 실시간 전력/활용률 프로파일 (Jetson AGX Thor, bf16){emc_note}\n"
        f"LPDDR5X Peak: {DRAM_BW_PEAK} GB/s  |  tegrastats 100ms 샘플링",
        fontsize=11, fontweight="bold"
    )

    # ── (a) EMC BW 또는 GPU 전력 ────────────────────────────────────────────
    ax = axes[0]
    if has_emc:
        ax.plot(ts_ms, bw_arr, color="#2166AC", lw=1.3, label="Total EMC BW (GB/s)")
        ax.axhline(DRAM_BW_PEAK,        color="red",    ls="--", lw=1, alpha=0.6,
                   label=f"Peak {DRAM_BW_PEAK} GB/s")
        ax.axhline(DRAM_BW_PEAK * 0.80, color="orange", ls=":",  lw=1, alpha=0.7,
                   label="80% BW-bound threshold")
        ax.set_ylabel("EMC BW (GB/s)", fontsize=9)
        ax.set_ylim(0, DRAM_BW_PEAK * 1.15)
        ax.set_title("(a) Total DRAM Bandwidth — EMC (CPU+GPU+IO)", fontsize=9, fontweight="bold")
        _add_phase_overlay(ax, measure_recs, t0, DRAM_BW_PEAK * 1.05)
    elif has_power:
        # VDD_GPU 전력 (보정값)
        ax.fill_between(ts_ms, gpu_pw_corr / 1000.0, color="#D65F5F", alpha=0.45,
                        label=f"VDD_GPU (W, baseline-corrected, offset={gpu_pw_offset:.0f}mW)")
        ax.plot(ts_ms, gpu_pw_corr / 1000.0, color="#C0392B", lw=1.0)
        # 전체 VIN 전력 (secondary)
        ax2 = ax.twinx()
        ax2.plot(ts_ms, vin_pw / 1000.0, color="#7F8C8D", lw=0.8, ls="--",
                 alpha=0.6, label="VIN total (W)")
        ax2.set_ylabel("System Power (W)", fontsize=8, color="#7F8C8D")
        ax2.tick_params(axis='y', labelcolor='#7F8C8D')
        ax.set_ylabel("GPU Power (W)", fontsize=9)
        ax.set_title(
            "(a) GPU 전력 소비 — BW-bound(낮음) vs Compute-bound(높음) 판별\n"
            f"    EMC_FREQ 미제공(JetPack 7 Thor) → VDD_GPU proxy 사용",
            fontsize=9, fontweight="bold"
        )
        _add_phase_overlay(ax, measure_recs, t0, ax.get_ylim()[1] * 0.92)
        # 범례 통합
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    else:
        ax.text(0.5, 0.5, "전력/BW 데이터 없음\n(tegrastats 데이터 확인 필요)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_title("(a) GPU 전력 / EMC BW — 데이터 없음", fontsize=9, fontweight="bold")

    # ── (b) GPU util (GR3D% from nvidia-smi) ────────────────────────────────
    ax = axes[1]
    gr3d_src = "nvidia-smi" if all_samples[0].gr3d_pct > 0 or gr3d_arr.max() > 0 else "없음"
    ax.fill_between(ts_ms, gr3d_arr, color="#D65F5F", alpha=0.50,
                    label=f"GR3D % ({gr3d_src})")
    ax.plot(ts_ms, gr3d_arr, color="#C0392B", lw=0.9)
    ax.set_ylabel("GPU Core Util (%)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_title(
        f"(b) GPU Core Utilization — GR3D% ({gr3d_src})\n"
        "    높음=compute-bound, 낮음이라도 메모리 BW는 최대일 수 있음",
        fontsize=9, fontweight="bold"
    )
    ax.legend(fontsize=8)
    _add_phase_overlay(ax, measure_recs, t0, 105)

    # ── (c) CPU util ────────────────────────────────────────────────────────
    ax = axes[2]
    ax.fill_between(ts_ms, cpu_arr, color="#6ACC65", alpha=0.50, label="CPU avg %")
    ax.plot(ts_ms, cpu_arr, color="#3A7A35", lw=0.9)
    # CPU+SOC 전력 secondary
    if has_power and cpu_pw.max() > 100:
        ax2c = ax.twinx()
        ax2c.plot(ts_ms, cpu_pw / 1000.0, color="#8E44AD", lw=0.8, ls=":",
                  alpha=0.7, label="VDD_CPU_SOC_MSS (W)")
        ax2c.set_ylabel("CPU+SOC+MEM Power (W)", fontsize=8, color="#8E44AD")
        ax2c.tick_params(axis='y', labelcolor='#8E44AD')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2c.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    else:
        ax.legend(fontsize=8)
    ax.set_ylabel("CPU Util (%)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_title(
        "(c) CPU 사용률 + VDD_CPU_SOC_MSS 전력\n"
        "    메모리 서브시스템(MSS) 포함 → decode 시 전력↑",
        fontsize=9, fontweight="bold"
    )
    _add_phase_overlay(ax, measure_recs, t0, 105)

    # ── (d) RAM ─────────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(ts_ms, ram_arr, color="#9370DB", lw=1.3, label="RAM used (GB)")
    ax.set_ylabel("RAM (GB)", fontsize=9)
    ax.set_ylim(max(0, ram_arr.min() - 2), ram_arr.max() + 2)
    ax.set_title(
        "(d) DRAM 점유량 추이 (가중치 + KV cache + 런타임)\n"
        "    decode 중 KV cache +481MB 증가 예상",
        fontsize=9, fontweight="bold"
    )
    ax.set_xlabel("Wall Clock Time (ms from monitor start)", fontsize=9)
    ax.legend(fontsize=8)
    _add_phase_overlay(ax, measure_recs, t0, ram_arr.max() + 1.5)

    plt.savefig(out_dir / "fig_bw_timeline.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.savefig(out_dir / "fig_bw_timeline.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  [Fig] fig_bw_timeline.png")


def plot_phase_box(phase_samples_list: list[dict],
                   measure_recs: list[RunRecord],
                   out_dir: Path):
    """
    Fig 2: Phase별 BW 분포 (박스플롯 + 주석)
    """
    # 여러 run에서 각 phase 샘플 합산
    combined: dict[str, list[float]] = {
        "vision": [], "lm_prefill": [], "decode": [], "flow": [], "other": []
    }
    for ps in phase_samples_list:
        for phase, slist in ps.items():
            if phase in combined:
                combined[phase].extend([s.emc_GBps for s in slist])

    # decode BW (CUDA Events 기반)
    cuda_bw = {}
    for rec in measure_recs:
        if rec.decode_ms > 0 and rec.n_tok > 0:
            bw = MODEL_GB * rec.n_tok / (rec.decode_ms / 1000.0)
            cuda_bw["decode"] = cuda_bw.get("decode", []) + [bw]

    phases_to_plot = [p for p in ["vision", "lm_prefill", "decode", "flow"]
                      if combined.get(p)]
    if not phases_to_plot:
        print("  [시각화] phase_box: 데이터 없음 — 스킵")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Phase별 DRAM 대역폭 분포", fontsize=12, fontweight="bold")

    # (a) Box plot
    ax = axes[0]
    ax.set_facecolor("white")
    data   = [combined[p] for p in phases_to_plot]
    labels = [_phase_label(p) for p in phases_to_plot]
    colors = [PHASE_COLORS.get(p, "#888888") for p in phases_to_plot]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(DRAM_BW_PEAK,        color="red",    ls="--", lw=1, alpha=0.6)
    ax.axhline(DRAM_BW_PEAK * 0.80, color="orange", ls=":",  lw=1, alpha=0.8,
               label="80% BW-bound threshold")
    ax.set_ylabel("DRAM Bandwidth (GB/s)", fontsize=10)
    ax.set_title("(a) EMC BW 분포 (tegrastats 샘플)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")
    ax.set_ylim(0, DRAM_BW_PEAK * 1.15)

    # 통계 주석
    for i, (p, d) in enumerate(zip(phases_to_plot, data), 1):
        if d:
            mu = np.mean(d)
            ax.text(i, mu + 5, f"{mu:.0f}", ha="center", fontsize=8,
                    color="black", fontweight="bold")

    # (b) Mean BW bar chart + CUDA Events BW
    ax = axes[1]
    ax.set_facecolor("white")
    means     = [np.mean(combined[p]) if combined[p] else 0 for p in phases_to_plot]
    stds      = [np.std(combined[p])  if combined[p] else 0 for p in phases_to_plot]
    x = np.arange(len(phases_to_plot))
    bars = ax.bar(x, means, color=colors, alpha=0.80, edgecolor="white",
                  yerr=stds, capsize=4, error_kw={"lw": 1.5})
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3,
                f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")

    # CUDA Events decode BW 오버레이 (더 정확한 값)
    if "decode" in cuda_bw and cuda_bw["decode"]:
        di = phases_to_plot.index("decode")
        cuda_mean = np.mean(cuda_bw["decode"])
        ax.scatter([di], [cuda_mean], color="black", s=80, zorder=5,
                   label=f"CUDA Events decode BW\n{cuda_mean:.1f} GB/s (정밀)")
        ax.legend(fontsize=8)

    ax.axhline(DRAM_BW_PEAK,        color="red",    ls="--", lw=1, alpha=0.6)
    ax.axhline(DRAM_BW_PEAK * 0.80, color="orange", ls=":",  lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean DRAM Bandwidth (GB/s)", fontsize=10)
    ax.set_title("(b) Phase별 평균 BW (±σ)\n+ CUDA Events 정밀 측정값", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")
    ax.set_ylim(0, DRAM_BW_PEAK * 1.15)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_dir / "fig_bw_phase_box.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.savefig(out_dir / "fig_bw_phase_box.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  [Fig] fig_bw_phase_box.png")


def plot_cpu_gpu_split(all_samples: list[BwSample],
                       measure_recs: list[RunRecord],
                       out_dir: Path):
    """
    Fig 3: CPU vs GPU 활동 split.
    Thor JetPack 7: EMC 없음 → 전력(VDD_GPU / VDD_CPU_SOC_MSS) 기반 분석.
    bwmon 있으면 BW 실측, 없으면 전력/GR3D 기반 대체 시각화.
    """
    if not all_samples:
        return

    ts_arr  = np.array([s.t_wall_s         for s in all_samples])
    bw_tot  = np.array([s.emc_GBps         for s in all_samples])
    gpu_raw = np.array([s.gpu_client_GBps  for s in all_samples])
    cpu_raw = np.array([s.cpu_client_GBps  for s in all_samples])

    has_bwmon = gpu_raw.max() > 0 or cpu_raw.max() > 0

    if not has_bwmon:
        # 추정: GPU BW = decode 구간에서 CUDA Events 값, 나머지는 비율 추정
        # GPU util (GR3D%) 기반 비례 배분
        gr3d = np.array([s.gr3d_pct for s in all_samples]) / 100.0
        # GPU BW ≈ total × (gr3d_pct / 100) × scaling_factor
        # scaling_factor 추정: decode 구간 GPU BW는 알고 있음
        if measure_recs:
            rec = measure_recs[0]
            cuda_gpu_bw = MODEL_GB * rec.n_tok / (rec.decode_ms / 1000.0) if rec.decode_ms > 0 else 0
        else:
            cuda_gpu_bw = 211.0  # 이전 측정값 fallback
        # decode 구간에서 gr3d ≈ 100% → GPU BW ≈ cuda_gpu_bw
        # 따라서: gpu_bw = bw_tot × (gr3d / 1.0) × (cuda_gpu_bw / bw_tot_decode_mean)
        # bwmon 없고 EMC도 없으면 전력 기반 추정으로 전환
        has_emc_data = bw_tot.max() > 1.0
        if not has_emc_data:
            # 전력 기반 GPU vs CPU 활동 split
            gpu_pw   = np.array([max(0.0, s.vdd_gpu_mW    - min(s.vdd_gpu_mW for s in all_samples))
                                  for s in all_samples])
            cpu_pw   = np.array([max(0.0, s.vdd_cpu_soc_mW) for s in all_samples])
            gr3d_arr = np.array([s.gr3d_pct for s in all_samples])
            t0    = ts_arr[0]
            ts_ms = (ts_arr - t0) * 1000.0

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig.patch.set_facecolor("white")
            fig.suptitle(
                "GPU vs CPU 활동 분석 (Thor JetPack 7 — EMC_FREQ 미제공)\n"
                "전력/GR3D% 기반 proxy 측정",
                fontsize=11, fontweight="bold"
            )
            # (a) 전력 stacked
            ax = axes[0]
            ax.set_facecolor("white")
            ax.fill_between(ts_ms, gpu_pw / 1000.0, color="#D65F5F", alpha=0.6,
                            label="VDD_GPU (W, baseline 보정)")
            ax.fill_between(ts_ms, cpu_pw / 1000.0, color="#6ACC65", alpha=0.35,
                            label="VDD_CPU_SOC_MSS (W, CPU+메모리서브시스템)")
            ax.set_ylabel("Power (W)", fontsize=10)
            ax.legend(fontsize=9)
            ax.set_title("(a) GPU / CPU+SOC 전력 — Compute vs BW-bound 판별\n"
                         "    Prefill(compute-bound): GPU 전력↑  |  Decode(BW-bound): GPU 전력↓",
                         fontsize=9, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, ls="--")
            _add_phase_overlay(ax, measure_recs, t0, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 10)

            # (b) GR3D%
            ax = axes[1]
            ax.set_facecolor("white")
            ax.fill_between(ts_ms, gr3d_arr, color="#D65F5F", alpha=0.5,
                            label="GR3D util %")
            ax.set_ylabel("GPU Core Util (%)", fontsize=10)
            ax.set_ylim(0, 110)
            ax.set_xlabel("Wall Clock Time (ms)", fontsize=10)
            ax.legend(fontsize=9)
            ax.set_title("(b) GPU Core Utilization (nvidia-smi GR3D%)", fontsize=9, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, ls="--")
            _add_phase_overlay(ax, measure_recs, t0, 105)

            plt.tight_layout()
            plt.savefig(out_dir / "fig_bw_cpu_gpu.png", dpi=200, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
            print(f"  [Fig] fig_bw_cpu_gpu.png (전력 proxy 모드)")
            return

        # EMC 있는 경우 기존 BW split
        decode_mask = gr3d > 0.8
        if decode_mask.any():
            scale = cuda_gpu_bw / (bw_tot[decode_mask].mean() + 1e-6)
        else:
            scale = 0.75
        gpu_est = bw_tot * gr3d * min(scale, 1.2)
        cpu_est = np.maximum(bw_tot - gpu_est, 0)
        note = "(추정치 — bwmon 미사용)"
    else:
        gpu_est = gpu_raw
        cpu_est = cpu_raw
        note = "(bwmon 실측)"

    t0    = ts_arr[0]
    ts_ms = (ts_arr - t0) * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"CPU vs GPU DRAM Bandwidth Split {note}", fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.set_facecolor("white")
    ax.fill_between(ts_ms, 0, gpu_est, color="#D65F5F", alpha=0.6, label="GPU BW (추정)")
    ax.fill_between(ts_ms, gpu_est, gpu_est + cpu_est,
                    color="#6ACC65", alpha=0.6, label="CPU+Other BW")
    ax.plot(ts_ms, bw_tot, color="#2166AC", lw=1, ls="--", label="Total EMC")
    ax.axhline(DRAM_BW_PEAK, color="red", ls=":", lw=1, alpha=0.5)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=10)
    ax.set_ylim(0, DRAM_BW_PEAK * 1.15)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, ls="--")
    ax.set_title("(a) Stacked Bandwidth: GPU / CPU+Other", fontsize=10, fontweight="bold")
    _add_phase_overlay(ax, measure_recs, t0, DRAM_BW_PEAK * 1.05)

    ax = axes[1]
    ax.set_facecolor("white")
    ratio = cpu_est / (bw_tot + 1e-6) * 100
    ax.plot(ts_ms, ratio, color="#3A7A35", lw=1.2, label="CPU BW 비율 (%)")
    ax.fill_between(ts_ms, ratio, alpha=0.3, color="#3A7A35")
    ax.set_ylabel("CPU BW / Total BW (%)", fontsize=10)
    ax.set_xlabel("Wall Clock Time (ms)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, ls="--")
    ax.set_title("(b) CPU 대역폭 비율", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_bw_cpu_gpu.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.savefig(out_dir / "fig_bw_cpu_gpu.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  [Fig] fig_bw_cpu_gpu.png")


def plot_decode_zoom(phase_samples_list: list[dict],
                     measure_recs: list[RunRecord],
                     out_dir: Path):
    """
    Fig 4: Decode 구간 확대 — BW의 선형성 / 파동 검증.
    """
    all_decode_samples: list[BwSample] = []
    for ps in phase_samples_list:
        all_decode_samples.extend(ps.get("decode", []))

    if len(all_decode_samples) < 3:
        print("  [시각화] decode_zoom: 샘플 부족 (tegrastats 100ms vs 짧은 decode) → 스킵")
        _plot_decode_fallback(measure_recs, out_dir)
        return

    vals = np.array([s.emc_GBps for s in all_decode_samples])
    t    = np.arange(len(vals)) * 0.1  # 100ms 간격 → 초

    lin  = decode_linearity({"decode": all_decode_samples})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Decode Phase Bandwidth Deep-Dive", fontsize=11, fontweight="bold")

    # (a) BW vs sample index
    ax = axes[0]
    ax.set_facecolor("white")
    ax.plot(t, vals, "o-", color="#D65F5F", lw=1.5, ms=6, label="Decode EMC BW")
    if lin.get("sufficient_data"):
        slope = lin["slope_GBps_per_step"]
        intercept = vals[0] - slope * 0
        fit_line = slope * t + intercept
        ax.plot(t, fit_line, "k--", lw=1, alpha=0.6,
                label=f"Linear fit (slope={slope:+.2f} GB/s/step)")
    ax.axhline(DRAM_BW_PEAK,        color="red",    ls=":", lw=1, alpha=0.5)
    ax.axhline(DRAM_BW_PEAK * 0.80, color="orange", ls=":", lw=1, alpha=0.7)

    # CUDA Events 정밀값 오버레이
    if measure_recs:
        cuda_bw_vals = [MODEL_GB * r.n_tok / (r.decode_ms / 1000.0)
                        for r in measure_recs if r.decode_ms > 0 and r.n_tok > 0]
        if cuda_bw_vals:
            cuda_mean = np.mean(cuda_bw_vals)
            ax.axhline(cuda_mean, color="navy", ls="-.", lw=1.5,
                       label=f"CUDA Events avg: {cuda_mean:.1f} GB/s")

    ax.set_xlabel("Time within Decode (s)", fontsize=10)
    ax.set_ylabel("DRAM BW (GB/s)", fontsize=10)
    ax.set_title("(a) Decode BW 시계열\n선형성 검증", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, ls="--")

    # (b) BW 히스토그램
    ax = axes[1]
    ax.set_facecolor("white")
    ax.hist(vals, bins=max(5, len(vals)//2),
            color="#D65F5F", alpha=0.75, edgecolor="white", label="Decode BW")
    ax.axvline(np.mean(vals), color="black", ls="--", lw=1.5,
               label=f"mean={np.mean(vals):.0f} GB/s")
    ax.axvline(DRAM_BW_PEAK * 0.80, color="orange", ls=":", lw=1,
               label="80% threshold")
    ax.set_xlabel("DRAM BW (GB/s)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"(b) Decode BW 분포\nCV={lin.get('cv', 0):.3f}  "
                 f"({lin.get('verdict', 'N/A')})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")

    plt.tight_layout(pad=1.5)
    plt.savefig(out_dir / "fig_bw_decode_zoom.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  [Fig] fig_bw_decode_zoom.png")


def _plot_decode_fallback(measure_recs: list[RunRecord], out_dir: Path):
    """
    tegrastats 100ms > decode 길이라 샘플이 없을 때:
    CUDA Events 기반 BW를 기준으로 최소 시각화.
    """
    bw_vals = [MODEL_GB * r.n_tok / (r.decode_ms / 1000.0)
               for r in measure_recs if r.decode_ms > 0 and r.n_tok > 0]
    if not bw_vals:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(range(len(bw_vals)), bw_vals, color="#D65F5F", alpha=0.8)
    ax.axhline(DRAM_BW_PEAK, color="red", ls="--")
    ax.axhline(DRAM_BW_PEAK * 0.80, color="orange", ls=":")
    ax.set_xlabel("Measure Run #")
    ax.set_ylabel("Decode BW (GB/s)")
    ax.set_title("Decode BW (CUDA Events)\n[tegrastats 해상도 부족 → 런별 평균만 표시]",
                 fontweight="bold")
    for i, v in enumerate(bw_vals):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_bw_decode_zoom.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 리포트 생성
# ──────────────────────────────────────────────────────────────────────────────

def write_report(analysis: dict, out_dir: Path):
    """Markdown 분석 리포트 생성."""
    lines = [
        "# Alpamayo 1.5 — Memory Bandwidth Analysis Report",
        f"**보드**: Jetson AGX Thor  |  **LPDDR5X Peak**: {DRAM_BW_PEAK} GB/s  |  **모델**: bf16  22.16 GB",
        "",
        "## 측정 방법",
        "- **tegrastats** (100ms 샘플링): 총 EMC 대역폭 (CPU+GPU+IO 합산)",
        "- **CUDA Events** (μs 정밀도): Decode Phase GPU BW 직접 계산",
        "- bwmon per-client: " + ("활성화" if analysis.get("bwmon_used") else "미사용 (tegrastats만)"),
        "",
        "## Phase별 대역폭 통계",
        "",
        "| Phase | 샘플 수 | 평균 BW | ±σ | 최대 | P95 | CV | EMC% | 판정 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for phase in ["vision", "lm_prefill", "decode", "flow"]:
        st = analysis.get("phase_stats", {}).get(phase, {})
        if not st or st["n"] == 0:
            lines.append(f"| {_phase_label(phase)} | 0 | — | — | — | — | — | — | 데이터 없음 |")
            continue
        verdict = ("**★ BW-BOUND**"  if st["peak_pct"] >= 80
                   else "BW-bound"   if st["peak_pct"] >= 60
                   else "compute"    if phase in ("vision", "lm_prefill")
                   else "mixed")
        lines.append(
            f"| {_phase_label(phase)} | {st['n']} "
            f"| {st['mean']:.1f} GB/s | ±{st['std']:.1f} "
            f"| {st['max']:.1f} | {st['p95']:.1f} "
            f"| {st['cv']:.3f} | {st['peak_pct']:.1f}% | {verdict} |"
        )

    # CUDA Events decode
    cuda = analysis.get("cuda_events_decode", {})
    if cuda:
        lines += [
            "",
            "## Decode BW (CUDA Events — 고정밀)",
            "",
            f"- **평균**: {cuda.get('mean', 0):.2f} GB/s",
            f"- **±σ**: {cuda.get('std', 0):.2f} GB/s",
            f"- **MBU**: {cuda.get('mean', 0) / DRAM_BW_PEAK * 100:.1f}% of {DRAM_BW_PEAK} GB/s",
            f"- **토큰**: {cuda.get('n_tok', 0):.0f} tok",
        ]

    # Spike 분석
    spikes = analysis.get("spikes", [])
    lines += [
        "",
        f"## 대역폭 급변 구간 (Z > 2.5σ): {len(spikes)}건",
        "",
    ]
    if spikes:
        lines.append("| 시각 (s) | BW (GB/s) | Z-score | 방향 |")
        lines.append("|---|---|---|---|")
        for sp in spikes[:10]:
            lines.append(
                f"| {sp['t_wall_s']:.3f} | {sp['bw_GBps']:.1f} | {sp['z_score']:+.2f} | {sp['direction']} |"
            )
    else:
        lines.append("급변 구간 없음 (대역폭이 안정적)")

    # Decode 선형성
    lin = analysis.get("decode_linearity", {})
    lines += [
        "",
        "## Decode 구간 선형성 분석",
        "",
        f"- **샘플 수**: {lin.get('n_samples', 0)}",
        f"- **Slope**: {lin.get('slope_GBps_per_step', 0):+.3f} GB/s/step",
        f"- **R²**: {lin.get('r2', 0):.4f}",
        f"- **CV**: {lin.get('cv', 0):.4f}",
        f"- **결론**: {lin.get('verdict', 'insufficient data')}",
        "",
        "> KV cache 증가 효과: decode step당 1 토큰 추가 → BW ≈ +model_GB×1tok/step_ms (무시할 수준)",
        "",
        "## 해석 요약",
        "",
        "| 구간 | CPU BW 기여 | 비고 |",
        "|---|---|---|",
        "| Vision Encoder | 낮음 (<5 GB/s) | GPU compute-bound |",
        "| LM Prefill | 낮음 (<5 GB/s) | GPU compute-bound, GEMM 지배 |",
        "| Decode | 중간 (5~15 GB/s) | GPU BW-bound, CPU는 EOS 체크 등 housekeeping |",
        "| Flow | 낮음~중간 | 64-tile GEMM, partial compute-bound |",
        "",
        "*Note: Jetson Thor는 unified memory이므로 CPU BW = Total EMC - GPU BW로 추정.*",
    ]

    rpt = out_dir / "bw_report.md"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [저장] {rpt}")


# ──────────────────────────────────────────────────────────────────────────────
# 메인 프로파일링
# ──────────────────────────────────────────────────────────────────────────────

def run_profiling(warmup: int = 1, runs: int = 2,
                  interval_ms: int = 100,
                  use_bwmon: bool = False):

    print("\n[1/5] 모델 로드...")
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    t0 = time.perf_counter()
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16,
    ).cuda().eval()
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - t0
    print(f"  모델 로드: {model_load_s:.1f}s")

    print("[2/5] 입력 준비...")
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
    model_inputs = helper.to_device({
        "tokenized_data"  : inputs,
        "ego_history_xyz" : data["ego_history_xyz"],
        "ego_history_rot" : data["ego_history_rot"],
    }, "cuda")
    input_tok_len = inputs["input_ids"].shape[-1]
    print(f"  입력 토큰: {input_tok_len}")

    print("[3/5] 패치 등록...")
    det     = PhaseDetectorBW()
    t_vlm   = CUDATimer()
    t_flow  = CUDATimer()
    tok_n   = [0]
    euler_n = [0]
    patch_vlm_forward(model.vlm, det)
    patch_lm_forward(model.vlm, det)
    wrap_generate(model.vlm, det, t_vlm, tok_n, input_tok_len)
    wrap_diffusion(model.diffusion, t_flow, euler_n)

    print("[4/5] BW 모니터 시작...")
    monitor = BandwidthMonitor(interval_ms=interval_ms, use_bwmon=use_bwmon)
    monitor.start()

    # 모니터링 시작 후 약간 대기해서 baseline 샘플 수집
    time.sleep(0.5)

    print("[5/5] 프로파일링...")
    all_recs: list[RunRecord] = []

    for i in range(warmup):
        r = run_one_bw(f"run_{i+1:02d}", True,
                       model, model_inputs, det, monitor,
                       t_vlm, t_flow, tok_n, euler_n)
        all_recs.append(r)
        time.sleep(0.3)  # run간 BW가 0으로 돌아오는지 관찰

    measure_recs: list[RunRecord] = []
    for i in range(runs):
        r = run_one_bw(f"run_{i+1:02d}", False,
                       model, model_inputs, det, monitor,
                       t_vlm, t_flow, tok_n, euler_n)
        all_recs.append(r)
        measure_recs.append(r)
        time.sleep(0.3)

    # 마지막 몇 샘플 더 수집 (post-inference baseline)
    time.sleep(1.0)
    monitor.stop()
    all_samples = monitor.drain()

    if all_samples:
        span_s = all_samples[-1].t_wall_s - all_samples[0].t_wall_s
        print(f"\n  수집된 BW 샘플 총계: {len(all_samples)}개 "
              f"(간격 {interval_ms}ms, span {span_s:.1f}s)")
    else:
        print(f"\n  수집된 BW 샘플: 0개 — CUDA Events 기반 결과만 출력됩니다.")

    return all_recs, measure_recs, all_samples, input_tok_len, model_load_s


# ──────────────────────────────────────────────────────────────────────────────
# 결과 집계 / 저장
# ──────────────────────────────────────────────────────────────────────────────

def analyze_and_save(all_recs, measure_recs, all_samples,
                     input_tok_len, model_load_s):

    warmup_recs = [r for r in all_recs if r.is_warmup]

    # ── Phase별 BW 분류 ──────────────────────────────────────────────────────
    phase_samples_list = []
    for rec in measure_recs:
        ps, ranges = assign_phases_to_samples(rec, all_samples)
        phase_samples_list.append(ps)

    # ── 통계 분석 ─────────────────────────────────────────────────────────────
    combined: dict[str, list[BwSample]] = {
        k: [] for k in ["vision", "lm_prefill", "decode", "flow", "other"]
    }
    for ps in phase_samples_list:
        for k in combined:
            combined[k].extend(ps.get(k, []))

    phase_stats_dict = {k: phase_stats(v) for k, v in combined.items()}
    spikes = detect_spikes(all_samples)
    lin    = decode_linearity(combined)

    # CUDA Events decode BW
    cuda_bw_vals = [MODEL_GB * r.n_tok / (r.decode_ms / 1000.0)
                    for r in measure_recs if r.decode_ms > 0 and r.n_tok > 0]
    cuda_events_decode = {
        "mean": float(np.mean(cuda_bw_vals)) if cuda_bw_vals else 0,
        "std" : float(np.std(cuda_bw_vals))  if cuda_bw_vals else 0,
        "n_tok": float(np.mean([r.n_tok for r in measure_recs])),
        "mbu_pct": float(np.mean(cuda_bw_vals) / DRAM_BW_PEAK * 100) if cuda_bw_vals else 0,
    }

    analysis = {
        "dram_bw_peak_GBps"  : DRAM_BW_PEAK,
        "model_gb"           : MODEL_GB,
        "input_tok_len"      : input_tok_len,
        "model_load_s"       : model_load_s,
        "bwmon_used"         : any(s.gpu_client_GBps > 0 for s in all_samples) if all_samples else False,
        "n_samples_total"    : len(all_samples),
        "n_measure_runs"     : len(measure_recs),
        "phase_stats"        : phase_stats_dict,
        "cuda_events_decode" : cuda_events_decode,
        "spikes"             : spikes,
        "decode_linearity"   : lin,
        "warmup_vs_measure"  : {
            "vision_ms"    : {"w": np.mean([r.vision_ms     for r in warmup_recs]) if warmup_recs else 0,
                              "m": np.mean([r.vision_ms     for r in measure_recs])},
            "lm_prefill_ms": {"w": np.mean([r.lm_prefill_ms for r in warmup_recs]) if warmup_recs else 0,
                              "m": np.mean([r.lm_prefill_ms for r in measure_recs])},
            "decode_ms"    : {"w": np.mean([r.decode_ms     for r in warmup_recs]) if warmup_recs else 0,
                              "m": np.mean([r.decode_ms     for r in measure_recs])},
            "flow_ms"      : {"w": np.mean([r.flow_ms       for r in warmup_recs]) if warmup_recs else 0,
                              "m": np.mean([r.flow_ms       for r in measure_recs])},
        },
    }

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    ts_data = [{
        "t_wall_s":        s.t_wall_s,
        "emc_pct":         s.emc_pct,
        "emc_GBps":        s.emc_GBps,
        "gr3d_pct":        s.gr3d_pct,
        "cpu_pct":         s.cpu_pct,
        "ram_GB":          s.ram_GB,
        "vdd_gpu_mW":      s.vdd_gpu_mW,       # KEY: compute vs BW-bound proxy
        "vdd_cpu_soc_mW":  s.vdd_cpu_soc_mW,   # CPU+SOC+Memory Subsystem
        "vin_mW":          s.vin_mW,            # total board power
        "emc_GBps_derived":s.emc_GBps_derived,  # CUDA-calibrated estimate
        "gpu_client_GBps": s.gpu_client_GBps,
        "cpu_client_GBps": s.cpu_client_GBps,
    } for s in all_samples]

    (OUT / "bw_timeseries.json").write_text(
        json.dumps({"samples": ts_data}, indent=2, default=float))
    (OUT / "bw_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=float))
    print(f"  [저장] bw_timeseries.json  bw_analysis.json")

    # ── 터미널 요약 출력 ──────────────────────────────────────────────────────
    W = 80
    print(f"\n{'═'*W}")
    print(f"  DRAM Bandwidth Analysis — Alpamayo 1.5 on Jetson AGX Thor")
    print(f"{'═'*W}")
    print(f"  {'Phase':<14} {'샘플':>5}  {'평균 BW':>10}  {'±σ':>8}  {'피크%':>7}  {'판정'}")
    print(f"  {'-'*14} {'-'*5}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*20}")
    for phase in ["vision", "lm_prefill", "decode", "flow"]:
        st = phase_stats_dict.get(phase, {})
        if not st or st["n"] == 0:
            print(f"  {_phase_label(phase):<14} {'—':>5}  {'데이터 없음':>10}")
            continue
        verd = ("★ BW-BOUND"    if st["peak_pct"] >= 80
                else "BW-bound" if st["peak_pct"] >= 60
                else "compute"  if phase in ("vision", "lm_prefill")
                else "mixed")
        print(f"  {_phase_label(phase):<14} {st['n']:>5}  "
              f"{st['mean']:>8.1f} GB/s  ±{st['std']:>5.1f}  "
              f"{st['peak_pct']:>6.1f}%  {verd}")
    print(f"\n  CUDA Events Decode: {cuda_events_decode['mean']:.1f} GB/s "
          f"(MBU {cuda_events_decode['mbu_pct']:.1f}%)")
    print(f"  BW 급변 구간: {len(spikes)}건")
    if lin.get("sufficient_data"):
        print(f"  Decode 선형성: {lin['verdict']}  (CV={lin['cv']:.4f})")
    print(f"{'═'*W}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    if all_samples:
        plot_timeline(all_samples, measure_recs, warmup_recs, phase_samples_list, FIGD)
        plot_cpu_gpu_split(all_samples, measure_recs, FIGD)
    else:
        print("  [시각화] BW 샘플 없음 → timeline/cpu_gpu 그림 스킵")
    plot_phase_box(phase_samples_list, measure_recs, FIGD)
    plot_decode_zoom(phase_samples_list, measure_recs, FIGD)
    write_report(analysis, OUT)

    print(f"\n  [완료] 결과 디렉토리: {OUT}")
    print(f"  그림: {FIGD}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Alpamayo 1.5 DRAM Bandwidth Profiler"
    )
    ap.add_argument("--warmup",      type=int,  default=1,
                    help="Warmup 횟수 (기본 1)")
    ap.add_argument("--runs",        type=int,  default=2,
                    help="측정 횟수 (기본 2)")
    ap.add_argument("--interval-ms", type=int,  default=100,
                    help="tegrastats 샘플링 간격 ms (기본 100, 최소 100)")
    ap.add_argument("--bwmon",       action="store_true",
                    help="bwmon per-client BW 활성화 (sudo 필요)")
    ap.add_argument("--nsys",        action="store_true",
                    help="nsys 모드 (안내 메시지 억제)")
    args = ap.parse_args()

    if not args.nsys:
        print("=" * 72)
        print("  [이 스크립트 실행 방법]")
        print()
        print("  # 기본 실행")
        print(f"  python {Path(__file__).name} --warmup 1 --runs 2")
        print()
        print("  # bwmon 활성화 (per-client GPU/CPU 분리)")
        print(f"  sudo python {Path(__file__).name} --bwmon --warmup 1 --runs 2")
        print()
        print("  # nsys 동시 캡처")
        print(f"  nsys profile --trace=cuda,nvtx \\")
        print(f"    --output=profiling_results/260514_bw/nsys_bw \\")
        print(f"    python {Path(__file__).name} --nsys --warmup 1 --runs 2")
        print("=" * 72)

    all_recs, measure_recs, all_samples, input_tok_len, model_load_s = \
        run_profiling(warmup=args.warmup, runs=args.runs,
                      interval_ms=args.interval_ms, use_bwmon=args.bwmon)

    analyze_and_save(all_recs, measure_recs, all_samples,
                     input_tok_len, model_load_s)


if __name__ == "__main__":
    main()
