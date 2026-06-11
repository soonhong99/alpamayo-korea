"""
tegrastats_monitor.py
─────────────────────
Jetson AGX Thor 전용 하드웨어 모니터.
tegrastats 출력을 파싱해서 추론 중 CPU/GPU/메모리 사용률을 시계열로 기록한다.

사용법:
    # 백그라운드에서 실행 후 추론 시작
    python tegrastats_monitor.py --output profiling_results/tegrastats.json &
    MONITOR_PID=$!
    python profile_alpamayo.py
    kill $MONITOR_PID

    # 또는 duration 지정
    python tegrastats_monitor.py --duration 120 --interval 100

tegrastats 출력 예시:
    RAM 45623/131072MB (lfb 1x4MB) SWAP 0/65536MB CPU [45%@2035,45%@2035,...] \
    GR3D_FREQ 98%@1300 GPU@62.8C VDD_IN 79264mW/79264mW
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# tegrastats 파서
# ──────────────────────────────────────────────
class TegrastatsParser:
    """
    tegrastats 한 줄을 파싱해서 딕셔너리로 반환.
    Thor (JetPack 7) 포맷 기준.
    """

    # RAM 45623/131072MB
    RAM_PATTERN = re.compile(r"RAM\s+(\d+)/(\d+)MB")
    # SWAP 0/65536MB
    SWAP_PATTERN = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
    # CPU [45%@2035,45%@2035,off,off,...]
    CPU_PATTERN = re.compile(r"CPU\s+\[([^\]]+)\]")
    # GR3D_FREQ 98%@1300
    GPU_FREQ_PATTERN = re.compile(r"GR3D_FREQ\s+(\d+)%@(\d+)")
    # GPU@62.8C
    GPU_TEMP_PATTERN = re.compile(r"GPU@([\d.]+)C")
    # CPU@52.3C
    CPU_TEMP_PATTERN = re.compile(r"CPU@([\d.]+)C")
    # VDD_IN 79264mW/79264mW
    POWER_PATTERN = re.compile(r"VDD_IN\s+(\d+)mW/(\d+)mW")
    # VDD_CPU_GPU_CV 74120mW/74120mW
    GPU_POWER_PATTERN = re.compile(r"VDD_CPU_GPU_CV\s+(\d+)mW/(\d+)mW")
    # EMC_FREQ 0%@6400  (메모리 컨트롤러)
    EMC_PATTERN = re.compile(r"EMC_FREQ\s+(\d+)%@(\d+)")

    @classmethod
    def parse(cls, line: str) -> Optional[dict]:
        line = line.strip()
        if not line:
            return None

        result = {"timestamp": time.time()}

        # RAM
        m = cls.RAM_PATTERN.search(line)
        if m:
            used, total = int(m.group(1)), int(m.group(2))
            result["ram_used_mb"]  = used
            result["ram_total_mb"] = total
            result["ram_used_pct"] = round(used / total * 100, 1)

        # SWAP
        m = cls.SWAP_PATTERN.search(line)
        if m:
            result["swap_used_mb"]  = int(m.group(1))
            result["swap_total_mb"] = int(m.group(2))

        # CPU cores
        m = cls.CPU_PATTERN.search(line)
        if m:
            cores = m.group(1).split(",")
            active_loads = []
            active_freqs = []
            for c in cores:
                c = c.strip()
                if c.lower() == "off":
                    continue
                cm = re.match(r"(\d+)%@(\d+)", c)
                if cm:
                    active_loads.append(int(cm.group(1)))
                    active_freqs.append(int(cm.group(2)))
            if active_loads:
                result["cpu_avg_load_pct"]  = round(sum(active_loads) / len(active_loads), 1)
                result["cpu_max_load_pct"]  = max(active_loads)
                result["cpu_avg_freq_mhz"]  = round(sum(active_freqs) / len(active_freqs), 0)
                result["cpu_core_loads"]    = active_loads

        # GPU utilization
        m = cls.GPU_FREQ_PATTERN.search(line)
        if m:
            result["gpu_util_pct"]  = int(m.group(1))
            result["gpu_freq_mhz"]  = int(m.group(2))

        # EMC (메모리 대역폭 컨트롤러)
        m = cls.EMC_PATTERN.search(line)
        if m:
            result["emc_util_pct"] = int(m.group(1))
            result["emc_freq_mhz"] = int(m.group(2))

        # 온도
        m = cls.GPU_TEMP_PATTERN.search(line)
        if m:
            result["gpu_temp_c"] = float(m.group(1))

        m = cls.CPU_TEMP_PATTERN.search(line)
        if m:
            result["cpu_temp_c"] = float(m.group(1))

        # 전력
        m = cls.POWER_PATTERN.search(line)
        if m:
            result["total_power_mw"] = int(m.group(1))

        m = cls.GPU_POWER_PATTERN.search(line)
        if m:
            result["gpu_power_mw"] = int(m.group(1))

        return result


# ──────────────────────────────────────────────
# 모니터 메인 루프
# ──────────────────────────────────────────────
class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 100, output_path: Path = None):
        self.interval_ms = interval_ms
        self.output_path = output_path or Path("profiling_results/tegrastats.json")
        self.records: list[dict] = []

    def run(self, duration_sec: Optional[float] = None):
        """
        tegrastats를 subprocess로 실행하고 실시간 파싱.
        duration_sec이 None이면 Ctrl+C 또는 SIGTERM까지 계속.
        """
        cmd = ["tegrastats", f"--interval", str(self.interval_ms)]
        print(f"[Monitor] tegrastats 시작 (interval={self.interval_ms}ms)")

        start_time = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                record = TegrastatsParser.parse(line)
                if record:
                    self.records.append(record)
                    self._print_live(record)

                if duration_sec and (time.time() - start_time) >= duration_sec:
                    proc.terminate()
                    break

        except KeyboardInterrupt:
            if "proc" in dir():
                proc.terminate()
            print("\n[Monitor] 중단됨")

        self._save()

    def _print_live(self, r: dict):
        ram_pct  = r.get("ram_used_pct", 0)
        gpu_util = r.get("gpu_util_pct", 0)
        cpu_avg  = r.get("cpu_avg_load_pct", 0)
        power    = r.get("total_power_mw", 0)
        print(
            f"\r  RAM {ram_pct:5.1f}%  GPU {gpu_util:3d}%  "
            f"CPU {cpu_avg:5.1f}%  {power/1000:.1f}W",
            end="", flush=True
        )

    def _save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 통계 요약
        def avg(key):
            vals = [r[key] for r in self.records if key in r]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        def peak(key):
            vals = [r[key] for r in self.records if key in r]
            return max(vals) if vals else 0.0

        summary = {
            "records": self.records,
            "summary": {
                "duration_sec":      round(
                    (self.records[-1]["timestamp"] - self.records[0]["timestamp"])
                    if len(self.records) > 1 else 0, 1
                ),
                "num_samples":        len(self.records),
                "ram_avg_used_pct":   avg("ram_used_pct"),
                "ram_peak_used_mb":   peak("ram_used_mb"),
                "cpu_avg_load_pct":   avg("cpu_avg_load_pct"),
                "cpu_peak_load_pct":  peak("cpu_max_load_pct"),
                "gpu_avg_util_pct":   avg("gpu_util_pct"),
                "gpu_peak_util_pct":  peak("gpu_util_pct"),
                "emc_avg_util_pct":   avg("emc_util_pct"),
                "gpu_avg_temp_c":     avg("gpu_temp_c"),
                "gpu_peak_temp_c":    peak("gpu_temp_c"),
                "avg_total_power_w":  round(avg("total_power_mw") / 1000, 2),
                "avg_gpu_power_w":    round(avg("gpu_power_mw") / 1000, 2),
            },
        }

        with open(self.output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Monitor] 저장 완료: {self.output_path}")
        s = summary["summary"]
        print(f"  GPU 평균 이용률: {s['gpu_avg_util_pct']:.1f}%  "
              f"피크: {s['gpu_peak_util_pct']:.1f}%")
        print(f"  RAM 평균 사용:   {s['ram_avg_used_pct']:.1f}%  "
              f"피크: {s['ram_peak_used_mb']} MB")
        print(f"  총 소비 전력:    {s['avg_total_power_w']:.1f}W")


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Jetson Thor tegrastats 모니터")
    parser.add_argument("--interval", type=int, default=100,
                        help="샘플링 간격 (ms, 기본 100)")
    parser.add_argument("--duration", type=float, default=None,
                        help="기록 시간(초). 미지정 시 Ctrl+C까지")
    parser.add_argument("--output", type=str,
                        default="profiling_results/tegrastats.json")
    args = parser.parse_args()

    monitor = TegrastatsMonitor(
        interval_ms=args.interval,
        output_path=Path(args.output),
    )
    monitor.run(args.duration)


if __name__ == "__main__":
    main()
