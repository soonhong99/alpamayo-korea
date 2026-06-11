"""
260522_cpu_bw_sweep.py — CPU Cache Hierarchy Verification via Bandwidth Sweep
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[v1 → v2 fix]
  v1: np.sum() → compute-bound (float32 addition ~15 GB/s ceiling regardless of
      where data lives). Cache hierarchy invisible.
  v2: np.copyto(dst, src) → pure memory copy, no arithmetic → bandwidth-bound.
      Expected cliffs: L1D (~80 GB/s) > L2 (~40 GB/s) > L3 (~20 GB/s) > DRAM.
      Outlier filtering via trimmed mean (drop top/bottom 10%).

Pinned to a single core so L1/L2 are per-core (not shared).
L3 (16 MB official) should appear as a cliff around 16 MB.

Run on Thor (no model, no GPU needed):
    python3 ~/alpamayo1.5/scripts/cache_experiments/260522_cpu_bw_sweep.py

Output: ~/alpamayo1.5/profiling_results/260522_gpu_cache/cpu_bw_sweep.json
"""

import json
import os
import time
from pathlib import Path

import numpy as np

N_WARMUP = 5
N_MEASURE = 30  # trimmed mean uses middle 80% → need enough samples


def pin_to_core(core: int = 0) -> None:
    try:
        os.sched_setaffinity(0, {core})
    except (AttributeError, OSError):
        pass


def parse_cache_size(s: str) -> int:
    s = s.strip()
    if s.endswith("K"):
        return int(s[:-1]) * 1024
    if s.endswith("M"):
        return int(s[:-1]) * 1024 * 1024
    return int(s)


def read_sys_caches() -> list[dict]:
    cache_base = Path("/sys/devices/system/cpu/cpu0/cache")
    if not cache_base.exists():
        return []
    caches = []
    seen: set[tuple] = set()
    for idx_dir in sorted(cache_base.glob("index*")):
        try:
            level = int((idx_dir / "level").read_text().strip())
            ctype = (idx_dir / "type").read_text().strip()
            size_str = (idx_dir / "size").read_text().strip()
            shared = (idx_dir / "shared_cpu_list").read_text().strip()
            size_bytes = parse_cache_size(size_str)
            key = (level, ctype, size_bytes)
            if key in seen:
                continue
            seen.add(key)
            caches.append({
                "level": level,
                "type": ctype,
                "size_str": size_str,
                "size_bytes": size_bytes,
                "shared_cpus": shared,
            })
        except Exception:
            pass
    return caches


def measure_cpu_copy_bw(size_bytes: int) -> tuple[float, float]:
    """
    Pure memory copy: np.copyto(dst, src).
    No arithmetic → bandwidth-bound, not compute-bound.
    Bytes transferred = 2 × size_bytes (one read + one write).
    Uses trimmed mean (drop top/bottom 10%) to reject OS-interrupt outliers.
    Returns (mean_bw_gbps, std_bw_gbps).
    """
    n_elem = size_bytes // 4  # float32
    src = np.ones(n_elem, dtype=np.float32)
    dst = np.empty_like(src)

    for _ in range(N_WARMUP):
        np.copyto(dst, src)

    times: list[float] = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    del src, dst
    t = np.array(times)
    # Trimmed mean: drop slowest 10% (OS interrupts) and fastest 10% (noise)
    lo, hi = np.percentile(t, [10, 90])
    trimmed = t[(t >= lo) & (t <= hi)]
    mean_t = trimmed.mean() if len(trimmed) > 0 else t.mean()
    std_t = trimmed.std() if len(trimmed) > 0 else t.std()

    mean_bw = 2 * size_bytes / mean_t / 1e9   # ×2: read src + write dst
    std_bw = mean_bw * (std_t / mean_t) if mean_t > 0 else 0.0
    return mean_bw, std_bw


def main() -> None:
    pin_to_core(0)

    print("=== CPU Cache Info from /sys ===")
    sys_caches = read_sys_caches()
    if sys_caches:
        for c in sys_caches:
            print(f"  L{c['level']} {c['type']:12s}: {c['size_str']:8s}  shared_by={c['shared_cpus']}")
    else:
        print("  (not available in /sys)")

    print("\n=== Copy Bandwidth Sweep (pinned to core 0) ===")
    print("  Method: np.copyto(dst, src) — pure memory copy, no arithmetic")
    print("  BW = 2×size / time  (counts both read and write)")
    print(f"\n{'Size':>8}  {'BW (GB/s)':>12}  {'±Std':>8}  Note")
    print("─" * 60)

    sizes_bytes = [
        8 * 1024,           # 8 KB
        16 * 1024,          # 16 KB
        32 * 1024,          # 32 KB
        64 * 1024,          # 64 KB   ← L1D boundary
        128 * 1024,         # 128 KB
        256 * 1024,         # 256 KB
        512 * 1024,         # 512 KB
        1 * 1024 ** 2,      # 1 MB    ← L2/core boundary
        2 * 1024 ** 2,      # 2 MB
        4 * 1024 ** 2,      # 4 MB
        8 * 1024 ** 2,      # 8 MB
        12 * 1024 ** 2,     # 12 MB
        16 * 1024 ** 2,     # 16 MB   ← L3 boundary (official)
        24 * 1024 ** 2,     # 24 MB
        32 * 1024 ** 2,     # 32 MB
        64 * 1024 ** 2,     # 64 MB
        128 * 1024 ** 2,    # 128 MB
        256 * 1024 ** 2,    # 256 MB
    ]

    BOUNDARY_NOTES = {
        64 * 1024:       "◄ L1D boundary (expected)",
        1 * 1024 ** 2:   "◄ L2/core boundary (expected)",
        16 * 1024 ** 2:  "◄ L3 boundary (official spec)",
    }

    results: list[dict] = []
    for size in sizes_bytes:
        label = (
            f"{size // 1024}KB" if size < 1024 ** 2
            else f"{size // 1024 ** 2}MB"
        )
        bw, std = measure_cpu_copy_bw(size)
        note = BOUNDARY_NOTES.get(size, "")
        print(f"{label:>8}  {bw:>12.2f}  {std:>8.2f}  {note}")
        results.append({
            "size_bytes": size,
            "label": label,
            "bw_gbps": round(bw, 3),
            "std_gbps": round(std, 3),
        })

    out_dir = Path("profiling_results/260522_gpu_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cpu_bw_sweep.json"
    out_path.write_text(json.dumps({
        "sys_caches": sys_caches,
        "method": "np.copyto (pure copy, 2×size bytes transferred). v1 used np.sum() which was compute-bound.",
        "note": "Pinned to core 0. Bandwidth cliffs indicate cache boundaries.",
        "results": results,
    }, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
