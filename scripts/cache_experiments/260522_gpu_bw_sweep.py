"""
260522_gpu_bw_sweep.py — GPU L2 vs DRAM Bandwidth Characterization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[v1 → v2 fix]
  v1: separate x.sum() calls per measurement.
      Problem: CUDA kernel launch overhead (~5 µs) dominates for small tensors.
      A 4 MB tensor reads in ~16 µs, but overhead adds ~5 µs → measured BW
      is artificially low. Bandwidth appeared to "ramp up" from small to large
      — that was just overhead amortization, NOT a real cache effect.

  v2: CUDA Graphs. N_GRAPH_ITERS reads are bundled into one graph replay.
      Launch overhead is paid once per replay, not per iteration.
      Within a single replay of 100 iterations:
        - Iter 1  : miss  → load from DRAM → fills L2
        - Iters 2-100: hit → read from L2 (if tensor ≤ L2 = 32 MB)
      The average per-iter BW reveals whether data is L2-resident or DRAM-bound.

Expected result (v2):
  - sizes < ~32 MB  → iterations 2-100 hit L2  → high BW (>> DRAM)
  - sizes > ~32 MB  → all iterations miss L2    → BW ≈ DRAM (~247 GB/s)

Run on Thor (no model needed):
    python3 ~/alpamayo1.5/scripts/cache_experiments/260522_gpu_bw_sweep.py

Output: ~/alpamayo1.5/profiling_results/260522_gpu_cache/bw_sweep.json
"""

import json
from pathlib import Path

import numpy as np
import torch

N_WARMUP       = 5    # separate launches before graph capture
N_GRAPH_ITERS  = 100  # reads bundled inside one CUDA graph replay
N_MEASURE      = 20   # graph replays to average over


def measure_read_bw(size_bytes: int) -> tuple[float, float]:
    """
    Measures per-iteration read bandwidth using CUDA graphs.
    Within each graph replay, N_GRAPH_ITERS sum() calls run back-to-back
    without Python or kernel-launch overhead between them.
    For tensors that fit in L2, iterations 2+ are served from L2.
    Returns (mean_bw_gbps, std_bw_gbps) using trimmed mean (drop top/bottom 10%).
    """
    n_elem = size_bytes // 4   # float32
    x = torch.zeros(n_elem, dtype=torch.float32, device="cuda")
    y = torch.empty(n_elem, dtype=torch.float32, device="cuda")  # copy destination
    torch.cuda.synchronize()

    # Warmup: loads x (and y) into L2 if they fit together (x+y ≤ L2 = 32 MB)
    for _ in range(N_WARMUP):
        y.copy_(x)
    torch.cuda.synchronize()

    # Capture N_GRAPH_ITERS copies into a single CUDA graph.
    # y.copy_(x) reads all of x and writes all of y — pure bandwidth operation.
    # Within one replay, iterations 2-N hit L2 if x+y fits in L2 (2×size ≤ 32 MB).
    # BW = 2 × size / time  (counts read of x + write of y).
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(N_GRAPH_ITERS):
            y.copy_(x)

    times_ms: list[float] = []
    for _ in range(N_MEASURE):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        g.replay()
        end_ev.record()
        torch.cuda.synchronize()
        times_ms.append(start_ev.elapsed_time(end_ev) / N_GRAPH_ITERS)

    del x, y, g
    torch.cuda.empty_cache()

    arr = np.array(times_ms)
    lo, hi   = np.percentile(arr, [10, 90])
    trimmed  = arr[(arr >= lo) & (arr <= hi)]
    mean_ms  = trimmed.mean() if len(trimmed) > 0 else arr.mean()
    std_ms   = trimmed.std()  if len(trimmed) > 0 else arr.std()

    mean_bw = 2 * size_bytes / (mean_ms * 1e-3) / 1e9  # ×2: read x + write y
    std_bw  = mean_bw * std_ms / mean_ms if mean_ms > 0 else 0.0
    return mean_bw, std_bw


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"

    props = torch.cuda.get_device_properties(0)
    l2_bytes = props.L2_cache_size
    l2_mb = l2_bytes / 1024 ** 2

    print(f"Device : {props.name}")
    print(f"SM     : {props.major}.{props.minor}  |  SMs: {props.multi_processor_count}")
    print(f"L2     : {l2_mb:.2f} MB  ({l2_bytes:,} bytes)")
    print(f"Total  : {props.total_memory / 1024**3:.1f} GB\n")

    # y.copy_(x) working set = x + y = 2×size.
    # L2 cliff is at 2×size = L2 = 32 MB  →  size = 16 MB.
    # Dense sweep around 16 MB boundary.
    sizes_bytes = [
        512 * 1024,          # 512 KB
        1 * 1024 ** 2,       # 1 MB
        2 * 1024 ** 2,       # 2 MB
        4 * 1024 ** 2,       # 4 MB
        8 * 1024 ** 2,       # 8 MB
        10 * 1024 ** 2,      # 10 MB
        12 * 1024 ** 2,      # 12 MB
        14 * 1024 ** 2,      # 14 MB  ← approaching L2/2
        16 * 1024 ** 2,      # 16 MB  = L2/2 (x+y = 32 MB = L2)
        18 * 1024 ** 2,      # 18 MB  ← just over L2/2
        24 * 1024 ** 2,      # 24 MB
        32 * 1024 ** 2,      # 32 MB
        48 * 1024 ** 2,      # 48 MB
        64 * 1024 ** 2,      # 64 MB
        96 * 1024 ** 2,      # 96 MB
        128 * 1024 ** 2,     # 128 MB
        256 * 1024 ** 2,     # 256 MB
    ]

    results: list[dict] = []

    print(f"{'Size':>8}  {'BW (GB/s)':>12}  {'±Std':>8}  Note")
    print("─" * 55)

    dram_bw_samples: list[float] = []

    for size in sizes_bytes:
        label = (
            f"{size // 1024}KB" if size < 1024 ** 2
            else f"{size // 1024 ** 2}MB"
        )
        bw, std = measure_read_bw(size)

        note = ""
        if size == l2_bytes // 2:
            note = "◄ L2 boundary (x+y = L2)"
        elif size == l2_bytes // 2 + 2 * 1024 ** 2:
            note = "◄ just over"

        if size >= 128 * 1024 ** 2:
            dram_bw_samples.append(bw)

        print(f"{label:>8}  {bw:>12.1f}  {std:>8.1f}  {note}")
        results.append({
            "size_bytes": size,
            "label": label,
            "bw_gbps": round(bw, 2),
            "std_gbps": round(std, 2),
        })

    dram_bw = float(np.mean(dram_bw_samples)) if dram_bw_samples else None
    if dram_bw:
        # v2: small tensors are L2-resident → high BW; large tensors → DRAM BW
        # Report the peak L2 BW (smallest sub-L2 tensor, fully L2-resident after warmup)
        sub_l2 = [r for r in results if r["size_bytes"] <= l2_bytes // 4]
        if sub_l2:
            l2_bw = max(r["bw_gbps"] for r in sub_l2)
            best  = max(sub_l2, key=lambda r: r["bw_gbps"])
            print(f"\nPeak L2 BW (at {best['label']:>4})   : {l2_bw:.1f} GB/s")
            print(f"Estimated DRAM BW           : {dram_bw:.1f} GB/s")
            print(f"L2 / DRAM speedup           : {l2_bw / dram_bw:.2f}×")

    out_dir = Path("profiling_results/260522_gpu_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bw_sweep.json"
    out_path.write_text(json.dumps({
        "l2_cache_mb": l2_mb,
        "dram_bw_estimate_gbps": dram_bw,
        "results": results,
    }, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
