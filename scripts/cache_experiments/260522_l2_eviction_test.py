"""
260522_l2_eviction_test.py — Does a Weight Matrix Stay in L2 Between Decode Steps?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Simulates the decode-phase access pattern:
  For each of 10 steps (like autoregressive decode):
    1. Read "other layers" (to simulate the rest of the network flushing L2)
    2. Read the "target weight" and measure bandwidth

If the weight matrix is small (< L2 = 32 MB) AND the flush is smaller than
L2, the weight may survive in L2 → high bandwidth on re-access.
If the flush completely evicts the weight → bandwidth drops to DRAM bandwidth.

This experiment directly answers:
  "Can any weight matrix stay L2-resident across decode steps?"

Expected results:
  - 4 MB weight + 4 MB flush  → weight survives in L2 → high BW
  - 4 MB weight + 64 MB flush → weight evicted        → DRAM BW
  - 33 MB weight (LM Q-proj)  → borderline; mostly evicted even with small flush
  - 200 MB weight (LM layer)  → always DRAM BW

Also measures the theoretical DRAM bandwidth baseline using a large tensor.

Run on Thor (no model needed):
    python3 ~/alpamayo1.5/scripts/cache_experiments/260522_l2_eviction_test.py

Output: ~/alpamayo1.5/profiling_results/260522_gpu_cache/l2_eviction_test.json
"""

import json
from pathlib import Path

import numpy as np
import torch

N_STEPS = 10          # simulate 10 decode steps per scenario
DTYPE = torch.bfloat16
BYTES_PER_ELEM = 2


def make_tensor(size_mb: float) -> torch.Tensor:
    n = int(size_mb * 1024 * 1024 / BYTES_PER_ELEM)
    return torch.zeros(n, dtype=DTYPE, device="cuda")


def read_bw_single(t: torch.Tensor) -> float:
    """Single timed read, returns GB/s."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = t.sum()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return t.numel() * BYTES_PER_ELEM / (elapsed_ms * 1e-3) / 1e9


N_GRAPH_ITERS = 100  # iterations inside each CUDA graph replay


def measure_bw_cuda_graph(t: torch.Tensor) -> float:
    """
    Measures read bandwidth using a CUDA graph with N_GRAPH_ITERS iterations.
    Within one replay, iterations 2-N hit L2 if tensor fits.
    Returns GB/s (read-only: size bytes per iteration).
    """
    out = torch.empty_like(t)
    # Warmup
    for _ in range(3):
        out.copy_(t)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(N_GRAPH_ITERS):
            out.copy_(t)  # read t, write out: 2×size per iter

    # Measure
    times_ms: list[float] = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end) / N_GRAPH_ITERS)

    del out, g
    arr = np.array(times_ms)
    mean_ms = np.median(arr)  # median to reject outliers
    # ×2: counts read of t + write of out
    return 2 * t.numel() * BYTES_PER_ELEM / (mean_ms * 1e-3) / 1e9


def measure_dram_baseline(size_mb: float = 256.0) -> float:
    """
    DRAM baseline: tensor >> L2, measured with CUDA graph.
    L2 cannot hold 256 MB tensor (L2 = 32 MB), so all accesses go to DRAM.
    """
    t = make_tensor(size_mb)
    bw = measure_bw_cuda_graph(t)
    del t
    torch.cuda.empty_cache()
    return bw


def simulate_decode(
    weight_mb: float,
    flush_mb: float,
    n_steps: int = N_STEPS,
) -> dict:
    """
    Simulate n_steps decode steps with CUDA-graph-based measurement.

    Each step:
      1. Read flush tensor (single kernel — simulates other-layer computation,
         potentially evicting weight from L2)
      2. Measure weight bandwidth using CUDA graph (N_GRAPH_ITERS iterations
         in one replay)

    Why CUDA graph for weight measurement?
      - Without CUDA graph: single kernel can't distinguish fast L2 from slow
        DRAM due to GPU bus saturation issues for small tensors
      - With CUDA graph: within one replay, iterations 2-N hit L2 if weight
        fits → much higher BW than DRAM baseline
      - If weight was evicted by flush in step 1: all iterations miss L2
        → BW drops to DRAM baseline level
    """
    weight = make_tensor(weight_mb)
    flush  = make_tensor(flush_mb)
    torch.cuda.synchronize()

    # Pre-build CUDA graph for weight measurement
    weight_bw_graph_out = torch.empty_like(weight)
    for _ in range(3):
        weight_bw_graph_out.copy_(weight)
    torch.cuda.synchronize()

    g_weight = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_weight):
        for _ in range(N_GRAPH_ITERS):
            weight_bw_graph_out.copy_(weight)

    bw_steps: list[float] = []
    for _ in range(n_steps):
        # Step 1: flush — read flush tensor (single kernel, evicts L2 if flush > L2)
        _ = flush.sum()
        torch.cuda.synchronize()

        # Step 2: measure weight BW via graph replay
        times_ms: list[float] = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            g_weight.replay()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end) / N_GRAPH_ITERS)
        mean_ms = float(np.median(times_ms))
        bw = 2 * weight.numel() * BYTES_PER_ELEM / (mean_ms * 1e-3) / 1e9
        bw_steps.append(bw)

    del weight, flush, weight_bw_graph_out, g_weight
    torch.cuda.empty_cache()

    return {
        "bw_per_step_gbps": [round(b, 1) for b in bw_steps],
        "mean_bw_gbps": round(float(np.mean(bw_steps)), 1),
        "std_bw_gbps": round(float(np.std(bw_steps)), 1),
        "min_bw_gbps": round(float(np.min(bw_steps)), 1),
        "max_bw_gbps": round(float(np.max(bw_steps)), 1),
    }


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"

    props = torch.cuda.get_device_properties(0)
    l2_mb = props.L2_cache_size / 1024 ** 2
    print(f"Device : {props.name}")
    print(f"L2     : {l2_mb:.1f} MB\n")

    print("Measuring DRAM baseline bandwidth (256 MB tensor) ...")
    dram_bw = measure_dram_baseline(256.0)
    print(f"DRAM baseline: {dram_bw:.1f} GB/s\n")

    # Weight scenarios
    # Format: (label, weight_mb, description)
    weight_scenarios = [
        ("small_4mb",      4.0,  "Small weight (fits in L2 comfortably)"),
        ("medium_16mb",   16.0,  "Medium weight (fits in L2)"),
        ("lm_qproj_33mb", 33.0,  "LM Q-proj approx (d_model=4096, BF16) — at L2 boundary"),
        ("lm_layer_200mb",200.0, "Approx one full LM transformer layer — 6× over L2"),
    ]

    # Flush scenarios (how much "other layer" data is read between accesses)
    flush_mbs = [4.0, 32.0, 64.0, 200.0]

    all_results: dict = {
        "hardware": {
            "device": props.name,
            "l2_mb": l2_mb,
            "dram_baseline_gbps": dram_bw,
        },
        "scenarios": {},
    }

    for label, weight_mb, description in weight_scenarios:
        fits = weight_mb <= l2_mb
        print(f"── {label}  ({weight_mb:.0f} MB)  {'[fits L2]' if fits else '[> L2]'}")
        print(f"   {description}")
        print(f"   {'flush_mb':>10}  {'mean BW':>10}  {'std':>8}  {'vs DRAM':>10}  interpretation")
        print("   " + "─" * 65)

        scenario_results: dict = {
            "weight_mb": weight_mb,
            "description": description,
            "fits_l2": fits,
            "flush_tests": {},
        }

        for flush_mb in flush_mbs:
            result = simulate_decode(weight_mb, flush_mb, N_STEPS)
            ratio = result["mean_bw_gbps"] / dram_bw

            # With CUDA graph, BW is the average across N_GRAPH_ITERS iters.
            # If weight is in L2 (not evicted by flush): iters 2-N hit L2 → BW >> DRAM
            # If weight evicted: all iters miss → BW ≈ DRAM baseline
            if ratio > 3.0:
                interp = "L2 resident ✓"
            elif ratio > 1.5:
                interp = "partial L2"
            else:
                interp = "DRAM (evicted)"

            print(f"   {flush_mb:>9.0f}M  "
                  f"{result['mean_bw_gbps']:>10.1f}  "
                  f"{result['std_bw_gbps']:>8.1f}  "
                  f"{ratio:>9.2f}×  {interp}")

            scenario_results["flush_tests"][f"flush_{flush_mb:.0f}mb"] = result
        print()
        all_results["scenarios"][label] = scenario_results

    out_dir = Path("profiling_results/260522_gpu_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "l2_eviction_test.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
