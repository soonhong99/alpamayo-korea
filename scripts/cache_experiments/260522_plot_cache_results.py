"""
260522_plot_cache_results.py — Visualize Cache Experiment Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run AFTER the other 3 experiments have produced JSON results.
Reads from profiling_results/260522_gpu_cache/ and generates:
  - fig1_gpu_bw_sweep.pdf/png
  - fig2_cpu_bw_sweep.pdf/png
  - fig3_l2_eviction.pdf/png

Run on Thor:
    python3 ~/alpamayo1.5/scripts/cache_experiments/260522_plot_cache_results.py

Or on Windows after scp-ing the JSON files back.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULT_DIR = Path("profiling_results/260522_gpu_cache")
FIG_DIR = RESULT_DIR / "figures"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


def plot_bw_sweep(json_path: Path, out_prefix: str, title: str, boundary_labels: dict[int, str]) -> None:
    data = json.loads(json_path.read_text())
    results = data["results"]

    sizes_mb = [r["size_bytes"] / 1024 ** 2 for r in results]
    bws = [r["bw_gbps"] for r in results]
    stds = [r["std_gbps"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(sizes_mb, bws, yerr=stds, marker="o", linewidth=1.8,
                markersize=5, capsize=4, color="#2563eb", ecolor="#93c5fd")

    # Shade the L2 / L3 regions
    for boundary_bytes, label in boundary_labels.items():
        boundary_mb = boundary_bytes / 1024 ** 2
        ax.axvline(x=boundary_mb, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(boundary_mb * 1.03, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else max(bws) * 0.95,
                label, color="#dc2626", fontsize=9, va="top")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Tensor / Array Size (MB)")
    ax.set_ylabel("Read Bandwidth (GB/s)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x*1024:.0f}K"
    ))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{out_prefix}.{ext}")
    plt.close(fig)
    print(f"  Saved {out_prefix}.{{png,pdf}}")


def plot_l2_eviction(json_path: Path) -> None:
    data = json.loads(json_path.read_text())
    dram_bw = data["hardware"]["dram_baseline_gbps"]
    l2_mb = data["hardware"]["l2_mb"]
    scenarios = data["scenarios"]

    flush_labels = ["4MB", "32MB", "64MB", "200MB"]
    flush_keys = [f"flush_{f}mb" for f in ["4", "32", "64", "200"]]

    scenario_labels = list(scenarios.keys())
    n_scenarios = len(scenario_labels)
    n_flush = len(flush_keys)

    fig, axes = plt.subplots(1, n_scenarios, figsize=(4 * n_scenarios, 5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]

    for ax, label in zip(axes, scenario_labels):
        sc = scenarios[label]
        weight_mb = sc["weight_mb"]
        fits = sc["fits_l2"]
        flush_tests = sc["flush_tests"]

        means = []
        stds = []
        for fk in flush_keys:
            if fk in flush_tests:
                means.append(flush_tests[fk]["mean_bw_gbps"])
                stds.append(flush_tests[fk]["std_bw_gbps"])
            else:
                means.append(0.0)
                stds.append(0.0)

        colors = ["#16a34a" if m > dram_bw * 1.3 else "#dc2626" for m in means]
        bars = ax.bar(flush_labels, means, yerr=stds, capsize=4, color=colors, alpha=0.85, width=0.6)
        ax.axhline(y=dram_bw, color="#6b7280", linestyle="--", linewidth=1.2, label=f"DRAM baseline ({dram_bw:.0f} GB/s)")
        ax.axhline(y=dram_bw * 2, color="#2563eb", linestyle=":", linewidth=1.0, alpha=0.5)

        title_color = "#16a34a" if fits else "#dc2626"
        ax.set_title(f"{label}\n({weight_mb:.0f} MB, {'≤' if fits else '>'} L2={l2_mb:.0f}MB)",
                     color=title_color, fontsize=10)
        ax.set_xlabel("Flush size between accesses")
        if ax is axes[0]:
            ax.set_ylabel("Read BW (GB/s)")
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("L2 Eviction Test: Weight BW vs Flush Size\n(green = L2 resident; red = evicted to DRAM)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig3_l2_eviction.{ext}")
    plt.close(fig)
    print("  Saved fig3_l2_eviction.{png,pdf}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    gpu_path = RESULT_DIR / "bw_sweep.json"
    cpu_path = RESULT_DIR / "cpu_bw_sweep.json"
    evict_path = RESULT_DIR / "l2_eviction_test.json"

    if gpu_path.exists():
        print("Plotting GPU bandwidth sweep ...")
        gpu_data = json.loads(gpu_path.read_text())
        l2_bytes = int(gpu_data["l2_cache_mb"] * 1024 ** 2)
        plot_bw_sweep(
            gpu_path,
            "fig1_gpu_bw_sweep",
            "GPU Memory Bandwidth vs Tensor Size (Jetson AGX Thor)",
            {l2_bytes: f"L2 = {gpu_data['l2_cache_mb']:.0f}MB"},
        )
    else:
        print(f"  Skipping GPU sweep (not found: {gpu_path})")

    if cpu_path.exists():
        print("Plotting CPU bandwidth sweep ...")
        plot_bw_sweep(
            cpu_path,
            "fig2_cpu_bw_sweep",
            "CPU Memory Bandwidth vs Array Size (Jetson AGX Thor, core 0)",
            {
                64 * 1024:        "L1D = 64KB",
                1 * 1024 ** 2:    "L2 = 1MB/core",
                16 * 1024 ** 2:   "L3 = 16MB",
            },
        )
    else:
        print(f"  Skipping CPU sweep (not found: {cpu_path})")

    if evict_path.exists():
        print("Plotting L2 eviction test ...")
        plot_l2_eviction(evict_path)
    else:
        print(f"  Skipping eviction test (not found: {evict_path})")

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
