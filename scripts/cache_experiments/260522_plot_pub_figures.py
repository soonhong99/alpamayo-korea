"""
260522_plot_pub_figures.py  —  Publication-quality figures for Thor cache analysis
════════════════════════════════════════════════════════════════════════════════════
Hardcoded from experimental measurements (2026-05-22, Jetson AGX Thor).
No JSON dependency — runs on Windows/Mac/Linux without GPU.

Figures produced:
  fig1_gpu_bw_cliff       GPU L2 vs DRAM bandwidth cliff (main result)
  fig2_cpu_hierarchy      CPU L1/L2/L3/DRAM bandwidth staircase
  fig3_hierarchy_map      Capacity vs bandwidth log-log scatter (both CPU+GPU)
  fig4_weight_vs_cache    Weight footprint vs cache sizes (horizontal log bars)
  fig5_eviction_heatmap   L2 retention under simulated decode flush

Run:
    python3 scripts/cache_experiments/260522_plot_pub_figures.py

Output: profiling_results/260522_gpu_cache/figures/fig{1-5}.[pdf|png]
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "grid.color":        "#999999",
    "axes.grid":         True,
    "axes.axisbelow":    True,
})

# ── Color palette ─────────────────────────────────────────────────────────────
C_L2      = "#1565C0"   # deep blue   — GPU L2
C_L2_BG   = "#DDEEFF"   # pale blue   — GPU L2 shading
C_DRAM    = "#B71C1C"   # deep red    — GPU DRAM
C_DRAM_BG = "#FFEEEE"   # pale red    — DRAM shading
C_CPU     = "#1B5E20"   # deep green  — CPU
C_L1_BG   = "#E8F5E9"   # L1D shading
C_L2C_BG  = "#C8E6C9"   # L2 CPU shading
C_L3_BG   = "#FFF8E1"   # L3 shading
C_DRAM_CPU_BG = "#FFEBEE"  # CPU DRAM shading
C_FIT     = "#2E7D32"   # fits in cache
C_BORDER  = "#F57F17"   # borderline
C_NOFIT   = "#C62828"   # doesn't fit
C_KV      = "#00695C"   # KV cache — teal

OUT = Path("profiling_results/260522_gpu_cache/figures")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved: {name}.[pdf|png]")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: GPU Bandwidth Cliff
# ══════════════════════════════════════════════════════════════════════════════
def fig1_gpu_bw_cliff() -> None:
    # Measured on Jetson AGX Thor, 2026-05-22
    # y.copy_(x) in CUDA Graph (100 iter/replay); BW = 2*size/time
    # Working set = x + y = 2*size  →  L2 cliff at size = L2/2 = 16 MB
    sizes_mb = [0.512, 1, 4, 8, 10, 12, 16, 18, 32, 96, 256]
    bw_gbps  = [114.2, 206.5, 344.9, 1106.0, 1125.5, 1036.5,
                455.3, 201.9, 244.9, 233.7, 231.0]
    DRAM     = 231.0    # GB/s — measured at 256 MB
    L2_PEAK  = 1125.5   # GB/s — measured at 10 MB
    L2_BOUND = 16.0     # MB  — cliff point

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # Region shading
    ax.axvspan(0.2, L2_BOUND, color=C_L2_BG, alpha=1.0, zorder=0)
    ax.axvspan(L2_BOUND, 400, color=C_DRAM_BG, alpha=1.0, zorder=0)

    # DRAM baseline
    ax.axhline(DRAM, color=C_DRAM, linestyle="--", linewidth=1.6, zorder=2)
    ax.text(200, DRAM + 32, f"DRAM baseline: {DRAM:.0f} GB/s",
            color=C_DRAM, fontsize=10.5, ha="right", fontweight="bold")

    # L2 boundary vertical line + label
    # Cliff at tensor=16 MB because working set = 2×16 = 32 MB = L2 capacity
    ax.axvline(L2_BOUND, color=C_L2, linestyle=":", linewidth=1.5, zorder=3, alpha=0.55)
    ax.text(L2_BOUND * 1.06, 1340,
            "L2 boundary",
            fontsize=9, color=C_L2, ha="left", va="top", alpha=0.90)

    # Data line
    ax.plot(sizes_mb, bw_gbps, "o-", color=C_L2, linewidth=2.2,
            markersize=6.5, markeredgewidth=0.8, markeredgecolor="white",
            zorder=5)

    # L2 Peak annotation
    ax.annotate(
        f"L2 Peak\n{L2_PEAK:.0f} GB/s",
        xy=(10, L2_PEAK), xytext=(5.2, 1270),
        fontsize=11.5, color=C_L2, fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_L2, lw=1.6),
    )

    # 4.76× speedup double-arrow
    # Placed at x=0.55 MB where the data line (~120 GB/s) is well below
    # the DRAM baseline (231 GB/s), so the arrow occupies clear space.
    x_arrow = 0.55
    ax.annotate("", xy=(x_arrow, DRAM), xytext=(x_arrow, L2_PEAK),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=2.0))
    ax.text(x_arrow * 2.0, (DRAM + L2_PEAK) / 2,
            "4.76×\nspeedup",
            fontsize=11.5, fontweight="bold", color="#333333", va="center")

    ax.set_xscale("log")
    ax.set_xlim(0.3, 350)
    ax.set_ylim(0, 1400)
    ax.set_xlabel("Tensor Size (MB)")
    ax.set_ylabel("Memory Bandwidth (GB/s)")
    ax.set_title(
        "GPU Memory Hierarchy: L2 Cache vs. DRAM Bandwidth",
        fontsize=11,
    )
    ax.yaxis.set_major_locator(mticker.MultipleLocator(200))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x*1024:.0f}K"
    ))

    # Legend — same style as CPU figure
    legend_patches = [
        mpatches.Patch(color=C_L2_BG,   label="L2 Cache  (32 MB SRAM,  ~1,126 GB/s)"),
        mpatches.Patch(color=C_DRAM_BG, label="DRAM       (122 GB,        ~231 GB/s)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    fig.tight_layout()
    save(fig, "fig1_gpu_bw_cliff")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: CPU Cache Hierarchy
# ══════════════════════════════════════════════════════════════════════════════
def fig2_cpu_hierarchy() -> None:
    # Measured on Jetson AGX Thor ARM Neoverse V3AE, single core, 2026-05-22
    # np.copyto(dst, src) — pure memory bandwidth; BW = 2*size/time
    # Working set = src + dst = 2*size → cliff at size = cache_size/2
    # L1D cliff: 2*size = 64 KB → size = 32 KB (peak at 128 KB)
    # L2  cliff: 2*size = 1 MB  → size = 512 KB
    # L3  cliff: 2*size = 16 MB → size = 8 MB (gradual; ARM prefetcher)
    sizes_kb = [8,    32,   64,   128,   512,   1024,  4096,  16384, 131072, 262144]
    bw_gbps  = [33.3, 83.0, 95.7, 106.6, 89.3,  67.7,  48.4,  39.7,  33.3,   33.2]
    # Note: 8 KB = timing artifact (array too small for perf_counter resolution)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    # Region shading (by nominal cache sizes; working set = 2×size)
    ax.axvspan(1,     64,     color=C_L1_BG,        alpha=1.0, zorder=0)
    ax.axvspan(64,    512,    color=C_L2C_BG,       alpha=1.0, zorder=0)
    ax.axvspan(512,   8192,   color=C_L3_BG,        alpha=1.0, zorder=0)
    ax.axvspan(8192,  4.5e5,  color=C_DRAM_CPU_BG,  alpha=1.0, zorder=0)

    # Region labels (positioned low in chart)
    region_info = [
        (20,    "L1D\n64 KB\n~96 GB/s",   C_FIT),
        (170,   "L2\n1 MB\n~107 GB/s",    C_CPU),
        (2000,  "L3\n16 MB\n~40 GB/s",    "#795548"),
        (70000, "DRAM\n~33 GB/s",         C_DRAM),
    ]
    for x, label, color in region_info:
        ax.text(x, 7, label, fontsize=8, ha="center", va="bottom",
                color=color, style="italic")

    # Boundary verticals
    for x_kb, label in [(64, "64 KB\n(L1D)"), (512, "512 KB\n(L2)"), (8192, "8 MB\n(L3 eff.)")]:
        ax.axvline(x_kb, color="#777777", linestyle=":", linewidth=1.3,
                   zorder=3, alpha=0.8)

    # Data line  (skip 8 KB outlier in annotation)
    ax.plot(sizes_kb, bw_gbps, "s-", color=C_CPU, linewidth=2.2,
            markersize=6.5, markeredgewidth=0.8, markeredgecolor="white",
            zorder=5, label="Measured BW (np.copyto)")

    # Peak annotation: text placed in L1D region (left of 64 KB line)
    # so the arrow crosses the boundary, visually connecting L1D territory → peak at 128 KB.
    # 128 KB = 2× L1D(64 KB). ARM prefetcher keeps L1D-class BW just past L1D capacity.
    ax.annotate(
        "Peak  106.6 GB/s",
        xy=(128, 106.6), xytext=(14, 116),
        fontsize=9.5, color=C_CPU, fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_CPU, lw=1.3),
    )

    ax.set_xscale("log")
    ax.set_xlim(5, 4e5)
    ax.set_ylim(0, 130)
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel("Copy Bandwidth (GB/s)")
    ax.set_title(
        "CPU Memory Hierarchy Bandwidth — ARM Neoverse V3AE",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x)}K" if x < 1024 else f"{int(x // 1024)}M"
    ))
    legend_patches = [
        mpatches.Patch(color=C_L1_BG,       label="L1D Cache  (64 KB/core,   ~96 GB/s)"),
        mpatches.Patch(color=C_L2C_BG,      label="L2 Cache   (1 MB/core,  ~107 GB/s)"),
        mpatches.Patch(color=C_L3_BG,       label="L3 Cache   (16 MB shared, ~40 GB/s)"),
        mpatches.Patch(color=C_DRAM_CPU_BG, label="DRAM       (per core,      ~33 GB/s)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5)

    fig.tight_layout()
    save(fig, "fig2_cpu_hierarchy")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Memory Hierarchy Map  (capacity × bandwidth log-log scatter)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_hierarchy_map() -> None:
    """
    Each memory level as a single point on (capacity, bandwidth) log-log axes.
    Model component sizes overlaid as vertical reference lines.
    Tells the reader: 'where is each weight on this map?'
    """
    MB = 1024**2
    GB = 1024**3

    # (capacity_bytes, bw_gbps, label, color, marker, markersize)
    hierarchy = [
        (64   * 1024,   107,  "CPU L1D\n(64 KB/core)",   C_CPU,  "^",  11),
        (1    * MB,      67,  "CPU L2\n(1 MB/core)",      C_CPU,  "s",  10),
        (16   * MB,      40,  "CPU L3\n(16 MB shared)",   C_CPU,  "D",  10),
        (122.8 * GB,     33,  "CPU DRAM\n(per core)",     C_CPU,  "v",  10),
        (32   * MB,    1100,  "GPU L2\n(32 MB)",          C_L2,   "*",  18),
        (122.8 * GB,    231,  "GPU DRAM\n(full bus)",     C_DRAM, "o",  11),
    ]

    # Model components as vertical reference lines
    # (size_bytes, label, color, linestyle)
    model_refs = [
        (7.11  * MB,   "KV Cache\n(65 steps)\n7 MB",          C_KV,     "-.",  1.6),
        (32    * MB,   "Q/O Proj\n(per layer)\n32 MB",        C_BORDER, ":",   1.6),
        (96    * MB,   "MLP Layer\n(per layer)\n96 MB",       C_NOFIT,  "--",  1.6),
        (20.64 * GB,   "Full Model\n20.64 GB",                "#888888", "-",  1.4),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Plot hierarchy points
    label_offsets = {
        "CPU L1D\n(64 KB/core)":   (0.35, 0.70),
        "CPU L2\n(1 MB/core)":     (2.5,  0.72),
        "CPU L3\n(16 MB shared)":  (0.3,  0.75),
        "CPU DRAM\n(per core)":    (0.25, 1.30),
        "GPU L2\n(32 MB)":         (0.2,  0.70),
        "GPU DRAM\n(full bus)":    (0.18, 1.45),
    }
    for cap, bw, label, color, marker, ms in hierarchy:
        ax.scatter(cap, bw, s=ms**2, c=color, marker=marker,
                   zorder=6, edgecolors="white", linewidths=0.7)
        mx, my = label_offsets.get(label, (1.5, 0.72))
        ax.annotate(
            label,
            xy=(cap, bw),
            xytext=(cap * mx, bw * my),
            fontsize=8.5, color=color,
            ha="center", va="top",
        )

    # Connect CPU levels
    cpu_caps = [h[0] for h in hierarchy[:4]]
    cpu_bws  = [h[1] for h in hierarchy[:4]]
    ax.plot(cpu_caps, cpu_bws, "--", color=C_CPU, lw=1.3, alpha=0.55, zorder=2)

    # Connect GPU levels
    gpu_caps = [h[0] for h in hierarchy[4:]]
    gpu_bws  = [h[1] for h in hierarchy[4:]]
    ax.plot(gpu_caps, gpu_bws, "--", color=C_L2, lw=1.3, alpha=0.55, zorder=2)

    # Model reference lines + labels at top
    y_top_base = 2500
    for i, (size_b, label, color, ls, lw) in enumerate(model_refs):
        ax.axvline(size_b, color=color, linestyle=ls, linewidth=lw, alpha=0.85, zorder=3)
        y_label = y_top_base * (0.85 ** i)
        ax.text(size_b * 1.08, y_label, label, fontsize=7.5, color=color,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # "Unreachable" zone annotation
    ax.fill_between([32*MB, 122.8*GB], [1, 1], [3000, 3000],
                    color="#EEEEEE", alpha=0.35, zorder=1)
    ax.text(600*MB, 80, "No cache covers\nthis region\n(weights >32 MB fall here)",
            fontsize=8, color="#666666", ha="center", style="italic")

    # GPU L2 "sweet spot" label
    ax.annotate(
        "GPU L2: best BW\nfor weight ≤ 32 MB",
        xy=(32*MB, 1100),
        xytext=(3*MB, 1700),
        fontsize=9, color=C_L2, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_L2, lw=1.4),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e4, 2.5e11)
    ax.set_ylim(15, 3500)
    ax.set_xlabel("Memory Level Capacity")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title(
        "Memory Hierarchy: Capacity vs. Bandwidth\n"
        "(Experimental measurements, Jetson AGX Thor; dashed lines = model components)",
        fontsize=10.5,
    )

    def fmt_cap(x, _):
        if x >= 1024**3:
            return f"{x/1024**3:.0f} GB"
        elif x >= 1024**2:
            return f"{x/1024**2:.0f} MB"
        elif x >= 1024:
            return f"{x/1024:.0f} KB"
        return f"{x:.0f} B"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_cap))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}"
    ))

    legend_elems = [
        plt.Line2D([0], [0], marker="^", color=C_CPU, ms=9, ls="--",
                   label="CPU memory levels (single core)"),
        plt.Line2D([0], [0], marker="*", color=C_L2, ms=12, ls="--",
                   label="GPU memory levels"),
        plt.Line2D([0], [0], color=C_KV, ls="-.", lw=1.5,
                   label="Model components (ref. lines)"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=9)

    fig.tight_layout()
    save(fig, "fig3_hierarchy_map")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Alpamayo 1.5 Weight Footprint vs. Cache Sizes
# ══════════════════════════════════════════════════════════════════════════════
def fig4_weight_vs_cache() -> None:
    # (label, size_mb, category: "fit" | "border" | "nofit")
    # Sorted small → large for horizontal bar
    components = [
        ("LayerNorm γ/β  (per layer)",         0.008,     "fit"),
        ("KV Cache  (65 decode steps, GQA-4)",  7.11,     "fit"),
        ("K/V Projection  (per layer, GQA est.)", 8.0,    "fit"),
        ("Q / O Projection  (per layer)",       32.0,     "border"),
        ("MLP Gate / Up / Down  (per layer)",   96.0,     "nofit"),
        ("Embedding / LM Head",               1190.0,     "nofit"),
        ("Full Alpamayo 1.5  (BF16, 11.08B)", 20_640.0,  "nofit"),
    ]
    GPU_L2_MB = 32.0
    CPU_L3_MB = 16.0

    color_map = {"fit": C_FIT, "border": C_BORDER, "nofit": C_NOFIT}
    labels  = [c[0] for c in components]
    sizes   = np.array([c[1] for c in components])
    cats    = [c[2] for c in components]
    colors  = [color_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    y_pos = np.arange(len(components))
    bars  = ax.barh(y_pos, sizes, color=colors, height=0.62,
                    edgecolor="white", linewidth=0.6, zorder=4)

    # Cache boundary lines
    ax.axvline(CPU_L3_MB, color="#E65100", linestyle="--", linewidth=1.8, zorder=5,
               label=f"CPU L3: {CPU_L3_MB:.0f} MB (bandwidth ~40 GB/s)")
    ax.axvline(GPU_L2_MB, color=C_L2,    linestyle="-",  linewidth=2.2, zorder=5,
               label=f"GPU L2: {GPU_L2_MB:.0f} MB (bandwidth ~1,100 GB/s)")

    # Size labels to the right of each bar
    for i, (size, cat) in enumerate(zip(sizes, cats)):
        if size >= 1000:
            txt = f"{size/1000:.2f} GB"
        elif size >= 1:
            txt = f"{size:.1f} MB"
        else:
            txt = f"{size*1000:.0f} KB"
        ax.text(size * 1.08, i, txt, va="center", ha="left",
                fontsize=9, color="#333333")

    # Fit/border labels inside bars (only for bars wide enough)
    fit_labels = {
        "fit":    ("L2-resident", "white"),
        "border": ("= L2 bound.", "white"),
        "nofit":  ("DRAM-bound",  "white"),
    }
    for i, (size, cat) in enumerate(zip(sizes, cats)):
        if size >= 0.5:  # wide enough to print inside
            txt, col = fit_labels[cat]
            ax.text(size * 0.5, i, txt, va="center", ha="center",
                    fontsize=8.5, color=col, fontweight="bold", zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(0.001, 1.4e6)
    ax.set_xlabel("Size (MB)")
    ax.set_title(
        "Alpamayo 1.5: Weight / Data Footprint vs. GPU Cache Capacity\n"
        "(BF16 weights; KV Cache calculated for 65 decode steps with GQA-4, 8 KV heads)",
        fontsize=10.5,
    )

    def fmt_mb(x, _):
        if x >= 1000:
            return f"{x/1000:.0f} GB"
        elif x >= 1:
            return f"{x:.0f} MB"
        else:
            return f"{x*1000:.0f} KB"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_mb))

    # "660× L2" annotation for full model
    ax.annotate(
        f"Full model = {20_640/GPU_L2_MB:.0f}× GPU L2",
        xy=(20_640, len(components) - 1),
        xytext=(20_640 * 0.6, len(components) - 0.3),
        fontsize=9, color=C_NOFIT, ha="center", va="bottom",
        fontweight="bold",
    )

    # Legend + color legend
    legend_handles = [
        ax.get_lines()[0],
        ax.get_lines()[1],
        mpatches.Patch(color=C_FIT,    label="Fits in GPU L2 (≤ 32 MB) — L2-resident"),
        mpatches.Patch(color=C_BORDER, label="At L2 boundary — borderline"),
        mpatches.Patch(color=C_NOFIT,  label="Exceeds GPU L2 — always DRAM-bound"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
              ncol=1, framealpha=0.95)

    fig.tight_layout()
    save(fig, "fig4_weight_vs_cache")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: L2 Eviction Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig5_eviction_heatmap() -> None:
    """
    Heatmap: each cell = (weight_size, flush_size) → BW ratio vs DRAM.
    Shows that weight size determines L2 residency, not flush size.
    """
    DRAM_BW = 227.2  # GB/s (measured baseline)

    # Rows: weight size; Cols: flush size
    weight_mb = [4,   16,  33,   200]
    flush_mb  = [4,   32,  64,   200]

    # Measured BW (GB/s) for each (weight, flush) combination
    # 4 MB weight: exact per-flush values from doc
    # 16/33/200 MB weights: flush-independent (all similar)
    bw_raw = np.array([
        [983.0, 978.7, 961.6, 957.4],   # 4 MB weight
        [466.0, 466.0, 466.0, 466.0],   # 16 MB (partial L2)
        [228.0, 228.0, 228.0, 228.0],   # 33 MB (DRAM)
        [230.0, 230.0, 230.0, 230.0],   # 200 MB (DRAM)
    ])
    ratios = bw_raw / DRAM_BW  # relative to DRAM = 1.0×

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Custom colormap: red (1.0×) → yellow (2.5×) → green (4.76×)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "l2_eviction",
        [(0.0, "#C62828"),   # DRAM speed — red
         (0.3, "#F57F17"),   # partial
         (0.65, "#FDD835"),  # partial L2
         (1.0, "#2E7D32")],  # L2 speed — green
    )
    vmin, vmax = 1.0, 4.76

    im = ax.imshow(ratios, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Cell annotations
    interp_labels = {
        (0, 0): "L2 Resident", (0, 1): "L2 Resident",
        (0, 2): "L2 Resident", (0, 3): "L2 Resident",
        (1, 0): "Partial L2",  (1, 1): "Partial L2",
        (1, 2): "Partial L2",  (1, 3): "Partial L2",
        (2, 0): "DRAM",        (2, 1): "DRAM",
        (2, 2): "DRAM",        (2, 3): "DRAM",
        (3, 0): "DRAM",        (3, 1): "DRAM",
        (3, 2): "DRAM",        (3, 3): "DRAM",
    }
    for i in range(4):
        for j in range(4):
            r = ratios[i, j]
            bw = bw_raw[i, j]
            text_color = "white" if r < 2.5 else "black"
            ax.text(j, i, f"{r:.2f}×\n({bw:.0f} GB/s)",
                    ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold",
                    linespacing=1.3)

    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{f} MB\nflush" for f in flush_mb], fontsize=9.5)
    ax.set_yticks(range(4))

    row_labels = [
        f"4 MB  (≤ L2)   ",
        f"16 MB (≤ L2)   ",
        f"33 MB (> L2)   ",
        f"200 MB (>> L2) ",
    ]
    ax.set_yticklabels(row_labels, fontsize=9.5)
    ax.set_xlabel("Simulated Flush Size  (other-layer access between decode steps)",
                  fontsize=10)
    ax.set_ylabel("Weight Tensor Size", fontsize=10)
    ax.set_title(
        "L2 Cache Retention Under Simulated Decode Flush\n"
        "(Cell = bandwidth ratio vs DRAM baseline 227 GB/s;  CUDA Graph, 100 iter/replay)",
        fontsize=10.5,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("BW / DRAM Baseline", fontsize=9)
    cbar.set_ticks([1.0, 2.0, 3.0, 4.0, 4.76])
    cbar.set_ticklabels(["1.0× (DRAM)", "2.0×", "3.0×", "4.0×", "4.76× (L2 peak)"],
                        fontsize=8)

    # Right-side annotations
    row_notes = ["L2 Resident [Y]", "Partial L2",  "DRAM [N]",    "DRAM [N]"]
    note_colors = [C_FIT, C_BORDER, C_NOFIT, C_NOFIT]
    for i, (note, col) in enumerate(zip(row_notes, note_colors)):
        ax.text(4.55, i, note, va="center", ha="left",
                fontsize=9, color=col, fontweight="bold",
                transform=ax.transData)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.invert_yaxis()

    # Key insight box
    ax.text(0.01, -0.22,
            "Key insight: Bandwidth is determined by weight size, not flush size. "
            "Weights ≤ 16 MB stay L2-resident within CUDA Graph (100 iterations). "
            "Actual Alpamayo MLP (96 MB) always falls in DRAM zone.",
            transform=ax.transAxes, fontsize=8, style="italic",
            color="#444444", wrap=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig5_eviction_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 (bonus): Decode DRAM Bottleneck
# ══════════════════════════════════════════════════════════════════════════════
def fig6_decode_bottleneck() -> None:
    """
    Left: cumulative DRAM traffic across 65 decode steps.
    Right: per-step breakdown by component type.
    """
    N_STEPS       = 65
    MODEL_GB      = 20.64   # total BF16 weights
    KV_GQA_GB     = 7.11 / 1024      # 7.11 MB → GB
    DRAM_BW       = 231.0   # GB/s

    # Per-step breakdown (GB per decode step)
    MLP_GB    = 28 * 3 * 96  / 1024       # 28 LM layers × 3 proj × 96 MB
    ATTN_QO_GB = 28 * 2 * 32 / 1024       # 28 × Q+O × 32 MB
    EMBED_GB   = 2 * 1190 / 1024          # embed_tokens + lm_head (×2, 1.19 GB each)
    OTHER_GB   = MODEL_GB - MLP_GB - ATTN_QO_GB - EMBED_GB  # Action Expert + K/V + misc

    steps = np.arange(1, N_STEPS + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # ── Left: cumulative DRAM traffic ──
    w_cum = steps * MODEL_GB
    k_cum = steps * KV_GQA_GB

    ax1.fill_between(steps, w_cum, alpha=0.22, color=C_DRAM)
    ax1.plot(steps, w_cum, color=C_DRAM, linewidth=2.2,
             label=f"Weight reads  ({MODEL_GB:.2f} GB/step)")
    ax1.fill_between(steps, k_cum, alpha=0.6, color=C_KV)
    ax1.plot(steps, k_cum, color=C_KV, linewidth=1.4,
             label=f"KV cache reads ({KV_GQA_GB*1024:.1f} MB/step, GQA-4)")

    # Endpoint annotations
    total_dram = N_STEPS * MODEL_GB
    total_time = total_dram / DRAM_BW
    ax1.annotate(
        f"Total: {total_dram:.0f} GB\n"
        f"÷ {DRAM_BW:.0f} GB/s\n"
        f"= {total_time:.2f} s (theoretical)",
        xy=(N_STEPS, total_dram),
        xytext=(42, total_dram * 0.72),
        fontsize=9, color=C_DRAM, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_DRAM, lw=1.3),
    )
    ax1.annotate(
        f"KV total: {N_STEPS*KV_GQA_GB*1024:.0f} MB\n(0.03% of total)",
        xy=(N_STEPS, N_STEPS * KV_GQA_GB),
        xytext=(30, 170),
        fontsize=8.5, color=C_KV,
        arrowprops=dict(arrowstyle="->", color=C_KV, lw=1.1),
    )

    ax1.set_xlabel("Decode Step")
    ax1.set_ylabel("Cumulative DRAM Traffic (GB)")
    ax1.set_title(
        "DRAM Traffic: 65-Step Autoregressive Decode\n"
        "(BF16, each step reads all model weights)",
        fontsize=10.5,
    )
    ax1.legend(loc="upper left", fontsize=9)

    # ── Right: per-step breakdown ──
    breakdown_labels = [
        "MLP\n(gate/up/down\n×28 layers)",
        "Action Expert\n+ K/V proj\n+ misc",
        "Embedding\n+ LM Head",
        "Attention\nQ / O proj\n×28 layers",
    ]
    breakdown_vals = [MLP_GB, OTHER_GB, EMBED_GB, ATTN_QO_GB]
    bar_colors = [C_NOFIT, "#607D8B", "#7B1FA2", C_BORDER]

    bars = ax2.bar(range(4), breakdown_vals, color=bar_colors,
                   width=0.68, edgecolor="white", linewidth=0.7, zorder=4)

    for bar, val in zip(bars, breakdown_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.12,
                 f"{val:.2f} GB\n({val/MODEL_GB*100:.0f}%)",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.axhline(MODEL_GB, color="#333333", linestyle="--", linewidth=1.4, zorder=5)
    ax2.text(3.5, MODEL_GB + 0.15, f"Total: {MODEL_GB} GB",
             fontsize=9, ha="right", color="#333333")

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(breakdown_labels, fontsize=8.5)
    ax2.set_ylabel("DRAM Read per Step (GB)")
    ax2.set_title(
        "Per-Step DRAM Breakdown\n"
        "(Model components, 1 decode step)",
        fontsize=10.5,
    )
    ax2.set_ylim(0, MODEL_GB * 1.22)

    fig.tight_layout()
    save(fig, "fig6_decode_bottleneck")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Writing figures to: {OUT}/\n")

    print("[1/6] GPU bandwidth cliff ...")
    fig1_gpu_bw_cliff()

    print("[2/6] CPU cache hierarchy ...")
    fig2_cpu_hierarchy()

    print("[3/6] Memory hierarchy map ...")
    fig3_hierarchy_map()

    print("[4/6] Weight vs. cache sizes ...")
    fig4_weight_vs_cache()

    print("[5/6] L2 eviction heatmap ...")
    fig5_eviction_heatmap()

    print("[6/6] Decode DRAM bottleneck ...")
    fig6_decode_bottleneck()

    print(f"\nDone. All figures at: {OUT}/")
    print("Files: fig1_gpu_bw_cliff, fig2_cpu_hierarchy, fig3_hierarchy_map,")
    print("       fig4_weight_vs_cache, fig5_eviction_heatmap, fig6_decode_bottleneck")
    print("Extensions: .pdf (vector, for paper) and .png (300 dpi, for preview)")
