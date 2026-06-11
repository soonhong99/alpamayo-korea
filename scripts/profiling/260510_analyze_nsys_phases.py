"""
260510_analyze_nsys_phases.py  ·  v2.0
─────────────────────────────────────────────────────────────────────────────
nsys_phase.sqlite에서 Phase별 CUDA kernel 분포를 추출.

[확인된 스키마]
  CUPTI_ACTIVITY_KIND_KERNEL (137,392행)
    start, end, shortName, demangledName, gridX/Y/Z, blockX/Y/Z
  NVTX_EVENTS (186행)
    start, end, text, textId  (text 직접 또는 StringIds FK)
  StringIds (3,586행)
    id, value

[실행]
  python scripts/profiling/260510_analyze_nsys_phases.py
  (Thor 또는 Windows 양쪽 가능)
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
OUT = Path("profiling_results/260510_memory_utilization")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DB  = OUT / "nsys_phase.sqlite"

DRAM_BW  = 273.0
MODEL_GB = 22.157

# NVTX text → 논문 Phase 레이블
PHASE_MAP = {
    "Phase/Prefill"     : "Prefill",
    "Phase/Decode"      : "Decode",
    "Phase/Flow"        : "Flow",
    "Phase/VLM_Generate": "VLM_total",
}


# ─────────────────────────────────────────────────────────────────────────────
# kernel 이름 분류
# ─────────────────────────────────────────────────────────────────────────────

def classify_kernel(name: str) -> str:
    n = name.lower()
    if "gemv" in n:
        return "GEMV (decode matmul)"
    if "splitk" in n or "split_k" in n or "splitksum" in n:
        return "splitK GEMM (decode)"
    if "nvjet" in n:
        if any(x in n for x in ["256x128", "128x256", "128x128", "256x256"]):
            return "large-tile GEMM (prefill)"
        return "GEMM-nvjet (other)"
    if "gemm" in n or "cutlass" in n or "wgmma" in n:
        return "GEMM (other)"
    if any(x in n for x in ["flash", "fmha", "sdpa", "attention", "scaled_dot"]):
        return "Attention / Flash-Attn"
    if any(x in n for x in ["layernorm", "layer_norm", "rmsnorm", "rms_norm"]):
        return "LayerNorm / RMSNorm"
    if any(x in n for x in ["catarray", "kv_cache", "appendkv", "paged_copy"]):
        return "KV Cache Ops"
    if any(x in n for x in ["elementwise", "vectorized_", "_add_", "_mul_",
                              "fused_bias", "gelu", "silu", "act_fn"]):
        return "Elementwise / Activation"
    if any(x in n for x in ["memcpy", "memset", "d2d", "h2d"]):
        return "Memory Copy / Set"
    if any(x in n for x in ["embedding", "lookup", "gather"]):
        return "Embedding / Gather"
    if any(x in n for x in ["softmax", "topk", "sample", "argmax"]):
        return "Sampling / Softmax"
    if any(x in n for x in ["reduce", "reduction", "sum_"]):
        return "Reduction"
    if any(x in n for x in ["rope", "rotary"]):
        return "RoPE"
    if any(x in n for x in ["conv", "depthwise"]):
        return "Conv (vision)"
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_nvtx(conn: sqlite3.Connection) -> list[dict]:
    """
    NVTX_EVENTS 로드.
    text 컬럼이 NULL인 경우 StringIds에서 textId로 조인.
    """
    rows = conn.execute("""
        SELECT
            n.start,
            n.end,
            COALESCE(n.text, s.value) AS text
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE COALESCE(n.text, s.value) IS NOT NULL
          AND n.end IS NOT NULL
          AND n.end > n.start
    """).fetchall()
    return [{"start_ns": r[0], "end_ns": r[1], "text": r[2]} for r in rows]


def load_kernels(conn: sqlite3.Connection) -> list[dict]:
    """
    CUPTI_ACTIVITY_KIND_KERNEL 로드.
    shortName / demangledName 이 정수 FK인 경우 StringIds로 조인.
    """
    # 먼저 shortName 컬럼 타입 확인 (int FK vs text 직접 저장)
    sample = conn.execute(
        "SELECT shortName FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1"
    ).fetchone()
    name_is_fk = sample is not None and isinstance(sample[0], int)

    if name_is_fk:
        # StringIds 조인 방식
        rows = conn.execute("""
            SELECT
                k.start,
                k.end,
                (k.end - k.start)                                            AS dur_ns,
                COALESCE(sn.value, sd.value, sm.value, 'unknown')            AS name,
                k.gridX, k.gridY, k.gridZ,
                k.blockX, k.blockY, k.blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds sn ON sn.id = k.shortName
            LEFT JOIN StringIds sd ON sd.id = k.demangledName
            LEFT JOIN StringIds sm ON sm.id = k.mangledName
            WHERE k.end > k.start
        """).fetchall()
    else:
        # 문자열 직접 저장 방식
        rows = conn.execute("""
            SELECT
                start,
                end,
                (end - start)                                                AS dur_ns,
                COALESCE(shortName, demangledName, mangledName, 'unknown')   AS name,
                gridX, gridY, gridZ,
                blockX, blockY, blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > start
        """).fetchall()

    print(f"    name FK 방식: {name_is_fk}")
    return [
        {
            "start_ns": r[0], "end_ns": r[1], "dur_ns": r[2],
            "name": str(r[3]) if r[3] is not None else "unknown",
            "grid":  (r[4], r[5], r[6]),
            "block": (r[7], r[8], r[9]),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 필터링: NVTX 범위 안에 있는 kernel 수집
# ─────────────────────────────────────────────────────────────────────────────

def map_kernels_to_phases(nvtx: list[dict], kernels: list[dict]) -> dict[str, list]:
    """
    Phase별 exclusive 시간 범위로 kernel을 분류.

    [Decode NVTX 76ms 문제 해결]
    Phase/Decode NVTX는 CPU dispatch 시간만 측정 (GPU는 비동기 실행 → 1738ms).
    따라서 Decode range를 NVTX 직접 사용 대신
    "Prefill 종료 시각 ~ Flow 시작 시각" 갭으로 유도.

    [Priority 원칙]
    Prefill / Decode / Flow 는 상호 배타적 → 시간 순서로 명확히 분리.
    VLM_total은 참고용으로만 사용.
    """
    # Phase별 NVTX 범위 수집 (warmup + run 각 1회씩)
    phase_occurrences: dict[str, list[tuple]] = defaultdict(list)
    for r in nvtx:
        for key, label in PHASE_MAP.items():
            if r["text"] == key:
                phase_occurrences[label].append((r["start_ns"], r["end_ns"]))

    # 마지막 발생 = 측정 run 선택
    last_ranges: dict[str, tuple] = {}
    for label, occs in phase_occurrences.items():
        occs_sorted = sorted(occs, key=lambda x: x[0])
        last_ranges[label] = occs_sorted[-1]
        dur_ms = (occs_sorted[-1][1] - occs_sorted[-1][0]) / 1e6
        print(f"  [NVTX] {label}: {len(occs)}회 → "
              f"마지막 사용 ({dur_ms:.0f} ms) "
              f"{'← CPU-async, 보정 필요' if label=='Decode' and dur_ms < 200 else ''}")

    # ── 시간 범위 구성 ──────────────────────────────────────────────────────
    # Prefill: NVTX 직접 사용 (stop() 내부 synchronize → CPU/GPU 시간 일치)
    # Decode:  Prefill end ~ Flow start (NVTX Decode는 CPU dispatch만 측정)
    # Flow:    NVTX 직접 사용 (별도 Phase, 독립적)

    phase_ranges: dict[str, tuple] = {}

    if "Prefill" in last_ranges:
        phase_ranges["Prefill"] = last_ranges["Prefill"]

    if "Flow" in last_ranges:
        phase_ranges["Flow"] = last_ranges["Flow"]

    # Decode = Prefill 종료 ~ Flow 시작
    if "Prefill" in last_ranges and "Flow" in last_ranges:
        pf_end    = last_ranges["Prefill"][1]
        flow_start = last_ranges["Flow"][0]
        if flow_start > pf_end:
            phase_ranges["Decode"] = (pf_end, flow_start)
            dc_ms = (flow_start - pf_end) / 1e6
            print(f"  [Decode 보정] Prefill 종료 ~ Flow 시작 = {dc_ms:.0f} ms "
                  f"(CUDA Events 측정: 1738 ms 참고)")

    # VLM_total 은 kernel 분류에 사용하지 않음 (Prefill+Decode 상위 범위로 중복)
    print(f"  [Phase 범위 요약]")
    for p, (rs, re) in phase_ranges.items():
        print(f"    {p}: {(re-rs)/1e6:.0f} ms")

    # ── kernel → phase 배정 (배타적 시간 범위) ──────────────────────────
    result: dict[str, list] = defaultdict(list)
    unmatched = 0
    for k in kernels:
        ks = k["start_ns"]
        matched = False
        for phase, (rs, re) in phase_ranges.items():
            if ks >= rs and ks < re:   # kernel 시작 시각 기준 (end 포함 불필요)
                result[phase].append(k)
                matched = True
                break
        if not matched:
            unmatched += 1

    print(f"  [매핑 결과] 미매핑: {unmatched:,}개 "
          f"(warmup 구간 kernel 등 정상)")
    return dict(result)


# ─────────────────────────────────────────────────────────────────────────────
# 통계 집계
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(phase_kernels: dict) -> dict:
    stats = {}
    for phase, kernels in phase_kernels.items():
        if not kernels:
            continue

        total_ns = sum(k["dur_ns"] for k in kernels)

        by_name  = defaultdict(lambda: {"count": 0, "total_ns": 0,
                                         "grids": [], "blocks": []})
        by_class = defaultdict(lambda: {"count": 0, "total_ns": 0})

        for k in kernels:
            nm = k["name"]
            cl = classify_kernel(nm)
            by_name[nm]["count"]    += 1
            by_name[nm]["total_ns"] += k["dur_ns"]
            by_name[nm]["grids"].append(k["grid"])
            by_name[nm]["blocks"].append(k["block"])
            by_class[cl]["count"]    += 1
            by_class[cl]["total_ns"] += k["dur_ns"]

        top15 = sorted(by_name.items(),
                       key=lambda x: x[1]["total_ns"], reverse=True)[:15]

        class_dist = {
            cl: {
                "count":    v["count"],
                "total_ms": v["total_ns"] / 1e6,
                "pct":      v["total_ns"] / total_ns * 100 if total_ns else 0,
            }
            for cl, v in sorted(by_class.items(),
                                 key=lambda x: x[1]["total_ns"], reverse=True)
        }

        stats[phase] = {
            "n_kernels" : len(kernels),
            "total_ms"  : total_ns / 1e6,
            "top15"     : [
                {
                    "name"     : nm,
                    "class"    : classify_kernel(nm),
                    "count"    : v["count"],
                    "total_ms" : v["total_ns"] / 1e6,
                    "pct"      : v["total_ns"] / total_ns * 100 if total_ns else 0,
                    "avg_us"   : v["total_ns"] / v["count"] / 1e3,
                }
                for nm, v in top15
            ],
            "class_dist": class_dist,
        }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(stats: dict):
    W = 92
    for phase in ["Prefill", "Decode", "Flow", "VLM_total"]:
        if phase not in stats:
            continue
        s = stats[phase]
        print(f"\n{'═'*W}")
        print(f"  ▶ {phase}  —  {s['n_kernels']:,}개 kernel  /  {s['total_ms']:.1f} ms")
        print(f"{'═'*W}")

        print(f"  {'Kernel Class':<30} {'Time(ms)':>10}  {'%':>6}  {'#kernel':>8}")
        print(f"  {'-'*30} {'-'*10}  {'-'*6}  {'-'*8}")
        for cl, v in s["class_dist"].items():
            bar = "█" * max(1, int(v["pct"] / 3))
            print(f"  {cl:<30} {v['total_ms']:>10.1f}  "
                  f"{v['pct']:>5.1f}%  {v['count']:>8}  {bar}")

        print(f"\n  Top-15 kernels:")
        print(f"  {'#':>3}  {'Kernel (truncated to 56 chars)':<57} "
              f"{'ms':>8}  {'%':>6}  {'N':>6}  {'avg μs':>8}")
        print(f"  {'-'*3}  {'-'*57} {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}")
        for i, k in enumerate(s["top15"], 1):
            nm = k["name"][:56]
            print(f"  {i:>3}  {nm:<57} "
                  f"{k['total_ms']:>8.2f}  {k['pct']:>5.1f}%  "
                  f"{k['count']:>6}  {k['avg_us']:>8.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 논문용 분석 요약 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_paper_summary(stats: dict):
    print("\n" + "★" * 80)
    print("  논문 기재용 핵심 수치")
    print("★" * 80)

    for phase in ["Prefill", "Decode", "Flow"]:
        if phase not in stats:
            continue
        s = stats[phase]
        cd = s["class_dist"]

        print(f"\n[{phase}]  {s['total_ms']:.0f} ms  /  {s['n_kernels']:,} kernels")

        # 상위 3개 class
        top3 = list(cd.items())[:3]
        for cl, v in top3:
            print(f"  {cl:<32} {v['total_ms']:>8.1f} ms  ({v['pct']:.1f}%)")

        # Phase별 특이점 분석
        if phase == "Prefill":
            gemm_pct = sum(v["pct"] for cl, v in cd.items()
                           if "GEMM" in cl or "Attention" in cl)
            print(f"  → GEMM+Attention 합계: {gemm_pct:.1f}%  "
                  f"[Compute-bound 근거]")

        elif phase == "Decode":
            gemv_pct = cd.get("GEMV (decode matmul)", {}).get("pct", 0)
            splitk_pct = cd.get("splitK GEMM (decode)", {}).get("pct", 0)
            bw_bound_pct = gemv_pct + splitk_pct
            print(f"  → GEMV + splitK 합계: {bw_bound_pct:.1f}%  "
                  f"[BW-bound 커널 직접 증거]")
            n_tok = 17
            print(f"  → 토큰당 kernel 수: {s['n_kernels']/n_tok:.0f}개/tok")

        elif phase == "Flow":
            small_kernels = sum(1 for k in s["top15"]
                                if k["avg_us"] < 100)
            print(f"  → 평균 실행시간 <100μs kernel: {small_kernels}개  "
                  f"[overhead 지배 근거]")

    print("\n" + "★" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# 마크다운 저장
# ─────────────────────────────────────────────────────────────────────────────

def write_md(stats: dict):
    lines = [
        "# nsys CUDA Kernel Phase 분석 — 논문 기재용",
        "**측정**: nsys 2025.6.1 + NVTX + CUPTI | **보드**: Jetson AGX Thor (SM 11.0)",
        "",
    ]
    for phase in ["Prefill", "Decode", "Flow"]:
        if phase not in stats:
            continue
        s = stats[phase]
        lines += [
            f"## {phase} ({s['total_ms']:.0f} ms, {s['n_kernels']:,} kernels)",
            "",
            "### Kernel Class 분포",
            "| Class | Time (ms) | % | Count |",
            "|---|---|---|---|",
        ]
        for cl, v in list(s["class_dist"].items())[:10]:
            lines.append(
                f"| {cl} | {v['total_ms']:.1f} | {v['pct']:.1f}% | {v['count']} |"
            )
        lines += [
            "",
            "### Top-10 Kernels",
            "| Rank | Kernel | ms | % | Count | avg μs |",
            "|---|---|---|---|---|---|",
        ]
        for i, k in enumerate(s["top15"][:10], 1):
            nm = k["name"][:70]
            lines.append(
                f"| {i} | `{nm}` | {k['total_ms']:.2f} | "
                f"{k['pct']:.1f}% | {k['count']} | {k['avg_us']:.1f} |"
            )
        lines.append("")

    p = OUT / "nsys_phase_analysis.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "GEMV (decode matmul)"      : "#D65F5F",
    "splitK GEMM (decode)"      : "#E8906A",
    "large-tile GEMM (prefill)" : "#6ACC65",
    "GEMM-nvjet (other)"        : "#A0CC80",
    "GEMM (other)"              : "#B0D090",
    "Attention / Flash-Attn"    : "#4878CF",
    "LayerNorm / RMSNorm"       : "#8FA0C8",
    "KV Cache Ops"              : "#B47CC7",
    "Elementwise / Activation"  : "#C4B0D0",
    "Memory Copy / Set"         : "#AAAAAA",
    "Embedding / Gather"        : "#FFD070",
    "Sampling / Softmax"        : "#F0B050",
    "Reduction"                 : "#80C0A0",
    "RoPE"                      : "#D0A060",
    "Conv (vision)"             : "#70B0D0",
    "Other"                     : "#CCCCCC",
}


def plot_all(stats: dict):
    phases = [p for p in ["Prefill", "Decode", "Flow"] if p in stats]
    if not phases:
        return

    # ── Fig1: 각 phase별 class pie chart ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Alpamayo 1.5 on Jetson AGX Thor — CUDA Kernel Distribution per Phase\n"
        "(nsys 2025.6.1, NVTX-correlated, bf16, measurement run only)",
        fontsize=11, fontweight="bold"
    )

    for ax, phase in zip(axes, phases):
        s = stats[phase]
        cd = s["class_dist"]
        # threshold: 1% 미만은 Other로 합산
        labels, pcts, cols = [], [], []
        other_pct = 0.0
        for cl, v in cd.items():
            if v["pct"] >= 1.0:
                labels.append(cl)
                pcts.append(v["pct"])
                cols.append(CLASS_COLORS.get(cl, "#CCCCCC"))
            else:
                other_pct += v["pct"]
        if other_pct > 0:
            labels.append(f"Other (<1% each)")
            pcts.append(other_pct)
            cols.append("#CCCCCC")

        wedges, _, autotexts = ax.pie(
            pcts, labels=None, colors=cols,
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            startangle=90, pctdistance=0.78,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title(
            f"{phase}\n{s['total_ms']:.0f} ms  ·  {s['n_kernels']:,} kernels",
            fontsize=10, fontweight="bold", pad=10
        )

    # 공통 범례
    all_used = set()
    for p in phases:
        all_used |= set(stats[p]["class_dist"].keys())
    patches = [
        mpatches.Patch(color=CLASS_COLORS.get(c, "#CCC"), label=c)
        for c in CLASS_COLORS if c in all_used
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.06), framealpha=0.95, edgecolor="#CCC")

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    for ext in ("png", "pdf"):
        fp = FIG / f"fig_kernel_pie.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Fig] {fp}")
    plt.close(fig)

    # ── Fig2: stacked bar (phase × class) ─────────────────────────────────
    all_classes = []
    for p in phases:
        for cl in stats[p]["class_dist"]:
            if cl not in all_classes:
                all_classes.append(cl)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x       = np.arange(len(phases))
    bottoms = np.zeros(len(phases))
    used    = []

    for cl in all_classes:
        vals = [stats[p]["class_dist"].get(cl, {}).get("total_ms", 0.0)
                for p in phases]
        if max(vals) < 1.0:
            continue
        ax.bar(x, vals, bottom=bottoms,
               color=CLASS_COLORS.get(cl, "#CCC"), label=cl,
               edgecolor="white", linewidth=0.8, alpha=0.92)
        bottoms += np.array(vals)
        used.append(cl)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n{stats[p]['total_ms']:.0f} ms\n({stats[p]['n_kernels']:,} kernels)"
         for p in phases],
        fontsize=10
    )
    ax.set_ylabel("GPU Time captured by NVTX (ms)", fontsize=11)
    ax.set_title(
        "Phase별 CUDA Kernel Class 시간 분해\n"
        "(Alpamayo 1.5 · Jetson AGX Thor · nsys NVTX-correlated)",
        fontsize=11, fontweight="bold"
    )
    handles = [mpatches.Patch(color=CLASS_COLORS.get(c, "#CCC"), label=c)
               for c in used]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#CCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, ls="--")
    ax.set_ylim(0, max(bottoms) * 1.12)

    # 각 bar 위에 합계 표시
    for i, (p, tot) in enumerate(zip(phases, bottoms)):
        ax.text(i, tot + max(bottoms) * 0.01,
                f"{tot:.0f} ms", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fp = FIG / f"fig_kernel_stack.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Fig] {fp}")
    plt.close(fig)

    # ── Fig3: Decode 내 BW-bound kernel 비중 (논문 핵심 그래프) ──────────
    if "Decode" in stats:
        s  = stats["Decode"]
        cd = s["class_dist"]
        gemv_ms   = cd.get("GEMV (decode matmul)", {}).get("total_ms", 0)
        splitk_ms = cd.get("splitK GEMM (decode)", {}).get("total_ms", 0)
        other_ms  = s["total_ms"] - gemv_ms - splitk_ms

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            "Decode Phase: BW-Bound Kernel Analysis\n"
            f"(Jetson AGX Thor · DRAM BW={DRAM_BW} GB/s · Achieved 216.8 GB/s = 79.4%)",
            fontsize=11, fontweight="bold"
        )

        # (a) pie: BW-bound vs rest
        ax = axes[0]
        ax.set_facecolor("white")
        slices = [gemv_ms, splitk_ms, other_ms]
        lbls   = [f"GEMV\n{gemv_ms:.0f}ms",
                  f"splitK GEMM\n{splitk_ms:.0f}ms",
                  f"Other\n{other_ms:.0f}ms"]
        clrs   = ["#D65F5F", "#E8906A", "#CCCCCC"]
        ws, _, ats = ax.pie(slices, labels=lbls, colors=clrs,
                            autopct="%1.1f%%", startangle=90,
                            pctdistance=0.72,
                            wedgeprops={"edgecolor": "white", "lw": 1.5})
        for at in ats:
            at.set_fontsize(9)
        ax.set_title("(a) Decode kernel 분해\n(BW-bound kernel 비중)", fontsize=10,
                     fontweight="bold")

        # (b) BW bar: achieved vs peak
        ax = axes[1]
        ax.set_facecolor("white")
        bw_achieved = 216.8
        bw_gemv     = MODEL_GB / ((gemv_ms / 1000.0) / 17) if gemv_ms > 0 else 0
        cats   = ["DRAM Peak", "Decode\n(measured)", "GEMV-only\n(estimated)"]
        bws    = [DRAM_BW, bw_achieved, min(bw_gemv, DRAM_BW * 1.05)]
        colors = ["#AAAAAA", "#D65F5F", "#E8906A"]
        bars   = ax.bar(cats, bws, color=colors, alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, bws):
            ax.text(bar.get_x() + bar.get_width()/2, v + 4,
                    f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
        ax.axhline(DRAM_BW * 0.70, color="orange", ls=":", lw=2,
                   label="BW-bound threshold (70%)")
        ax.set_ylabel("Bandwidth (GB/s)", fontsize=10)
        ax.set_ylim(0, DRAM_BW * 1.2)
        ax.set_title("(b) 달성 메모리 대역폭", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2, ls="--")

        plt.tight_layout()
        for ext in ("png", "pdf"):
            fp = FIG / f"fig_decode_bw_kernel.{ext}"
            plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"[Fig] {fp}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not DB.exists():
        print(f"[오류] {DB} 없음")
        return

    print(f"[로드] {DB}  ({DB.stat().st_size/1e6:.1f} MB)")
    conn = sqlite3.connect(str(DB))

    try:
        print("\n[1] NVTX 범위 로드...")
        nvtx = load_nvtx(conn)
        print(f"    {len(nvtx)}개 NVTX 이벤트")
        # 어떤 텍스트가 있는지 출력
        texts = sorted(set(r["text"] for r in nvtx))
        print(f"    고유 텍스트 ({len(texts)}개): {texts[:20]}")

        print("\n[2] GPU kernel 로드...")
        kernels = load_kernels(conn)
        print(f"    {len(kernels):,}개 kernel  "
              f"(총 {sum(k['dur_ns'] for k in kernels)/1e6:.0f} ms)")

        print("\n[3] Phase 매핑...")
        phase_kernels = map_kernels_to_phases(nvtx, kernels)
        for p, ks in phase_kernels.items():
            print(f"    {p}: {len(ks):,}개 kernel "
                  f"({sum(k['dur_ns'] for k in ks)/1e6:.0f} ms)")

        print("\n[4] 통계 집계...")
        stats = aggregate(phase_kernels)

        print_stats(stats)
        print_paper_summary(stats)

        # 저장
        out_json = OUT / "nsys_phase_analysis.json"
        out_json.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False, default=float)
        )
        print(f"\n[저장] {out_json}")
        write_md(stats)
        print("\n[시각화 생성 중...]")
        plot_all(stats)

    finally:
        conn.close()
        print("\n완료.")


if __name__ == "__main__":
    main()
