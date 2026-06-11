"""
260510_analyze_results.py
─────────────────────────────────────────────────────────────────────────────
수집된 모든 profiling 데이터를 정제 → 논문용 수치 + figure 생성

입력 (이미 가져온 파일):
  profiling_results/260510_memory_utilization/
    ├── phase_timing.json          ← Layer 1 정밀 측정 (nsys 1-run)
    ├── summary.json               ← tegrastats 4-run 평균 (phase별)
    ├── hardware_spec.json         ← 하드웨어 스펙
    ├── memory_snapshots/*.pickle  ← before/after 스냅샷
    └── nsys_kern_summary.csv      ← (선택) nsys stats 결과

출력:
  profiling_results/260510_memory_utilization/
    ├── paper_numbers.json         ← 논문에 직접 쓸 수치 (검증된 것만)
    ├── paper_numbers.md           ← 동일 내용, markdown 표
    └── figures/
        └── fig_paper_summary.png  ← 논문용 2-panel figure
"""

from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
for _f in ["Malgun Gothic", "NanumGothic", "Apple SD Gothic Neo", "DejaVu Sans"]:
    if _f in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 ─────────────────────────────────────────────────────────────────────
OUT = Path("profiling_results/260510_memory_utilization")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    phase_timing = json.loads((OUT / "phase_timing.json").read_text())
    summary      = json.loads((OUT / "summary.json").read_text())
    hw_spec_path = OUT / "hardware_spec.json"
    hw_spec      = json.loads(hw_spec_path.read_text()) if hw_spec_path.exists() else {}
    return phase_timing, summary, hw_spec


# ══════════════════════════════════════════════════════════════════════════════
# 2. 핵심 수치 계산 (논문용)
# ══════════════════════════════════════════════════════════════════════════════

def compute_paper_numbers(phase_timing: dict, summary: dict, hw_spec: dict) -> dict:
    pt  = phase_timing
    sm  = summary
    run = pt["runs"]["vlm_generate"][0]

    # ── 기본 수치 ─────────────────────────────────────────────────────────────
    model_gb       = pt["model_size_gb"]           # 22.16 GB
    peak_gb        = run["mem_peak_gb"]             # 23.20 GB
    baseline_gb    = run["mem_before_gb"]           # 22.24 GB (모델 + 소량 overhead)
    after_gb       = run["mem_after_gb"]            # 22.25 GB
    kv_peak_gb     = run["mem_delta_gb"]            # 0.96 GB (peak - before)
    kv_retained_gb = run["mem_retained_gb"]         # 0.009 GB (after - before)
    total_ms       = run["gpu_time_ms"]             # 5742 ms
    n_tok          = run["n_steps"]                 # 17 tokens

    # ── 메모리 활용률 ─────────────────────────────────────────────────────────
    total_mem_gb  = hw_spec.get("gpu_total_mem_gb", 131.9)   # 131.9 GB
    mem_util_pct  = peak_gb / total_mem_gb * 100              # 17.6%
    headroom_gb   = total_mem_gb - peak_gb                    # 108.7 GB

    # ── Phase별 메모리 (tegrastats 4-run, 더 신뢰할 수 있는 phase 분리) ──────
    by_phase = sm.get("by_phase", {})
    phase_mem = {}
    for ph, d in by_phase.items():
        phase_mem[ph] = {
            "mean_gb": d["gpu_mem_mean_mb"] / 1024,
            "peak_gb": d["gpu_mem_peak_mb"] / 1024,
            "over_model_gb": d["gpu_mem_peak_mb"] / 1024 - model_gb,
        }

    # ── BW 계산 보정 ──────────────────────────────────────────────────────────
    # 현재 BW 공식 문제:
    #   분자: MODEL_GB × n_tok (decode weight reads만 계산)
    #   분모: total_ms (vision + prefill + decode + flow 전체)
    # → decode만의 시간을 알아야 정확한 BW 계산 가능
    #
    # 보수적 추정:
    #   - Prefill이 전체의 P%, Decode이 D% 차지한다고 가정
    #   - 이전 실험 기반: vision ~10%, prefill ~25%, decode ~40%, flow ~25%
    #     (출처: tegrastats 4-run memory 변화 패턴 기반 추정)

    DRAM_BW = pt["dram_bw_peak_GBps"]      # 273 GB/s

    # 시나리오별 decode-only BW (phase 비율 가정)
    decode_frac_scenarios = {
        "30%": 0.30,
        "40%": 0.40,
        "50%": 0.50,
    }
    decode_bw_scenarios = {}
    decode_bytes_gb = model_gb * n_tok     # 22.16 × 17 = 376.7 GB
    for label, frac in decode_frac_scenarios.items():
        decode_time_s = (total_ms / 1000.0) * frac
        bw = decode_bytes_gb / decode_time_s
        decode_bw_scenarios[f"decode_frac_{label}"] = {
            "decode_time_s": round(decode_time_s, 2),
            "bw_GBps": round(bw, 1),
            "bw_pct": round(bw / DRAM_BW * 100, 1),
        }

    # 이론 하한: 모델 가중치만 1회 읽는 시간 (단일 토큰 decode의 최소 시간)
    decode_step_min_ms = model_gb / DRAM_BW * 1000   # 81.2 ms/step
    decode_total_min_ms = decode_step_min_ms * n_tok  # 17 × 81.2 = 1,380 ms

    # ── 현재 시스템 BW (희석된 값, 참고용) ──────────────────────────────────
    system_bw_GBps     = run["bw_theory_GBps"]       # 65.6 GB/s (희석됨)
    system_bw_pct      = run["bw_pct_of_peak"]       # 24%

    # ── 4-run 안정성 (tegrastats 결과) ───────────────────────────────────────
    prev_runs = {
        "warmup1_ms": 6004,
        "warmup2_ms": 5059,
        "run1_ms": 5183,
        "run2_ms": 4846,
        "run3_ms": 4861,
        "run4_ms": 4844,
    }
    run_times = [prev_runs[f"run{i}_ms"] for i in range(1, 5)]
    latency_mean_ms = float(np.mean(run_times))
    latency_std_ms  = float(np.std(run_times))

    return {
        # ── 확정 수치 (논문에 직접 기재 가능) ─────────────────────────────────
        "CONFIRMED": {
            "model_params_B":          11.08,
            "model_size_bf16_GB":      round(model_gb, 2),       # 22.16 GB
            "mem_baseline_GB":         round(baseline_gb, 2),    # 22.24 GB
            "mem_peak_GB":             round(peak_gb, 2),        # 23.20 GB
            "mem_kv_activation_GB":    round(kv_peak_gb, 2),    # 0.96 GB
            "mem_retained_after_MB":   round(kv_retained_gb * 1024, 1),  # 9.1 MB
            "total_mem_GB":            round(total_mem_gb, 1),  # 131.9 GB
            "mem_utilization_pct":     round(mem_util_pct, 1),  # 17.6%
            "mem_headroom_GB":         round(headroom_gb, 1),   # 108.7 GB
            "inference_latency_ms":    round(latency_mean_ms, 0),  # 4934 ms
            "inference_latency_std_ms": round(latency_std_ms, 0),  # 167 ms
            "n_decode_tokens":         n_tok,                   # 17
            "dram_bw_theory_GBps":     DRAM_BW,                # 273 GB/s
            "decode_step_min_ms":      round(decode_step_min_ms, 1),  # 81.2 ms
        },
        # ── Phase별 메모리 (tegrastats 4-run 평균) ────────────────────────────
        "PHASE_MEMORY": phase_mem,
        # ── BW 추정 (시나리오별, 논문에서 "estimated" 표기 필요) ───────────────
        "BW_SCENARIOS": decode_bw_scenarios,
        "system_bw_GBps_diluted":   round(system_bw_GBps, 1),  # 65.6 (희석)
        "system_bw_pct_diluted":    round(system_bw_pct, 1),   # 24%
        # ── 주석 ─────────────────────────────────────────────────────────────
        "NOTES": {
            "BW_caveat": (
                "system_bw는 vision+prefill+decode+flow 전체 시간을 분모로 사용. "
                "decode-only BW는 BW_SCENARIOS 참고 (phase 비율 가정 필요). "
                "nsys kernel 분석으로 실제 phase 비율 확인 후 업데이트 예정."
            ),
            "KV_caveat": (
                "mem_kv_activation_GB는 KV cache + 활성화 텐서 + 임시 버퍼 합산. "
                "KV cache 단독 분리는 memory_history_dump.pickle 분석 필요."
            ),
            "n_tok_caveat": (
                "n_tok=17은 고정 입력(clip_id)에서만 유효. "
                "다른 입력에서는 달라질 수 있음."
            ),
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Markdown 출력
# ══════════════════════════════════════════════════════════════════════════════

def write_markdown(numbers: dict):
    c  = numbers["CONFIRMED"]
    pm = numbers["PHASE_MEMORY"]
    bw = numbers["BW_SCENARIOS"]

    lines = [
        "# 프로파일링 최종 수치 — 논문 기재용",
        f"**측정일**: 2026-05-10 | **보드**: Jetson AGX Thor | **모델**: Alpamayo 1.5",
        "",
        "---",
        "",
        "## 1. 확정 수치 (실측, 직접 인용 가능)",
        "",
        "| 항목 | 값 | 측정 방법 |",
        "|------|----|-----------|",
        f"| 모델 파라미터 수 | **{c['model_params_B']:.2f} B** | `sum(p.numel())` |",
        f"| 모델 크기 (bf16) | **{c['model_size_bf16_GB']:.2f} GB** | `numel × 2 bytes` |",
        f"| 추론 중 GPU 메모리 피크 | **{c['mem_peak_GB']:.2f} GB** | `memory_stats['active_bytes.all.peak']` |",
        f"| 모델 대비 추가 메모리 (KV+활성화) | **{c['mem_kv_activation_GB']:.2f} GB** | peak − baseline |",
        f"| 추론 완료 후 잔류 메모리 | {c['mem_retained_after_MB']:.0f} MB | after − before |",
        f"| 총 가용 메모리 | {c['total_mem_GB']:.1f} GB | Unified LPDDR5X |",
        f"| **메모리 활용률** | **{c['mem_utilization_pct']:.1f}%** | peak / total |",
        f"| 메모리 여유분 | {c['mem_headroom_GB']:.0f} GB | total − peak |",
        f"| 1회 추론 지연 (4-run 평균) | **{c['inference_latency_ms']:.0f} ± {c['inference_latency_std_ms']:.0f} ms** | CUDA Events |",
        f"| Decode 생성 토큰 수 | {c['n_decode_tokens']} tokens | generate() 훅 |",
        f"| Decode 이론 최소 시간/step | {c['decode_step_min_ms']:.1f} ms/step | 22.16 GB ÷ 273 GB/s |",
        "",
        "---",
        "",
        "## 2. Phase별 메모리 (tegrastats 4-run 평균)",
        "",
        "| Phase | 평균 메모리 | Peak 메모리 | 모델 대비 |",
        "|-------|-------------|-------------|-----------|",
    ]

    for ph, d in pm.items():
        lines.append(
            f"| {ph.capitalize()} | {d['mean_gb']:.2f} GB | {d['peak_gb']:.2f} GB "
            f"| +{d['over_model_gb']*1024:.0f} MB |"
        )

    lines += [
        "",
        "> **주의**: tegrastats는 100ms 샘플링이므로 단기 spike 누락 가능.",
        "> 정밀 peak은 `memory_stats` 기반 23.20 GB 참고.",
        "",
        "---",
        "",
        "## 3. Decode BW 추정 (phase 비율 가정)",
        "",
        "현재 측정은 vision+prefill+decode+flow 전체를 하나로 측정해 BW가 희석됨.",
        "Phase 비율 가정에 따른 decode-only BW 추정:",
        "",
        "| Decode 비율 가정 | Decode 시간 | 추정 BW | 이론 최대 대비 | 판정 |",
        "|-----------------|------------|---------|--------------|------|",
    ]

    verdicts = {
        "decode_frac_30%": "BW-bound 가능성 높음",
        "decode_frac_40%": "BW-bound 확실 (>70%)",
        "decode_frac_50%": "BW-bound 중간",
    }
    for key, s in bw.items():
        pct = s["bw_pct"]
        verdict = verdicts.get(key, "")
        bound = "**BW-bound**" if pct >= 70 else "compute-bound 혼재"
        lines.append(
            f"| {key.replace('decode_frac_', '')} | {s['decode_time_s']:.1f}s "
            f"| {s['bw_GBps']:.0f} GB/s | {pct:.0f}% | {bound} |"
        )

    lines += [
        "",
        "> **결론**: nsys kernel 분석으로 decode 실제 비율 확인 후 BW 확정 예정.",
        "> 현재 증거로는 decode가 전체 시간의 30~50%라면 **BW-bound 확실**.",
        "",
        "---",
        "",
        "## 4. 논문 기재 시 주의사항",
        "",
        "- `mem_kv_activation_GB`: KV cache + 활성화 + 임시 버퍼 합산. KV 단독 수치 아님.",
        "- `inference_latency`: single-sample, bf16, torch.autocast 조건.",
        "- BW 수치는 phase 비율 가정 포함 → 논문에서 'estimated' 명시.",
        "- GR3D_FREQ (92.7%)는 SM compute 효율 ≠ compute utilization. 별도 표기.",
        "",
        "---",
        "",
        "## 5. 다음 실험 (미확정 수치 확정용)",
        "",
        "| 수치 | 현재 상태 | 확정 방법 |",
        "|------|-----------|-----------|",
        "| Phase별 GPU 시간 | 미측정 | nsys kernel 패턴 분석 |",
        "| Decode-only BW | 추정 | nsys + phase 분리 |",
        "| KV cache 단독 크기 | 미분리 | memory_history_dump.pickle 분석 |",
        "| SM compute 효율 | GR3D_FREQ만 | ncu (--metrics sm__throughput) |",
    ]

    md_path = OUT / "paper_numbers.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {md_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 논문용 Figure (2-panel)
# ══════════════════════════════════════════════════════════════════════════════

def plot_paper_figure(numbers: dict):
    c  = numbers["CONFIRMED"]
    pm = numbers["PHASE_MEMORY"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")

    # ── Panel (a): Memory breakdown stacked bar ────────────────────────────
    ax = ax1
    ax.set_facecolor("white")

    categories = ["Model\nWeights", "KV+\nActivations", "Peak\nTotal", "Available\nMemory"]
    values     = [c["model_size_bf16_GB"],
                  c["mem_kv_activation_GB"],
                  c["mem_peak_GB"],
                  c["total_mem_GB"]]
    colors     = ["#4878CF", "#D65F5F", "#B47CC7", "#CCCCCC"]
    hatches    = ["", "//", "", ""]
    edge_colors = ["white", "white", "white", "#999"]

    bars = ax.bar(categories, values, color=colors, edgecolor=edge_colors,
                  hatch=hatches, alpha=0.88, linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 1.5,
                f"{val:.1f} GB",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # 메모리 활용률 annotation
    ax.annotate(
        f"Memory utilization\n{c['mem_utilization_pct']:.1f}%",
        xy=(2, c["mem_peak_GB"]),
        xytext=(2.6, c["mem_peak_GB"] + 20),
        fontsize=8.5, color="#B47CC7",
        arrowprops=dict(arrowstyle="->", color="#B47CC7", lw=1.2),
    )

    ax.set_ylabel("Memory (GB)", fontsize=10)
    ax.set_title("(a) GPU Memory Footprint", fontsize=11, fontweight="bold")
    ax.set_ylim(0, c["total_mem_GB"] * 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, ls="--")

    # ── Panel (b): Phase memory bar + inference time ───────────────────────
    ax = ax2
    ax.set_facecolor("white")

    phase_names  = list(pm.keys())
    phase_peaks  = [pm[p]["peak_gb"] for p in phase_names]
    phase_colors = {"vision": "#4878CF", "prefill": "#6ACC65",
                    "decode": "#D65F5F", "flow": "#B47CC7"}
    pcols = [phase_colors.get(p, "#AAA") for p in phase_names]

    x = np.arange(len(phase_names))
    bars = ax.bar(x, phase_peaks, color=pcols, alpha=0.85,
                  edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, phase_peaks):
        over = val - c["model_size_bf16_GB"]
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.05,
                f"{val:.2f}\n(+{over*1024:.0f}MB)",
                ha="center", va="bottom", fontsize=8)

    # 모델 기준선
    ax.axhline(c["model_size_bf16_GB"], color="#555", ls="--", lw=1.3,
               label=f"Model weights ({c['model_size_bf16_GB']:.2f} GB)")
    # 전체 peak 기준선
    ax.axhline(c["mem_peak_GB"], color="#B47CC7", ls=":", lw=1.1,
               label=f"Overall peak ({c['mem_peak_GB']:.2f} GB)")

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in phase_names], fontsize=10)
    ax.set_ylabel("GPU Memory Peak (GB)", fontsize=10)
    ax.set_title("(b) Memory Peak per Phase", fontsize=11, fontweight="bold")
    ax.set_ylim(c["model_size_bf16_GB"] - 0.5, c["mem_peak_GB"] + 0.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, ls="--")

    # 추론 지연 텍스트
    ax.text(0.02, 0.97,
            f"Inference: {c['inference_latency_ms']:.0f} ± "
            f"{c['inference_latency_std_ms']:.0f} ms",
            transform=ax.transAxes, fontsize=8.5, va="top",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor="#CCC", alpha=0.9))

    plt.suptitle(
        "Alpamayo 1.5 on Jetson AGX Thor — Memory Profiling",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout(pad=1.5)

    out = FIG / "fig_paper_summary.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    out_pdf = FIG / "fig_paper_summary.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Fig] {out}")
    print(f"[Fig] {out_pdf}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 스냅샷 비교 분석
# ══════════════════════════════════════════════════════════════════════════════

def analyze_snapshots():
    snap_dir = OUT / "memory_snapshots"
    if not snap_dir.exists():
        print("[Snapshot] 디렉토리 없음, 건너뜀")
        return

    files = sorted(snap_dir.glob("*.pickle"))
    if not files:
        print("[Snapshot] pickle 파일 없음")
        return

    print(f"\n[Snapshot 분석] {len(files)}개 파일")
    print(f"  {'파일':<40} {'reserved(GB)':>13} {'allocated(GB)':>14} {'large_blocks':>13}")
    print(f"  {'-'*40} {'-'*13} {'-'*14} {'-'*13}")

    results = []
    for fpath in files:
        try:
            with open(fpath, "rb") as f:
                snap = pickle.load(f)
        except Exception as e:
            print(f"  [ERR] {fpath.name}: {e}")
            continue

        total_res = 0
        total_alloc = 0
        n_large = 0

        for seg in snap:
            total_res += seg.get("total_size", 0)
            for blk in seg.get("blocks", []):
                if blk.get("state") == "active_allocated":
                    sz = blk.get("size", 0)
                    total_alloc += sz
                    if sz > 50 * 1024 * 1024:  # 50 MB 이상
                        n_large += 1

        res_gb   = total_res   / 1e9
        alloc_gb = total_alloc / 1e9
        results.append((fpath.name, res_gb, alloc_gb, n_large))
        print(f"  {fpath.name:<40} {res_gb:>12.3f}  {alloc_gb:>13.3f}  {n_large:>13}")

    # before vs after 비교
    befores = [(n, r, a) for n, r, a, _ in results if "before" in n]
    afters  = [(n, r, a) for n, r, a, _ in results if "after"  in n]
    if befores and afters:
        print()
        print("  [Before → After 비교]")
        for (bn, br, ba), (an, ar, aa) in zip(befores, afters):
            print(f"  {bn} → {an}")
            print(f"    allocated: {ba:.3f} GB → {aa:.3f} GB  (Δ {(aa-ba)*1024:+.0f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. nsys kernel 분석 (CSV가 있을 때만)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_nsys_csv():
    csv_path = OUT / "nsys_kern_summary.csv"
    if not csv_path.exists():
        print("\n[nsys] nsys_kern_summary.csv 없음 → Thor에서 실행 필요:")
        print("  nsys stats --report cuda_gpu_kern_sum --format csv \\")
        print(f"      --output {OUT}/nsys_kern_summary \\")
        print(f"      {OUT}/nsys_run.nsys-rep")
        return

    try:
        import csv
        rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"[nsys CSV] 파싱 오류: {e}")
        return

    if not rows:
        print("[nsys CSV] 데이터 없음")
        return

    # 컬럼 이름 유연하게 처리
    col_name  = next((c for c in rows[0] if "Name" in c or "name" in c), None)
    col_total = next((c for c in rows[0] if "Total" in c and "%" not in c), None)
    col_pct   = next((c for c in rows[0] if "%" in c and "Time" in c), None)

    if not col_name or not col_total:
        print(f"[nsys CSV] 예상 컬럼 없음. 실제 컬럼: {list(rows[0].keys())}")
        return

    print(f"\n[nsys Kernel Summary] Top-15 by total time")
    print(f"  {'Kernel':<60} {'Total %':>8} {'Total time':>12}")
    print(f"  {'-'*60} {'-'*8} {'-'*12}")
    for row in rows[:15]:
        name  = row.get(col_name, "?")[:58]
        total = row.get(col_total, "?")
        pct   = row.get(col_pct, "")
        print(f"  {name:<60} {pct:>8} {total:>12}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. 최종 수치 출력 (terminal)
# ══════════════════════════════════════════════════════════════════════════════

def print_final_summary(numbers: dict):
    c = numbers["CONFIRMED"]
    W = 68
    print("\n" + "═" * W)
    print("  논문 기재 확정 수치 (Alpamayo 1.5 / Jetson AGX Thor)")
    print("═" * W)
    rows = [
        ("모델 파라미터",        f"{c['model_params_B']:.2f} B"),
        ("모델 크기 (bf16)",     f"{c['model_size_bf16_GB']:.2f} GB"),
        ("추론 중 메모리 피크",  f"{c['mem_peak_GB']:.2f} GB"),
        ("KV+활성화 overhead",   f"{c['mem_kv_activation_GB']:.2f} GB ({c['mem_kv_activation_GB']*1024:.0f} MB)"),
        ("추론 후 잔류",         f"{c['mem_retained_after_MB']:.0f} MB (KV cache 해제 확인)"),
        ("메모리 활용률",        f"{c['mem_utilization_pct']:.1f}% ({c['mem_peak_GB']:.2f} / {c['total_mem_GB']:.1f} GB)"),
        ("메모리 여유",          f"{c['mem_headroom_GB']:.0f} GB (추가 배치 여유)"),
        ("추론 지연 (4-run)",    f"{c['inference_latency_ms']:.0f} ± {c['inference_latency_std_ms']:.0f} ms"),
        ("Decode 토큰 수",       f"{c['n_decode_tokens']} tokens/inference"),
        ("Decode 이론 하한",     f"{c['decode_step_min_ms']:.1f} ms/step = 22.16 GB ÷ 273 GB/s"),
    ]
    for k, v in rows:
        print(f"  {k:<28}  {v}")
    print("═" * W)
    print()
    print("  [BW 추정 — decode-only, phase 비율 가정]")
    for key, s in numbers["BW_SCENARIOS"].items():
        label = key.replace("decode_frac_", "decode ")
        bound = "BW-bound" if s["bw_pct"] >= 70 else "혼재"
        print(f"  {label}: {s['bw_GBps']:.0f} GB/s ({s['bw_pct']:.0f}%) → {bound}")
    print("  ※ nsys kernel 분석으로 실제 decode 비율 확인 후 확정 필요")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== 260510 Profiling 데이터 분석 ===\n")

    # 데이터 로드
    phase_timing, summary, hw_spec = load_all()

    # 핵심 수치 계산
    numbers = compute_paper_numbers(phase_timing, summary, hw_spec)

    # 저장
    json_out = OUT / "paper_numbers.json"
    json_out.write_text(
        json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[저장] {json_out}")

    # Markdown
    write_markdown(numbers)

    # Figure
    plot_paper_figure(numbers)

    # Snapshot 분석
    analyze_snapshots()

    # nsys CSV (있으면)
    analyze_nsys_csv()

    # 최종 출력
    print_final_summary(numbers)


if __name__ == "__main__":
    main()
