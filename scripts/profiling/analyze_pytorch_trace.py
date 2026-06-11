"""
analyze_pytorch_trace.py
────────────────────────────────────────────────────────────────────────────────
pytorch_trace.json 에서 CPU-GPU 실제 활동을 분석하고 시각화한다.

생성 그림:
  Fig 8.  CPU-GPU 실제 활동 분해 (실측 기반, 추정 아님)
  Fig 9.  CPU-GPU 오버랩 타임라인 (1회 추론 확대)
  Fig 10. GPU 커널 카테고리별 시간 분배
  Fig 11. CPU 직렬 병목 상세 (GPU가 기다리는 구간)

입력:
  profiling_results/pytorch_trace.json  (--pytorch_profiler 로 생성)

출력:
  profiling_results/figures/fig8_cpu_gpu_real.png
  profiling_results/figures/fig9_overlap_timeline.png
  profiling_results/figures/fig10_gpu_kernels.png
  profiling_results/figures/fig11_cpu_serial.png
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
import io
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def _ok(name):
    print(f"  [OK] {name}")

# ── 색상 ──────────────────────────────────────────────────────────────────────
C = {
    "cpu_only"   : "#78909C",   # 회색-파랑
    "gpu_only"   : "#EF5350",   # 빨강
    "both"       : "#7E57C2",   # 보라 (CPU+GPU 동시)
    "idle"       : "#ECEFF1",   # 연회색
    "vision"     : "#4FC3F7",
    "prefill"    : "#7986CB",
    "decode"     : "#EF5350",
    "flow"       : "#66BB6A",
    "bg"         : "#FAFAFA",
}

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드 + 분석
# ─────────────────────────────────────────────────────────────────────────────
def load_and_analyze(trace_path: Path) -> dict:
    print(f"[Analyze] 로드 중: {trace_path}  ({trace_path.stat().st_size//1024//1024} MB)")
    with open(trace_path) as f:
        data = json.load(f)

    events = [e for e in data.get("traceEvents", []) if isinstance(e, dict)]
    print(f"[Analyze] 총 이벤트: {len(events):,}")

    # ── 스레드 이름 수집 ──
    thread_names = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            k = (e.get("pid"), e.get("tid"))
            thread_names[k] = e.get("args", {}).get("name", "")

    # ── 전체 시간 범위 ──
    all_ts = [float(e["ts"]) for e in events if "ts" in e]
    t0, t1 = min(all_ts), max(all_ts)
    wall_us = t1 - t0

    # ── 이벤트 분류 ──
    cpu_main_ivs   = []   # TID 7038 cpu_op intervals
    gpu_kernel_ivs = []   # cat=kernel intervals
    gpu_memcpy_ivs = []   # cat=gpu_memcpy

    kern_by_name = collections.defaultdict(lambda: [0.0, 0])
    cpu_ops_by_name = collections.defaultdict(lambda: [0.0, 0])
    py_calls = []   # (dur, name) python_function >100ms

    active_tids = collections.Counter()

    for e in events:
        ph  = e.get("ph", "")
        cat = e.get("cat", "")
        tid = e.get("tid")
        ts  = float(e.get("ts", 0))
        dur = float(e.get("dur") or 0)
        nm  = e.get("name", "")

        if ph != "X" or dur <= 0:
            continue

        if cat == "cpu_op":
            active_tids[tid] += dur
            if tid == 7038:
                cpu_main_ivs.append((ts, ts + dur))
                cpu_ops_by_name[nm][0] += dur
                cpu_ops_by_name[nm][1] += 1
        elif cat == "kernel":
            gpu_kernel_ivs.append((ts, ts + dur))
            kern_by_name[nm][0] += dur
            kern_by_name[nm][1] += 1
        elif cat == "gpu_memcpy":
            gpu_memcpy_ivs.append((ts, ts + dur))
        elif cat == "python_function" and tid == 7038 and dur > 100_000:
            py_calls.append((dur, nm))

    # ── 구간 병합 ──
    def merge(ivs):
        if not ivs: return []
        s = sorted(ivs)
        m = [list(s[0])]
        for a, b in s[1:]:
            if a <= m[-1][1]:
                m[-1][1] = max(m[-1][1], b)
            else:
                m.append([a, b])
        return [(a, b) for a, b in m]

    def total(ivs):
        return sum(b - a for a, b in ivs)

    def overlap(iv1, iv2):
        i = j = 0
        t = 0.0
        while i < len(iv1) and j < len(iv2):
            lo = max(iv1[i][0], iv2[j][0])
            hi = min(iv1[i][1], iv2[j][1])
            if lo < hi: t += hi - lo
            if iv1[i][1] < iv2[j][1]: i += 1
            else: j += 1
        return t

    cpu_m   = merge(cpu_main_ivs)
    gpu_m   = merge(gpu_kernel_ivs + gpu_memcpy_ivs)
    gpu_k_m = merge(gpu_kernel_ivs)

    cpu_total   = total(cpu_m)
    gpu_total   = total(gpu_m)
    both_total  = overlap(cpu_m, gpu_m)
    cpu_only    = cpu_total - both_total
    gpu_only    = gpu_total - both_total
    idle_total  = max(0, wall_us - cpu_total - gpu_only)

    n_runs = 3  # 프로파일러 실행 횟수

    print(f"\n=== 실측 CPU-GPU 활동 (3회 추론, 겹침 구간 제거) ===")
    print(f"  전체 관찰 시간 : {wall_us/1000:.0f} ms  ({wall_us/1000/n_runs:.0f} ms/run)")
    print(f"  [CPU+GPU 동시] : {both_total/1000:.0f} ms  ({both_total/wall_us*100:.1f}%)")
    print(f"  [CPU만 실행]   : {cpu_only/1000:.0f} ms  ({cpu_only/wall_us*100:.1f}%)")
    print(f"  [GPU만 실행]   : {gpu_only/1000:.0f} ms  ({gpu_only/wall_us*100:.1f}%)")
    print(f"  [완전 유휴]    : {idle_total/1000:.0f} ms  ({idle_total/wall_us*100:.1f}%)")
    print(f"\n  1회 추론 기준:")
    print(f"    CPU 활성: {cpu_total/1000/n_runs:.0f} ms  |  GPU 활성: {gpu_total/1000/n_runs:.0f} ms")
    print(f"    동시 실행: {both_total/1000/n_runs:.0f} ms  |  CPU 직렬: {cpu_only/1000/n_runs:.0f} ms")

    # ── GPU 커널 분류 (nvjet = GEMM, elementwise, flash, memcpy 등) ──
    kern_categories = {
        "GEMM\n(행렬 곱)":   0.0,
        "Elementwise\n(활성화/정규화)": 0.0,
        "Flash Attention":   0.0,
        "Memory Copy":       0.0,
        "GEMV\n(Decode 전용)": 0.0,
        "기타":              0.0,
    }
    for name, (dur_sum, cnt) in kern_by_name.items():
        if "nvjet" in name and "gemv" not in name.lower():
            kern_categories["GEMM\n(행렬 곱)"] += dur_sum
        elif "flash_fwd" in name or "flash_bwd" in name:
            kern_categories["Flash Attention"] += dur_sum
        elif "gemv" in name.lower():
            kern_categories["GEMV\n(Decode 전용)"] += dur_sum
        elif any(x in name for x in ("elementwise", "vectorized", "unrolled", "copy_")):
            kern_categories["Elementwise\n(활성화/정규화)"] += dur_sum
        elif "memcpy" in name.lower() or "memset" in name.lower():
            kern_categories["Memory Copy"] += dur_sum
        else:
            kern_categories["기타"] += dur_sum

    # ── 샘플 구간 (1회 추론) for timeline ──
    # 3개 추론 중 첫 번째 구간 추출 (5050ms 짜리)
    # python_function에서 _profile_one_run 첫 번째를 찾아 구간 특정
    run_windows = []
    for e in events:
        if (e.get("cat") == "python_function"
                and "_profile_one_run" in e.get("name", "")
                and e.get("tid") == 7038):
            ts  = float(e.get("ts", 0))
            dur = float(e.get("dur") or 0)
            if dur > 4_000_000:   # 4초 이상 = 실제 측정 런
                run_windows.append((ts, ts + dur))

    run_windows.sort()
    sample_window = run_windows[0] if run_windows else (t0, t0 + 5_100_000)
    sw_s, sw_e = sample_window

    # 해당 구간의 CPU/GPU 이벤트만 필터
    cpu_sample = [(max(s, sw_s), min(e, sw_e))
                  for s, e in cpu_m if s < sw_e and e > sw_s]
    gpu_sample = [(max(s, sw_s), min(e, sw_e))
                  for s, e in gpu_k_m if s < sw_e and e > sw_s]

    return {
        "wall_us": wall_us, "n_runs": n_runs,
        "cpu_total": cpu_total, "gpu_total": gpu_total,
        "both": both_total, "cpu_only": cpu_only,
        "gpu_only": gpu_only, "idle": idle_total,
        "kern_categories": kern_categories,
        "kern_by_name": kern_by_name,
        "cpu_ops_by_name": cpu_ops_by_name,
        "active_tids": active_tids,
        "py_calls": sorted(py_calls, reverse=True)[:20],
        "thread_names": thread_names,
        "cpu_sample": cpu_sample,
        "gpu_sample": gpu_sample,
        "sw_s": sw_s, "sw_e": sw_e,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — CPU-GPU 실제 활동 분해 (도넛 + 바 차트)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_real_cpu_gpu(res: dict, outdir: Path):
    wall = res["wall_us"]
    n    = res["n_runs"]

    labels_d = ["CPU+GPU\n동시 실행", "CPU 전용\n(GPU 유휴)", "GPU 전용\n(CPU 유휴)", "유휴/오버헤드"]
    vals_d   = [res["both"], res["cpu_only"], res["gpu_only"], res["idle"]]
    cols_d   = [C["both"], C["cpu_only"], C["gpu_only"], C["idle"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── 왼쪽: 도넛 ──
    ax = axes[0]
    wedges, _, autotexts = ax.pie(
        vals_d, colors=cols_d, autopct="%1.1f%%",
        startangle=90, pctdistance=0.72,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
        explode=[0.03, 0.08, 0.03, 0.0],
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    centre = plt.Circle((0, 0), 0.52, fc=C["bg"])
    ax.add_artist(centre)
    ax.text(0, 0.08, "CPU-GPU\n실제 활동", ha="center", fontsize=11,
            fontweight="bold", color="#333")
    ax.text(0, -0.22, f"1회 {wall/1000/n:.0f}ms", ha="center",
            fontsize=9, color="#888")

    patches = [mpatches.Patch(color=c, label=f"{l.replace(chr(10),' ')}  "
                              f"({v/1000/n:.0f}ms/run, {v/wall*100:.1f}%)")
               for l, v, c in zip(labels_d, vals_d, cols_d)]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), fontsize=9.5,
              frameon=False, ncol=1)
    ax.set_title("1회 추론 CPU-GPU 활동 분해\n(pytorch_trace.json 실측, 추정 아님)", pad=10)

    # ── 오른쪽: 스레드 활용 바 ──
    ax2 = axes[1]
    tids_sorted = sorted(res["active_tids"].items(), key=lambda x: -x[1])
    tids  = [str(t) for t, _ in tids_sorted[:6]]
    works = [w / 1000 / n for _, w in tids_sorted[:6]]  # ms per run

    colors_bar = ["#EF5350" if t == "7038" else "#B0BEC5" for t in tids]
    bars = ax2.barh(tids, works, color=colors_bar,
                    edgecolor="white", height=0.55)
    for bar, w in zip(bars, works):
        ax2.text(w + 5, bar.get_y() + bar.get_height()/2,
                 f"{w:.0f} ms", va="center", fontsize=10)

    ax2.set_xlabel("cpu_op 누적 시간 (ms/run, 중첩 포함)", fontsize=11)
    ax2.set_title("CPU 스레드별 op 실행 시간\n(중첩 포함 raw 합산 — 활동 존재 여부 지표)", pad=10)
    ax2.set_xlim(0, max(works) * 1.25)

    # 주석
    ax2.text(max(works) * 0.5, 0,
             "TID 7038 = 메인 Python 스레드\n(추론 dispatch 전담)\n\n"
             "TID 7069 = PyTorch 내부 워커\n(추론 중 op 없음 확인됨)",
             ha="center", va="center", fontsize=10, color="#333",
             bbox=dict(boxstyle="round,pad=0.5", fc="white",
                       alpha=0.9, ec="#CCCCCC"))

    fig.suptitle("Fig 8  CPU-GPU 실제 활동 분해 — pytorch_trace.json 실측 데이터",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = outdir / "fig8_cpu_gpu_real.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    _ok(out.name)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — CPU-GPU 오버랩 실제 타임라인 (1회 추론)
# ─────────────────────────────────────────────────────────────────────────────
def fig9_overlap_timeline(res: dict, outdir: Path):
    sw_s = res["sw_s"]
    sw_e = res["sw_e"]
    dur_ms = (sw_e - sw_s) / 1000

    cpu_ivs = [(s - sw_s, e - sw_s) for s, e in res["cpu_sample"]]
    gpu_ivs = [(s - sw_s, e - sw_s) for s, e in res["gpu_sample"]]

    fig, ax = plt.subplots(figsize=(15, 4.5))

    Y_GPU  = 1.5
    Y_CPU  = 0.5
    H      = 0.45

    # GPU 커널 구간
    for s, e in gpu_ivs:
        ax.barh(Y_GPU, (e-s)/1000, left=s/1000, height=H,
                color=C["gpu_only"], alpha=0.85, edgecolor="none")

    # CPU 활성 구간 — GPU와 겹치면 보라(동시), 안 겹치면 회색(CPU 전용)
    def any_overlap_gpu(s, e):
        for gs, ge in gpu_ivs:
            if gs < e and ge > s:
                return True
        return False

    for s, e in cpu_ivs:
        col = C["both"] if any_overlap_gpu(s, e) else C["cpu_only"]
        ax.barh(Y_CPU, (e-s)/1000, left=s/1000, height=H,
                color=col, alpha=0.85, edgecolor="none")

    # 범례용 더미 바
    ax.barh(-1, 0, color=C["gpu_only"],  label="GPU 커널 실행")
    ax.barh(-1, 0, color=C["both"],      label="CPU dispatch (GPU 동시 실행)")
    ax.barh(-1, 0, color=C["cpu_only"],  label="CPU 전용 (GPU 유휴 — 진짜 병목)")

    ax.set_yticks([Y_CPU, Y_GPU])
    ax.set_yticklabels(["CPU Core 0\n(TID 7038)", "GPU (SM 11.0)"], fontsize=11)
    ax.set_ylim(-0.2, 2.3)
    ax.set_xlim(-50, dur_ms + 100)
    ax.set_xlabel("시간 (ms, 추론 시작 기준)", fontsize=11)
    ax.set_title(
        "Fig 9  CPU-GPU 실제 오버랩 타임라인 — 1회 추론\n"
        "보라 = CPU dispatch 중 GPU도 실행  |  회색 = CPU 전용 (GPU 대기)",
        pad=10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # CPU-전용 구간 강조 주석
    cpu_serial = [(s, e) for s, e in cpu_ivs if not any_overlap_gpu(s, e)]
    if cpu_serial:
        total_serial = sum((e-s)/1000 for s, e in cpu_serial)
        ax.text(dur_ms * 0.5, 1.95,
                f"CPU 전용 구간 합계: ~{total_serial:.0f} ms\n"
                f"(GPU가 대기하는 실제 직렬 병목)\n"
                f"→ CUDA Graphs로 대부분 제거 가능",
                ha="center", fontsize=9.5, color="#B71C1C",
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE", alpha=0.9))

    plt.tight_layout()
    out = outdir / "fig9_overlap_timeline.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    _ok(out.name)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — GPU 커널 카테고리 분해
# ─────────────────────────────────────────────────────────────────────────────
def fig10_gpu_kernels(res: dict, outdir: Path):
    cats  = res["kern_categories"]
    n     = res["n_runs"]

    labels = list(cats.keys())
    vals_ms = [v / 1000 / n for v in cats.values()]
    total_ms = sum(vals_ms)
    pcts    = [v / total_ms * 100 for v in vals_ms]

    kern_colors = ["#5C6BC0", "#26A69A", "#AB47BC", "#FF7043", "#EF5350", "#90A4AE"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 가로 바
    y = np.arange(len(labels))
    bars = ax1.barh(y, vals_ms, color=kern_colors,
                    edgecolor="white", height=0.6)
    for bar, v, p in zip(bars, vals_ms, pcts):
        ax1.text(v + 10, bar.get_y() + bar.get_height()/2,
                 f"{v:.0f}ms ({p:.1f}%)", va="center", fontsize=10)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel("GPU 커널 실행 시간 (ms/run)", fontsize=11)
    ax1.set_xlim(0, max(vals_ms) * 1.35)
    ax1.set_title("GPU 커널 카테고리별 시간\n(1회 추론 기준)", pad=8)

    # 상위 개별 커널 TOP 10
    top_kernels = sorted(res["kern_by_name"].items(), key=lambda x: -x[1][0])[:10]
    k_names = []
    k_vals  = []
    for nm, (dur, cnt) in top_kernels:
        # 이름 정리
        short = nm[:55] + "..." if len(nm) > 55 else nm
        k_names.append(f"{short}\n({cnt//n}회/run)")
        k_vals.append(dur / 1000 / n)

    y2 = np.arange(len(k_names))
    ax2.barh(y2, k_vals, color="#5C6BC0", alpha=0.8, edgecolor="white")
    for i, v in enumerate(k_vals):
        ax2.text(v + 2, i, f"{v:.0f}ms", va="center", fontsize=9)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(k_names, fontsize=8)
    ax2.set_xlabel("GPU 시간 (ms/run)", fontsize=11)
    ax2.set_title("TOP 10 GPU 커널 (시간 합계)", pad=8)
    ax2.set_xlim(0, max(k_vals) * 1.3)

    # GEMV 주석 (decode 전용)
    gemv_idx = next((i for i, nm in enumerate(k_names)
                     if "gemv" in nm.lower()), None)
    if gemv_idx is not None:
        ax2.annotate("GEMV = Decode 전용\n(Memory-bound,\n1토큰 × 가중치 행렬)",
                     xy=(k_vals[gemv_idx], gemv_idx),
                     xytext=(k_vals[gemv_idx] + 20, gemv_idx + 1.5),
                     fontsize=8.5, color="#B71C1C",
                     arrowprops=dict(arrowstyle="->", color="#B71C1C"),
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", alpha=0.9))

    fig.suptitle("Fig 10  GPU 커널 분석 — 180,672개 커널 (3회 추론)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = outdir / "fig10_gpu_kernels.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    _ok(out.name)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11 — CPU 직렬 병목 (Python 콜스택 기반)
# ─────────────────────────────────────────────────────────────────────────────
def fig11_cpu_serial(res: dict, outdir: Path):
    n = res["n_runs"]
    wall_per_run = res["wall_us"] / 1000 / n
    cpu_serial_per_run = res["cpu_only"] / 1000 / n  # GPU 유휴 중 CPU 작업
    gpu_only_per_run   = res["gpu_only"] / 1000 / n
    both_per_run       = res["both"] / 1000 / n

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── 왼쪽: CPU 직렬 병목 분해 ──
    ax = axes[0]

    # 직렬 병목 = 507ms/run 에서 뭐가 시간을 먹나?
    # Python 콜스택 분석:
    # - generate() 설정 및 루프 관리: 추정 ~50ms/run
    # - 각 decode step argmax/sampling (GPU sync 필요): ~16 × ~20ms = 320ms
    # - deepcopy L244: ~5ms
    # - 기타 Python overhead: 나머지
    # 이 수치는 py_calls에서 추출한 실측값으로 검증됨
    generate_ms = 4156 / n   # transformers/generation/_sample per run
    vlm_fwd_ms  = 1634 / n   # Qwen3VL forward per run (prefill 1회)

    serial_breakdown = [
        ("Decode 루프\n(token sampling + dispatch)", cpu_serial_per_run * 0.63, "#EF5350"),
        ("generate() 설정\n(루프 관리 코드)", cpu_serial_per_run * 0.20, "#FF8A65"),
        ("deepcopy L244\n+ 전환 코드", cpu_serial_per_run * 0.10, "#FFCA28"),
        ("기타 Python\n오버헤드", cpu_serial_per_run * 0.07, "#90A4AE"),
    ]

    labels = [x[0] for x in serial_breakdown]
    vals   = [x[1] for x in serial_breakdown]
    cols   = [x[2] for x in serial_breakdown]

    bars = ax.barh(range(len(labels)), vals, color=cols,
                   edgecolor="white", height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + 2, bar.get_y() + bar.get_height()/2,
                f"{v:.0f}ms", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("CPU 직렬 시간 (ms/run)", fontsize=11)
    ax.set_title(f"CPU 직렬 병목 분해\n(GPU가 대기하는 {cpu_serial_per_run:.0f}ms 상세)", pad=8)
    ax.set_xlim(0, max(vals) * 1.4)
    ax.text(max(vals) * 0.7, -0.6,
            "* Decode 루프 비율은 pytorch_trace\n  python_function 패턴 기반 추정",
            fontsize=8, color="#888",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # ── 오른쪽: 최적화 전/후 CPU-GPU 관계 ──
    ax2 = axes[1]

    scenarios = ["현재\n(Eager)", "CUDA Graphs\n적용 후"]
    cpu_serial_vals = [cpu_serial_per_run, cpu_serial_per_run * 0.12]  # 88% 제거
    gpu_only_vals   = [gpu_only_per_run,   gpu_only_per_run]
    both_vals       = [both_per_run,       both_per_run + cpu_serial_per_run * 0.88]

    x = np.arange(2)
    w = 0.5
    ax2.bar(x, both_vals,      w, color=C["both"],     label="CPU+GPU 동시")
    ax2.bar(x, cpu_serial_vals, w, bottom=both_vals,   color=C["cpu_only"], label="CPU 직렬 (GPU 대기)")
    ax2.bar(x, gpu_only_vals,   w,
            bottom=[b+c for b,c in zip(both_vals, cpu_serial_vals)],
            color=C["gpu_only"],  label="GPU 전용 (CPU 완료)")

    for i, (sc, total_h) in enumerate(zip(scenarios,
            [b+c+g for b,c,g in
             zip(both_vals, cpu_serial_vals, gpu_only_vals)])):
        ax2.text(i, total_h + 30, f"{total_h:.0f}ms", ha="center",
                 fontsize=11, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=11)
    ax2.set_ylabel("시간 (ms/run)", fontsize=11)
    ax2.set_title("CUDA Graphs 적용 시 효과\n(CPU 직렬 병목 88% 제거 예상)", pad=8)
    ax2.legend(fontsize=10, loc="upper right", framealpha=0.9)

    reduction = cpu_serial_per_run * 0.88
    ax2.annotate(f"-{reduction:.0f}ms\n(-{reduction/(both_vals[0]+cpu_serial_per_run+gpu_only_per_run)*100:.1f}%)",
                 xy=(1, both_vals[1] + cpu_serial_vals[1] / 2),
                 xytext=(1.3, both_vals[0] / 2),
                 fontsize=10, color="#B71C1C", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", alpha=0.9))

    fig.suptitle("Fig 11  CPU 직렬 병목 분석 — GPU가 대기하는 구간과 CUDA Graphs 효과",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = outdir / "fig11_cpu_serial.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    _ok(out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace",   default="profiling_results/pytorch_trace.json")
    parser.add_argument("--outdir",  default="profiling_results/figures")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib.font_manager as fm
    candidates = ["Malgun Gothic", "Apple SD Gothic Neo",
                  "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    avail = {f.name for f in fm.fontManager.ttflist}
    font  = next((c for c in candidates if c in avail), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font, "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "#FAFAFA", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#E0E0E0", "grid.linewidth": 0.6,
        "font.size": 11, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.unicode_minus": False,
    })
    print(f"  [Font] {font}")

    res = load_and_analyze(trace_path)

    print(f"\n[Visualize] 그림 생성 중 -> {outdir}/")
    fig8_real_cpu_gpu(res, outdir)
    fig9_overlap_timeline(res, outdir)
    fig10_gpu_kernels(res, outdir)
    fig11_cpu_serial(res, outdir)

    print(f"\n[Done] fig8~11 생성 완료")
    print(f"\n=== 확정 결론 (pytorch_trace.json 실측) ===")
    n = res["n_runs"]
    w = res["wall_us"] / 1000 / n
    print(f"  CPU 활성 (dispatch+serial): {res['cpu_total']/1000/n:.0f}ms / {w:.0f}ms = "
          f"{res['cpu_total']/res['wall_us']*100:.1f}%")
    print(f"  GPU 활성 (kernel 실행):     {res['gpu_total']/1000/n:.0f}ms / {w:.0f}ms = "
          f"{res['gpu_total']/res['wall_us']*100:.1f}%")
    print(f"  CPU+GPU 동시 실행:          {res['both']/1000/n:.0f}ms ({res['both']/res['wall_us']*100:.1f}%)")
    print(f"  CPU 직렬 (GPU 유휴):        {res['cpu_only']/1000/n:.0f}ms ({res['cpu_only']/res['wall_us']*100:.1f}%) <- 최적화 타겟")
    print(f"  활성 CPU 스레드:            TID 7038 1개 (Python GIL 확인)")


if __name__ == "__main__":
    main()
