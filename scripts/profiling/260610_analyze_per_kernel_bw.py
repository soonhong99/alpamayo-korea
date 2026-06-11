#!/usr/bin/env python3
"""
260610_analyze_per_kernel_bw.py
목적: 커널별 순간 DRAM 대역폭 및 SM 활용률 분석

두 가지 분석 모드:
  --mode ncu   : ncu CSV에서 커널별 instantaneous BW 계산
  --mode nsys  : nsys SQLite에서 커널 갭(idle time) 분석
  --mode both  : 두 결과 통합

핵심 질문:
  1. DRAM이 실제로 전송할 때 속도는? (계산 시간 제외)
  2. 커널과 커널 사이 GPU가 얼마나 놀고 있나?
  3. 커널 내에서 SM은 전송 중 계산을 하고 있나, 놀고 있나?
"""

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import statistics

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
DRAM_PEAK_GBS = 231.0          # Jetson AGX Thor LPDDR5X 이론 peak (GB/s)
SECTOR_BYTES  = 32             # LTS sector = 32 bytes
LARGE_KERNEL_THRESHOLD_MB = 10 # 이 이상: "대형 커널" (GEMV/GEMM 등), 이하: "소형 커널" (utility)

M_DRAM_READ = "lts__d_sectors_fill_sysmem.sum"
M_DRAM_WRITE = "lts__t_sectors_aperture_sysmem_op_write.sum"
M_L2_HIT    = "lts__t_request_hit_rate.pct"
M_DURATION  = "gpu__time_duration.sum"       # ns
M_SM_ACTIVE = "sm__active_cycles.sum"
M_ELAPSED   = "gpc__cycles_elapsed.max"

TARGET_METRICS = {M_DRAM_READ, M_DRAM_WRITE, M_L2_HIT, M_DURATION, M_SM_ACTIVE, M_ELAPSED}

RESULTS_DIR = Path.home() / "alpamayo1.5/profiling_results/260610_per_kernel_bw"


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def fmt_gb(v: float) -> str:
    return f"{v:.3f} GB"

def fmt_gbs(v: float) -> str:
    return f"{v:.1f} GB/s ({v/DRAM_PEAK_GBS*100:.0f}% peak)"

def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"

def fmt_ms(ns: float) -> str:
    return f"{ns/1e6:.3f} ms"


# ──────────────────────────────────────────────
# Mode 1: ncu CSV → 커널별 instantaneous BW
# ──────────────────────────────────────────────
def parse_ncu_csv(csv_path: Path) -> list[dict]:
    """ncu CSV를 파싱하여 커널별 metric dict 목록 반환."""
    # kernel_id → metric dict
    kernels: dict[str, dict] = defaultdict(lambda: {
        "kernel_name": "",
        M_DRAM_READ:  0.0,
        M_DRAM_WRITE: 0.0,
        M_L2_HIT:     [],   # list → 나중에 평균
        M_DURATION:   0.0,
        M_SM_ACTIVE:  0.0,
        M_ELAPSED:    0.0,
    })

    n_lines = 0
    n_parsed = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            n_lines += 1
            if not any(m in raw_line for m in TARGET_METRICS):
                continue
            try:
                parts = next(csv.reader([raw_line]))
            except Exception:
                continue
            # 헤더: ID(0) ProcessID(1) ProcessName(2) HostName(3)
            #       NVTX_push(4) NVTX_id(5) KernelName(6) Context(7) Stream(8)
            #       BlockSize(9) GridSize(10) Device(11) CC(12)
            #       SectionName(13) MetricName(14) MetricUnit(15) MetricValue(16)
            if len(parts) < 17:
                continue

            kernel_id   = parts[0].strip('"').strip()
            kernel_name = parts[6].strip('"').strip()
            metric_name = parts[14].strip('"').strip()
            metric_val  = parts[16].strip('"').strip().replace(",", "")

            if metric_name not in TARGET_METRICS:
                continue
            try:
                val = float(metric_val)
            except ValueError:
                continue

            k = kernels[kernel_id]
            k["kernel_name"] = kernel_name

            if metric_name == M_L2_HIT:
                k[M_L2_HIT].append(val)
            elif metric_name in (M_DRAM_READ, M_DRAM_WRITE, M_SM_ACTIVE):
                k[metric_name] += val
            elif metric_name == M_DURATION:
                # 일부 버전은 sum, 일부는 max — 가장 큰 값 사용
                k[M_DURATION] = max(k[M_DURATION], val)
            elif metric_name == M_ELAPSED:
                k[M_ELAPSED] = max(k[M_ELAPSED], val)
            n_parsed += 1

    print(f"  파싱: {n_lines:,} lines → {len(kernels):,} 커널 인스턴스 ({n_parsed:,} metric rows)")

    # L2 hit rate 리스트 → 평균
    result = []
    for kid, k in kernels.items():
        if k[M_DURATION] <= 0:
            continue
        k[M_L2_HIT] = statistics.mean(k[M_L2_HIT]) if k[M_L2_HIT] else 0.0
        k["kernel_id"] = kid

        dram_read_bytes  = k[M_DRAM_READ]  * SECTOR_BYTES
        dram_write_bytes = k[M_DRAM_WRITE] * SECTOR_BYTES
        duration_ns      = k[M_DURATION]
        duration_s       = duration_ns / 1e9

        # 순간 DRAM read BW (계산 시간이 kernel 내에 있어도 kernel 단위 최소 분모)
        k["dram_read_GB"]   = dram_read_bytes / 1e9
        k["dram_write_GB"]  = dram_write_bytes / 1e9
        k["duration_ms"]    = duration_ns / 1e6
        k["instant_bw_gbs"] = (dram_read_bytes / 1e9) / duration_s if duration_s > 0 else 0.0

        # SM 활용률: SM이 실제 연산한 사이클 / 커널 경과 사이클
        if k[M_ELAPSED] > 0:
            k["sm_util_pct"] = k[M_SM_ACTIVE] / k[M_ELAPSED] * 100
        else:
            k["sm_util_pct"] = 0.0

        result.append(k)

    return result


def analyze_ncu(stage_name: str = "decode") -> None:
    csv_path = RESULTS_DIR / f"{stage_name}_per_kernel.csv"
    if not csv_path.exists():
        print(f"[오류] 파일 없음: {csv_path}")
        print(f"  먼저 실행: sudo bash 260610_run_ncu_per_kernel_bw.sh")
        return

    print(f"\n{'='*70}")
    print(f"  ncu 커널별 instantaneous BW 분석 — {stage_name.upper()}")
    print(f"  파일: {csv_path}")
    print(f"{'='*70}")

    kernels = parse_ncu_csv(csv_path)
    if not kernels:
        print("  [오류] 파싱된 커널 없음. CSV 형식 확인 필요.")
        return

    # 커널 분류: 대형(DRAM read >= threshold) vs 소형
    threshold_gb = LARGE_KERNEL_THRESHOLD_MB / 1024
    large = [k for k in kernels if k["dram_read_GB"] >= threshold_gb]
    small = [k for k in kernels if k["dram_read_GB"] <  threshold_gb]

    total_dram_read_gb = sum(k["dram_read_GB"] for k in kernels)
    large_dram_read_gb = sum(k["dram_read_GB"] for k in large)

    print(f"\n  커널 수: 전체 {len(kernels):,}  |  대형(≥{LARGE_KERNEL_THRESHOLD_MB}MB read) {len(large):,}  |  소형 {len(small):,}")
    print(f"  DRAM read 총량: {total_dram_read_gb:.3f} GB")
    print(f"    대형 커널: {large_dram_read_gb:.3f} GB ({large_dram_read_gb/total_dram_read_gb*100:.1f}%)")
    print(f"    소형 커널: {total_dram_read_gb-large_dram_read_gb:.3f} GB ({(total_dram_read_gb-large_dram_read_gb)/total_dram_read_gb*100:.1f}%)")

    # ── 대형 커널 분석 (GEMV/GEMM — 가중치 이동 주체) ──
    if large:
        bw_list  = [k["instant_bw_gbs"] for k in large]
        sm_list  = [k["sm_util_pct"] for k in large]
        dur_list = [k["duration_ms"] for k in large]

        # byte-weighted 평균 BW (가장 정확한 "실제 DRAM 속도")
        total_bytes = sum(k["dram_read_GB"] for k in large)
        weighted_bw = sum(k["dram_read_GB"] * k["instant_bw_gbs"] for k in large) / total_bytes

        print(f"\n  ┌─── 대형 커널 instantaneous BW (byte-weighted = 실제 전송 속도) ───┐")
        print(f"  │  byte-weighted 평균 BW : {fmt_gbs(weighted_bw)}")
        print(f"  │  단순 평균 BW          : {fmt_gbs(statistics.mean(bw_list))}")
        print(f"  │  중앙값 BW             : {fmt_gbs(statistics.median(bw_list))}")
        print(f"  │  최솟값 BW             : {fmt_gbs(min(bw_list))}")
        print(f"  │  최댓값 BW             : {fmt_gbs(max(bw_list))}")
        print(f"  ├─── SM 활용률 (SM이 연산 중인 비율) ─────────────────────────────┤")
        print(f"  │  평균 SM 활용률        : {fmt_pct(statistics.mean(sm_list))}")
        print(f"  │  중앙값 SM 활용률      : {fmt_pct(statistics.median(sm_list))}")
        print(f"  │  ↑ 낮을수록 SM은 놀고 있음 (순수 메모리 대기)")
        print(f"  ├─── 커널 실행 시간 ──────────────────────────────────────────────┤")
        print(f"  │  평균 duration         : {statistics.mean(dur_list):.3f} ms")
        print(f"  │  총 duration 합산      : {sum(dur_list):.1f} ms")
        print(f"  └─────────────────────────────────────────────────────────────────┘")

        # DRAM 전송에만 쓰인 이론 시간 (= bytes / peak BW)
        min_transfer_ms = large_dram_read_gb / DRAM_PEAK_GBS * 1000
        print(f"\n  이론 최소 전송 시간 (231 GB/s 가정): {min_transfer_ms:.1f} ms")
        print(f"  실제 대형 커널 총 실행 시간       : {sum(dur_list):.1f} ms")
        print(f"  overhead                          : {sum(dur_list)-min_transfer_ms:.1f} ms ({(sum(dur_list)-min_transfer_ms)/sum(dur_list)*100:.1f}%)")

    # ── BW 분포 히스토그램 ──
    print(f"\n  instantaneous BW 분포 (대형 커널, byte-weighted bins):")
    bins = [(0,50),(50,100),(100,150),(150,180),(180,210),(210,231),(231,999)]
    bin_bytes = defaultdict(float)
    bin_count = defaultdict(int)
    for k in large:
        bw = k["instant_bw_gbs"]
        for lo, hi in bins:
            if lo <= bw < hi:
                bin_bytes[(lo,hi)] += k["dram_read_GB"]
                bin_count[(lo,hi)] += 1
                break
    for lo, hi in bins:
        label = f"{lo:3d}–{min(hi,231):3d} GB/s"
        cnt   = bin_count[(lo,hi)]
        gb    = bin_bytes[(lo,hi)]
        bar   = "█" * int(gb / total_dram_read_gb * 50)
        print(f"  {label} : {bar} {gb:.2f} GB ({cnt} 커널)")

    # ── 상위 20개 DRAM-heavy 커널 (byte-weighted BW 관점) ──
    print(f"\n  DRAM read 상위 20개 커널 (= 실질적 대역폭 점유자):")
    top20 = sorted(large, key=lambda k: k["dram_read_GB"], reverse=True)[:20]
    print(f"  {'커널 이름':45s} {'read GB':>8s} {'BW(GB/s)':>10s} {'SM%':>6s} {'dur(ms)':>8s}")
    print(f"  {'-'*82}")
    for k in top20:
        name = k["kernel_name"][:44]
        print(f"  {name:45s} {k['dram_read_GB']:8.3f} "
              f"{k['instant_bw_gbs']:10.1f} "
              f"{k['sm_util_pct']:6.1f}% "
              f"{k['duration_ms']:8.3f}")

    # ── SM 활용률 낮은 커널 (= 메모리 대기 중, compute 놀고 있음) ──
    pure_mem = [k for k in large if k["sm_util_pct"] < 20]
    if pure_mem:
        pure_gb = sum(k["dram_read_GB"] for k in pure_mem)
        print(f"\n  SM 활용률 < 20% 커널 (거의 순수 메모리 대기):")
        print(f"    {len(pure_mem)}개, {pure_gb:.2f} GB ({pure_gb/large_dram_read_gb*100:.1f}% of large kernels)")
        print(f"    → 이 커널들에서 SM은 계산 없이 DRAM 데이터만 기다림")

    # ── 소형 커널 요약 ──
    if small:
        sm_bw_list = [k["instant_bw_gbs"] for k in small if k["instant_bw_gbs"] > 0]
        small_dur  = sum(k["duration_ms"] for k in small)
        print(f"\n  소형 커널 ({len(small)}개): 총 {total_dram_read_gb-large_dram_read_gb:.3f} GB read, "
              f"총 duration {small_dur:.1f} ms")
        if sm_bw_list:
            print(f"  소형 커널 평균 BW: {statistics.mean(sm_bw_list):.1f} GB/s")
            print(f"  ↑ 이 값이 높아도 의미 없음 — bytes가 작아 전체 트래픽에 기여 미미")


# ──────────────────────────────────────────────
# Mode 2: nsys SQLite → 진짜 instantaneous BW + 커널 갭 분석
# ──────────────────────────────────────────────

# ncu로 측정한 GEMV 커널당 DRAM read (고정값, 260610 측정 확정)
GEMV_DRAM_READ_GB = 1.276   # gemv2T_kernel_val 1회당 read (BF16 4096×4096 weight)
GEMV_NAME_SUBSTR  = "gemv2T_kernel_val"


def _find_kernel_table(cur, tables: list[str]):
    """nsys 버전별로 다른 커널 테이블 이름 자동 탐지."""
    for cand in ["CUPTI_ACTIVITY_KIND_KERNEL", "GPU_KERNELS", "cuda_kernels"]:
        if cand in tables:
            return cand
    # 컬럼 기반 추측
    for t in tables:
        try:
            cols = {r[1].lower() for r in cur.execute(f"PRAGMA table_info({t})").fetchall()}
            if "start" in cols and "end" in cols:
                return t
        except Exception:
            pass
    return None


def _find_string_table(cur, tables: list[str]):
    """StringIds 테이블 탐지 (커널 이름 조회용)."""
    for cand in ["StringIds", "STRINGIDS", "strings", "StringTable"]:
        if cand in tables:
            return cand
    for t in tables:
        try:
            cols = [r[1].lower() for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
            if "id" in cols and "value" in cols:
                return t
        except Exception:
            pass
    return None


def _find_nvtx_table(cur, tables: list[str]):
    """NVTX 이벤트 테이블 탐지."""
    for cand in ["NVTX_EVENTS", "NVTX_PAYLOAD_EVENTS", "nvtx_events"]:
        if cand in tables:
            return cand
    for t in tables:
        if "nvtx" in t.lower():
            return t
    return None


def analyze_nsys() -> None:
    sqlite_path = RESULTS_DIR / "decode_timeline.sqlite"
    if not sqlite_path.exists():
        print(f"[오류] 파일 없음: {sqlite_path}")
        print(f"  먼저 nsys export 실행:")
        print(f"  sudo -E /usr/local/cuda/bin/nsys export --type sqlite \\")
        print(f"    --output {sqlite_path} --force-overwrite true \\")
        print(f"    {RESULTS_DIR}/decode_timeline.nsys-rep")
        return

    print(f"\n{'='*70}")
    print(f"  nsys 실제 커널 타임라인 분석 (replay 오버헤드 없음)")
    print(f"  파일: {sqlite_path}")
    print(f"  핵심: GEMV 실제 duration + ncu bytes → 진짜 instantaneous BW")
    print(f"{'='*70}")

    conn = sqlite3.connect(sqlite_path)
    cur  = conn.cursor()

    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"\n  테이블 목록: {', '.join(tables)}")

    ktable = _find_kernel_table(cur, tables)
    stable = _find_string_table(cur, tables)
    ntable = _find_nvtx_table(cur, tables)

    if ktable is None:
        print("  [오류] CUDA 커널 테이블을 찾지 못함. 스키마 덤프:")
        for t in tables[:8]:
            cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
            print(f"    {t}: {', '.join(cols[:10])}")
        conn.close()
        return

    # ── 커널 테이블 컬럼 매핑 ──
    kcols = [r[1] for r in cur.execute(f"PRAGMA table_info({ktable})").fetchall()]
    print(f"\n  커널 테이블: {ktable}")
    print(f"  컬럼: {', '.join(kcols)}")

    start_col = next((c for c in kcols if c.lower() in
                      ("start", "start_ns", "startns", "ts_start")), None)
    end_col   = next((c for c in kcols if c.lower() in
                      ("end", "end_ns", "endns", "ts_end")), None)
    # 커널 이름: shortName / demangledName (integer → StringIds FK) 또는 직접 text
    name_col_int  = next((c for c in kcols if c.lower() in
                          ("shortname", "demangled_name", "demangledname", "name_id")), None)
    name_col_text = next((c for c in kcols if c.lower() in
                          ("kernelname", "kernel_name", "name") and
                          c not in (name_col_int or "")), None)

    if start_col is None or end_col is None:
        print(f"  [오류] start/end 컬럼 미발견. 컬럼 목록: {kcols}")
        conn.close()
        return

    print(f"  → start='{start_col}', end='{end_col}'")
    print(f"  → name_int='{name_col_int}', name_text='{name_col_text}'")
    print(f"  StringIds: {stable},  NVTX: {ntable}")

    # ── 커널 이름 조회 방식 결정 ──
    if name_col_int and stable:
        # StringIds JOIN 방식 (일반적)
        scols = [r[1] for r in cur.execute(f"PRAGMA table_info({stable})").fetchall()]
        val_col = next((c for c in scols if c.lower() in ("value", "str", "text")), scols[-1])
        id_col  = next((c for c in scols if c.lower() in ("id", "rowid")), scols[0])
        print(f"  StringIds 컬럼: id='{id_col}', value='{val_col}'")

        query = (f"SELECT k.{start_col}, k.{end_col}, s.{val_col} "
                 f"FROM {ktable} k "
                 f"LEFT JOIN {stable} s ON k.{name_col_int} = s.{id_col} "
                 f"ORDER BY k.{start_col}")
    elif name_col_text:
        query = (f"SELECT {start_col}, {end_col}, {name_col_text} "
                 f"FROM {ktable} ORDER BY {start_col}")
    else:
        query = (f"SELECT {start_col}, {end_col}, NULL "
                 f"FROM {ktable} ORDER BY {start_col}")

    rows = cur.execute(query).fetchall()

    # ── NVTX DecodeAll 범위 추출 (있으면 필터링) ──
    decode_start_ns, decode_end_ns = None, None
    if ntable:
        try:
            ncols = [r[1] for r in cur.execute(f"PRAGMA table_info({ntable})").fetchall()]
            print(f"  NVTX 컬럼: {', '.join(ncols)}")
            # text/value 컬럼에서 'DecodeAll' 검색
            text_col = next((c for c in ncols if c.lower() in
                             ("text", "value", "message", "msg", "name")), None)
            t_start  = next((c for c in ncols if c.lower() in
                             ("start", "start_ns", "startns", "timestamp")), None)
            t_end    = next((c for c in ncols if c.lower() in
                             ("end", "end_ns", "endns")), None)
            if text_col and t_start:
                nvtx_rows = cur.execute(
                    f"SELECT {t_start}, {t_end or t_start}, {text_col} "
                    f"FROM {ntable} WHERE {text_col} LIKE '%DecodeAll%'"
                ).fetchall()
                if nvtx_rows:
                    decode_start_ns = min(r[0] for r in nvtx_rows)
                    decode_end_ns   = max(r[1] for r in nvtx_rows if r[1])
                    print(f"\n  DecodeAll NVTX 범위: "
                          f"{decode_start_ns/1e6:.1f} ms ~ {decode_end_ns/1e6:.1f} ms "
                          f"(span: {(decode_end_ns-decode_start_ns)/1e6:.1f} ms)")
        except Exception as e:
            print(f"  NVTX 범위 추출 실패: {e}")

    conn.close()

    if not rows:
        print("  [오류] 커널 데이터 없음.")
        return

    starts_all = [r[0] for r in rows]
    ends_all   = [r[1] for r in rows]
    names_all  = [str(r[2] or "") for r in rows]

    # ── DecodeAll 범위 내 커널만 필터링 ──
    if decode_start_ns and decode_end_ns:
        decode_rows = [(s, e, n) for s, e, n in zip(starts_all, ends_all, names_all)
                       if s >= decode_start_ns and e <= decode_end_ns]
        print(f"  DecodeAll 내 커널: {len(decode_rows):,} / 전체 {len(rows):,}")
    else:
        print("  ⚠ DecodeAll NVTX 범위 미발견 → 전체 커널 분석")
        decode_rows = list(zip(starts_all, ends_all, names_all))

    if not decode_rows:
        print("  [오류] DecodeAll 내 커널 없음. NVTX 범위를 확인하세요.")
        return

    starts = [r[0] for r in decode_rows]
    ends   = [r[1] for r in decode_rows]
    names  = [r[2] for r in decode_rows]

    # ── 전체 타임라인 요약 ──
    total_elapsed_ns = ends[-1] - starts[0]
    total_active_ns  = sum(e - s for s, e in zip(starts, ends))
    idle_ns          = total_elapsed_ns - total_active_ns
    idle_pct         = idle_ns / total_elapsed_ns * 100

    print(f"\n  ┌─── Decode 커널 타임라인 요약 ──────────────────────────────────┐")
    print(f"  │  총 커널 수          : {len(decode_rows):,}")
    print(f"  │  전체 elapsed        : {total_elapsed_ns/1e6:.1f} ms (nsys 오버헤드 포함)")
    print(f"  │  커널 active 합산    : {total_active_ns/1e6:.1f} ms")
    print(f"  │  커널 간 idle 합산   : {idle_ns/1e6:.1f} ms")
    print(f"  │  idle 비율           : {idle_pct:.1f}%  ← GPU가 이 시간만큼 쉼")
    print(f"  └────────────────────────────────────────────────────────────────┘")

    # ── GEMV 커널 분리 → 진짜 instantaneous BW 계산 ──
    gemv_rows = [(s, e, n) for s, e, n in zip(starts, ends, names)
                 if GEMV_NAME_SUBSTR in n]
    other_rows = [(s, e, n) for s, e, n in zip(starts, ends, names)
                  if GEMV_NAME_SUBSTR not in n]

    print(f"\n  커널 분류 (이름 기준):")
    print(f"    GEMV ({GEMV_NAME_SUBSTR}): {len(gemv_rows):,} 개")
    print(f"    기타 (소형 utility)       : {len(other_rows):,} 개")

    if gemv_rows:
        gemv_dur_ns  = [e - s for s, e, _ in gemv_rows]
        gemv_dur_ms  = [d / 1e6 for d in gemv_dur_ns]
        gemv_bw_list = [GEMV_DRAM_READ_GB / (d / 1e9) for d in gemv_dur_ns]

        gemv_total_active_ns = sum(gemv_dur_ns)
        gemv_total_active_ms = gemv_total_active_ns / 1e6

        # byte-weighted 평균 (모든 GEMV가 같은 bytes → 단순 평균 = byte-weighted)
        mean_dur_ms = statistics.mean(gemv_dur_ms)
        mean_bw     = statistics.mean(gemv_bw_list)
        median_bw   = statistics.median(gemv_bw_list)

        print(f"\n  ┌─── GEMV 커널 진짜 instantaneous BW (nsys 실제 타이밍) ─────────┐")
        print(f"  │  커널당 DRAM read (ncu 확정)   : {GEMV_DRAM_READ_GB:.3f} GB")
        print(f"  │  평균 실제 실행 시간            : {mean_dur_ms:.3f} ms/커널")
        print(f"  │")
        print(f"  │  ★ 평균 instantaneous BW       : {fmt_gbs(mean_bw)}")
        print(f"  │    (= {GEMV_DRAM_READ_GB:.3f} GB / {mean_dur_ms:.3f} ms)")
        print(f"  │  중앙값 BW                     : {fmt_gbs(median_bw)}")
        print(f"  │  최솟값 BW                     : {fmt_gbs(min(gemv_bw_list))}")
        print(f"  │  최댓값 BW                     : {fmt_gbs(max(gemv_bw_list))}")
        print(f"  ├─── ncu replay vs nsys 실제 비교 ──────────────────────────────┤")
        print(f"  │  ncu replay 평균 duration      : 17.219 ms  (BW: 77.8 GB/s)")
        print(f"  │  nsys 실제 평균 duration        : {mean_dur_ms:.3f} ms  (BW: {mean_bw:.1f} GB/s)")
        print(f"  │  replay 오버헤드 배율           : {17.219/mean_dur_ms:.2f}×")
        print(f"  ├─── GEMV 시간 분포 ──────────────────────────────────────────┤")
        print(f"  │  GEMV 총 active 시간            : {gemv_total_active_ms:.1f} ms")
        print(f"  │  전체 elapsed 대비              : {gemv_total_active_ms/total_elapsed_ns*1e6*100:.1f}%")
        print(f"  └────────────────────────────────────────────────────────────────┘")

        # BW 분포
        print(f"\n  GEMV instantaneous BW 분포:")
        bins = [(0,100),(100,150),(150,180),(180,200),(200,215),(215,231),(231,999)]
        for lo, hi in bins:
            cnt = sum(1 for bw in gemv_bw_list if lo <= bw < hi)
            bar = "█" * min(cnt * 40 // max(len(gemv_bw_list), 1) + (1 if cnt else 0), 40)
            print(f"  {lo:3d}–{min(hi,231):3d} GB/s : {bar} {cnt} 커널")

    # ── 커널 간 갭 분석 ──
    gaps_ns = [starts[i+1] - ends[i] for i in range(len(starts)-1)]
    gaps_ms = [g / 1e6 for g in gaps_ns]
    pos_gaps = [g for g in gaps_ms if g > 0.001]   # 1μs 이상만 유의미한 갭

    print(f"\n  ┌─── 커널 간 갭 (CPU dispatch / sync 오버헤드) ───────────────────┐")
    print(f"  │  유의미한 갭 수 (>1μs)     : {len(pos_gaps):,}")
    if pos_gaps:
        print(f"  │  총 idle 시간 (>1μs 갭)   : {sum(pos_gaps):.1f} ms")
        print(f"  │  평균 갭                   : {statistics.mean(pos_gaps)*1000:.1f} μs")
        print(f"  │  중앙값 갭                 : {statistics.median(pos_gaps)*1000:.1f} μs")
        print(f"  │  최대 갭                   : {max(pos_gaps):.3f} ms")
        small_gaps  = sum(g for g in pos_gaps if g < 0.1)
        medium_gaps = sum(g for g in pos_gaps if 0.1 <= g < 1)
        large_gaps  = sum(g for g in pos_gaps if g >= 1)
        print(f"  │  갭 합계 분해:")
        print(f"  │    <0.1ms (kernel launch): {small_gaps:.1f} ms")
        print(f"  │    0.1–1ms (Python dispatch): {medium_gaps:.1f} ms")
        print(f"  │    ≥1ms   (sync/alloc):    {large_gaps:.1f} ms  ← 최적화 우선 대상")
    print(f"  └────────────────────────────────────────────────────────────────┘")

    # 갭 분포 히스토그램
    print(f"\n  갭 분포:")
    gap_bins = [(0,0.001),(0.001,0.01),(0.01,0.1),(0.1,1),(1,10),(10,9999)]
    labels   = ["<1μs (무시)", "1–10μs", "10–100μs", "0.1–1ms", "1–10ms", ">10ms"]
    for (lo, hi), label in zip(gap_bins, labels):
        cnt   = sum(1 for g in gaps_ms if lo <= g < hi)
        total = sum(g for g in gaps_ms if lo <= g < hi)
        bar   = "█" * min(cnt * 30 // max(len(gaps_ms), 1) + (1 if cnt else 0), 30)
        print(f"  {label:18s}: {bar} {cnt:5d} 갭  합계 {total:7.2f} ms")

    # 상위 10개 큰 갭
    top10 = sorted(range(len(gaps_ms)), key=lambda i: gaps_ms[i], reverse=True)[:10]
    print(f"\n  최대 갭 TOP 10 (최적화 대상):")
    print(f"  {'#':>3}  {'갭(ms)':>8}  {'직전 커널':35}  {'다음 커널':35}")
    print(f"  {'-'*90}")
    for rank, idx in enumerate(top10):
        if gaps_ms[idx] <= 0:
            break
        p = names[idx][:34] if idx < len(names) else "?"
        n = names[idx+1][:34] if idx+1 < len(names) else "?"
        print(f"  {rank+1:3d}  {gaps_ms[idx]:8.3f}  {p:35}  {n:35}")

    # ── 최종 해석 ──
    print(f"\n  {'─'*68}")
    print(f"  결론 (nsys 실측 기반)")
    print(f"  {'─'*68}")
    if gemv_rows:
        print(f"  ① 진짜 GEMV instantaneous BW : {mean_bw:.0f} GB/s ({mean_bw/DRAM_PEAK_GBS*100:.0f}% of peak)")
        print(f"     ncu replay(77.8 GB/s)의 {17.219/mean_dur_ms:.1f}배 → replay 오버헤드 확인")
    print(f"  ② GPU idle 비율 : {idle_pct:.1f}%  (커널 간 dispatch 갭)")
    if idle_pct < 5:
        print(f"     ✅ idle 거의 없음 → DRAM 전송이 거의 연속적")
    elif idle_pct < 15:
        print(f"     ⚠ 약간의 CPU dispatch 오버헤드 → CUDA Graph 적용 시 단축 가능")
    else:
        print(f"     ❌ idle {idle_pct:.0f}% → Python/PyTorch dispatch가 주 병목")
    print(f"  ③ SM util 0% (ncu 확정) + DRAM-bound BW = Decode는 순수 memory-bound")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="커널별 instantaneous BW 및 GPU idle time 분석"
    )
    parser.add_argument(
        "--mode", choices=["ncu", "nsys", "both"], default="both",
        help="분석 모드: ncu (커널별 BW), nsys (갭 분석), both (둘 다)"
    )
    parser.add_argument(
        "--stage", default="decode",
        help="ncu 분석 단계 이름 (기본: decode)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="결과 디렉토리 (기본: ~/alpamayo1.5/profiling_results/260610_per_kernel_bw)"
    )
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = args.results_dir

    if not RESULTS_DIR.exists():
        print(f"[오류] 결과 디렉토리 없음: {RESULTS_DIR}")
        print("  먼저 ncu/nsys 측정 스크립트를 실행하세요.")
        sys.exit(1)

    if args.mode in ("ncu", "both"):
        analyze_ncu(args.stage)

    if args.mode in ("nsys", "both"):
        analyze_nsys()

    print(f"\n{'='*70}")
    print(f"  분석 완료")
    print(f"  결과 디렉토리: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
