#!/usr/bin/env python3
"""
260611_analyze_nvjet_bw.py

FFN/QKV 가중치 GEMV(nvjet_tst_*) 순간 DRAM 대역폭 측정

추가 실험 없이 기존 데이터로 분석:
  - 260610 ncu CSV  : nvjet 타입별 평균 DRAM read bytes (전체 템플릿 이름 기반)
  - 260610 nsys SQLite: nvjet 타입별 평균 실행 시간 (shortName JOIN)

핵심 문제 해결:
  기존 분석(260610_analyze_per_kernel_bw.py)은 nsys에서 demangledName JOIN을 사용해
  lm_head(19개)만 매칭됐다. shortName JOIN으로 바꾸면 nvjet_tst_* 전체가 잡힌다.

BW 계산:
  순간 BW = (ncu DRAM bytes per kernel) / (nsys 실행 시간 per kernel)

실행:
  python3 260611_analyze_nvjet_bw.py
  python3 260611_analyze_nvjet_bw.py --results-dir /path/to/results
"""

import argparse
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import statistics

# ──────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────
DRAM_PEAK_GBS = 231.0
SECTOR_BYTES  = 32

RESULTS_DIR = Path.home() / "alpamayo1.5/profiling_results/260610_per_kernel_bw"
NCU_CSV     = RESULTS_DIR / "decode_per_kernel.csv"
NSYS_DB     = RESULTS_DIR / "decode_timeline.sqlite"

M_DRAM_READ = "lts__d_sectors_fill_sysmem.sum"
M_DURATION  = "gpu__time_duration.sum"
M_SM_ACTIVE = "sm__active_cycles.sum"
M_ELAPSED   = "gpc__cycles_elapsed.max"
TARGET_METRICS = {M_DRAM_READ, M_DURATION, M_SM_ACTIVE, M_ELAPSED}


# ──────────────────────────────────────────────────────
# nvjet 이름 추출
# ──────────────────────────────────────────────────────

def extract_nvjet_prefix(demangled_name: str) -> str:
    """
    ncu demangled 이름에서 nvjet prefix 추출.

    ncu 예:  void gemv2T_kernel_val<int, int, __nv_bfloat16, ..., 256, 128, 64, 5, 2, 1, ...>
    nsys 예: nvjet_tst_256x128_64x5_2x1_v_bz_TNT
    공통 prefix: nvjet_tst_256x128_64x5_2x1

    식별 규칙 (관측된 모든 nvjet 타입의 공통 패턴):
      - 세 번째 파라미터 c == 64  (K-tile, 항상 64)
      - 다섯 번째 파라미터 e == 2  (warp rows)
      - 여섯 번째 파라미터 f == 1  (warp cols)
      - 첫 번째 파라미터 a >= 64  (tile 크기)

    lm_head 전용 gemv2T는 vocab=152064으로 이 패턴에 맞지 않아 자동 제외됨.
    """
    if "gemv2T_kernel_val" not in demangled_name:
        return ""

    nums = [int(x) for x in re.findall(r"\b(\d+)\b", demangled_name)]
    for i in range(len(nums) - 5):
        a, b, c, d, e, f = nums[i], nums[i+1], nums[i+2], nums[i+3], nums[i+4], nums[i+5]
        # nvjet 식별 조건
        if c == 64 and e == 2 and f == 1 and a >= 64 and b >= 1 and d >= 1:
            return f"nvjet_tst_{a}x{b}_{c}x{d}_{e}x{f}"
    return ""


def nsys_shortname_to_prefix(short_name: str) -> str:
    """
    nsys shortName에서 nvjet prefix 추출.

    예: nvjet_tst_256x128_64x5_2x1_v_bz_TNT
        → split("_") = [nvjet, tst, 256x128, 64x5, 2x1, v, bz, TNT]
        → parts[:5]  = [nvjet, tst, 256x128, 64x5, 2x1]
        → "nvjet_tst_256x128_64x5_2x1"

    2cta 변형도 동일하게 매핑됨:
        nvjet_tst_448x64_64x3_2x1_2cta_v_bz_TNT → nvjet_tst_448x64_64x3_2x1
    """
    if not short_name.startswith("nvjet_tst_"):
        return ""
    parts = short_name.split("_")
    # parts: [nvjet, tst, AxB, CxD, ExF, ...]
    if len(parts) < 5:
        return ""
    return "_".join(parts[:5])


# ──────────────────────────────────────────────────────
# Step 1: ncu CSV → nvjet 타입별 평균 DRAM bytes
# ──────────────────────────────────────────────────────

def load_ncu_nvjet(csv_path: Path) -> dict:
    """
    ncu CSV 파싱 → nvjet 타입별 {count, mean_dram_gb, total_dram_gb} 반환.

    핵심: k["kernel_name"]에는 전체 템플릿 이름이 있음 (표시만 44자 잘렸을 뿐).
    전체 이름에서 extract_nvjet_prefix()로 타입 구분.
    """
    kernels: dict = defaultdict(lambda: {
        "name": "", M_DRAM_READ: 0.0, M_DURATION: 0.0,
        M_SM_ACTIVE: 0.0, M_ELAPSED: 0.0,
    })

    n_lines = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            n_lines += 1
            if not any(m in raw_line for m in TARGET_METRICS):
                continue
            try:
                parts = next(csv.reader([raw_line]))
            except Exception:
                continue
            if len(parts) < 17:
                continue

            kid   = parts[0].strip('"').strip()
            name  = parts[6].strip('"').strip()
            mname = parts[14].strip('"').strip()
            mval  = parts[16].strip('"').strip().replace(",", "")

            if mname not in TARGET_METRICS:
                continue
            try:
                val = float(mval)
            except ValueError:
                continue

            k = kernels[kid]
            k["name"] = name
            if mname == M_DRAM_READ:
                k[M_DRAM_READ] += val
            elif mname == M_DURATION:
                k[M_DURATION] = max(k[M_DURATION], val)
            elif mname == M_SM_ACTIVE:
                k[M_SM_ACTIVE] += val
            elif mname == M_ELAPSED:
                k[M_ELAPSED] = max(k[M_ELAPSED], val)

    print(f"  ncu: {n_lines:,} lines → {len(kernels):,} 커널 인스턴스")

    # nvjet 타입별 그룹핑
    nvjet_groups: dict = defaultdict(list)  # prefix → list of dram_read_GB
    lmhead_list = []
    n_matched = 0

    for kid, k in kernels.items():
        if k[M_DURATION] <= 0:
            continue
        dram_gb = k[M_DRAM_READ] * SECTOR_BYTES / 1e9

        # ★ 핵심 수정: ncu도 nvjet 커널은 shortName(nvjet_tst_*)으로 직접 저장한다.
        #   gemv2T 템플릿 이름을 쓰는 것은 lm_head뿐.
        if k["name"].startswith("nvjet_tst_"):
            prefix = nsys_shortname_to_prefix(k["name"])
            if prefix:
                nvjet_groups[prefix].append(dram_gb)
                n_matched += 1
        elif "gemv2T_kernel_val" in k["name"]:
            # lm_head: gemv2T 경로 사용 (vocab이 커서 nvjet 타일 범위 초과)
            lmhead_list.append(dram_gb)

    print(f"  nvjet 타입 수: {len(nvjet_groups)}  (매칭 커널: {n_matched}개)")
    print(f"  lm_head (gemv2T non-nvjet): {len(lmhead_list)}개")

    # 결과 조합
    result = {}
    for prefix, lst in nvjet_groups.items():
        result[prefix] = {
            "count":         len(lst),
            "mean_dram_gb":  statistics.mean(lst),
            "total_dram_gb": sum(lst),
        }
    if lmhead_list:
        result["lmhead"] = {
            "count":         len(lmhead_list),
            "mean_dram_gb":  statistics.mean(lmhead_list),
            "total_dram_gb": sum(lmhead_list),
        }
    return result


# ──────────────────────────────────────────────────────
# Step 2: nsys SQLite → nvjet 타입별 평균 실행 시간
# ──────────────────────────────────────────────────────

def _auto_detect(cur, tables, candidates):
    """테이블/컬럼 자동 탐지 헬퍼."""
    for c in candidates:
        if c in tables:
            return c
    return None


def load_nsys_nvjet(db_path: Path) -> dict:
    """
    nsys SQLite에서 shortName JOIN으로 nvjet 타입별 평균 실행 시간(ns) 반환.

    기존 분석(260610)의 demangledName JOIN과 달리 shortName을 명시적으로 지정.
    shortName은 CUPTI 커널 테이블의 컬럼 중 하나 (INTEGER FK → StringIds).
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"  nsys 테이블: {', '.join(tables)}")

    # 테이블 탐지
    ktable = _auto_detect(cur, tables,
        ["CUPTI_ACTIVITY_KIND_KERNEL", "GPU_KERNELS", "cuda_kernels"])
    if ktable is None:
        ktable = next((t for t in tables
                       if "kernel" in t.lower() and "string" not in t.lower()), None)
    stable = _auto_detect(cur, tables, ["StringIds", "STRINGIDS", "StringTable"])
    if stable is None:
        stable = next((t for t in tables if "string" in t.lower()), None)
    ntable = _auto_detect(cur, tables, ["NVTX_EVENTS", "NVTX_PAYLOAD_EVENTS"])
    if ntable is None:
        ntable = next((t for t in tables if "nvtx" in t.lower()), None)

    if ktable is None or stable is None:
        print(f"  [오류] 필요 테이블 미발견. tables={tables}")
        conn.close()
        return {}

    # 컬럼 탐지
    kcols = [r[1] for r in cur.execute(f"PRAGMA table_info({ktable})").fetchall()]
    scols = [r[1] for r in cur.execute(f"PRAGMA table_info({stable})").fetchall()]

    # ★ shortName 컬럼을 명시적으로 찾는다 (기존 스크립트는 demangledName이 먼저 선택됨)
    short_col = next((c for c in kcols if c.lower() == "shortname"), None)
    start_col = next((c for c in kcols if c.lower() in ("start", "start_ns", "startns")), None)
    end_col   = next((c for c in kcols if c.lower() in ("end", "end_ns", "endns")), None)
    val_col   = next((c for c in scols if c.lower() in ("value", "str", "text")), None)
    id_col    = next((c for c in scols if c.lower() == "id"), None)

    if not all([short_col, start_col, end_col, val_col, id_col]):
        print(f"  [오류] 컬럼 미발견: shortName={short_col}, start={start_col}, "
              f"end={end_col}, StringIds.value={val_col}")
        print(f"  kcols: {kcols}")
        conn.close()
        return {}

    print(f"  커널 테이블: {ktable}, shortName 컬럼: {short_col}")

    # ── DecodeAll NVTX 범위 ──
    decode_start, decode_end = None, None
    if ntable:
        try:
            ncols = [r[1] for r in cur.execute(f"PRAGMA table_info({ntable})").fetchall()]
            txt_c = next((c for c in ncols if c.lower() in
                          ("text", "value", "message", "msg")), None)
            ts_c  = next((c for c in ncols if c.lower() in
                          ("start", "start_ns", "startns", "timestamp")), None)
            te_c  = next((c for c in ncols if c.lower() in
                          ("end", "end_ns", "endns")), None)
            if txt_c and ts_c:
                nvtx = cur.execute(
                    f"SELECT {ts_c}, {te_c or ts_c}, {txt_c} "
                    f"FROM {ntable} WHERE {txt_c} LIKE '%DecodeAll%'"
                ).fetchall()
                if nvtx:
                    decode_start = min(r[0] for r in nvtx)
                    decode_end   = max(r[1] for r in nvtx if r[1])
                    print(f"  DecodeAll: {decode_start/1e6:.1f}ms ~ {decode_end/1e6:.1f}ms "
                          f"(span {(decode_end-decode_start)/1e6:.1f}ms)")
        except Exception as e:
            print(f"  NVTX 범위 추출 실패: {e}")

    # ── shortName JOIN 쿼리 (핵심 변경점) ──
    query = (f"SELECT k.{start_col}, k.{end_col}, s.{val_col} "
             f"FROM {ktable} k "
             f"LEFT JOIN {stable} s ON k.{short_col} = s.{id_col} "
             f"ORDER BY k.{start_col}")
    rows = cur.execute(query).fetchall()
    conn.close()
    print(f"  nsys: 총 {len(rows):,} 커널 로드")

    # DecodeAll 필터링
    if decode_start and decode_end:
        rows = [(s, e, n) for s, e, n in rows
                if s >= decode_start and (e or s) <= decode_end]
        print(f"  nsys: DecodeAll 내 {len(rows):,} 커널")

    # nvjet 타입별 그룹핑
    nvjet_groups: dict = defaultdict(list)   # prefix → list of duration_ns
    lmhead_durs = []
    n_nvjet = 0

    for start_ns, end_ns, name in rows:
        if name is None or end_ns is None:
            continue
        dur_ns = end_ns - start_ns
        if dur_ns <= 0:
            continue
        name_str = str(name)

        if name_str.startswith("nvjet_tst_"):
            prefix = nsys_shortname_to_prefix(name_str)
            if prefix:
                nvjet_groups[prefix].append(dur_ns)
                n_nvjet += 1
        elif "gemv2T" in name_str:
            lmhead_durs.append(dur_ns)

    print(f"  nsys: nvjet 타입 {len(nvjet_groups)}개, 총 {n_nvjet}개 커널 매칭")
    if lmhead_durs:
        print(f"  nsys: lm_head(gemv2T) {len(lmhead_durs)}개 "
              f"(avg {statistics.mean(lmhead_durs)/1e6:.3f}ms)")

    result = {}
    for prefix, durs in nvjet_groups.items():
        result[prefix] = {
            "count":       len(durs),
            "mean_dur_ns": statistics.mean(durs),
            "std_dur_ms":  statistics.stdev(durs)/1e6 if len(durs) > 1 else 0.0,
            "total_dur_ms": sum(durs) / 1e6,
        }
    if lmhead_durs:
        result["lmhead"] = {
            "count":       len(lmhead_durs),
            "mean_dur_ns": statistics.mean(lmhead_durs),
            "std_dur_ms":  statistics.stdev(lmhead_durs)/1e6 if len(lmhead_durs) > 1 else 0.0,
            "total_dur_ms": sum(lmhead_durs) / 1e6,
        }
    return result


# ──────────────────────────────────────────────────────
# Step 3: 교차검증 및 BW 계산
# ──────────────────────────────────────────────────────

# 어느 투영층인지 힌트 (DRAM 크기와 개수로 역산)
# 36 layers × 19 steps = 684, 36×2×19 = 1368, 등
def guess_projection(prefix: str, count: int, mean_mb: float) -> str:
    n = count
    if n == 19:
        return "lm_head [4096×152064]"
    if n == 216:
        return "FFN gate+up 융합 or Action Expert FFN"
    if n == 684:  # 36×19
        if mean_mb > 100:
            return "FFN down proj [11008→4096] (36L×19s)"
        return "attention or misc (36L×19s)"
    if n == 1368:  # 36×19×2
        if mean_mb > 80:
            return "FFN gate/up [4096→11008] (36L×19s×2)"
        return "Q/K/V proj (36L×19s×N)"
    if n == 720:   # 36×20 or other
        return "proj 변형 (GQA head or O_proj)"
    if n == 360:
        return "O proj [4096→4096] or similar"
    return f"미분류 ({n}개)"


def main():
    parser = argparse.ArgumentParser(
        description="nvjet GEMV 순간 DRAM BW 분석 (ncu + nsys 교차검증)"
    )
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="결과 디렉토리 (기본: ~/alpamayo1.5/profiling_results/260610_per_kernel_bw)")
    args = parser.parse_args()

    global RESULTS_DIR, NCU_CSV, NSYS_DB
    if args.results_dir:
        RESULTS_DIR = args.results_dir
        NCU_CSV     = RESULTS_DIR / "decode_per_kernel.csv"
        NSYS_DB     = RESULTS_DIR / "decode_timeline.sqlite"

    if not NCU_CSV.exists():
        print(f"[오류] ncu CSV 없음: {NCU_CSV}")
        sys.exit(1)
    if not NSYS_DB.exists():
        print(f"[오류] nsys SQLite 없음: {NSYS_DB}")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  FFN/QKV nvjet GEMV 순간 DRAM BW (ncu bytes ÷ nsys 시간)")
    print(f"  추가 실험 불필요 — 기존 데이터 재분석")
    print(f"{'='*72}")

    print(f"\n[1/3] ncu CSV 파싱 (전체 템플릿 이름 기반 그룹핑)...")
    ncu = load_ncu_nvjet(NCU_CSV)

    print(f"\n[2/3] nsys SQLite 분석 (shortName JOIN)...")
    nsy = load_nsys_nvjet(NSYS_DB)

    print(f"\n[3/3] BW 계산 및 출력...")

    # 매칭
    all_keys = set(ncu.keys()) | set(nsy.keys())
    rows = []

    for key in all_keys:
        n = ncu.get(key, {})
        s = nsy.get(key, {})

        bw = None
        peak = None
        if n.get("mean_dram_gb") and s.get("mean_dur_ns") and s["mean_dur_ns"] > 0:
            bw   = n["mean_dram_gb"] / (s["mean_dur_ns"] / 1e9)
            peak = bw / DRAM_PEAK_GBS * 100

        rows.append({
            "prefix":        key,
            "ncu_count":     n.get("count", 0),
            "nsys_count":    s.get("count", 0),
            "mean_dram_mb":  (n.get("mean_dram_gb") or 0) * 1000,
            "total_dram_gb": n.get("total_dram_gb") or 0,
            "mean_dur_ms":   (s.get("mean_dur_ns") or 0) / 1e6,
            "bw":            bw,
            "peak":          peak,
        })

    # 총 DRAM 기준 내림차순 정렬
    rows.sort(key=lambda r: r["total_dram_gb"], reverse=True)

    # ── 결과 테이블 ──
    print(f"\n  {'커널 타입':<28} {'투영층 추정':<28} {'ncu':>4} {'nsys':>4} "
          f"{'DRAM/k':>7} {'총GB':>6} {'시간/k':>7} {'순간BW':>22}")
    print(f"  {'-'*108}")

    matched = [r for r in rows if r["bw"] is not None]
    unmatched = [r for r in rows if r["bw"] is None]

    for r in rows:
        pname  = r["prefix"].replace("nvjet_tst_", "")[:27]
        guess  = guess_projection(r["prefix"], r["nsys_count"] or r["ncu_count"],
                                  r["mean_dram_mb"])[:27]
        dram_k = f"{r['mean_dram_mb']:.0f}MB"   if r["mean_dram_mb"] else "?"
        dram_t = f"{r['total_dram_gb']:.2f}GB"  if r["total_dram_gb"] else "?"
        dur_k  = f"{r['mean_dur_ms']:.3f}ms"    if r["mean_dur_ms"] else "?"

        if r["bw"] is not None:
            bw_str = f"★ {r['bw']:.1f} GB/s ({r['peak']:.0f}% peak)"
        else:
            bw_str = "─ 매칭 실패 (ncu/nsys 개수 불일치 가능)"

        print(f"  {pname:<28} {guess:<28} {r['ncu_count']:4d} {r['nsys_count']:4d} "
              f"{dram_k:>7} {dram_t:>6} {dur_k:>7} {bw_str}")

    # ── 요약 통계 ──
    if matched:
        total_bytes = sum(r["total_dram_gb"] for r in matched)
        w_bw        = (sum(r["total_dram_gb"] * r["bw"] for r in matched) / total_bytes
                       if total_bytes > 0 else 0)

        print(f"\n  ┌─── 요약 (nvjet GEMV 전체) {'─'*46}┐")
        print(f"  │ byte-weighted 평균 BW : {w_bw:.1f} GB/s ({w_bw/DRAM_PEAK_GBS*100:.0f}% peak)")
        print(f"  │ 매칭 성공 타입        : {len(matched)}개 / 전체 {len(rows)}개")
        print(f"  │ 총 DRAM (매칭 타입)   : {total_bytes:.2f} GB")
        print(f"  ├─── 비교 기준 {'─'*58}┤")
        print(f"  │ lm_head 순간 BW       : 212.0 GB/s (92% peak) ← 이전 측정")
        print(f"  │ stage 평균 BW         : 204.6 GB/s (89% peak) ← 260609")
        print(f"  └{'─'*70}┘")

        # BW 분포 분석
        bw_vals = [r["bw"] for r in matched if "lmhead" not in r["prefix"]]
        if bw_vals:
            print(f"\n  nvjet GEMV BW 범위:")
            print(f"    최솟값: {min(bw_vals):.1f} GB/s ({min(bw_vals)/DRAM_PEAK_GBS*100:.0f}%)")
            print(f"    최댓값: {max(bw_vals):.1f} GB/s ({max(bw_vals)/DRAM_PEAK_GBS*100:.0f}%)")
            if len(bw_vals) > 1:
                print(f"    표준편차: {statistics.stdev(bw_vals):.1f} GB/s")
            print(f"\n    BW가 lm_head(212 GB/s)와 유사하면 → 모든 GEMV가 DRAM-bound 확정")
            print(f"    크게 낮은 타입 → 해당 커널은 다른 병목 존재 (compute? startup latency?)")

    if unmatched:
        print(f"\n  ⚠ 매칭 실패 타입 ({len(unmatched)}개):")
        for r in unmatched:
            reason = "nsys에서 미발견" if r["nsys_count"] == 0 else "ncu에서 미발견"
            print(f"    {r['prefix']}: {reason}")
        print(f"  → ncu/nsys 동일 실행 기준 확인 필요 (단계 범위, 배치 등)")

    print(f"\n  분석 완료. 데이터: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
