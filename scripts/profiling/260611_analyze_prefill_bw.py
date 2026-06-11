#!/usr/bin/env python3
"""
260611_analyze_prefill_bw.py
============================
LM Prefill 단계 커널별 순간 DRAM 대역폭 + SM 활용률 분석.

방법: ncu DRAM 바이트 + nsys 실제 커널 시간 교차검증 (Decode와 동일).
핵심 질문: Prefill의 각 커널 타입이 compute-bound인가, memory-bound인가?

Usage:
    python3 260611_analyze_prefill_bw.py \\
        --ncu  ~/alpamayo1.5/profiling_results/260611_prefill_bw/prefill_per_kernel.csv \\
        --nsys ~/alpamayo1.5/profiling_results/260610_per_kernel_bw/decode_timeline.sqlite

ncu CSV: 260611_run_ncu_prefill_bw.sh 실행 결과
nsys SQLite: 260610_run_nsys_decode.sh 실행 결과 (전체 추론 포함, Prefill 타이밍도 있음)

[2026-06-11 수정 사항]
1. ncu 2-pass 이중 산정 수정
   - --replay-mode kernel + 6개 메트릭 → 2 pass 수집
   - 각 커널이 CSV에 2회 등장 (pass1: DRAM 메트릭, pass2: SM/시간 메트릭)
   - 총 DRAM 합계가 2× 과산정됨 → 그룹 요약에 실제값(절반) 표기 추가
   - BW 교차검증은 영향 없음: n_match = min(ncu, nsys) = nsys 수 → pass1 바이트만 사용

2. normalize_kernel_name() 추가
   - ncu는 full demangled C++ 이름 (e.g. void flash_fwd_kernel<...>(args))
   - nsys는 shortName (e.g. flash_fwd_kernel)
   - 교차검증 시 정규화된 이름으로 2차 매칭 → FlashAttention 등 포착 가능
"""

import argparse
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DRAM_PEAK_GB_S = 231.0   # Jetson AGX Thor LPDDR5X 이론 peak
SECTOR_BYTES   = 32      # L2 sector = 32 bytes

# NVTX 범위 이름 후보 (nsys SQLite에서 Prefill 범위 탐색)
PREFILL_NVTX_CANDIDATES = [
    "LM_Prefill",        # ← nsys에서 확인된 실제 이름 (2026-06-11 확인)
    "Phase/LM_Prefill",  # 단일 push 방식 (구버전)
    "PrefillOnly",
    "Prefill",
    "prefill",
    "LMPrefill",
]

# ──────────────────────────────────────────────────────────────────
# 커널 이름 정규화: demangled C++ → shortName
# ──────────────────────────────────────────────────────────────────

def normalize_kernel_name(name: str) -> str:
    """
    ncu full demangled C++ 이름 → nsys shortName으로 정규화.

    예:
      "void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...>>(pytorch_flash::Flash_fwd_params)"
       → "flash_fwd_kernel"          ← 네임스페이스 pytorch_flash:: 제거 필수

      "void flash_fwd_kernel<...>(args)" → "flash_fwd_kernel"
      "unrolled_elementwise_kernel<..." → "unrolled_elementwise_kernel"
      "nvjet_tst_256x128_..."           → "nvjet_tst_256x128_..."  (그대로)
      "reduce_kernel<...>"              → "reduce_kernel"

    규칙:
      1. 'void ' 접두사 제거
      2. 첫 번째 '<' 또는 '(' 이전까지 잘라서 qualified name 추출
         (예: "pytorch_flash::flash_fwd_kernel")
      3. '::'가 있으면 마지막 '::' 이후만 사용 (네임스페이스 제거)
         (예: "flash_fwd_kernel")
      4. 첫 번째 \w+ 토큰 반환
    """
    n = name.strip()

    # Step 1: 'void ' 접두사 제거
    if n.startswith("void "):
        n = n[5:].strip()

    # Step 2: 첫 '<' 또는 '(' 이전 위치 (qualified name 끝)
    lt = n.find("<")
    lp = n.find("(")
    bracket = min(
        lt if lt != -1 else len(n),
        lp if lp != -1 else len(n),
    )
    prefix = n[:bracket]  # e.g. "pytorch_flash::flash_fwd_kernel"

    # Step 3: 네임스페이스 제거 (마지막 '::' 이후)
    if "::" in prefix:
        prefix = prefix.rsplit("::", 1)[1]  # "flash_fwd_kernel"

    # Step 4: 첫 \w+ 토큰
    m = re.match(r"(\w+)", prefix)
    if m:
        return m.group(1)

    # fallback: 원래 이름에서 첫 단어
    m2 = re.match(r"(\w+)", n)
    return m2.group(1) if m2 else n


# 커널 분류 규칙 (shortName prefix 기반)
def classify_kernel(name: str) -> str:
    """커널 이름을 기능 그룹으로 분류."""
    n = name.lower()
    if "flash_fwd" in n or "flash_bwd" in n:
        return "FlashAttention (GEMM)"
    if "gemv2t_kernel_val" in n or "gemv2t" in n:
        return "lm_head (GEMV, 구형)"
    if n.startswith("nvjet_tst_"):
        return "nvjet GEMV/GEMM"
    if "xmma_gemm" in n or "_gemm_" in n:
        return "cuBLAS GEMM (대형)"
    if "catarraybatched" in n:
        return "KV concat"
    if "elementwise" in n or "vectorized_elementwise" in n:
        return "elementwise"
    if "softmax" in n:
        return "softmax"
    if "layernorm" in n or "rms_norm" in n:
        return "LayerNorm/RMSNorm"
    if "rotary" in n or "rope" in n:
        return "RoPE"
    if "reduce" in n:
        return "reduction"
    return "기타"


# ──────────────────────────────────────────────────────────────────
# ncu CSV 파서
# ──────────────────────────────────────────────────────────────────

def load_ncu_prefill(csv_path: str) -> list[dict]:
    """
    Prefill ncu CSV에서 커널별 DRAM bytes, SM util, ncu 시간을 파싱.

    반환값: 커널 레코드 리스트
    {
      name:        str          커널 shortName
      dram_read_b: float        DRAM read bytes (L2 fill from DRAM)
      dram_write_b:float        DRAM write bytes
      dram_total_b:float        read + write
      sm_active:   float        sm__active_cycles.sum
      gpc_elapsed: float        gpc__cycles_elapsed.max
      sm_util_pct: float        SM 활용률 (%)
      ncu_dur_us:  float        ncu 시간 (μs) — replay로 인해 부풀려짐
      l2_hit_pct:  float        L2 hit율 (%)
    }
    """
    records = []
    current: dict = {}
    current_name = ""

    def flush(d: dict, name: str) -> dict | None:
        """ncu 멀티행 레코드를 정리해서 반환."""
        if not d or not name:
            return None
        dram_r  = d.get("lts__d_sectors_fill_sysmem.sum", 0.0) * SECTOR_BYTES
        dram_w  = d.get("lts__t_sectors_aperture_sysmem_op_write.sum", 0.0) * SECTOR_BYTES

        # ★ SM util: GB10B (SM 11.0) 확정된 메트릭 (2026-06-11 --list-metrics 실측)
        #
        #   sm__active_cycles.sum      → GB10B 존재하지 않음 (Ampere/Hopper 전용) → 항상 0
        #   gpc__cycles_elapsed.max    → GB10B 존재하지 않음
        #   sm__cycles_active.sum      → GB10B 존재하지 않음
        #
        #   smsp__cycles_active.sum    → ✅ GB10B 지원 확인 (SM 서브파티션 active cycles)
        #   smsp__cycles_elapsed.sum   → ✅ GB10B 지원 확인 (분모, 단위 일치)
        #   sm__throughput.avg.pct_of_peak_sustained_elapsed → ✅ GB10B 지원 (직접 %)
        #
        #   공식: smsp__cycles_active.sum / smsp__cycles_elapsed.sum × 100
        #         = 모든 SMSP의 active cycles 합 / elapsed cycles 합
        #         = SM 평균 활성도 (%) — SM 단위 비율과 동일
        smsp_act    = d.get("smsp__cycles_active.sum", None)    # ✅ GB10B 확인
        smsp_el     = d.get("smsp__cycles_elapsed.sum", None)   # ✅ GB10B 확인 (분모)
        throughput  = d.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", None)  # ✅ GB10B
        ncu_dur     = d.get("gpu__time_duration.sum", 0.0) / 1e3  # ns → μs
        l2_hit      = d.get("lts__t_request_hit_rate.pct", 0.0)

        # SM util 계산 우선순위:
        #   1순위: smsp cycles 비율 (GB10B 확인됨)
        #   2순위: sm throughput % (교차검증)
        #   3순위: None
        #
        # ⚠ 중요 해석 주의 (2026-06-10 실측 확인):
        #   smsp__cycles_active/elapsed = "warp 스케줄링 활성도 (occupancy)"
        #   ≠ "SM compute utilization"
        #   stall 중인 warp도 active로 카운트됨!
        #   → 96.9% = warp occupancy 높음 (GEMM 특성) ≠ compute-bound
        #   → compute vs memory bound 판별은 DRAM BW%가 올바른 지표
        #   → 이 수치가 0%에 가까우면 occupancy 낮음 (Decode GEMV 가능성)
        #      이 수치가 높으면 occupancy 높음 (GEMM 특성, bound 미결정)
        if smsp_act is not None and smsp_el is not None and smsp_el > 0:
            sm_util = min((smsp_act / smsp_el) * 100.0, 100.0)  # 101% cap
            sm_src  = "smsp_occupancy"  # 이름 변경: compute가 아니라 occupancy
        elif throughput is not None and throughput >= 0:
            sm_util = min(throughput, 100.0)
            sm_src  = "sm_throughput_pct"
        else:
            sm_util = None
            sm_src  = "unsupported"

        return {
            "name":         name,
            "dram_read_b":  dram_r,
            "dram_write_b": dram_w,
            "dram_total_b": dram_r + dram_w,
            "smsp_active":  smsp_act or 0.0,
            "smsp_elapsed": smsp_el or 0.0,
            "sm_util_pct":  sm_util,   # None = 미지원/구 데이터
            "sm_util_src":  sm_src,
            "ncu_dur_us":   ncu_dur,
            "l2_hit_pct":   l2_hit,
        }

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            # ncu CSV 헤더: 처음 몇 줄이 주석 ("==PROF==") 또는 공백일 수 있음
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[ERROR] ncu CSV 없음: {csv_path}")
        sys.exit(1)

    # ncu CSV 포맷: 각 커널이 여러 행에 걸쳐 metric 값을 나열
    # 헤더 행 탐색 (ID, Process ID, Process Name, ... 형태)
    header_row = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"ID"') or stripped.startswith("ID,"):
            header_row = i
            data_start = i + 1
            break
        # 또는 "Kernel Name" 포함 행
        if "Kernel Name" in stripped or "kernel_name" in stripped.lower():
            header_row = i
            data_start = i + 1
            break

    if header_row is None:
        print("[ERROR] ncu CSV에서 헤더 행을 찾지 못함. 파일 처음 10줄:")
        for l in lines[:10]:
            print(" ", l.rstrip())
        sys.exit(1)

    # CSV 파싱
    reader = csv.DictReader(lines[header_row:])
    # ncu 컬럼 이름 예시: "Metric Name", "Metric Value", "Kernel Name", "ID" 등
    # 실제 컬럼은 실행 환경에 따라 다를 수 있으므로 유연하게 처리
    fieldnames = reader.fieldnames or []
    lower_fields = {f.lower(): f for f in fieldnames}

    # 핵심 컬럼 찾기
    kernel_col  = lower_fields.get("kernel name") or lower_fields.get("name")
    metric_col  = lower_fields.get("metric name")
    value_col   = lower_fields.get("metric value")

    if not kernel_col or not metric_col or not value_col:
        # 대안: ID + Metric Name + Metric Value 컬럼 구조
        id_col = lower_fields.get("id")
        if id_col and metric_col and value_col:
            # ID 기반 파싱 (커널 이름은 로그 파일에서 별도 추출)
            print("[WARNING] 'Kernel Name' 컬럼 없음. ID 기반 파싱 시도.")
        else:
            print(f"[ERROR] ncu CSV 컬럼 구조 인식 불가. 컬럼: {fieldnames[:10]}")
            sys.exit(1)

    metric_map: dict[str, float] = {}
    cur_name: str = ""

    for row in reader:
        name_val = row.get(kernel_col or "", "").strip().strip('"')
        metric    = row.get(metric_col or "", "").strip()
        value_str = row.get(value_col or "", "").strip().replace(",", "")

        if not metric or not value_str:
            continue

        # 새 커널로 전환
        if name_val and name_val != cur_name:
            if cur_name and metric_map:
                rec = flush(metric_map, cur_name)
                if rec:
                    records.append(rec)
            cur_name = name_val
            metric_map = {}

        try:
            metric_map[metric] = float(value_str)
        except ValueError:
            pass

    # 마지막 커널 처리
    if cur_name and metric_map:
        rec = flush(metric_map, cur_name)
        if rec:
            records.append(rec)

    return records


# ──────────────────────────────────────────────────────────────────
# nsys SQLite 파서 — Prefill 범위 내 커널 타이밍 추출
# ──────────────────────────────────────────────────────────────────

def find_prefill_range_nsys(db_path: str) -> tuple[int, int] | None:
    """nsys SQLite에서 Prefill NVTX 범위의 (start_ns, end_ns)를 반환."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # NVTX 이벤트 테이블 탐색
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0].lower() for row in cur.fetchall()}

    nvtx_tables = [t for t in tables if "nvtx" in t or "marker" in t or "range" in t]
    print(f"[nsys] NVTX 관련 테이블: {nvtx_tables}")

    if not nvtx_tables:
        print("[WARNING] nsys SQLite에 NVTX 테이블 없음. Prefill 범위 추출 불가.")
        conn.close()
        return None

    # NVTX range 이름 목록 출력 (디버깅용)
    for tbl in nvtx_tables:
        try:
            cur.execute(f"PRAGMA table_info({tbl})")
            cols = [c[1] for c in cur.fetchall()]
            print(f"[nsys] {tbl} 컬럼: {cols}")

            # text/name 컬럼 찾기
            text_col = next((c for c in cols if c.lower() in ("text", "name", "value")), None)
            if text_col:
                cur.execute(f"SELECT DISTINCT {text_col} FROM {tbl} LIMIT 30")
                names = [r[0] for r in cur.fetchall() if r[0]]
                print(f"[nsys] {tbl} 이름 목록 (최대 30개): {names}")
        except Exception as e:
            print(f"[nsys] {tbl} 조회 오류: {e}")

    # Prefill 범위 탐색
    for tbl in nvtx_tables:
        try:
            cur.execute(f"PRAGMA table_info({tbl})")
            cols = [c[1] for c in cur.fetchall()]
            cols_lower = {c.lower(): c for c in cols}

            text_col  = cols_lower.get("text") or cols_lower.get("name") or cols_lower.get("value")
            start_col = cols_lower.get("start") or cols_lower.get("startns") or cols_lower.get("timestamp")
            end_col   = cols_lower.get("end")   or cols_lower.get("endns")

            if not text_col or not start_col:
                continue

            for candidate in PREFILL_NVTX_CANDIDATES:
                if end_col:
                    cur.execute(
                        f"SELECT {start_col}, {end_col} FROM {tbl} "
                        f"WHERE {text_col} = ? ORDER BY {start_col} LIMIT 1",
                        (candidate,)
                    )
                else:
                    # push/pop 방식: timestamp만 있는 경우 → start/end 쌍 직접 계산 어려움
                    cur.execute(
                        f"SELECT {start_col} FROM {tbl} "
                        f"WHERE {text_col} = ? ORDER BY {start_col}",
                        (candidate,)
                    )
                row = cur.fetchone()
                if row:
                    if end_col and len(row) == 2:
                        print(f"[nsys] Prefill NVTX 범위 발견: '{candidate}' → {row[0]:,} ~ {row[1]:,} ns")
                        conn.close()
                        return (int(row[0]), int(row[1]))
                    elif len(row) == 1:
                        print(f"[nsys] Prefill NVTX push 발견: '{candidate}' @ {row[0]:,} ns (end 컬럼 없음)")
        except Exception as e:
            print(f"[nsys] {tbl} Prefill 탐색 오류: {e}")

    print("[WARNING] nsys에서 Prefill NVTX 범위를 찾지 못함.")
    print("          아래 후보 이름을 확인 후 --prefill-nvtx 인자로 직접 지정하세요.")
    conn.close()
    return None


def load_nsys_prefill_kernels(
    db_path: str,
    prefill_start_ns: int,
    prefill_end_ns: int,
) -> list[dict]:
    """
    nsys SQLite에서 Prefill 범위 내 커널 이름 + 시작/종료 시간을 로드.

    반환: [{"name": str, "start_ns": int, "end_ns": int, "dur_us": float}, ...]
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 커널 테이블 탐색
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0].lower(): row[0] for row in cur.fetchall()}

    # CUDA kernel 테이블 후보
    kernel_table = None
    for candidate in ("cupti_activity_kind_kernel", "cuda_kernel", "kernel", "gpu_kernel"):
        if candidate in tables:
            kernel_table = tables[candidate]
            break

    if not kernel_table:
        # 부분 이름 매칭
        kernel_table_found = [v for k, v in tables.items() if "kernel" in k]
        if kernel_table_found:
            kernel_table = kernel_table_found[0]

    if not kernel_table:
        print(f"[nsys] 커널 테이블을 찾지 못함. 테이블 목록: {list(tables.values())[:20]}")
        conn.close()
        return []

    cur.execute(f"PRAGMA table_info({kernel_table})")
    cols = {c[1].lower(): c[1] for c in cur.fetchall()}

    start_col = cols.get("start") or cols.get("startns") or cols.get("start_time")
    end_col   = cols.get("end")   or cols.get("endns")   or cols.get("end_time")

    if not start_col or not end_col:
        print(f"[nsys] 커널 테이블 '{kernel_table}' 시간 컬럼 없음. 컬럼: {list(cols.keys())}")
        conn.close()
        return []

    # ── shortName 텍스트 복원: 커널 테이블의 shortName은 INTEGER FK → StringIds JOIN 필수
    # ※ 버그 수정: shortName 컬럼이 커널 테이블에 직접 있어도 INTEGER FK이므로
    #              직접 읽으면 정수가 나온다 → 항상 StringIds와 JOIN해야 함
    #              (260611_analyze_nvjet_bw.py와 동일한 방식)

    # StringIds 테이블 탐색
    shortname_table = None
    for candidate in ("StringIds", "StringTable", "string_table", "stringids", "strings"):
        if candidate.lower() in tables:
            shortname_table = tables[candidate.lower()]
            break

    results = []

    # shortName FK 컬럼 (커널 테이블 내 INTEGER, StringIds를 가리킴)
    sname_fk_col = (
        cols.get("shortname") or cols.get("short_name") or
        cols.get("shortnameid") or cols.get("kernelnameid")
    )

    if shortname_table and sname_fk_col:
        cur.execute(f"PRAGMA table_info({shortname_table})")
        str_cols = {c[1].lower(): c[1] for c in cur.fetchall()}
        id_col  = str_cols.get("id") or str_cols.get("_id_")
        val_col = str_cols.get("value") or str_cols.get("name") or str_cols.get("text")

        if id_col and val_col:
            # JOIN: 커널.shortName (int) = StringIds.id → StringIds.value (text)
            query = (
                f"SELECT k.{start_col}, k.{end_col}, s.{val_col} "
                f"FROM {kernel_table} k "
                f"LEFT JOIN {shortname_table} s ON k.{sname_fk_col} = s.{id_col} "
                f"WHERE k.{start_col} >= ? AND k.{end_col} <= ? "
                f"ORDER BY k.{start_col}"
            )
            cur.execute(query, (prefill_start_ns, prefill_end_ns))
            for start, end, name in cur.fetchall():
                dur_us = (end - start) / 1e3
                results.append({"name": name or "", "start_ns": start, "end_ns": end, "dur_us": dur_us})

            # 디버그: 처음 5개 이름 출력
            sample = [r["name"] for r in results[:5]]
            print(f"[nsys] 첫 5개 커널 이름: {sample}")
        else:
            print(f"[nsys] StringIds 컬럼 탐색 실패. str_cols={list(str_cols.keys())}")
    else:
        print(f"[nsys] shortName FK 컬럼 또는 StringIds 테이블 없음.")
        print(f"       sname_fk_col={sname_fk_col}, shortname_table={shortname_table}")
        print(f"       kernel table cols: {list(cols.keys())[:20]}")

    conn.close()
    print(f"[nsys] Prefill 범위 내 커널 {len(results)}개 로드")
    return results


# ──────────────────────────────────────────────────────────────────
# SM util 및 DRAM BW 요약 출력
# ──────────────────────────────────────────────────────────────────

def summarize_by_group(ncu_records: list[dict]) -> None:
    """
    ncu 레코드를 커널 그룹별로 집계하여 SM util + ncu BW 출력.

    ⚠ 2× 이중 산정 주의:
    ncu --replay-mode kernel + 6 메트릭 → 2 pass → CSV에 각 커널 2회 등장.
    총 DRAM 합계가 2× 과산정됨. 아래 표에 "실제(÷2)" 열 추가.
    """

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in ncu_records:
        g = classify_kernel(r["name"])
        groups[g].append(r)

    print("\n" + "=" * 88)
    print("  [ncu 기반] 그룹별 DRAM + SM util  ⚠ DRAM은 2× 이중산정 (실제=÷2)")
    print("  ⚠ SM% = smsp warp occupancy (≠ compute util). bound 판별은 DRAM BW%로 함.")
    print("=" * 88)
    print(f"{'그룹':<30} {'N':>5} {'DRAM(GB)':>10} {'실제(GB)':>10} {'SM%':>7} {'ncu BW':>10}")
    print("-" * 88)

    total_dram = 0.0
    for group, recs in sorted(groups.items(), key=lambda x: -sum(r["dram_total_b"] for r in x[1])):
        count  = len(recs)
        dram   = sum(r["dram_total_b"] for r in recs)
        real   = dram / 2.0   # 2× 이중산정 보정
        # SM util: None이면 미지원 표시
        sm_vals = [r["sm_util_pct"] for r in recs if r["sm_util_pct"] is not None]
        sm_avg  = sum(sm_vals) / len(sm_vals) if sm_vals else None
        sm_str  = f"{sm_avg:>5.1f}%" if sm_avg is not None else "  N/A"
        ncu_s  = sum(r["ncu_dur_us"] for r in recs) / 1e6
        bw_ncu = (dram / 1e9) / ncu_s if ncu_s > 0 else 0.0
        # bound 판별: DRAM BW% 기반 (smsp_occupancy는 compute/memory 구분 불가)
        # DRAM BW > 60%: DRAM이 bottleneck → memory-bound
        # DRAM BW < 50%: DRAM 한산 → compute/SRAM-bound 가능성
        dram_peak_pct = (real / 1e9) / (ncu_s * DRAM_PEAK_GB_S) * 100 if ncu_s > 0 else 0
        if sm_avg is None:
            occ_str = "occ=N/A"
        else:
            occ_str = f"occ={sm_avg:.0f}%"
        if dram_peak_pct > 60:
            bound = f"DRAM-bound ({occ_str})"
        elif dram_peak_pct > 30:
            bound = f"mixed ({occ_str})"
        else:
            bound = f"compute/SRAM-bound ({occ_str})"
        total_dram += dram

        print(
            f"  {group:<28} {count:>5} {dram/1e9:>9.2f} {real/1e9:>9.2f}"
            f" {sm_str:>7} {bw_ncu:>8.1f} GB/s  ← {bound}"
        )
        # smsp% 설명 주석 (헤더에도 추가)
        # SM%는 warp occupancy (≠ compute utilization). bound 판별은 DRAM BW% 기준.

    print("-" * 80)
    real_total = total_dram / 2.0
    print(
        f"  {'합계':<28} {sum(len(v) for v in groups.values()):>5} "
        f"{total_dram/1e9:>9.2f} {real_total/1e9:>9.2f}"
    )
    print(f"  ※ 실제 총 DRAM ≈ {real_total/1e9:.1f} GB (2× 이중산정 보정 후)")


def summarize_with_nsys(
    ncu_records: list[dict],
    nsys_kernels: list[dict],
) -> None:
    """
    ncu 바이트 + nsys 시간으로 실효 DRAM BW 계산.

    매칭 전략:
    1. 1차: 정확한 이름 일치 (nvjet 계열은 이 경로로 매칭)
    2. 2차: normalize_kernel_name() 적용 후 매칭
       → FlashAttention 등 C++ demangled 이름과 shortName을 연결
    """

    if not nsys_kernels:
        print("\n[WARNING] nsys 커널 데이터 없음. nsys BW 교차검증 생략.")
        return

    # ── ncu 인덱스: 원래 이름 + 정규화 이름으로 이중 인덱스
    ncu_by_name:      dict[str, list[float]] = defaultdict(list)  # 원래 이름
    ncu_by_norm:      dict[str, list[float]] = defaultdict(list)  # 정규화 이름
    ncu_norm_to_orig: dict[str, str]         = {}                 # norm → orig 역매핑
    sm_by_name:       dict[str, list[float]] = defaultdict(list)
    sm_by_norm:       dict[str, list[float]] = defaultdict(list)

    for r in ncu_records:
        orig = r["name"]
        norm = normalize_kernel_name(orig)
        ncu_by_name[orig].append(r["dram_total_b"])
        ncu_by_norm[norm].append(r["dram_total_b"])
        ncu_norm_to_orig[norm] = orig
        sm_by_name[orig].append(r["sm_util_pct"])
        sm_by_norm[norm].append(r["sm_util_pct"])

    # ── nsys 인덱스: 원래 이름 + 정규화 이름
    nsys_by_name: dict[str, list[float]] = defaultdict(list)
    nsys_by_norm: dict[str, list[float]] = defaultdict(list)
    for k in nsys_kernels:
        orig = k["name"]
        norm = normalize_kernel_name(orig)
        nsys_by_name[orig].append(k["dur_us"])
        nsys_by_norm[norm].append(k["dur_us"])

    # ── 매칭: 1차(정확히) → 2차(정규화)
    # 이미 1차 매칭된 이름은 2차에서 제외
    exact_match_names  = set(ncu_by_name.keys()) & set(nsys_by_name.keys())
    norm_match_names   = (set(ncu_by_norm.keys()) & set(nsys_by_norm.keys())) - {
        normalize_kernel_name(n) for n in exact_match_names
    }

    print("\n" + "=" * 92)
    print("  [ncu+nsys 교차검증] 커널 타입별 실효 순간 DRAM BW")
    print("  ⚠ ncu 커널 수가 nsys의 2×인 경우: ncu 2-pass 이중산정 (BW 값은 정확)")
    print("=" * 92)
    print(
        f"{'커널 이름':>42} {'N':>5} {'DRAM/k':>9} "
        f"{'dur/k(μs)':>10} {'SM%':>6} {'BW':>10} {'peak%':>7} {'매칭방식':<6}"
    )
    print("-" * 92)

    matched_dram = 0.0
    rows = []

    # 1차 매칭 (exact)
    for name in sorted(exact_match_names, key=lambda n: -sum(ncu_by_name.get(n, [0]))):
        rows.append((name, ncu_by_name[name], nsys_by_name[name], sm_by_name[name], "exact"))

    # 2차 매칭 (normalized → flash/elementwise 등 demangled 이름 포착)
    for norm in sorted(norm_match_names, key=lambda n: -sum(ncu_by_norm.get(n, [0]))):
        orig = ncu_norm_to_orig.get(norm, norm)
        rows.append((orig, ncu_by_norm[norm], nsys_by_norm[norm], sm_by_norm[norm], f"norm:{norm}"))

    for name, ncu_bytes, nsys_durs, sm_vals, match_kind in rows:
        n_ncu  = len(ncu_bytes)
        n_nsys = len(nsys_durs)
        n_match = min(n_ncu, n_nsys)

        if n_match < 1:
            continue

        # ncu 2-pass 이중산정 경고 (비율이 ~2×인 경우)
        ratio = n_ncu / max(n_nsys, 1)
        if 1.7 < ratio < 2.4:
            flag = f"  [2× pass]"
        elif abs(n_ncu - n_nsys) / max(n_ncu, 1) > 0.2:
            flag = f"  [ncu={n_ncu} nsys={n_nsys}]"
        else:
            flag = ""

        avg_dram_mb  = sum(ncu_bytes[:n_match]) / n_match / 1e6
        avg_dur_us   = sum(nsys_durs[:n_match]) / n_match
        # sm_vals 안에 None이 있을 수 있음 (SM 11.0 미지원)
        sm_valid = [v for v in sm_vals[:n_match] if v is not None]
        avg_sm   = sum(sm_valid) / len(sm_valid) if sm_valid else None
        total_dram_b = sum(ncu_bytes[:n_match])
        total_dur_s  = sum(nsys_durs[:n_match]) / 1e6
        bw_gb_s      = (total_dram_b / 1e9) / total_dur_s if total_dur_s > 0 else 0.0
        pct          = bw_gb_s / DRAM_PEAK_GB_S * 100
        matched_dram += total_dram_b

        short_name = name[-42:] if len(name) > 42 else name
        if avg_sm is None:
            sm_str2 = " N/A%"
        else:
            sm_str2 = f"{avg_sm:>5.1f}%"
        # bound 판별: DRAM BW% 기반 (smsp_occupancy는 compute/memory 구분 불가)
        if pct > 60:
            bound = f"DRAM-bound (occ={avg_sm:.0f}%)" if avg_sm is not None else "DRAM-bound"
        elif pct > 30:
            bound = f"mixed (occ={avg_sm:.0f}%)" if avg_sm is not None else "mixed"
        else:
            bound = f"compute/SRAM-bound (occ={avg_sm:.0f}%)" if avg_sm is not None else "compute/SRAM-bound"
        nsys_err_flag = " ⚠하한" if avg_dur_us < 100 else ""  # nsys 오버헤드 % 클 경우
        print(
            f"  {short_name:>42} {n_match:>5} {avg_dram_mb:>8.1f}MB "
            f"{avg_dur_us:>9.1f} {sm_str2:>6} "
            f"{bw_gb_s:>8.1f}GB/s {pct:>6.1f}%  ← {bound}{flag}{nsys_err_flag}"
        )

    print("-" * 92)
    print(f"\n  매칭된 DRAM 합계: {matched_dram/1e9:.2f} GB (교차검증 범위)")
    print(f"\n  ★ bound 판별 기준: DRAM BW% > 60% → DRAM-bound / 30~60% → mixed / < 30% → compute/SRAM-bound")
    print(f"  ★ SM% = smsp warp occupancy (≠ compute util). 높아도 DRAM-bound일 수 있음.")


def top_kernels_by_dram(ncu_records: list[dict], top_n: int = 20) -> None:
    """DRAM 사용량 상위 커널 출력 (커널 타입 탐색용)."""

    by_name: dict[str, list[float]] = defaultdict(list)
    sm_by:   dict[str, list[float]] = defaultdict(list)
    for r in ncu_records:
        by_name[r["name"]].append(r["dram_total_b"])
        sm_by[r["name"]].append(r["sm_util_pct"])

    print("\n" + "=" * 72)
    print(f"  [ncu] DRAM 상위 {top_n} 커널 타입 (Prefill 커널 탐색)")
    print("=" * 72)
    print(f"{'커널 이름':>45} {'N':>5} {'총DRAM(GB)':>11} {'avg_DRAM(MB)':>13} {'avg_SM(%)':>10}")
    print("-" * 72)

    ranked = sorted(by_name.items(), key=lambda x: -sum(x[1]))
    for name, vals in ranked[:top_n]:
        n    = len(vals)
        tot  = sum(vals) / 1e9
        avg  = sum(vals) / n / 1e6
        sm_raw = [v for v in sm_by[name] if v is not None]
        sm   = sum(sm_raw) / len(sm_raw) if sm_raw else None
        short = name[-45:] if len(name) > 45 else name
        if sm is None:
            sm_str3 = "  N/A"
        else:
            sm_str3 = f"{sm:>5.1f}%"
        # 상위 목록은 DRAM 합계 기준; bound 판별은 그룹 요약(summarize_by_group)에서 수행
        print(f"  {short:>45} {n:>5} {tot:>10.2f} {avg:>12.1f} {sm_str3:>9}")
    print("-" * 72)


# ──────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prefill 순간 DRAM BW + SM util 분석")
    parser.add_argument("--ncu",  required=True,  help="Prefill ncu CSV 경로")
    parser.add_argument("--nsys", required=True,  help="nsys SQLite 경로 (전체 추론, Prefill 구간 포함)")
    parser.add_argument("--prefill-nvtx", default=None,
                        help="Prefill NVTX 범위 이름 (미지정 시 자동 탐색)")
    parser.add_argument("--top",  type=int, default=20, help="상위 커널 출력 수 (기본: 20)")
    args = parser.parse_args()

    print("=" * 72)
    print("  Alpamayo Prefill 단계 커널별 DRAM BW + SM util 분석")
    print(f"  ncu  : {args.ncu}")
    print(f"  nsys : {args.nsys}")
    print("=" * 72)

    # ── Step 1: ncu CSV 로드
    print("\n[1/4] ncu CSV 파싱 중...")
    ncu_records = load_ncu_prefill(args.ncu)
    print(f"  로드된 커널 레코드: {len(ncu_records)}개")

    if not ncu_records:
        print("[ERROR] ncu 레코드 0개. CSV 경로/포맷 확인 필요.")
        sys.exit(1)

    # ── Step 2: ncu 기반 그룹 요약 (SM util + ncu BW)
    print("\n[2/4] ncu 기반 그룹별 SM util + DRAM 분석...")
    top_kernels_by_dram(ncu_records, top_n=args.top)
    summarize_by_group(ncu_records)

    # ── Step 3: nsys Prefill 범위 탐색
    print("\n[3/4] nsys에서 Prefill NVTX 범위 탐색...")
    if args.prefill_nvtx:
        print(f"  지정된 NVTX 이름: {args.prefill_nvtx}")
        # 지정된 이름으로 탐색
        PREFILL_NVTX_CANDIDATES.insert(0, args.prefill_nvtx)

    nsys_range = find_prefill_range_nsys(args.nsys)

    # ── Step 4: nsys 커널 타이밍 로드 + 교차검증
    if nsys_range:
        start_ns, end_ns = nsys_range
        duration_ms = (end_ns - start_ns) / 1e6
        print(f"\n[4/4] nsys Prefill 범위 내 커널 로드...")
        print(f"  범위: {start_ns:,} ~ {end_ns:,} ns ({duration_ms:.1f} ms)")

        nsys_kernels = load_nsys_prefill_kernels(args.nsys, start_ns, end_ns)
        summarize_with_nsys(ncu_records, nsys_kernels)
    else:
        print("\n[4/4] nsys Prefill 범위 없음. nsys BW 교차검증 생략.")
        print("  → ncu 기반 SM util 분석 결과만 유효.")
        print("  → nsys 범위 이름을 --prefill-nvtx 인자로 직접 지정하거나,")
        print("     nsys SQLite의 NVTX 이름 목록을 확인 후 재실행하세요.")

    # ── 결론 요약
    print("\n" + "=" * 72)
    print("  결론 가이드")
    print("=" * 72)
    print("""
  ★ DRAM BW% — Async Pipeline 설계 기준:
    DRAM BW > 70% : BW 여유 < 30% → Stream 2 DMA 삽입 여지 적음
                    → 커널 간 갭(GPU idle) 제거(CUDA Graph)가 주요 타겟
    DRAM BW 30~60%: BW 여유 40~70% → Stream 2 cudaMemPrefetchAsync 최적 구간
                    → 현재 커널 실행 중 다음 레이어 가중치 DMA 가능
    DRAM BW < 30% : BW 여유 > 70% → DMA 기회 가장 많음 (SRAM 재사용 구간)

  SM% (smsp warp occupancy) 해석:
    - 89~100%: warp가 거의 항상 스케줄링됨 (GEMM/GEMV 공통)
    - DRAM stall 중인 warp도 "active" 카운트 → SM "빈 슬롯" 없음
    - "SM이 논다" ≠ smsp 낮음. GPU idle은 커널 간 dispatch 갭으로 측정해야 함.

  [2026-06-11 실측 결론]
    smsp_occupancy 89~100% (이전 SM=0%는 GB10B 미지원 메트릭으로 인한 오류).

    Async Pipeline 기회 (DRAM BW 여유 기준, Prefill):
      - nvjet GEMM (DRAM 70%):      여유 69 GB/s  → 레이어 간 DMA 부분 중첩 가능
      - FlashAttention (DRAM 39%):  여유 140 GB/s → ★ 최대 DMA 기회 (41ms × 140 GB/s = 5.7 GB)
      - elementwise (DRAM 74~98%): 여유 5~60 GB/s → 제한적

    GPU idle (Decode, 커널 간 갭):
      10.6% = 229 ms → 35,498건 × 평균 4.5 μs gap
      → CUDA Graph 적용 시 제거 가능

  ncu 2× 이중산정 처리:
    - 그룹 요약의 '실제(GB)' 열 = 실제 DRAM (÷2)
    - 교차검증 BW 값은 정확 (n_match = nsys 수 → pass1 바이트만 사용)
""")


if __name__ == "__main__":
    main()
