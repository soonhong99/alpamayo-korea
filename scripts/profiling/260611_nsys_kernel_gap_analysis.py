#!/usr/bin/env python3
"""
260611_nsys_kernel_gap_analysis.py
===================================
nsys sqlite에서 각 추론 단계별 커널 간 유휴 시간(inter-kernel gap)을 직접 측정.

실행:
  python3 260611_nsys_kernel_gap_analysis.py \
    --sqlite ~/alpamayo1.5/profiling_results/260610_per_kernel_bw/decode_timeline.sqlite
"""

import argparse
import sqlite3
import os
import sys
from collections import defaultdict


def ns_to_ms(ns: int) -> float:
    return ns / 1e6


def load_string_ids(conn: sqlite3.Connection) -> dict:
    """StringIds 테이블 전체 로드 → {id: string}"""
    cursor = conn.cursor()
    str_map = {}
    for row in cursor.execute("SELECT id, value FROM StringIds"):
        str_map[row[0]] = row[1]
    return str_map


def list_all_nvtx_ranges(conn: sqlite3.Connection, str_map: dict):
    """디버그용: sqlite 안의 모든 NVTX range 이름 출력"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > 0
        ORDER BY text
    """)
    rows = cursor.fetchall()

    print("\n  [디버그] sqlite 안의 모든 NVTX range 이름:")
    seen = set()
    for text, text_id in rows:
        # text가 None이면 textId로 lookup
        name = text if text else str_map.get(text_id, f"ID:{text_id}")
        if name and name not in seen:
            seen.add(name)
            print(f"    '{name}'")


def query_nvtx_ranges(conn: sqlite3.Connection, str_map: dict,
                      stage_map: dict) -> dict:
    """각 단계에 해당하는 NVTX range의 (start_ns, end_ns) 목록 반환"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT start, end, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND end > 0
    """)

    stage_ranges = defaultdict(list)

    for t_start, t_end, text, text_id in cursor.fetchall():
        name = text if text else str_map.get(text_id, "")
        if not name:
            continue
        for stage, names in stage_map.items():
            if any(n.lower() in name.lower() for n in names):
                stage_ranges[stage].append((t_start, t_end))
                break

    print()
    for stage, ranges in stage_ranges.items():
        total_ms = sum(ns_to_ms(e - s) for s, e in ranges)
        print(f"  {stage}: {len(ranges)}개 range, 총 {total_ms:.1f} ms")
    for stage in stage_map:
        if stage not in stage_ranges:
            print(f"  [WARNING] {stage}: range를 찾지 못함")

    return dict(stage_ranges)


def query_kernels(conn: sqlite3.Connection, str_map: dict) -> list:
    """모든 CUDA 커널의 (start_ns, end_ns, name) 반환"""
    cursor = conn.cursor()
    cursor.execute("SELECT start, end, shortName, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")

    kernels = []
    for start, end, short_id, demangled_id in cursor.fetchall():
        # shortName, demangledName 모두 StringIds FK일 수 있음
        name = None
        for nid in [demangled_id, short_id]:
            if nid is None:
                continue
            if isinstance(nid, int):
                name = str_map.get(nid)
            else:
                name = str(nid)
            if name:
                # 너무 긴 템플릿 파라미터 축약
                if len(name) > 80:
                    name = name[:77] + "..."
                break
        kernels.append((start, end, name or "unknown"))

    print(f"  총 커널 수: {len(kernels):,}")
    return kernels


def analyze_stage(stage_name: str, stage_ranges: list,
                  all_kernels: list) -> dict:
    """단계 안 커널 간 gap 분석"""
    # 단계 전체 시간 범위
    stage_start = min(r[0] for r in stage_ranges)
    stage_end   = max(r[1] for r in stage_ranges)
    stage_ns    = stage_end - stage_start

    # 이 범위 안의 커널만 필터 (GPU 실행 기준)
    kernels_in = [(s, e, n) for s, e, n in all_kernels
                  if s >= stage_start and e <= stage_end + 1_000_000]  # 1ms tolerance

    if len(kernels_in) < 2:
        return {
            "stage": stage_name, "n_kernels": len(kernels_in),
            "stage_ms": ns_to_ms(stage_ns), "total_gap_ms": 0,
            "gap_ratio": 0, "top_gaps": [], "all_gaps": []
        }

    kernels_sorted = sorted(kernels_in, key=lambda x: x[0])

    gaps = []
    for i in range(1, len(kernels_sorted)):
        prev_end   = kernels_sorted[i-1][1]
        curr_start = kernels_sorted[i][0]
        gap_ns = curr_start - prev_end
        if gap_ns > 0:
            gaps.append({
                "gap_ms":    ns_to_ms(gap_ns),
                "prev_name": kernels_sorted[i-1][2],
                "curr_name": kernels_sorted[i][2],
            })

    total_gap_ms = sum(g["gap_ms"] for g in gaps)
    stage_ms     = ns_to_ms(stage_ns)
    gap_ratio    = total_gap_ms / stage_ms * 100 if stage_ms > 0 else 0

    return {
        "stage":        stage_name,
        "n_kernels":    len(kernels_sorted),
        "stage_ms":     stage_ms,
        "total_gap_ms": total_gap_ms,
        "gap_ratio":    gap_ratio,
        "top_gaps":     sorted(gaps, key=lambda x: x["gap_ms"], reverse=True)[:10],
        "all_gaps":     gaps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--list-nvtx", action="store_true",
                        help="sqlite 안의 모든 NVTX 이름 출력 후 종료")
    args = parser.parse_args()

    if not os.path.exists(args.sqlite):
        print(f"[ERROR] 파일 없음: {args.sqlite}")
        sys.exit(1)

    print("=" * 70)
    print(f"  nsys 커널 간 유휴 분석")
    print(f"  파일: {os.path.basename(args.sqlite)}")
    print("=" * 70)

    conn = sqlite3.connect(args.sqlite)

    print("\n[0] StringIds 로딩...")
    str_map = load_string_ids(conn)
    print(f"  {len(str_map):,}개 문자열 로드 완료")

    # --list-nvtx 옵션: 이름 목록만 보고 종료
    if args.list_nvtx:
        list_all_nvtx_ranges(conn, str_map)
        conn.close()
        return

    # 먼저 실제 NVTX 이름 목록 출력 (디버그)
    list_all_nvtx_ranges(conn, str_map)

    # ── 단계별 NVTX 이름 매핑 ──
    # 아래 이름들은 실제 sqlite에서 발견된 이름에 맞게 자동 탐색
    STAGE_MAP = {
        "VE":         ["Vision_Encoder", "vision_encoder", "VE"],
        "LM_Prefill": ["LM_Prefill", "Prefill", "prefill", "PrefillOnly"],
        "LM_Decode":  ["LM_Decode", "Decode", "decode", "DecodeOnly",
                       "autoregressive", "generate"],
        "Flow":       ["FlowODE", "FlowStep", "Flow", "flow",
                       "ODE", "diffusion"],
    }

    print("\n[1] NVTX 단계 범위 탐색...")
    stage_ranges = query_nvtx_ranges(conn, str_map, STAGE_MAP)

    print("\n[2] 커널 타임라인 로딩...")
    all_kernels = query_kernels(conn, str_map)

    print("\n[3] 단계별 gap 분석...")
    results = []
    for stage, ranges in stage_ranges.items():
        r = analyze_stage(stage, ranges, all_kernels)
        results.append(r)
        print(f"  {stage}: 완료")

    conn.close()

    # ── 출력 ──
    print()
    print("=" * 70)
    print("  결과: 단계별 커널 간 유휴 시간 (GPU 완전 유휴 구간)")
    print("=" * 70)

    print(f"\n{'단계':<16} {'커널 수':>8} {'단계 시간':>10} "
          f"{'유휴 합계':>10} {'유휴 비율':>9}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["stage_ms"]):
        print(f"{r['stage']:<16} {r['n_kernels']:>8,} "
              f"{r['stage_ms']:>9.1f}ms "
              f"{r['total_gap_ms']:>9.1f}ms "
              f"{r['gap_ratio']:>8.1f}%")

    for r in sorted(results, key=lambda x: x["stage_ms"]):
        print()
        print(f"  ── {r['stage']} ──")
        print(f"  커널 수: {r['n_kernels']:,}  |  "
              f"단계 시간: {r['stage_ms']:.1f} ms  |  "
              f"유휴 합계: {r['total_gap_ms']:.1f} ms ({r['gap_ratio']:.1f}%)")
        if not r["top_gaps"]:
            print("  (gap 없음)")
            continue
        print(f"  상위 gap (큰 순서):")
        for i, g in enumerate(r["top_gaps"], 1):
            if g["gap_ms"] < 0.01:
                break
            print(f"    {i:2}. {g['gap_ms']:7.3f} ms"
                  f"  |  {g['prev_name'][:50]:50s}"
                  f"  →  {g['curr_name'][:50]}")

    # 전체 요약
    print()
    print("=" * 70)
    print("  전체 요약")
    print("=" * 70)
    total_stage = sum(r["stage_ms"] for r in results)
    total_gap   = sum(r["total_gap_ms"] for r in results)
    print(f"\n  측정된 전체 시간: {total_stage:.1f} ms")
    print(f"  전체 유휴 합계:   {total_gap:.1f} ms ({total_gap/total_stage*100:.1f}%)\n")
    for r in sorted(results, key=lambda x: x["total_gap_ms"], reverse=True):
        bar = "█" * max(1, int(r["gap_ratio"] / 1.5))
        print(f"  {r['stage']:<16} {bar:<50} "
              f"{r['gap_ratio']:.1f}%  ({r['total_gap_ms']:.1f} ms)")

    # LM_Decode 미발견 시 안내
    if "LM_Decode" not in stage_ranges:
        print()
        print("  [!] LM_Decode를 찾지 못했습니다.")
        print("  위의 '[디버그] NVTX range 이름' 목록에서 Decode 관련 이름을 확인하고")
        print("  STAGE_MAP의 'LM_Decode' 항목에 추가해 주세요.")
        print("  또는 아래 명령으로 이름 목록만 확인할 수 있습니다:")
        print(f"    python3 {os.path.basename(__file__)} --sqlite {args.sqlite} --list-nvtx")


if __name__ == "__main__":
    main()
