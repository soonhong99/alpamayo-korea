#!/bin/bash
# =============================================================================
# 260611_diagnose_sm_metric.sh
# 목적: SM 11.0에서 실제로 지원되는 SM utilization 메트릭 찾기
#
# 배경:
#   sm__active_cycles.sum / gpc__cycles_elapsed.max 공식이 모든 커널에서
#   0.0%를 반환함. 원인: SM 11.0에서 sm__active_cycles.sum이 미지원(→0)이거나
#   공식의 집계 단위 불일치.
#
# 이 스크립트가 하는 것:
#   1. SM 11.0에서 지원되는 SM activity 메트릭 목록 출력
#   2. 소형 test 커널에 대해 후보 메트릭 값을 직접 확인
#   3. 올바른 메트릭 조합 결정
# =============================================================================

set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
INFERENCE_SCRIPT="$HOME/alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260611_sm_diag"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "  SM 메트릭 진단 - SM 11.0 (Blackwell Jetson AGX Thor)"
echo "================================================================"

# ------------------------------------------------------------------
# 단계 1: 지원되는 SM activity 관련 메트릭 목록 확인
# ------------------------------------------------------------------
echo ""
echo "[단계 1] SM 11.0에서 지원되는 SM activity 메트릭 목록..."

SM_METRIC_LIST="$RESULTS_DIR/sm_metrics_supported.txt"

sudo -E /usr/local/cuda/bin/ncu --query-metrics-mode detailed 2>/dev/null \
    | grep -iE "sm__active|smsp__active|sm__busy|sm__throughput|sm__inst|smsp__issue" \
    > "$SM_METRIC_LIST" || true

echo "  결과 저장: $SM_METRIC_LIST"
echo "  발견된 메트릭 수: $(wc -l < "$SM_METRIC_LIST" 2>/dev/null || echo 0)"
echo ""
head -40 "$SM_METRIC_LIST" 2>/dev/null || echo "  (결과 없음 - sm__active_cycles 미지원 확인됨)"

# ------------------------------------------------------------------
# 단계 2: lm_head GEMV 커널 하나에 대해 후보 메트릭 직접 측정
# (decode 1 step, lm_head 커널이 포함된 범위)
# ------------------------------------------------------------------
echo ""
echo "[단계 2] Decode 단계에서 후보 SM 메트릭 직접 측정..."

# 현재 공식의 원본 메트릭 + SM 후보 메트릭들
CANDIDATE_METRICS="sm__active_cycles.sum,\
gpc__cycles_elapsed.max,\
smsp__active_cycles.sum,\
smsp__active_cycles.avg.pct_of_peak_sustained_active,\
smsp__inst_issued.sum,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
sm__throughput.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum"

OUT_CSV="$RESULTS_DIR/sm_diag_decode.csv"
LOG_FILE="$RESULTS_DIR/sm_diag_decode.log"

echo "  측정 중 (Decode NVTX 범위, 1 step)..."

sudo -E /usr/local/cuda/bin/ncu \
    --nvtx \
    --nvtx-include "Decode" \
    --kernel-regex "gemv2T_kernel|nvjet" \
    --kernel-id :::1 \
    --replay-mode kernel \
    --set none \
    --metrics "$CANDIDATE_METRICS" \
    --csv \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
    > "$OUT_CSV" 2> "$LOG_FILE" || true

echo "  CSV 크기: $(du -sh "$OUT_CSV" 2>/dev/null | cut -f1 || echo '0')"
echo ""
echo "  첫 50행 (원본 sm__active_cycles.sum 값 확인):"
head -60 "$OUT_CSV" 2>/dev/null | grep -E "sm__|smsp__|gpu__time|Metric Name" | head -30 || echo "  (출력 없음)"

# ------------------------------------------------------------------
# 단계 3: 원본 값 확인용 간단 python 스크립트 실행
# ------------------------------------------------------------------
echo ""
echo "[단계 3] 원본 메트릭 값 확인 (sm_active vs gpc_elapsed 비교)..."

"$PYTHON" - "$OUT_CSV" << 'PYEOF'
import csv, sys

path = sys.argv[1]
try:
    lines = open(path).readlines()
except:
    print("  CSV 없음")
    sys.exit(0)

# 헤더 찾기
header_idx = None
for i, l in enumerate(lines):
    if '"ID"' in l or 'Metric Name' in l:
        header_idx = i
        break

if header_idx is None:
    print("  헤더 없음 - ncu 오류 발생")
    for l in lines[:5]: print(" ", l.rstrip())
    sys.exit(0)

reader = csv.DictReader(lines[header_idx:])
fields = {f.lower(): f for f in (reader.fieldnames or [])}
kn_col = fields.get("kernel name") or fields.get("name")
mn_col = fields.get("metric name")
mv_col = fields.get("metric value")

if not (kn_col and mn_col and mv_col):
    print(f"  컬럼 찾기 실패. 컬럼 목록: {list(fields.keys())[:8]}")
    sys.exit(0)

# 첫 번째 커널의 모든 메트릭 값 출력
first_kernel = None
metrics = {}
for row in reader:
    kn = row.get(kn_col, "").strip('"')
    mn = row.get(mn_col, "").strip()
    mv = row.get(mv_col, "").strip().replace(",", "")
    if not kn or not mn:
        continue
    if first_kernel is None:
        first_kernel = kn
    if kn == first_kernel:
        try:
            metrics[mn] = float(mv)
        except:
            metrics[mn] = mv

print(f"\n  커널: {first_kernel[:60]}")
print(f"  {'메트릭':<55} {'값':>20}")
print(f"  {'-'*77}")
for m, v in sorted(metrics.items()):
    print(f"  {m:<55} {str(v):>20}")

sm_act = metrics.get("sm__active_cycles.sum", None)
gpc_el = metrics.get("gpc__cycles_elapsed.max", None)
smsp_act = metrics.get("smsp__active_cycles.avg.pct_of_peak_sustained_active", None)
gpu_time = metrics.get("gpu__time_duration.sum", None)

print(f"\n  ★ 진단:")
if sm_act is not None:
    print(f"    sm__active_cycles.sum     = {sm_act:>15,.0f}")
if gpc_el is not None:
    print(f"    gpc__cycles_elapsed.max   = {gpc_el:>15,.0f}")
if sm_act is not None and gpc_el is not None and gpc_el > 0:
    ratio = sm_act / gpc_el * 100
    print(f"    기존 공식 결과             = {ratio:>14.4f}%")
    if sm_act == 0:
        print(f"    → sm__active_cycles.sum = 0 → 메트릭 미지원 (SM 11.0 호환 문제)")
    elif ratio > 1000:
        print(f"    → {ratio:.0f}% = 집계 단위 불일치 (N_SMSP × util_fraction)")
    else:
        print(f"    → 값 있음, 단 0% 표시는 반올림 이슈")
if smsp_act is not None:
    print(f"    smsp__active_cycles.avg.pct_of_peak = {smsp_act:.2f}%  ← 올바른 SM util")
if gpu_time is not None:
    print(f"    gpu__time_duration.sum (ns) = {gpu_time:,.0f}  ({gpu_time/1e6:.2f} ms)")
PYEOF

sudo chown -R "$(logname)":"$(logname)" "$RESULTS_DIR" 2>/dev/null || true

echo ""
echo "================================================================"
echo "  요약:"
echo "  1. sm_metrics_supported.txt - SM 11.0 지원 메트릭 목록"
echo "  2. sm_diag_decode.csv       - 실측 값"
echo "  3. 위 Python 출력에서 sm__active_cycles.sum = 0이면 미지원 확정"
echo "  4. smsp__active_cycles.avg.pct_of_peak_sustained_active 값이 있으면"
echo "     그것이 올바른 SM util"
echo "================================================================"
