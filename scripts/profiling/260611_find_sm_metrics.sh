#!/bin/bash
# =============================================================================
# 260611_find_sm_metrics.sh
# SM 11.0에서 실제로 지원되는 SM activity 메트릭 탐색
# =============================================================================

set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260611_sm_diag"
mkdir -p "$RESULTS_DIR"
NCU="/usr/local/cuda/bin/ncu"

echo "================================================================"
echo "  SM 11.0 지원 메트릭 탐색"
echo "================================================================"

# ------------------------------------------------------------------
# 1. 전체 메트릭 목록 덤프 (SM 관련 키워드로 필터)
# ------------------------------------------------------------------
echo ""
echo "[1] 전체 지원 메트릭 덤프 → SM/GPC/SMSP 관련 grep..."
ALL_METRICS="$RESULTS_DIR/all_metrics.txt"

$NCU --query-metrics-mode detailed 2>/dev/null > "$ALL_METRICS" || true
TOTAL=$(wc -l < "$ALL_METRICS" 2>/dev/null || echo 0)
echo "  전체 메트릭 수: $TOTAL 줄"

echo ""
echo "  ── SM/SMSP/GPC 관련 메트릭 ──"
grep -iE "^sm__|^smsp__|^gpc__|^gr__" "$ALL_METRICS" 2>/dev/null | head -60 \
    || echo "  (없음)"

echo ""
echo "  ── 'active' 포함 메트릭 ──"
grep -i "active" "$ALL_METRICS" 2>/dev/null | head -40 \
    || echo "  (없음)"

echo ""
echo "  ── 'util' 또는 'throughput' 포함 메트릭 ──"
grep -iE "util|throughput" "$ALL_METRICS" 2>/dev/null | head -40 \
    || echo "  (없음)"

echo ""
echo "  ── 'cycle' 포함 메트릭 ──"
grep -i "cycle" "$ALL_METRICS" 2>/dev/null | head -40 \
    || echo "  (없음)"

# ------------------------------------------------------------------
# 2. --query-metrics 방식도 시도 (버전에 따라 옵션명 다름)
# ------------------------------------------------------------------
echo ""
echo "[2] 대안 쿼리 방식 시도..."
$NCU --help 2>&1 | grep -iE "query|metric|list" | head -20 || true

echo ""
echo "  ncu 버전:"
$NCU --version 2>/dev/null | head -5 || true

# ------------------------------------------------------------------
# 3. 실제 커널 1개에 후보 메트릭 측정 시도
#    SM 11.0에서 작동하는 것만 값이 나옴
# ------------------------------------------------------------------
echo ""
echo "[3] 실제 커널 1개에 후보 메트릭 측정 (값 있는 것 = 지원 확인)..."

INFERENCE_SCRIPT="$HOME/alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"

# lts 계열(DRAM)은 이미 작동 확인됨. SM 후보 메트릭들 추가 측정
CANDIDATE_METRICS="\
smsp__active_cycles.sum,\
smsp__active_cycles.avg,\
sm__active_cycles.sum,\
sm__cycles_active.avg,\
gr__active_cycles.avg,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__thread_inst_executed.sum,\
sm__throughput.avg.pct_of_peak_sustained_active,\
smsp__active_cycles.avg.pct_of_peak_sustained_active,\
lts__d_sectors_fill_sysmem.sum"

OUT="$RESULTS_DIR/sm_candidate_test.csv"
LOG="$RESULTS_DIR/sm_candidate_test.log"

echo "  측정 중 (Decode 1 step, --kernel-id :::1 사용)..."
sudo -E $NCU \
    --nvtx \
    --nvtx-include "DecodeAll" \
    --kernel-id :::1 \
    --replay-mode kernel \
    --set none \
    --metrics "$CANDIDATE_METRICS" \
    --csv \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
    > "$OUT" 2> "$LOG" || true

echo "  CSV 크기: $(du -sh "$OUT" 2>/dev/null | cut -f1 || echo '?')"

"$PYTHON" - "$OUT" << 'PYEOF'
import csv, sys

path = sys.argv[1]
try:
    lines = open(path).readlines()
except:
    print("  CSV 없음")
    sys.exit(0)

header_idx = None
for i, l in enumerate(lines):
    if '"ID"' in l or '"Metric Name"' in l or 'Metric Name' in l:
        header_idx = i
        break

if header_idx is None:
    print("  ncu 오류 (헤더 없음). 로그:")
    for l in lines[:8]: print("   ", l.rstrip())
    sys.exit(0)

reader = csv.DictReader(lines[header_idx:])
fields = {f.lower().strip('"'): f for f in (reader.fieldnames or [])}
kn_col = fields.get("kernel name") or fields.get('"kernel name"')
mn_col = fields.get("metric name") or fields.get('"metric name"')
mv_col = fields.get("metric value") or fields.get('"metric value"')

if not (kn_col and mn_col and mv_col):
    print(f"  컬럼 인식 실패. 필드: {list(fields.keys())[:10]}")
    sys.exit(0)

first_kernel = None
metrics = {}
for row in reader:
    kn = row.get(kn_col, "").strip().strip('"')
    mn = row.get(mn_col, "").strip().strip('"')
    mv = row.get(mv_col, "").strip().strip('"').replace(",", "")
    if not kn or not mn: continue
    if first_kernel is None:
        first_kernel = kn
    if kn == first_kernel:
        try:
            metrics[mn] = float(mv)
        except:
            if mv and mv not in ("-", "N/A", ""):
                metrics[mn] = mv

print(f"\n  커널: {(first_kernel or '')[:70]}")
print(f"\n  {'메트릭':<60} {'값':>15}  {'지원여부'}")
print(f"  {'-'*90}")

SM_CANDIDATES = [
    "smsp__active_cycles.sum",
    "smsp__active_cycles.avg",
    "sm__active_cycles.sum",
    "sm__cycles_active.avg",
    "gr__active_cycles.avg",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__thread_inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_active",
    "smsp__active_cycles.avg.pct_of_peak_sustained_active",
    "lts__d_sectors_fill_sysmem.sum",
]
for m in SM_CANDIDATES:
    v = metrics.get(m, "NOT_IN_CSV")
    supported = "✅" if v != "NOT_IN_CSV" and v != 0.0 else ("⚠ 0" if v == 0.0 else "❌")
    print(f"  {m:<60} {str(v):>15}  {supported}")

print(f"\n  ★ 올바른 SM util 메트릭 (값>0인 것):")
for m, v in metrics.items():
    if isinstance(v, float) and v > 0 and "sm" in m.lower():
        print(f"    {m} = {v:.4f}")
PYEOF

sudo chown -R "$(logname)":"$(logname)" "$RESULTS_DIR" 2>/dev/null || true

echo ""
echo "================================================================"
echo "  결과 파일: $RESULTS_DIR/"
echo "  - all_metrics.txt       → SM 11.0 지원 메트릭 전체 목록"
echo "  - sm_candidate_test.csv → 후보 메트릭 실측값"
echo ""
echo "  여기서 지원 확인된 SM 메트릭으로 ncu 스크립트 업데이트 예정"
echo "================================================================"
