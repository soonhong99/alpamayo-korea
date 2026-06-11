#!/bin/bash
# =============================================================================
# 260611_find_sm_metrics_v2.sh
# ncu 2025.3.0.0 / SM 11.0 (Blackwell) 환경에서 SM 메트릭 조사
#
# 이전 스크립트 버그 수정:
#   1. --query-metrics-mode detailed → full/base로 변경
#   2. --kernel-id :::1 제거 (NVTX 필터와 충돌)
#   3. 올바른 DecodeAll 필터 사용
# =============================================================================

set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
INFERENCE_SCRIPT="$HOME/alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260611_sm_diag"
mkdir -p "$RESULTS_DIR"
NCU="/usr/local/cuda/bin/ncu"

echo "================================================================"
echo "  SM 11.0 / ncu 2025.3.0.0 지원 메트릭 조사 (수정판)"
echo "================================================================"

# ------------------------------------------------------------------
# 1. 올바른 메트릭 쿼리 (base / full 모드)
# ------------------------------------------------------------------
echo ""
echo "[1] 메트릭 쿼리 (--query-metrics-mode full)..."
FULL_METRICS="$RESULTS_DIR/metrics_full.txt"
$NCU --query-metrics-mode full 2>/dev/null > "$FULL_METRICS" || true
TOTAL=$(wc -l < "$FULL_METRICS" 2>/dev/null || echo 0)
echo "  전체 메트릭 수: $TOTAL 줄"
echo "  처음 20줄:"
head -20 "$FULL_METRICS" 2>/dev/null || echo "  (비어있음)"

echo ""
echo "[1b] 메트릭 쿼리 (--query-metrics-mode base)..."
BASE_METRICS="$RESULTS_DIR/metrics_base.txt"
$NCU --query-metrics-mode base 2>/dev/null > "$BASE_METRICS" || true
BASE_TOTAL=$(wc -l < "$BASE_METRICS" 2>/dev/null || echo 0)
echo "  base 모드 메트릭 수: $BASE_TOTAL 줄"
head -20 "$BASE_METRICS" 2>/dev/null || echo "  (비어있음)"

echo ""
echo "[1c] --query-metrics 단독 시도..."
$NCU --query-metrics 2>/dev/null | head -30 || true

# ------------------------------------------------------------------
# 2. ncu --list-sections 로 사용 가능한 섹션 확인
#    섹션 기반 접근이 아키텍처 독립적으로 SM util을 제공
# ------------------------------------------------------------------
echo ""
echo "[2] 사용 가능한 섹션 목록 (--list-sections)..."
SECTIONS="$RESULTS_DIR/sections.txt"
$NCU --list-sections 2>/dev/null > "$SECTIONS" || true
echo "  섹션 수: $(wc -l < "$SECTIONS" 2>/dev/null || echo 0)"
echo ""
echo "  SM util 관련 섹션:"
grep -iE "SpeedOfLight|WarpState|Compute|SM|Throughput|Occupancy" "$SECTIONS" 2>/dev/null \
    || echo "  (없음)"
echo ""
echo "  전체 섹션:"
cat "$SECTIONS" 2>/dev/null || echo "  (비어있음)"

# ------------------------------------------------------------------
# 3. SpeedOfLight 섹션으로 SM util 측정 시도
#    ncu 내장 섹션은 아키텍처 차이를 추상화함
# ------------------------------------------------------------------
echo ""
echo "[3] SpeedOfLight 섹션 기반 SM util 측정..."
SOL_CSV="$RESULTS_DIR/speedoflight_decode.csv"
SOL_LOG="$RESULTS_DIR/speedoflight_decode.log"

sudo -E $NCU \
    --nvtx \
    --nvtx-include "DecodeAll" \
    --replay-mode kernel \
    --section SpeedOfLight \
    --csv \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
    > "$SOL_CSV" 2> "$SOL_LOG" || true

KERNEL_COUNT=$(grep -c '"' "$SOL_CSV" 2>/dev/null || echo 0)
echo "  CSV 크기: $(du -sh "$SOL_CSV" 2>/dev/null | cut -f1 || echo '?')"
echo "  커널 행 수 (따옴표 기준): $KERNEL_COUNT"
echo ""
echo "  SpeedOfLight 결과 (처음 60줄):"
head -60 "$SOL_CSV" 2>/dev/null || echo "  (없음)"

# ------------------------------------------------------------------
# 4. 특정 SM 메트릭 후보를 직접 --metrics로 측정
#    ncu 2025.3 Blackwell에서 실제로 작동하는 것 확인
# ------------------------------------------------------------------
echo ""
echo "[4] SM 메트릭 후보 직접 수집 (Decode 1 kernel)..."

# ncu 2025.3에서 Blackwell용으로 변경된 가능성 있는 이름들
BLACKWELL_SM_CANDIDATES="\
sm__active_cycles.sum,\
smsp__active_cycles.sum,\
sm__cycles_active.sum,\
smsp__cycles_active.sum,\
sm__active_cycles.avg,\
smsp__active_cycles.avg,\
sm__throughput.avg.pct_of_peak_sustained_active,\
smsp__inst_executed.sum,\
sm__inst_executed.sum,\
gpc__cycles_elapsed.max,\
gpu__time_duration.sum,\
lts__d_sectors_fill_sysmem.sum"

CANDIDATE_CSV="$RESULTS_DIR/sm_candidate_v2.csv"
CANDIDATE_LOG="$RESULTS_DIR/sm_candidate_v2.log"

echo "  측정 중..."
sudo -E $NCU \
    --nvtx \
    --nvtx-include "DecodeAll" \
    --replay-mode kernel \
    --set none \
    --metrics "$BLACKWELL_SM_CANDIDATES" \
    --csv \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
    > "$CANDIDATE_CSV" 2> "$CANDIDATE_LOG" || true

echo "  CSV 크기: $(du -sh "$CANDIDATE_CSV" 2>/dev/null | cut -f1 || echo '?')"

"$PYTHON" - "$CANDIDATE_CSV" "$CANDIDATE_LOG" << 'PYEOF'
import csv, sys, re

csv_path = sys.argv[1]
log_path = sys.argv[2]

# ncu 에러 메시지 확인
try:
    log_lines = open(log_path).readlines()
    invalid = [l for l in log_lines if 'invalid' in l.lower() or 'not found' in l.lower() or 'error' in l.lower()]
    if invalid:
        print("  ncu 에러/경고:")
        for l in invalid[:10]: print("   ", l.rstrip())
except: pass

try:
    lines = open(csv_path).readlines()
except:
    print("  CSV 없음")
    sys.exit(0)

header_idx = None
for i, l in enumerate(lines):
    if '"ID"' in l or '"Metric Name"' in l:
        header_idx = i
        break

if header_idx is None:
    print("  CSV 헤더 없음:")
    for l in lines[:6]: print("   ", l.rstrip())
    sys.exit(0)

reader = csv.DictReader(lines[header_idx:])
fields = {f.strip().strip('"').lower(): f for f in (reader.fieldnames or [])}

kn_col = next((f for k, f in fields.items() if 'kernel name' in k), None)
mn_col = next((f for k, f in fields.items() if 'metric name' in k), None)
mv_col = next((f for k, f in fields.items() if 'metric value' in k), None)

if not all([kn_col, mn_col, mv_col]):
    print(f"  컬럼 인식 실패: {list(fields.keys())[:8]}")
    sys.exit(0)

# 첫 번째 커널의 메트릭 수집
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
        try: metrics[mn] = float(mv)
        except: metrics[mn] = mv if mv else "N/A"

print(f"\n  ★ 커널: {(first_kernel or 'NONE')[:70]}")
print(f"\n  {'메트릭':<58} {'값':>15}  상태")
print(f"  {'-'*85}")

CANDIDATES = [
    "sm__active_cycles.sum",
    "smsp__active_cycles.sum",
    "sm__cycles_active.sum",
    "smsp__cycles_active.sum",
    "sm__active_cycles.avg",
    "smsp__active_cycles.avg",
    "sm__throughput.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "sm__inst_executed.sum",
    "gpc__cycles_elapsed.max",
    "gpu__time_duration.sum",
    "lts__d_sectors_fill_sysmem.sum",
]
for m in CANDIDATES:
    v = metrics.get(m, "NOT_IN_CSV")
    if v == "NOT_IN_CSV":
        status = "❌ 미지원/미수집"
    elif isinstance(v, float) and v == 0.0:
        status = "⚠  0 (수집됐으나 0)"
    elif isinstance(v, float) and v > 0:
        status = "✅ 지원됨"
    else:
        status = f"?? {v}"
    print(f"  {m:<58} {str(v):>15}  {status}")

print(f"\n  ★ SM util 측정 가능 후보 (값>0):")
found = False
for m, v in metrics.items():
    if isinstance(v, float) and v > 0:
        if any(kw in m.lower() for kw in ['sm', 'smsp', 'active', 'throughput', 'inst']):
            print(f"    ✅ {m} = {v:.2f}")
            found = True
if not found:
    print("    없음 → SM 11.0에서 SM activity 메트릭 미지원 가능성 높음")
PYEOF

# ------------------------------------------------------------------
# 5. ncu --set full 로 수집 가능한 모든 메트릭 목록 확인
# ------------------------------------------------------------------
echo ""
echo "[5] ncu --list-metrics 로 현재 GPU 지원 메트릭 목록..."
LIST_METRICS="$RESULTS_DIR/list_metrics.txt"
sudo -E $NCU \
    --list-metrics \
    --nvtx \
    --nvtx-include "DecodeAll" \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
    > "$LIST_METRICS" 2>/dev/null || true

LCOUNT=$(wc -l < "$LIST_METRICS" 2>/dev/null || echo 0)
echo "  --list-metrics 결과: $LCOUNT 줄"
echo ""
echo "  SM/SMSP/GPC/active/cycle 관련:"
grep -iE "sm|smsp|gpc|active|cycle|throughput|util" "$LIST_METRICS" 2>/dev/null | head -50 \
    || echo "  (없음)"

sudo chown -R "$(logname)":"$(logname)" "$RESULTS_DIR" 2>/dev/null || true

echo ""
echo "================================================================"
echo "  결과 파일:"
echo "  - metrics_full.txt     : --query-metrics-mode full 결과"
echo "  - metrics_base.txt     : --query-metrics-mode base 결과"
echo "  - sections.txt         : 사용 가능한 ncu 섹션 목록"
echo "  - speedoflight_decode  : SpeedOfLight 섹션 결과"
echo "  - sm_candidate_v2.csv  : SM 후보 메트릭 직접 측정"
echo "  - list_metrics.txt     : --list-metrics 결과"
echo "================================================================"
