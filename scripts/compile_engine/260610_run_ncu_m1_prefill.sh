#!/bin/bash
# =============================================================================
# 260610_run_ncu_m1_prefill.sh — P5 fused Prefill 전체 DRAM 측정 (ncu)
#
# Baseline (eager, 260609 확정): Prefill read 179.899 GB + write 52.067 GB
#                               = 231.966 GB
# 이 스크립트는 fused 1회만 측정해 위 기준선과 비교한다.
#
# 실행: bash 260610_run_ncu_m1_prefill.sh   (내부에서 sudo 사용)
# =============================================================================
set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
SCRIPT="$HOME/alpamayo1.5/scripts/compile_engine/260610_m1_prefill_e2e.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260610_m1_prefill_e2e"
mkdir -p "$RESULTS_DIR"

METRICS="lts__d_sectors_fill_sysmem.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum,\
lts__t_request_hit_rate.pct,\
gpu__time_duration.sum"

OUT="$RESULTS_DIR/prefill_fused.csv"
echo "[$(date '+%H:%M:%S')] ncu fused prefill 측정 시작 (모델 로딩 포함, 20~40분 예상)..."

sudo -E env PYTHONPATH="$HOME/alpamayo1.5/src" /usr/local/cuda/bin/ncu \
    --nvtx --nvtx-include "Phase/LM_Prefill" \
    --replay-mode kernel \
    --set none --metrics "$METRICS" --csv \
    "$PYTHON" "$SCRIPT" --fuse 1 --mode ncu_single_run \
    > "$OUT" 2> "$RESULTS_DIR/prefill_fused.log"

sudo chown -R "$(logname)":"$(logname)" "$RESULTS_DIR" 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] 완료"
echo ""
echo "=== Prefill DRAM (fused) vs baseline 231.966 GB (eager, 260609) ==="
PYTHONPATH="$HOME/alpamayo1.5/src" "$PYTHON" \
    "$HOME/alpamayo1.5/scripts/compile_engine/260610_m1_mlp_ncu_test.py" \
    summarize "$OUT"
