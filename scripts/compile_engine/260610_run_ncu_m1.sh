#!/bin/bash
# =============================================================================
# 260610_run_ncu_m1.sh — M1 step 1: P5 fusion DRAM byte saving (ncu 실측)
#
# eager MLP (5 kernels: gate mm, up mm, silu, mul, down mm) vs
# fused  MLP (2 kernels: gate_silu_mul triton, down mm)
# 동일 NVTX 범위(Phase/MLP)에서 DRAM read/write byte 비교.
#
# 실행: sudo -E bash 260610_run_ncu_m1.sh
# =============================================================================
set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
SCRIPT="$HOME/alpamayo1.5/scripts/compile_engine/260610_m1_mlp_ncu_test.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260610_m1_ffn"
mkdir -p "$RESULTS_DIR"

METRICS="lts__d_sectors_fill_sysmem.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum,\
lts__t_request_hit_rate.pct,\
gpu__time_duration.sum"

for MODE in ncu_eager ncu_fused; do
  OUT="$RESULTS_DIR/${MODE}.csv"
  echo "[$(date '+%H:%M:%S')] ncu $MODE ..."
  sudo -E env PYTHONPATH="$HOME/alpamayo1.5/src" /usr/local/cuda/bin/ncu \
      --nvtx --nvtx-include "Phase/MLP" \
      --replay-mode kernel \
      --set none --metrics "$METRICS" --csv \
      "$PYTHON" "$SCRIPT" "$MODE" \
      > "$OUT" 2> "$RESULTS_DIR/${MODE}.log"
done

sudo chown -R "$(logname)":"$(logname)" "$RESULTS_DIR" 2>/dev/null || true

echo ""
echo "=== DRAM byte 비교 (Phase/MLP, 1 forward) ==="
PYTHONPATH="$HOME/alpamayo1.5/src" "$PYTHON" "$SCRIPT" summarize \
    "$RESULTS_DIR/ncu_eager.csv" "$RESULTS_DIR/ncu_fused.csv"
