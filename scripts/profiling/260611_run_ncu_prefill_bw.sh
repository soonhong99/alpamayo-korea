#!/bin/bash
# =============================================================================
# 260611_run_ncu_prefill_bw.sh
# 목적: LM Prefill 단계 커널별 순간 DRAM 대역폭 + SM 활용률 측정
#
# Decode 측정(260610_run_ncu_per_kernel_bw.sh)과 동일한 방법,
# NVTX 필터만 Phase/PrefillOnly로 변경.
#
# 핵심 질문:
#   - SM util > 0%? → compute-bound (FlashAttention GEMM 지배)
#   - SM util = 0%? → memory-bound (가중치 전송 병목)
#
# 배경:
#   Decode(seq=1)에서 가중치 투영은 GEMV(intensity≈1 op/byte).
#   Prefill(seq=3086)에서는 GEMM으로 전환: intensity ≈ 1761 op/byte.
#   Ridge point(BF16) ≈ 수천 op/byte → weight GEMM도 여전히 DRAM-bound 가능성.
#   Attention(FlashAttention)은 O(n²) FLOPs, O(n) DRAM → compute-bound 예상.
#
# Prefill은 Decode(44,000 커널)보다 커널 수가 훨씬 적음:
#   36층 × 1회 패스 ≈ 수천 개 커널 → 예상 소요 5~15분
#
# 실행: sudo -E bash 260611_run_ncu_prefill_bw.sh
#   ※ sudo -E 필수: $HOME 환경변수 보존
#
# 분석: python3 260611_analyze_prefill_bw.py (자동 실행됨)
# =============================================================================

set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
INFERENCE_SCRIPT="$HOME/alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260611_prefill_bw"

mkdir -p "$RESULTS_DIR"

# ★ SM 메트릭 수정 (2026-06-11 --list-metrics 실측):
#   sm__active_cycles.sum / gpc__cycles_elapsed.max → GB10B 미존재 → 항상 0
#   smsp__cycles_active.sum / smsp__cycles_elapsed.sum → GB10B 유일 지원 공식
METRICS="lts__d_sectors_fill_sysmem.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum,\
lts__t_request_hit_rate.pct,\
gpu__time_duration.sum,\
smsp__cycles_active.sum,\
smsp__cycles_elapsed.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

# ------------------------------------------------------------------
# Prefill 단계 측정
# ------------------------------------------------------------------
echo "================================================================"
echo "  Prefill 단계 커널별 instantaneous BW + SM util 측정"
echo "  NVTX 필터: Phase/LM_Prefill"
echo "  핵심 질문: SM util > 0? (compute-bound) vs 0? (memory-bound)"
echo "  예상 소요: 5~15분"
echo "================================================================"
echo "[$(date '+%H:%M:%S')] 시작..."

OUT_CSV="$RESULTS_DIR/prefill_per_kernel.csv"
LOG_FILE="$RESULTS_DIR/prefill_per_kernel.log"

sudo -E /usr/local/cuda/bin/ncu \
    --nvtx \
    --nvtx-include "Phase/LM_Prefill" \
    --replay-mode kernel \
    --set none \
    --metrics "$METRICS" \
    --csv \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
    > "$OUT_CSV" 2> "$LOG_FILE"

echo "[$(date '+%H:%M:%S')] ncu 완료"
echo "  CSV 크기: $(du -sh "$OUT_CSV" | cut -f1)"
KERNEL_COUNT=$(grep -c 'smsp__cycles_active\|lts__d_sectors' "$OUT_CSV" 2>/dev/null || echo 0)
echo "  커널 행 수: $KERNEL_COUNT"

# ------------------------------------------------------------------
# nsys 수집 (기존 SQLite 재사용 가능하면 스킵 가능)
# decode_timeline.sqlite에 Prefill 타이밍이 이미 포함됨.
# 새로 nsys를 실행하고 싶다면 아래 주석 해제.
# ------------------------------------------------------------------
# NSYS_OUT="$RESULTS_DIR/prefill_timeline"
# echo ""
# echo "[$(date '+%H:%M:%S')] nsys 수집..."
# sudo -E /usr/local/cuda/bin/nsys profile \
#     --trace=cuda,nvtx \
#     --cuda-graph-trace=node \
#     --sample=none \
#     --output "$NSYS_OUT" \
#     --force-overwrite true \
#     "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run
# /usr/local/cuda/bin/nsys export \
#     --type sqlite \
#     --output "${NSYS_OUT}.sqlite" \
#     --force-overwrite true \
#     "${NSYS_OUT}.nsys-rep"

echo ""
echo "================================================================"
echo "  파일 권한 수정 중..."
sudo chown -R "$(logname)":"$(logname)" "$RESULTS_DIR" 2>/dev/null || true
echo ""
echo "  분석 실행:"
echo "  python3 ~/alpamayo1.5/scripts/profiling/260611_analyze_prefill_bw.py \\"
echo "    --ncu   $OUT_CSV \\"
echo "    --nsys  $HOME/alpamayo1.5/profiling_results/260610_per_kernel_bw/decode_timeline.sqlite"
echo ""
echo "  ※ nsys는 기존 decode_timeline.sqlite 재사용 (Prefill 타이밍 포함)"
echo "  ※ 새 nsys 필요시 스크립트 내 주석 해제"
echo "================================================================"
