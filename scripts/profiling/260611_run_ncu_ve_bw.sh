#!/bin/bash
# =============================================================================
# 260611_run_ncu_ve_bw.sh
# 목적: Vision Encoder(VE) 단계 커널별 순간 DRAM 대역폭 + SM 활용률 측정
#
# 핵심 질문:
#   - SM util > 0%? → compute-bound (ViT, Conv2D 연산 지배)
#   - SM util = 0%? → memory-bound (가중치 전송 병목)
#
# 배경:
#   VE stage BW = 80 GB/s (35%) ← 4단계 중 가장 낮음
#   낮은 stage BW의 두 가지 해석:
#     (A) compute-bound: SM이 바빠서 DRAM을 못 긁음 → BW 낮음
#     (B) 비효율 접근: DRAM 접근 패턴이 나쁨 → 실효 BW 낮음
#   SM util로 (A) vs (B)를 판별함.
#
#   Vision Encoder는 ViT(Vision Transformer) 기반.
#   seq_len << 모델 hidden_dim이 아닌 경우, attention이 compute-bound 가능성.
#
# NVTX 구조 (nsys 확인됨):
#   Phase / Vision_Encoder  ← 이 범위 측정
#
# 실행: sudo -E bash 260611_run_ncu_ve_bw.sh
# 분석:
#   python3 260611_analyze_prefill_bw.py \
#     --ncu  ~/alpamayo1.5/profiling_results/260611_ve_bw/ve_per_kernel.csv \
#     --nsys ~/alpamayo1.5/profiling_results/260610_per_kernel_bw/decode_timeline.sqlite \
#     --prefill-nvtx Vision_Encoder
# =============================================================================

set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
INFERENCE_SCRIPT="$HOME/alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260611_ve_bw"

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

echo "================================================================"
echo "  Vision Encoder 커널별 instantaneous BW + SM util 측정"
echo "  NVTX 필터: Phase/Vision_Encoder"
echo "  핵심 질문: SM util > 0? (compute-bound) vs 0? (memory-bound)"
echo "  Stage BW 35%(80 GB/s)가 낮은 이유 판별"
echo "  예상 소요: 5~20분"
echo "================================================================"
echo "[$(date '+%H:%M:%S')] 시작..."

OUT_CSV="$RESULTS_DIR/ve_per_kernel.csv"
LOG_FILE="$RESULTS_DIR/ve_per_kernel.log"

sudo -E /usr/local/cuda/bin/ncu \
    --nvtx \
    --nvtx-include "Phase/Vision_Encoder" \
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

sudo chown -R "$(logname)":"$(logname)" "$RESULTS_DIR" 2>/dev/null || true

echo ""
echo "================================================================"
echo "  분석 실행:"
echo "  python3 ~/alpamayo1.5/scripts/profiling/260611_analyze_prefill_bw.py \\"
echo "    --ncu  $OUT_CSV \\"
echo "    --nsys $HOME/alpamayo1.5/profiling_results/260610_per_kernel_bw/decode_timeline.sqlite \\"
echo "    --prefill-nvtx Vision_Encoder"
echo ""
echo "  SM util > 0% → compute-bound  → 최적화: FP4, 알고리즘 개선"
echo "  SM util = 0% → memory-bound  → 최적화: 가중치 압축, prefetch"
echo "================================================================"
