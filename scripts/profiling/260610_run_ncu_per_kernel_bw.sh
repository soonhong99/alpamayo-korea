#!/bin/bash
# =============================================================================
# 260610_run_ncu_per_kernel_bw.sh
# 목적: 커널별 순간(instantaneous) DRAM 대역폭 측정
#
# 260609와의 차이:
#   - gpu__time_duration.sum 추가    → 각 커널의 실제 실행 시간(ns)
#   - sm__active_cycles.sum 추가     → 커널 내 SM이 실제로 연산한 사이클 수
#   - gpc__cycles_elapsed.max 추가   → 커널 실행 중 경과한 전체 사이클 수
#   → 이 세 값으로 커널별 instantaneous BW 및 SM 활용률 계산 가능
#
# 질문: DRAM에서 데이터를 가져오는 시간만 따졌을 때 실제 전송 속도는?
#       GPU SM은 전송 중에 계산하고 있는가, 놀고 있는가?
#
# 대상: Decode 전체 (Phase/DecodeAll, 19 steps, ~44,000 커널)
#       → 가장 DRAM-bound한 단계, 실측이 가장 의미 있음
#       Flow는 선택적으로 추가 가능 (FLOW_MEASURE=1 환경변수)
#
# 실행: sudo -E bash 260610_run_ncu_per_kernel_bw.sh
#   ※ sudo -E 필수: $HOME 등 환경변수 보존 + ncu 전체 경로 사용
#   ※ sudo bash (without -E)로 실행하면 $HOME=/root → 경로 오류로 즉시 종료
#
# 예상 소요 시간: Decode 약 60~90분 (44,000 커널 × replay)
# =============================================================================

set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
INFERENCE_SCRIPT="$HOME/alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260610_per_kernel_bw"

mkdir -p "$RESULTS_DIR"

# SM 11.0 (Thor) 지원 metric 목록
# lts__d_sectors_fill_sysmem.sum        : DRAM read (sector × 32 = bytes)
# lts__t_sectors_aperture_sysmem_op_write.sum : DRAM write
# lts__t_request_hit_rate.pct           : L2 hit rate (per-kernel arithmetic mean)
# gpu__time_duration.sum                           : 커널 실행 시간 (ns)
# smsp__cycles_active.sum                          : SM 서브파티션 active cycles (GB10B 확인됨 2026-06-11)
# smsp__cycles_elapsed.sum                         : SM 서브파티션 elapsed cycles (분모, 단위 일치)
# sm__throughput.avg.pct_of_peak_sustained_elapsed : SM throughput % (교차검증용 직접 %)
#
# ★ 변경 이유 (2026-06-11 --list-metrics 실측 확인):
#   sm__active_cycles.sum / gpc__cycles_elapsed.max → GB10B(SM 11.0)에 존재하지 않음 → 항상 0
#   smsp__cycles_active.sum / smsp__cycles_elapsed.sum → GB10B 유일하게 지원되는 SM 활성도 공식
METRICS="lts__d_sectors_fill_sysmem.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum,\
lts__t_request_hit_rate.pct,\
gpu__time_duration.sum,\
smsp__cycles_active.sum,\
smsp__cycles_elapsed.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

# ------------------------------------------------------------------
# Decode 단계 측정
# ------------------------------------------------------------------
echo "================================================================"
echo "  Decode 단계 커널별 instantaneous BW 측정"
echo "  NVTX 필터: Phase/DecodeAll (19 steps 전체)"
echo "  예상 소요: 60~90분"
echo "================================================================"
echo "[$(date '+%H:%M:%S')] 시작..."

OUT_CSV="$RESULTS_DIR/decode_per_kernel.csv"
LOG_FILE="$RESULTS_DIR/decode_per_kernel.log"

sudo -E /usr/local/cuda/bin/ncu \
    --nvtx \
    --nvtx-include "Phase/DecodeAll" \
    --replay-mode kernel \
    --set none \
    --metrics "$METRICS" \
    --csv \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
    > "$OUT_CSV" 2> "$LOG_FILE"

echo "[$(date '+%H:%M:%S')] Decode 완료"
echo "  CSV 크기: $(du -sh "$OUT_CSV" | cut -f1)"
echo "  커널 행 수: $(grep -c 'gpu__time_duration' "$OUT_CSV" 2>/dev/null || echo 'N/A')"

# ------------------------------------------------------------------
# Flow 단계 측정 (선택적, FLOW_MEASURE=1 설정 시 실행)
# ------------------------------------------------------------------
if [ "${FLOW_MEASURE:-0}" = "1" ]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Flow 단계 측정 시작..."
    OUT_CSV_FLOW="$RESULTS_DIR/flow_per_kernel.csv"
    LOG_FILE_FLOW="$RESULTS_DIR/flow_per_kernel.log"

    sudo -E /usr/local/cuda/bin/ncu \
        --nvtx \
        --nvtx-include "FlowODE/FlowStep" \
        --replay-mode kernel \
        --set none \
        --metrics "$METRICS" \
        --csv \
        "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
        > "$OUT_CSV_FLOW" 2> "$LOG_FILE_FLOW"

    echo "[$(date '+%H:%M:%S')] Flow 완료"
    echo "  CSV 크기: $(du -sh "$OUT_CSV_FLOW" | cut -f1)"
fi

echo ""
echo "================================================================"
echo "  측정 완료. 분석 실행:"
echo "  python3 260610_analyze_per_kernel_bw.py"
echo "================================================================"
