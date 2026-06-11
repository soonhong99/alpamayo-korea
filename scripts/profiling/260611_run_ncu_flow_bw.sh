#!/bin/bash
# =============================================================================
# 260611_run_ncu_flow_bw.sh
# 목적: Flow (ODE) 단계 커널별 순간 DRAM 대역폭 + SM 활용률 측정
#
# 핵심 질문:
#   - SM util ≈ 0%? → memory-bound (가중치 전송 병목)  ← 예상
#   - SM util > 0%? → compute-bound (ODE solver의 연산 집중적)
#
# 배경:
#   Flow stage BW = 203 GB/s (88%) ← 4단계 중 가장 높음
#   88%는 BW 포화에 가까운 수준 → memory-bound가 강력한 예상.
#   그러나 ODE 계산 과정에서 수치적분(Runge-Kutta 등)이
#   연산을 늘릴 수 있음 → SM util로 최종 확인 필요.
#
#   Flow(Diffusion Flow Matching)는:
#   - U-Net 또는 Transformer 기반 denoising 네트워크
#   - ODE step × N_steps 반복 (일반적으로 4~20 step)
#   - 각 step마다 동일한 가중치 재사용 (가중치 1회 로드 per step)
#
# NVTX 구조 (nsys 실측 확인 2026-06-11):
#   FlowODE (wrapper push-pop)    ← 커널 없음, ncu 0개 캡처 (2026-06-11 실패 확인)
#    └── FlowStep (per-ODE-step)  ← 실제 커널은 여기 안에 있음
#
#   FlowODE 필터는 wrapper range라서 직접 커널이 없음 → 0 kernels profiled.
#   FlowStep으로 변경 (2026-06-11 수정).
#
# 실행: sudo -E bash 260611_run_ncu_flow_bw.sh
# 분석:
#   python3 260611_analyze_prefill_bw.py \
#     --ncu  ~/alpamayo1.5/profiling_results/260611_flow_bw/flow_per_kernel.csv \
#     --nsys ~/alpamayo1.5/profiling_results/260610_per_kernel_bw/decode_timeline.sqlite \
#     --prefill-nvtx FlowStep
# =============================================================================

set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
INFERENCE_SCRIPT="$HOME/alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260611_flow_bw"

mkdir -p "$RESULTS_DIR"

# ★ SM 메트릭 수정 (2026-06-11 --list-metrics 실측):
#   sm__active_cycles.sum / gpc__cycles_elapsed.max → GB10B 미존재 → 항상 0
#   smsp__cycles_active.sum / smsp__cycles_elapsed.sum → GB10B 유일 지원 공식
# SM 메트릭 제거 — DRAM BW만 측정하면 충분 (SM util은 연구 범위 밖)
METRICS="lts__d_sectors_fill_sysmem.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum,\
lts__t_request_hit_rate.pct,\
gpu__time_duration.sum"

echo "================================================================"
echo "  Flow(ODE) 단계 커널별 instantaneous BW 측정 (SM 메트릭 제외)"
echo "  NVTX 필터 전략 (3단계 시도):"
echo "    1차: FlowODE/FlowStep  (부모/자식 경로)"
echo "    2차: Measure/run_01/FlowODE (전체 경로)"
echo "    3차: 전체 캡처 후 시간 범위 필터링"
echo "  핵심 질문: SM util ≈ 0? (memory-bound, 예상) vs > 0? (compute-bound)"
echo "  Stage BW 88%(203 GB/s): 이미 매우 높음 → BW 포화 확인"
echo "  예상 소요: 10~30분"
echo "================================================================"
echo "[$(date '+%H:%M:%S')] 시작..."

OUT_CSV="$RESULTS_DIR/flow_per_kernel.csv"
LOG_FILE="$RESULTS_DIR/flow_per_kernel.log"

# ──────────────────────────────────────────────────────────────────────
# [단계 0] 먼저 추론 로그 확인 (FlowODE 훅 등록 여부)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "[0/3] 추론 로그 확인 (훅 등록 상태)..."
"$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run 2>&1 | head -40
echo ""

# ──────────────────────────────────────────────────────────────────────
# [단계 1] 1차 시도: FlowODE/FlowStep (부모/자식 경로)
# VE 필터 "Phase/Vision_Encoder" 와 동일한 형식
# ──────────────────────────────────────────────────────────────────────
echo "[1/3] ncu 시도 1: --nvtx-include \"FlowODE/FlowStep\"..."
ATTEMPT1_CSV="$RESULTS_DIR/flow_attempt1_FlowODE_FlowStep.csv"
ATTEMPT1_LOG="$RESULTS_DIR/flow_attempt1.log"

sudo -E /usr/local/cuda/bin/ncu \
    --nvtx \
    --nvtx-include "FlowODE/FlowStep" \
    --replay-mode kernel \
    --set none \
    --metrics "$METRICS" \
    --csv \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
    > "$ATTEMPT1_CSV" 2> "$ATTEMPT1_LOG" || true

COUNT1=$(grep -c 'smsp__cycles_active\|lts__d_sectors' "$ATTEMPT1_CSV" 2>/dev/null || echo 0)
echo "  커널 행 수: $COUNT1"

if [ "$COUNT1" -gt "0" ]; then
    echo "  ✅ 성공! FlowODE/FlowStep 필터 작동"
    cp "$ATTEMPT1_CSV" "$OUT_CSV"
    cp "$ATTEMPT1_LOG" "$LOG_FILE"
else
    echo "  ❌ 실패 (0 kernels). 2차 시도로..."

    # ──────────────────────────────────────────────────────────────────
    # [단계 2] 2차 시도: Measure/run_01/FlowODE (전체 경로)
    # ──────────────────────────────────────────────────────────────────
    echo ""
    echo "[2/3] ncu 시도 2: --nvtx-include \"Measure/run_01/FlowODE\"..."
    ATTEMPT2_CSV="$RESULTS_DIR/flow_attempt2_full_path.csv"
    ATTEMPT2_LOG="$RESULTS_DIR/flow_attempt2.log"

    sudo -E /usr/local/cuda/bin/ncu \
        --nvtx \
        --nvtx-include "Measure/run_01/FlowODE" \
        --replay-mode kernel \
        --set none \
        --metrics "$METRICS" \
        --csv \
        "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run \
        > "$ATTEMPT2_CSV" 2> "$ATTEMPT2_LOG" || true

    COUNT2=$(grep -c 'smsp__cycles_active\|lts__d_sectors' "$ATTEMPT2_CSV" 2>/dev/null || echo 0)
    echo "  커널 행 수: $COUNT2"

    if [ "$COUNT2" -gt "0" ]; then
        echo "  ✅ 성공! Measure/run_01/FlowODE 필터 작동"
        cp "$ATTEMPT2_CSV" "$OUT_CSV"
        cp "$ATTEMPT2_LOG" "$LOG_FILE"
    else
        echo "  ❌ 실패 (0 kernels). 3차 시도로..."

        # ────────────────────────────────────────────────────────────
        # [단계 3] 3차 시도: FlowODE 없이 전체 캡처 후 Python 시간 필터
        # 가장 신뢰성 높은 방법 — 모든 커널 캡처 후 nsys FlowODE 시간창 적용
        # ⚠ 전체 추론 캡처로 CSV가 크고 시간이 매우 오래 걸림 (30~90분)
        # ────────────────────────────────────────────────────────────
        echo ""
        echo "[3/3] ncu 시도 3: NVTX 없이 전체 캡처 (FlowODE 시간창 기반 필터링)"
        echo "  ⚠ 주의: Decode ~44000 커널 포함 → CSV 매우 크고 시간 오래 걸림"
        echo "  ⚠ 이 시도는 건너뜁니다. 훅 등록 문제 해결 후 재시도 권장."
        echo ""
        echo "  ── 진단 정보 ──"
        echo "  시도 1 로그 (마지막 20줄):"
        tail -20 "$ATTEMPT1_LOG" 2>/dev/null || echo "    (로그 없음)"
        echo ""
        echo "  시도 2 로그 (마지막 20줄):"
        tail -20 "$ATTEMPT2_LOG" 2>/dev/null || echo "    (로그 없음)"

        # 빈 파일 생성 (에러 방지)
        echo "==PROF== No kernels profiled after 3 attempts" > "$OUT_CSV"
        echo "See attempt logs for details" >> "$OUT_CSV"
        cp "$ATTEMPT1_LOG" "$LOG_FILE"
    fi
fi

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
echo "    --prefill-nvtx FlowStep"
echo ""
echo "  SM util ≈ 0% → memory-bound 확인 → 양자화, prefetch 대상"
echo "  SM util > 0% → compute-bound → FP4 변환, 알고리즘 최적화"
echo ""
echo "  ※ FlowODE NVTX 이름 불일치 시 nsys에서 실제 이름 확인:"
echo "  sqlite3 $HOME/alpamayo1.5/profiling_results/260610_per_kernel_bw/decode_timeline.sqlite \\"
echo "    \"SELECT text FROM StringIds WHERE text LIKE '%flow%' OR text LIKE '%Flow%' COLLATE NOCASE;\""
echo "================================================================"
