#!/usr/bin/env bash
# 260609_run_ncu_full.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alpamayo 1.5 — 4단계 DRAM 대역폭 전체 측정 (Decode 전체 수정판)
#
# 260607 버전 대비 변경점:
#   [수정] Decode: step_010 1개 → DecodeAll (EOS까지 전체) 로 변경
#          --nvtx-include "DecodeAll"  (신규 NVTX 필터)
#   [유지] VE     : Phase/Vision_Encoder
#   [유지] Prefill: Phase/LM_Prefill
#   [유지] Flow   : FlowODE/FlowStep
#
# SM 11.0 확정 metrics (dram__bytes_read.sum 없음):
#   lts__d_sectors_fill_sysmem.sum           × 32 = DRAM read bytes
#   lts__t_sectors_aperture_sysmem_op_write.sum × 32 = DRAM write bytes
#   lts__t_request_hit_rate.pct
#
# 사용법:
#   chmod +x 260609_run_ncu_full.sh
#   ./260609_run_ncu_full.sh              # 전체 실행
#   SKIP_VE=1 ./260609_run_ncu_full.sh   # VE 스킵 (기존 결과 재사용)
#   DECODE_ONLY=1 ./260609_run_ncu_full.sh  # Decode만 재측정
#
# 예상 소요시간:
#   Step 0 (타이밍): ~10분
#   Step 1 (VE):     ~30분  (1,755 커널 × ncu replay)
#   Step 2 (Prefill): ~40분  (2,070 커널 × ncu replay)
#   Step 3 (Decode):  ~90분  (EOS까지 전체 커널 × ncu replay) ← 260607 대비 큰 폭 증가
#   Step 4 (Flow):    ~60분  (24,116 커널 × ncu replay)
#   Step 5 (분석):    ~1분
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

# ─── 경로 설정 ────────────────────────────────────────────────────────
NCU_BIN="/usr/local/cuda/bin/ncu"
PYTHON="/home/ice401/alpamayo1.5/a1_5_venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/260609_ncu_full_bandwidth.py"
ANALYZE_SCRIPT="$SCRIPT_DIR/260609_analyze_ncu_full.py"
RESULTS_DIR="/home/ice401/alpamayo1.5/profiling_results/260609_ncu_full"

# ─── 스킵 플래그 (1=스킵) ──────────────────────────────────────────────
SKIP_TIMING="${SKIP_TIMING:-0}"
SKIP_VE="${SKIP_VE:-0}"
SKIP_PREFILL="${SKIP_PREFILL:-0}"
SKIP_DECODE="${SKIP_DECODE:-0}"
SKIP_FLOW="${SKIP_FLOW:-0}"
SKIP_ANALYZE="${SKIP_ANALYZE:-0}"

# DECODE_ONLY=1 → 나머지 단계 모두 스킵
if [[ "${DECODE_ONLY:-0}" == "1" ]]; then
    SKIP_TIMING=1; SKIP_VE=1; SKIP_PREFILL=1; SKIP_FLOW=1
    SKIP_DECODE=0
fi

# ─── SM 11.0 확정 metrics ─────────────────────────────────────────────
# dram__bytes_read.sum은 SM 11.0에서 미지원 → LTS 섹터로 측정
METRICS="lts__d_sectors_fill_sysmem.sum,lts__t_sectors_aperture_sysmem_op_write.sum,lts__t_request_hit_rate.pct"

# ─── 결과 파일 경로 ───────────────────────────────────────────────────
OUT_VE="$RESULTS_DIR/ncu_ve.csv"
OUT_PREFILL="$RESULTS_DIR/ncu_prefill.csv"
OUT_DECODE="$RESULTS_DIR/ncu_decode_all.csv"    # ★ 전체 Decode
OUT_FLOW="$RESULTS_DIR/ncu_flow.csv"
TIMING_JSON="$RESULTS_DIR/timing_results.json"

# ─── 함수 ─────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_prereqs() {
    log "환경 확인..."
    if [[ ! -x "$NCU_BIN" ]]; then
        echo "ERROR: ncu 없음: $NCU_BIN" >&2; exit 1
    fi
    if [[ ! -f "$PYTHON" ]]; then
        echo "ERROR: Python 없음: $PYTHON" >&2; exit 1
    fi
    if [[ ! -f "$PY_SCRIPT" ]]; then
        echo "ERROR: Python 스크립트 없음: $PY_SCRIPT" >&2; exit 1
    fi
    "$NCU_BIN" --version | head -1
    "$PYTHON" -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
    mkdir -p "$RESULTS_DIR"
    log "결과 저장: $RESULTS_DIR"
}

# ncu 공통 실행 함수
# 인자: <nvtx_filter> <output_csv> <log_file>
run_ncu() {
    local NVTX_FILTER="$1"
    local OUT_CSV="$2"
    local LOG_FILE="$3"

    log "ncu 실행: --nvtx-include \"$NVTX_FILTER\""
    log "  출력: $OUT_CSV"
    log "  로그: $LOG_FILE"

    # ★ 리다이렉션 규칙:
    #   stdout (ncu --csv 출력) → OUT_CSV
    #   stderr (ncu 진행 메시지 + Python 로그) → LOG_FILE
    #
    # 잘못된 방식: "> OUT_CSV 2>&1 | tee LOG"
    #   bash에서 2>&1이 파이프보다 먼저 적용되어 stderr도 OUT_CSV로 가고
    #   tee는 빈 스트림을 받아 LOG가 0바이트가 됨
    #
    # 올바른 방식: stdout → OUT_CSV, stderr → LOG_FILE (별도)
    sudo -E "$NCU_BIN" \
        --nvtx \
        --nvtx-include "$NVTX_FILTER" \
        --replay-mode kernel \
        --set none \
        --metrics "$METRICS" \
        --csv \
        "$PYTHON" "$PY_SCRIPT" --mode ncu_single_run --results-dir "$RESULTS_DIR" \
        > "$OUT_CSV" \
        2> "$LOG_FILE"

    local EXIT_CODE=$?
    local LINES
    LINES=$(wc -l < "$OUT_CSV" 2>/dev/null || echo "0")
    log "  ncu 종료코드: $EXIT_CODE | CSV: $LINES lines"

    # stderr 로그 마지막 20줄 출력 (진행 확인용)
    if [[ -s "$LOG_FILE" ]]; then
        log "  ── stderr 로그 (마지막 20줄) ──"
        tail -20 "$LOG_FILE" | while IFS= read -r line; do
            echo "    $line"
        done
    else
        log "  ⚠ stderr 로그 비어있음 (ncu 실행 자체 실패 가능)"
    fi
}

# ─── Step 0: 타이밍 기준 측정 ─────────────────────────────────────────
step_timing() {
    log "════════════════════════════════════════"
    log "Step 0: 타이밍 기준 측정 (ncu 없음)"
    log "════════════════════════════════════════"
    "$PYTHON" "$PY_SCRIPT" \
        --mode timing \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/timing_run.log"
    log "[Step 0 완료] → $TIMING_JSON"
}

# ─── Step 1: VE (Phase/Vision_Encoder) ────────────────────────────────
step_ve() {
    log "════════════════════════════════════════"
    log "Step 1: ncu — VE (Phase/Vision_Encoder)"
    log "  예상 소요: ~30분 (1,755 커널)"
    log "════════════════════════════════════════"
    run_ncu "Phase/Vision_Encoder" "$OUT_VE" "$RESULTS_DIR/run_ve.log"
    local LINES
    LINES=$(wc -l < "$OUT_VE" 2>/dev/null || echo "0")
    if [[ "$LINES" -lt 100 ]]; then
        log "WARNING: VE CSV가 너무 작음 ($LINES lines). NVTX 캡처 실패 가능."
        log "  확인사항: Phase push → Vision_Encoder push 순서가 스크립트에 있는지 확인"
    fi
}

# ─── Step 2: Prefill (Phase/LM_Prefill) ──────────────────────────────
step_prefill() {
    log "════════════════════════════════════════"
    log "Step 2: ncu — Prefill (Phase/LM_Prefill)"
    log "  예상 소요: ~40분 (2,070 커널)"
    log "════════════════════════════════════════"
    run_ncu "Phase/LM_Prefill" "$OUT_PREFILL" "$RESULTS_DIR/run_prefill.log"
    local LINES
    LINES=$(wc -l < "$OUT_PREFILL" 2>/dev/null || echo "0")
    if [[ "$LINES" -lt 100 ]]; then
        log "WARNING: Prefill CSV가 너무 작음 ($LINES lines). NVTX 캡처 실패 가능."
    fi
}

# ─── Step 3: Decode ALL (Phase/DecodeAll) ← ★ 핵심 수정 ─────────────────
step_decode_all() {
    log "════════════════════════════════════════"
    log "Step 3: ncu — Decode 전체 (Phase/DecodeAll)"
    log "  NVTX 필터: Phase/DecodeAll  ← ncu는 2레벨 parent/child 필터 필요"
    log "    단일 레벨 'DecodeAll' → ncu가 캡처 안 함 (확인됨 260609)"
    log "    2레벨 'Phase/DecodeAll' → Phase(parent) + DecodeAll(child) 구조"
    log "  OPEN : on_vlm_pre에서 seq==1 첫 감지 시 (Phase 후 DecodeAll push)"
    log "  CLOSE: sep.ode_step==0에서 Flow 시작 직전 (Phase+DecodeAll 모두 pop)"
    log "  예상 소요: ~90분 (seed=42, 19steps × ~2,259 커널/step)"
    log "════════════════════════════════════════"
    run_ncu "Phase/DecodeAll" "$OUT_DECODE" "$RESULTS_DIR/run_decode_all.log"

    local LINES
    LINES=$(wc -l < "$OUT_DECODE" 2>/dev/null || echo "0")
    log "  Decode 전체 CSV: $LINES lines"
    if [[ "$LINES" -lt 100 ]]; then
        log "ERROR: Decode CSV가 너무 작음 ($LINES lines)."
        log "  가능한 원인:"
        log "  1) DecodeAll NVTX가 push/pop 되지 않음 — PhaseSeparator.on_vlm_pre 확인"
        log "  2) Flow step_start가 DecodeAll을 너무 빨리 닫음"
        log "  ★ 원인 분석 (260609 확인):"
        log "     ncu는 단일 레벨 NVTX 필터를 지원하지 않음"
        log "     → 'DecodeAll' 단독 → 0 커널 캡처"
        log "     → 'Phase/DecodeAll' 2레벨 → 정상 캡처"
        log "     Python 스크립트가 Phase push 없이 DecodeAll만 push 하고 있는지 확인"
        return 1
    fi
}

# ─── Step 4: Flow (FlowODE/FlowStep) ─────────────────────────────────
step_flow() {
    log "════════════════════════════════════════"
    log "Step 4: ncu — Flow (FlowODE/FlowStep)"
    log "  예상 소요: ~60분 (24,116 커널 = 10 ODE steps × ~2,411 커널/step)"
    log "════════════════════════════════════════"
    run_ncu "FlowODE/FlowStep" "$OUT_FLOW" "$RESULTS_DIR/run_flow.log"
    local LINES
    LINES=$(wc -l < "$OUT_FLOW" 2>/dev/null || echo "0")
    if [[ "$LINES" -lt 100 ]]; then
        log "WARNING: Flow CSV가 너무 작음 ($LINES lines). NVTX 캡처 실패 가능."
    fi
}

# ─── Step 5: 분석 ─────────────────────────────────────────────────────
step_analyze() {
    log "════════════════════════════════════════"
    log "Step 5: 결과 분석"
    log "════════════════════════════════════════"
    if [[ ! -f "$ANALYZE_SCRIPT" ]]; then
        log "WARNING: 분석 스크립트 없음 ($ANALYZE_SCRIPT)"
        log "  260609_analyze_ncu_full.py 를 별도로 실행하세요."
        return 0
    fi
    "$PYTHON" "$ANALYZE_SCRIPT" \
        --results-dir "$RESULTS_DIR" \
        --timing-json "$TIMING_JSON" \
        2>&1 | tee "$RESULTS_DIR/analysis.log"
}

# ─── 전체 실행 ────────────────────────────────────────────────────────
main() {
    log "Alpamayo 1.5 — 4단계 전체 DRAM 대역폭 측정 (260609 Decode 전체 수정판)"
    log "시작: $(date)"
    echo ""

    check_prereqs

    [[ "$SKIP_TIMING"  == "0" ]] && step_timing   || log "[Step 0 스킵]"
    [[ "$SKIP_VE"      == "0" ]] && step_ve        || log "[Step 1 스킵] — VE"
    [[ "$SKIP_PREFILL" == "0" ]] && step_prefill   || log "[Step 2 스킵] — Prefill"
    [[ "$SKIP_DECODE"  == "0" ]] && step_decode_all || log "[Step 3 스킵] — Decode"
    [[ "$SKIP_FLOW"    == "0" ]] && step_flow       || log "[Step 4 스킵] — Flow"
    [[ "$SKIP_ANALYZE" == "0" ]] && step_analyze    || log "[Step 5 스킵] — 분석"

    echo ""
    log "════════════════════════════════════════════════════════════"
    log "전체 완료: $(date)"
    log "결과 파일 목록:"
    ls -lh "$RESULTS_DIR"/ 2>/dev/null | grep -E '\.(csv|json|log)$' || true
    log "════════════════════════════════════════════════════════════"
}

main "$@"
