#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# EXP-0 ~ EXP-4 순차 실행 런처
# 실험 계획서: docs/260515_mig_pipeline_experiment_plan.md
#
# 실행 방법 (Thor):
#   source ~/alpamayo1.5/a1_5_venv/bin/activate
#   bash ~/alpamayo1.5/scripts/profiling/260515_exp_runner.sh
#
# 실험 결과:
#   profiling_results/260515_exp0/   ← MIG 환경 확인
#   profiling_results/260515_exp1/   ← MIG 슬라이스 스케일링
#   profiling_results/260515_exp3/   ← 크로스프레임 파이프라인 (핵심)
#   profiling_results/260515_exp4/   ← GPU 용량 스케일링
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="$(cd "$SCRIPT_DIR/../../profiling_results" 2>/dev/null && pwd || echo "profiling_results")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "═══════════════════════════════════════"
log "Alpamayo 1.5 병렬화 실험 시작"
log "스크립트 디렉터리: $SCRIPT_DIR"
log "═══════════════════════════════════════"

# ── EXP-0: MIG 환경 확인 ──────────────────────────────────────
log ""
log ">>> EXP-0: MIG 활성화 확인 및 환경 진단"
python3 "$SCRIPT_DIR/260515_exp0_mig_check.py"
log "EXP-0 완료"

# EXP-0 결과 읽기 (MIG 지원 여부 확인)
EXP0_JSON="$RESULTS_BASE/260515_exp0/exp0_mig_check.json"
if [[ -f "$EXP0_JSON" ]]; then
    MIG_SUPPORTED=$(python3 -c "
import json
with open('$EXP0_JSON') as f:
    d = json.load(f)
print(d['verdict']['mig_supported'])
" 2>/dev/null || echo "False")
    log "MIG 지원: $MIG_SUPPORTED"
else
    MIG_SUPPORTED="False"
fi

# ── EXP-1: MIG 슬라이스 스케일링 ─────────────────────────────
log ""
log ">>> EXP-1: MIG 슬라이스 크기별 성능 스케일링 (proxy 모드)"
python3 "$SCRIPT_DIR/260515_exp1_mig_scaling.py"
log "EXP-1 완료"

# ── EXP-3: 크로스프레임 파이프라인 (핵심!) ──────────────────
log ""
log ">>> EXP-3: 크로스프레임 파이프라인 (핵심 실험)"
python3 "$SCRIPT_DIR/260515_exp3_pipeline.py" --frames 10 --warmup 2
log "EXP-3 완료"

# ── EXP-4: GPU 용량 스케일링 ─────────────────────────────────
log ""
log ">>> EXP-4: GPU 용량 스케일링"
python3 "$SCRIPT_DIR/260515_exp4_capacity_scaling.py" --max-gb 32
log "EXP-4 완료"

# ── 최종 요약 ─────────────────────────────────────────────────
log ""
log "═══════════════════════════════════════"
log "모든 실험 완료. 결과 요약:"
log "═══════════════════════════════════════"

for exp_dir in exp0 exp1 exp3 exp4; do
    dir="$RESULTS_BASE/260515_$exp_dir"
    if [[ -d "$dir" ]]; then
        file_count=$(ls "$dir"/*.{json,md,png} 2>/dev/null | wc -l)
        log "  $exp_dir: $file_count 파일 생성됨 ($dir)"
    fi
done

log ""
log "결과 파일 전송 (Windows로):"
log "  scp -r ice401@100.95.177.101:~/alpamayo1.5/profiling_results/260515_exp* \\"
log "      /mnt/c/Users/nanay/Desktop/Alphamayo/profiling_results/"
log ""
log "핵심 결과 (EXP-3 Pipeline):"
EXP3_JSON="$RESULTS_BASE/260515_exp3/pipeline_results.json"
if [[ -f "$EXP3_JSON" ]]; then
    python3 -c "
import json
with open('$EXP3_JSON') as f:
    d = json.load(f)
a = d['analysis']
print(f\"  Baseline FPS: {a['baseline_fps']:.4f}\")
print(f\"  Pipeline FPS: {a['pipeline_fps']:.4f}\")
print(f\"  Speedup:      {a['measured_speedup']:.2f}×\")
print(f\"  H2 확인:      {a['hypothesis_H2_confirmed']}\")
"
fi
