#!/usr/bin/env bash
# MIG 인스턴스 제거 및 MIG 비활성화
# 실행: sudo bash ~/alpamayo1.5/scripts/profiling/260515_exp2_mig_teardown.sh

set -euo pipefail
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "MIG 인스턴스 제거 중..."
nvidia-smi mig -dci 2>/dev/null || true   # Compute Instance 제거
nvidia-smi mig -dgi 2>/dev/null || true   # GPU Instance 제거
sleep 1

log "MIG 비활성화..."
nvidia-smi -mig 0
sleep 1

log "현재 상태:"
nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader
nvidia-smi -L
log "MIG teardown 완료."
