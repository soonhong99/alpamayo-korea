#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# EXP-2 Step 1: MIG 활성화 및 인스턴스 생성
#
# 실행:
#   sudo bash ~/alpamayo1.5/scripts/profiling/260515_exp2_mig_setup.sh
#
# 이 스크립트가 하는 일:
#   1. MIG 활성화
#   2. 사용 가능한 GI 프로파일 출력
#   3. 실험용 인스턴스 생성 (Vision=1g, VLM=최대, Action=1g)
#   4. 생성된 UUID 목록을 파일로 저장
#      → exp2_real_mig_measure.py 가 이 파일을 읽어 실행
# ─────────────────────────────────────────────────────────────────

set -euo pipefail
OUT_DIR="$HOME/alpamayo1.5/profiling_results/260515_exp2"
mkdir -p "$OUT_DIR"
UUID_FILE="$OUT_DIR/mig_uuids.json"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. MIG 활성화 ─────────────────────────────────────────────
log "MIG 활성화 중..."
nvidia-smi -mig 1
sleep 2

log "MIG 현재 상태:"
nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader

# ── 2. 사용 가능한 GI 프로파일 출력 ──────────────────────────
log ""
log "사용 가능한 GI 프로파일:"
nvidia-smi mig -lgip
log ""

# ── 3. 프로파일 자동 감지 및 인스턴스 생성 ───────────────────
# Thor SM=20 기준 예상 프로파일:
#   1g.Xgb  → ID 확인 필요
#   2g.Xgb  → ID 확인 필요
#   4g.Xgb  → ID 확인 필요
#
# 실험 목표 배정:
#   Vision  : 가장 작은 슬라이스 (1g)
#   VLM     : 가장 큰 슬라이스 (나머지)
#   Action  : 가장 작은 슬라이스 (1g)
#
# nvidia-smi mig -lgip 출력에서 ID 자동 파싱

log "GI 프로파일 ID 파싱 중..."

# 가장 작은 슬라이스 ID 감지 (SM이 가장 적은 것)
SMALL_ID=$(nvidia-smi mig -lgip 2>/dev/null \
    | grep -E '^\|[[:space:]]+[0-9]+' \
    | awk '{print $3, $9}' \
    | sort -k2 -n \
    | head -1 \
    | awk '{print $1}')

# 가장 큰 슬라이스 ID 감지 (SM이 가장 많은 것)
LARGE_ID=$(nvidia-smi mig -lgip 2>/dev/null \
    | grep -E '^\|[[:space:]]+[0-9]+' \
    | awk '{print $3, $9}' \
    | sort -k2 -n \
    | tail -1 \
    | awk '{print $1}')

log "감지된 프로파일 ID: SMALL=$SMALL_ID, LARGE=$LARGE_ID"

# 프로파일 ID를 찾지 못하면 수동 입력 안내
if [[ -z "$SMALL_ID" || -z "$LARGE_ID" ]]; then
    log ""
    log "⚠️  프로파일 자동 감지 실패."
    log "위의 'nvidia-smi mig -lgip' 출력에서 ID를 확인하고"
    log "아래 변수를 직접 수정하세요:"
    log ""
    log "  SMALL_ID=<1g 프로파일 ID>"
    log "  LARGE_ID=<가장 큰 프로파일 ID>"
    log ""
    log "그 후 다시 실행:"
    log "  sudo nvidia-smi mig -cgi <SMALL_ID>,<LARGE_ID>,<SMALL_ID> -C"
    exit 1
fi

# ── 4. GI 인스턴스 생성: Vision(small) + VLM(large) + Action(small) ──
log ""
log "GI 인스턴스 생성: $SMALL_ID (Vision), $LARGE_ID (VLM), $SMALL_ID (Action)"
nvidia-smi mig -cgi "${SMALL_ID},${LARGE_ID},${SMALL_ID}" -C
sleep 2

log ""
log "생성된 MIG 인스턴스:"
nvidia-smi mig -lgi
log ""
nvidia-smi -L

# ── 5. UUID 파싱 및 저장 ─────────────────────────────────────
log ""
log "MIG UUID 파싱 중..."

# nvidia-smi -L 출력에서 MIG UUID 추출
# 출력 예:
#   GPU 0: NVIDIA Thor (...)
#     MIG 1g.Xgb Device 0: (UUID: MIG-aaaa...)
#     MIG 4g.Xgb Device 1: (UUID: MIG-bbbb...)
#     MIG 1g.Xgb Device 2: (UUID: MIG-cccc...)

UUIDS=()
while IFS= read -r line; do
    if [[ "$line" =~ UUID:\ (MIG-[a-f0-9\-]+) ]]; then
        UUIDS+=("${BASH_REMATCH[1]}")
    fi
done < <(nvidia-smi -L)

if [[ ${#UUIDS[@]} -lt 3 ]]; then
    log "⚠️  UUID가 3개 미만 감지됨 (${#UUIDS[@]}개). 인스턴스 생성을 확인하세요."
    nvidia-smi -L
    exit 1
fi

log "감지된 UUID:"
for i in "${!UUIDS[@]}"; do
    log "  Device $i: ${UUIDS[$i]}"
done

# JSON으로 저장 (Python 스크립트가 읽음)
python3 - <<EOF
import json
uuids = ${UUIDS[@]+"${UUIDS[@]}"}
data = {
    "vision_uuid":  "${UUIDS[0]}",
    "vlm_uuid":     "${UUIDS[1]}",
    "action_uuid":  "${UUIDS[2]}",
    "small_gi_id":  "${SMALL_ID}",
    "large_gi_id":  "${LARGE_ID}",
}
with open("${UUID_FILE}", "w") as f:
    json.dump(data, f, indent=2)
print(f"UUID 저장 완료: ${UUID_FILE}")
EOF

log ""
log "════════════════════════════════════════"
log "MIG 설정 완료."
log "다음 단계:"
log "  python3 ~/alpamayo1.5/scripts/profiling/260515_exp2_mig_real_measure.py"
log "════════════════════════════════════════"
