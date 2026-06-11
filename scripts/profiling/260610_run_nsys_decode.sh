#!/bin/bash
# =============================================================================
# 260610_run_nsys_decode.sh
# 목적: 커널 간 갭(idle time) 및 CPU-GPU 타임라인 분석
#
# ncu --replay-mode kernel은 커널을 개별 재실행하므로
# 커널 사이의 실제 간격(갭)을 알 수 없다.
# nsys는 replay 없이 실제 실행 흐름을 타임스탬프로 기록한다.
# → 커널 시작/종료 시각, 커널 간 갭, CPU 오버헤드 측정 가능
#
# 측정 항목:
#   - CUDA 커널 실행 타임라인 (시작/종료 시각 ns 정밀도)
#   - 커널 간 갭 (GPU idle time)
#   - CPU-side 스케줄링 오버헤드 (Python/PyTorch dispatch latency)
#   - SM occupancy 타임라인 (SM이 실제 사용되는 시간 비율)
#
# 실행: sudo bash 260610_run_nsys_decode.sh
#
# 예상 소요: 약 5~15분 (replay 없어서 ncu보다 훨씬 빠름)
# 출력: decode_timeline.nsys-rep (Nsight Systems GUI에서 열기 가능)
#       decode_timeline.sqlite    (Python 분석용)
# =============================================================================

set -euo pipefail

PYTHON="$HOME/alpamayo1.5/a1_5_venv/bin/python3"
INFERENCE_SCRIPT="$HOME/alpamayo1.5/scripts/profiling/260609_ncu_full_bandwidth.py"
RESULTS_DIR="$HOME/alpamayo1.5/profiling_results/260610_per_kernel_bw"

mkdir -p "$RESULTS_DIR"

OUT_BASE="$RESULTS_DIR/decode_timeline"

echo "================================================================"
echo "  nsys 커널 타임라인 수집 (Decode 단계)"
echo "  replay 없음 → 실제 실행 시간 그대로 기록"
echo "  예상 소요: 5~15분"
echo "================================================================"
echo "[$(date '+%H:%M:%S')] 시작..."

# nsys profile 옵션 설명:
#   --trace=cuda,nvtx   : CUDA 커널 + NVTX 마커 기록
#   --cuda-graph-trace=node : CUDA Graph 내 개별 노드도 추적
#   --sample=none       : CPU 샘플링 비활성 (오버헤드 최소화)
#   --force-overwrite   : 기존 파일 덮어쓰기
sudo -E /usr/local/cuda/bin/nsys profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --sample=none \
    --output "$OUT_BASE" \
    --force-overwrite true \
    "$PYTHON" "$INFERENCE_SCRIPT" --mode ncu_single_run

echo "[$(date '+%H:%M:%S')] nsys profile 완료: ${OUT_BASE}.nsys-rep"

# SQLite로 변환 (Python 분석 스크립트에서 사용)
echo "[$(date '+%H:%M:%S')] SQLite 변환 중..."
/usr/local/cuda/bin/nsys export \
    --type sqlite \
    --output "${OUT_BASE}.sqlite" \
    --force-overwrite true \
    "${OUT_BASE}.nsys-rep"

echo "[$(date '+%H:%M:%S')] SQLite 완료: ${OUT_BASE}.sqlite"
echo ""
echo "================================================================"
echo "  분석 실행:"
echo "  python3 260610_analyze_per_kernel_bw.py --mode nsys"
echo ""
echo "  GUI 뷰어 (Windows에서):"
echo "  Nsight Systems에서 ${OUT_BASE}.nsys-rep 열기"
echo "================================================================"
