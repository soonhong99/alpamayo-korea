#!/usr/bin/env bash
# run_nsight.sh
# ─────────────────────────────────────────────────────────────────────────────
# NSight Systems + NSight Compute 프로파일링 래퍼
# Jetson AGX Thor (aarch64, CUDA 13.0, SM 11.0) 기준
#
# 사용법:
#   chmod +x scripts/profiling/run_nsight.sh
#   ./scripts/profiling/run_nsight.sh
#
# 출력:
#   profiling_results/
#   ├── nsight_sys.nsys-rep       ← NSight Systems GUI에서 열기
#   ├── nsight_sys_summary.txt    ← CLI 텍스트 요약
#   ├── nsight_compute.ncu-rep    ← NSight Compute GUI에서 열기
#   └── nsight_compute_summary.csv
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/profiling_results"
VENV="${HOME}/alpamayo1.5/a1_5_venv"

mkdir -p "${OUTPUT_DIR}"

# ── 환경 활성화 ──────────────────────────────────────────────────────────────
source "${VENV}/bin/activate"
echo "[NSight] Python: $(which python)"
echo "[NSight] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[NSight] 출력 디렉토리: ${OUTPUT_DIR}"

# ── NSight Systems — 시스템 레벨 타임라인 ───────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: NSight Systems (시스템 타임라인)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas,cusparse \
    --cuda-memory-usage=true \
    --output="${OUTPUT_DIR}/nsight_sys" \
    --force-overwrite=true \
    --sample=cpu \
    --backtrace=dwarf \
    --duration=120 \
    python "${SCRIPT_DIR}/profile_alpamayo.py" \
        --warmup 3 \
        --runs 10

echo "[NSight] nsys 완료: ${OUTPUT_DIR}/nsight_sys.nsys-rep"

# ── NSight Systems CLI 요약 출력 ─────────────────────────────────────────────
nsys stats \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
    --format csv \
    --output "${OUTPUT_DIR}/nsight_sys_summary" \
    "${OUTPUT_DIR}/nsight_sys.nsys-rep" 2>/dev/null || true

echo "[NSight] 텍스트 요약 저장됨"

# ── NSight Compute — 커널 레벨 분석 ─────────────────────────────────────────
# 주의: 커널 하나씩 재실행하므로 시간이 매우 오래 걸림 (10~30분)
# --target-processes: 자식 프로세스까지 추적
# --set full: 모든 메트릭 수집 (roofline 포함)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: NSight Compute (커널 분석)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ncu \
    --set full \
    --target-processes all \
    --kernel-name-base function \
    --launch-count 5 \
    --clock-control none \
    --export "${OUTPUT_DIR}/nsight_compute" \
    --force-overwrite \
    python "${SCRIPT_DIR}/profile_alpamayo.py" \
        --warmup 1 \
        --runs 1 2>&1 | head -200 || true

# CSV 내보내기
ncu \
    --import "${OUTPUT_DIR}/nsight_compute.ncu-rep" \
    --csv \
    --page raw \
    > "${OUTPUT_DIR}/nsight_compute_summary.csv" 2>/dev/null || true

echo "[NSight] ncu 완료: ${OUTPUT_DIR}/nsight_compute.ncu-rep"

# ── tegrastats 병행 수집 ─────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: tegrastats 하드웨어 모니터링"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python "${SCRIPT_DIR}/tegrastats_monitor.py" \
    --interval 50 \
    --duration 60 \
    --output "${OUTPUT_DIR}/tegrastats.json" &
MONITOR_PID=$!

# 추론 재실행 (tegrastats와 동시)
python "${SCRIPT_DIR}/profile_alpamayo.py" \
    --warmup 5 \
    --runs 20 \
    --pytorch_profiler

kill ${MONITOR_PID} 2>/dev/null || true
wait ${MONITOR_PID} 2>/dev/null || true

echo "[NSight] tegrastats 완료"

# ── 리포트 생성 ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4: 시각화 리포트 생성"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python "${SCRIPT_DIR}/report_generator.py" \
    --input_dir "${OUTPUT_DIR}"

echo ""
echo "✅ 프로파일링 완료"
echo "   결과 폴더: ${OUTPUT_DIR}"
echo "   NSight GUI: nsight_sys.nsys-rep / nsight_compute.ncu-rep"
echo "   요약 보고서: report_alpamayo_profiling.pdf"
