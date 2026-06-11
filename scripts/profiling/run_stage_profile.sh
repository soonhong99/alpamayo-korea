#!/usr/bin/env bash
# run_stage_profile.sh
# ────────────────────────────────────────────────────────────────────────────
# Alpamayo 1.5  논문급 단계 분리 프로파일링 실행 스크립트
#
# 사용법:
#   chmod +x scripts/profiling/run_stage_profile.sh
#   ./scripts/profiling/run_stage_profile.sh
#
# 출력:
#   profiling_results/
#   ├── raw_timings.json       ← 모든 런의 원시 타이밍 (vision/prefill/decode/action)
#   ├── summary.json           ← 통계 (mean/std/p50/p95/p99)
#   ├── stage_breakdown.csv    ← matplotlib 입력용 CSV
#   └── nsight_stage.nsys-rep  ← NSight 타임라인 (선택)
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/profiling_results"
VENV="${HOME}/alpamayo1.5/a1_5_venv"

mkdir -p "${OUTPUT_DIR}"

source "${VENV}/bin/activate"
echo "[Stage Profile] Python: $(which python)"
echo "[Stage Profile] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[Stage Profile] 출력 디렉토리: ${OUTPUT_DIR}"

# ── 옵션 A: NSight 없이 빠른 실행 ────────────────────────────────────────
# 훅이 제대로 동작하는지 먼저 확인할 때 사용
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: 단계별 타이밍 측정 (NSight 없이)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "${PROJECT_ROOT}"
python "${SCRIPT_DIR}/profile_alpamayo.py" \
    --warmup 3 \
    --runs 8

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: NSight Systems + 단계별 NVTX 마커"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# NSight 재실행 → 올바른 NVTX 마커로 타임라인 생성
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --output="${OUTPUT_DIR}/nsight_stage" \
    --force-overwrite=true \
    --sample=cpu \
    --duration=180 \
    python "${SCRIPT_DIR}/profile_alpamayo.py" \
        --warmup 2 \
        --runs 5

echo "[Stage Profile] NSight 완료: ${OUTPUT_DIR}/nsight_stage.nsys-rep"

# ── NVTX 단계별 시간 추출 ────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: NVTX 단계별 시간 추출"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nsys stats \
    --report nvtx_sum \
    --format csv \
    "${OUTPUT_DIR}/nsight_stage.nsys-rep" \
    | tee "${OUTPUT_DIR}/nvtx_stage_breakdown.csv"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4: CUDA 커널 요약"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    "${OUTPUT_DIR}/nsight_stage.nsys-rep" \
    | head -30 \
    | tee "${OUTPUT_DIR}/kernel_summary.csv"

echo ""
echo "✅ 단계 분리 프로파일링 완료"
echo "   raw_timings.json   ← Python CUDA Event 기반 정밀 수치"
echo "   summary.json       ← 통계 (mean/std/p50/p95)"
echo "   nvtx_stage_breakdown.csv  ← NSight NVTX 단계별 시간"
echo ""
echo "   nsight_stage.nsys-rep 에서 확인할 항목:"
echo "   [vlm_generate] 안에 [vision_encoding], [llm_prefill], [llm_decode]"
echo "   [alpamayo_full_inference] - [vlm_generate] = Flow Matching 시간"
