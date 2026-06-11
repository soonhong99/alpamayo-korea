#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# 260607_run_ncu_stages.sh
#
# Alpamayo 1.5 — 4단계 실제 DRAM 대역폭 측정 파이프라인
#
# 실행 순서:
#   Step 0: ncu 버전 확인 + SM 11.0 에서 사용 가능한 DRAM metrics 조회
#   Step 1: 타이밍 측정 (ncu 없이 CUDA Event로, 빠름)
#   Step 2: 이론값 추정 출력 (ncu 없이, 1분)
#   Step 3: ncu — LM Decode steady-state 1 step (가장 중요, 30초)
#   Step 4: ncu — LM Prefill (compute-bound 검증, 5~10분)
#   Step 5: ncu — VE (Vision Encoder, 3~5분)
#   Step 6: ncu — Flow (diffusion, 3분)
#   Step 7: 분석 및 최종 BW 표 출력
#
# 사용법:
#   # Thor 보드에서:
#   source ~/alpamayo1.5/a1_5_venv/bin/activate
#   cd ~/alpamayo1.5
#   bash scripts/profiling/260607_run_ncu_stages.sh 2>&1 | tee profiling_results/260607_ncu_bandwidth/run_log.txt
#
#   # 특정 Step만 실행:
#   SKIP_TIMING=1 SKIP_VE=1 bash scripts/profiling/260607_run_ncu_stages.sh
#
# 예상 소요 시간:
#   Step 1 (timing)  : ~15분  (warmup 1 + 측정 2)
#   Step 3 (decode)  : ~10분  (1 step replay, ncu overhead ~10×)
#   Step 4 (prefill) : ~30분  (seq=3086, ncu replay heavy)
#   Step 5 (VE)      : ~20분
#   Step 6 (flow)    : ~15분
#   Total            : ~90분
#
# 권한 문제 발생 시:
#   ncu: ERR_NVGPUCTRPERM → sudo ncu ... 로 실행
#   또는: sudo sh -c "echo 0 > /proc/sys/kernel/perf_event_paranoid"
# ═══════════════════════════════════════════════════════════════════════

set -e

# ─── 경로 설정 ────────────────────────────────────────────────────────
VENV_PYTHON="/home/ice401/alpamayo1.5/a1_5_venv/bin/python3"
SCRIPT_DIR="/home/ice401/alpamayo1.5/scripts/profiling"
SCRIPT="$SCRIPT_DIR/260607_ncu_bandwidth_measurement.py"
ANALYSIS="$SCRIPT_DIR/260607_analyze_ncu_bandwidth.py"
RESULTS_DIR="/home/ice401/alpamayo1.5/profiling_results/260607_ncu_bandwidth"
mkdir -p "$RESULTS_DIR"

# ─── 스킵 플래그 (환경 변수로 제어) ─────────────────────────────────
SKIP_TIMING=${SKIP_TIMING:-0}
SKIP_ESTIMATE=${SKIP_ESTIMATE:-0}
SKIP_DECODE=${SKIP_DECODE:-0}
SKIP_PREFILL=${SKIP_PREFILL:-0}
SKIP_VE=${SKIP_VE:-0}
SKIP_FLOW=${SKIP_FLOW:-0}

# ─── ncu 공통 설정 ───────────────────────────────────────────────────
# SM 11.0 (Thor Blackwell) 에서 유효한 DRAM metric 이름:
#   lts__d_sectors_fill_sysmem.sum          ← DRAM→L2 fill sectors (DRAM reads), ×32=bytes
#   lts__t_sectors_aperture_sysmem_op_write.sum ← DRAM write sectors
#   lts__t_request_hit_rate                 ← L2 hit rate (%)
# (dram__bytes_read.sum 은 SM 11.0에서 "n/a" — 사용 불가)
LTS_METRICS="lts__d_sectors_fill_sysmem.sum,lts__t_sectors_aperture_sysmem_op_write.sum,lts__t_request_hit_rate"

# NVTX 범위 구조 (이중 push, ncu hierarchy 필터 호환):
#   Phase/Vision_Encoder  → push("Phase") + push("Vision_Encoder")
#   Phase/LM_Prefill      → push("Phase") + push("LM_Prefill")
#   Phase/Decode_all      → push("Phase") + push("Decode_all")
#   Decode/step_010       → push("Decode") + push("step_010")   [Decode_all 내부]
#   FlowODE/FlowStep      → push("FlowODE") + push("FlowStep")
#     ★ Flow는 "Phase" 이름을 쓰지 않는다:
#       Flow 실행 시점에 Phase(Decode)/Decode_all이 아직 스택에 살아있기 때문.
#       ncu "Phase/Flow" 필터는 FIRST Phase의 direct child = Decode_all 에서
#       매칭 실패함. 충돌 없는 고유 이름 FlowODE/FlowStep 사용.

# sudo 환경변수 전달 필수:
#   HF_HOME: Alpamayo 모델 로컬 캐시 경로
#   ldconfig: /etc/ld.so.conf.d/pytorch-local.conf 로 PyTorch so 경로 등록됨

# ─── ncu 명령 탐색 (CUDA toolkit 경로) ──────────────────────────────
NCU=$(command -v ncu 2>/dev/null || echo "/usr/local/cuda/bin/ncu")
if [ ! -x "$NCU" ]; then
    echo "[ERROR] ncu 를 찾을 수 없습니다."
    echo "  시도: sudo apt install cuda-tools-13-0 또는"
    echo "        export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

echo "ncu 경로: $NCU"
echo "Python: $VENV_PYTHON"
echo "결과 저장: $RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────
# Step 0: ncu 버전 + SM 11.0 사용 가능 metrics 확인
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "Step 0: ncu 환경 확인"
echo "════════════════════════════════════════════"

$NCU --version 2>&1 | head -3

METRICS_FILE="$RESULTS_DIR/available_dram_metrics.txt"
echo "[metrics 조회 중...]"
$NCU --query-metrics 2>/dev/null | grep -iE "dram|l2cache|lts__t" \
    > "$METRICS_FILE" || echo "(metrics 조회 실패 — 권한 부족 가능)"
echo "DRAM/L2 관련 metrics → $METRICS_FILE"
wc -l "$METRICS_FILE" 2>/dev/null && true

# SM 11.0 지원 metric set 결정
# 기본값 (Ampere/Hopper 호환, SM 11.0에서도 동작 예상)
BASE_METRICS="dram__bytes_read.sum,dram__bytes_write.sum"
EXTENDED_METRICS="${BASE_METRICS},dram__throughput.avg.pct_of_peak_sustained_elapsed,l2cache__read_hit_rate.pct"

# SM utilization도 포함 (메모리 bound vs compute bound 확인)
FULL_METRICS="${EXTENDED_METRICS},sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed"

echo ""
echo "사용할 metrics:"
echo "  BASE:     $BASE_METRICS"
echo "  EXTENDED: $EXTENDED_METRICS"
echo "  FULL:     $FULL_METRICS"

# ─────────────────────────────────────────────────────────────────────
# Step 1: CUDA Event 타이밍 측정 (ncu 없이 빠르게)
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_TIMING" = "0" ]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "Step 1: 타이밍 측정 (CUDA Event, ncu 없음)"
    echo "════════════════════════════════════════════"
    $VENV_PYTHON "$SCRIPT" --mode timing --warmup 1 --runs 2 \
        2>&1 | tee "$RESULTS_DIR/step1_timing.log"
    echo "[Step 1 완료] → $RESULTS_DIR/timing_results.json"
else
    echo "[Step 1 스킵]"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 2: 이론 추정값 출력 (모델 로드만, 추론 없음)
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_ESTIMATE" = "0" ]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "Step 2: 이론 추정값 (가중치 크기 + 실측 시간 기반)"
    echo "════════════════════════════════════════════"
    $VENV_PYTHON "$SCRIPT" --mode estimate_only \
        2>&1 | tee "$RESULTS_DIR/step2_estimate.log"
    echo "[Step 2 완료] → $RESULTS_DIR/estimate_only.json"
else
    echo "[Step 2 스킵]"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 3: ncu — LM Decode steady-state (가장 중요)
#
# 목표: Decode step 10 (steady-state) 에서 실제 DRAM bytes 측정
#   - "Phase/Decode" NVTX 범위 내에서 "Decode/step_010" 만 캡처
#   - KV cache가 누적된 상태에서의 실제 DRAM 접근 측정
#   - 예상: dram__bytes_read ~ 16.5 GB per step (LM 가중치 + KV cache)
#
# ncu --nvtx-include "Decode/step_010":
#   warmup run에서 생긴 step_010 NVTX 범위를 ncu가 타겟
#   단, ncu는 모든 커널을 replays 하므로 실제 실행 시간보다 훨씬 느림
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_DECODE" = "0" ]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "Step 3: ncu — LM Decode steady-state"
    echo "        타겟: Decode/step_010  (이중 push 계층 필터)"
    echo "        결과: ncu_decode_v8.csv"
    echo "        예상 소요: 10~20분"
    echo "        ✅ v7에서 이미 성공 확인됨 (재실행 필요 시만)"
    echo "════════════════════════════════════════════"

    export HF_HOME=/home/ice401/.cache/huggingface
    sudo -E $NCU \
        --nvtx \
        --nvtx-include "Decode/step_010" \
        --metrics "$LTS_METRICS" \
        --csv \
        --force-overwrite \
        $VENV_PYTHON "$SCRIPT" --mode ncu_single_run \
        > "$RESULTS_DIR/ncu_decode_v8.csv" 2>&1

    echo "[Step 3 완료] → $RESULTS_DIR/ncu_decode_v8.csv"
    wc -l "$RESULTS_DIR/ncu_decode_v8.csv" 2>/dev/null && true
else
    echo "[Step 3 스킵] — 기존 ncu_decode_v7.csv 사용"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 4: ncu — LM Prefill
#
# 목표: prefill (seq=3086) 에서 실제 DRAM bytes 측정
#   - compute-bound → 실제 DRAM << 이론 (가중치 tile reuse)
#   - L2 hit rate 높을 것 → GEMM은 tile 단위로 L2에 남음
#   - 이론: 15.168 GB / 1,437 ms = 10.6 GB/s (이것이 실제 BW보다 클 수도)
#   - NVTX: push("Phase") → push("LM_Prefill") → ncu "Phase/LM_Prefill"
#
# ⚠️ 소요시간 주의: seq=3086 prefill 커널 수가 많아 ncu replay ~30~90분
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_PREFILL" = "0" ]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "Step 4: ncu — LM Prefill"
    echo "        타겟: Phase/LM_Prefill  (이중 push 계층 필터)"
    echo "        결과: ncu_lm_prefill_v2.csv"
    echo "        ⚠️  예상 소요: 30~90분 (seq=3086 replay heavy)"
    echo "════════════════════════════════════════════"

    export HF_HOME=/home/ice401/.cache/huggingface
    sudo -E $NCU \
        --nvtx \
        --nvtx-include "Phase/LM_Prefill" \
        --metrics "$LTS_METRICS" \
        --csv \
        --force-overwrite \
        $VENV_PYTHON "$SCRIPT" --mode ncu_single_run \
        > "$RESULTS_DIR/ncu_lm_prefill_v2.csv" 2>&1

    echo "[Step 4 완료] → $RESULTS_DIR/ncu_lm_prefill_v2.csv"
    wc -l "$RESULTS_DIR/ncu_lm_prefill_v2.csv" 2>/dev/null && true
else
    echo "[Step 4 스킵]"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 5: ncu — Vision Encoder
#
# 목표: VE (Qwen2.5-VL ViT) 에서 실제 DRAM bytes 측정
#   - compute-bound (다중 카메라 patch → attention O(n²))
#   - 가중치 1.153 GB / 738 ms → 이론 1.6 GB/s (매우 낮음)
#   - 실제 DRAM << 이론: Tensor Core가 포화 상태
#   - NVTX: push("Phase") → push("Vision_Encoder") → ncu "Phase/Vision_Encoder"
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_VE" = "0" ]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "Step 5: ncu — Vision Encoder"
    echo "        타겟: Phase/Vision_Encoder  (이중 push 계층 필터)"
    echo "        결과: ncu_ve_v2.csv"
    echo "        예상 소요: 15~30분"
    echo "════════════════════════════════════════════"

    export HF_HOME=/home/ice401/.cache/huggingface
    sudo -E $NCU \
        --nvtx \
        --nvtx-include "Phase/Vision_Encoder" \
        --metrics "$LTS_METRICS" \
        --csv \
        --force-overwrite \
        $VENV_PYTHON "$SCRIPT" --mode ncu_single_run \
        > "$RESULTS_DIR/ncu_ve_v2.csv" 2>&1

    echo "[Step 5 완료] → $RESULTS_DIR/ncu_ve_v2.csv"
    wc -l "$RESULTS_DIR/ncu_ve_v2.csv" 2>/dev/null && true
else
    echo "[Step 5 스킵]"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 6: ncu — Flow (Action Expert DiT, 65-step ODE)
#
# 목표: Flow matching 단계 실제 DRAM bytes 측정
#   - 65 ODE steps × (action_in_proj + expert + action_out_proj) forward
#   - 동일 가중치 반복 재사용 → L2 hit rate 높을 것으로 예상
#   - NVTX: push("FlowODE") + push("FlowStep") → ncu "FlowODE/FlowStep"
#
# ★ "Phase/Flow"를 쓰지 않는 이유 (v2 실패 원인 확정):
#   Flow 실행 시점에 Phase(Decode)/Decode_all이 스택에 살아있음.
#   on_generate_end()가 sample_trajectories... 반환 AFTER Flow이기 때문.
#   → Phase를 또 push 시 NVTX stack = Phase/Decode_all/Phase/Flow
#   → ncu "Phase/Flow" 필터: FIRST Phase의 direct child = Decode_all → 매칭 실패
#   → v3에서 FlowODE/FlowStep으로 이름 변경, v3 파일명 사용
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_FLOW" = "0" ]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "Step 6: ncu — Flow (diffusion, 65-step ODE)"
    echo "        타겟: FlowODE/FlowStep  (Phase 충돌 방지 고유 이름)"
    echo "        결과: ncu_flow_v3.csv"
    echo "        예상 소요: 20~40분 (65 ODE steps × replay)"
    echo "        ★ v3: Phase→FlowODE, Flow→FlowStep (NVTX 스택 충돌 수정)"
    echo "════════════════════════════════════════════"

    export HF_HOME=/home/ice401/.cache/huggingface
    sudo -E $NCU \
        --nvtx \
        --nvtx-include "FlowODE/FlowStep" \
        --metrics "$LTS_METRICS" \
        --csv \
        --force-overwrite \
        $VENV_PYTHON "$SCRIPT" --mode ncu_single_run \
        > "$RESULTS_DIR/ncu_flow_v3.csv" 2>&1

    echo "[Step 6 완료] → $RESULTS_DIR/ncu_flow_v3.csv"
    wc -l "$RESULTS_DIR/ncu_flow_v3.csv" 2>/dev/null && true
else
    echo "[Step 6 스킵]"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 7: 분석 및 최종 BW 표
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "Step 7: 결과 분석"
echo "════════════════════════════════════════════"

if [ -f "$ANALYSIS" ]; then
    $VENV_PYTHON "$ANALYSIS" --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/step7_analysis.log"
else
    echo "[WARNING] $ANALYSIS 없음. 분석 스크립트를 생성하세요."
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "전체 완료. 결과 파일 목록:"
ls -lh "$RESULTS_DIR"/*.csv "$RESULTS_DIR"/*.json 2>/dev/null || true
echo "═══════════════════════════════════════════════════"
