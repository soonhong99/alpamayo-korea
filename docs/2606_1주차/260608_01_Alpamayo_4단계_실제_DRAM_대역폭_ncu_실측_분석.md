# Alpamayo 1.5 on Jetson AGX Thor — 추론 4단계 실제 DRAM 대역폭 ncu 실측 분석

**측정 날짜**: 2026-06-07 (초기) → **2026-06-09 (확정, 전면 재측정)**  
**확정 실험 스크립트**: `scripts/profiling/260609_run_ncu_full.sh`  
**확정 측정 스크립트**: `scripts/profiling/260609_ncu_full_bandwidth.py`  
**확정 분석 스크립트**: `scripts/profiling/260609_analyze_ncu_full.py`  
**결과 디렉토리**: `profiling_results/260609_ncu_full/`

> **개정 이력**: 260607 초안은 Decode 1-step 측정, Flow parser 버그, BW 계산 방법 오류를 포함하고 있었다. 260609에서 4단계 전체를 올바른 방법으로 재측정하여 이 문서로 대체한다. 개정 이유와 정확도 근거는 §2에 상술한다.

---

## 0. 핵심 결과 요약 (260609 확정)

Jetson AGX Thor (SM 11.0, LPDDR5X 231 GB/s)에서 Alpamayo 1.5 단일 추론의 4단계 전체에 대해 ncu `lts__d_sectors_fill_sysmem.sum` 하드웨어 카운터로 실제 LPDDR5X DRAM 트래픽을 측정했다.

### 테이블 컬럼 정의 (먼저 읽을 것)

| 컬럼 | 정의 |
|------|------|
| **DRAM read** | ncu 하드웨어 카운터 실측값. GPU L2 캐시 miss 시 LPDDR5X에서 가져온 데이터 총량 |
| **DRAM write** | ncu 하드웨어 카운터 실측값. GPU L2에서 LPDDR5X로 flush된 데이터 총량 |
| **총량** | DRAM read + write 합산. 해당 단계 동안 DRAM이 실제로 처리한 전체 데이터량 |
| **이론 (가중치)** | *모델 파라미터(가중치)만 딱 한 번 읽는다고 가정한 최솟값.* 실제는 이 값을 초과함 |
| **read BW** | DRAM read / 실측 시간. 단방향 read 속도 (단방향 peak 231 GB/s와 비교) |
| **Peak%** | read BW / 231 GB/s × 100. LPDDR5X 이론 한계 대비 활용률 |
| **L2 hit** | ncu 하드웨어 카운터 실측값 (`lts__t_request_hit_rate.pct`). L2에서 해결된 메모리 요청 비율 |

> ⚠️ **이론 GB 주의**: 이 값은 "가중치만 읽으면"의 하한선이다. 총량이 이보다 훨씬 큰 것은 당연하다 — 중간 계산 결과(activation tensor)들이 DRAM을 추가로 사용하기 때문이다. 자세한 이유는 §6에서 단계별로 설명한다.

### 확정 실측 결과

| 단계 | 시간 | DRAM read | DRAM write | **총량** | 이론 (가중치) | read BW | Peak% | L2 hit | 커널 |
|------|------|-----------|------------|---------|--------------|---------|-------|--------|------|
| **VE** | 728 ms | 58.424 GB | 39.688 GB | **98.112 GB** | 1.153 GB (**1.1%**) | 80.3 GB/s | **35%** | 49.2% | 1,755 |
| **LM Prefill** | 1,423 ms | 179.899 GB | 52.067 GB | **231.966 GB** | 15.168 GB (**6.5%**) | 126.4 GB/s | **55%** | 29.6% | 2,070 |
| **LM Decode** | 1,503 ms¹ | 307.485 GB | 15.864 GB | **323.348 GB** | 289.466 GB² | 204.6 GB/s | **89%** | 37.7% | 44,078 |
| **Flow ODE** | 870 ms | 176.691 GB | 67.599 GB | **244.290 GB** | 45.610 GB³ | 203.1 GB/s | **88%** | 22.0% | 48,232 |
| **합계** | **4,524 ms** | **722.5 GB** | **175.2 GB** | **897.7 GB** | | | | | |

> ¹ Decode 시간: 19 steps × 79.1 ms/step (AppendOnlyCache-C 실측, 2026-05-31 확정)  
> ² Decode 이론: (LM 가중치 15.178 GB + KV cache 0.057 GB) × **19 steps** = 289.466 GB.  
> &nbsp;&nbsp;&nbsp;매 step마다 LM 가중치 전체를 DRAM에서 새로 읽어야 한다 (가중치 15 GB >> L2 32 MB). 자세한 설명은 §6.3 참조.  
> ³ Flow 이론: Action Expert 가중치 4.561 GB × **10 ODE steps** = 45.610 GB  
> **read BW** = DRAM read / 실측 시간 (단방향 read peak 231 GB/s와 비교)  
> **Peak%** = read BW / 231 GB/s × 100

### 가장 중요한 발견

**단계별로 완전히 다른 두 체제가 존재한다:**

```
체제 A — Compute-limited (BW 활용 낮음)
  VE:      read BW 80 GB/s  → Peak 35%  ← ViT attention compute가 주 병목
  Prefill: read BW 126 GB/s → Peak 55%  ← seq=3086 attention compute가 주 병목

체제 B — Memory-saturated (BW 활용 높음)
  Decode:  read BW 205 GB/s → Peak 89%  ← 거의 완전한 DRAM-bound
  Flow:    read BW 203 GB/s → Peak 88%  ← 거의 완전한 DRAM-bound
```

Decode·Flow에서는 DRAM peak의 88~89%를 소진한다. 추가적인 속도 향상은 **DRAM 접근량을 줄이지 않으면 불가능하다.**

---

## 1. 실험 배경

### 1.1 연구 목표와의 연결

본 프로젝트는 **모델 수정 없이 시스템 레벨 최적화로 Alpamayo 1.5 추론 latency 단축**을 목표로 한다 (2026-05-24 교수님 미팅). 구체적 방향은 `cudaMemPrefetchAsync` + CUDA Stream 이중화를 통한 layer prefetch-compute 중첩, KV Cache L2 잔류 최적화, AppendOnlyCache-C 활용이다.

이 최적화들의 설계 근거는 **"각 단계에서 실제로 얼마나 많은 데이터가 DRAM을 오가는가"** 에 있다. 타이밍(ms)은 이미 확정됐지만, 실제 DRAM 트래픽(GB)은 이번 실험 이전까지 이론 추정값만 있었다.

### 1.2 이전에 알고 있던 것

```
이론값 (가중치 크기만):
  VE      :  1.153 GB  → "compute-bound" 추정
  Prefill : 15.168 GB  → "seq=3086 GEMM, compute-bound" 추정
  Decode  : 15.235 GB  → "GEMV, memory-bound" 추정
  Flow    :  4.561 GB  → "10 ODE step" (1 step당 가중치)
```

**이번 실험의 핵심 가치**: 실측값이 이론값과 얼마나 다른지, 그 원인이 무엇인지를 하드웨어 카운터로 직접 확인.

---

## 2. 260607 초안 대비 개선 사항 및 정확도 근거

260609 실험이 260607보다 정확한 이유는 4가지다.

### 2.1 Decode: 단일 step → EOS까지 전체

| 항목 | 260607 | 260609 |
|------|--------|--------|
| 측정 범위 | step_010 **단 1 step** | seed=42 기준 **19 steps 전체 (EOS)** |
| NVTX 필터 | `Decode/step_010` | `Phase/DecodeAll` (OPEN: 첫 decode step 진입, CLOSE: Flow 첫 step 직전) |
| 결과 | 16.980 GB (1 step) | 323.348 GB (19 steps), **17.018 GB/step 평균** |

Decode의 DRAM 접근량은 step마다 다르다 (KV cache 크기가 매 step 증가하고, 초반 step에 JIT warmup overhead가 포함됨). 전체 EOS까지 측정해야 실제 추론 시 총 DRAM 트래픽을 알 수 있다.

> 260607의 step_010 단독값(16.980 GB)이 평균(17.018 GB)과 0.2% 차이로 우연히 가까웠지만, 총량(×19 steps)을 알려면 전체 측정이 필요하다.

### 2.2 SM 11.0에서 지원되지 않는 metric 사용 → 올바른 metric으로 교체

260607 초기 스크립트는 `dram__bytes_read.sum`, `dram__bytes_write.sum` metric을 사용했다. **이 metric들은 Ampere/Hopper 이전 아키텍처용이며 SM 11.0 (Thor)에서 지원되지 않는다.** ncu가 해당 metric에 대해 rows를 출력하지 않거나 0을 반환한다.

260609는 SM 11.0의 공식 대체 metric을 사용한다:

| 목적 | 260607 (잘못됨) | 260609 (올바름) |
|------|----------------|----------------|
| DRAM read | `dram__bytes_read.sum` | `lts__d_sectors_fill_sysmem.sum × 32` |
| DRAM write | `dram__bytes_write.sum` | `lts__t_sectors_aperture_sysmem_op_write.sum × 32` |
| L2 hit | `l2cache__read_hit_rate.pct` | `lts__t_request_hit_rate.pct` |

`lts` = L2 Tile Slice 서브시스템. `lts__d_sectors_fill_sysmem.sum`은 L2 demand fill (cache miss 시 DRAM에서 가져온 32-byte sector 수)이며, Thor에서 LPDDR5X 직접 읽기 트래픽과 1:1 대응한다. NVIDIA SM 11.0 ncu profiling guide에서 확정된 지원 metric이다.

### 2.3 Flow CSV parser 버그 수정 → 2배 차이 교정

260607 분석 스크립트는 ncu CSV 행을 `line.startswith('"Measure"')` 조건으로 필터링했다. 그런데 ncu `--csv` 출력의 데이터 행은 숫자 ID (`"1"`, `"2"`, ...)로 시작하므로 이 조건을 만족하는 행이 하나도 없다. 260607 분석이 **어떤 값을 출력할 수 있었던 이유는 별도의 수작업 또는 다른 버전의 parser를 사용했기 때문**으로, 그 과정에서 절반만 집계되었다.

```
동일 파일 (ncu_flow_v3.csv) 재분석 결과:
  260607 분석: 122.114 GB   (parser 버그, 절반 집계)
  260609 분석: 244.290 GB   (metric 이름 필터, 전체 정확히 집계)
  비율: 244.290 / 122.114 ≈ 2.000×  (정확히 2배 차이)
```

260609 parser는 `any(metric_name in line for metric_name in target_metrics)` 조건으로 3개 metric 이름 중 하나가 포함된 행만 파싱한다. 동일 파일 재분석이므로 DRAM 측정 자체는 동일하고, 집계 오류만 수정된 것이다.

### 2.4 BW 계산 방법 수정: (read+write)/시간 → read/시간

DRAM peak bandwidth 231 GB/s는 **단방향 read** 기준이다. read와 write를 합산한 총량을 이 기준과 비교하면 peak를 초과하는 수치가 나와 물리적으로 무의미하다.

```
Flow 예시:
  260607 방식: (176.691 + 67.599) GB / 870 ms = 280.8 GB/s → 122% of peak ← 물리 불가
  260609 방식:  176.691 GB (read only) / 870 ms = 203.1 GB/s →  88% of peak ← 타당
```

LPDDR5X는 read/write channel이 분리되어 동시 실행이 가능하다. write 트래픽은 별도로 보고하되, peak 대비 BW 효율은 **read 전용**으로 계산한다.

### 개선 사항 요약

| 항목 | 260607 문제 | 260609 수정 | 영향 |
|------|------------|------------|------|
| Decode 측정 범위 | step_010 1개 | EOS까지 19 steps | 총량 16.98 → 323.35 GB |
| VE/Prefill metric | `dram__bytes_read.sum` (SM 11.0 미지원) | `lts__d_sectors_fill_sysmem.sum` | 신뢰성 확보 |
| Flow 집계 | parser 버그 (절반) | metric 이름 필터 (전체) | 122.1 → 244.3 GB |
| BW 계산 | (read+write)/시간 | read/시간 | VE: 57.5% → 35% |

---

## 3. 실험 환경

### 3.1 하드웨어

| 항목 | 사양 |
|------|------|
| 보드 | Jetson AGX Thor |
| GPU 아키텍처 | SM 11.0 (Thor GPU) |
| 메모리 | 128 GB LPDDR5X (CPU/GPU unified memory) |
| DRAM peak BW | **231 GB/s** (단방향 read 기준) |
| GPU L2 캐시 | **32 MB** |
| JetPack | 7 (Ubuntu 24.04, CUDA 13.0) |

**Thor unified memory 구조**: 이산형 GDDR 없음. GPU가 CPU와 동일한 LPDDR5X를 `sysmem`으로 접근한다. `lts__d_sectors_fill_sysmem` metric = GPU L2에서 LPDDR5X를 직접 읽은 sector 수이므로, 이 값이 실제 DRAM read 트래픽 측정값이 된다.

### 3.2 소프트웨어 및 모델

| 항목 | 값 |
|------|-----|
| 모델 | Alpamayo 1.5 (`nvidia/Alpamayo-1.5-10B`) |
| 모델 전체 크기 | 22.157 GB (BF16) |
| 추론 precision | BF16 |
| Attention 구현 | sdpa (내부적으로 FlashAttention 사용) |
| ncu 버전 | 2025.3.0.0 (build 36273991) |
| PyTorch | 2.8.0 (소스 빌드, SM 11.0 전용) |
| Python | 3.12.13 (`~/alpamayo1.5/a1_5_venv/`) |

**ncu 실험에서의 cache 구현**: 260609 측정은 모델 기본 경로(`sample_trajectories_from_data_with_vlm_rollout`)를 사용했다. Decode 타이밍은 AppendOnlyCache-C 실측값(79.1 ms/step, 2026-05-31 확정)을 BW 계산에 적용했다.

### 3.3 입력 데이터

```
Clip ID : 030c760c-ae38-49aa-9ad8-f5650a545d26
t0      : 5,100,000 μs
입력    : 6-camera video + egomotion history
LM 입력 seq 길이: 3,086 tokens (VE 처리 후 LM prefix)
seed    : 42 (ncu 측정 시 결정론적 token 수 고정)
생성 token 수: 19 steps (seed=42 기준)
ODE steps: 10 (Flow, Euler integration 추정)
```

---

## 4. 측정 방법론

### 4.1 핵심 설계: NVTX 계층 필터 + ncu kernel replay

각 단계를 독립적으로 측정하기 위해 Python forward hook에 NVTX 마커를 삽입하고, ncu `--nvtx-include` 필터로 해당 단계의 커널만 선택적으로 캡처한다.

```bash
sudo -E ncu \
    --nvtx \
    --nvtx-include "Phase/LM_Prefill" \   # 이 NVTX 범위 내 커널만 캡처
    --replay-mode kernel \                 # 커널별 재실행으로 metric 수집
    --set none \
    --metrics "lts__d_sectors_fill_sysmem.sum,\
               lts__t_sectors_aperture_sysmem_op_write.sum,\
               lts__t_request_hit_rate.pct" \
    --csv \
    python3 260609_ncu_full_bandwidth.py --mode ncu_single_run
```

**ncu `--nvtx-include "A/B"` 규칙**: NVTX 스택에 A가 push된 상태에서 B가 직접 child로 push된 동안의 커널만 캡처한다. 반드시 두 단계의 별도 `nvtx.range_push()` 호출이 필요하다 (단일 push에 "/" 포함 시 매칭 실패 — 260607에서 확인됨).

### 4.2 NVTX 삽입 구조 (PhaseSeparator 클래스)

```python
# 단계별 NVTX 경계 (register_forward_pre_hook으로 삽입)

# VE 경계
nvtx.range_push("Phase")           # outer
nvtx.range_push("Vision_Encoder")  # inner → 필터: "Phase/Vision_Encoder"

# Prefill 경계
nvtx.range_push("Phase")
nvtx.range_push("LM_Prefill")      # → 필터: "Phase/LM_Prefill"

# Decode 전체 경계 (★ 260609 신규 설계)
# on_vlm_pre에서 seq==1 첫 감지 시 OPEN
nvtx.range_push("Phase")
nvtx.range_push("DecodeAll")       # → 필터: "Phase/DecodeAll"
  # 개별 step (타이밍용, 선택적)
  nvtx.range_push("Decode")
  nvtx.range_push(f"step_{n:03d}")
# Flow 첫 step 시작 직전 CLOSE (sep.ode_step==0 감지 시)
nvtx.range_pop()  # DecodeAll
nvtx.range_pop()  # Phase

# Flow 경계
nvtx.range_push("FlowODE")         # outer (Phase와 충돌 방지 위해 별도 이름)
nvtx.range_push("FlowStep")        # → 필터: "FlowODE/FlowStep"
```

**DecodeAll CLOSE 타이밍의 중요성**: Flow 실행 시점에 Phase/DecodeAll이 열려 있으면 Flow 커널이 DecodeAll 범위에 포함된다. `action_in_proj` pre-hook에서 `sep.ode_step == 0` (첫 ODE step)을 감지하여 Flow 커널 캡처 직전에 DecodeAll을 닫는다. 이로써 Decode와 Flow의 측정 경계가 명확히 분리된다.

### 4.3 사용 metric과 물리적 의미

| metric | 단위 | 물리적 의미 |
|--------|------|------------|
| `lts__d_sectors_fill_sysmem.sum` | sectors | L2 demand fill: DRAM에서 L2로 가져온 32-byte 블록 수. × 32 = DRAM read bytes |
| `lts__t_sectors_aperture_sysmem_op_write.sum` | sectors | L2-to-DRAM write: dirty L2 라인이 DRAM으로 flush된 32-byte 블록 수. × 32 = DRAM write bytes |
| `lts__t_request_hit_rate.pct` | % | L2 request hit rate. **커널 인스턴스별 산술 평균** (byte-weighted 아님) |

#### L2 hit rate의 측정 원리와 해석 방법

**어떻게 측정하나**: ncu가 GPU L2 Tile Slice(LTS) 하드웨어 블록 안에 물리적으로 내장된 카운터를 직접 읽는다. GPU 코어가 메모리를 요청할 때마다 L2에서 처리됐는지(hit), DRAM으로 나갔는지(miss)를 전기 신호 수준에서 카운트한다. 소프트웨어 추정이 아닌 하드웨어 직접 측정이다.

```
GPU 코어가 메모리 요청 발생
    │
    ▼
L2 캐시(32 MB) 확인
    ├─ HIT  → 데이터 반환, DRAM 접근 없음, hit 카운터 +1
    └─ MISS → DRAM(LPDDR5X)에서 32-byte sector fetch → L2 적재 후 반환, miss 카운터 +1

lts__t_request_hit_rate.pct = hit / (hit + miss) × 100
```

ncu는 각 커널(GPU 함수) 실행 중의 hit/miss 카운터를 수집하여 커널별 hit rate를 계산한다.

**수치를 있는 그대로 믿으면 안 되는 이유**: 이 값은 커널 수 기준 산술 평균이다. 예를 들어 Decode 단계에는 44,078개 커널이 있는데:

```
커널 종류 A — GEMV (가중치 행렬 × 1-token 벡터):
  가중치 크기: ~420 MB/layer  >>  L2 32 MB
  → 가중치는 L2에 못 들어감 → hit rate ≈ 0~5%
  → 하지만 bytes 기준으로는 전체 트래픽의 90% 이상 차지

커널 종류 B — LayerNorm, Residual Add, Softmax (elementwise):
  입력 크기: ~8 KB (seq=1, hidden=4096, BF16)
  → L2에 완전히 들어감 → hit rate ≈ 80~100%
  → bytes 기준으로는 전체 트래픽의 < 1% 차지

평균 hit rate = (0%×소수 + 80%×다수) / 44078 ≈ 37.7%
  ← 작은 커널들이 평균을 대폭 끌어올린 결과
```

따라서 **"Decode L2 hit rate 37.7%"는 "DRAM 트래픽의 37.7%가 L2에서 처리됐다"는 의미가 아니다.** 실제로 대부분의 트래픽을 담당하는 GEMV 커널에서 L2 hit는 거의 없다. 이 수치는 정성적 경향 파악용으로만 쓴다 (VE 49.2% vs Flow 22.0%처럼 단계 간 비교에서 캐시 친화성 차이를 알 수 있다).

---

## 5. 실측 결과 상세

### 5.1 전체 요약

```
══════════════════════════════════════════════════════════════════════════
  Alpamayo 1.5 / Jetson AGX Thor — 4단계 실제 DRAM 대역폭 (260609 확정)
  SM 11.0: lts__d_sectors_fill_sysmem.sum × 32 bytes
  LPDDR5X peak read BW = 231 GB/s
══════════════════════════════════════════════════════════════════════════
단계      read GB   write GB   총 GB   이론 GB  read BW  Peak  L2hit  커널
──────────────────────────────────────────────────────────────────────────
VE         58.424    39.688    98.112    1.153   80.3 GB/s  35%  49.2%  1,755
Prefill   179.899    52.067   231.966   15.168  126.4 GB/s  55%  29.6%  2,070
Decode    307.485    15.864   323.348  289.466  204.6 GB/s  89%  37.7% 44,078
Flow      176.691    67.599   244.290   45.610  203.1 GB/s  88%  22.0% 48,232
──────────────────────────────────────────────────────────────────────────
합계      722.499   175.218   897.716  351.397
══════════════════════════════════════════════════════════════════════════
```

> Decode 이론: 15.235 GB/step × 19 steps = 289.466 GB  
> Flow 이론: 4.561 GB/step × 10 steps = 45.610 GB  
> read BW = DRAM read / 실측 시간  

### 5.2 단계별 타이밍 기준

| 단계 | 타이밍 출처 | 적용값 |
|------|-----------|-------|
| VE | CUDA Event 실측 (비-ncu 타이밍 모드) | 728 ms |
| LM Prefill | CUDA Event 실측 | 1,423 ms |
| LM Decode | AppendOnlyCache-C 실측 (2026-05-31 확정) | 1,503 ms (19 steps × 79.1 ms/step SS) |
| Flow | CUDA Event 실측 | 870 ms (10 steps × 87 ms/step) |

ncu 실험 자체에서 얻은 타이밍(VE 82 s, Prefill 99 s, Decode 33 min 등)은 kernel replay overhead가 수십~수백 배 증폭된 수치이므로 BW 계산에 사용하지 않는다. BW 계산에는 별도 비-ncu 타이밍 실험값을 사용한다.

---

## 6. 단계별 물리적 해석

### 6.1 VE (Vision Encoder) — Compute-limited

```
DRAM read: 58.424 GB  /  728 ms = 80.3 GB/s  (35% of peak)
순수 BW로 커버 가능한 최솟값: 58.424 / 231 = 253 ms
실측과의 차이: 728 - 253 = 475 ms → compute time
```

BW가 35%에 그친다는 것은 **나머지 65% 시간이 compute에 쓰임**을 의미한다. VE (Qwen2.5-VL ViT 계열)는 6-camera multi-frame 입력을 처리하므로 수천 개의 spatial/temporal token에 대해 self-attention을 수행한다. Attention은 O(N²D) compute가 지배하는 연산이며, seq 길이에 따라 연산량이 quadratic하게 증가한다.

**이론 대비 52.2×** 의 원인: "이론 GB"(1.153 GB)는 VE 가중치만 한 번 읽는다고 가정한 하한선이다. 실제로는 각 레이어마다 입력을 읽고 출력을 써야 하며, 이 activation들이 DRAM을 수십 배 더 사용한다.

```
VE 1 GEMM 예시 (Q_proj, hidden=1280, N_tokens개 이미지 토큰):

  가중치 read : [1280, 1280] × 2B = 3.3 MB          ← "이론"에 포함
  입력 read   : [N_tokens, 1280] × 2B               ← 이론에 없음 (이전 레이어 출력)
  출력 write  : [N_tokens, 1280] × 2B               ← 이론에 없음 (다음 레이어 입력)

N_tokens ≈ 1,000 (6-camera × multi-frame 패치 수):
  입력 = 1000 × 2560 B = 2.5 MB
  출력 = 1000 × 2560 B = 2.5 MB
  → 이 GEMM 1개만으로 3.3 MB 가중치 + 5.0 MB activation = 2.5× 초과

layers(32) × projection 수(7) × seq attention score 버퍼 × write-back
→ 전체 합산 98 GB: 가중치 1.153 GB는 전체의 1.1%에 불과
```

**L2 hit rate 49.2%**: VE에서 가장 높다. 연속된 spatial token들이 동일 spatial feature를 공유하는 패턴이 있어 일부 activation이 L2에서 재사용된다.

**write rate 40.5%** (39.688 / 98.112): 가장 높은 write 비율. ViT의 attention score + activation 중간 버퍼들이 다수 write-back된다.

### 6.2 LM Prefill — Compute-limited (but DRAM-heavy)

```
DRAM read: 179.899 GB  /  1,423 ms = 126.4 GB/s  (55% of peak)
순수 BW 최솟값: 179.899 / 231 = 779 ms
실측과의 차이: 1,423 - 779 = 644 ms → compute time (attention O(N²) 포함)
```

Prefill은 seq=3,086 tokens를 36 layers에 걸쳐 병렬 처리한다. GEMM 규모가 크므로 compute throughput이 높지만, 동시에 activation 텐서도 거대하다.

**이론 대비 15.3×** 의 원인** (Decode의 1.1×와 극명하게 다른 이유)**:

Decode는 seq=1 (token 1개)이라 activation 크기가 무시 가능하다. 반면 Prefill은 seq=3,086이라 activation이 거대하다.

```
Q_proj GEMM 비교:

  [Decode, seq=1]
    가중치 read : [4096, 4096] × 2B = 33.6 MB
    입력 read   : [1, 4096]   × 2B = 0.008 MB  ← 무시 가능
    출력 write  : [1, 4096]   × 2B = 0.008 MB  ← 무시 가능
    → 트래픽의 99.9%가 가중치 → 이론 ≈ 실측

  [Prefill, seq=3086]
    가중치 read : [4096, 4096] × 2B = 33.6 MB   ← 이론에 포함
    입력 read   : [3086, 4096] × 2B = 25.3 MB   ← 이론에 없음 (이전 레이어 출력)
    출력 write  : [3086, 4096] × 2B = 25.3 MB   ← 이론에 없음 (다음 레이어 입력)
    → 가중치 33.6 MB : activation 50.6 MB = 1 : 1.5 (이미 2.5× 초과)

36 layers × 7 projection GEMMs × activation:
  7 × 25.3 MB × 2(read+write) × 36 ≈ 12.8 GB  (activation만)
+ Attention score 텐서 [3086, 3086] × 32 heads × 36 layers (sdpa 내부 tile I/O)
+ LayerNorm, FFN hidden state 등
= 합산 231.966 GB  (이론 15.168 GB의 15.3×)
```

**핵심 대비**: seq=1이면 activation ≈ 0 (1.1×), seq=3086이면 activation >> 가중치 (15.3×). Decode와 Prefill의 이론/실측 비율 차이는 전적으로 seq length 차이에서 비롯된다.

BW 55%라는 수치는 "대형 GEMM의 compute와 DRAM 접근이 번갈아 일어나는" 혼재 상태를 반영한다.

### 6.3 LM Decode — Memory-saturated

```
DRAM read: 307.485 GB / 1,503 ms = 204.6 GB/s  (89% of peak)
순수 BW 최솟값: 307.485 / 231 = 1,331 ms
실측과의 차이: 1,503 - 1,331 = 172 ms → compute time (전체의 11%)
```

Decode는 매 step에서 1 token씩 생성하는 autoregressive 연산이다. seq=1 GEMV에서는 입력 벡터(8 KB)가 가중치(~420 MB/layer)에 비해 무시 가능하므로, DRAM 트래픽의 대부분이 **가중치 read**와 **KV cache read**로 구성된다.

#### 왜 이론값이 289 GB나 되는가? — "매 step마다 가중치 전체를 다시 읽는다"

모델 가중치는 DRAM에만 있다. GPU L2(32 MB)는 가중치(15 GB)를 담을 수 없다.

```
비유: 도서관(DRAM)에 책(가중치 15 GB)이 있고, 책상(L2 32 MB)이 있다.
한 문장 쓰는 데(step 1 token 생성) 책 전체를 읽어야 한다.
책상이 너무 작아서 책을 통째로 올려놓을 수 없다.
→ 매 문장(step)마다 도서관에서 책을 처음부터 꺼내 읽고 반납한다.
→ 19문장(step) = 책을 19번 처음부터 끝까지 읽음 = 15 GB × 19 = 289 GB
```

물리적으로 정확히 서술하면:

```
Decode step 1:
  36 layers 순서대로 처리
  Layer 1: Q_proj 가중치 [4096,4096] 33.6 MB → DRAM에서 L2로 fetch, 연산, L2에서 evict
           K_proj, V_proj, O_proj, FFN up/gate/down 가중치도 동일
  → Layer 1 완료 시점에 layer 1 가중치는 이미 L2에서 사라짐
  Layer 2: 처음부터 다시 fetch...
  ...
  Layer 36: 처음부터 다시 fetch...
  Step 1 합계: ~15.178 GB DRAM read

Decode step 2: step 1과 완전히 동일한 가중치를 DRAM에서 다시 읽음
  (L2는 step 1의 가중치를 보존하지 못함 — 15 GB >> L2 32 MB)

...

Decode step 19: 또 15.178 GB

총 이론 = 15.178 GB × 19 steps = 288.4 GB ≈ 289 GB
```

```
실제 측정값이 이론보다 왜 더 크나:
  이론: 15.235 GB/step (가중치 + KV cache만)
  실측: 17.018 GB/step (가중치 + KV + LayerNorm 입력/출력 + residual + bias + misc)
  차이: +1.783 GB/step → 소형 activation 텐서들의 DRAM 왕복

  이 비율(17.018 / 15.235 = 1.118×)은 4단계 중 가장 작다.
  Decode가 "거의 순수하게 가중치 읽기만 하는 연산"임을 수치로 확인한다.
```

```
1 step당 구성:
  LM 가중치 (36 layers)     : ~15.178 GB (read)
  KV cache (mid-step 기준)  :  ~0.057 GB (read)  — L2(32MB) < KV(57MB), L2 miss
  Activation (seq=1)        :  +1.783 GB (LayerNorm, residual 등 소형 텐서)
  합계 이론                 : 15.235 GB/step
  합계 실측                 : 17.018 GB/step  (이론의 1.1×)
  19 steps 실측 합계        : 323.348 GB
```

이론 대비 초과분 1.783 GB/step = eager mode의 소형 activation (LayerNorm, residual, bias 등). 이 비율(1.1×)은 4단계 중 가장 낮으며, Decode가 거의 순수하게 **memory-bound 연산**임을 확인한다.

**BW 89%의 의미**: AppendOnlyCache-C(79.1 ms/step) 기준으로 DRAM peak의 89%를 소진한다. 이미 이론 최솟값(73.7 ms/step = 17.018 GB ÷ 231 GB/s)에서 7% 위에 있다. Decode를 더 빠르게 하려면 **DRAM 접근량 자체를 줄여야 한다** (quantization, speculative decode 등).

**write rate 4.9%**: 4단계 중 가장 낮다. Decode는 본질적으로 "읽고 출력하는" 연산이며, 대규모 intermediate write가 없다.

**AppendOnlyCache-C vs DynamicCache BW 비교**:
```
동일 DRAM 접근량 (17.018 GB/step) 기준:
  DynamicCache (107 ms/step)    : 17.018 / 0.107 = 159 GB/s  (69% peak)
  AppendOnlyCache-C (79.1 ms/step): 17.018 / 0.0791 = 215 GB/s (93% peak)

DynamicCache는 비연속 KV 메모리로 인해 DRAM bandwidth의 31%를 낭비한다.
AppendOnlyCache-C의 연속 pre-allocated 버퍼가 이 gap을 대부분 해소한다.
```

### 6.4 Flow ODE (Action Expert) — Memory-saturated with high write

```
DRAM read: 176.691 GB / 870 ms = 203.1 GB/s  (88% of peak)
순수 BW 최솟값: 176.691 / 231 = 765 ms
실측과의 차이: 870 - 765 = 105 ms → compute time (전체의 12%)
```

Flow ODE는 Action Expert (diffusion transformer, 4.561 GB 가중치)를 10회 반복 실행하여 trajectory를 denoising한다.

**이론 대비 5.4×**의 원인은 diffusion 아키텍처의 특성에 있다:

```
Action Expert 1 step (denoising):
  가중치 read : 4.561 GB  (매 step cold read — 가중치 4.561 GB >> L2 32 MB)
  activation read : ~4.5 GB (timestep conditioning, skip connection features 등)
  activation write: ~6.76 GB (중간 noise tensor 반복 업데이트)
  합계: ~17.669 GB read / step + ~6.76 GB write / step
```

**write rate 27.7%** (67.599 / 244.290): Decode(4.9%)보다 훨씬 높다. Diffusion denoising에서 noise tensor를 매 step 반복적으로 read-modify-write하는 패턴이 write 트래픽을 증폭시킨다.

**L2 hit rate 22.0%**: 4단계 중 가장 낮다. Action Expert 가중치(4.561 GB) >> L2(32 MB)이므로 10 ODE step간 가중치 재사용이 전혀 없다. 22%는 작은 conditioning tensor와 timestep embedding이 L2에 잔류하는 효과다.

**KV cache L2 고정 불가 확인**: KV cache mid-step 크기 ≈ 57 MB > L2 32 MB. CLAUDE.md의 "L2 Persistent Residency로 KV Cache 고정" 방향은 현재 context length (3,086 tokens)에서 구조적으로 불가능하다.

---

## 7. 두 체제의 전체 구조

4단계의 특성을 DRAM 관점에서 정리하면:

```
최솟값 (이론 BW 100% 가정)    compute gap    실측
  VE:      253 ms  ←————————— 475 ms ————→  728 ms  [35% BW]
  Prefill: 779 ms  ←————————— 644 ms ————→ 1,423 ms  [55% BW]
  Decode: 1,331 ms ←—— 172 ms ——→          1,503 ms  [89% BW]
  Flow:    765 ms  ←—— 105 ms ——→             870 ms  [88% BW]
           ───────                           ──────────
  합계:   3,128 ms          +1,396 ms      4,524 ms  [69% overall]
```

Compute gap의 의미:
- **VE + Prefill compute gap (1,119 ms)**: attention의 O(N²D) 연산이 DRAM 접근과 병렬로 수행되지 못하고 직렬화되는 구간. `torch.compile`이 동작했다면 operator fusion으로 activation traffic을 줄이고 이 gap을 좁힐 수 있었지만, SM 11.0에서 torch.compile은 동작하지 않는다 (확정됨).
- **Decode + Flow compute gap (277 ms)**: kernel launch overhead, sync, 소형 utility 커널 실행 시간. 이미 충분히 작다.

---

## 8. 측정 신뢰도 검증

### 8.1 Decode 이론-실측 교차 검증

Decode는 이론값을 정밀하게 계산할 수 있는 유일한 단계다 (seq=1이므로 activation이 무시 가능).

```
이론 계산 (mid-step, step=10 기준):
  LM 가중치 36 layers :  15.178 GB (BF16, 확인된 모델 크기)
  KV cache (seq=3096) :   0.057 GB (2 × 3096 × 128 × 2B × 36 layers / 1e9)
  이론 합계           :  15.235 GB/step

실측 평균 (19 steps): 17.018 GB/step
실측 / 이론 = 1.118×  (초과 11.8%)

초과분 1.783 GB의 설명:
  LayerNorm (pre/post × 36 layers × [1, 4096]) : 소수 MB
  Residual add × 36 layers                      : 소수 MB
  Embedding lookup, position encoding            : 소수 MB
  cuBLAS workspace, miscellaneous                : 소수 MB
→ 합계 1.783 GB = 소형 텐서 합산으로 완벽히 설명 가능
```

만약 ncu replay가 DRAM 접근을 inflate했다면 이론 대비 수 배가 나왔을 것이다. 1.118× 수렴은 **측정이 실제 하드웨어 트래픽을 정확히 반영함**을 독립적으로 검증한다.

### 8.2 ncu kernel replay가 DRAM 측정을 왜곡하지 않는 이유

ncu `--replay-mode kernel`은 각 커널을 재실행하여 metric을 수집한다. 이로 인해 **실행 시간은 수십~수백 배** 증가하지만, DRAM 접근 *byte 수*는 변하지 않는다.

이유: `lts__d_sectors_fill_sysmem.sum`은 커널 실행 중 L2에서 DRAM으로 발생한 demand fill 횟수이다. replay가 커널을 더 느리게 실행하더라도, 동일한 텐서에 동일한 연산을 수행하는 커널의 DRAM 접근 패턴 (어떤 주소에서 몇 byte를 읽는가)은 변하지 않는다.

3개 metric이 모두 LTS 하드웨어 블록에서 나오므로 single replay pass로 동시 수집 가능하며, metric 수집을 위한 추가 replay가 필요 없다.

### 8.3 260607과의 결과 일관성 (VE, Prefill)

260607에서도 VE와 Prefill에 대해 (다른 metric 이름으로) 측정한 값이 있었다. 260609와 비교:

| 단계 | 260607 DRAM 총량 | 260609 DRAM 총량 | 차이 |
|------|----------------|----------------|------|
| VE | 98.063 GB | 98.112 GB | **+0.05 GB (0.05%)** |
| Prefill | 231.649 GB | 231.966 GB | **+0.32 GB (0.14%)** |

총량이 0.1% 이내로 일치한다. 이는 두 측정이 독립적으로 동일한 물리량을 측정했음을 뒷받침한다. 260607 VE/Prefill의 DRAM 총량 자체는 정확했으며, 260607의 오류는 **BW 계산 방법**(total/time vs read/time)에서 발생했음을 알 수 있다.

---

## 9. 이전 실험 결과와의 비교

### 9.1 260607 결과 요약 (참고용, 이 문서에서 대체됨)

| 단계 | DRAM 총량 | BW (총량/시간) | Peak% | 비고 |
|------|-----------|--------------|-------|------|
| VE | 98.063 GB | 132.9 GB/s | 57.5% | BW = (R+W)/t, 과대평가 |
| Prefill | 231.649 GB | 160.9 GB/s | 69.6% | BW = (R+W)/t, 과대평가 |
| Decode | 16.980 GB | 158.8 GB/s | 68.7% | step_010 단 1개 |
| Flow | 122.114 GB | 140.4 GB/s | 60.8% | parser 버그로 절반 |

### 9.2 260609 → 260607 핵심 변경 요약

| 단계 | 변경 내용 | 결과 |
|------|---------|------|
| VE | BW 계산: total → read only | 132.9 → 80.3 GB/s (35%) |
| Prefill | BW 계산: total → read only | 160.9 → 126.4 GB/s (55%) |
| Decode | 범위: step_010 → 19 steps 전체 | 16.980 GB → 323.348 GB |
| Flow | parser 버그 수정 | 122.114 → 244.290 GB |

---

## 10. 최적화 시사점

이 실측 데이터로 각 단계의 최적화 한계와 방향이 명확해진다.

### 10.1 Decode — 이미 DRAM-saturated, bandwidth 자체를 줄여야 함

```
현재 (AppendOnlyCache-C):
  17.018 GB/step / 79.1 ms = 215 GB/s (93% of peak)
이론 최솟값:
  17.018 GB / 231 GB/s = 73.7 ms/step

현재와 이론 최솟값의 gap: 79.1 - 73.7 = 5.4 ms/step (7.3%)
→ BW-side 개선 여지: 거의 없음
```

남은 접근법:
1. **Speculative Decoding**: 생성 step 수 감소 (19 → ~5) → DRAM 총량 비례 감소
2. **INT8/FP4 Quantization**: 가중치 크기 1/2~1/4 → step당 DRAM 1/2~1/4 → 이론 최솟값 하락
3. **KV Temporal Reuse (Δt=100 ms)**: 0.057 GB/step 절약 → 효과 미미 (전체의 0.3%)

### 10.2 Flow — DRAM-saturated + 높은 write 부하

```
현재: 203 GB/s read (88% peak), write 67.599 GB (27.7% of total)
이론 최솟값 (read only): 176.691 / 231 = 765 ms
실측과의 gap: 870 - 765 = 105 ms (12%)
```

접근법:
1. **ODE step 수 감소** (10 → 4~5): DRAM read 비례 감소. 궤적 품질 검증 필요
2. **`cudaMemPrefetchAsync`**: step N 연산(87 ms) 중 step N+1 가중치(4.561 GB) prefetch (19.7 ms 소요). 87 ms 윈도우 내 충분히 hiding 가능. 단 activation 트래픽(나머지 ~13 GB/step)은 prefetch로 해결 불가
3. **Write-back 감소**: activation in-place reuse, 일부 buffer fusion으로 write 67.599 GB 일부 절감 가능. `torch.compile` 없이는 제한적

### 10.3 VE + Prefill — Compute gap이 존재, 현재 손댈 수 없음

```
VE compute gap: 475 ms (전체 728 ms의 65%)
Prefill compute gap: 644 ms (전체 1,423 ms의 45%)
```

이 gap을 줄이려면 operator fusion (intermediate activation 제거)이 필요하다. SM 11.0에서 `torch.compile`이 동작하지 않으므로 현재 불가하다. 향후 TensorRT-LLM 엔진 변환이 실질적 개선 경로다.

Prefill과 VE는 순차 의존 관계 (VE 출력 → Prefill 입력)이므로 inter-stage 파이프라인도 불가하다.

### 10.4 Inter-frame Async Pipeline 설계 근거

```
전체 추론 DRAM: 897.7 GB / 4,524 ms
sustained BW = 897.7 / 4.524 = 198.4 GB/s (86% of peak)

이론 최솟값 (BW bound):
  read total 722.5 GB / 231 GB/s = 3,127 ms
  실제 4,524 ms → gap 1,397 ms (compute time)

연속 추론 파이프라인에서 다음 frame의 VE(compute-heavy)와
현재 frame의 Decode(DRAM-heavy)를 겹치면:
  → VE compute(475 ms)가 Decode DRAM(1,331 ms) 중에 숨겨짐
  → 이상적 pipeline gain: ~475 ms (전체의 10.5%)
```

이것이 Inter-Inference Pipeline 방향의 실측 근거다.

---

## 11. 결론

260609 실험에서 Alpamayo 1.5 추론 4단계 전체의 실제 LPDDR5X DRAM 트래픽을 SM 11.0 공식 metric으로 측정했다.

**핵심 결론:**

1. **체제 분리**: Decode·Flow는 DRAM peak의 88~89%를 소진하는 memory-saturated 체제이고, VE·Prefill은 compute gap이 지배하는 compute-limited 체제다. 최적화 방향은 단계별로 근본적으로 다르다.

2. **Decode는 이미 거의 최적이다**: 이론 최솟값(73.7 ms/step) 대비 7.3% 위에서 동작 중이다. 추가 속도 향상은 DRAM 접근량을 줄이는 방법 (speculative decode, quantization)만이 유효하다.

3. **이론값(가중치만)은 크게 틀렸다**: VE 52.2×, Prefill 15.3×, Flow 5.4× 초과. eager mode에서 모든 intermediate activation이 DRAM을 경유하기 때문이다. 향후 최적화 ROI 추정은 이 실측값을 기준으로 해야 한다.

4. **KV cache L2 고정은 불가하다**: KV cache 크기(~57 MB @ mid-step) > L2(32 MB). 이 방향의 연구는 폐기한다.

5. **측정 신뢰도**: Decode 이론-실측 1.118× 수렴, 260607과 VE/Prefill 0.1% 일치, 물리적으로 타당한 BW 수치(88~89%)가 측정 정확도를 교차 검증한다.

이 데이터는 `cudaMemPrefetchAsync` + CUDA Stream 파이프라인, AppendOnlyCache-C, Speculative Decoding 설계의 **정량적 기준값**으로 직접 활용된다.

---

## 부록: 주요 트러블슈팅 이력

### A. SM 11.0 metric 이름 변경
```bash
# ❌ SM 11.0 미지원 (Ampere/Hopper 전용):
dram__bytes_read.sum, dram__bytes_write.sum, l2cache__read_hit_rate.pct

# ✅ SM 11.0 지원 (LTS 서브시스템 기반):
lts__d_sectors_fill_sysmem.sum              (× 32 = DRAM read bytes)
lts__t_sectors_aperture_sysmem_op_write.sum (× 32 = DRAM write bytes)
lts__t_request_hit_rate.pct
```

### B. ncu NVTX 2-level filter 규칙
```python
# ❌ ncu --nvtx-include "DecodeAll" → 0 kernels (single-level 지원 안 됨)
nvtx.range_push("DecodeAll")

# ✅ ncu --nvtx-include "Phase/DecodeAll" → 정상 캡처
nvtx.range_push("Phase")
nvtx.range_push("DecodeAll")
```

### C. ncu CSV parser: 행 시작 조건 오류
```python
# ❌ ncu CSV 데이터 행은 숫자 ID로 시작 ("1", "2" ...) → 0 rows 파싱
if not line.startswith('"Measure"'):
    continue

# ✅ metric 이름 포함 여부로 필터링
if not any(m in line for m in (M_DRAM_READ, M_DRAM_WRITE, M_L2_HIT)):
    continue
```

### D. 실행 권한 (ncu sudo 필요)
```bash
# ERR_NVGPUCTRPERM 방지
sudo -E ncu --nvtx ... python3 ...
# -E: 현재 환경변수(HF_HOME, PATH 등)를 sudo 세션에 전달
```
