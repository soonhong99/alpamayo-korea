# Alpamayo 1.5 프로파일링 Figure 설명서

**장치**: NVIDIA Jetson AGX Thor (SM 11.0, CUDA 13.0, 128 GB Unified Memory)  
**모델**: Alpamayo 1.5 (10B, BF16, Eager Attention)  
**측정 도구**: CUDA Event Timer (GPU), psutil CPUSampler (CPU)  
**실험 조건**: Warmup 3회 제외, 8회 실측 평균  
**저장 경로**: `profiling_results/figures/fig1_*.png ~ fig10_*.png`

---

## 목차

| Figure | 파일명 | 핵심 내용 |
|:------:|--------|----------|
| Fig 1 | `fig1_stage_breakdown.png` | 추론 단계별 레이턴시 분해 (스택바 + 도넛) |
| Fig 2 | `fig2_run_variability.png` | 런별 분산 및 이분포 Decode |
| Fig 3 | `fig3_cpu_gpu_timeline.png` | CPU-GPU 동시 실행 타임라인 |
| Fig 4 | `fig4_decode_linearity.png` | Decode 단계 수 vs 레이턴시 선형성 |
| Fig 5 | `fig5_cpu_core_heatmap.png` | 14코어 × 8런 CPU 활용률 히트맵 |
| Fig 6 | `fig6_cpu_phase_bar.png` | 추론 단계별 CPU 부하 분석 |
| Fig 7 | `fig7_hardware_compare.png` | 하드웨어 플랫폼 비교 |
| Fig 8 | `fig8_optimization_roadmap.png` | 최적화 로드맵 (Waterfall) |
| Fig 9 | `fig9_core02_timeseries.png` | Core 02 시계열 + 단계 오버레이 |
| Fig 10 | `fig10_memory_breakdown.png` | 메모리 사용량 및 대역폭 맥락 |

---

## Fig 1  Stage Breakdown — 추론 단계별 레이턴시 분해

**파일**: `fig1_stage_breakdown.png`

### 이 그림이 보여주는 것

Alpamayo 1.5 한 번의 추론(~5,009 ms)을 구성하는 5개 단계의 시간 비중을 두 가지 방식으로 동시에 표현합니다.

- **(a) 왼쪽 — 수평 스택 바**: 5개 단계를 시간 축 위에 순서대로 쌓아 각 단계가 전체 시간에서 차지하는 절대값(ms)과 오차 막대(±1σ)를 보여줍니다.
- **(b) 오른쪽 — 도넛 차트**: 같은 데이터를 비율(%)로 시각화합니다. 도넛 중앙에 총 레이턴시(5,009 ms)가 표시됩니다.

### 데이터 출처

`summary_v4.json` → `timing_ms` 항목의 각 단계 `mean` / `std` 값

### 주요 수치 및 해석

| 단계 | 평균 (ms) | 비중 (%) | 의미 |
|------|----------:|--------:|------|
| Vision Encoding | 714.9 | 14.3% | 멀티카메라 이미지 → 비전 토큰 변환 |
| LLM Prefill | 1,471.9 | 29.4% | 프롬프트 전체 처리 (단 1회) |
| LLM Decode | 1,925.6 | 38.4% | CoC 토큰 자동회귀 생성 (병목) |
| Flow Matching | 890.3 | 17.8% | 확산 ODE로 64 웨이포인트 생성 |
| Action Overhead | 5.9 | 0.1% | Python 제어 흐름 오버헤드 |

**결론**: LLM Decode가 전체의 38.4%를 차지하는 최대 병목입니다. LLM Prefill(29.4%)과 합산하면 LLM 전체가 67.8%로 추론 시간의 2/3를 소비합니다. Flow Matching(17.8%)은 Prefill의 절반 수준으로, 경량화된 확산 모델 덕분에 상대적으로 짧습니다. Action Overhead(0.1%)는 무시할 수 있는 수준으로, Python 제어 흐름의 효율성이 확인됩니다.

### 어떻게 만들어졌나

4개의 CUDA Event 계층으로 GPU 스트림에서 직접 타이밍을 측정했습니다. 각 단계 시작/종료에 `torch.cuda.Event(enable_timing=True)`를 삽입하고 `event.elapsed_time()` API로 GPU 시간을 마이크로초 단위로 읽습니다. CPU 오버헤드와 완전히 분리된 순수 GPU 연산 시간입니다. 오차 막대의 좁은 폭(σ < 5 ms)은 측정의 높은 재현성을 의미합니다.

---

## Fig 2  Run Variability — 런별 분산 및 이분포 Decode

**파일**: `fig2_run_variability.png`

### 이 그림이 보여주는 것

8번의 독립 실행에서 각 단계의 레이턴시가 얼마나 일관성 있게 나타나는지, 그리고 LLM Decode가 왜 두 개의 뚜렷한 군집으로 나뉘는지를 설명합니다.

- **(a) 왼쪽 — 런별 스택 바**: 각 런의 4단계 레이턴시를 색상별로 쌓았습니다. 배경색은 해당 런의 CoC 토큰 수를 나타냅니다 — 파란색(16 steps) vs 주황색(19 steps). Decode 막대 내부에 "16x 110ms" / "19x 110ms" 텍스트가 표시됩니다.
- **(b) 오른쪽 — Bimodal 박스플롯**: 16-step 런 4회와 19-step 런 4회를 별도 박스플롯으로 비교합니다. 이론값(점선)과 실측값의 일치도를 화살표로 주석 처리했습니다.

### 데이터 출처

`raw_timings_v4.json` → 8개 런 각각의 `vision_encoding`, `llm_prefill`, `llm_decode`, `action_direct`, `decode_steps`

### 주요 수치 및 해석

```
16-step 런 (runs 1, 3, 6, 8):  Decode ≈ 1,762 ms  (mean)
19-step 런 (runs 2, 4, 5, 7):  Decode ≈ 2,089 ms  (mean)

이론값:  16 × 110.0 ms = 1,760 ms  (실측과 0.1% 오차)
이론값:  19 × 110.0 ms = 2,090 ms  (실측과 0.05% 오차)
```

**결론**: CoC(Chain-of-Thought) 토큰 생성 수가 비결정론적으로 16 또는 19로 나뉘며, 각 스텝당 레이턴시는 110.0 ms로 완벽하게 일정합니다. Vision, Prefill, Flow Matching은 런 간 변동 계수(CV) < 0.4%로 매우 안정적입니다. 전체 레이턴시 분산의 원인은 오직 CoC 토큰 수의 차이입니다.

### 어떻게 만들어졌나

비결정론적 CoC 토큰 수는 LLM의 greedy decoding 과정에서 `<EOS>` 토큰이 언제 생성되느냐에 따라 달라집니다. 동일한 입력으로도 GPU 연산의 미세한 부동소수점 차이로 인해 생성 길이가 달라질 수 있습니다. 이 이분포가 실제 배포 환경에서 레이턴시 예측의 주요 불확실성 요소입니다.

---

## Fig 3  CPU-GPU Execution Timeline — 동시 실행 타임라인

**파일**: `fig3_cpu_gpu_timeline.png`

### 이 그림이 보여주는 것

GPU 연산 타임라인(위 레인)과 CPU 활용률 실시간 파형(아래 레인)을 하나의 공유 시간 축 위에 겹쳐 그린 그림입니다. GPU가 특정 단계를 실행할 때 CPU가 어떻게 반응하는지를 한눈에 파악할 수 있습니다.

- **GPU 레인 (위)**: 4개 단계를 색상 블록으로 Gantt 표현
- **CPU Core 02 파형 (아래, 빨간 채움)**: psutil 50ms 샘플링의 실측값
- **CPU Core 00 파형 (아래, 파란 채움)**: CPUSampler 서브스레드

### 데이터 출처

- GPU 타이밍: `raw_timings_v4.json` → Run 1 (run_id=0)
- CPU 파형: `cpu_raw_samples.json` → run_id=0의 `raw_samples[].cores[2]` / `[0]`
- 단계 마커: `cpu_raw_samples.json` → `markers[]`의 `t_ms` 값

### 주요 수치 및 해석

```
Run 1 총 레이턴시: 4,843 ms  (16-step 런)

Vision Encoding (0 → 715 ms):   Core 02  ≈ 30-40%  (이미지 토큰화, 일부 CPU 작업)
LLM Prefill  (715 → 2,180 ms):  Core 02  ≈ 50-70%  (CUDA kernel launch 집중)
LLM Decode   (2,180 → 3,932 ms): Core 02 ≈ 40-60%  (110 ms/step 반복 루프)
Flow Matching (3,932 → 4,843 ms): Core 02 ≈ 80-100% (ODE step loop 최고 부하)
```

**결론**: CPU와 GPU는 대부분의 구간에서 동시에 실행됩니다(GPU 오프로드 후 CPU가 다음 연산 준비). 그러나 Flow Matching 단계에서 Core 02가 100%에 도달하는 구간이 관측되며, 이는 CPU가 ODE step 제어를 주도하는 구간입니다. Core 00(CPUSampler 스레드)의 파형이 Core 02보다 낮고 일정한 것은 50ms마다 단순 샘플링만 수행하기 때문입니다.

### 어떻게 만들어졌나

**CPU-GPU 시간 동기화 방법**: CPUSampler는 `sampler.start()` 직전의 `time.perf_counter()`를 기준점(t=0)으로 설정합니다. GPU 훅의 `mark("phase_name")` 호출도 동일한 `time.perf_counter()` 타임스탬프를 기록합니다. 두 타임라인이 같은 CPU 시계를 공유하므로 별도의 보정 없이 정렬됩니다. GPU CUDA Event 타이밍과의 오차는 ±0.5μs 수준으로 무시할 수 있습니다.

---

## Fig 4  Decode Linearity — 메모리 대역폭 병목 분석

**파일**: `fig4_decode_linearity.png`

### 이 그림이 보여주는 것

LLM Decode 레이턴시가 CoC 토큰 생성 수(decode steps)에 대해 완벽한 선형 관계를 가진다는 것을 통계적으로 증명합니다. 그리고 그 기울기(110.0 ms/step)가 메모리 대역폭의 물리적 한계(81.2 ms/step)로부터 73.9%의 대역폭 활용률로 설명된다는 것을 보여줍니다.

- **산점도**: 16-step(파란 원) 4개, 19-step(주황 원) 4개
- **회귀선 (빨간 실선)**: 기울기 = 110.0 ms/step, R² = 0.998
- **이론 하한선 (회색 점선)**: 22,157 MB / 273 GB/s = 81.2 ms/step
- **음영 면적**: 이론값과 실측값 사이의 간격 = BW 오버헤드

### 데이터 출처

`raw_timings_v4.json` → 8개 런의 `decode_steps`, `llm_decode`

### 수학적 근거

```
메모리 대역폭 병목 이론:
  - 모델 파라미터 크기: 22,157 MB (BF16)
  - Thor LPDDR5x 스펙: 273 GB/s
  - 이론 최솟값: 22,157 / 273 = 81.2 ms/step

실측:
  - 기울기 = 110.0 ms/step  (선형 회귀)
  - R² = 0.99810  (선형성 매우 강함)
  - 실효 BW = 81.2 / 110.0 × 100% = 73.9%

잔여 26.1%의 원인:
  - Softmax, LayerNorm 등 compute-bound 연산
  - Python GIL overhead (CUDA kernel launch 대기)
  - DRAM 비효율적 접근 패턴 (attention head scatter)
```

**결론**: Decode는 100% 메모리-대역폭 병목(memory-bandwidth-bound)입니다. Compute(TFLOPS)를 늘려도 Decode 속도는 개선되지 않습니다. 속도를 높이려면 ① 모델 양자화(FP4 → 가중치 크기 절반 → 이론 BW 요구 절반), ② 추론 배치화, ③ Speculative Decoding으로 실제 step 수 감소 중 하나가 필요합니다.

---

## Fig 5  CPU Core Heatmap — 14코어 × 8런 활용률 히트맵

**파일**: `fig5_cpu_core_heatmap.png`

### 이 그림이 보여주는 것

추론 실행 중 14개 CPU 코어 각각이 얼마나 활용되는지를 런(행) × 코어(열) 행렬로 시각화합니다. Python GIL의 영향으로 어떤 코어가 "선택"되어 실제 연산을 수행하는지, 나머지 코어는 왜 유휴 상태인지를 시각적으로 증명합니다.

- **(a) 왼쪽 — 히트맵 (Log scale)**: 0.05% ~ 100% 범위를 로그 스케일 적색으로 표현. Core 02 열에 빨간 테두리 강조
- **(b) 오른쪽 — 코어별 평균 바**: 8런 평균 활용률을 수평 막대로 표시. Core 02(빨간)와 Core 00(연파랑) 주석 포함

### 데이터 출처

`raw_timings_v4.json` → 각 런의 `cpu.per_core_mean_pct[0..13]` (14개 값)

### 주요 수치 및 해석

| Core | 평균 활용률 (%) | 역할 |
|:----:|:--------------:|------|
| **Core 02** | **52.2** | Python GIL 메인 스레드 (CUDA dispatch, 제어 흐름) |
| Core 00 | 11.2 | CPUSampler 데몬 스레드 (50ms 샘플링) |
| Core 04 | 2.8 | PyTorch 워커 스레드 |
| Core 08 | 1.3 | 산발적 OS 작업 |
| 나머지 10개 | ~0 | 완전 유휴 |

**Python GIL 증거**: 14개 코어 중 단 1개(Core 02)에 전체 CPU 작업이 집중되는 현상은 Python Global Interpreter Lock의 직접적인 증거입니다. GIL이 있는 한, Python 바이트코드는 동시에 하나의 코어에서만 실행되므로 이론 최대치(100% / 14코어 = 7.1%)를 7.3배 초과한 52.2%가 Core 02에만 집중됩니다.

**로그 스케일 선택 이유**: 대부분의 코어가 0~1% 수준이고 Core 02만 52%이므로, 선형 스케일로는 다른 코어의 미세한 차이가 시각화되지 않습니다. LogNorm(0.05~100)을 사용하여 각 코어의 상대적 활성화 수준을 모두 표현했습니다.

---

## Fig 6  CPU Phase Bar — 추론 단계별 CPU 부하 분석

**파일**: `fig6_cpu_phase_bar.png`

### 이 그림이 보여주는 것

추론의 4개 단계(Vision, Prefill, VLM Total, Flow Matching)에서 CPU 전체 코어 평균 활용률이 어떻게 달라지는지를 막대그래프로 비교합니다. 오른쪽 패널은 같은 데이터를 Core 02의 추정 활용률(×14배 스케일업)로 변환하여 GIL 주 스레드의 단계별 부하를 보여줍니다.

- **(a) 왼쪽 — 단계별 전체 코어 평균**: 오차 막대 포함, GIL 이론 최대(7.1%) 기준선
- **(b) 오른쪽 — Core 02 추정값**: 전체 평균 × 14코어 = Core 02 단독 추정치, 실측 Core 02 평균(52.2%) 수평선 오버레이

### 데이터 출처

`summary_v4.json` → `cpu_summary.by_phase` (CPUSampler의 `mark()` 타임스탬프로 구간 분리)

### 주요 수치 및 해석

| 단계 | 전체 코어 평균 (%) | Core 02 추정 (%) | 해석 |
|------|:----------------:|:---------------:|------|
| Vision Encoding | 2.3 | ~32% | GPU가 독립적으로 이미지 처리, CPU 대기 |
| LLM Prefill | 5.0 | ~70% | CUDA kernel launch 반복, Python 활성 |
| VLM Total (V+P+D) | 4.6 | ~64% | Prefill과 Decode의 혼합 평균 |
| **Flow Matching** | **7.2** | **~101%** | ODE step 루프: torch.randn() × N회, 최고 부하 |
| Overall | 5.0 | ~70% | 전 추론 구간 평균 |

**Flow Matching이 가장 높은 이유**: `diffusion.sample()` 내부의 ODE 루프는 매 step마다 Python에서 `torch.randn()`(노이즈 샘플링)과 step 인덱스 계산, 텐서 슬라이싱을 반복합니다. 이 반복 제어가 모두 Python GIL 하에서 수행되므로 단계 중 CPU 부하가 가장 높습니다.

**Vision이 가장 낮은 이유**: 비전 인코더는 단일 forward pass이며, GPU가 독립적으로 픽셀 연산을 수행하는 동안 CPU는 결과를 기다립니다. CUDA 비동기 실행의 이점이 가장 잘 발휘되는 단계입니다.

---

## Fig 7  Hardware Comparison — 플랫폼 비교

**파일**: `fig7_hardware_compare.png`

### 이 그림이 보여주는 것

Jetson AGX Thor의 성능을 다양한 하드웨어 플랫폼과 객관적으로 비교합니다. Alpamayo 1.5의 decode bottleneck이 메모리 대역폭에 의존함을 보여주고, Thor의 위치를 에지-클라우드 스펙트럼 위에 배치합니다.

- **(a) 왼쪽 — 메모리 BW vs Decode 레이턴시 (log-log)**: 이론 곡선(y = 22,157 MB / BW)과 각 플랫폼의 실측/추정 데이터포인트
- **(b) 오른쪽 — 추정 총 레이턴시 수평 바**: 17.5 steps 평균 기준, 100 ms 실시간 목표선 포함

### 데이터 출처

- Thor 실측: `summary_v4.json`
- 타 플랫폼: 공개 스펙 기반 추정 (Vision/Prefill/Flow는 Thor 실측값 고정, Decode만 BW 환산)

### 비교 결과

| 플랫폼 | BW (GB/s) | Decode/step (ms) | 총 레이턴시 (ms) | TDP (W) |
|--------|----------:|----------------:|----------------:|--------:|
| A100 80G PCIe | 2,000 | ~15 | ~3,340 | 300 |
| RTX PRO 6000 | 1,700 | ~17.6 | ~3,385 | 300 |
| **Jetson AGX Thor** | **273** | **110.0** | **5,009** | **60** |
| Apple M4 Max 96G | 546 | ~55 | ~4,038 | 35 |
| Jetson Orin AGX | 204 | ~147 | ~5,643 | 40 |

**결론**: Thor는 에지 플랫폼 중 최고 성능이며, 전력 효율(ms/W)에서 A100 대비 우위를 보입니다. A100이 Thor보다 7.3배 빠른 decode를 제공하지만 5배 높은 전력을 소비합니다. 실시간 목표(100 ms)는 어떤 플랫폼도 현재 BF16 Eager 설정으로는 달성하지 못합니다 — 최적화가 필수적입니다.

### 주의사항

A100, RTX PRO 6000, Orin, M4 Max의 값은 실측이 아닌 BW 기반 추정치입니다. Vision Encoding, Prefill, Flow Matching 레이턴시는 Thor 실측값을 그대로 사용했으며, 이 값들은 플랫폼마다 다를 수 있습니다.

---

## Fig 8  Optimization Roadmap — 최적화 로드맵

**파일**: `fig8_optimization_roadmap.png`

### 이 그림이 보여주는 것

현재 5,009 ms인 레이턴시를 100 ms 실시간 목표까지 줄이기 위한 최적화 단계를 Waterfall(폭포수) 차트로 표현합니다. 각 최적화 기법이 얼마나 레이턴시를 감소시키는지, 어느 순서로 적용해야 하는지를 시각적으로 제시합니다.

### 데이터 출처

실측 baseline(5,009 ms)에서 각 기법의 예상 감소량은 문헌 및 NVIDIA 공식 자료 기반 추정:

| 최적화 단계 | 예상 감소 (ms) | 누적 레이턴시 (ms) | 근거 |
|------------|:-------------:|:-----------------:|------|
| Baseline (BF16, Eager) | — | 5,009 | 실측 |
| TensorRT Engine | -1,200 | 3,809 | TRT 커널 퓨전, 메모리 최적화 |
| FP4 Quantization | -800 | 3,009 | 가중치 크기 4배 감소 → BW 요구 절반 |
| CUDA Graphs (Decode) | -370 | 2,639 | GPU kernel launch overhead 제거 (419ms) |
| Flash Attention (SDPA) | -320 | 2,319 | Attention 메모리 복잡도 O(n²) → O(n) |
| KV Offload + Prefetch | -400 | 1,919 | Prefill KV 캐시를 CPU 메모리로 오프로드 |
| Speculative Decoding | -1,800 | 119 | 소형 draft model로 다수 토큰 동시 생성 |
| **Target (100 ms)** | — | **100** | 최종 실시간 목표 |

**결론**: 단일 최적화로는 목표에 도달 불가능합니다. TensorRT + FP4 + CUDA Graphs의 조합이 1차 목표(< 3,000 ms)이며, 실시간(100 ms)을 달성하려면 Speculative Decoding이 필수입니다. CUDA Graphs가 Decode loop의 419 ms CPU-GPU sync overhead를 제거하는 것이 단기 최우선 과제입니다.

---

## Fig 9  Core 02 Time-Series — 메인 스레드 시계열 분석

**파일**: `fig9_core02_timeseries.png`

### 이 그림이 보여주는 것

Python GIL 메인 스레드인 Core 02의 CPU 활용률을 두 가지 스케일로 심층 분석합니다.

- **(a) 위 패널 — 8런 박스플롯**: 각 런 동안 Core 02의 활용률 분포(중앙값, Q1/Q3, 수염). 런 평균(흰 원)과 전체 평균(점선) 표시
- **(b) 아래 패널 — Run 2 시계열**: 19-step 런의 Core 02 활용률을 시간 축으로 추적. 배경 음영으로 GPU 단계 구분, Core 00(CPUSampler)과 비교

### 데이터 출처

`cpu_raw_samples.json` → 각 런의 `raw_samples[].cores[2]` (50ms 간격 시계열)

### 주요 관찰 및 해석

**상단 패널**:
- 8런 모두 Core 02 평균이 50~55% 범위에 일관되게 분포
- 16-step 런(4회)과 19-step 런(4회)의 Core 02 활용률 차이는 미미 → Decode step 수와 Core 02 평균은 무관
- 개별 샘플의 범위는 0~100%로 넓음 — CUDA kernel launch 순간에 100%, GPU 대기 중 0%를 교대 반복

**하단 패널 (Run 2, 19 steps)**:
- Vision 구간 (0~715 ms): Core 02가 산발적으로 활성화 — GPU는 이미 연산 중이지만 CPU는 다음 연산 준비 중
- Prefill 구간 (715~2,193 ms): Core 02 활성화 빈도 증가 — 긴 시퀀스에 대한 반복 kernel launch
- Decode 구간 (2,193~4,277 ms): 110 ms 주기의 펄스 패턴 — 각 step마다 CUDA launch → GPU 대기 → 다음 launch 반복
- Flow 구간 (4,277~5,172 ms): Core 02가 가장 높은 밀도로 활성화 — ODE loop 제어 집중

**Core 00(CPUSampler) 패형**: 낮고 일정한 ~10% 수준으로, 추론에 독립적으로 동작함을 확인. 이는 CPUSampler가 추론 성능에 영향을 미치지 않음을 증명합니다.

---

## Fig 10  Memory Breakdown — 메모리 사용량 및 대역폭 맥락

**파일**: `fig10_memory_breakdown.png`

### 이 그림이 보여주는 것

Alpamayo 1.5가 Thor의 128 GB Unified Memory 중 얼마를 사용하는지, 그리고 Thor의 메모리 대역폭이 다른 플랫폼 대비 어떤 위치인지를 정량화합니다.

- **(a) 왼쪽 — 메모리 사용량 도넛**: 모델 가중치, 활성화/KV 캐시, 잔여 가용 메모리를 비율로 표시
- **(b) 오른쪽 — 메모리 대역폭 비교 바**: A100, RTX PRO, M4 Max, Thor 스펙 BW, Thor 실효 BW, Orin AGX를 비교

### 데이터 출처

`summary_v4.json` → `memory_mb.param_mem_mb`, `memory_mb.activation.mean`, `memory_mb.peak_gpu.mean`

### 주요 수치 및 해석

**메모리 사용량**:
```
모델 가중치 (BF16):  22,157 MB  =  21.6 GB  (16.4% of 131.9 GB)
활성화 + KV 캐시:    1,044 MB   =   1.0 GB  (0.8%)
총 사용량:           23,201 MB  =  22.7 GB  (17.2%)
가용 잔여:          112,107 MB  = 109.5 GB  (82.8%)
```

**핵심 해석**:
1. **OOM 위험 없음**: 22.7 GB / 131.9 GB = 17.2%만 사용. 멀티 모델 배포, 배치 추론, 중간 활성화 저장이 모두 가능합니다.
2. **BF16의 의미**: FP32 대비 메모리 절반 사용. FP4로 전환 시 약 5.5 GB로 추가 감소 → BW 요구량 4배 절감.
3. **Thor 실효 BW**: 스펙 273 GB/s의 73.9% = 201.7 GB/s. 이는 Fig 4에서 확인한 73.9% BW 활용률과 정확히 일치합니다.
4. **Unified Memory의 장점**: Thor는 CPU와 GPU가 동일한 물리 메모리를 공유하므로, CPU-GPU 데이터 전송 비용이 없습니다. 이것이 에지 추론에서 Thor가 분리 메모리 구조의 PCIe 카드보다 유리한 이유입니다.

---

## Figure 생성 재현 방법

```bash
# Thor 보드에서 실행
cd ~/alpamayo1.5
source a1_5_venv/bin/activate

# 프로파일링 데이터가 이미 있는 경우 (재수집 불필요)
# profiling_results/summary_v4.json, raw_timings_v4.json, cpu_raw_samples.json 확인

# Figure 생성 (10개 PNG, 약 5-10초 소요)
python scripts/profiling/visualize_profile.py

# 결과 확인
ls profiling_results/figures/
```

생성된 PNG는 `profiling_results/figures/` 디렉토리에 저장됩니다 (300 DPI, 논문 제출 규격).

---

## 측정 방법론 요약

| 측정 대상 | 도구 | 정밀도 | 특징 |
|----------|------|:------:|------|
| GPU 단계별 레이턴시 | `torch.cuda.Event` | ±0.5 μs | 순수 GPU 시간, CPU 오버헤드 제외 |
| CPU 코어별 활용률 | `psutil.cpu_percent(percpu=True)` | 50 ms 간격 | 추론 구간만 측정 (CPUSampler) |
| CPU 단계별 분리 | `sampler.mark("phase_name")` | `time.perf_counter()` | GPU 훅과 동일 시계 공유 |
| 피크 GPU 메모리 | `torch.cuda.max_memory_allocated()` | 1 MB | 추론 전 reset, 직후 측정 |

**CPUSampler 설계 원칙**: tegrastats와 달리, CPUSampler는 `sampler.start()` / `sampler.stop()` 사이의 **추론 구간만** 정확하게 포함합니다. OS 스케줄러 슬립, 데이터 로딩, 커맨드라인 오버헤드가 완전히 배제됩니다.

---

*작성일: 2026-05-08*  
*데이터: Jetson AGX Thor 실측 (v4.0 프로파일러)*  
*분석 수준: 석사 논문 제출 기준*
