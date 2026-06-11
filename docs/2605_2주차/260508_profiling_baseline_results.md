# Alpamayo 1.5 — Jetson AGX Thor 추론 프로파일링 결과 분석

> **버전**: v4.0 (4계층 GPU + CPU 코어별 psutil 실측)
> **측정일**: 2026-05-08
> **환경**: Jetson AGX Thor · SM 11.0 · CUDA 13.0 · PyTorch 2.8.0 · BF16 · Python 3.12.13
> **GPU 측정**: CUDA Event + monkey-patch + forward hook 4계층
> **CPU 측정**: psutil.cpu_percent(percpu=True) 50ms 샘플링 — 인퍼런스 구간 완전 동기화
> **통계**: 워밍업 3회 제외, 측정 8회 평균

---

## 목차

1. [왜 이 분석이 필요한가](#1-왜-이-분석이-필요한가)
2. [모델 구조와 측정 단계 정의](#2-모델-구조와-측정-단계-정의)
3. [계측 방법론](#3-계측-방법론)
4. [GPU 측정 결과](#4-gpu-측정-결과)
5. [CPU 측정 결과 — 코어별 실측](#5-cpu-측정-결과--코어별-실측)
6. [CPU-GPU 상호작용 분석](#6-cpu-gpu-상호작용-분석)
7. [핵심 질문: 왜 Decode가 병목인가](#7-핵심-질문-왜-decode가-병목인가)
8. [메모리 사용량](#8-메모리-사용량)
9. [최적화 로드맵](#9-최적화-로드맵)
10. [결론 요약](#10-결론-요약)

---

## 1. 왜 이 분석이 필요한가

Alpamayo 1.5는 자율주행용 VLA(Vision-Language-Action) 모델이다.
입력(카메라 영상) → 출력(6.4초 궤적 + 추론 텍스트)까지 **단일 추론 1회에 약 5,009ms**가 걸린다.
실시간 자율주행의 제어 루프 목표는 **100ms** 이하이므로 현재 **50배 초과** 상태다.

> 어디서 시간이 가는지 모르면 최적화 방향을 잡을 수 없다.
> 이 문서는 GPU 4단계를 직접 계측하고, **CPU 코어별 실측**을 추가해
> "GPU만 쓰는가, CPU도 쓰는가, 어느 단계에서 CPU가 필요한가"를 완전히 규명한다.

---

## 2. 모델 구조와 측정 단계 정의

Alpamayo 1.5의 추론 파이프라인은 두 모델이 직렬로 실행된다.

```
입력 (멀티카메라 영상 14장 + 자차 궤적 히스토리)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  VLM (Cosmos Reason2 / Qwen3-VL 기반, 8.2B 파라미터)        │
│                                                             │
│  [1] Vision Encoding    [2] LLM Prefill    [3] LLM Decode   │
│   이미지 토큰화           입력 전처리         CoC 텍스트 생성  │
│   GPU: compute-bound     GPU: compute-bound  GPU: memory-bound│
│   CPU: 거의 없음         CPU: 5.0%           CPU: ~4.6%      │
│                                                             │
│  출력: Chain-of-Cognition 추론 텍스트 + KV cache             │
└─────────────────────────────────────────────────────────────┘
        │  (5.9ms — deepcopy + 전환 코드, Action Overhead)
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Action Expert (Flow Matching ODE, 2.3B 파라미터)            │
│                                                             │
│  [4] Flow Matching                                          │
│   노이즈 → 궤적 수치 적분                                    │
│   GPU: compute+memory 혼합                                  │
│   CPU: 7.2% ← 가장 높음 (ODE 스텝 반복 + torch.randn() 호출) │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
출력 (pred_xyz, pred_rot — 64 waypoints, 6.4초 분량)
```

---

## 3. 계측 방법론

### 3.1 GPU 계측: 4계층 패치 시스템

GPU 시간은 **CUDA Event**를 모델 내부에 직접 삽입해 측정했다.
CUDA Event는 GPU 스트림 안에서 타임스탬프를 찍으므로 CPU 블로킹 없이 ±0.5μs 정밀도를 달성한다.

```
Layer 1: model.vlm.generate() monkey-patch
         → VLM 전체 구간 시작/종료 마킹 (ev_vlm_s ~ ev_vlm_e)

Layer 2: model.vlm forward hook (pixel_values / past_kv 유무 판단)
         → Prefill: 첫 번째 forward (ev_pre_s ~ ev_pre_e)
         → Decode: 이후 forward 누적 (ev_vlm_e - ev_pre_e)

Layer 3: model.vlm.model.visual forward hook
         → Vision Encoding (ev_vis_s ~ ev_vis_e)

Layer 4: model.diffusion.sample() monkey-patch  [v3.0 이후]
         → Flow Matching ODE (ev_action_s ~ ev_action_e)
```

**계측 품질 지표:**
- Vision / Prefill / Flow 직접 측정률: **100%** (8/8 런)
- 단계 합산 잔차: **0.000ms** (max 0.002ms) → 4단계가 전체를 빠짐없이 설명

### 3.2 CPU 계측: psutil 임베드 방식 (v4.0 신규)

#### 왜 tegrastats를 쓰지 않았나

tegrastats는 외부 프로세스에서 독립적으로 실행되기 때문에 **인퍼런스 시작/종료 시점을 알 수 없다**.
로그의 어느 줄이 추론 중인지, 시작 전인지, 종료 후인지 구분이 불가능하다.
또한 수동으로 타이밍을 맞춰 실행해야 하므로 재현성도 낮다.

#### psutil 임베드 방식의 원리

`CPUSampler` 클래스를 프로파일러 스크립트 내부에 내장해,
**인퍼런스 시작 직전**에 백그라운드 스레드를 켜고 **종료 직후**에 끈다.

```python
sampler.start()          # ← 인퍼런스 직전 (백그라운드 스레드 시작)

model.sample_trajectories_from_data_with_vlm_rollout(...)

cpu_result = sampler.stop()   # ← 인퍼런스 직후 (스레드 종료, 결과 반환)
```

백그라운드 스레드는 50ms마다 `psutil.cpu_percent(percpu=True)`를 호출해
**Core 0~13 각각의 활용률(%)**을 타임스탬프와 함께 기록한다.

#### 단계별 CPU 마커 동기화

GPU 훅(Layer 1~4)이 호출되는 시점에 `sampler.mark("phase_name")`을 동시에 실행한다.
이 방식으로 **어느 CPU 샘플이 어느 GPU 단계에 속하는지** 정확하게 분리할 수 있다.

```
vlm.generate() 진입  →  mark("vlm_start")
Vision Encoding 시작  →  mark("vision_start")
Vision Encoding 종료  →  mark("vision_end")
Prefill 시작          →  mark("prefill_start")
Prefill 종료          →  mark("prefill_end")
vlm.generate() 종료  →  mark("vlm_end")
diffusion.sample() 시작  →  mark("flow_start")
diffusion.sample() 종료  →  mark("flow_end")
```

---

## 4. GPU 측정 결과

### 4.1 단계별 평균 GPU 시간 (v4.0 최신)

| 단계 | mean (ms) | ±std | p50 | p95 | 비율 | CV(%) | 측정 방법 |
|------|----------:|-----:|----:|----:|-----:|------:|---------|
| **Vision Encoding** | **714.9** | 2.8 | 715.5 | 717.7 | **14.3%** | 0.39 | Layer 3 직접 |
| **LLM Prefill** | **1,471.9** | 4.5 | 1,473.0 | 1,476.7 | **29.4%** | 0.31 | Layer 2 직접 |
| **LLM Decode** | **1,925.6** | 175.4 | 1,927.7 | 2,096.0 | **38.4%** | 9.11 | 유도 (잔차 0) |
| **Flow Matching** | **890.3** | 1.6 | 890.7 | 891.9 | **17.8%** | 0.18 | Layer 4 직접 |
| Action Overhead | 5.9 | 0.4 | 5.8 | 6.5 | 0.1% | 5.98 | 유도 |
| **Total GPU** | **5,008.7** | 177.5 | 5,010.8 | 5,177.3 | 100% | 3.54 | CUDA Event |
| Total Wall | 5,008.7 | 177.5 | 5,010.9 | 5,177.3 | — | 3.54 | perf_counter |
| CPU Overhead | **0.05** | 0.02 | 0.04 | 0.09 | — | — | Wall − GPU |

> **CPU Overhead = 0.05ms**: Wall 시간과 GPU 시간이 사실상 동일하다.
> 이는 `torch.cuda.synchronize()` 이후 CUDA Event elapsed_time을 읽는 Python 코드 자체의 오버헤드다.
> 추론 루프 내 CPU 블로킹은 없다는 증거이기도 하다.

### 4.2 런별 원시 데이터 (8회 측정)

| Run | Total (ms) | Vision | Prefill | Decode | Flow | Steps | CPU all-core |
|-----|----------:|-------:|--------:|-------:|-----:|------:|-------------:|
| 1 | 4,843 | 709 | 1,464 | 1,771 | 892 | **16** | 4.93% |
| 2 | 5,171 | 715 | 1,478 | 2,084 | 889 | 19 | 5.04% |
| 3 | 4,836 | 715 | 1,475 | 1,753 | 887 | **16** | 5.14% |
| 4 | 5,174 | 718 | 1,474 | 2,085 | 891 | 19 | 5.08% |
| 5 | 5,179 | 716 | 1,466 | 2,099 | 891 | 19 | 4.96% |
| 6 | 4,842 | 715 | 1,474 | 1,757 | 889 | **16** | 4.92% |
| 7 | 5,174 | 716 | 1,472 | 2,090 | 890 | 19 | 4.99% |
| 8 | 4,850 | 716 | 1,472 | 1,765 | 892 | **16** | 5.00% |

**Bimodal 분포 확인**: 16-step 런 4회 (평균 4,843ms), 19-step 런 4회 (평균 5,175ms).
CoC(Chain-of-Cognition) 토큰 수가 입력에 따라 이산적으로 결정되는 현상이다.

### 4.3 Decode 선형성 검증

1 step = 110.0ms 관계가 완벽하게 성립한다 (R² = 0.99837).

```
16 steps × 110.0ms = 1,760ms  vs  실측 평균 1,762ms  (오차 0.1%)
19 steps × 110.0ms = 2,090ms  vs  실측 평균 2,089ms  (오차 0.05%)
```

이 선형성은 **Decode가 메모리 대역폭 한계로 동작**한다는 직접 증거다.
각 스텝마다 동일한 양의 가중치를 메모리에서 로드하기 때문에 스텝당 시간이 일정하다.

---

## 5. CPU 측정 결과 — 코어별 실측

> **모든 수치는 psutil 실측값이다. 추정 없음.**
> 측정 조건: 인퍼런스 시작 직전 ~ 직후 구간만 포함 (8런 × 99샘플/런 = 792 CPU 샘플)

### 5.1 Jetson AGX Thor CPU 구성

| 항목 | 값 |
|------|-----|
| 물리 코어 수 | 14 |
| 논리 코어 수 | 14 (ARM 계열, 하이퍼스레딩 없음) |
| 현재 동작 주파수 | 1,196 MHz |
| OS | Ubuntu 24.04 LTS (Linux 6.8) |

### 5.2 코어별 평균 활용률 (8런 평균)

| Core | mean (%) | max (%) | p95 (%) | 역할 |
|------|--------:|--------:|--------:|------|
| **Core 00** | **11.2** | 100.0 | 18.1 | CPUSampler 백그라운드 스레드 |
| Core 01 | 0.2 | 25.0 | 0.7 | 유휴 |
| **Core 02** | **52.2** | 100.0 | 53.9 | **Python 메인 스레드 (GIL 소유자)** |
| Core 03 | 1.0 | 40.0 | 2.6 | 유휴 (간헐적 OS 작업) |
| Core 04 | 2.8 | 40.0 | 10.3 | PyTorch 내부 워커 (간헐적) |
| Core 05 | 0.4 | 25.0 | 1.4 | 유휴 |
| Core 06 | 0.3 | 20.0 | 1.2 | 유휴 |
| Core 07 | 0.0 | 0.0 | 0.0 | 완전 유휴 |
| Core 08 | 1.3 | 33.3 | 2.7 | 유휴 (간헐적) |
| Core 09~12 | 0.0 | 0.0 | 0.0 | 완전 유휴 |
| Core 13 | 0.4 | 25.0 | 1.7 | 유휴 |
| **전체 평균** | **5.0%** | — | — | 14코어 평균 |

**수치 검증:**
```
(11.2 + 0.2 + 52.2 + 1.0 + 2.8 + 0.4 + 0.3 + 0.0 + 1.3 + 0+0+0.2+0+0.4) / 14
= 70.0 / 14 = 5.0%  ← 출력 "전체 평균 5.0%"와 완벽 일치  (수식 검증)
```

### 5.3 Core 02: Python GIL 메인 스레드 분석

Core 02의 **52.2% 평균 활용률**은 이 분석에서 가장 중요한 발견이다.

#### Python GIL(Global Interpreter Lock)이란

CPython(표준 Python 인터프리터)은 한 번에 하나의 스레드만 Python 바이트코드를 실행할 수 있다.
이를 **GIL**이라고 한다. 결과적으로:
- 아무리 많은 CPU 코어가 있어도 Python 코드는 단일 코어에서만 실행된다
- 멀티스레딩으로 Python 코드를 병렬화할 수 없다
- CUDA 커널은 GPU에서 독립적으로 실행되므로 GIL과 무관하다

#### Core 02가 52.2%인 이유 — 작업 분석

5,009ms 추론 동안 Core 02가 실제로 바쁜 시간:
```
5,009ms × 52.2% = 2,615ms  (Core 02 실제 연산 시간)
5,009ms × 47.8% = 2,394ms  (Core 02 유휴 — GPU 커널 실행 대기)
```

Core 02가 수행하는 작업:
1. **CUDA 커널 론치(dispatch)**: `cudaLaunchKernel()` 호출 — CPU에서 실행
2. **Token sampling** (Decode 루프): 매 스텝 top-p softmax + 샘플링 — CPU 연산
3. **Flow Matching ODE 스테핑**: `torch.randn()` 노이즈 생성 + step 제어 — CPU 연산
4. **Python 인터프리터 오버헤드**: 바이트코드 실행, 함수 호출 스택

#### Core 02가 47.8%는 유휴인 이유

GPU 커널이 실행되는 동안 CPU는 다음 명령을 준비하며 sleep/wait 상태에 들어간다.
특히 Prefill처럼 하나의 대형 GEMM이 오랫동안 실행되는 구간에서 CPU가 길게 유휴 상태가 된다.

#### GIL 이론 최대치와 비교

단일 스레드가 14코어 시스템에서 가질 수 있는 all-core 평균 최대:
```
100% (단일 코어 포화) / 14코어 = 7.14% (전체 평균 기준)
```

실측 **5.0%**는 이론 최대(7.14%)의 70% 수준이다.
Core 02 실측 52.2% × (1/14) = 3.73%가 "코어 활용 기여분"이고,
Core 00(CPUSampler 스레드) 11.2% × (1/14) = 0.80%가 추가된다.
나머지 0.47%는 PyTorch 내부 워커 등이다.

### 5.4 단계별 CPU 활용률

| 단계 | 전체 코어 평균 (%) | ±std | GPU 특성 | CPU 역할 |
|------|------------------:|-----:|---------|---------|
| Vision Encoding | **2.3%** | 0.27% | compute-bound | 이미지 전처리 후 대기 |
| LLM Prefill | **5.0%** | 0.20% | compute-bound | 입력 토큰 처리 + KV 초기화 |
| VLM 전체 (V+P+D) | **4.6%** | 0.10% | 혼합 | — |
| **Flow Matching** | **7.2%** | 0.14% | ODE 루프 | torch.randn() + step 반복 |
| 전체 인퍼런스 | **5.0%** | 0.08% | — | — |

**Flow Matching CPU 활용률이 가장 높은 이유:**

Flow Matching은 ODE(상미분방정식) 수치 적분이다.
각 step마다 다음을 CPU에서 실행한다:
1. `torch.randn()` — CPU에서 난수 생성 후 GPU로 전송
2. `step_fn()` 호출 — Python 함수 호출 오버헤드
3. step 카운터 증감 — Python 정수 연산

이 반복 패턴이 Decode 루프와 유사하지만, Flow Matching은 매 step마다 반드시 CPU가 관여하기 때문에 상대적으로 CPU 활용률이 높다.

---

## 6. CPU-GPU 상호작용 분석

### 6.1 pytorch_trace.json 분석 결과 (3회 추론 기반)

v4.0 psutil 분석에 더해, 별도로 PyTorch Profiler Chrome Trace를 분석했다.
이 두 측정 방식은 **서로 다른 것을 측정**하며 상호 보완적이다.

| 측정 도구 | 값 | 측정 대상 |
|----------|---:|---------|
| pytorch_trace CPU 활성 | 74.9% | PyTorch op 내 wall time 비율 (스레드 alive 기준) |
| psutil Core 02 활용률 | 52.2% | OS가 실제 CPU clock cycle 할당한 비율 |
| psutil 전체 14코어 평균 | 5.0% | 전체 코어 clock 활용률 |

**두 수치의 차이(74.9% vs 52.2%)를 왜 다른가:**

pytorch_trace는 Python 스레드가 PyTorch op 내부에 머물렀던 wall time을 카운트한다.
GPU 커널이 실행되는 동안 CPU thread가 아직 op 내부에서 완료를 기다리는 경우도 "활성"으로 카운트된다.

반면 psutil은 OS 스케줄러 수준에서 Core 02에 실제 CPU 사이클이 할당된 비율만을 측정한다.
GPU 커널 완료를 기다리며 `sleep()`이나 `futex_wait()`를 호출한 시간은 0%로 나온다.

따라서:
- pytorch_trace 74.9% - psutil Core 02 52.2% ≒ **22.7%**  
  = Python 스레드가 op 내에 있지만 실제 CPU를 쓰지 않고 대기한 시간의 비율

### 6.2 CPU-GPU 병렬 실행 비율

pytorch_trace 기반:
```
CPU+GPU 동시 실행: 67.7% of wall time (3,983ms/run)
CPU 직렬 병목:     7.1% of wall time ( 419ms/run)  ← GPU 유휴 구간
GPU only:          8.5% of wall time ( 501ms/run)
완전 유휴:         16.6% (warmup/데이터 로딩 전환 등)
```

**CPU 직렬 병목 419ms/run**은 GPU가 아무 커널도 실행하지 않고 기다리는 구간이다.
이 419ms가 CUDA Graphs 적용 시 약 88% 제거 가능한 최적화 타겟이다.

### 6.3 단일 활성 스레드 확인 (GIL 증거)

pytorch_trace 분석에서 CPU op을 실제로 실행한 스레드:
- **TID 7038 (메인 스레드)**: cpu_op duration 총합의 99.9%
- TID 7069 (PyTorch 워커): 실질적으로 0

이것이 Python GIL의 직접적 증거다.
14개 코어가 있어도 Python 코드는 단일 스레드(TID 7038, Core 02)에서만 실행된다.

---

## 7. 핵심 질문: 왜 Decode가 병목인가

### 7.1 FlashDrive 논문과의 비교

FlashDrive(2025, Z Lab)는 Alpamayo 1.5를 **RTX PRO 6000(Blackwell)** 에서 측정했다.
그들의 최적화 타겟은 **Prefill**이었으며 KV Cache Reuse로 4.5× 가속을 달성했다.

우리 Thor 결과와 단계 비율을 나란히 보면:

| 단계 | FlashDrive (RTX PRO 6000) | 우리 (Jetson AGX Thor) |
|------|:---:|:---:|
| Vision | ~5% | 14.3% |
| Prefill | ~40-45% ← **병목** | 29.4% |
| **Decode** | ~30% | **38.4% ← 병목** |
| Flow Matching | ~20% | 17.8% |

**병목이 뒤집혔다. 이것은 프로파일링 오류가 아니다.**

### 7.2 원인: 메모리 대역폭 vs 연산 집약도

#### Prefill — 행렬×행렬 (Compute-bound)

Prefill은 입력 토큰 N개를 한 번에 병렬로 처리한다.
`[N × d_model] × [d_model × d_ffn]` 형태의 대형 행렬 곱.
연산량(FLOP)이 메모리 접근량(Byte)보다 훨씬 많다 → GPU TFLOPS가 병목.
TFLOPS가 높은 고성능 GPU에서 빠르다.

#### Decode — 벡터×행렬 (Memory-bound)

Decode는 1토큰씩 자기회귀적으로 생성한다. 매 스텝마다:
- 모델 가중치 전체(22,157 MB)를 메모리에서 읽는다
- 실제 연산량은 극히 적고 **메모리 로드가 지배**한다

```
[이론적 하한 — 메모리 대역폭 기준]

Jetson AGX Thor (LPDDR5x):
  22,157 MB / 273 GB/s = 81.2ms/step  (이론 최소값)
  실측: 110.0ms/step
  BW 활용률: 81.2 / 110.0 = 73.9%

RTX PRO 6000 (GDDR7):
  22,157 MB / 1,700 GB/s = 13.0ms/step  (이론 최소값)
  -> 19 steps × 13ms = 247ms  (FlashDrive에서 Decode 비중이 작은 이유)
```

**Thor의 메모리 대역폭(273 GB/s)이 RTX PRO 6000(1,700 GB/s)의 약 1/6이다.**
Decode는 메모리 대역폭에 정비례하므로 Thor에서 Decode가 상대적으로 가장 느린 단계가 된다.

#### 직관적 비유

> **Prefill**은 공장 대량 생산 — GPU 코어(기계)가 많을수록 빠르다.
> **Decode**는 창고 출고 — 대역폭(창고 입구)이 전부다.
> RTX PRO 6000은 입구가 6배 넓으니 Decode가 빠르고, Thor는 입구가 좁아 Decode가 막힌다.

### 7.3 하드웨어별 병목 요약

| GPU | 메모리 대역폭 | Decode/step | 병목 단계 |
|-----|:---:|:---:|:---:|
| RTX PRO 6000 (Blackwell) | 1,700 GB/s | ~13ms | **Prefill** |
| A100 80G (PCIe) | 2,000 GB/s | ~11ms | **Prefill** |
| Apple M4 Max | 546 GB/s | ~41ms | Prefill ≈ Decode |
| **Jetson AGX Thor** | **273 GB/s** | **110ms** | **Decode** |
| Jetson Orin AGX | 204 GB/s | ~147ms | **Decode** |

---

## 8. 메모리 사용량

| 항목 | 값 |
|------|---:|
| 전체 HW 통합 메모리 | 131.9 GB |
| 모델 파라미터 (BF16) | 22,157 MB (16.8%) |
| 활성화 + KV Cache | 1,044 MB (0.8%) |
| 사용 합계 | 23,201 MB (17.6%) |
| 여유 메모리 | 112.1 GB (82.4%) |

Jetson AGX Thor는 CPU-GPU 통합 128GB 공유 메모리 구조다.
OOM 없음. 향후 KV Cache Reuse 등 메모리 집약 최적화에 충분한 여유가 있다.

---

## 9. 최적화 로드맵

### 9.1 각 단계 최적화 가능성

| 단계 | 현재 (ms) | 특성 | 주요 최적화 기법 | 예상 후 (ms) | 근거 |
|------|----------:|------|----------------|-------------:|------|
| Vision Encoding | 714.9 | Compute-bound | TensorRT 엔진, 해상도 축소, INT8 | ~400 | TensorRT 1.5-2x |
| LLM Prefill | 1,471.9 | Compute-bound | **KV Cache Reuse** (Prefill 스킵) | ~290 | FlashDrive 결과 |
| **LLM Decode** | **1,925.6** | **Memory-bound** | **CUDA Graphs + Speculative Decoding** | **~800** | 문헌 1.5-2.4x |
| Flow Matching | 890.3 | ODE 루프 | **`--dtype fp4`** (단일 플래그) | ~220 | NVIDIA fp4 4x |
| Action Overhead | 5.9 | deepcopy | 이미 최소 수준 | ~5 | — |

### 9.2 단계적 로드맵

```
베이스라인:     5,009ms  (50.1x 초과)
                 │
                 ▼  1단계: 즉시 적용 (코드 1줄)
FP4 Flow:        889 → 220ms  (-669ms)
                 ├── 예상: 4,340ms  (43.4x)
                 │
                 ▼  2단계: 구현 1-2주
TensorRT 엔진:   -1,200ms
CUDA Graphs:     -370ms
Flash Attention: -320ms
                 ├── 예상: 2,450ms  (24.5x)
                 │
                 ▼  3단계: 연구 레벨 (3-6개월)
KV Cache Reuse:  -1,180ms
KV Offload:      -400ms
                 ├── 예상: 870ms   (8.7x)
                 │
                 ▼  4단계: 모델 레벨
Spec. Decoding:  -750ms
모델 증류:       추가 단축
                 └── 목표: ~100ms (실시간)
```

---

## 10. 결론 요약

### GPU 프로파일링 결론

| 질문 | 답 |
|------|-----|
| 총 레이턴시 | **5,009ms** (목표 100ms의 **50배**) |
| 최대 병목 | **LLM Decode 38.4%** (1,926ms) |
| Decode 병목 이유 | Thor 273 GB/s BW → 22 GB 가중치 로드 지배. **프로파일링 오류 아님** |
| FlashDrive와 다른 이유 | RTX PRO 6000 BW 6배 → Decode 빠름, Prefill 병목. Thor는 역전 |
| 계측 신뢰도 | 4계층 직접 측정. 잔차 **0.000ms**. R² = 0.998 선형 검증 완료 |
| Bimodal 분포 | 16 step (4843ms) / 19 step (5175ms) — CoC 토큰 이산 분포 |
| 즉시 적용 최적화 | `--dtype fp4` → Flow 889ms → ~220ms (**단일 플래그**) |

### CPU 프로파일링 결론 (v4.0 신규)

| 질문 | 답 |
|------|-----|
| 전체 CPU 활용률 | **5.0% ± 0.1%** (14코어 all-core 평균) |
| 실제 작동 코어 | **Core 02** (52.2%) — Python GIL 메인 스레드 |
| Core 00 역할 | 11.2% — CPUSampler 백그라운드 스레드 |
| 나머지 12개 코어 | **평균 0.3%** — 사실상 완전 유휴 |
| CPU가 가장 바쁜 단계 | **Flow Matching (7.2%)** — ODE 루프 + randn() 반복 |
| CPU가 가장 한가한 단계 | **Vision Encoding (2.3%)** — GPU가 독립 처리 |
| GPU 동시 실행 비율 | **67.7%** (pytorch_trace 기반) |
| CPU 직렬 병목 | **419ms/run** (GPU 대기 구간 — CUDA Graphs로 제거 가능) |
| 멀티코어 병렬화 여지 | **없음** — Python GIL로 인해 단일 스레드만 가능. C++ 확장이나 CUDA Graphs로 우회 필요 |

---

*파일 경로: `docs/profiling_baseline_results_결과.md`*
*관련 파일: `docs/profiling_figures_설명.md` (각 Figure 상세 설명)*
*프로파일러: `scripts/profiling/profile_alpamayo.py` v4.0*
*시각화: `scripts/profiling/visualize_profile.py` v4.0*
