# RT-Swap / Demand Layering 논문 개념 적용 실험 보고서

**날짜**: 2026-05-27  
**실험자**: Alphamayo 프로젝트 팀  
**하드웨어**: NVIDIA Jetson AGX Thor (iGPU, 128GB 통합 메모리)  
**대상 모델**: Alpamayo 1.5 (10B, `nvidia/Alpamayo-1.5-10B`)

---

## 목차

1. [실험 동기 및 목표](#1-실험-동기-및-목표)
2. [참고 논문 1 — RT-Swap (RTAS 2024)](#2-참고-논문-1--rt-swap-rtas-2024)
3. [참고 논문 2 — Demand Layering (RTSS 2022)](#3-참고-논문-2--demand-layering-rtss-2022)
4. [두 논문의 핵심 공통 아이디어](#4-두-논문의-핵심-공통-아이디어)
5. [Thor iGPU 환경에의 적용 전략](#5-thor-igpu-환경에의-적용-전략)
6. [실험 설계 — 3단계 검증 체계](#6-실험-설계--3단계-검증-체계)
7. [대역폭 측정 실험 (Phase 0-A)](#7-대역폭-측정-실험-phase-0-a)
8. [레이어 compute time 측정 (Phase 0-B)](#8-레이어-compute-time-측정-phase-0-b)
9. [비동기 파이프라인 실행 결과 (Phase 1)](#9-비동기-파이프라인-실행-결과-phase-1)
10. [실패 원인 분석 — 3가지 층위](#10-실패-원인-분석--3가지-층위)
11. [핵심 발견: Decode는 Memory-Bandwidth-Bound](#11-핵심-발견-decode는-memory-bandwidth-bound)
12. [결론 및 다음 연구 방향](#12-결론-및-다음-연구-방향)

---

## 1. 실험 동기 및 목표

### 현재 성능 격차

Alpamayo 1.5 모델의 Thor 보드 실측 추론 속도와 자율주행 실시간 요구사항 사이에는 다음과 같은 격차가 존재한다.

| 항목 | 수치 |
|------|------|
| 실측 추론 속도 (full pipeline) | **6.13 초** |
| 자율주행 실시간 요구사항 | **100 ms 이하** |
| **격차** | **약 60배** |

이 격차의 대부분은 **VLM autoregressive decode** 단계에서 발생한다.

```
Vision Encoder  →  VLM Prefill  →  [VLM Decode × 65 steps]  →  Action Expert
   (빠름)            (빠름)              ← 최대 병목 →              (빠름)
```

- Decode 1 step = 36개 Transformer 레이어를 순차 통과
- 실측: 1 step ≈ **94 ms**, 65 step = **6.13 s**

### 연구 방향 (2026-05-24 교수님 미팅 결정)

> **모델 구조 변경 없이, 시스템 레벨 최적화만으로 throughput 향상 가능성을 탐색한다.**

참고 논문 2편(RT-Swap, Demand Layering)의 아이디어를 차용하여, decode 단계의 레이어 간 비동기 파이프라인을 구성하는 것을 목표로 설정하였다.

---

## 2. 참고 논문 1 — RT-Swap (RTAS 2024)

> **Ji, M. et al.** "RT-Swap: Addressing GPU Memory Bottlenecks for Real-Time Multi-DNN Inference"  
> *IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS), 2024*

### 2.1 논문이 해결하고자 하는 문제

자율주행, 로봇 등 실시간 엣지 시스템에서는 여러 DNN 모델을 **동시에** 실행해야 한다. 그런데 각 모델의 가중치(weight)를 전부 GPU VRAM에 올려 두면 GPU 메모리를 초과하는 경우가 빈번하다. RT-Swap은 이 문제를 해결하기 위해, **CPU DRAM을 GPU의 스왑 공간(swap device)** 처럼 사용하여 VRAM 용량을 가상으로 확장하는 런타임 메모리 관리 프레임워크이다.

### 2.2 핵심 메커니즘: Prefetch와 Overlap

RT-Swap의 핵심은 **"지금 레이어를 계산하는 동안, 다음 레이어 가중치를 미리 가져온다"** 는 비동기 파이프라인이다.

```
[기존 순차 실행]
레이어 i:   ─── 가중치 로딩 (CPU DRAM → GPU VRAM) ───▶─── 계산 ───▶
레이어 i+1:                                                    ─── 가중치 로딩 ───▶─── 계산 ───▶
              ← 가중치 로딩 시간 + 계산 시간 직렬 합산 →

[RT-Swap: 비동기 overlap]
레이어 i:   ─── 가중치 로딩 ───▶─────────────── 계산 ────────────▶
레이어 i+1:                    ─── 가중치 미리 로딩(비동기) ───▶    ← 이미 준비됨!
              ← max(로딩 시간, 계산 시간) 으로 단축 →
```

이를 수식으로 표현하면:

$$T_{async} = \sum_{i=1}^{N} \max(t_{\text{prefetch}_{i+1}}, \; t_{\text{compute}_i})$$

$$T_{sequential} = \sum_{i=1}^{N} (t_{\text{prefetch}_i} + t_{\text{compute}_i})$$

**이론적 speedup 조건**: `t_compute_i > t_prefetch_{i+1}` 이 모든 레이어에서 성립할 때, prefetch 비용이 compute 시간 뒤에 완전히 숨겨져(hidden) 로딩 오버헤드가 0에 수렴한다.

### 2.3 RT-Swap의 핵심 하드웨어 전제 조건

RT-Swap이 효과를 발휘하려면 다음 하드웨어 조건이 필수이다.

```
[dGPU 구조 — RT-Swap의 전제]

CPU DRAM ──────── PCIe ──────▶ GPU VRAM
                                    │
                              ┌─────┴──────┐
                              │  DMA 엔진  │  ← SM과 물리적으로 독립된 전용 하드웨어
                              └─────┬──────┘
                                    │
                              ┌─────┴──────┐
                              │  SM (연산) │
                              └────────────┘

→ DMA 엔진이 가중치 이동을 담당하는 동안, SM은 독립적으로 연산을 수행
→ 두 작업이 진정한 하드웨어 병렬(true parallelism) 실행 가능
```

| 구성 요소 | 역할 |
|-----------|------|
| **DMA 엔진** | CPU DRAM → GPU VRAM 데이터 이동 전담 (PCIe 경유) |
| **SM (Streaming Multiprocessors)** | GEMM 등 연산 전담 |
| **PCIe** | CPU-GPU 간 데이터 통로 (별도 물리 버스) |

논문은 72% 이상 추가 task set schedulability 향상, GPU 물리 메모리 대비 **96.2% 초과 메모리**를 사용하는 DNN task set까지 처리 가능함을 보였다.

---

## 3. 참고 논문 2 — Demand Layering (RTSS 2022)

> **Ji, M. et al.** "Demand Layering for Real-Time DNN Inference with Minimized Memory Usage"  
> *IEEE Real-Time Systems Symposium (RTSS), 2022*

### 3.1 논문이 해결하고자 하는 문제

임베디드 iGPU 환경(CPU-GPU 통합 메모리)에서 대형 DNN을 추론할 때 **GPU 메모리 용량 초과** 문제를 해결한다. CPU-GPU가 동일한 물리 DRAM을 공유하기 때문에, 기존의 "CPU DRAM을 스왑 공간으로 사용"하는 방식이 통하지 않는다.

### 3.2 핵심 아이디어: SSD를 파트너로 한 레이어 단위 on-demand 로딩

운영체제의 **Demand Paging**에서 영감을 받은 기법이다. 전체 모델을 메모리에 올리는 대신, **NVMe SSD에 모델 가중치를 저장하고, 레이어 실행 직전에 그 레이어만 로딩**한다.

```
[기존 방식]
전체 모델 (DRAM 상주) → [L0 실행] → [L1 실행] → ... → [LN 실행]
     ↑ 전체 모델 크기만큼 DRAM 점유

[Demand Layering]
NVMe SSD ──▶ L0 로딩 ──▶ [L0 실행] ──▶ L0 해제
NVMe SSD ──▶ L1 로딩 ──▶ [L1 실행] ──▶ L1 해제  ← 항상 1~2개 레이어만 DRAM 점유
             ...
```

실험에서 메모리 사용량 **96.5% 감소**를 달성하였다.

### 3.3 파이프라인 아키텍처: 3단계 중첩과 Copy Engine의 역할

Demand Layering이 iGPU에서도 성공적으로 작동한 핵심 이유는, 데이터 이동을 **SM(Streaming Multiprocessors)이 아닌 GPU Copy Engine(CE)** 이 처리하기 때문이다.

```
[Demand Layering 3단계 파이프라인 — 핵심 HW 구조]

         SSD                 CPU DRAM               GPU DRAM/SM
          │                     │                        │
  [Read]  ├── NVMe I/O 비동기 ──▶│ CPU 버퍼               │
  [Copy]  │                     ├── cudaMemcpyAsync ──▶   │ (GPU Copy Engine 담당)
  [Exec]  │                     │                        ├── GEMM 실행 (SM 담당)
          │                     │                        │

→ Read: CPU가 SSD에서 데이터 읽기  (CPU NVMe 컨트롤러 사용)
→ Copy: GPU Copy Engine(CE)이 CPU 버퍼 → GPU 메모리 전송
→ Exec: GPU SM이 연산 수행

[파이프라인 타임라인]

CE (Copy):  ─[L1 복사]─▶─[L2 복사]─▶─[L3 복사]─▶ ...
SM (Exec):             ─[L1 실행]─▶─[L2 실행]─▶─[L3 실행]─▶ ...
                          ↑               ↑
                   CE ∥ SM 하드웨어 병렬 실행
```

**왜 iGPU에서도 효과가 있는가:**  
iGPU(Xavier, Orin 등)도 내부에 SM과 물리적으로 독립된 **Copy Engine(CE)** 하드웨어를 보유한다.  
`cudaMemcpyAsync`를 비동기로 발행하면 CE가 DRAM 간 복사를 담당하며, SM은 이와 무관하게 연산을 계속한다.

> ⚠️ **Xavier의 Zero-Copy 최적화**: Xavier는 CPU-GPU가 물리 메모리를 공유하므로, CPU 버퍼와 GPU 버퍼가 동일한 물리 주소를 가리킬 수 있다 (zero-copy). 이 경우 Copy 단계가 사라져 2단계 파이프라인으로 단순화된다.

이를 통해 **near-zero overhead (< 1 ms 추가 지연)** 를 달성하면서도 메모리는 88.4% 절감하는 메모리-지연 트레이드오프를 제공한다.

---

## 4. 두 논문의 핵심 공통 아이디어

두 논문은 서로 다른 문제(메모리 용량 vs. 실시간 스케줄링)를 다루지만, **동일한 핵심 원리**를 공유한다.

| 항목 | RT-Swap | Demand Layering |
|------|---------|-----------------|
| **핵심 원리** | 레이어 i 계산 ∥ 레이어 i+1 가중치 이동 | 레이어 i 실행 ∥ 레이어 i+1 SSD→GPU 로딩 |
| **이동 매체** | CPU DRAM → GPU VRAM (PCIe) | NVMe SSD → CPU 버퍼 → GPU DRAM |
| **병렬 실현 수단** | **GPU DMA 엔진** (SM 독립) | **GPU Copy Engine (CE)** (SM 독립) |
| **목적** | VRAM 용량 확장 + 실시간 보장 | GPU 메모리 최소화 |
| **핵심 조건** | `t_compute > t_prefetch (DMA)` | `t_compute > t_SSD_read + t_copy (CE)` |

> **공통 수식**: 직렬 실행 시간 `Σ(t_load + t_compute)`를 비동기 중첩으로 `Σ max(t_load, t_compute)`으로 단축한다.

**두 논문의 공통된 핵심 전제**:  
"데이터 이동"과 "SM 연산"을 담당하는 **하드웨어가 물리적으로 분리**되어 있어야 한다.  
RT-Swap은 dGPU의 DMA 엔진을, Demand Layering은 iGPU의 Copy Engine(CE)을 이 분리 수단으로 사용한다.

---

## 5. Thor iGPU 환경에의 적용 전략

### 5.1 당초 적용 아이디어 및 핵심 오류

두 논문의 아이디어를 Alpamayo decode 단계에 다음과 같이 적용하려 하였다.

> **"레이어 i를 compute stream에서 실행하는 동안, prefetch stream에서 레이어 i+1의 가중치를 GPU L2 캐시로 워밍한다."**

```
[목표한 Thor iGPU 파이프라인 (당초 설계)]

Compute stream:  ─[Layer i 실행 (GEMM)]──────────────────▶─[Layer i+1 실행]─▶
Prefetch stream:           ─[Layer i+1 가중치 sum() L2 워밍]─▶

→ GEMM 실행 중 L2 캐시가 채워져, i+1 레이어는 DRAM 대신 L2에서 읽음
→ 이론적 speedup: 227 GB/s(DRAM) → 1,126 GB/s(L2)로 업그레이드
```

### 5.2 ⚠️ Demand Layering 이식 시 간과한 결정적 차이

Demand Layering은 분명히 **iGPU 환경**을 대상으로 설계된 논문이다. 그렇다면 왜 동일한 iGPU 환경인 Thor에서 이식에 실패했는가?

**답**: Demand Layering이 사용하는 `cudaMemcpyAsync` (Copy Engine 담당)와, 우리가 사용한 `tensor.sum()` (SM 담당)은 **전혀 다른 하드웨어 경로**이다.

| | Demand Layering의 prefetch | 본 실험의 prefetch |
|-|---------------------------|-------------------|
| **방법** | `cudaMemcpyAsync` (비동기 복사) | `tensor.sum()` (GPU 커널) |
| **처리 하드웨어** | **Copy Engine (CE)** — SM과 독립된 전용 HW | **SM** — compute와 동일한 자원 |
| **DRAM BW 점유** | CE 전용 경로로 분리됨 | SM과 함께 DRAM BW 경쟁 |
| **SM 부하** | **0%** (CE가 전담) | SM 사용량 증가 → compute 지연 |
| **결과** | compute ∥ copy 진정한 병렬 ✅ | compute + prefetch 직렬화 ❌ |

```
[Demand Layering이 실제로 하는 것]

          GPU 내부 하드웨어
    ┌─────────────────────────────┐
    │  Copy Engine (CE)           │  ← cudaMemcpyAsync 담당
    │  (SM과 물리적으로 독립)      │     DRAM 복사 중에도 SM 0% 소모
    │                             │
    │  SM (Streaming MP)          │  ← 연산 담당 (CE와 동시 실행)
    └─────────────────────────────┘

[본 실험이 실제로 한 것]

          GPU 내부 하드웨어
    ┌─────────────────────────────┐
    │  SM (Streaming MP)          │  ← compute stream: GEMM 실행
    │                             │     + prefetch stream: sum() 실행
    │  → 두 stream이 모두 SM 경쟁 │     → 동일 자원에서 시분할
    └─────────────────────────────┘
```

**이것이 이식 실패의 핵심 원인이다.**  
Demand Layering의 "비동기 파이프라인"을 차용하면서,  
그 파이프라인이 작동하는 **하드웨어 기반(CE)을 `sum()` 커널(SM)로 잘못 대체**하였다.

### 5.3 Thor iGPU의 Copy Engine 유무

Thor iGPU에도 Copy Engine은 존재한다. CUDA의 `asyncEngineCount`로 확인 가능하다.  
그러나 모든 가중치가 이미 DRAM에 올라와 있는 상황에서, **DRAM → DRAM 간 `cudaMemcpyAsync`는 L2 캐시 워밍 효과가 없다**.  

| 문제 | 설명 |
|------|------|
| **저장 매체 부재** | Demand Layering은 SSD → DRAM 경로가 핵심. Thor는 모든 가중치가 이미 DRAM에 있어 이 경로 자체가 없음 |
| **L2 워밍 불가** | DRAM → DRAM copy는 L2를 거치지 않는 streaming path를 사용. CE로 복사해도 L2에 잔류하지 않음 |
| **실질적 이동 없음** | CE가 DRAM 내에서 복사해도, GEMM은 원래 주소에서 읽기 때문에 접근 위치가 동일 |

### 5.4 파이프라인 성립 조건 사전 검증

파이프라인이 이론적으로 성립하려면 아래 조건이 필요하다.

$$t_{\text{compute per layer}} > t_{\text{prefetch per layer}}$$

이 조건을 사전에 측정으로 검증하는 **3단계 실험 체계**를 설계하였다.

---

## 6. 실험 설계 — 3단계 검증 체계

```
┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│     Phase 0-A          │   │     Phase 0-B          │   │       Phase 1          │
│   DRAM/L2 대역폭 측정  │   │  레이어 t_compute 측정 │   │  실제 비동기 파이프라인 │
│                        │   │                        │   │       실행 및 검증     │
│ 260526_prefetch_       │   │ 260524_layer_compute   │   │ 260524_async_pipeline  │
│ effect_test.py         │   │ _profile.py (v3)       │   │ .py (v3)               │
│                        │   │                        │   │                        │
│ L2 = 1,126 GB/s        │   │ t_compute = 2.62 ms    │   │ Baseline vs Async      │
│ DRAM = 227 GB/s        │   │ 36/36 조건 성립 ✅     │   │ speedup 측정           │
└────────────┬───────────┘   └──────────┬─────────────┘   └───────────┬────────────┘
             │                          │                               │
             ▼                          ▼                               ▼
      t_prefetch 기준값 산출    파이프라인 가능 여부 판단        실제 효과 검증
```

---

## 7. 대역폭 측정 실험 (Phase 0-A)

### 7.1 측정 방법론

**대역폭 (Bandwidth)** 의 정의:  
> 단위 시간당 물리적 메모리(DRAM)에서 연산 장치(CPU 또는 GPU)로 데이터를 얼마나 많이 가져올 수 있는가 (단위: GB/s)

**캐시 히트 원천 차단 방법**:
- L3 캐시 용량(16 MB), GPU L2 캐시(32 MB)를 크게 초과하는 **500 MB 크기의 독립 텐서**를 각 연산 장치마다 할당
- 동일 텐서에 `x.sum()` 연산을 반복 → **Cache Thrashing** 발생으로 매번 DRAM에서 로딩 강제
- 결과적으로 순수 DRAM 대역폭만 측정 가능

**멀티코어 CPU-GPU 동시 측정 (contention 실험)**:
- 다수의 CPU 코어와 GPU가 동시에 DRAM에 접근하는 상황에서 각 연산 장치의 유효 대역폭 측정
- 코어 수를 늘릴수록 전체 DRAM 대역폭이 어떻게 분배되는지 관찰

### 7.2 측정 결과

**실험 파일**: `scripts/profiling/260526_prefetch_effect_test.py`

| 측정 항목 | 결과 | 의미 |
|-----------|------|------|
| **GPU L2 캐시 대역폭** | **1,126 GB/s** | L2 → SM 전송 속도 |
| **GPU DRAM 대역폭** | **227 GB/s** | DRAM → GPU 전송 속도 |

### 7.3 이론 prefetch 시간 산출

위 대역폭 수치를 기반으로 레이어별 prefetch 소요 시간을 산출하였다.

```
Transformer 레이어 가중치 크기:
  MLP 가중치 1개 (gate_proj) = 100.7 MB
  전체 레이어 가중치 합      = 385.9 MB

DRAM 기준 prefetch 시간:
  MLP 1개 : 100.7 MB ÷ 227 GB/s = 0.444 ms  ← pipeline 성립 판단 기준선
  전체 레이어: 385.9 MB ÷ 227 GB/s = 1.70 ms

L2 기준 prefetch 시간 (캐시 히트 시):
  전체 레이어: 385.9 MB ÷ 1,126 GB/s = 0.343 ms
```

L2 캐시 대역폭(1,126 GB/s)이 DRAM 대역폭(227 GB/s)보다 **약 5배 빠르다**는 사실이 prefetch 아이디어의 이론적 근거이다.

---

## 8. 레이어 compute time 측정 (Phase 0-B)

### 8.1 측정 방법

**실험 파일**: `scripts/profiling/260524_layer_compute_profile.py` (v3, hook-based)

CUDA Event를 각 Transformer 레이어의 앞뒤에 `forward_pre_hook` / `forward_hook`으로 등록하여, `lm_model.forward()` 실행 중 per-layer elapsed time을 수집하였다.

> **구현 이슈**: 초기 v1/v2는 `Qwen3VLTextDecoderLayer`를 직접 호출 시도 → 실패.  
> Qwen3VL 구조에서 `rotary_emb`가 model-level에만 존재하고 개별 레이어에 없기 때문.  
> → v3에서 `lm_model.forward()`를 실행 주체로 사용하는 hook 방식으로 해결.

### 8.2 측정 결과

| 레이어 | mean_ms | std_ms | 판정 (조건: >0.444 ms) |
|--------|---------|--------|------------------------|
| L00 | 2.931 | 0.130 | ✅ overlap 가능 |
| L01 | 2.642 | 0.166 | ✅ overlap 가능 |
| L17 | 2.538 | 0.143 | ✅ overlap 가능 |
| L24 | 2.628 | 0.204 | ✅ overlap 가능 |
| L35 | 2.670 | 0.124 | ✅ overlap 가능 |

| 집계 항목 | 값 |
|-----------|-----|
| 레이어 평균 compute time | **2.62 ms** |
| 표준편차 | ~0.15 ms (CV ≈ 6%) |
| overlap 가능 레이어 수 | **36 / 36 (100%)** |
| 1 decode step 합산 | **94.28 ms** |
| 65 step 전체 추론 | **6.13 s** |

### 8.3 파이프라인 조건 판단

$$t_{\text{compute}} = 2.62 \text{ ms} \gg t_{\text{prefetch (MLP 1개)}} = 0.444 \text{ ms} \quad (\text{약 5.9배})$$

$$t_{\text{compute}} = 2.62 \text{ ms} > t_{\text{prefetch (전체 레이어)}} = 1.70 \text{ ms} \quad (\text{약 1.5배})$$

**→ 이론적 파이프라인 조건: 전 레이어(36/36) 성립 ✅**

---

## 9. 비동기 파이프라인 실행 결과 (Phase 1)

### 9.1 구현 방식

**실험 파일**: `scripts/inference/260524_async_pipeline.py` (v3, hook-based)

| 구현 요소 | 내용 |
|-----------|------|
| Prefetch 방법 | `tensor.view(-1).sum()` on `prefetch_stream` |
| 동기화 | `torch.cuda.current_stream().wait_stream(prefetch_stream)` |
| 트리거 | 각 레이어 `forward_pre_hook` 에서 다음 레이어 prefetch 발행 |

### 9.2 측정 결과

| 실행 모드 | ms / step | 65-step 합산 |
|-----------|-----------|--------------|
| **Baseline (순차)** | **95.96 ms** | **6,237 ms (6.24 s)** |
| **Async Pipeline** | **147.82 ms** | **9,609 ms (9.61 s)** |
| **Speedup** | **0.6491×** | **−3,371 ms (54% 더 느림)** |

**→ 파이프라인이 이론 예측과 정반대로 54% 느려지는 결과 발생 ❌**

---

## 10. 실패 원인 분석 — 3가지 층위

### 원인 1 (기술적): `tensor.sum()`은 "읽기 2번"이다

prefetch 수단으로 사용한 `sum()` 연산이 L2 캐시를 채우는 것이 아니라, 오히려 **추가 DRAM 읽기**를 유발하였다.

```
[기대했던 동작]
sum()  → 레이어 가중치를 L2에 적재 → 이후 GEMM이 L2에서 빠르게 읽음

[실제 동작]
sum()  → 386 MB 전체를 DRAM에서 읽음
         (스트리밍 방식, L2엔 마지막 ~32 MB만 잔류)
GEMM   → 386 MB를 처음부터 다시 DRAM에서 읽음
         (앞 354 MB는 L2 miss → DRAM 재접근)

→ 실제 DRAM 트래픽 = 원래 386 MB × 2 = 772 MB / layer
```

추가 지연 추정:

```
36 layers × 386 MB = 13.9 GB 추가 DRAM 트래픽
13.9 GB ÷ 227 GB/s = 약 61 ms 추가 부하

실측 overhead: 147.82 − 95.96 = 51.86 ms  ← 이론값과 정합
```

### 원인 2 (구조적, 치명적): `tensor.sum()`은 CE가 아닌 SM 커널이다

Demand Layering은 iGPU 환경임에도 효과를 냈다. 이유는 **Copy Engine(CE)** 이 SM과 물리적으로 독립된 하드웨어이기 때문이다. 본 실험은 CE 대신 SM 커널(`tensor.sum()`)을 사용하여 이 이점을 스스로 포기하였다.

```
[Demand Layering 논문의 실제 구현 — iGPU Xavier]

┌──────────────────────────────────────────────────┐
│              iGPU 내부 (통합 DRAM)              │
│                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │  Copy Engine (CE)   │  │  SM (연산 유닛)    │  │
│  │  cudaMemcpyAsync  │  │  GEMM, Attention  │  │
│  │  담당             │  │  담당             │  │
│  └──────────┬──────────┘  └──────────┬──────────┘  │
│             │  물리적으로 독립       │              │
│             └────────────────────────┘              │
│                  통합 DRAM                       │
└──────────────────────────────────────────────────┘

→ CE(copy) ∥ SM(compute): 진정한 하드웨어 병렬 ✅
→ iGPU여도 CE ≠ SM 이므로 중첩 가능


[본 실험의 구현 — prefetch를 SM 커널로 대체]

┌──────────────────────────────────────────────────┐
│              iGPU 내부 (통합 DRAM)              │
│                                                │
│  ┌─────────────────────────────────────────────┐  │
│  │              SM (연산 유닛)                │  │
│  │  compute stream: GEMM, Attention           │  │
│  │  prefetch stream: tensor.sum() 커널        │  │
│  │  → 두 stream이 동일한 SM 자원 경쟁         │  │
│  └─────────────────────────────────────────────┘  │
│                  통합 DRAM                       │
└──────────────────────────────────────────────────┘

→ SM(compute) + SM(sum): 동일 자원 시분할 경쟁 ❌
→ DRAM BW 경쟁 + SM 경쟁 → 두 작업 모두 느려짐
```

**핵심 실수**: Demand Layering의 "비동기 prefetch" 개념을 차용하면서,  
그것이 작동하는 하드웨어(Copy Engine)를 SM 커널(`tensor.sum()`)로 잘못 대체하였다.  
결과적으로 논문의 핵심 전제인 **"데이터 이동 하드웨어 ≠ 연산 하드웨어"** 조건을 스스로 위반하였다.

### 원인 3 (인식 오류): `t_compute = 2.62 ms`의 정체

Phase 0-B에서 측정한 2.62 ms를 "계산 시간"으로 해석한 것이 오류였다.

실제 FLOPs 분석:

```
seq=1 decode 기준, 레이어 1개의 FLOPs:
  Q proj     [1, 4096] × [4096, 4096]  ≈   33 M FLOPs
  MLP gate   [1, 4096] × [4096, 14336] ≈  117 M FLOPs
  전체 합산                             ≈  450 M FLOPs

Thor iGPU 이론 성능: ~100 TFLOPS
순수 compute 시간 = 450 M ÷ 100 T = 0.0045 ms  ← 사실상 무시 가능
```

| 항목 | 값 |
|------|----|
| 실측 레이어 시간 | 2.62 ms |
| 이론 DRAM 읽기 시간 (386 MB ÷ 227 GB/s) | 1.70 ms |
| **유효 DRAM 대역폭** (386 MB ÷ 2.62 ms) | **147 GB/s** (피크의 **65%**) |
| 순수 compute (FLOPs 기반 이론) | ~0.005 ms (**2.62 ms의 0.2%**) |

**→ 레이어 실행 시간 2.62 ms의 99.8%가 DRAM 읽기 대기 시간이었다.**

따라서 compute stream과 prefetch stream이 "overlap"할 수 있는 독립적인 계산 시간이 사실상 존재하지 않았으며, 두 stream 모두 동일한 DRAM 대역폭을 경쟁하는 구조가 될 수밖에 없었다.

---

## 11. 핵심 발견: Decode는 Memory-Bandwidth-Bound

### 11.1 Roofline 분석

```
Roofline 분석 (Thor iGPU):

연산 강도 (Arithmetic Intensity):
  I = FLOPs / Bytes = 450 M / 386 M ≈ 1.16 FLOPs/Byte

Ridge Point (연산 제한 ↔ 메모리 제한 경계):
  Ridge = 피크 TFLOPS ÷ 피크 BW = 100 T ÷ 0.227 T = 441 FLOPs/Byte

I = 1.16  ≪  Ridge = 441
→ 완전한 Memory-Bandwidth-Bound 영역에 위치
```

### 11.2 확인된 핵심 사실 요약

| 확인 사실 | 수치 | 의미 |
|-----------|------|------|
| 레이어 실행 시간 대비 순수 FLOPs 비율 | 0.2% | 거의 전부 메모리 대기 |
| 유효 DRAM 대역폭 | **147 GB/s** | 피크(227 GB/s)의 65% |
| L2 캐시 크기 vs 레이어 가중치 | 32 MB vs 386 MB | L2 워밍의 근본적 한계 |
| iGPU 내 DMA 엔진 | **없음** | RT-Swap 전제 조건 불성립 |
| `tensor.sum()` 연산 성격 | SM 사용 compute 커널 | prefetch가 아닌 자원 경쟁 |

### 11.3 논문 아이디어 적용의 구조적 한계 — 수정된 분석

| 논문 | 전제 환경 | 논문이 쓰는 HW 분리 수단 | 본 실험 | 불일치 원인 |
|------|-----------|--------------------------|---------|-------------|
| **RT-Swap** | dGPU + PCIe | GPU DMA 엔진 (SM 독립) | SM 커널 `sum()` | SM ≠ DMA 엔진 |
| **Demand Layering** | iGPU + **NVMe SSD** | GPU **Copy Engine (CE)** + SSD | SM 커널 `sum()`, SSD 없음 | SM ≠ CE, 저장 매체 부재 |
| **공통 전제** | 데이터 이동 HW ≠ 연산 HW | CE or DMA 엔진 | SM (연산 HW와 동일) | 자원 분리 불가 |

> ⚠️ **Demand Layering이 iGPU에서 성공한 이유는 "iGPU이기 때문"이 아니라,  
> "Copy Engine(CE)이 SM과 독립된 하드웨어이기 때문"이다.**  
>
> 본 실험의 근본 오류는 "비동기 파이프라인"이라는 소프트웨어 개념은 차용하면서,  
> 그것이 실현되는 하드웨어 경로(CE)를 SM 커널로 대체한 데 있다.

**올바른 이식 경로 (가능하다면):**
```
[Demand Layering을 Thor에 올바르게 이식하려면]

조건 1: SSD에 가중치를 저장할 것 (현재: 전체 DRAM 상주)
조건 2: cudaMemcpyAsync(CE)로 prefetch 발행할 것 (현재: sum() SM 커널)
조건 3: 단, DRAM BW가 compute + copy 동시 처리에 충분한지 확인 필요

하지만 Thor의 현실적 제약:
  - 모든 가중치가 이미 DRAM에 상주 → SSD 로딩 필요성 없음 (메모리 여유 128 GB)
  - DRAM → DRAM copy는 CE를 사용해도 BW를 추가로 소모
  - DRAM BW 자체(227 GB/s)가 이미 병목 → 추가 copy 트래픽은 오히려 악화

결론: Thor 환경에서 Demand Layering의 핵심 이점(SSD→DRAM의 새로운 데이터 경로)이
       재현될 여지가 없다. 접근 자체의 재설계가 필요하다.
```

---

## 12. 결론 및 다음 연구 방향

### 12.1 이번 실험의 가치

이번 실험 체인(대역폭 측정 → compute time 측정 → 파이프라인 시도)은 원하는 speedup을 달성하는 데 실패하였으나, 다음의 중요한 사실들을 정량적으로 확인하였다.

1. **Alpamayo decode 병목의 정확한 정체**: 순수 memory-bandwidth-bound (99% DRAM 읽기)
2. **유효 대역폭 수치**: 147 GB/s (피크의 65%) — 소규모 GEMM의 DRAM 접근 비효율성 반영
3. **레이어 간 일관성**: 36개 전 레이어가 2.3~2.9 ms 범위로 매우 안정적
4. **올바른 최적화 방향**: prefetch/overlap이 아닌, **DRAM 트래픽 자체를 줄이는 접근** 필요

### 12.2 다음 연구 방향

병목이 DRAM 대역폭임이 명확해진 만큼, 다음 접근들이 유효할 것으로 판단된다.

| 접근법 | 예상 효과 | 근거 |
|--------|-----------|------|
| **INT4 가중치 양자화** | 2~4× | DRAM 트래픽 1/4로 감소 → 직접 효과 |
| **다중 샘플 배치 (`num_samples > 1`)** | 2~4× / sample | seq 길이 증가 → compute-bound로 전환 |
| **컴포넌트 간 파이프라인** | 1.5~2× | Vision Encoder ∥ VLM Decode 중첩 |
| **Speculative Decoding** | 2~4× | Draft+Verify → seq > 1 달성 |
| **CUDAGraph** | ~1.2× | kernel launch overhead 제거 |

### 12.3 최종 정리

```
실험 결론 한 줄 요약:

  RT-Swap / Demand Layering 아이디어를 Thor iGPU에 적용한 결과,
  파이프라인 조건은 이론상 성립했으나(36/36 레이어),
  실제 speedup은 0.65× (54% 더 느림)이었다.
  
  핵심 오류: Demand Layering의 비동기 prefetch는 Copy Engine(CE)이 담당하지만,
             본 실험은 이를 SM 커널(tensor.sum())로 잘못 대체 → CE ≠ SM
  추가 문제: Thor는 전체 가중치가 이미 DRAM 상주 → SSD→DRAM 새 경로 자체 없음
  수확: Decode가 완전한 memory-bandwidth-bound임을 정량적으로 확인 (유효 BW 147 GB/s).
  방향: DRAM 트래픽 감소 (INT4, 배치 증대) 로의 전환.
```

---

## 부록: 실험 파일 목록

| 파일 | 목적 | 주요 결과 |
|------|------|-----------|
| `scripts/profiling/260526_prefetch_effect_test.py` | L2/DRAM 대역폭 측정 | L2 = 1,126 GB/s, DRAM = 227 GB/s |
| `scripts/profiling/260524_layer_compute_profile.py` | 레이어별 t_compute 측정 | 2.62 ms/layer, 36/36 조건 통과 |
| `scripts/inference/260524_async_pipeline.py` | 비동기 파이프라인 PoC | speedup = 0.65× (실패) |

| 결과 파일 | 내용 |
|-----------|------|
| `profiling_results/260524_layer_compute_profile.json` | 레이어별 상세 측정값 (36개) |
| `profiling_results/260524_async_pipeline.json` | Baseline vs Async 비교 데이터 |

---

*보고서 작성일: 2026-05-27*  
*Alpamayo 프로젝트 — 4주차 실험 결과*
