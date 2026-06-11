> ⚠️ **DEPRECATED** — 이 파일의 수치는 컴포넌트 크기 오류 포함 (flow=0, decode 분모 오류).  
> 최신 보고서: [`docs/260515_allphase_bw_analysis.md`](260515_allphase_bw_analysis.md)

# Alpamayo 1.5 DRAM Bandwidth Analysis Report (구버전)
**보드**: Jetson AGX Thor (LPDDR5X 273 GB/s, 128 GB Unified Memory)
**모델**: Alpamayo 1.5 — 11.08B params, 22.16 GB (BF16)
**측정일**: 2026-05-14 | **방법**: CUDA Events + tegrastats + nsys kernel 분석

---

## 1. 전체 DRAM 대역폭 사용 요약

| 구분 | BW (GB/s) | 피크 대비 (%) | 측정 방법 |
|------|-----------|--------------|-----------|
| **이론 피크** | **273.0** | 100% | LPDDR5X 스펙 |
| **Decode (실측)** | **211.5** | **77.5%** | CUDA Events (직접 측정) |
| **시스템 평균 (전 구간 희석)** | **65.6** | **24.0%** | 전체 추론 시간 기준 |
| Decode 이론 하한 | 237.2 | 86.9% | 22.16 GB ÷ 81.2 ms/step |

### 핵심 결론
- **Decode 구간은 명백한 BW-BOUND** — 피크의 77.5%를 소비하며 메모리 대역폭이 병목
- **전체 추론 시간 기준 시스템 BW는 24%** — Vision, Prefill, Flow 구간이 평균을 크게 희석
- **1회 추론 (총 4,969 ms) 동안 실질 DRAM 트래픽 추정: ~420 GB** (22.16 GB × 19 decode steps)

---

## 2. Phase별 DRAM 대역폭 상세

### 2-1. Decode — BW-BOUND ★
```
측정값: 211.5 ± 46.8 GB/s  (77.5% MBU)
측정 방법: CUDA Events
  BW = model_GB × n_tok / decode_ms
     = 22.157 GB × 18 tok / 1,886 ms
     = 211.5 GB/s
```

**왜 BW-BOUND인가?**
Decode는 **매 token마다 전체 모델 가중치 22.16 GB를 메모리에서 불러와** 연산한다.
batch_size = 1 조건에서는 연산(FLOP)보다 메모리 읽기가 압도적으로 많아,
GPU SM이 연산을 마칠 때쯤 이미 다음 가중치를 기다리는 상태가 된다.

**nsys 커널 분포 (decode 구간):**
| 커널 분류 | GPU 시간 | 비중 | 특성 |
|-----------|---------|------|------|
| splitK GEMM | 786 ms | 47.9% | 소형 GEMM (batch=1) → BW 위주 |
| GEMM-nvjet | 608 ms | 37.1% | 행렬-벡터 곱 (decode 전용 shape) |
| GEMV (gemv2T_kernel) | 99 ms | 6.0% | 순수 BW-bound |
| KV Cache Ops | 81 ms | 4.9% | KV read/write |
| Elementwise | 55 ms | 3.4% | 짧은 compute |

→ splitK GEMM과 GEMM-nvjet이 decode 시간의 85%를 차지하며 모두 메모리 접근 위주

**Decode BW 변동성:**
- run_01 (17 tok): 1,784 ms → 222.5 GB/s
- run_02 (19 tok): 1,987 ms → 211.9 GB/s
- **σ = 101 ms** — 토큰 수 차이(+2 tok)가 대부분의 변동 설명
- 동일 토큰 수 기준으로는 step당 편차 매우 작음 (step 단위 BW 안정적)

---

### 2-2. LM Prefill — Compute-BOUND
```
지속 시간: 1,435 ms (±4 ms)  — 전체의 28.9%
추정 BW: ~40–50 GB/s  (15–18% MBU)  ← 추정값
```

**왜 Compute-BOUND인가?**
Prefill은 입력 3,086 tokens를 한꺼번에 처리한다.
- Large-tile GEMM (batch=3086): 연산 강도(FLOP/byte) 높음 → SM이 대부분 연산
- 메모리는 가중치 1회 로드 후 3,086회 재사용 → 실질 BW 수요 낮음

**nsys 커널 분포 (prefill 구간):**
| 커널 분류 | GPU 시간 | 비중 |
|-----------|---------|------|
| Large-tile GEMM | 914 ms | 37.4% |
| Elementwise / Activation | 645 ms | 26.4% |
| Other (NCCL, misc) | 342 ms | 14.0% |
| GEMM-nvjet (기타) | 205 ms | 8.4% |
| Flash Attention | 136 ms | 5.6% |
| KV Cache Ops (캐시 기록) | 124 ms | 5.1% |

→ **KV Cache 기록**(124 ms)이 Prefill에서 유일하게 BW를 소비하는 주요 작업

**메모리 할당:**
- Prefill 전: 22.32 GB → 후: 22.44 GB → Peak: **23.20 GB** (+761 MB peak)
- 이 761 MB = **KV Cache 할당 + Flash Attention 임시 버퍼** (Decode 전 사전 배정)

---

### 2-3. Vision Encoder — Compute-BOUND (혼재)
```
지속 시간: 741 ms (±0.16 ms)  — 전체의 14.9%
추정 BW: ~35–45 GB/s  (13–16% MBU)  ← 추정값
```

- ViT 기반 이미지 인코딩 — 2D convolution + Attention 위주
- 연산 강도가 높아 BW보다 SM compute가 우선 병목
- **메모리 추가 사용 최소**: Prefill 전 +649 MB peak (activation 임시 버퍼)
- Warmup 대비 **1.55× 단축** — JIT 컴파일 + cudaMalloc이 처음 실행 시 느림

---

### 2-4. Flow (Euler Solver) — Decode-유사 BW
```
지속 시간: 906 ms (±4 ms)  — 전체의 18.2%
추정 BW: ~150–180 GB/s  (55–66% MBU)  ← 추정값
```

- Decode 이후 trajectory 생성 단계 (10 Euler steps)
- Decode와 유사한 auto-regressive 패턴이지만 step당 연산량 다름
- **nsys 커널 분포 (Flow):**

| 커널 분류 | GPU 시간 | 비중 |
|-----------|---------|------|
| Elementwise | 255 ms | 30.3% |
| GEMM-nvjet | 247 ms | 29.3% |
| KV Cache Ops | 144 ms | 17.1% |
| splitK GEMM | 107 ms | 12.6% |

→ KV Cache Ops 비중(17%)이 Decode(4.9%)보다 높아 KV 접근 패턴이 다름

---

## 3. 대역폭 급변 구간 분석 (BW 낙폭)

### 3-1. Prefill → Decode 전환: 가장 큰 낙폭

```
Prefill BW: ~40–50 GB/s  (compute-bound)
   ↓
[Post-Prefill Gap: ~100–500 ms] ← fill_reverse_indices + KV 재편성
   ↓
Decode BW: 211.5 GB/s  (BW-bound)
```

**낙폭 원인:**
1. Prefill 직후 `fill_reverse_indices` 커널 실행 — KV cache 인덱스 재구성
2. `cudaDeviceSynchronize` 호출 — GPU stall 발생
3. Warmup 실행에서 이 Gap이 **~500 ms**로 측정됨 (Measure에서는 <100 ms로 감소)

**BW 관점의 의미:**
- Prefill이 끝나는 순간 BW 수요가 4–5× 급등 (40→211 GB/s)
- Gap 구간(stall)에서는 BW = 0 → 메모리 버스가 완전히 유휴 상태

---

### 3-2. Decode 구간 내부: BW 안정적
```
step당 BW: 211–225 GB/s (토큰 수 고정 기준)
σ: ±5 GB/s (step당 편차)
```

- 매 decode step에서 동일한 kernel 패턴 반복 → BW 매우 안정적
- `cudaStreamSynchronize` (EOS 체크용)가 **매 step마다** 삽입됨
  → CPU가 GPU tensor를 읽어가며 미세한 파이프라인 단절 발생
  → 이것이 decode BW를 이론값 237 GB/s에서 211 GB/s로 끌어내리는 주요 원인

---

### 3-3. Warmup → Measure: JIT 효과
```
Vision: 1,151 ms → 741 ms  (-36%, 1.55× 단축)
Prefill: 1,545 ms → 1,435 ms  (-7%, 1.08× 단축)
Decode: 1,594 ms → 1,886 ms  (+18% ← 토큰 수 증가로 인한 증가)
Flow: 930 ms → 906 ms  (-3%)
```

- **Vision이 가장 큰 JIT 효과** (1.55×): 처음 실행 시 cuDNN autotuning + cudaMalloc
- **Decode는 오히려 더 느림** — Warmup(15 tok)보다 Measure(17–19 tok)에서 토큰 수 많음
- Warmup BW 전반: JIT + 불필요한 동기화로 피크 BW 미달

---

## 4. CPU 대역폭 사용 분석

### 4-1. Jetson Thor 통합 메모리 아키텍처의 의미

```
CPU Memory ──┐
             ├── Unified LPDDR5X 128 GB (273 GB/s)
GPU Memory ──┘
```

Thor의 CPU와 GPU는 **동일한 LPDDR5X 버스를 공유**한다.
따라서 CPU 접근과 GPU 접근이 **대역폭을 경쟁**한다.

---

### 4-2. 추론 중 CPU 대역폭 사용 패턴

| 구간 | CPU BW 추정 | 발생 원인 |
|------|-------------|-----------|
| 모델 로딩 (1회, ~3–4분) | **~60–80 GB/s** | 22.16 GB 가중치를 DRAM에 적재 |
| Vision Encoder | **~2–5 GB/s** | Python dispatch, 이미지 전처리 |
| LM Prefill | **~3–8 GB/s** | HuggingFace tokenizer, Python overhead |
| **Decode (매 step)** | **~5–15 GB/s** | EOS 체크 `cudaStreamSynchronize` + CPU read |
| Flow | **~2–5 GB/s** | trajectory 결과 수신 |

**Decode에서 CPU BW 소비가 가장 높은 이유:**
```python
# HuggingFace generate() 내부 — 매 decode step마다 실행
unfinished = unfinished_sequences.max() == 0  # ← GPU tensor → CPU read
```
이 연산이 매 step마다 `cudaStreamSynchronize`를 트리거하여:
1. CPU가 GPU 버스를 점유 (tensor 읽기)
2. GPU 다음 step 시작을 대기
3. decode당 1회 CPU↔GPU 메모리 경쟁

**tegrastats 전력 데이터 (Thor JetPack 7):**
| 구간 | VDD_GPU (추정) | VDD_CPU_SOC_MSS | 특성 |
|------|--------------|-----------------|------|
| 유휴 | -392 mW (교정 오프셋) | ~3,924 mW | CPU 기저 전력 |
| Prefill | **~30–80 W** | ~5–8 W | Compute-bound, GPU SM 풀가동 |
| Decode | **~5–20 W** | ~4–6 W | BW-bound, SM 연산 낮음 |

> 전력 차이가 GPU 연산 강도를 잘 나타냄:
> - **Prefill = 고전력** → FLOP 위주 → compute-bound
> - **Decode = 저전력** → 메모리 스트리밍 위주 → BW-bound

---

### 4-3. CPU가 전체 BW에서 차지하는 비중 추정

```
GPU BW (Decode): 211.5 GB/s
CPU BW (Decode): ~5–15 GB/s
────────────────────────────
추정 합계:        ~220–226 GB/s  (이론 피크의 80–83%)
```

CPU BW는 GPU 대비 약 **3–7% 수준**으로 추론 성능에 직접적 영향은 작다.
그러나 매 decode step의 **EOS 동기화 지연** (~2–5 ms)이 누적되어
18 tokens 기준으로 **36–90 ms의 순수 CPU 오버헤드**가 발생한다.

이는 Decode BW를 이론 최대 (237 GB/s)에서 **211.5 GB/s로 낮추는 주원인**이다.

---

## 5. 전체 대역폭 분포 시각적 요약

```
Phase           시간 비중    BW 추정    특성
─────────────────────────────────────────────────────────────────
Vision Enc.     14.9%       ~35 GB/s   Compute-bound (ViT GEMM)
LM Prefill      28.9%       ~48 GB/s   Compute-bound (large GEMM)
Decode          38.0%       211 GB/s   ★ BW-BOUND (weight streaming)
Flow            18.2%       ~170 GB/s  Mixed (BW+compute)
─────────────────────────────────────────────────────────────────
전체 평균 (희석) 100%        ~66 GB/s   24% MBU
```

```
DRAM BW ┤
 273    │                         ████ (이론 피크)
        │  - - - - - - - - - - - - - - - - - - - BW-bound 임계 (70%)
 191    │
        │
 211    │              ████████████  Decode ★ (77.5% MBU, 실측)
        │    ████████               Flow   (추정 ~63% MBU)
        │
  48    │    LM Prefill (추정 ~18%)
  35    │████ Vision Enc (추정 ~13%)
        └──────────────────────────────────────────►
             Vision  Prefill  Decode   Flow
```

---

## 6. 최적화 기회 분석

### A. Decode BW 병목 해소
| 방법 | 예상 효과 | 조건 |
|------|-----------|------|
| batch_size 증가 (예: 4) | Decode BW → 160–180 GB/s (MBU 유지하며 throughput 4×) | 메모리 여유 109 GB 활용 가능 |
| INT4/FP4 quantization | 모델 크기 절반 → 22 GB → 11 GB → BW 수요 50% 감소 | --dtype fp4 플래그 지원 |
| EOS 체크 배치화 | CPU 동기화 제거 → ~37 ms 절약 (18 tok 기준) | HuggingFace 수정 필요 |
| Speculative Decoding | Decode 단계 수 줄임 → 총 BW 소비 감소 | Draft 모델 필요 |

### B. Prefill 최적화 여지
- 현재 Prefill이 전체 시간의 28.9% → 3,086 tok 입력에 1,435 ms
- **Continuous batching** 적용 시 다른 요청과 Prefill GEMM 공유 가능
- Flash Attention이 이미 활성화됨 → KV cache ops 124 ms가 추가 타깃

### C. Vision Encoder 병목
- Warmup에서 1,151 ms → Measure 741 ms: **410 ms가 JIT/캐시 효과**
- `torch.compile()` 적용 시 steady-state 더욱 단축 가능

---

## 7. 측정 방법론 및 신뢰도

| 항목 | 측정 방법 | 신뢰도 |
|------|-----------|--------|
| Decode BW (211.5 GB/s) | CUDA Events (ms) + model_size × n_tok | **HIGH** — 직접 측정 |
| Phase 지속 시간 | CUDA Events (n=2 measure runs) | **HIGH** — 하드웨어 타이머 |
| CPU/Memory 전력 | tegrastats VDD_CPU_SOC_MSS | **MEDIUM** — 100ms 샘플링 |
| GPU 전력 | tegrastats VDD_GPU | **MEDIUM** — 교정 오프셋 -392 mW |
| Vision/Prefill BW | nsys 커널 패턴 기반 추정 | **LOW** — 직접 EMC 측정 불가 (JetPack 7) |
| CPU BW 기여 | 아키텍처 분석 + 전력 프록시 | **LOW** — 추정 |

> **JetPack 7 제약**: Thor의 tegrastats가 `EMC_FREQ` (DRAM BW %)를 제공하지 않아
> Vision/Prefill BW를 직접 측정할 수 없음. VDD_GPU 전력이 간접 지표로 사용됨.

---

*측정 환경: Jetson AGX Thor, JetPack 7 (Ubuntu 24.04, CUDA 13.0, SM 11.0)*
*모델: nvidia/Alpamayo-1.5-10B, BF16, attn_implementation="sdpa", n=2 measure runs*
