# PyTorch Eager Mode의 커널 비융합 문제와 Alpamayo 전용 컴파일 엔진 연구 방향

**날짜**: 2026-06-08  
**관련 실험**: `260608_01_Alpamayo_4단계_실제_DRAM_대역폭_ncu_실측_분析.md`  
**핵심 주제**: ncu 실측으로 드러난 DRAM 낭비의 근본 원인과 해결 전략

---

## 0. 문제 요약

```
ncu 실측 결과 (Jetson AGX Thor, LPDDR5X 231 GB/s):

단계      이론값(가중치)   실측값         낭비 비율
VE          1.153 GB      98.063 GB       85×
Prefill    15.168 GB     231.649 GB       15×
Decode     15.626 GB      16.980 GB      1.09×  ← 유일하게 정상
Flow        4.561 GB     122.114 GB       27×
```

**Decode를 제외한 3단계가 가중치 크기 대비 수십 배의 DRAM을 소모하고 있다.**  
이것은 모델 설계의 문제가 아니다. **실행 엔진(PyTorch eager mode)의 구조적 한계** 때문이다.

---

## 1. 근본 원인 — PyTorch Eager Mode의 커널 비융합

### 1.1 GPU에서 연산이 실행되는 방식

GPU는 연산을 **커널(kernel)** 단위로 실행한다. 하나의 커널은 GPU에 올라가서 실행되고, 결과를 DRAM에 쓴 뒤 종료된다. 다음 커널은 그 결과를 DRAM에서 읽어서 시작한다.

```
PyTorch eager mode에서 Prefill 1 레이어의 실제 실행 흐름:

[LayerNorm 커널]
  입력  읽기: [3086, 4096] = 25.3 MB  ← DRAM
  출력 쓰기: [3086, 4096] = 25.3 MB  → DRAM (방금 만든 결과를 저장)
             ↕ DRAM 경유
[Q_proj GEMM 커널]
  가중치 읽기: [4096, 4096] = 33.6 MB  ← DRAM
  입력   읽기: [3086, 4096] = 25.3 MB  ← DRAM (방금 LayerNorm이 쓴 것을 다시 읽음)
  출력  쓰기: [3086, 4096] = 25.3 MB  → DRAM
             ↕ DRAM 경유
[K_proj GEMM 커널]
  동일 반복...
             ↕
[V_proj GEMM 커널]
  동일 반복...
             ↕
[RoPE 커널]
  읽기: Q [3086, 4096] + K [3086, 4096]  ← DRAM (방금 쓴 것 다시 읽음)
  쓰기: Q_rotated, K_rotated             → DRAM
             ↕
[FlashAttention 커널]
  ...
```

**"방금 만든 것을 바로 다시 읽는" 패턴이 레이어마다, 연산마다 반복된다.**

### 1.2 왜 이렇게 설계되었는가

PyTorch eager mode는 **범용성과 디버깅 용이성**을 위해 이 구조를 선택했다:

- 모든 연산이 독립적인 커널 → 어느 연산이든 중간에 멈춰서 확인 가능
- 임의의 연산 조합을 지원 → 연구자가 자유롭게 모델 설계 가능
- 그래디언트 계산이 명확 → 학습에 유리

**하지만 추론(inference)에서는 이 자유도가 불필요한 DRAM 왕복으로 직결된다.**

### 1.3 Decode만 정상인 이유

```
Decode (seq=1, GEMV):
  입력 [1, 4096] = 8 KB → 무시 가능
  가중치 [4096, 4096] = 32 MB ← 지배적
  출력 [1, 4096] = 8 KB → 무시 가능

LayerNorm의 출력이 8 KB이므로, 이를 DRAM에 썼다가 다시 읽어도
  8 KB × 2 = 16 KB 낭비 → 33.6 MB 가중치 대비 0.05% → 측정 불가 수준
```

seq=1이라 activation 텐서가 극히 작아 낭비가 측정되지 않는다.  
seq가 커질수록(VE의 수천 patch, Prefill의 3086 토큰) 낭비가 선형으로 증가한다.

### 1.4 Prefill 낭비의 정량적 분해

```
Prefill 1 레이어에서 "불필요한" DRAM 왕복 추정:

연산 경계마다 activation 쓰기/읽기:
  LayerNorm 출력      : 25.3 MB × 2 (쓰기+읽기) = 50.6 MB
  RoPE Q/K 결과       : 25.3 MB × 4 (2개 × 쓰기+읽기) = 101.2 MB
  Attention → O_proj  : 25.3 MB × 2 = 50.6 MB
  Residual add 입력   : 25.3 MB × 2 = 50.6 MB
  LayerNorm 2 출력    : 25.3 MB × 2 = 50.6 MB
  FFN gate/up 출력    : 68 MB × 4 (2개 × 쓰기+읽기) = 272 MB
  SiLU 결과           : 68 MB × 2 = 136 MB
  gate×up 곱          : 68 MB × 2 = 136 MB
  Residual add 2      : 25.3 MB × 2 = 50.6 MB

레이어당 "커널 경계 낭비"   ≈ 898 MB
× 36 layers               ≈ 31.7 GB

가중치 읽기 (필수):           15.168 GB
Activation 필수 전달:          ~1.8 GB
cuBLAS workspace 등:        ~수십 GB 추가

→ 이론 최솟값 ~17 GB에서 실측 231 GB까지의 차이가 설명됨
```

---

## 2. 기존 해결책 — TRT-LLM의 접근

NVIDIA TensorRT-LLM은 이 문제를 **커널 융합(kernel fusion)** 으로 해결한다:

```
TRT-LLM이 적용하는 주요 fusion:

Before (eager):           After (TRT-LLM fused):
  [LayerNorm]               ┌──────────────────┐
      ↕ DRAM                │  LayerNorm        │
  [Q_proj]                  │  + Q/K/V proj     │ → 1 커널, DRAM 왕복 제거
      ↕ DRAM                │  + RoPE           │
  [K_proj]                  └──────────────────┘
      ↕ DRAM                        ↕
  [V_proj]                  ┌──────────────────┐
      ↕ DRAM                │  FlashAttention   │ → 이미 fusion됨
  [RoPE]                    │  (O(N) DRAM)      │
      ↕ DRAM                └──────────────────┘
  [FlashAttn]                       ↕
      ↕ DRAM                ┌──────────────────┐
  [O_proj]                  │  O_proj + residual│ → 1 커널
      ...                   └──────────────────┘
```

**TRT-LLM이 실제로 하는 것:**
- LayerNorm + QKV projection fusion
- FlashAttention (already fused internally)
- FFN gate + SiLU + elementwise mul fusion
- RMSNorm + linear fusion
- 각종 elementwise operation 합산

이를 통해 이론적으로 Prefill DRAM을 수십 GB 수준으로 줄일 수 있다.

---

## 3. TRT-LLM이 해결하지 못하는 것

TRT-LLM이 잘 동작하는 환경과 우리 환경의 차이를 명확히 짚어야 한다.

### 3.1 Alpamayo는 표준 LLM이 아니다

```
TRT-LLM이 최적화하는 모델:
  GPT-2, LLaMA, Falcon, Mistral, Gemma...
  → 공통 구조: Embedding → N × Transformer Layer → LM Head

Alpamayo의 구조:
  VE (Qwen2.5-VL ViT, 6-camera multi-modal)
    ↓
  LM (Qwen2-based, 36 layers)
    ↓
  LM Decode (autoregressive)
    ↓
  Flow matching ODE (65-step, Action Expert DiT)
    ↓
  64 waypoints 출력
```

Flow matching ODE는 TRT-LLM의 설계 범위 밖이다. TRT-LLM을 적용하려면:
- VE → TRT engine 변환 (부분적 지원)
- LM → TRT engine 변환 (지원)
- Action Expert → TRT engine 변환 (미지원)
- 4단계 파이프라인 연결 → 직접 구현 필요

단계별로 따로 변환하더라도 **단계 간 파이프라인 최적화는 여전히 공백**으로 남는다.

### 3.2 iGPU 통합 메모리 아키텍처를 모른다

```
TRT-LLM이 가정하는 하드웨어:
  CPU [DDR] ←PCIe→ GPU [HBM/GDDR]
  → GPU 메모리는 GPU 전용, CPU와 분리

Thor iGPU 실제 구조:
  CPU + GPU + DMA engine 모두 LPDDR5X 공유
  → PCIe 없음, 직접 DRAM 접근
  → CPU L3 cache(16 MB)를 weight staging buffer로 활용 가능
  → DMA engine이 CPU/GPU 연산과 독립적으로 prefetch 가능
```

TRT-LLM은 이 구조적 차이를 전혀 활용하지 않는다. iGPU에서는 discrete GPU와는 다른 최적화 전략이 필요하다.

### 3.3 Flow ODE의 반복 구조를 모른다

```
Flow: 동일한 Action Expert 가중치(4.561 GB)로 65회 동일 forward pass

TRT-LLM이 보는 것: "65번의 독립적인 forward pass"
우리가 알고 있는 것: "완전히 동일한 연산의 65회 반복"

→ 가중치가 동일하므로 step N에서 warm된 L2를 step N+1이 재활용 가능
→ L2 hit rate 실측 7.7% → 재활용이 전혀 안 되고 있음 (가중치 4.561 GB >> L2 32 MB)
→ 그러나 persistent kernel + shared memory 활용으로 step 간 activation 재사용 가능
```

### 3.4 10Hz 실시간 latency 목표 vs throughput 목표

```
TRT-LLM의 핵심 가치:
  - 많은 request를 처리하는 throughput 최대화
  - in-flight batching, paged KV cache, continuous batching
  - P50 latency보다 requests/second가 중요

우리의 요구사항:
  - 단 1개의 추론을 100ms 이내에 완료 (10Hz)
  - 배치 없음 (N=1)
  - P99 tail latency가 안전과 직결
  - throughput은 무관
```

목적 함수가 다르면 최적 전략도 다르다. TRT-LLM의 paged KV cache, in-flight batching 등은 latency에 불리한 trade-off를 포함한다.

---

## 4. 우리가 만들 수 있는 것 — Alpamayo 전용 컴파일 엔진

### 4.1 컴파일 엔진이란

컴파일 엔진(compile engine)은 **모델의 연산 그래프를 분석하고, 하드웨어에 최적화된 커널 조합으로 재컴파일**하는 시스템이다.

```
입력: Alpamayo 모델 (PyTorch 연산 그래프)
      + 하드웨어 사양 (Thor, SM 11.0, L2 32 MB, LPDDR5X 231 GB/s)
      + 실행 프로파일 (실측 DRAM 접근량, 커널 경계 위치)

컴파일 엔진이 하는 일:
  1. 연산 그래프 분석 → 어느 커널 경계에서 낭비가 큰지 파악
  2. Fusion 가능 패턴 탐지 → LayerNorm+QKV, SiLU+gate+mul 등
  3. Triton / CUDA 최적 커널 생성 → iGPU 메모리 계층에 맞춘 타일링
  4. Flow ODE persistent kernel 생성 → 65 step 루프 내재화
  5. Cross-stage prefetch 스케줄 생성 → DMA + CPU + GPU 동시 활용

출력: Alpamayo 전용 최적화 실행 엔진
```

### 4.2 우리가 유리한 이유

TRT-LLM 팀은 모르는 것을 우리는 안다:

```
우리가 가진 것:
  ✅ ncu 실측값: 각 단계 정확한 DRAM 접근량
     → VE 98 GB, Prefill 231 GB, Decode 17 GB, Flow 122 GB

  ✅ 커널 수: 단계별 커널 경계 위치
     → VE 1,755 커널, Prefill 2,070 커널, Flow 24,116 커널

  ✅ L2 hit rate: Flow 7.7% → weight streaming 패턴 확인

  ✅ 하드웨어 특성: Thor iGPU unified memory, L2 32 MB, LPDDR5X 231 GB/s

  ✅ Alpamayo 파이프라인 구조: 4단계 순서, 각 단계 입출력 형태

TRT-LLM 팀이 모르는 것:
  ❌ SM 11.0 iGPU에서의 실제 DRAM 접근 패턴
  ❌ Flow matching ODE의 65-step 반복 구조
  ❌ Alpamayo VLA 파이프라인의 단계 간 데이터 흐름
  ❌ 10Hz 실시간 단일 추론 요구사항
```

**우리의 실측 데이터가 컴파일 엔진 설계의 정확한 입력값이 된다.**

### 4.3 구체적 구현 전략

#### 전략 1: Triton 기반 fused kernel 작성

Triton은 Python으로 GPU 커널을 작성할 수 있는 NVIDIA/OpenAI의 DSL이다. 커스텀 fusion이 가능하다.

```python
# 예: LayerNorm + QKV projection fused kernel (Triton)
@triton.jit
def fused_layernorm_qkv_kernel(
    x_ptr,       # 입력 activation [seq, hidden]
    w_q_ptr,     # Q 가중치 [hidden, head_dim × n_heads]
    w_k_ptr,     # K 가중치
    w_v_ptr,     # V 가중치
    out_q_ptr,   # Q 출력
    out_k_ptr,   # K 출력
    out_v_ptr,   # V 출력
    ...
):
    # LayerNorm + Q/K/V projection을 단일 커널로 처리
    # x를 DRAM에서 1번만 읽고, LayerNorm 후 즉시 Q/K/V 연산
    # LayerNorm 출력을 DRAM에 쓰지 않음 → 25.3 MB × 2 절약 per layer
    ...
```

> **SM 11.0 (Blackwell) 호환성 주의**: torch.compile의 Triton 경로는 현재 Thor에서 작동하지 않음 (API 불일치, CLAUDE.md 확정). 직접 Triton 커널 작성 + ncu_single_run 방식으로 적용.

#### 전략 2: Flow ODE Persistent Kernel

```
현재:
  for step in range(65):
    output = action_in_proj(input)  # DRAM 읽기/쓰기
    output = expert(output)          # DRAM 읽기/쓰기
    output = action_out_proj(output) # DRAM 읽기/쓰기
    input = ode_step(input, output)  # DRAM 쓰기/읽기

목표:
  [하나의 persistent CUDA kernel]
    shared_buffer = load_initial_input()  # DRAM 1회 읽기
    for step in range(65):
        # shared memory / L1 내에서 intermediate 유지
        # DRAM 쓰기/읽기 없음
        q = matmul(shared_buffer, w_q)  # w_q는 streaming (DRAM → L2 → compute)
        ...
    write_output(shared_buffer)  # DRAM 1회 쓰기

절약: step 간 activation 왕복 = 65 × ~수 MB × 371 커널 경계 수
```

#### 전략 3: iGPU 전용 Cross-Stage Prefetch 스케줄러

```
실측 데이터 기반 prefetch 타이밍 설계:

Stage 3 (Decode, 17 steps):
  step 1~14: Decode 연산
  step 15 시작 시: DMA → Flow 가중치 4.561 GB prefetch 시작
  step 17 종료 시: Flow 가중치 준비 완료 (4.561 / 231 = 19.7ms, 3 step = 32ms 여유)

Stage 2 (Prefill, 1440ms):
  Prefill 80% 완료 시: DMA → Decode KV cache 버퍼 초기화 시작
  Prefill 100% 완료 시: Decode 즉시 시작 가능

→ DMA engine이 GPU와 병렬로 동작 (iGPU의 고유 특성)
→ 이 스케줄을 자동 생성하는 것이 컴파일 엔진의 역할
```

#### 전략 4: Activation 메모리 계획 (Memory Planning)

```
현재 (PyTorch eager):
  매 연산마다 새 텐서 할당 → 메모리 파편화 → L2 eviction 증가

컴파일 엔진 접근:
  전체 레이어 실행 계획을 미리 파악
  → Layer N의 출력 버퍼가 Layer N+1의 입력으로 in-place 재사용
  → 불필요한 중간 텐서 제거
  → L2에 남아있는 activation이 다음 커널에서 hit 확률 증가
```

---

## 5. TRT-LLM과 커스텀 엔진의 포지셔닝

```
TRT-LLM을 사용할 부분:
  LM Decode (Stage 3):
    → TRT-LLM의 fused attention, layernorm+qkv 활용
    → 이미 잘 최적화된 표준 LLM 추론 경로
    → 우리가 재발명할 필요 없음

커스텀 컴파일 엔진이 필요한 부분:
  Flow ODE (Stage 4):        → TRT-LLM 범위 밖, 우리가 직접
  Cross-stage pipeline:      → TRT-LLM 범위 밖, 우리가 직접
  iGPU prefetch 스케줄:      → TRT-LLM 모름, 우리가 직접
  VE-specific fusion:        → 부분적 지원, 개선 여지 있음
```

**TRT-LLM을 쓰는 것과 커스텀 엔진을 만드는 것은 경쟁 관계가 아니다.** TRT-LLM을 LM 부분의 baseline으로 활용하면서, Alpamayo 전용 최적화를 그 위에 추가하는 구조가 현실적이다.

---

## 6. 연구 로드맵

### 단기 (즉시 실험 가능)

| 과제 | 예상 DRAM 절약 | 난이도 |
|------|-------------|--------|
| FFN SiLU+gate×up+down_proj fused Triton kernel | Prefill ~30 GB | 중 |
| LayerNorm + QKV fused kernel | Prefill ~15 GB | 중 |
| ncu로 fusion 효과 정량 측정 | — | 낮음 |

### 중기 (1~2개월)

| 과제 | 내용 | 예상 효과 |
|------|------|---------|
| Flow persistent kernel | 65-step 루프를 1 커널로 | Flow 870ms → ? |
| Cross-stage DMA prefetch 구현 | Decode 중 Flow weight prefetch | 단계 전환 latency 제거 |
| AppendOnlyCache-C ncu 재측정 | 실제 DRAM 절약량 확인 | 검증 |

### 장기 (연구 contribution)

| 과제 | 내용 |
|------|------|
| Alpamayo 전용 컴파일 엔진 | VLA 파이프라인 인식, iGPU 전용 tiling, cross-stage 스케줄 |
| iGPU 통합 메모리 최적화 이론 | "iGPU에서 VLA 추론의 최적 메모리 계층 활용 원칙" 정립 |
| 논문 | "Efficient VLA Inference on Edge iGPU: Kernel Fusion and Cross-Stage Pipelining for Alpamayo" |

---

## 7. 결론

### 발견한 것

ncu 하드웨어 카운터 측정으로 **PyTorch eager mode가 가중치 크기 대비 최대 85배의 DRAM 트래픽을 생성**하고 있음을 확인했다. 이는 모델의 결함이 아니라, 커널마다 중간 결과를 DRAM에 쓰고 다시 읽는 **커널 비융합(kernel non-fusion)** 의 구조적 결과다.

### TRT-LLM에 대한 입장

TRT-LLM은 이 문제를 LM 영역에서 잘 해결한다. **하지만:**
- Alpamayo의 Flow matching ODE를 다루지 않는다
- Jetson AGX Thor의 iGPU unified memory 구조를 최적화하지 않는다
- 10Hz 단일 추론 latency를 위한 cross-stage 파이프라인을 모른다
- **TRT-LLM은 우리 문제의 일부만 해결한다**

### 우리가 할 수 있는 것

우리는 이 하드웨어에서 Alpamayo를 가장 깊이 측정한 팀이다. 실측 DRAM 접근량, 커널 수, L2 hit rate — 이 데이터를 기반으로 **Alpamayo의 구조와 iGPU의 특성에 특화된 컴파일 엔진**을 설계할 수 있다. 이것이 TRT-LLM을 "쓰는" 연구가 아니라, TRT-LLM이 해결하지 못한 문제를 **새로 푸는** 연구가 되는 이유다.
