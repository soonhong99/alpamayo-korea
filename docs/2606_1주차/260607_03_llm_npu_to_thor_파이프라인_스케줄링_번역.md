# llm.npu → Thor 파이프라이닝 및 스케줄링 번역 분석

**작성일**: 2026-06-07  
**배경**: ASPLOS 2025 "Fast On-device LLM Inference with NPUs" 기법을  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Alpamayo 1.5 on Jetson AGX Thor에 시스템 수준에서 적용하는 방향 분석  
**전제**: 양자화 없음 — 시스템 파이프라이닝 및 스케줄링으로만 최적화

---

## 1. 핵심 전제: llm.npu의 원리는 NPU에 종속되지 않는다

논문의 표면적 기여는 "NPU를 LLM에 활용하는 방법"이지만,  
논문이 실제로 해결하는 문제는 다음이다:

> **"여러 개의 독립적 하드웨어 자원이 있을 때, 의존성을 지키면서 각 자원의 유휴 시간(bubble)을 제거하는 스케줄링"**

이 원리는 하드웨어에 무관하다. Thor에 그대로 적용 가능하다.

---

## 2. 하드웨어 구조 동형성 (Isomorphism)

### llm.npu의 하드웨어 (Qualcomm Snapdragon)

```
┌─────────────────────────────────────────────────────────────┐
│ Qualcomm Snapdragon                                         │
│                                                             │
│  NPU (Hexagon DSP)    CPU (ARM big.LITTLE)                 │
│  73 TOPS INT8         moderate FP16                        │
│  INT8 MatMul 전담     FP Attention, outlier, residual      │
│  ↕ 독립 DMA          ↕ 독립 메모리 버스                    │
│                                                             │
│  NPU ← → CPU: 동시 실행 가능, 의존점에서만 동기화          │
└─────────────────────────────────────────────────────────────┘
NPU bubble (37%): CPU op이 끝날 때까지 NPU가 기다리는 시간
```

### Thor의 하드웨어 구조

```
┌─────────────────────────────────────────────────────────────┐
│ NVIDIA Jetson AGX Thor (SM 11.0)                            │
│                                                             │
│  GPU SM (20개)                 CPU (ARM Neoverse V3AE × 14) │
│  1,600 TFLOPS BF16             경량 연산                    │
│  GEMM, Attention 전담          전처리, 스케줄               │
│  ↕ 단일 SM Pool                ↕ Python thread              │
│  (multi-stream → 직렬화됨)     (GPU inference 중 독립 실행) │
│                                                             │
│  CUDA Unified Memory Manager (Page Migration Engine)        │
│  cudaMemPrefetchAsync → SM과 44% overlap 가능               │
│  [128 GB LPDDR5X 231 GB/s — CPU+GPU 공유 단일 DRAM]         │
└─────────────────────────────────────────────────────────────┘
GPU bubble: 커널 간 Python dispatch 대기, 직렬 fetch-compute
```

⚠️ **Thor에는 DLA가 없다** (2026-06-04 실험 + 하드웨어 문서 확인).  
이전 버전에서 "DLA × 2 (NVDLA v3.0)" 언급은 오기(誤記)임.

⚠️ **Thor iGPU에서 CUDA multi-stream (SM ∥ SM)은 직렬 실행된다** (260604_04 실험: 0.97~1.00×).  
Thor는 단일 SM Pool 구조 → GPU 스케줄러가 stream을 순서대로 처리.  
dGPU의 GPC(Graphics Processing Cluster) 분리 구조가 없어 동시 실행 불가.

### 대응 관계 (260604 실험 결과 반영)

| llm.npu 자원 | Thor 대응 자원 | 실제 동시 실행 가능 여부 | 역할 |
|---|---|---|---|
| NPU (INT8 MatMul) | GPU SM (BF16 GEMM) | — (주 연산) | GEMM, Attention |
| CPU (FP Attention, outlier) | CPU (14-core) | **✅ 가능** (CPU time < GPU time 조건) | 전처리 병렬 실행 |
| NPU-CPU 독립 DMA | cudaMemPrefetchAsync | **⚠️ 44% overlap** (크기 균형 조건) | layer 가중치 prefetch |
| NPU-CPU 동기화 포인트 | CUDA Event | — | 의존성 관리 |
| NPU bubble (37%) | inter-kernel gap + serial fetch | — | 제거 목표 |

---

## 3. 3가지 기법의 Thor 번역

### 기법 ① Chunk-sharing Graph → CUDA Static Dependency DAG

**원래 목적**: NPU의 정적 shape 제약을 우회하기 위해 가변 프롬프트를 고정 청크로 분할.

**Thor에서의 의미**: CUDA는 동적 shape를 지원하므로 NPU의 shape 제약은 없다.  
그러나 논문의 더 깊은 원리 — **"연산 그래프를 사전에 명시하여 런타임 오버헤드를 제거"** — 는 적용된다.

**현재 문제 (Python dispatch overhead):**
```
decode 1 step: 약 252개 CUDA 커널 launch
  = 36 layers × (q, k, v, norm, attn, o, gate, up, down, residual, ...) per layer

Python이 각 커널을 순서대로 launch:
  [Python] → [CUDA launch q_proj] → [Python] → [CUDA launch k_proj] → ...
  
  각 Python dispatch: ~10~50 μs
  총 overhead: 252 × 30 μs ≈ 7.5 ms / step (= 79 ms의 9.5%)
```

**CUDA Graph 적용 (= llm.npu의 Chunk-sharing 대응):**
```python
# 사전 컴파일 단계 (한 번만)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = lm_model(inputs_embeds, position_ids, ...)

# 추론 시 (매 step)
graph.replay()  # Python dispatch overhead 없이 직접 GPU 실행
# → 7.5 ms overhead 제거
```

⚠️ **주의**: CLAUDE.md 확인됨 — Qwen3VL의 `_deepstack_process`에서 dynamic boolean indexing이  
CUDA Graph 캡처와 근본 비호환. 커스텀 decode 루프 재작성이 선행되어야 함.

---

### 기법 ② Shadow Outlier Execution → CPU 병렬 전처리

**원래 목적**: activation outlier(0.1-0.3%)를 CPU에서 NPU와 동시 처리 → NPU는 일반 채널에 집중.

**Thor에서의 의미**: BF16 유지로 quantization outlier 문제 없음.  
그러나 **"GPU가 무거운 연산을 하는 동안 CPU가 다른 독립적 작업을 병렬 실행"** 원리는 직접 적용.

**현재 문제:**
```
현재 Inference k+1 시작 전 CPU 준비 과정:
  GPU: Inference k 완료 (4,366 ms)
                              ↓ GPU 완료 신호
  CPU: 다음 프레임 수신 + 이미지 전처리 + CUDA 메모리 staged
       → 이 시간 동안 GPU가 대기 (CPU가 준비할 때까지)
```

**Shadow 방식 적용:**
```
GPU가 Inference k의 Decode step 10~17을 실행하는 동안 (마지막 ~560 ms):
  CPU Thread:
    ① 카메라 프레임 k+1 수신
    ② 이미지 resize, normalize (Vision preprocessing)
    ③ 비동기 CUDA memcpy → GPU 메모리로 staged
    ④ 텍스트 토큰 준비, embedding lookup

GPU Inference k 완료 → Inference k+1 VE 입력 이미 GPU에 있음 → 즉시 VE 시작

절약: CPU 준비 시간 (현재 측정 필요) → 약 20-50 ms 추정
```

---

### 기법 ③ Out-of-Order Subgraph Execution → CUDA 멀티스트림 스케줄링

**원래 목적**: NPU bubble 37% → 0.7%. CPU op과 NPU op이 서로를 기다리지 않도록 의존성을 분석해 재배치.

**Thor에서**: 이것이 가장 직접적이고 중요한 대응이다.

#### 3.1 레이어 내부 (Intra-layer) — Thor에서 SM 병렬화 불가

**의존성 분석 (논리적으로는 병렬 실행 가능):**
```
한 블록에서:
  q_proj ──────────────────┐
  k_proj → k_norm → mrope ─┼─→ Attention ─→ o_proj ─→ residual
  v_proj ──────────────────┘
  
  gate_proj ─→ silu ─→ mul ─→ down_proj ─→ residual
  up_proj ──────────┘
  
  Q/K/V의 입력은 동일한 hidden → 상호 의존성 없음 (이론적으로 동시 실행 가능)
  gate/up의 입력은 동일한 hidden → 상호 의존성 없음 (이론적으로 동시 실행 가능)
```

**❌ Thor에서 CUDA Multi-Stream SM 병렬화 불가 (260604_04 실험 결과)**
```
실험 (260604_cuda_stream_concurrency_test.py):
  stream0 (compute-bound): torch.mm 반복 → Tensor Core 집중
  stream1 (memory-bound) : tensor.clone() → DRAM 대역폭 집중
  결과: 0.97~1.00× (직렬과 동일)

원인: Thor iGPU는 단일 SM Pool — 독립 GPC 없음
  dGPU: GPC 0 → stream A 전용 실행 ┐ → 동시 실행 ✅
         GPC 1 → stream B 전용 실행 ┘
  Thor:  단일 SM Pool → stream A 완료 후 stream B → 직렬 ❌
```

**→ Intra-layer CUDA multi-stream 최적화는 Thor에서 효과 없음.**  
Q/K/V 분리, gate/up 분리를 별도 stream으로 실행해도 실제로는 순차 처리된다.

⚠️ **dGPU나 Xavier를 대상으로 한 논문(RT-Swap, Demand Layering) 기법이**  
**Thor iGPU에서 그대로 작동하지 않는 핵심 이유가 바로 이것이다.**

#### 3.2 레이어 간 (Inter-layer, llm.npu의 out-of-order 핵심 대응)

**논문의 핵심 insight**: NPU 연산이 진행되는 동안 CPU가 다음 연산 데이터를 준비.  
Thor 대응: SM이 Block k를 계산하는 동안 DMA가 Block k+1 가중치를 prefetch.

```
현재 (직렬):
  ├── Block k 가중치 DRAM fetch ──┐
                                  └── Block k 연산 ──┐
                                                      └── Block k+1 가중치 fetch ──→ ...
  DMA와 SM이 절대 겹치지 않음

cudaMemPrefetchAsync 적용:
  ├── Block k 연산 (SM) ──────────────────────────────────────────────→
  └── Block k+1 가중치 prefetch (DMA) ────────────────────────────────→
      ↑ SM과 DMA는 독립 하드웨어 → 진짜 동시 실행

  Block k 완료 → Block k+1 가중치 이미 L2에 있음 → DMA 대기 없이 즉시 시작
```

**Contribution C 스케줄러의 Thor 번역:**

논문 Eq.5: `C(g) = +ΣT_i` (CPU ops) / `-ΣT_i` (NPU ops)

Thor 버전:
```python
def contribution_c(op, pending_ops):
    """
    op을 지금 실행하면 unlock되는 downstream 작업들의 합산 시간
    """
    if op.stream == "DMA_prefetch":
        # prefetch 완료 → SM이 unlock됨 → SM 작업 시간의 합
        return +sum(t for t in downstream_SM_ops(op))
    elif op.stream == "SM_compute":
        # SM 완료 → DMA가 다음 prefetch 시작 → DMA 작업 시간의 합
        return -sum(t for t in downstream_DMA_ops(op))

# 스케줄링: C가 가장 큰 op을 먼저 실행
# (Alpamayo는 고정 아키텍처 → 오프라인 정적 계산 가능)
```

**실용적 구현**: Alpamayo의 아키텍처가 고정이므로 OOE 스케줄러를 매 step 실행할 필요 없이,  
최적 schedule을 오프라인에서 한 번 계산하고 고정 CUDA event graph로 구워넣는다.

---

## 4. 레벨별 파이프라이닝 전체 구조

```
레벨 3: Inference 간 스트리밍 (교수님 핵심 제안)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  → llm.npu의 연속 프롬프트 처리와 동일한 원리
  → 100ms마다 새 Inference launch → 파이프라인 10 Hz
  → 단일 inference 4,366ms이지만 throughput = 10Hz 달성

  Inference A: [VE─728ms][Prefill─1423ms][Decode─1345ms][Flow─870ms]
  Inference B:            [VE][Prefill][Decode][Flow]
  Inference C:                     [VE][Prefill]...
  
  43개 concurrent inferences → pipeline 가득 참

레벨 2: 단계 간 CPU-GPU 협력 (Shadow Outlier 대응)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GPU: [Decode step 10~17 = ~560ms]
  CPU:    [다음 프레임 전처리 + CUDA staged]
  
  GPU Decode 완료 → 다음 VE 입력 즉시 준비 → 지연 제거

레벨 1: 레이어 간 DMA-SM 중첩 (OOE Block level 대응)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SM Stream: [Block k 연산 ─────────]
  DMA Stream:    [Block k+1 prefetch ─]
  → DMA idle 완전 제거

레벨 0: 레이어 내 연산자 병렬화 (OOE Intra-chunk 대응)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ❌ Thor iGPU에서 SM ∥ SM 불가 (단일 SM Pool, 직렬화됨)
  Stream 0: [q]─────→[attn]→[o]→[gate]─────→[down]
  Stream 1:   [k][v]              [up]         ← 실제로는 직렬 실행
  → Thor에서 효과 없음 (260604_04 실험 0.97× 확인)
```

---

## 5. llm.npu와의 직접 대응 및 가능성 평가

| llm.npu 기법 | Thor 대응 구현 | Thor 실현 여부 | 이득 추정 | 난이도 | 선행 조건 |
|---|---|---|---|---|---|
| Chunk-sharing graph | CUDA Graph | ⚠️ 조건부 | -7.5 ms/step | 높음 | dynamic indexing 제거 필요 |
| Shadow outlier → CPU | CPU 병렬 전처리 | **✅ 가능** | -20~50 ms/step | 낮음 | 즉시 가능 |
| OOE (inter-layer) | cudaMemPrefetchAsync | **⚠️ 44% overlap** | -10~20 ms/step | 중간 | ncu 측정 선행 |
| OOE (intra-layer) | CUDA multi-stream | **❌ 불가** | ~~-5~18 ms~~→ 0 | 해당없음 | 단일 SM Pool (260604 확인) |
| ~~DLA 활용~~ | ~~TensorRT FP16 FFN~~ | **❌ DLA 없음** | — | — | Thor에 DLA 미탑재 |

---

## 6. ~~DLA 활용 방향~~ → **확정: Thor에 DLA 없음**

**2026-06-07 확인**: Thor(Jetson AGX Thor, SM 11.0)에는 **DLA가 탑재되어 있지 않다.**

근거:
- `docs/2605_4주차/260524_02_Thor_아키텍처_심층분석.md`: Thor 스펙에 DLA 언급 없음
- `docs/2605_5주차/260604_04_Thor_iGPU_병렬실행_가능성_정리.md`: "Thor에는 DLA가 없거나 활용 방법이 다르다" (미검증 기술 → 이후 없음으로 확정)
- Thor는 Blackwell SM 11.0 기반 iGPU — DLA는 Orin, Xavier 계열에 탑재

DLA가 있는 이전 모델과의 비교:
```
Jetson Xavier NX: 2× NVDLA (Deep Learning Accelerator) → CNN/DNN 오프로드 가능
Jetson Orin:      2× NVDLA v2.0 → INT8 가속
Jetson AGX Thor:  DLA 없음 → GPU SM만으로 모든 연산
```

→ **llm.npu의 "NPU와 GPU를 같이 쓰는" 구조를 Thor에서 그대로 재현하는 것은 불가능.**  
가장 가까운 대응은 cudaMemPrefetchAsync(page migration engine) + CPU thread 병렬 실행이다.

---

## 7. 구현 우선순위

### 즉시 시작 가능

```python
# Step 1: CUDA Stream 2개 생성
stream_a = torch.cuda.Stream()  # q/attn/o/gate/down
stream_b = torch.cuda.Stream()  # k/v/up

# Step 2: 이벤트 기반 동기화
evt_qkv_done = torch.cuda.Event()

with torch.cuda.stream(stream_b):
    k = self.k_proj(hidden)
    v = self.v_proj(hidden)
    evt_qkv_done.record()

with torch.cuda.stream(stream_a):
    q = self.q_proj(hidden)
    stream_a.wait_event(evt_qkv_done)  # k, v 완료 대기
    attn_out = sdpa(q, k, v, ...)

# Step 3: inter-layer prefetch (다음 레이어 가중치 prefetch)
# decode loop에서:
for i, layer in enumerate(layers):
    if i + 1 < len(layers):
        # 다음 레이어 가중치를 비동기 prefetch
        for param in layers[i+1].parameters():
            torch.cuda.default_stream().wait_stream(stream_a)
            # cudaMemPrefetchAsync equivalent
```

### 측정 선행 필요

1. `ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed` per kernel
   → q_proj, k_proj, v_proj, gate_proj의 DRAM 활용률 측정
   → 100% 미만이면 DRAM 대역폭에 여유가 있는 것

2. Inter-kernel gap 측정 (`260607_idle_device_profiler.py` 활용)
   → Python dispatch overhead가 실제로 얼마인지 (nsys로 timeline 확인)

⚠️ DLA 측정 항목 삭제 — Thor에 DLA 없음 (Section 6 참조)

---

## 8. 예상 누적 효과 (260604 실험 결과 반영)

```
현재 decode: 79 ms/step (17 steps = 1,345 ms)

Thor에서 실제로 사용 가능한 최적화:
  ① CPU 병렬 전처리: Inference 간 지연 -20~50 ms     ← ✅ 즉시 구현 가능
  ② cudaMemPrefetchAsync (inter-layer):               ← ⚠️ 조건부 가능
       단일 레이어 prefetch ≈ 44% overlap, 1.25×
       그러나 decode는 이미 DRAM-bound → "prefetch해서 올 곳이 더 빠른 메모리"가 없음
       → pixel_values page migration (VE 시작 전 staged) 적용이 현실적
  ③ ~~CUDA multi-stream Q/K/V~~: 0× (단일 SM Pool) ← ❌ 불가
  ④ Python dispatch 제거 (CUDA Graph):               ← ⚠️ 조건부 가능
       -7.5 ms/step 가능 시 → dynamic indexing 재작성 선행 필요

보수적 추정 (①+② 픽셀 prefetch만):
  Inference 간 지연 -20~50 ms → 4,366 ms → ~4,316~4,346 ms
  큰 효과는 ① CPU 전처리 파이프라인 (간접 latency 제거)
  
  CUDA Graph 성공 시: 79 → ~71 ms/step (17 steps = 1,207 ms)
  → 전체 4,366 → ~4,228 ms
  
1단계 목표 (CLAUDE.md ~3,500 ms): KV Temporal Reuse 병행 필요
```

---

## 9. 논문이 우리에게 주는 진짜 교훈

llm.npu가 NPU bubble 37% → 0.7%로 만든 방법은 "더 좋은 하드웨어를 쓴 것"이 아니다.  
**"있는 하드웨어의 유휴 시간을 측정하고, 의존성이 없는 작업으로 그 시간을 채운 것"이다.**

우리가 해야 할 일:
```
1. 측정: ncu로 각 커널의 DRAM utilization, SM utilization, inter-kernel gap
2. 분류: 어느 커널이 실제로 독립적으로 실행 가능한가? (의존성 DAG 분석)
3. 스케줄: Contribution C 방식으로 독립 커널을 병렬 배치
4. 검증: 병렬화 후 실측 시간이 예측과 일치하는가?
```

이것이 교수님이 "원자 단위로 쪼개서 낭비를 찾으라"고 하신 말씀의 정확한 의미다.

---

*관련 파일:*
- `docs/2606_1주차/260607_02_llm_npu_ASPLOS25_논문분석.md` — 원 논문 분석
- `docs/2606_1주차/260607_01_교수님_피드백_5주차_아키텍처_레이어구조_심층분석.md` — 교수님 질문
- `scripts/profiling/260607_idle_device_profiler.py` — 유휴 device 측정 스크립트
- `CLAUDE.md` — "cudaMemPrefetchAsync + CUDA Stream 이중화" 연구 방향
