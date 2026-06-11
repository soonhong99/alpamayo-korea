# llm.npu 논문 철저 분석 — ASPLOS 2025

**논문 제목**: Fast On-device LLM Inference with NPUs  
**학회**: ASPLOS 2025 (Rotterdam, Netherlands)  
**저자**: Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, Xuanzhe Liu  
**소속**: Peking University / BUPT  
**코드**: https://github.com/UbiquitousLearning/mllm  
**분석일**: 2026-06-07  
**분석 목적**: 교수님 피드백 "레이어 단위 분해, 낭비 구간 식별, 병렬 처리 설계" 연구 방향 이해

---

## 1. 한 줄 요약 및 동기

### 한 줄 요약

> 모바일 LLM의 Prefill이 전체 추론 시간의 90% 이상을 차지하는 문제를, 모바일 NPU를 **operator 단위로 분해·재스케줄링**함으로써 22.4× 속도 향상을 달성한 최초의 NPU 오프로딩 시스템.

### 동기 (왜 이 문제인가)

모바일 기기에서 LLM(Qwen1.5-1.8B 등)을 실행하면:

```
UI 자동화 태스크 (600~800 토큰 입력)
  → 1 step당 8.1초 (llama.cpp, Qwen1.5-1.8B, CPU)
  → 5-step task = 40초 이상 → 실용 불가

시간 분해:
  prefill : 94.4%~98.8% (입력 처리)
  decode  :  1.2%~ 5.6% (출력 생성)
```

Prefill이 병목인데, 모바일 CPU/GPU는 병렬 연산 능력이 낮다.  
반면 모바일 NPU(Hexagon: 73 TOPS INT8)는 유휴 상태다 — **이 유휴 NPU를 활용하는 것이 논문의 핵심 기회.**

---

## 2. 핵심 기여 분석

저자가 명시한 기여:

| 기여 | 내용 | Novelty 평가 |
|------|------|-------------|
| ① Chunk-sharing graph | 가변 길이 프롬프트를 고정 크기 청크로 분할, 정적 연산자 공유 | **높음** — NPU 그래프 재빌드 문제 최초 체계적 해결 |
| ② Shadow outlier execution | activation outlier를 CPU에서 병렬 처리, per-tensor quantization 유지 | **높음** — NPU per-tensor 정밀도 손실 없이 해결 |
| ③ Out-of-order subgraph execution | 청크 간 데이터 의존성 유지하면서 서브그래프 비순차 실행 | **매우 높음** — NPU bubble 37% → 0.7% 감소, NP-hard 문제의 실용적 근사 |
| ④ 최초 1,000 tokens/sec 달성 | COTS 모바일 기기에서 10억 파라미터 모델 prefill >1,000 tok/s | **기록적 결과** |

기여 간 상호 의존성:
- ①은 ③을 가능하게 한다 (청크 분할이 있어야 서브그래프 재배치 가능)
- ②는 ①③과 독립적 (tensor 수준 최적화)
- ③은 ①②가 만들어낸 서브그래프들을 최적 순서로 스케줄링

---

## 3. 수학적 구조 분석

### 3.1 Shadow Outlier Execution 수식 (논문 Eq.1)

$$x_s \odot w = \left\{\min\left[\max(x_s, -127), 128\right]\right\} \odot w \Big|_{\text{NPU}} + \text{extract}\left(\lfloor x_s/128 \rfloor \times 128\right) \odot w \Big|_{\text{CPU}}$$

**기호 정의:**
- $x$: 원래 float activation
- $w$: INT8 weights
- $s$: quantization scale factor
- $x_s = x/s$: 정규화된 activation
- $\odot$: MatMul 연산
- $\text{extract}(\cdot)$: outlier 채널만 추출하여 compact tensor로 변환

**직관적 해석:**

비유: 큰 회사의 급여 계산. 대부분 직원(normal range)은 NPU에서 일괄 처리하고, 특수 계약직(outlier)만 별도 팀(CPU)에서 계산 → 두 결과를 합산.

```
x_s 범위 -127~128 이내: NPU에서 W8A8 per-tensor MatMul (빠름)
x_s가 128 초과:         lfloor x_s/128 rfloor * 128 = 초과분만 추출
                        → CPU에서 compact MatMul (병렬 실행)
최종 결과 = NPU 결과 + CPU 결과 (결합 법칙)
```

**가정:** outlier가 sparse할수록 CPU 부하가 적음. 논문 측정: outlier = 전체 채널의 0.1%~0.3% (5~15개/layer).

**outlier importance 정의:**
$$\text{importance}(l) = \frac{\max(\text{outlier values in layer } l)}{s_l}$$
- 비율이 클수록 activation 분포가 넓다 → quantization 오류가 크다 → 해당 레이어의 outlier 처리 중요
- 85% 비중요 레이어의 outlier는 pruning → CPU-NPU 동기화 비용 제거

### 3.2 Out-of-Order Scheduling 수식 (논문 Eq.2-5)

**Cross-chunk dependency (Eq.2):**

$$G_{i,j} \leftarrow G_{0,j-1},\, G_{1,j-1},\, \ldots,\, G_{i,j-1}$$

i번째 청크의 j번째 서브그래프는, 0~i번 청크의 j-1번 서브그래프가 모두 끝나야 시작 가능.  
(Attention은 KV Cache를 통해 이전 청크를 모두 참조)

**Intra-chunk dependency (Eq.3):**

$$G_{i,j} \leftarrow G_{i,j-1}$$

같은 청크 내에서는 순서 의존성만 있음 (LayerNorm, Linear 등).

**단일 프로세서 제약 (Eq.4):**

$$\sum_{i=0}^{N}\sum_{j=0}^{M} P_{i,j,t} = 1, \quad \forall t$$

모바일 프로세서의 약한 선점(preemption) 능력으로 인해 어느 시점에도 프로세서 하나는 서브그래프 하나만 실행.

**스케줄링 목적함수 - Contribution C (Eq.5):**

$$C(g) = \begin{cases} +\sum_{i \in S} T_i & \text{if } g \text{ is on CPU/GPU} \\ -\sum_{i \in S} T_i & \text{if } g \text{ is on NPU} \end{cases}$$

- $S$: 서브그래프 $g$ 완료 후 새롭게 실행 가능해지는 서브그래프 집합
- $T_i$: 서브그래프 $i$의 실행 시간

**해석:**
- CPU에서 $g$를 실행할 때 → $g$ 완료 후 NPU가 긴 $S$를 실행하면 좋음 → $C = +\sum T_i$ (크면 우선 선택)
- NPU에서 $g$를 실행할 때 → $g$ 완료 후 연이어 실행할 NPU 작업이 짧을수록 좋음 → $C = -\sum T_i$ (절댓값이 작으면 즉 $C$가 덜 음수이면 우선 선택)

**왜 NP-Hard인가:**
이 스케줄링 문제는 Traveling Salesman Problem (TSP)으로 환원된다. 최적해 탐색 불가 → 온라인 그리디 휴리스틱 사용 (마이크로초 수준 오버헤드).

---

## 4. 시스템 아키텍처 분석

### 4.1 전체 파이프라인

```
[오프라인 준비 단계]

LLM 가중치 (HuggingFace)
  ↓
W8A8 per-tensor quantization
  + outlier importance 프로파일링 (대규모 코퍼스 사용)
  + 하위 85% 비중요 레이어 outlier pruning 결정
  ↓
Fixed-size chunk-sharing graph 빌드 (청크 크기 256)
  ↓
각 서브그래프 실행 시간 및 의존성 프로파일링
  ↓
모델 파일 + 청크 그래프 + 메타데이터 저장


[온라인 실행 단계]

입력 프롬프트 (가변 길이)
  ↓
청크 분할 (256 토큰 단위)
  ↓
각 청크를 서브그래프 집합으로 분해
  ┌─────────────────────────┐
  │ 정적 서브그래프 (공유)  │ → NPU: Linear (W8A8)
  │  - QKV projection       │
  │  - FFN (gate, up, down) │
  │  - Output projection    │
  └─────────────────────────┘
  ┌─────────────────────────┐
  │ 동적 서브그래프 (청크별)│ → CPU/GPU: Float
  │  - Attention (QK^T, V)  │
  │  - LayerNorm            │
  │  - Softmax              │
  └─────────────────────────┘
  ┌─────────────────────────┐
  │ Shadow outlier execution│ → CPU: 0.1-0.3% channels
  └─────────────────────────┘
  ↓
Out-of-order subgraph 스케줄러 (C 값 기반 온라인 그리디)
  ↓
결과 합산 → KV Cache 업데이트 → 다음 청크
  ↓
Decode 단계 (CPU 백엔드)
```

### 4.2 Transformer Block의 연산자별 하드웨어 배치

논문의 Figure 5와 Table 4를 기반으로 한 배치 전략:

```
한 개의 Transformer Block 내부:

LayerNorm            → CPU/GPU (FP32/FP16, NPU FP 성능 ~600× 열세)
  ↓
QKV Projection       → NPU (INT8 MatMul, 4.5-5.8× CPU 대비 빠름)
  ↓
[Shadow Outlier]     → CPU (0.1-0.3% outlier channels, NPU와 병렬)
  ↓
Attention (Q×K^T)    → CPU/GPU (FP16, causal mask 포함)
Softmax              → CPU/GPU (FP16)
Attention (score×V)  → CPU/GPU (FP16)
  ↓
Output Projection    → NPU (INT8 MatMul)
  ↓
Add (Residual)       → CPU/GPU (FP16)
  ↓
LayerNorm            → CPU/GPU (FP16)
  ↓
FFN (gate, up, down) → NPU (INT8 MatMul, 가장 큰 연산량)
  ↓
Add (Residual)       → CPU/GPU (FP16)
```

**설계 선택의 근거:**
- INT8 linear → NPU: 4.5-5.8× CPU INT8 대비 빠르고, GPU FP16 대비 1.8-3.5× 빠름
- FP16 Attention → CPU/GPU: NPU FP16 MatMul은 CPU INT8보다 193~759× 느림
- FP 연산 비중이 작아 CPU/GPU가 NPU보다 빠르게 처리 → out-of-order로 hidden

### 4.3 Chunk-sharing Graph의 연산자 분류

```
Prompt 길이 1024, 청크 크기 32 기준:

[Static Operators — 청크 간 공유]    [Dynamic Operators — 청크별 독립]
  QKV Linear weights                   Attention (K/V 크기 변화)
  FFN weights                          KV Cache
  Output Projection weights
  LayerNorm weights

Qwen1.5-1.8B 실측:
  전체 144개 서브그래프 중 120개 공유 가능 (83.3%)
  메모리 절약: 7.2 GB (1024-token 프롬프트, 256-chunk 기준)
```

---

## 5. 실험 결과 분석

### 5.1 주요 성능 수치 (논문 Figure 14 기반)

**Prefill 속도 (prompt=1024, Redmi K70 Pro):**

| Baseline | 비교 대상 | llm.npu 대비 |
|---------|-----------|-------------|
| llama.cpp-CPU | CPU 기반 | 18.17~38.4× 느림 |
| MNN-CPU | CPU 기반 | 7.3× 느림 |
| MLC-LLM-GPU | GPU 기반 | 32.5~43.6× 느림 |
| TFLite-GPU | GPU 기반 | 1.27~2.34× 느림 |
| PowerInfer-V2-NPU | NPU 기반 | 3.28~5.32× 느림 |

**에너지 소비 절감 (prompt=1024, Redmi K60 Pro):**
- vs llama.cpp-CPU: 35.63~59.52×
- vs MLC-LLM-GPU: 35.21~59.25×
- vs TFLite-GPU: 1.85~4.32×

**End-to-end 실제 애플리케이션 (Table 5, Qwen1.5-1.8B, LongBench 2wiki):**
- llm.npu: 1.7초 (prefill 1.49s, decode 0.24s)
- vs MLC-LLM: 45.6초 → **26.8× 단축**
- vs llama.cpp: 26.7초 → **15.7× 단축**

### 5.2 Ablation Study (Figure 19, prompt=512)

Qwen1.5-1.8B 기준 누적 기여도:

```
①  CPU baseline          :  65 tokens/sec (기준)
②  Naive NPU 단독        :  25 tokens/sec → 2.6× 느려짐!
   (그래프 빌드 오버헤드가 연산 이득을 상쇄)
③  + Chunk-sharing graph :  37 tokens/sec → CPU의 57%
   (그래프 빌드 시간 해소)
④  + Shadow outlier      : 395 tokens/sec → CPU의 6.1×
   (per-tensor NPU 완전 활용 가능해짐)
⑤  + Out-of-order exec   : 569 tokens/sec → CPU의 8.8×
   (NPU 버블 37% → 0.7%)
```

**가장 큰 기여:** Shadow outlier execution (3.91~8.68× 향상)  
**이유:** NPU가 per-tensor INT8 MatMul을 완전히 수행 가능해지면서 실질적인 하드웨어 활용율이 폭발적으로 증가.

### 5.3 정확도 (Table 6)

| 방법 | LAMBADA 평균 손실 | HellaSwag 평균 손실 |
|------|:-:|:-:|
| SmoothQuant | -14.9% | -5.7% |
| K-Quant | -31.3% | -0.8% |
| LLM.Int8() | 0% | -0.1% |
| **llm.npu** | **-1.2%** | **-0.0%** |

llm.npu는 FP16 대비 평균 **1% 미만 정확도 손실**.

### 5.4 청크 크기 선택 (Figure 8)

```
청크 크기 vs 처리 속도 관계:
  너무 작음: NPU SIMD 활용률 낮음 (언더로드)
  너무 큼:   NPU 메모리 한계 초과 + padding 낭비
  최적값: 256 (Xiaomi 14 기준)

근거: QKV Linear와 FFN의 처리 속도가 chunk=256에서 동시 최적화
```

---

## 6. 한계점 분석

### 6a. 저자가 명시한 한계

1. **Decode 단계 최적화 부재**: 현재 CPU 기반 decode → Persona-Chat처럼 출력이 많은 태스크에서 효과 감소 (1.1× 수준). 논문은 "GPU-NPU coordination으로 80~90ms decode 개선 가능"이라 언급하나 미구현.

2. **Hexagon NPU 전용 구현**: Qualcomm Hexagon에만 구현. MediaTek APU, Google Edge TPU, Huawei Ascend에 직접 적용 불가. 이식성 제한.

3. **리소스 경합 미고려**: `llm.npu does not consider resource detection and contention in the current implementation.` 다른 앱이 CPU/GPU를 쓰는 상황에서의 성능 변동 미검증.

4. **INT8 한계**: 정밀도 민감한 모델에서 1% 손실도 문제될 수 있음.

### 6b. 저자가 인정하지 않은 한계 *(분석자 독자적 관점)*

1. **Prefill만 최적화**: KV Cache를 decode에서 재활용하는 최적화(AppendOnlyCache 등)는 논문 범위 밖. 실제 decode latency는 여전히 병목 가능.

2. **메모리 오버헤드 실측 부재**: Shadow execution으로 weights를 CPU 메모리에 추가 복사 → 최대 2× 메모리 증가. Table만 제시, 장시간 운용 시 메모리 압박 미검증.

3. **Outlier 분포 가정**: "입력에 관계없이 outlier 채널 위치는 안정적"이라는 가정이 있다. 이 가정이 out-of-distribution input에서도 성립하는지 실험 없음.

4. **청크 내 causal mask 비효율**: 청크 단위 처리는 청크 내 tokens에 대해 full attention을 수행하지만, 이미 앞선 청크의 KV Cache가 있는 상황에서 attention 크기가 청크 순서에 따라 달라진다. 초기 청크는 small attention, 후기 청크는 large KV → 마지막 청크 attention이 병목 가능성.

5. **온라인 스케줄러의 최적성 보장 없음**: 그리디 휴리스틱은 국소 최적만 보장. 복잡한 의존성 구조에서 성능이 열화될 수 있으나 분석 없음.

---

## 7. 창의적 재검토 — 논문이 당연하게 넘어간 가정

### 7.1 "NPU는 정적 shape만 지원한다"는 가정 재검토

논문은 NPU의 동적 shape 미지원을 근본 제약으로 받아들이고 chunk-sharing으로 우회했다.  
그러나:
- 최신 NPU(Hexagon V79 등)는 dynamic shape 지원을 점진적으로 추가 중
- 논문 자체 Discussion에서 "Dynamic shape-aware hardware"를 미래 방향으로 제시
- 장기적으로는 llm.npu의 청크 분할 필요성이 사라질 수 있음

### 7.2 "Attention은 반드시 FP로 처리해야 한다"는 가정 재검토

모든 quantization 방법이 float attention을 유지했다. 그러나:
- Flash Attention의 일부 구현은 INT8 softmax를 허용
- Outlier 접근을 attention score에도 적용하면 NPU Attention 가능성 있음 *(추측)*
- 논문에서는 이 가능성을 탐색하지 않음

### 7.3 "Out-of-order execution이 correctness를 보장한다"는 가정 재검토

논문은 Cross-chunk와 Intra-chunk dependency 두 가지만 고려한다.  
그러나 Decode 단계에서 KV Cache를 공유하는 구조에서 청크 병렬 실행이 cache consistency를 보장하는지 명시적 증명이 없다.  
*(추론: 청크는 순서대로 KV에 append하므로 race condition은 없을 것으로 보이나, 논문에 공식 증명 없음)*

### 7.4 후속 연구 방향 (분석자 제안)

| 방향 | 아이디어 | Alpamayo 적용 가능성 |
|------|---------|---------------------|
| Decode 최적화 통합 | AppendOnlyCache 계열과 결합 | ✅ 직접 적용 가능 |
| GPU-NPU 협력 | Attention을 GPU로, Linear를 NPU로 → CPU 부하 이관 | ✅ Thor 구조에 직접 대응 |
| Prefill-Decode 오버랩 | 청크별 prefill 중 다음 인퍼런스 decode 병렬 실행 | ✅ Rolling Trajectory Pipeline과 일치 |
| VE에 chunk-sharing 적용 | Alpamayo의 VE도 16장 이미지를 청크화 가능 | ⚠️ VE는 양방향이므로 cross-frame attention 처리 필요 |
| 연산자별 ncu 측정 | 각 GEMM 종류별 실제 DRAM 접근량 측정 | ✅ 교수님 피드백 정확히 부합 |

---

## 8. Alpamayo 연구에 대한 직접적 시사점

교수님이 이 논문을 언급하신 이유와 우리가 배워야 할 것:

### 8.1 논문이 보여주는 "원자 단위 분해"의 방법

```
llm.npu의 분해 계층:

레벨 3: Block level
  → Transformer block을 NPU 적합/비적합으로 분류
  → 실측: FFN (INT8 weight)가 NPU 가장 적합
           Attention (float)이 CPU/GPU 필요

레벨 2: Tensor level
  → Activation 채널 단위 분석
  → 실측: 0.1-0.3% 채널만 outlier, 3% 채널이 80% 이상 outlier 담당

레벨 1: Prompt level
  → 토큰 단위 청크화
  → 실측: 청크 256이 NPU SIMD 활용률 최적
```

**Alpamayo에 적용할 동일한 분해:**

```
현재 우리가 측정한 것: 단계 전체 (728ms VE, 895ms Prefill, 1345ms Decode, 637ms Flow)

교수님이 원하시는 것 (llm.npu 방식):
  각 단계 내부의 연산자별 시간
  → W_Q: ?ms, W_K: ?ms, W_V: ?ms
  → Attention QK^T: ?ms, Softmax: ?ms, score×V: ?ms
  → FFN gate: ?ms, FFN up: ?ms, FFN down: ?ms
  → LayerNorm: ?ms (×2 per layer)
  → Residual add: ?ms
  각 연산자의 산술 강도 (FLOPs/byte)
  → 메모리 병목인가? compute 병목인가?
```

### 8.2 Out-of-order Execution → CUDA Stream 파이프라이닝

llm.npu의 핵심 아이디어를 Thor에 번역하면:

```
llm.npu (모바일):
  NPU Queue + CPU Queue
  → 의존성 지킨 채 out-of-order 스케줄링
  → NPU bubble 37% → 0.7%

Thor (iGPU):
  GPU CUDA Stream + CPU 비동기
  → compute stream (Layer k 연산)
  →   + memory stream (Layer k+1 weights prefetch)
  → cudaMemPrefetchAsync로 DRAM 로드 시간 숨기기
  → 이론: DRAM 로드 시간 제거 → 86ms/step → ~30ms/step (추정)
```

**llm.npu의 bubble rate 37% → 0.7% 감소는, Alpamayo decode 107ms → 79ms (-26%)와 같은 맥락이다.**  
두 논문 모두 "연산과 데이터 로드의 중첩"으로 성능을 끌어올렸다.

### 8.3 Shadow Outlier Execution → KV Temporal Reuse 영감

```
llm.npu Shadow:
  - 전체 activation 중 0.1-0.3%만 이례적 (outlier)
  - 이 부분만 분리해 CPU에서 처리
  - NPU는 일반 부분만 처리 → 효율 최대

Alpamayo KV Temporal Reuse 유사 구조:
  - 전체 vision KV 중 75%는 이전 프레임과 동일 (3/4 프레임 재사용)
  - 이 75%는 재계산 없이 그대로 사용
  - 나머지 25% (새 프레임)만 새로 계산
  - 절감 예상: 895ms Prefill → 44ms (-95%)
```

---

## 9. 요약 카드

| 항목 | 내용 |
|------|------|
| **핵심 문제** | 모바일 LLM prefill이 전체 추론의 90%를 차지. 모바일 NPU는 73 TOPS 보유하나 LLM 구조와 불일치로 미활용 |
| **핵심 방법** | ① 청크 분할 + 연산자 공유 (그래프 빌드 오버헤드 제거) ② 0.1-0.3% activation outlier만 CPU에서 병렬 처리 ③ NPU 버블 최소화 out-of-order 스케줄링 |
| **핵심 결과** | Prefill 22.4× 빠름, 30.7× 에너지 절감. COTS 모바일 최초 >1,000 tok/s. 정확도 손실 <1% |
| **핵심 한계** | Decode 미최적화, Hexagon 전용, 리소스 경합 무시, INT8 한계 |
| **후속 연구 방향** | GPU-NPU 협력 decode, 연산자별 ncu 측정, CUDA Stream prefetch와 결합, Dynamic shape NPU 활용 |

---

## 10. 논문과 현재 연구의 연결 지점

### llm.npu의 3단계 분해 → Alpamayo 적용 로드맵

| llm.npu | 대응 Alpamayo 방향 | 상태 |
|---------|-------------------|------|
| Chunk-sharing (Prompt level) | Async VE Pipeline (16장 → 청크별 비동기 처리) | 설계 완료 |
| Shadow outlier (Tensor level) | ncu per-kernel DRAM 실측 (anomalous layer 식별) | **미실시 — 최우선** |
| Out-of-order execution (Block level) | cudaMemPrefetchAsync + Dual CUDA Stream (레이어 k 연산 중 k+1 prefetch) | **미실시 — 핵심 방향** |

### 교수님 피드백과의 연결

> "모델을 원자 단위로 쪼개가면서 살펴보고, 낭비되는 부분이 어디있는지 세세하게 살펴보라"

llm.npu는 정확히 이것을 한다:
- 레이어 내 각 연산자의 hardware affinity를 실측
- NPU에 맞지 않는 연산자(FP Attention, LayerNorm)를 정확히 식별해 CPU로 이관
- 남은 bubble을 out-of-order execution으로 제거

Alpamayo에서 우리가 해야 할 동일한 작업:
1. `ncu`로 Decode 1 step의 각 GEMM kernel별 `dram__bytes_read.sum` 측정
2. 어떤 kernel이 compute-bound인지, memory-bound인지 roofline 상에 배치
3. compute-bound kernel → cudaMemPrefetchAsync로 다음 레이어 weights를 미리 로드
4. memory-bound kernel → FlashAttention 최적화 경로 유지

---

*관련 파일:*
- `docs/2606_1주차/260607_01_교수님_피드백_5주차_아키텍처_레이어구조_심층분석.md` — 교수님 질문 심층 분석
- `docs/2605_5주차/260605_03_Alpamayo_대역폭_측정_파이프라이닝_분析.md` — Alpamayo 대역폭 현황
- `CLAUDE.md` — "cudaMemPrefetchAsync + CUDA Stream 이중화" 연구 방향 (교수님 확정)
