# NSight 분석 — Decode Per-Token Overhead 해부

**작성 일시**: 2026-05-19  
**측정 도구**: NVIDIA NSight Systems (nsys), CUDA API trace  
**대상**: Alpamayo 1.5 on Jetson AGX Thor (BF16)  
**핵심 발견**: per-token 110ms = GPU 커널 57ms + 오버헤드 53ms

---

## 0. 핵심 수치 요약

```
┌─────────────────────────────────────────────────────────┐
│  1 token 사이클 = 110ms (기존 측정)                      │
│                                                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │  GPU 커널 실행       │  │  오버헤드 (CPU-GPU 교대)   │  │
│  │       57ms          │  │         53ms              │  │
│  │  (Transformer fwd)  │  │  (sync + prepare + copy)  │  │
│  └─────────────────────┘  └──────────────────────────┘  │
│                                                         │
│  → 사이클의 48%가 낭비되고 있음                           │
└─────────────────────────────────────────────────────────┘
```

---

## 1. NSight CUDA API Trace — 한 token 사이클의 전체 구조

### 실행 순서 (NSight에서 관찰된 순서)

```
[1] GPU Transformer forward (57ms)
    ─────────────────────────────────────────────────────
    QKV projection GEMV
    KV cache read (attention)
    FFN × 2 (GEMV)
    LayerNorm × 2
    Logit projection
    Sampling (top-p)
    ─────────────────────────────────────────────────────

[2] cudaMemsetAsync
    → 다음 token을 위한 버퍼 초기화
    → attention mask 새 position을 0으로 설정
    → KV cache 새 slot 할당 및 초기화

[3] fill_reverse_indice_kernel
    → KV cache ↔ sequence position 역방향 인덱스 갱신
    → Qwen3 RoPE(Rotary Position Embedding)에서
      새 token의 position이 KV cache의 어느 slot인지 매핑
    → DynamicCache가 매 step 커지면서 재계산 필요

[4] cudaMemcpyAsync  (H→D)
    → CPU에서 만든 position_ids GPU로 전송
    → 갱신된 attention_mask GPU로 전송
    → prepare_inputs_for_generation()의 결과물
    → 기존 profiling 확인: 112건/inference = 6.4건/step

[5] cudaStreamSynchronize  ← 가장 길다 (~30~40ms)
    → token.item() 또는 stopping_criteria() 호출 시 발생
    → CPU: "GPU 스트림에 있는 거 다 끝낼 때까지 내가 기다림"
    → GPU: 유휴 상태로 CPU 대기
    → CPU: 생성된 token 값 읽기 → EOS인지 확인
    → EOS 아니면 → 다음 step dispatch

[6] vectorized_elementwise_kernel  (sync 이후)
    → attention mask 1칸 확장 (elementwise)
    → position ID 업데이트

[7] unrolled_elementwise_kernel
    → embedding lookup 준비
    → 다음 forward를 위한 input 구성

     ↓ 다음 token 사이클 시작
[1] GPU Transformer forward (57ms) ...
```

---

## 2. 53ms 오버헤드 분해 (추정)

| 연산 | 추정 시간 | 원인 |
|---|---|---|
| `cudaStreamSynchronize` | **~30~40ms** | EOS 체크 GPU-CPU sync (blocking) |
| `cudaMemcpyAsync` (H→D) | ~5ms | position_ids, attention_mask 매 step 전송 |
| `fill_reverse_indice_kernel` | ~3ms | KV cache 인덱스 매 step 재계산 |
| `cudaMemsetAsync` | ~2ms | 버퍼 초기화 |
| `vectorized_elementwise_kernel` + `unrolled_elementwise_kernel` | ~3~5ms | attention mask 확장, embedding 준비 |
| **합계** | **~43~55ms** | ≈ 실측 53ms |

**결론**: 53ms 오버헤드의 60~75%는 `cudaStreamSynchronize` 단 하나에서 발생한다.

---

## 3. cudaStreamSynchronize 발생 경로 (코드 레벨)

```python
# HuggingFace generate() 내부 — 매 decode step 실행
# transformers/generation/utils.py

while True:
    outputs = self(**model_inputs)          # GPU forward pass (57ms)
    next_token_scores = outputs.logits[:, -1, :]
    
    # top-p sampling (GPU)
    next_tokens = torch.multinomial(
        torch.softmax(next_token_scores, dim=-1), 1
    )
    
    # ★ 여기서 cudaStreamSynchronize 발생
    # stopping_criteria가 next_tokens.item()을 호출
    # .item() → Python scalar → GPU-CPU 동기화 강제
    if stopping_criteria(input_ids, scores):
        break
    
    # prepare_inputs_for_generation() (CPU)
    # → position_ids 갱신, attention_mask 확장
    # → cudaMemcpyAsync (H→D)
    model_inputs = self.prepare_inputs_for_generation(
        next_tokens, past_key_values, attention_mask
    )
```

**근본 원인**: HuggingFace의 EOS 체크 메커니즘은 매 step마다 GPU 텐서를 CPU로 읽어야 한다. 이 `.item()` 호출이 cudaStreamSynchronize를 강제한다.

---

## 4. 57ms GPU 커널이 BW Wall(81.2ms)보다 빠른 이유

```
BW Wall 계산 (기존): 22,157 MB / 273 GB/s = 81.2ms/token
NSight 실측:                                  57ms/token

57ms < 81.2ms → 모순처럼 보이지만 계산 가정이 달랐다.

수정:
  BW Wall은 전체 모델(22,157 MB) 기준으로 계산했지만,
  Decode 단계에서는 Vision Encoder가 실행되지 않는다.
  
  실제 Decode에서 로드하는 모델 부분:
    Cosmos Reason2 LM 레이어만 ≈ 7~10B params × 2 bytes
    ≈ 14~20 GB
    (Vision Encoder 2~3B, Action Expert 2.3B 제외)
  
  수정된 BW Wall:
    14 GB / 273 GB/s ≈ 51ms/token
    20 GB / 273 GB/s ≈ 73ms/token
  
  실측 57ms는 이 범위 안에 있다.

→ GPU 커널 자체는 이미 BW Wall에 상당히 근접해서 실행 중
→ 커널 최적화 여지가 아니라 커널 간 간격(53ms)이 문제
```

---

## 5. CUDA Graph 적용 시 기대 효과 (실측 기반)

```
현재 per-token:
  GPU 커널:  57ms  (BW Wall에 근접, 큰 개선 불가)
  오버헤드:  53ms  (CUDA Graph로 제거 가능)
  합계:     110ms

CUDA Graph 이후 이론:
  GPU 커널:  57ms  (변화 없음)
  오버헤드:   0ms  (GPU가 연속 실행, CPU 개입 없음)
  합계:      57ms  (1.93× 가속)

Decode 전체:
  현재:  17.5 × 110ms = 1,925ms
  이후:  17.5 × 57ms  = 997ms   (-928ms, -48%)

Total inference:
  현재:  5,009ms
  이후:  ~4,081ms  (-928ms, -18.5%)
```

**중요한 함의**: FP4 변환과 CUDA Graph를 결합할 때:

```
FP4 + CUDA Graph 없음:
  GPU 커널: 57/4 = 14.25ms
  오버헤드: 53ms  (변화 없음)
  per-token: 67.25ms  → 110/67.25 = 1.64× 가속

FP4 + CUDA Graph 있음:
  GPU 커널: 14.25ms
  오버헤드:  0ms
  per-token: 14.25ms → 110/14.25 = 7.72× 가속

→ CUDA Graph 없이 FP4만 하면 decode 가속의 대부분을 놓친다
→ 두 기법은 독립이 아니라 조합해야 진짜 효과
```

---

## 6. 다른 Phase와의 비교

| Phase | NSight 특성 | 오버헤드 | 주요 개선 방법 |
|---|---|---|---|
| Vision (714ms) | Compute-bound, continuous | 거의 없음 | TensorRT, 해상도 축소 |
| Prefill (1,472ms) | Compute-bound, 1회 실행 | 거의 없음 | Flash Attention, KV Reuse |
| **Decode (1,926ms)** | **BW-bound, 반복 루프** | **53ms/token (48%)** | **CUDA Graph** |
| Flow (890ms) | ODE 루프 | 있음 (randn 반복) | CUDA Graph, FP4 |

Decode만 유일하게 커널 실행 시간보다 오버헤드가 비슷한 크기다. 다른 phase는 한 번 GPU kernel이 시작되면 CPU 개입 없이 오랫동안 실행된다. Decode만 17.5회 반복되면서 매번 CPU-GPU 교대가 일어난다.

---

## 7. 측정 환경 정보

```
플랫폼:  Jetson AGX Thor
GPU:     SM 11.0 (NVIDIA Blackwell), LPDDR5X 273 GB/s
모델:    Alpamayo 1.5 (BF16), 11.08B params, 22.16 GB
측정:    NSight Systems (nsys), CUDA API Trace
입력:    Mock (torch.randn), 실험 baseline과 동일 조건
```
