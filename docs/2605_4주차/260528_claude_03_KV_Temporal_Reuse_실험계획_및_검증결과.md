# KV Temporal Reuse — 실험 계획 및 검증 결과

**날짜**: 2026-05-28  
**작성자**: Alphamayo 프로젝트 팀  
**관련 스크립트**: `scripts/inference/260528_kv_temporal_reuse_poc.py`  
**원칙**: 모델 가중치 변경 없음 / 양자화 없음 / `past_key_values` 외부 관리만

---

## 0. 현황 및 동기

### 확정 베이스라인 (2026-05-28, sdpa + DynamicCache + BF16)

```
Vision Encoder  :   728ms  (15%)
LM Prefill      : 1,423ms  (29%)  ← seq=3,086 tokens
Decode          : 1,818ms  (38%)  ← 17 steps × 107ms/step
Flow            :   870ms  (18%)
──────────────────────────────────
합계            : 4,838ms
```

torch.compile은 Triton 3.7.0 API 비호환(Inductor) + `_deepstack_process` dynamic boolean indexing(cudagraphs) 두 경로 모두 비호환으로 확정 폐기.  
→ **다음 최적화 후보**: LM Prefill(1,423ms, 전체의 29%)를 시스템 레벨에서 줄이는 것.

### 입력 토큰 구조

```
총 3,086 tokens
├── [0   : ~100 ] text_prefix   ≈  100 tokens  (3.2%)  ← 프레임 간 불변
├── [100 : ~2990] vision        ≈ 2,890 tokens (93.6%) ← 매 프레임 VE로 재계산
├── [2990: ~3072] ego           ≈   82 tokens  (2.7%) ← 매 프레임 갱신
└── [3072: 3086 ] text_suffix   ≈   14 tokens  (0.5%) ← 프레임 간 불변
```

### 핵심 관찰

LM Prefill 1,423ms의 대부분은 3,086개 토큰 전체에 대한 multi-head attention 연산이다.  
만약 이전 프레임의 KV cache를 `past_key_values`로 그대로 주입하면,  
모델은 그 위치들을 다시 계산하지 않고 1토큰(또는 변경된 일부 토큰)만 forward한다.  
→ **LM Prefill 비용을 단일 decode step 수준(107ms)으로 낮출 수 있다는 가설**.

---

## 1. KV Temporal Reuse 개념

### 1.1 일반적인 LLM decode 흐름

```
프레임 t:
  [1] VE 실행         : pixel_values → vision_embeddings          (728ms)
  [2] LM Prefill      : 3,086 tokens → attention → KV_t 생성    (1,423ms)
  [3] Decode          : KV_t 기반 autoregressive decode           (1,818ms)
  [4] Flow            : trajectory 계산                            (870ms)
  합계: 4,838ms
```

### 1.2 KV Temporal Reuse 흐름 (제안)

```
프레임 t-1: (첫 번째 / cold start)
  [1] VE(t-1) + LM Prefill(t-1)  → KV_{t-1} 저장               (2,151ms)
  [2] Decode(t-1)                                                (1,818ms)
  [3] Flow(t-1)                                                   (870ms)

프레임 t: (steady state)
  [1] VE(t)                       : 새 픽셀 처리 (항상 필요)       (728ms)
  [2] KV_hybrid 구성               : KV_{t-1}에서 vision 부분 교체
      - text_prefix KV: t-1에서 재사용 (불변)
      - vision KV:      t의 새 embedding으로 교체
      - ego/suffix KV:  ~96토큰만 새로 forward                   (~44ms)
  [3] Decode(t)                                                  (1,818ms)
  [4] Flow(t)                                                     (870ms)
  합계: ~3,460ms  (예상, 실험 C 검증 필요)
```

### 1.3 모델 구조 변경 여부

**없음.** `model.vlm()` 호출 방식과 `past_key_values` 인자만 바꾼다.

```python
# 기존: 매 프레임 full prefill
out = model.vlm(input_ids, pixel_values=pv, use_cache=True)

# KV 재사용: 저장된 KV + 변경 부분만 forward
out = model.vlm(
    input_ids=suffix_ids_only,   # 변경된 토큰만
    past_key_values=kv_hybrid,   # 이전 프레임 KV (부분 교체)
    cache_position=cpos,
    use_cache=True,
)
```

---

## 2. 실험 설계 (3단계)

각 단계는 이전 단계의 성공을 전제로 한다.

### 실험 A — 동일 프레임 완전 재사용 (oracle 상한)

**질문**: KV 재사용 메커니즘 자체가 동작하는가?

```
1st pass: full_prefill(t0) → KV_t0 (3,086 slots)
2nd pass: KV_t0 제공 + next_token(1개) only forward
기대:  2nd prefill ≈ 107ms  (1 decode step)
       Speedup ≈ 1,423ms / 107ms = 13.3× (LM Prefill 기준)
       Speedup ≈ 2,151ms / 107ms = 20.1× (VE+LM Prefill 기준)
```

성공 조건:
- `reuse_prefill < 200ms`
- `speedup > 5×`
- `decode_2nd`에서 EOS 정상 생성 (MAX_DECODE_STEPS 미도달)

### 실험 B — Text prefix KV 재사용 (현실적 하한)

**질문**: 불변 구간(text_prefix 100토큰)만 재사용해도 이익이 있는가?

```
KV_prefix = KV_t0[:100]  (text_prefix 구간만 추출)
partial_prefill: KV_prefix 제공 + input_ids[100:] (2,986 tokens) forward
기대:  prefill 절약 ≈ 100/3086 = 3.2% → ~46ms 절약
       의미: text_prefix 재사용만으로는 이익이 미미함을 확인
```

실험 B의 목적은 "개선 확인"이 아닌 **이론 부합 여부 확인**이다.  
실제 이익이 작다는 것을 수치로 보여 "vision KV 교체"의 필요성을 정당화한다.

### 실험 C — 인접 프레임 vision KV 부분 교체 (핵심 실험)

**질문**: 다른 프레임의 vision KV로 교체해도 모델이 정상 동작하는가?

```
KV_hybrid = [t-1의 text_prefix KV (100tok) | t의 vision KV (2,890tok)]
suffix_prefill: KV_hybrid 제공 + t의 input_ids[2990:] (96 tokens) forward
기대:  prefill 비용 ≈ 96/3086 × 1,423ms ≈ 44ms
       절약 ≈ 1,423ms - 44ms = 1,379ms (97% 절약)
       전체 파이프라인: 4,838ms - 1,379ms = 3,459ms

검증 항목:
  - suffix_prefill 성공 여부 (에러 없이 완료)
  - decode 정상 완료 (EOS 생성, MAX_DECODE_STEPS 미도달)
  - decode step 수: t1 full prefill과 유사한지
```

---

## 3. 실험 A 결과 분석 (2026-05-28 측정)

### 3.1 원시 데이터

| trial | full_prefill (ms) | reuse_prefill (ms) | speedup | decode_1st steps | decode_2nd steps |
|-------|:-----------------:|:------------------:|:-------:|:----------------:|:----------------:|
| WARMUP 1 | 2,971 | 106.9 | 27.8× | 19 | 18 |
| MEASURE 1 | 2,470 | 107.7 | 22.9× | 19 | 18 |
| MEASURE 2 | 2,433 | 107.2 | 22.7× | 16 | 18 |
| MEASURE 3 | 2,413 | 106.3 | 22.7× | 19 | 15 |
| **평균 (MEASURE)** | **2,438** | **107.1** | **22.8×** | **18.0** | **17.0** |

```
결론: ✅ 메커니즘 동작 확인 (다음 단계 진행 가능)
```

### 3.2 수치 해석

**① full_prefill 2,438ms vs baseline LM Prefill 1,423ms — 차이의 이유**

스크립트의 `full_prefill()`은 `model.vlm(pixel_values=...)` 를 통째로 호출한다.  
이는 Vision Encoder + Vision Projection + LM Forward를 모두 포함한다.

```
full_prefill(2,438ms) 분해 추정:
  Vision Encoder          :   728ms  (calibrate_seqlen 측정값)
  Vision Projection 등    :  ~287ms  (2,438 - 728 - 1,423)
  LM Forward (3,086 tok)  : 1,423ms
  ─────────────────────────────────
  합계                    : 2,438ms
```

Vision Projection 오버헤드 287ms는 calibrate_seqlen에서 "Flow(870ms)" 내에 분산 측정되었거나  
또는 VE/LM 경계 계산 방식의 차이로 인해 별도 포착되지 않았던 부분이다.

→ KV 재사용으로 절약되는 부분은 **LM Forward(1,423ms) + Vision Projection(~287ms)** 이다.  
→ Vision Encoder(728ms)는 어떤 경우에도 새 프레임 처리를 위해 항상 재실행된다.

**② reuse_prefill = 107.1ms — 이론과 정확히 일치**

KV 재사용 시 1토큰 forward 비용 = decode 1 step 비용과 동일해야 한다.  
측정값 107.1ms ≈ baseline decode step 107ms/step → **이론 완벽 부합**.

수식으로 표현하면:
$$t_{reuse} = \frac{W_{LM}}{BW_{DRAM}} \approx \frac{22\,\text{GB}}{231\,\text{GB/s}} = 95\,\text{ms}$$

실측 107ms는 이론 하한 95ms 대비 12% 오버헤드 (sampling + Python + sync 포함시 타당).

**③ speedup = 22.8× (oracle, 동일 프레임)**

```
full_prefill(2,438ms) / reuse_prefill(107ms) = 22.8×

전체 파이프라인 관점:
  현재:   VE(728) + Prefill(1,423) + Decode(1,818) + Flow(870) = 4,838ms
  oracle: VE(728) + Reuse( 107)   + Decode(1,818) + Flow(870) = 3,523ms
  절약:   1,316ms (27% 단축)
```

주의: 이 27% 절약은 oracle case(동일 프레임 재사용)의 상한이다.

**④ decode 품질 유지 확인**

| 항목 | 1st (기준) | 2nd (KV 재사용) |
|------|:----------:|:---------------:|
| 평균 step 수 | 18.0 | 17.0 |
| MAX_DECODE_STEPS(80) 도달 | 없음 | 없음 |
| EOS 정상 생성 | ✅ | ✅ |

- step 수 차이(18.0 vs 17.0)는 top-p sampling의 확률적 특성으로 설명된다.
- 중요한 것은 **모두 EOS를 정상 생성**했다는 점이다.
- KV를 외부에서 주입해도 decode 품질이 유지된다는 것을 확인.

**⑤ WARMUP 효과 분석**

WARMUP 1의 full_prefill(2,971ms)이 MEASURE 평균(2,438ms)보다 **533ms 높다**.  
이는 첫 번째 실행 시:
- CUDA 커널 로딩 및 JIT 컴파일
- GPU L2 캐시 미스 (모델 가중치 cold)
- CUDA 스트림 스케줄링 초기화

등으로 인한 warm-up 효과이다.  
KV 재사용(107ms)은 WARMUP에서도 안정적 → 1토큰 forward는 이러한 초기화 비용이 없다.

### 3.3 실험 A 결론

| 검증 항목 | 기대 | 실측 | 판정 |
|----------|------|------|------|
| reuse_prefill < 200ms | ✅ | 107.1ms | ✅ |
| speedup > 5× | ✅ | 22.8× | ✅ |
| decode 정상 EOS | ✅ | 모든 trial | ✅ |
| 이론값 부합 (107ms ≈ decode step) | ✅ | 107ms | ✅ |

→ **KV 재사용 메커니즘은 정상 동작한다.** 실험 B, C 진행 가능.

---

## 4. 실험 B 예상 분석

### 4.1 예상 결과

text_prefix(100토큰)의 KV를 재사용하고 나머지 2,986토큰을 다시 forward.

```
partial_prefill(2,986 tokens) ≈ (2,986 / 3,086) × 1,423ms ≈ 1,377ms
절약: 1,423ms - 1,377ms = 46ms (3.2%)
```

이 수치는 의도적으로 **작다**. 실험 B의 목적:
- text_prefix 재사용만으로는 이익이 미미함을 수치로 확인
- 이를 통해 "대부분의 이익은 vision KV 교체에서 온다"는 명제를 뒷받침

### 4.2 주의사항: pixel_values와 partial forward

`input_ids[:, 100:]`에는 vision token placeholder ID가 포함된다.  
따라서 `model.vlm(input_ids=remaining_ids, pixel_values=..., past_key_values=prefix_kv)` 호출 시  
모델이 vision token 위치를 pixel_values에서 계산한 임베딩으로 교체한다.  
이것이 올바르게 동작하는지 검증이 필요하다 (attention_mask의 shape mismatch 가능).

---

## 5. 실험 C 예상 분석 및 리스크

### 5.1 예상 결과 (성공 시)

```
KV_hybrid 구성:
  레이어별: [t-1 text_prefix KV (100tok) | t vision KV (2,890tok)]
suffix_prefill: ego(82tok) + text_suffix(14tok) = 96tok만 새로 계산

suffix_prefill 비용 ≈ (96 / 3,086) × 1,423ms ≈ 44ms
전체: VE(728) + KV_hybrid(~1ms) + suffix_prefill(44) + Decode(1,818) + Flow(870) ≈ 3,461ms
      vs baseline 4,838ms → **1,377ms 절약 (28.5%)**
```

### 5.2 핵심 리스크: Vision KV 교체 후 모델 내부 일관성

**문제**: KV_hybrid는 t-1의 text_prefix attention context와 t의 vision attention context가 혼재한다.

트랜스포머의 self-attention에서 position i의 KV는  
position 0 ~ i-1의 모든 컨텍스트를 이미 흡수한 상태이다.  
즉, `KV_t[vision_layer]`는 `KV_t[text_prefix_layer]`와 **함께 계산**된 값이다.

```
KV_hybrid 혼재 상황:
  text_prefix KV: t-1의 컨텍스트를 흡수한 상태
  vision KV:      t의 컨텍스트를 흡수한 상태
  → 두 구간이 서로 다른 "attention 이력"을 가짐
  → suffix forward 시 이 불일치가 품질 저하를 유발할 수 있음
```

이 불일치가 실제로 문제가 되는지는 **실험 C로만 확인 가능**하다.

- 자율주행 도메인에서 연속 프레임은 시각적으로 매우 유사하다.
- t-1과 t의 vision KV가 충분히 유사하다면 혼재 효과는 미미할 수 있다.
- 반대로 급격한 장면 변화(교차로, 터널 진입) 시 품질 저하 우려.

### 5.3 실험 C 성공 판정 기준

| 항목 | 성공 기준 |
|------|---------|
| suffix_prefill 완료 | 에러 없이 완료 |
| decode EOS 생성 | MAX_DECODE_STEPS(80) 미도달 |
| decode step 수 | t1 full prefill 대비 ±5 steps 이내 |
| prefill 비용 | < 200ms (96token × 2ms/token 가정) |

---

## 6. 전체 로드맵과의 연계

### 6.1 현재 위치

```
베이스라인: 4,838ms
  └─ torch.compile: ❌ 비호환 확정
  └─ KV Temporal Reuse (실험 중)
       └─ 실험 A: ✅ 완료 (22.8× speedup on prefill, oracle 상한 확인)
       └─ 실험 B: 🔄 예정 (text_prefix 재사용, 소폭 절약 예상)
       └─ 실험 C: 🔄 예정 (vision KV 교체, 핵심 검증)
```

### 6.2 시나리오별 예상 전체 시간

| 시나리오 | LM Prefill | 전체 | 절약 | 가능 여부 |
|---------|:----------:|:----:|:----:|:-------:|
| baseline | 1,423ms | 4,838ms | — | ✅ 현재 |
| oracle (동일 프레임 재사용) | 107ms | 3,523ms | 1,315ms (27%) | ✅ 실험 A 확인 |
| text_prefix만 재사용 (실험 B) | ~1,377ms | 4,792ms | 46ms (1%) | 예측 (미미) |
| vision KV 교체 (실험 C) | ~44ms | 3,461ms | 1,377ms (28.5%) | 🔄 검증 중 |

### 6.3 Rolling Pipeline과의 관계

최종 목표 99ms는 단일 추론이 아닌 Rolling Trajectory Pipeline으로 재정의됨.  
KV Temporal Reuse는 그 중간 단계:
1. **1단계** (≤3,500ms): KV Temporal Reuse + 기타 최적화
2. **2단계** (≤1,500ms): Speculative Decoding으로 Decode 단축
3. **최종**: Rolling Trajectory — 이전 trajectory를 rolling window로 유지,  
   매 프레임 전체 추론이 아닌 incremental update

---

## 7. 실험 체크리스트

### 실험 A (완료)
- [x] 스크립트 작성: `260528_kv_temporal_reuse_poc.py`
- [x] 메커니즘 동작 확인: reuse_prefill = 107ms (speedup 22.8×)
- [x] Decode 품질 유지 확인: EOS 정상 생성
- [x] DynamicCache 버전 무관 접근 (`_cache_to_kv_pairs` 헬퍼)

### 실험 B (예정)
- [ ] partial prefill (pixel_values + remaining_ids) 실행 확인
- [ ] 절약량 측정 (기대: ~46ms)
- [ ] attention_mask shape mismatch 없는지 확인

### 실험 C (예정)
- [ ] t1 데이터 로드 (`t0_us + 500_000`)
- [ ] KV hybrid 구성 (`_cache_to_kv_pairs` + `_build_cache_from_kv`)
- [ ] suffix_prefill (96 tokens, `pixel_values=None`) 실행 확인
- [ ] decode 품질 비교: t1 full prefill vs hybrid
- [ ] 다른 Δt (100ms, 200ms, 500ms) 에서도 테스트

---

## 8. 핵심 발견사항 요약

| 항목 | 내용 |
|------|------|
| 메커니즘 | `past_key_values` 외부 주입으로 LM Prefill 비용 대폭 절감 |
| oracle 상한 (실험 A) | 1,423ms → 107ms (13.3×), 전체 27% 단축 |
| 이론값 부합 | reuse 107ms = decode step 107ms = 22GB ÷ 231GB/s 한계 |
| KV 구조 | 36 layers × [1, 8, 3086, 128] × BF16 = 455MB (128GB RAM에서 무제약) |
| 핵심 불확실성 | vision KV 교체 시 모델 출력 품질 유지 여부 (실험 C) |
| 모델 변경 여부 | **없음** — `model.vlm()` 호출 인자만 변경 |
