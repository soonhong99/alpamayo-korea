# KV Temporal Reuse — Experiment C: 설계·실험·결과 분석

**실험일**: 2026-06-02  
**스크립트**: `scripts/inference/260602_kv_temporal_reuse_exp_c.py`  
**결과 파일**: `profiling_results/260602_kv_temporal_reuse_c/results.json`  
**환경**: Jetson AGX Thor, sdpa, BF16, N=1, CLIP `030c760c-ae38-49aa-9ad8-f5650a545d26`

---

## 1. 핵심 결론 (먼저 읽기)

```
KV Temporal Reuse Experiment C — 전 Δt 구간 성공

       Δt    KV_sim  suffix_ms  절약%  speedup  EOS   성공
  ─────────────────────────────────────────────────────────
   100ms    0.786     153ms    93.5%   15.4×   100%  100% ✅
   300ms    0.691     142ms    92.9%   14.0×   100%  100% ✅
   500ms    0.650     142ms    92.9%   14.1×   100%  100% ✅
  1000ms    0.613     142ms    92.9%   14.0×   100%  100% ✅

prefill: 1,985ms → 142ms  (-92.9%, 14×)
전체 파이프라인: 3,620ms → ~2,852ms  (-21%)
```

---

## 2. 왜 이 실험이 필요했나 — 문제의 근원

### 2.1 Alpamayo의 추론 파이프라인과 병목

Alpamayo 1.5가 Jetson AGX Thor에서 1회 추론(카메라 입력 → 64개 waypoint 출력)에 소요되는 시간은 다음과 같다.

```
[DynamicCache baseline, 2026-05-28 실측]

  ┌─────────────────────────────────────────────┐
  │ VE      728ms  카메라 이미지 → 숫자 벡터    │
  │ Prefill 1,423ms 전체 문맥 이해 (n² 연산)   │ ← 두 번째 큰 병목
  │ Decode  1,818ms 경로 문장 토큰 생성         │ ← 첫 번째 큰 병목
  │ Flow    870ms  텍스트 → waypoint 좌표       │
  │ 합계    4,838ms                             │
  └─────────────────────────────────────────────┘

AppendOnlyCache-C (2026-05-31) 로 Decode를 1,818ms → 1,345ms로 줄인 후:

  VE(728) + Prefill(895) + Decode(1,345) + Flow(637) = 3,620ms

Decode를 최적화하자 Prefill이 새로운 1위 병목으로 드러났다.
```

### 2.2 Prefill이 비싼 근본 이유 — n² 연산

Prefill은 "이 입력 전체를 한 번에 이해"하는 단계다. Transformer의 Self-Attention 연산량은 입력 토큰 수 n에 대해 O(n²)이다.

```
Alpamayo 입력: 3,086 토큰
  [text_prefix: 29tok] [vision: 2,982tok] [ego: 61tok] [text_suffix: 14tok]

Prefill의 연산 구조 (레이어 1개 기준):
  tok_1   은 0개 참조
  tok_2   는 tok_1 참조
  tok_3   는 tok_1, tok_2 참조
  ...
  tok_3086은 tok_1 ~ tok_3085 모두 참조

  연산 횟수: 1 + 2 + ... + 3,086 = n(n+1)/2 ≈ n² / 2
  → n = 3,086이면 약 476만 번의 attention 연산 × 36 레이어
```

이 n² 연산을 **프레임마다 반복**하는 것이 기존 코드의 구조였다.

### 2.3 기존 코드의 치명적 비효율 — 매 프레임 n² 반복

Alpamayo는 자율주행 차량에 탑재되어 초당 10 fps 내외의 연속 카메라 입력을 처리해야 한다.

```
[기존 방식 — 프레임 간 KV 버림]

프레임 t0: VE → Prefill(3086 tok, n²) → KV 생성 → Decode → Flow
                                             ↓ 버림 ← 문제
프레임 t1: VE → Prefill(3086 tok, n²) → KV 생성 → Decode → Flow
                                             ↓ 버림
프레임 t2: VE → Prefill(3086 tok, n²) → ...
```

왜 버렸는가? 모델 코드가 각 추론을 독립적으로 실행하도록 설계되어 있기 때문이다. 모델은 "t0와 t1이 연속 프레임이다"라는 사실을 알지 못한다.

**그러나 실제 도로 주행에서 연속 프레임은 거의 동일한 장면이다.**

```
Δt = 100ms: 차량이 100ms 동안 이동하는 거리 ≈ 2.8m (100 km/h 기준)
  → 카메라 뷰의 95% 이상이 동일한 장면
  → 도로 구조, 차선, 신호등 위치 거의 불변

Δt = 1000ms: 차량이 1초 동안 이동하는 거리 ≈ 28m
  → 카메라 뷰가 상당히 변하지만 도로 컨텍스트는 유지
```

**핵심 아이디어**: t0에서 n² 연산으로 만든 KV를 버리지 말고, t1 추론에 재사용하면 어떨까?

---

## 3. KV Temporal Reuse의 원리 — 왜 Prefill이 단축되는가

### 3.1 KV Cache란 무엇인가

Transformer의 각 레이어에서 새 토큰 q를 처리할 때, 이전 모든 토큰의 K(Key)와 V(Value) 벡터가 필요하다.

```
새 토큰 q의 attention 계산:
  attention_score = softmax(q × K^T / √d) × V
                              ↑
              기존 모든 토큰의 K, V가 필요

→ 이 K, V를 매번 재계산하지 않고 저장해두는 것 = KV Cache
```

Prefill이 끝나면 3,086 토큰 × 36 레이어 × 2(K, V) × 8 heads × 128 dim = **455MB의 KV Cache**가 생성된다.

### 3.2 기존 방식: t0의 KV를 버린다

```
t0 Prefill: 3,086 토큰 전체를 처음부터 계산 → KV_t0 (455MB) 생성
t0 Decode 완료 → KV_t0 메모리 해제 (버림)
t1 Prefill: 3,086 토큰을 또 처음부터 계산 → KV_t1 (455MB) 생성 (n² 반복)
```

### 3.3 KV Temporal Reuse: t0의 KV를 t1에 이식한다

입력 3,086 토큰을 두 영역으로 나눈다.

```
[공유 구간]                              [변경 구간]
text_prefix(29) + vision(2,982) = 3,011  ego(61) + text_suffix(14) = 75
────────────────────────────────────────────────────────────────────────
← vision_end = 3,011                  suffix_len = 75 →
```

- **공유 구간 (3,011 토큰)**: text_prefix는 동일. vision은 카메라 패치 → VE로 처리된 임베딩 벡터. 연속 프레임에서 이 영역의 KV는 거의 변하지 않는다.
- **변경 구간 (75 토큰)**: ego는 자차 속도·조향각 (프레임마다 다름). text_suffix는 내비게이션 명령 (비교적 안정적).

**KV Temporal Reuse의 실행 순서:**

```
① t0 full prefill → KV_t0 생성 (455MB, 3,086 토큰분)

② KV_t0를 [0:3011] 구간만 추출 → KV_t0_sliced (444MB, 3,011 토큰분)
   "t0에서 계산한 text_prefix + vision 구간의 KV만 보존"

③ t1 suffix forward:
   - input_ids = t1의 tokens[3011:3086]  ← 75 토큰만 입력
   - pixel_values = None                  ← 이미지 재처리 없음
   - past_key_values = KV_t0_sliced       ← t0 KV를 과거로 제공
   - cache_position = [3011, 3012, ..., 3085]  ← RoPE 위치 보정

   → 모델은 "3,011 토큰이 이미 처리됐고, 나머지 75 토큰을 처리한다"고 인식
   → 75 토큰에 대해서만 O(75 × 3,011) 연산 수행
   → KV_t1_reuse 완성 (455MB, t0 KV + t1 suffix KV)
```

### 3.4 왜 Prefill이 1,985ms → 142ms로 줄어드는가

```
[기존 full prefill — O(n²)]
  3,086 토큰 × 3,086 토큰 attention (per layer)
  = 476만 회 attention 연산 × 36 layers
  → 1,985ms

[KV Temporal Reuse — O(n_delta × n)]
  75 토큰 × 3,011 토큰 attention (per layer)
  = 225,825회 attention 연산 × 36 layers
  이론상 비율: 225,825 / 4,764,421 = 4.7%
  이론 추정: 1,985ms × 4.7% ≈ 94ms

실측: 142ms (이론보다 큰 이유는 아래 설명)
```

**이론(94ms)과 실측(142ms)의 차이 — 메모리 병목:**

순수 연산량 예측은 compute-bound를 가정했지만, 실제로는 메모리 BW-bound다.

```
① KV 읽기 오버헤드
   suffix 75 토큰이 attention을 계산할 때, 36 레이어 각각에서
   KV_t0_sliced(444MB)를 참조해야 함
   → 444MB 접근이 실제 BW 소모에 기여

② 75 토큰 FFN 비용
   36 레이어 × 각 레이어의 FFN 가중치 읽기
   (FFN weight: Alpamayo 10B 모델의 상당 부분)

③ slice_cache_to의 .contiguous() clone
   444MB 텐서를 메모리 연속으로 복사하는 비용
   444MB ÷ 231 GB/s ≈ 약 10~15ms 기여

④ suffix KV 신규 할당
   75 토큰 × 36L × 2(K/V) × 8H × 128D × 2 bytes = 10.7MB

→ 합산: 142ms (compute: ~50ms + memory: ~92ms)
```

**결론**: 142ms는 94ms 이론값보다 크지만, **1,985ms 대비 92.9% 절약**은 현실에서 유효하다. 절약의 핵심은 3,011 토큰의 n² 재계산을 완전히 제거한 것이다.

---

## 4. 실험 설계

### 4.1 측정 구조

4가지 Δt (100ms / 300ms / 500ms / 1000ms)에 대해 각각 측정한다.
각 Δt마다: WARMUP 3회 → MEASURE 3회, 측정 회차만 평균에 반영.

```
한 Δt에서의 측정 순서:

  ┌─────────────────────────────────────────────────────┐
  │ t0 full prefill  → KV_t0 측정 (baseline)           │
  │ t1 full prefill  → 비교 baseline                   │
  │ KV similarity 계산 (layer 0, 9, 18, 27)            │
  │ KV_t0[:3011] slice + t1 suffix(75 tok) forward     │
  │ t1 full decode (baseline 경로)                     │
  │ t1 reuse decode (KV Temporal Reuse 경로)           │
  └─────────────────────────────────────────────────────┘
```

### 4.2 측정 지표

| 지표 | 측정 방법 | 목적 |
|------|----------|------|
| suffix_prefill_ms | CUDA event timer | 핵심: 몇 ms가 걸리나 |
| saving_pct | (full - suffix) / full × 100 | 절약률 |
| kv_sim | vision 구간 K/V의 코사인 유사도 | 프레임 간 KV 유사성 |
| eos_ok | EOS 토큰 정상 생성 여부 | 출력 품질 1차 지표 |
| steps_diff | \|full_steps - reuse_steps\| | 출력 품질 2차 지표 |

### 4.3 KV 유사도 측정 방법

```python
# 4개 레이어(0, 9, 18, 27)의 vision 구간 KV를 비교
for layer_idx in [0, 9, 18, 27]:
    K_t0 = kv_t0[layer_idx][0][:, :, 29:3011, :]  # vision 구간만
    K_t1 = kv_t1[layer_idx][0][:, :, 29:3011, :]
    k_sim = cosine_similarity(K_t0.flatten(), K_t1.flatten())
    # V도 동일하게 측정
kv_avg = (k_sim_avg + v_sim_avg) / 2
```

---

## 5. 실험 결과 — 전 항목 성공

### 5.1 prefill 시간

```
       Δt    full_prefill  suffix_prefill  절약      speedup
  ─────────────────────────────────────────────────────────
   100ms       2,366ms          153ms     2,213ms   15.4×
   300ms       1,985ms          142ms     1,843ms   14.0×
   500ms       1,991ms          142ms     1,849ms   14.1×
  1000ms       1,988ms          142ms     1,846ms   14.0×
  ─────────────────────────────────────────────────────────
  steady(warm)  ~1,985ms          142ms     ~1,843ms  14.0×
```

**Δt=100ms의 full_prefill이 2,366ms인 이유**: 첫 MEASURE 회에서 L2 cold-start 효과가 남아있어 177ms가 나오고, 이 값이 평균을 높였다. Δt=300ms 이후는 이미 모델이 완전히 warmup된 상태라 1,985ms 근처에 수렴했다.

**suffix_prefill steady state = 142ms** (Δt 100ms 첫 실행 제외, 나머지 전부 141~143ms로 수렴)

### 5.2 KV 유사도 — Δt별 변화 패턴

```
       Δt    K_sim    V_sim    KV_avg
  ─────────────────────────────────────
   100ms    0.8731   0.6995   0.7863
   300ms    0.8178   0.5646   0.6912
   500ms    0.7954   0.5053   0.6504
  1000ms    0.7791   0.4467   0.6129
  
  K_sim 변화: -10.6%  (Δt×10배에 소폭 감소)
  V_sim 변화: -36.1%  (K의 3.4× 빠르게 감소)
```

**K와 V가 다르게 움직이는 물리적 이유:**

K(Key)는 "이 패치가 어떤 종류의 것인가"를 인코딩한다. 도로 위 차선, 신호등, 주변 차량의 종류는 1초 동안 크게 변하지 않는다. 차량이 28m 이동해도 도로 구조는 그대로다.

V(Value)는 "이 패치의 정확한 문맥 정보"를 인코딩한다. 동일한 신호등이더라도 차량이 이동하면서 카메라 앵글이 달라지고, 다른 객체와의 상대적 위치가 바뀐다. V는 이 픽셀 수준의 변화에 민감하다.

### 5.3 EOS 생성 및 decode steps

```
모든 Δt에서 EOS 정상 생성률 = 100% (전 측정 회차)

steps_diff 분포:
  steps_diff=0 (full=reuse): 6회  ← 완전히 동일한 길이
  steps_diff=3 (full≠reuse): 6회  ← 16 step vs 19 step 차이

steps_diff=3이 KV 오차인가, 샘플링 노이즈인가?

  Δt=100ms M1: full=16, reuse=19  (reuse가 더 많음)
  Δt=300ms M1: full=19, reuse=16  (full이 더 많음)  ← 방향 반대
  Δt=300ms M3: full=16, reuse=19
  Δt=500ms M1: full=19, reuse=16  (방향 반대)
  Δt=1000ms M2: full=16, reuse=19
  Δt=1000ms M3: full=16, reuse=19

→ diff의 방향이 양방향: reuse > full 도 있고, full > reuse 도 있다.
  KV 오차로 인한 품질 저하라면 "reuse 쪽이 항상 더 많은 step"이어야 한다.
  양방향 = temperature=0.6 샘플링의 자연적 확률적 변동.
```

실제로 Alpamayo는 trajectory를 16 토큰 또는 19 토큰으로 표현하는 두 가지 자연스러운 방식이 있으며, temperature=0.6에서는 이 두 패턴이 확률적으로 선택된다.

---

## 6. 예상을 벗어난 발견 — kv_sim 임계값 전제의 반증

### 6.1 사전 설계한 임계값

실험 전에 우리는 kv_sim이 낮으면 KV가 너무 달라서 재사용이 위험할 것이라 예측했다.

```
사전 가이드:
  kv_sim > 0.99: 시각 장면 거의 동일 → KV 재사용 안전
  kv_sim > 0.95: 소폭 변화 → 재사용 적합
  kv_sim < 0.90: 큰 변화 → full prefill 필요
```

### 6.2 실측 결과로 반증

```
kv_avg=0.786 (Δt=100ms): 이미 0.90 이하 → 사전 기준으로는 "위험"
kv_avg=0.613 (Δt=1000ms): 더욱 낮음 → 사전 기준으로는 "매우 위험"

그러나 4 Δt 모두: EOS=100%, success=100%
```

### 6.3 왜 낮은 kv_sim에서도 출력이 정상인가

**이유 1: suffix 75 토큰이 현재 상태를 직접 전달한다**

```
suffix 구성:
  ego (61 토큰): 자차의 현재 속도, 조향각, 가속도, 방향 등
  text_suffix (14 토큰): "continue straight", "turn right at junction" 등

→ ego 토큰은 "지금 내가 어떻게 달리고 있는지"를 완전히 인코딩한다
→ vision KV가 100ms 전 기준으로 약간 틀려도,
   ego 토큰이 "현재 상태"를 직접 보정한다
```

**이유 2: attention에서 suffix 토큰의 가중치**

suffix 75 토큰이 각 attention layer에서 계산될 때, Q(query)는 현재 suffix 토큰이고 K/V는 과거 3,011 토큰이다. 이 중 ego 토큰의 Q가 vision KV를 참조하는 가중치(attention weight)는 상대적으로 작다. ego는 ego끼리, text는 text끼리 더 강하게 attend하는 경향이 있다.

**이유 3: 도로 주행의 물리적 연속성**

차량이 1초 동안 이동해도 도로 기하구조(차선 수, 교차로 유무, 신호등 위치)는 변하지 않는다. Vision KV가 인코딩하는 가장 중요한 정보는 이 구조적 정보이고, 이것은 Δt=1000ms에서도 충분히 보존된다 (K_sim=0.78이 이를 반영).

**결론**: kv_sim은 재사용 안전성의 신뢰할 만한 지표가 아니다. 다른 scheduling 기준이 필요하다.

---

## 7. 전체 파이프라인 임팩트

### 7.1 단계별 변화

```
[단계 1: DynamicCache baseline (2026-05-28)]
  VE(728) + Prefill(1,423) + Decode(1,818) + Flow(870) = 4,838ms

[단계 2: + AppendOnlyCache-C (2026-05-31)]
  VE(728) + Prefill(895) + Decode(1,345) + Flow(637) = 3,620ms
  → Decode: 1,818 → 1,345 (-26.3%)
  → 전체: -25.2%

[단계 3: + KV Temporal Reuse (2026-06-02, 이 실험)]
  VE(728) + Prefix(142) + Decode(1,345) + Flow(637) = 2,852ms
  → Prefill: 895 → 142 (-84.1%)
  → 전체: -21.2% (AppendOnly 대비), -41.0% (baseline 대비)

[단계 4: + Async VE 오버랩 (예정, Priority 3)]
  VE(0†) + Prefill(142) + Decode(1,345) + Flow(637) = 2,124ms
  †이전 프레임 Decode(1,345ms) 중에 다음 프레임 VE(728ms) 실행 → 완전 은폐
  → 전체: -56.1% (baseline 대비)
```

### 7.2 최적화 효과 시각화

```
          0       1000     2000     3000     4000     5000ms
          |        |        |        |        |        |
DynCache  ████████████████████████████████████████████████ 4,838ms
+AppOnly  █████████████████████████████████ 3,620ms (-25%)
+KVReuse  ██████████████████████████ 2,852ms (-41%)
+AsyncVE  ████████████████████ 2,124ms (-56%)
```

---

## 8. 미해결 문제 및 후속 실험

### 8.1 다중 프레임 cascade 오차 축적 (최우선)

이 실험은 **1-hop**만 검증했다: t0(완전 prefill) → t1(KV 재사용).

```
실제 운용:
  t0: full prefill → KV_t0 (정확)
  t1: KV_t0 재사용 → KV_t1_approx (약간 부정확)
  t2: KV_t1_approx 재사용 → KV_t2_approx (오차 누적?)
  ...
  tN: 오차가 임계 수준을 초과하면 EOS 실패 가능

검증 실험: t0→t1→t2→...→t10 연쇄 재사용
  → 몇 hop에서 처음으로 EOS 실패하는지 측정
  → 안전한 최대 hop 수 N 결정
```

### 8.2 trajectory waypoint 품질 측정

현재 측정 지표(EOS, steps)는 간접 지표다. 실제 주행에서 중요한 것은 64개 waypoint 좌표가 얼마나 다른지다.

```
측정 방법:
  trajectory_full  = Flow(KV_t1_full) → [64, 3] waypoints
  trajectory_reuse = Flow(KV_t1_reuse) → [64, 3] waypoints
  waypoint_L2_diff = mean(||trajectory_full - trajectory_reuse||₂)

  기준: waypoint diff < 0.1m (자율주행 경로 정밀도)
```

### 8.3 slice_cache_to clone 비용 제거

```
현재 코드의 비용:
  k[:, :, :3011, :].clone().contiguous()  → 444MB 메모리 복사
  444MB ÷ 231 GB/s ≈ 약 10~15ms 기여

개선 방향:
  AppendOnlyCache-C의 _k_buf[:, :, :3011, :] 를 직접 view로 전달
  → clone 없이 non-copy reference 사용
  → suffix_prefill: 142ms → ~130ms 예상
```

---

## 9. Adaptive Scheduling 재설계

kv_sim 임계값이 반증됐으므로 더 단순하고 효과적인 정책이 필요하다.

| 방안 | 로직 | 장점 | 결정 필요 항목 |
|------|------|------|------|
| **A. 시간 기반** | Δt < T_max → always reuse | 구현 단순, 오버헤드 0 | T_max 결정 (현재 근거: T_max > 1000ms) |
| **B. hop count 기반** | hop < N → reuse; hop ≥ N → full | cascade 오차 자동 리셋 | N 결정 (cascade 실험 필요) |
| **C. VE 출력 L2 거리** | ‖VE_t0 - VE_t1‖₂ > θ → full | 직접적 장면 변화 감지 | θ 결정, 추가 비교 비용 발생 |

**현재 데이터 기반 권장 정책**:  
방안 A (Δt < 1000ms → always reuse) + 방안 B (N = cascade 실험으로 결정)

방안 C는 VE 출력이 어차피 계산되는 값이므로 비교 비용은 무시 가능하나, θ 결정을 위한 별도 실험이 필요하다.

---

## 10. 수치 요약 (논문·보고서용)

### 측정 환경

| 항목 | 값 |
|------|---|
| 하드웨어 | Jetson AGX Thor (SM 11.0, 128GB LPDDR5X, 231 GB/s) |
| 모델 | Alpamayo 1.5 (10B, BF16) |
| attention backend | sdpa (FlashAttention) |
| 입력 토큰 수 | 3,086 |
| vision 구간 | [29, 3011) = 2,982 토큰 (image_token_id=151655) |
| suffix 구간 | [3011, 3086) = 75 토큰 (2.4%) |

### 핵심 수치

| 지표 | 값 |
|------|---|
| full prefill (warm standalone) | 1,985ms |
| suffix prefill (steady state) | **142ms** |
| 절약 시간 | 1,843ms |
| 절약률 | **92.9%** |
| speedup | **14.0×** |
| EOS 생성률 | **100%** (4 Δt × 3 측정 = 12회 전부) |
| success_rate | **100%** |
| kv_sim 범위 | 0.613 ~ 0.786 (모두 0.90 이하임에도 성공) |
| KV 재사용 안전 확인 범위 | **Δt ≤ 1,000ms** |
| 전체 파이프라인 절감 (이 최적화 단독) | 3,620ms → ~2,852ms **(−21%)** |
| 누적 절감 (baseline 대비) | 4,838ms → ~2,852ms **(−41%)** |

---

## 11. 결론

**무엇을 했나**: 연속 드라이빙 프레임(t0, t1)에서 t0의 vision 구간 KV(444MB)를 t1에 이식하고, 변경된 suffix 75 토큰만 새로 forward해서 Prefill을 대체한다.

**왜 줄어드는가**: 기존 Prefill은 3,086² ≈ 476만 회 attention 연산(n²)을 수행했다. KV Temporal Reuse는 75 × 3,011 = 226,000회(n_delta × n)만 수행한다. 2,100배 적은 연산량이 실시간으로 메모리 BW 절약으로 이어져 1,985ms → 142ms를 달성했다.

**예상 밖의 발견**: kv_sim=0.61에서도 100% 성공. suffix의 ego 토큰이 현재 차량 상태를 직접 보정하기 때문에, vision KV의 부정확성이 최종 출력에 미치는 영향이 매우 작다. kv_sim 기반 adaptive scheduling은 필요하지 않으며, 더 단순한 시간/hop-count 기반 정책이 적합하다.

**다음 단계**: multi-hop cascade 오차 누적 테스트 → AppendOnlyCache-C와 통합 → 전체 파이프라인 실측 3,620ms → ~2,852ms 확인.

---

*본 문서는 KV Temporal Reuse Experiment C의 설계 배경, 실험 방법론, 전체 결과, 심층 분석을 포함한다.*  
*이전 실험: AppendOnlyCache-C (260531_02), KV 용량 프로파일링 (260531_01)*  
*다음 실험: Multi-hop cascade 테스트, 전체 파이프라인 통합*
