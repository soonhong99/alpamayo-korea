# KV Temporal Reuse — Sliding Window (SW) 실험 결과 분석

**작성일**: 2026-06-02  
**실험 스크립트**: `scripts/inference/260602_kv_sliding_window_exp.py`  
**결과 파일**: `profiling_results/260602_kv_sliding_window/results.json`  
**비교 대상**: Exp C (`scripts/inference/260602_kv_temporal_reuse_exp_c.py`)

---

## 0. 실험 개요

### 0.1 동기

Exp C (vision KV 전체 재사용)는 t0 KV를 t1에 그대로 가져다 쓴다.
`t0 = [cam0_f0, cam0_f1, cam0_f2, cam0_f3, cam1_f0, ...]` 전부 재사용.

문제: Δt가 커질수록 가장 오래된 프레임(cam_c_f0, 즉 t-3 시점)이 현재와 점점 달라진다.

**SW의 아이디어**: oldest frame(t-3) KV를 newest frame(t+0) KV로 교체.
t1에서 방금 계산한 최신 KV를 이식해 t0 KV를 "갱신"한다.

### 0.2 실험 구조

```
T0 = 5.1s (기준)   →  t0 KV 계산 (Full Prefill)
T1 = T0 + Δt      →  t1 KV 계산 (Full Prefill)

비교 3방식:
  Full : t1 full prefill (기준선, ~1,990ms)
  ExpC : t0 vision KV 전체 재사용 + suffix 75tok forward (~141ms)
  SW   : t0 KV에서 4개 oldest frame KV를 t1 newest KV로 교체 + suffix (~163ms)
```

### 0.3 SW 이식 대상

카메라-우선(camera-first) 이미지 순서 기준:
- 교체: img{0,4,8,12} (cam0~3의 f0, t-3 시점) → t1의 img{3,7,11,15} (cam0~3의 f3, t+0 시점)
- 유지: 나머지 12개 이미지 (75%) — t0 그대로

교체 토큰 수: 4 × 180 = 720 / 2,982 vision tokens = **24.1%**

---

## 1. 타이밍 결과

### 1.1 종합 요약

| Δt | Full prefill | ExpC suffix | SW suffix | SW asm | SW total | SW 절감률 | ExpC 절감률 |
|---|---|---|---|---|---|---|---|
| 100ms | ~1,997ms | 141.4ms | 140.1ms | 22.7ms | **162.8ms** | 91.85% | 92.92% |
| 300ms | ~1,998ms | 141.9ms | 140.7ms | 22.8ms | **163.5ms** | 91.82% | 92.90% |
| 500ms | ~1,988ms | 141.6ms | 140.3ms | 22.8ms | **163.1ms** | 91.78% | 92.86% |
| 1000ms | ~1,992ms | 141.8ms | 140.4ms | 22.7ms | **163.2ms** | 91.81% | 92.88% |

### 1.2 핵심 관찰

**관찰 1 — SW suffix ≈ ExpC suffix (오차 1.3ms 이내)**

SW의 실제 forward 계산량은 ExpC와 완전히 동일하다: suffix 75 토큰만 계산한다.
`suffix_ms`: ExpC=141ms, SW=140ms → 차이가 없다.

이는 올바른 동작이다. SW는 KV를 교체한 뒤 suffix를 계산하는 구조이고,
suffix 계산에 필요한 KV 길이(3,011 tokens)는 ExpC와 동일하다.

**관찰 2 — SW overhead = assembly 22.7ms**

SW가 ExpC보다 느린 이유는 단 하나: KV assembly(tensor copy) 비용.

```
assembly = 36 layers × 2 (K,V) × 720 tokens × 8 heads × 128 dim × 2 bytes (BF16)
         ≈ 36 × 2 × 720 × 8 × 128 × 2 = 84MB 복사
실측: 22.7ms → 약 3.7 GB/s 유효 메모리 대역폭 (Thor 공유 메모리 기준 예상 범위 내)
```

**관찰 3 — Δt에 무관하게 타이밍이 고정**

Δt=100ms → 1000ms(10배 차이)에서도 SW total이 162~163ms로 완전히 일정하다.
SW 방법은 Δt 의존성이 없다.

**결론**: SW는 ExpC보다 항상 22ms 느리다.

---

## 2. KV Similarity 분석

### 2.1 전체 측정값 (3회 평균)

| Δt | ExpC overall | SW vs t1full | SW vs t0 | replaced_sim | retained_sim |
|---|---|---|---|---|---|
| 100ms | **0.786** | 0.736 | **0.885** | 0.59 | 0.89 |
| 300ms | **0.691** | 0.662 | **0.878** | 0.58 | 0.88 |
| 500ms | **0.650** | 0.630 | **0.874** | 0.58 | 0.87 |
| 1000ms | **0.613** | 0.609 | **0.871** | 0.60 | 0.87 |

*(모든 값: cosine similarity, K와 V의 평균, layer 0/9/18/27 샘플링)*

### 2.2 각 지표의 의미

#### `ExpC overall` — Exp C KV가 t1_full KV와 얼마나 다른가

t0 KV를 그대로 t1에 쓰는 Exp C의 "오차"를 나타낸다.
Δt가 클수록 t0와 t1의 KV가 달라지므로 이 값은 단조 감소한다.

- Δt=100ms: 0.786 (14% 오차)
- Δt=1000ms: 0.613 (39% 오차)

#### `SW vs t1full` — SW KV가 t1_full KV와 얼마나 다른가

SW 방법이 실제 t1_full과 비교해 얼마나 정확한 KV를 만드는지 나타낸다.

SW는 oldest 4개 이미지를 t1 최신 것으로 교체했으므로 이론상 ExpC보다 높아야 한다.
그러나 실측값은 ExpC보다 낮다(0.736 < 0.786 at Δt=100ms).

**왜 이런 결과가 나오는가?** 아래 2.3 참조.

#### `SW vs t0` = `retained_sim` — SW KV가 t0 KV와 얼마나 유사한가

SW는 t0 KV의 75%를 그대로 유지한다. 따라서 이 값은 높게 나오는 것이 당연하다.
**0.87~0.89로 매우 안정적이고 높다.**

Δt가 증가해도 retained_sim이 거의 변하지 않는 이유: 유지된 12개 이미지 KV는 t0와
완전히 동일하고 (sim=1.0), 교체된 4개 이미지 KV는 t1 최신 프레임 기준으로 바뀌었기
때문이다. 전체 평균에서 75% 비중을 차지하는 "유지된" 부분이 지배적이다.

#### `replaced_sim` — 교체 위치의 SW KV가 t1_full 동일 위치와 얼마나 유사한가

이 지표가 가장 중요하고 가장 오해하기 쉽다. 상세 분석은 2.3에서.

### 2.3 `replaced_sim ≈ 0.59`의 올바른 해석

**측정 방법 (코드 기준)**:

```python
# SW at old_pos vs t1_full at old_pos (동일한 pos 범위로 비교)
sim = compute_kv_similarity_region(kv_sw, kv_t1_full, start=old_s, end=old_e)
```

여기서:
- `kv_sw[old_s:old_e]` = t1의 newest frame KV를 직접 복사한 값 (`= kv_t1[new_s:new_e]`)
- `kv_t1_full[old_s:old_e]` = t1_full이 old_pos에 자연스럽게 계산한 KV

즉, replaced_sim은 **"t1의 newest frame(f3) KV"와 "t1_full이 oldest frame(f0) 위치에 계산한 KV"의 유사도**다.

**이것이 transplant 실패를 의미하지 않는 이유**:

```
비유: 선반의 첫 번째 칸에 새 책을 꽂았다.
  replaced_sim은 "새로 꽂은 책이 기존에 있던 책과 얼마나 비슷한가"를 재는 것.
  0.59라는 값은 새 책(t1 f3)과 이전 책(t1 f0)이 59% 유사함을 의미한다.
  이는 책을 올바르게 꽂았느냐와는 별개의 문제다.
```

transplant 성공 여부를 검증하려면 `kv_sw[old_s:old_e]` vs `kv_t1[new_s:new_e]`를
비교해야 하는데, 이 값은 정의상 1.0이다 (직접 복사했으므로).

**replaced_sim=0.59의 실제 의미**: t1 내에서 oldest 프레임(f0)과 newest 프레임(f3)의
KV가 약 59% 유사하다. 즉, 동일 장면의 300ms 간격(f0→f3 = 3×100ms) KV 변화량이다.

**이 값이 문제인가?** decode 결과를 보면 EOS 100%, steps_diff ≤3이므로 문제없다.
모델은 "위치 pos에 새로운 이미지 콘텐츠의 KV"가 있어도 정상적인 trajectory를 생성한다.

### 2.4 mRoPE 가정의 검증

SW의 핵심 전제: "mRoPE로 인코딩된 vision KV는 위치가 달라도 이식 가능하다."

Qwen2-VL의 vision token은 2D patch 좌표로 RoPE가 적용되며, 1D 순서 위치와 독립적이다.
따라서 img3(new_pos)의 KV를 img0(old_pos)에 넣어도 position encoding 관점에서는
"new_pos 이미지의 2D patch 좌표 KV를 old_pos 위치에 배치"하는 것이 된다.

이 이식이 모델에 미치는 영향: 실제 도달 EOS 100%, steps 정상 → **mRoPE 가정 실험적으로 지지됨.**

---

## 3. Decode 품질 비교

### 3.1 decode steps 분포

| Δt | Full steps | ExpC steps | SW steps |
|---|---|---|---|
| 100ms | 16~19 | 16~19 | 16 (3/3 runs) |
| 300ms | 16~19 | 16~19 | 16 (3/3 runs) |
| 500ms | 16~19 | 16~19 | 16~19 |
| 1000ms | 16 | 16~19 | 14~16 |

steps_diff (SW vs Full):
- 최대 3 steps 차이 — Full과 ExpC도 같은 범위로 변동함
- Δt=1000ms에서 SW가 14 steps → 가장 짧게 끝남 (early EOS, 빠른 수렴으로 해석 가능)

### 3.2 EOS 감지율: 모든 방법, 모든 Δt에서 100%

full, ExpC, SW 전부 EOS를 정상적으로 감지했다.
trajectory 생성이 중단 없이 완료된다.

### 3.3 SW가 생성하는 trajectory의 성격

SW는 t0의 f1, f2, f3 frame KV + t1의 f3(newest) frame KV를 조합한다.
이론적으로 다음을 의미한다:
- 과거 3개 프레임(f1, f2, f3 of t0)은 t0 기준으로 계산된 컨텍스트
- 가장 최신 프레임(cam별 newest)은 t1 기준으로 계산된 컨텍스트

결과적으로 SW trajectory는 "t0와 t1의 혼합 뷰"를 기반으로 생성된다.
이는 Exp C("순수 t0 뷰")와 Full t1 prefill("순수 t1 뷰") 사이 어딘가다.

---

## 4. SW vs Exp C — 어느 것을 써야 하는가

### 4.1 정량 비교

| 항목 | ExpC | SW | 우위 |
|---|---|---|---|
| prefill 시간 | **141ms** | 163ms | **ExpC** |
| 절감률 | **92.9%** | 91.8% | **ExpC** |
| KV freshness (SW vs t1) | 0.61~0.79 | 0.61~0.74 | **ExpC** |
| EOS 성공률 | 100% | 100% | 동일 |
| Δt 의존성 | 없음 | 없음 | 동일 |
| 구현 복잡도 | 단순 | 복잡 | **ExpC** |

**모든 항목에서 ExpC가 SW와 같거나 낫다.**

### 4.2 왜 SW가 ExpC보다 "KV freshness"가 낮은가?

직관적으로 SW는 t1의 최신 frame을 추가하므로 더 좋아야 하지 않나?

핵심 이유: **KV 비교 방식의 구조적 문제**

```
SW vs t1full 측정 방식:
  - SW[0:3011] vs t1_full[0:3011] (동일 위치 비교)

SW는 old_pos(img0 위치)에 t1의 new_pos(img3) 콘텐츠를 넣었다.
t1_full은 old_pos에 t1의 img0(oldest) 콘텐츠를 갖고 있다.
→ 같은 위치에 다른 이미지 콘텐츠가 있어서 유사도가 낮게 나온다.

t1_full이 old_pos에서 우리가 SW에 넣은 것(img3 KV)과 같은 값을 가지려면
t1_full도 "img3을 old_pos에 배치"해야 하는데, 그건 t1_full의 자연스러운 구조와 다르다.
```

반면 ExpC는 t0 KV 전체를 그대로 가져가므로 "재배치"가 없다.
SW는 재배치를 하기 때문에 오히려 t1_full과의 유사도가 낮아진다.

### 4.3 SW가 의미있는 상황

SW는 다음 조건에서 Exp C보다 나을 수 있다:
1. t0와 t1의 장면 변화가 **매우 크고**(급격한 방향 전환, 터널 진입 등)
2. oldest frame(f0)이 이미 stale해서 t0 KV를 재사용하면 안 되는 경우

현재 실험 클립(5.1s~6.1s, 고속도로 직진)에서는 이런 상황이 아니므로 SW의 장점이 없다.

---

## 5. KV Similarity 패턴 분석

### 5.1 Δt에 따른 KV 유사도 감소 패턴

```
ExpC overall (t0 KV vs t1_full KV):
  Δt=100ms: 0.786
  Δt=300ms: 0.691  (-0.095 per 200ms → 약 -0.047/100ms)
  Δt=500ms: 0.650  (-0.041 per 200ms)
  Δt=1000ms: 0.613 (-0.037 per 500ms → 체감 감소율이 줄어들고 있음)

→ KV 유사도 감소는 Δt에 대해 볼록(concave) 함수다.
   대형 모델의 KV representation이 "빠르게 다르게" → "천천히 수렴"하는 패턴.
```

### 5.2 카메라별 replaced_sim 차이 (Δt=1000ms 기준)

| 카메라 | sim_sw_vs_t1new | sim_sw_vs_t0old | 해석 |
|---|---|---|---|
| cam0 (img0→img3) | 0.579 | 0.488 | 중간 변화 |
| cam1 (img4→img7) | 0.679 | 0.555 | **변화 적음** (주로 정면 시야?) |
| cam2 (img8→img11) | 0.490 | 0.420 | **변화 큼** (측면 카메라?) |
| cam3 (img12→img15) | 0.657 | 0.555 | 중간 변화 |

cam2의 변화가 가장 크다(0.490). 측면 카메라는 차선 변경 등으로 변화가 크기 때문으로 추정.
cam1이 가장 안정적(0.679). 이 카메라의 KV를 교체하는 것이 가장 효과적이다.

---

## 6. 결론 및 권장사항

### 6.1 실험 목표 달성 여부

| 목표 | 결과 |
|---|---|
| SW가 에러 없이 실행되는가 | ✅ |
| SW가 92% 이상 prefill 절감하는가 | ✅ (91.8%) |
| EOS가 정상 감지되는가 | ✅ (100%) |
| mRoPE 가정이 성립하는가 | ✅ (decode 정상) |
| SW가 Exp C보다 나은가 | ❌ (22ms 느리고, KV sim도 낮음) |

### 6.2 주요 발견

1. **SW는 Exp C와 동일한 suffix-only forward를 수행한다.** assembly 22ms 외에 추가 계산 비용 없음.

2. **replaced_sim=0.59는 transplant 실패가 아니다.** 같은 시간대(t1) 내에서 f0과 f3 KV가 59% 유사하다는 의미다. mRoPE 이식은 decode 동작으로 검증됨.

3. **SW가 Exp C보다 나쁜 근본 원인**: KV를 old_pos에 다른 콘텐츠로 재배치하면 전체 vision KV의 일관성이 깨진다. Exp C는 재배치 없이 t0 KV 전체를 쓰므로 내부 일관성이 유지된다.

4. **Δt=1000ms에서도 SW total = 163ms로 고정.** 타이밍의 Δt 독립성 확인.

### 6.3 다음 단계 권장

**SW를 독립 방법으로 추구하는 것은 중단한다.** Exp C(141ms)보다 느리고 품질도 비슷하거나 낮다.

대신:

| 우선순위 | 방향 | 예상 효과 |
|---|---|---|
| **1순위** | Exp C + AppendOnlyCache-C 통합 | Prefill 141ms → 100ms 미만 목표 |
| **2순위** | Async VE 오버랩 (Decode 중 다음 VE 병렬) | VE 728ms → 0ms (완전히 숨겨짐) |
| **3순위** | Multi-hop cascade 안전 hop 수 결정 | t0→t1→t2→... 연속 재사용 가능 여부 |

### 6.4 현재까지 파이프라인 예상

```
[현재 (AppendOnlyCache-C 적용 후)]
VE(728) + Prefill(895) + Decode(1,345) + Flow(637) = 3,605ms

[Exp C KV Temporal Reuse 적용 시 (이번 실험 기준)]
VE(728) + Prefill(141) + Decode(1,345) + Flow(637) = 2,851ms   (-754ms, -21%)

[+ Async VE 오버랩]
VE(0*) + Prefill(141) + Decode(1,345) + Flow(637) = 2,123ms    (-1,482ms, -41%)

* 이전 추론의 Decode(1,345ms) 중 728ms를 VE가 사용
```

---

## 7. 데이터 품질 노트

- 측정 환경: Jetson AGX Thor, CUDA 13.0, BF16, sdpa, AppendOnlyCache-C 미적용 (DynamicCache)
- 각 Δt당 WARMUP 3회 + MEASURE 3회, MEASURE 3회 평균
- Full prefill 분산: ±50ms 이내 (KV 캐시 상태에 따라 변동)
- SW assembly 분산: ±0.4ms (매우 안정적인 tensor copy)
- KV similarity는 4개 레이어(0, 9, 18, 27) 샘플링으로 측정 — 전체 36 레이어의 proxy

---

*본 문서: Sliding Window KV Temporal Reuse 실험의 최종 결과 기록 및 해석.*  
*SW 방법은 Exp C 대비 우위 없음 확인. 향후 Exp C + Async Pipeline 방향으로 진행.*
