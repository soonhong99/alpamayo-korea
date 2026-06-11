# Experiment 1 — Decode Skip / Adaptive Decode
**실험 보고 (교수님께)** | 작성: 2026-05-20 | 플랫폼: Jetson AGX Thor

---

## 1. 실험 배경 — 왜 이 실험을 했는가

Alpamayo 1.5의 추론 파이프라인은 네 단계로 구성됩니다.

```
Vision (714ms) → Prefill (1,472ms) → Decode (1,926ms) → Flow Matching (890ms)
                                           ↑
                               이 단계가 정말 필요한가?
```

- **Prefill**: 카메라 입력 + 언어 프롬프트를 VLM이 한 번에 처리하여 hidden state를 만드는 단계
- **Decode**: VLM이 **Chain-of-Causation(CoC)** 추론 텍스트를 토큰 단위로 자동회귀 생성하는 단계
  - 예: *"Keep distance to the lead vehicle since it is directly ahead in our lane"*
  - 현재 16~19 토큰, 토큰당 ~107ms → **1,926ms 소요 (전체의 38%)**
- **Flow Matching(Action Expert)**: VLM 마지막 hidden state를 조건으로 받아 6.4초 주행 궤적(64 waypoints)을 생성

**핵심 질문**: CoC 텍스트를 끝까지 생성하지 않고 일부만 (또는 아예 0개) 생성해도 궤적 품질이 유지되는가?

이것이 사실이라면 Decode 단계를 단축하여 최대 **2.5초(32%)** 의 지연 시간을 절약할 수 있습니다.

---

## 2. 실험 설계

### 제어 변수 (모든 조건 동일)
- 동일한 카메라 입력 (PhysicalAI clip `030c760c`, t=5.1 s)
- 동일한 egomotion 이력
- 동일 시드 (`seed=42`) 적용
- 각 조건 10회 반복 → 평균값 사용

### 조작 변수 (조건별로 변경)
`max_coc_tokens = N` : VLM이 생성할 수 있는 CoC 토큰의 최대 개수

- N번째 토큰 이후 `ForceEarlyEOS` LogitsProcessor가 즉시 EOS(`<|traj_future_start|>`)를 강제 삽입
- N=0이면 Prefill 마지막 hidden state만 Action Expert에 전달

### 품질 판정 기준 (N=16 baseline 궤적 대비)

| 지표 | 정의 | **PASS** | FAIL |
|------|------|----------|------|
| **ADE** | 64개 waypoint 전체 평균 L2 거리 | < 0.2 m | > 0.5 m |
| **FDE** | 마지막 waypoint (t=6.4s) L2 거리 | < 1.0 m | > 3.0 m |

---

## 3. 각 그래프 설명 (X축·Y축)

### Fig 1(a) — Bird's-eye view (부감도)
| 축 | 의미 |
|----|------|
| **X축** (가로) | **전방 거리 (m)** — 차량이 6.4초 동안 얼마나 앞으로 이동할 것으로 예측하는가 |
| **Y축** (세로) | **측방 편차 (m)** — 차로 중앙선 기준 좌(-) / 우(+) 이탈 거리 |
| 각 선 | 조건별 예측 궤적 (64개 waypoint 연결). ★ = 6.4초 후 도착 지점 |
| 파란 점선 | 차로 경계선 (한국 표준 차로폭 3.5m 기준 ±1.75m) |

### Fig 1(b) — Forward distance over time
| 축 | 의미 |
|----|------|
| **X축** (가로) | **시간 (s)** — 현재(0)부터 6.4초 후까지 |
| **Y축** (세로) | **누적 전방 이동 거리 (m)** — 즉, 모델이 예측하는 차량 속도의 적분 |
| 선 기울기 | 가파를수록 빠른 속도를 예측함 |
| N=16 (검정) vs 나머지 | 기울기 차이가 곧 속도 계획의 차이 |

### Fig 1(c) — Lateral position over time
| 축 | 의미 |
|----|------|
| **X축** (가로) | **시간 (s)** |
| **Y축** (세로) | **측방 위치 (m)** — 차량이 시간에 따라 얼마나 우측으로 이동하는가 |

### Fig 2 — N=16 vs N=0 직접 비교
- 좌: **Full CoC** — 14개 토큰을 자연스럽게 완성한 N=16
- 우: **Decode Skip** — 토큰 없이 Prefill hidden state만 사용한 N=0
- 하단 박스: 두 궤적의 6.4초 후 endpoint 차이 수치

### Fig 3 — Per-waypoint error over time
| 축 | 의미 |
|----|------|
| **X축** (가로) | **시간 (s)** — 각 waypoint의 시간 위치 (waypoint 1 = 0.1s, waypoint 64 = 6.4s) |
| **Y축** (세로) | **N=16 baseline 대비 L2 거리 (m)** — 해당 시점에서 baseline 궤적과 얼마나 떨어져 있는가 |
| 점선 (빨강) | ADE pass threshold = 0.2m |
| 점선 (주황) | ADE marginal threshold = 0.5m |
| 점선 (초록) | FDE pass threshold = 1.0m |

### Fig 4 — Latency vs. Quality tradeoff
| 축 | 의미 |
|----|------|
| **X축** (가로) | **CoC 토큰 수 N** (좌 그래프) / **End-to-end 지연 시간 ms** (우 그래프) |
| **Y축** (가로) | **End-to-end 지연 시간 ms** (좌) / **Mean ADE vs baseline (m)** (우) |
| 이론 직선 | `5,122 + 107 × N ms` — NSight로 실측한 107ms/token 기반 |
| 초록 점선 | ADE pass threshold = 0.2m |

---

## 4. 실험 결과 수치표

> **지연 시간**: 실측 공식 `5,122 + 107×N ms` (Jetson AGX Thor, NSight 기반)
> **ADE**: Fig 4(b) 시각적 판독값 (N=0만 실험 로그로 확인됨: 0.907m)
> **N=16 CoC 내용**: "Keep distance to the lead vehicle since it is directly ahead in our lane" (14 tokens)

| 조건 | N (토큰 수) | 지연 시간 (ms) | N=16 대비 절약 (ms) | 절약률 | ADE (m) | FDE (m) | 판정 |
|------|------------|--------------|-------------------|--------|---------|---------|------|
| Decode Skip | **0** | 5,122 | **2,464** | **32.5%** | 0.907 | 3.32 | ❌ FAIL |
| 극소 | 1 | 5,229 | 2,357 | 31.1% | ~1.2 | — | ❌ FAIL |
| 최소 | 3 | 5,443 | 2,143 | 28.3% | ~0.95 | — | ❌ FAIL |
| 단축 | 5 | 5,657 | 1,929 | 25.4% | ~0.78 | — | ❌ FAIL |
| 중간 | 8 | 5,978 | 1,608 | 21.2% | ~1.02 | — | ❌ FAIL |
| 중간 | 10 | 6,192 | 1,394 | 18.4% | ~1.1 | — | ❌ FAIL |
| 중간 | 13 | 6,513 | 1,073 | 14.1% | ~1.35 | ~3.5 | ❌ FAIL |
| **Baseline** | **16** | **7,586** | 0 | 0% | **0** | **0** | ✅ PASS |

> ※ `~` 표기는 Fig 4 그래프 시각적 판독값 (±0.1m 오차 가능). 정확한 값은 Thor의 `summary.json` 참조.

### N=0 vs N=16 endpoint 정밀 비교 (Fig 2에서 직접 확인)

| | N=16 (Full CoC) | N=0 (Decode Skip) | 차이 |
|--|-----------------|-------------------|------|
| **6.4초 후 전방 위치** | 53.2 m | 49.9 m | **Δx = −3.3 m** |
| **6.4초 후 측방 위치** | 0.91 m | 0.68 m | **Δy = −0.23 m** |
| **FDE (endpoint L2)** | — | — | **3.32 m** |
| **ADE (평균 L2)** | — | — | **0.907 m** |

---

## 5. 세 가지 범주로 본 결과 요약

### 범주 A — Full CoC: N=16 (자연 완성) ✅

```
VLM이 CoC 추론을 자연스럽게 끝까지 생성 (14 tokens)
→ 문장 완성: "Keep distance to the lead vehicle since it is directly ahead in our lane"
→ Action Expert가 완전한 reasoning context를 conditioning으로 받음
→ 정상 궤적 생성
```

### 범주 B — Limited CoC: N=1~13 (강제 조기 종료) ❌

```
ForceEarlyEOS가 N번째 토큰 직후 EOS를 강제 삽입
→ 문장이 중간에 잘림: "Keep distance to the lead vehicle since it is..."  (미완성)
→ Action Expert가 추론이 완성되지 않은 hidden state를 받음
→ 분포 밖(out-of-distribution) 궤적 생성 — 모두 FAIL

가장 많이 아낀 조건(N=1, −31%)도 ADE ~1.2m로 pass 기준(0.2m)의 6배 초과
```

### 범주 C — No CoC: N=0 (Prefill hidden state만 사용) ❌

```
Decode 단계 전체를 건너뜀
→ Prefill의 마지막 hidden state가 그대로 Action Expert에 전달
→ 32.5% 지연 시간 절약 (−2.464초)
→ 그러나 ADE = 0.907m (pass 기준의 4.5배)
   FDE = 3.32m (6.4초 후 도착지점이 3.3m 어긋남)
→ FAIL
```

---

## 6. 핵심 결론

> **VLM의 CoC 추론은 중간에 잘라낼 수 없다.**
> 
> Action Expert가 올바른 궤적을 생성하려면 VLM이 추론 문장을 **자연스럽게 완성(EOS 자연 도달)** 해야 한다.
> 강제로 N개 토큰에서 종료하면, 설령 13개를 생성해도 모두 분포 밖 hidden state가 되어 궤적이 무너진다.

### 실험이 답한 질문

| 질문 | 답 |
|------|-----|
| Decode 없이(N=0)도 궤적이 나오는가? | 나오지만 품질 기준 **4.5× 초과 (FAIL)** |
| N=0과 N=16은 그래프상 비슷해 보이는가? | 방향은 비슷(직진), 하지만 **6.4초 후 3.3m 어긋남** |
| 토큰 수를 줄이면 품질이 점진적으로 나빠지는가? | 단조적이지 않음. 모든 조건이 threshold 이상으로 FAIL |
| 몇 개의 토큰이 최소로 필요한가? | **자연 완성(이 클립에서 14 tokens) 전체** — 임의 절단은 모두 FAIL |

### 다음 방향

이번 실험의 결과(시나리오 C)에 따르면 **Decode 단계 자체를 빠르게 만드는 것**이 유일한 경로입니다.

- **Exp 6**: CUDA Graph — 토큰당 오버헤드 107ms → 53ms로 감소
- **Exp 7**: EOS Sync 격리 — `cudaStreamSynchronize` 53ms 제거 검증
- **Adaptive Decode**: 장면 복잡도에 따라 CoC 길이를 동적으로 결정 (장기 방향)

---

*플랫폼: Jetson AGX Thor (JetPack 7, CUDA 13.0, SM 11.0, 128GB shared memory)*
*스크립트: `scripts/exp1_decode_skip.py`, 시각화: `scripts/viz_exp1_trajectories.py`*
