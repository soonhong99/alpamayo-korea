# Alpamayo 1.5 아키텍처 분석

Alpamayo 1.5가 왜 이렇게 설계됐는지, 각 구성 요소의 역할과 선택 이유를 정리한다.

---

## 전체 구조 한눈에

```
입력
 ├─ 멀티카메라 비디오 (front/rear/fisheye)
 ├─ Ego 이동 히스토리 (속도·조향·가속 시계열)
 └─ Navigation 힌트 (목적지 방향)
         │
         ▼
┌─────────────────────────────────────┐
│  Cosmos Reason2 (8.2B)              │  ← VLM backbone
│  = Qwen3-VL 기반 Vision-Language   │
│    + Chain-of-Causation 추론        │
└────────────────┬────────────────────┘
                 │  latent scene embedding (조건)
                 ▼
┌─────────────────────────────────────┐
│  Action Expert (2.3B)               │  ← trajectory generator
│  = Flow Matching 기반               │
│  = Unicycle 액션 공간               │
└────────────────┬────────────────────┘
                 │
                 ▼
출력
 ├─ 6.4초 궤적 (64 waypoints, Δacc + Δcurvature)
 └─ 한국어/영어 추론 텍스트 (Chain-of-Causation)
```

총 파라미터: **11.08B** (8.2B VLM + 2.3B Action Expert)

---

## 왜 두 모델로 분리했는가? (VLM + Action Expert)

### 핵심 문제: 언어 토큰 vs 연속 궤적

VLM(트랜스포머)의 출력은 **이산 토큰**이다. 궤적(waypoint 시계열)은 **연속 실수값**이다. 이 둘을 하나의 모델로 합치면:

| 접근법 | 문제 |
|---|---|
| 궤적을 토큰으로 양자화 | 해상도 손실, 멀티모달 분포 붕괴 (평균값으로 수렴) |
| Regression head 추가 | 단일 최적 궤적만 출력, 불확실성 표현 불가 |
| 통합 diffusion | 추론 속도 크게 저하, 언어 생성과 궤적 생성이 서로 간섭 |

**해결책**: 분리. VLM이 장면을 이해하고 latent embedding을 만들면, Action Expert가 그것을 조건으로 받아 궤적 분포를 샘플링한다.

분리의 이점:
- 각 모듈을 독립적으로 학습·업데이트 가능
- 언어 추론(느린 System 2)과 반사적 제어(빠른 System 1)가 자연스럽게 대응
- Action Expert만 교체해서 다른 차량 플랫폼에 적응 가능

---

## 왜 Flow Matching인가? (Diffusion이 아니라)

### Diffusion의 한계

Diffusion은 DDPM 기준 T=1000 스텝 역방향 샘플링이 필요하다. DDIM 등으로 줄여도 20~50스텝. 100ms 레이턴시 제약이 있는 Edge 추론에서는 너무 느리다.

### Flow Matching이란

Flow Matching(Rectified Flow, Lipman et al. 2022)은 노이즈 → 데이터 경로를 **직선**으로 만든다.

```
Diffusion:  x_T → ... (구불구불) ... → x_0   (많은 스텝 필요)
Flow Match: x_T ────────────────────→ x_0   (1~4 스텝도 가능)
```

ODE: `dx/dt = v_θ(x_t, t, condition)`

여기서 `v_θ`가 Action Expert가 학습하는 속도장(velocity field)이다.

### 궤적 생성에 Flow Matching이 적합한 이유

1. **멀티모달 분포 표현**: 같은 교차로에서 직진/좌회전/우회전 세 가지 궤적 분포를 동시에 표현 가능 (Regression head는 평균 하나만 출력)
2. **빠른 샘플링**: 4~8 function evaluation만으로 고품질 궤적 생성
3. **조건부 생성**: VLM latent를 조건으로 넣어 장면에 맞는 궤적 샘플링
4. **불확실성 정량화**: 여러 번 샘플링해 궤적 앙상블 → 불확실한 상황 감지

---

## 왜 Unicycle 액션 공간인가?

Alpamayo의 Action Expert 출력은 x/y 절대 좌표가 아니라:

```
(Δacceleration, Δcurvature) × 64 timesteps
```

### 다른 표현 방식과 비교

| 표현 | 예시 | 문제 |
|---|---|---|
| x/y 절대 좌표 | (1.2m, 0.4m), (2.5m, 0.7m)... | 차량 헤딩이 틀리면 모든 값 틀림 |
| x/y 상대 변위 | Δx, Δy per step | 측면 미끄러짐 없는 차량에 부자연스러움 |
| 속도 + 조향각 | v, δ | 물리적이나 곡률로 변환 필요 |
| **가속도 + 곡률 (Unicycle)** | Δa, Δκ | 차량 운동학과 직접 대응, 헤딩 독립적 |

**Unicycle 모델**: 차량을 단순화해 앞바퀴 방향만으로 선회하는 2D 모델.
- `v(t+1) = v(t) + a(t)·Δt`
- `κ = 1/R` (선회 반경의 역수)
- `heading(t+1) = heading(t) + v(t)·κ(t)·Δt`

이 표현의 장점:
- **헤딩 독립적**: 어느 방향을 보고 있어도 같은 모델
- **물리 제약 내재화**: 차량이 갑자기 옆으로 이동하는 물리적으로 불가능한 궤적 생성 방지
- **적분 가능**: 액션 시퀀스를 적분하면 바로 x/y 궤적

---

## 왜 Ego 이동 히스토리를 별도 입력으로 주는가?

비디오만으로는 자차 속도/가속도를 정확히 파악하기 어렵다. 광류(optical flow)에서 속도를 추정할 수 있지만:
- 이미지 해상도 제약으로 저속에서 노이즈가 큼
- VLM이 이미 이미지 이해에 모든 용량을 써야 함
- 정확한 ego state는 IMU/wheel encoder에서 오는 값이 훨씬 정확

따라서 **ego state(속도, 가속, 조향 히스토리 2초)를 별도 토큰**으로 VLM에 제공한다. 이렇게 하면:
- VLM이 "현재 60km/h로 직진 중"임을 정확히 알고 추론
- Action Expert도 현재 물리 상태를 conditioned input으로 받아 물리적으로 실현 가능한 궤적 생성

---

## Chain-of-Causation (CoC)이란

일반 언어 모델의 Chain-of-Thought과 유사하지만, 자율주행에 특화된 구조:

```
[인식] "전방 30m에 보행자가 도로를 횡단 중"
    ↓
[예측] "보행자가 현재 속도로 계속 이동하면 2.1초 후 차도 중앙 도달"
    ↓
[계획] "충돌 회피를 위해 감속 필요. 현재 50km/h → 20km/h 목표"
    ↓
[행동] "Δa = -2.1 m/s² 적용, 직진 유지"
```

이 텍스트 추론은 Action Expert가 궤적을 생성하는 데 **조건**으로 사용된다. 단순히 설명용이 아니라 실제 궤적 결정에 영향을 미친다.

한국어 추론 시 `--lang ko` 플래그를 전달하면 CoC가 한국어로 출력된다.

---

## 파일 구조와 역할

```
alpamayo1_5/
├── models/
│   ├── alpamayo1_5.py          ← 최상위 모델 클래스 (Alpamayo1_5)
│   ├── cosmos_reason2.py       ← VLM backbone (Qwen3-VL 기반)
│   └── action_expert.py        ← Flow Matching 궤적 생성기
├── tokenizers/
│   └── ego_tokenizer.py        ← Ego state → 토큰 변환
├── inference/
│   └── pipeline.py             ← 추론 파이프라인 (카메라→궤적)
└── utils/
    ├── trajectory.py           ← Unicycle 적분, waypoint 변환
    └── visualization.py        ← BEV 시각화
```

---

## 학습 파이프라인 (Fine-tuning 관점)

```
[사전학습된 Cosmos Reason2] + [사전학습된 Action Expert]
            │
            ▼
    [RL Post-training]
    보상 신호:
    - AlpaSim 충돌률 (주 신호)
    - 인간 전문가 궤적과의 거리 (보조)
    - 교통법규 위반 패널티
    - 승차감 (저크 최소화)
            │
            ▼
    [Alpamayo-Korea]
    한국 시나리오에 최적화된 가중치
```

RL에서 VLM과 Action Expert를 함께 학습한다. VLM의 추론이 더 좋아지면 Action Expert의 조건이 개선되고, Action Expert의 물리적 실현성이 높아지면 RL 보상이 커진다 — 두 모듈이 공동 진화한다.

---

## 한국 도로 환경에서 예상되는 구조적 취약점

| 취약점 | 원인 | 대응 |
|---|---|---|
| 수평 신호등 미인식 | 학습 데이터 대부분이 수직 신호등 | 시나리오 오버샘플링 + 보상 재설계 |
| 버스 전용차로 위반 | 버스 전용차로 규칙이 없는 국가 데이터 | 교통법규 패널티 추가 |
| 좁은 골목길 | 북미/유럽 도로폭 기준 학습 | 좁은 통로 시나리오 추가 |
| 이륜차 끼어들기 | 오토바이 밀도가 낮은 환경 학습 | AI Hub 이륜차 데이터셋 활용 |
| 유턴 구간 | 규칙이 국가마다 다름 | ETRI 데이터 + 시나리오 커버리지 확보 |

---

## 브레인스토밍: 미해결 질문들

1. **CoC 언어가 궤적에 얼마나 영향을 미치는가?**
   - 한국어 vs 영어 추론이 실제 궤적 출력에 차이를 만드는가?
   - 언어 자체가 아닌 토큰 패턴이 중요한 것 아닐까?

2. **SM 11.0 최적화 부재 → 실제 성능 갭은?**
   - flash_api.cpp 패치로 SD PA 대신 Flash Attention 사용 가능하게 됐지만
   - SM 11.0용 커널이 없어 일반 CUDA 코드 경로로 실행 → 진짜 Flash Attn 대비 얼마나 느린가?

3. **Action Expert의 샘플링 스텝 수 vs 레이턴시 트레이드오프**
   - 현재 몇 스텝인지 확인 필요 (`action_expert.py`의 `num_inference_steps`)
   - Thor에서 100ms 목표 달성하려면 최대 몇 스텝까지 가능한가?

4. **AlpaSim의 NuRec 포맷과 AI Hub 포맷 정렬**
   - AI Hub는 COCO-style JSON, AlpaSim은 NuRec scene format
   - 변환 스크립트가 핵심 병목 — 어떻게 설계할 것인가?
