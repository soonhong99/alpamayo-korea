# Alpamayo 1.5 모델 구조 및 논문 분석 — 주간 보고서

**작성일**: 2026-04-24  
**담당**: nanayah9911@gmail.com

---

## ▶ 즉시 참조

### 핵심 결론표

| 주제 | 결론 | 근거 |
| --- | --- | --- |
| Alpamayo 등장 배경 | 기존 자율주행 AI가 Long Tail 상황에서 실패했기 때문 | 기존 모델은 패턴 모방 방식으로 희귀 상황 대처 불가 |
| Alpamayo의 위치 | 차량 탑재용 현장 모델이 아니라 Teacher Model | NVIDIA 공식 발표: Fine-tuning·Distillation을 통한 경량 모델 생성용 |
| Alpamayo 모델 크기 | 10B 파라미터 (약 22GB) | 기존 자율주행 파이프라인 전체(~1B 미만) 대비 10배 이상 |
| Thor 보드 동작 확인 | Jetson AGX Thor(aarch64)에서 CUDA 추론 성공 | 소스 빌드 + 11종 패치 후 end-to-end 추론 완료 |

### 결론 한 줄

> 이번 주의 핵심은 Alpamayo가 왜 등장했는지, 왜 현장에 바로 쓸 수 없는지, 그리고 그 간극을 메우기 위한 양자화의 원리와 한계까지 순서대로 이해한 데 있다. 동시에 Jetson AGX Thor(aarch64) 실기기에서 Alpamayo 1.5를 CUDA로 구동하는 데 성공하여, 이후 한국어 추론 및 경량화 실험의 기반을 마련했다. 다음 주에는 토큰화 단계부터 VLM 백본 처리 흐름을 이어서 분석하고, Thor 보드에서 추론 지연 시간 측정 및 양자화 적용 가능성을 구체화할 예정이다.

- 알파마요 전체 구조 (입력 단계까지 학습 완료, 참고용)

    ![image.png](attachment:f00c66ca-843a-42c2-9614-98cb955071de:image.png)

---

## 0. Thor 보드 동작 확인 (2026-04-19~20 완료)

이번 주 학습과 병행하여, Jetson AGX Thor(aarch64) 실기기에서 Alpamayo 1.5가 실제로 동작하는지 검증하는 작업을 수행했다.

### 0.1 배경 및 난관

Jetson AGX Thor는 NVIDIA의 최신 엣지 AI 플랫폼(CUDA 13.0, SM 11.0 Blackwell, 128GB 통합 메모리)이다. Alpamayo 1.5(10B 파라미터)를 엣지에서 실시간 구동하기 위한 유일한 현실적 선택지이지만, 다음 세 가지 문제가 동시에 발생했다.

| 문제 | 원인 |
|---|---|
| PyTorch aarch64 + CUDA 13.0 공식 wheel 없음 | SM 11.0은 너무 신형이라 PyPI 배포 미지원 |
| CUDA 13.0의 CCCL 통합으로 기존 CUB API 대거 제거 | `TransformInputIterator`, `CountingInputIterator` 등 PyTorch 2.8.0 소스에서 직접 사용하는 API가 삭제됨 |
| FlashAttention SM 11.0 지원 누락 | `flash_api.cpp`가 SM 8.x/9.x/10.x/12.x만 체크하고 SM 11.x(Thor) 분기가 없었음 |

### 0.2 해결: 소스 빌드 + 11종 패치

PyTorch 2.8.0 소스를 클론하여 CUDA 13.0 호환성 패치를 직접 적용한 뒤 빌드했다 (빌드 시간 약 6시간, MAX_JOBS=8).

**주요 패치 내용**

| # | 대상 파일 | 변경 내용 |
|---|---|---|
| 1 | `cmake/Modules/FindCUB.cmake` | CUDA 13.0 sbsa 경로 추가 |
| 2 | `CuFFTUtils.h` | CUDA 13.0에서 삭제된 에러코드 3개 제거 |
| 3~10 | `cub.cuh`, `cub.cu`, `EmbeddingBag.cu` 등 7개 파일 | `cub::TransformInputIterator` → `thrust::make_transform_iterator`, `cub::Sum{}` → `thrust::plus<T>()` 등 CCCL 제거 API 전면 교체 |
| 11 | `flash_api.cpp` | `bool is_sm11x = dprops->major == 11` 분기 10곳 추가 |

**패치 핵심 예시 — CUB → Thrust 대체**

```cpp
// 패치 전 (CUDA 13.0에서 컴파일 오류)
cub::TransformInputIterator<int, NonZeroOp<T>, const T*> itr(ptr, op);

// 패치 후
auto itr = thrust::make_transform_iterator(ptr, NonZeroOp<T>());
```

**패치 핵심 예시 — FlashAttention SM 11.0 분기 추가**

```cpp
// 패치 전: SM 11.x 없음 → Thor에서 런타임 에러
bool is_sm10x = dprops->major == 10 && dprops->minor >= 0;

// 패치 후
bool is_sm10x = dprops->major == 10 && dprops->minor >= 0;
bool is_sm11x = dprops->major == 11 && dprops->minor >= 0;  // ← 추가
// 이후 is_sm11x를 모든 GPU 지원 체크 조건에 포함
```

### 0.3 동작 확인 결과

```
PyTorch version: 2.8.0a0+gitba56102
CUDA available:  True
CUDA device:     NVIDIA Thor
파라미터 수:      11.078526194 B
```

end-to-end 추론 실행 (`python -m alpamayo1_5.test_inference`):

```
Chain-of-Causation (per trajectory):
[['Nudge to the left to clear the construction equipment blocking the right side of our lane']]

minADE: 1.0375674 meters
WARNING: minADE (1.04m) is above 1.0m. Model sampling can be stochastic.
```

- **Chain-of-Causation**: 카메라 영상에서 공사 장비를 인식하고 "좌측으로 이동"이라는 행동의 이유를 언어로 출력 — Alpamayo의 핵심 기능인 인과 추론이 Thor에서 정상 동작함을 확인
- **minADE 1.04m**: `num_traj_samples=1`(단일 샘플링) 기준. 모델이 확률적이므로 다중 샘플링 시 낮아짐
- **추론 지연 시간**: 이번 실행에서는 명시적 타이밍을 측정하지 않았음. 공식 목표치는 ≤100ms이며, 다음 주 FP4 모드(`--dtype fp4`) 활성화 후 latency 프로파일링 예정

---

## 1. 이번 주 핵심 정리

이번 주에는 Alpamayo 1.5의 전체 구조를 이해하기 위한 배경 지식을 단계적으로 쌓았다. 핵심 확인 사항은 아래와 같다.

1. **Alpamayo는 Long Tail 문제를 해결하기 위해 등장했다.** 기존 모델은 자주 발생하는 상황은 잘 처리하지만, 드물고 예외적인 상황에서 실패한다.
2. **Alpamayo는 VLA(Vision-Language-Action) 모델이다.** 보고, 언어로 상황을 해석하고, 추론하고, 행동을 결정하는 네 단계를 하나의 신경망에서 처리한다.
3. **Alpamayo는 현장 배포용 모델이 아니라 Teacher Model이다.** NVIDIA는 이를 Fine-tuning과 Distillation의 출발점으로 사용할 것을 권장한다.
4. **입력 3종(카메라, 에고모션, 텍스트)은 각자 별도의 경로로 토큰화된다.** 통합 멀티모달 토큰 시퀀스에서 처음 결합된다.

이 4가지는 이후 토큰화, VLM 백본 분석, Thor 보드에서의 양자화 실험 설계 시 공통 전제로 작동한다.

---

## 2. Alpamayo 등장 배경과 위치

### 2.1 Long Tail 문제

기존 자율주행 AI의 핵심 한계는 **Long Tail** 상황에서의 실패였다. Long Tail이란 발생 빈도는 낮지만 종류가 매우 많고, 실제 사고의 핵심이 되는 예외 상황들의 집합이다.

- 갑자기 차도로 뛰어드는 어린이
- 역주행하는 차량
- 공사 구간의 예외적 교통 통제

기존 모델은 **"많은 데이터로 패턴 맞추기"** 방식이었기 때문에, 훈련 데이터에 드물게 등장하는 상황에서는 무엇을 해야 할지 알지 못했다.

Alpamayo는 이 문제를 **"상황을 이해하고, 인과관계를 따져서 판단하기"** 방식으로 접근한다.

### 2.2 Teacher Model로서의 위치

| 구분 | 내용 |
| --- | --- |
| 모델 크기 | 10B 파라미터, 약 22GB |
| 현장 배포 여부 | 직접 배포 불가 (너무 무거움) |
| NVIDIA 권장 사용법 | Teacher Model로 활용 → Fine-tuning·Distillation → 경량 학생 모델 생성 |
| 비교 대상 | 기존 자율주행 파이프라인 전체 합산 약 1B 미만 |

NVIDIA 공식 발표에 따르면, Alpamayo는 차량에서 직접 구동되는 것이 아니라 개발자들이 파인튜닝하고 증류해 자체 AV 스택의 백본으로 만들도록 설계됐다.

https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development

---

## 3. 양자화(Quantization): 개념과 기법

### 3.1 배경

모델의 학습은 FP32나 FP16으로 수행된다. 역전파(backpropagation)로 가중치를 미세 조정하려면 높은 정밀도가 필요하기 때문이다. 그러나 배포 시에는 학습이 완료됐으므로, 가중치를 더 적은 비트로 압축할 수 있다. 이것이 양자화다.

**양자화의 목적**: 메모리 절감 + 연산 속도 향상

### 3.2 양자화 5단계 과정 (FP32 → INT8 예시)

| 단계 | 내용 |
| --- | --- |
| 1. 범위 파악 | 가중치의 실제 분포(min, max) 측정 |
| 2. 스케일 팩터 계산 | `scale = max / 127` (비대칭 양자화 기준) |
| 3. INT8 변환 | `INT8값 = round(원래값 / scale)` |
| 4. 정수 연산 | 추론 시 INT8 × INT8 정수 연산만 수행 |
| 5. 역스케일 | 레이어 끝에서 scale 곱해 FP 범위로 복원 |

10B 파라미터 기준 예시:

- FP32 원본: 약 40GB
- INT8 양자화 후: 약 10GB + 스케일 팩터 약 100MB

### 3.3 오차 누적 해결 기법

| 기법 | 개념 | 특징 |
| --- | --- | --- |
| Calibration | 실제 데이터를 흘려 레이어별 실제 분포에 맞게 스케일 결정 | PTQ, 가장 단순 |
| QAT (Quantization-Aware Training) | 학습 중 INT8 오차를 역전파에 반영 | PTQ보다 정밀도 손실 적음, 추가 학습 비용 필요 |
| Mixed Precision | 민감한 레이어는 높은 정밀도, 둔감한 레이어는 낮은 정밀도 | 레이어별 오차 민감도 측정 필요 |
| GPTQ | 레이어 내 다른 가중치가 양자화 오차를 즉시 보상 | LLM처럼 레이어 깊은 모델에 적합 |
| AWQ | 활성화값에 큰 영향을 주는 Salient weight만 선별 보호 | LLM의 Outlier(1%) 문제 해결에 효과적 |
| KV Cache 분리 | 가중치는 INT4, KV Cache는 FP16 유지 | CoT 모델의 추론 체인 오차 누적 억제 |

---

## 4. Alpamayo-R1 논문 핵심 요소 분석

### 4.1 논문 제목의 의미

**"Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail"**

- **Bridging**: Vision(입력)과 Action(출력) 사이에 Reasoning(인과 추론)이라는 가교를 추가
- **Long Tail**: 희귀하고 예외적인 주행 상황

기존 자율주행 모델이 이미지를 보고 바로 핸들을 꺾는 반사 신경 모델이었다면, AR1은 중간에 추론 과정을 집어넣어 Vision과 Action을 연결한다.

### 4.2 Language와 Reasoning의 차이

| 구분 | 설명 |
| --- | --- |
| Language (기존 VLA) | 자유 형식의 자연어로 상황 묘사. 막연한 표현("주의해야 한다"), 피상적 요인("날씨가 맑다"), 인과적 혼란 발생 |
| Reasoning (AR1 CoC) | 관찰 가능한 증거만 기반으로 인과관계를 명확히 구축. 구체적 주행 결정과 직접 연결 |

### 4.3 AR1의 3대 핵심 요소

**① CoC(Chain of Causation) 데이터셋**

자동 라벨링과 인간 검수를 결합하여 구축한 300만 개의 인과 추론 트레이스. 막연한 설명이 아닌 관찰된 원인과 구체적 주행 결정을 논리적으로 연결한다.

**② 모듈형 VLA 구조**

전체 시스템을 레고 블록처럼 기능별 모듈로 분리했다. 핵심 이유는 특정 언어 모델에 영구적으로 종속되지 않기 위함이다. 더 뛰어난 모델이 등장하면 해당 모듈만 교체하면 된다.

또한 궤적 생성을 VLM이 직접 텍스트처럼 수행하지 않고 **Action Expert 모듈**이 전담한다. 이 모듈은 단순 좌표(x, y)가 아닌 가속도와 곡률 기반의 유니사이클 동역학으로 연속 궤적을 병렬 디코딩하여 실시간 처리가 가능하다.

**③ 멀티 스테이지 학습**

| 단계 | 내용 | 목적 |
| --- | --- | --- |
| Stage 1: Action Modality Injection | 이산 궤적 토큰 추가 + 디퓨전 전문가 추가 | VLM에 물리적 행동 출력 능력 주입 |
| Stage 2: SFT | CoC 데이터셋 3M으로 지도 미세조정 | 인과적 추론 능력 습득 |
| Stage 3: RL 사후 학습 (GRPO) | 대형 추론 모델이 평가자로 보상 신호 제공 | 추론 품질 +45%, 추론-행동 일관성 +37% 향상 |

SFT만으로는 말로는 "정지하겠다" 해놓고 실제 궤적은 가속하는 모순이 발생할 수 있다. RL은 텍스트 추론과 물리적 궤적의 일치를 명시적으로 강제한다.

---

## 5. Alpamayo 1.5 입력 구조 분석

### 5.1 입력 3종 개요

Alpamayo 1.5의 입력은 멀티카메라 영상, 에고모션 히스토리, 텍스트 명령으로 구성된다. 세 입력은 각자 별도의 토큰화 경로를 거친 뒤, 통합 멀티모달 토큰 시퀀스에서 처음 결합된다.

```
멀티카메라 영상  → 비전 인코더(ViT)        → 이미지 토큰     ┐
에고모션 히스토리 → TrajectoryFusionMixin   → 에고모션 토큰  ├→ 통합 시퀀스 → VLM
텍스트 명령      → 텍스트 토크나이저        → 언어 토큰      ┘
```

### 5.2 멀티카메라 영상

| 항목 | 사양 | 이유 |
| --- | --- | --- |
| 카메라 수 | 4개 (front-wide, front-tele, cross-left, cross-right) | 전방 원거리, 전방 근거리, 교차로 양측 동시 감지 |
| 프레임 | 카메라당 4 frame @ 10Hz (0.4초) | 물체의 속도·가속도·방향 변화 추정에 충분한 히스토리 |
| 해상도 | 1080×1920px → 320×576px 다운샘플 | 주행 결정에 필요한 정보 보존하면서 토큰 수 억제 |

### 5.3 에고모션 히스토리

| 항목 | 내용 |
| --- | --- |
| 표현 방식 | 가속도(a) + 곡률(κ) — 유니사이클 동역학 기반 |
| 원시 좌표(x, y) 미사용 이유 | 센서 노이즈에 취약, 물리적 연관성 없어 예측값이 불안정해짐, 모델 수렴 저하 |

에고모션의 이산 인덱스 변환은 VLM이 처리하는 최소 단위인 "정수 인덱스 → 임베딩 벡터" 형태를 맞추기 위한 것이다. VLM의 어텐션 레이어 입장에서 이 벡터는 텍스트 토큰의 벡터와 구조적으로 동일하므로, 에고모션 데이터가 텍스트처럼 처리된다.

### 5.4 텍스트 명령

카메라와 에고모션만으로는 모델이 **무엇을 해야 하는지의 의도**를 알 수 없다. 동일한 교차로 상황에서도 목적지에 따라 직진과 우회전이 완전히 다른 행동이기 때문이다.

| 종류 | 예시 |
| --- | --- |
| 내비게이션 가이던스 (1.5 신규) | "400m 앞에서 우회전" |
| 사용자 명령 | "저기 갓길에 세워줘", "천천히 움직여줘" |

---

## 6. 다음 주 계획

1. 남은 Alpamayo 구조 알아보기 (VLM 백본 처리 흐름, Action Expert 샘플링 스텝 수 측정)
2. Thor 보드 추론 지연 시간 프로파일링 (`--dtype fp4` 모드, `torch.cuda.synchronize()` 기반 측정)
3. 한국어 추론 검증 (`--lang ko` 플래그, CoC 한국어 출력 확인)
4. Alpamayo 파일 구조 파악하기 (소스 코드 레벨 분석)
5. 경량화 가능성 분석 (INT8/FP4 양자화 적용 검토)
