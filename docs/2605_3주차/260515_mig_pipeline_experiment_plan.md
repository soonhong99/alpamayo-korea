# Alpamayo 1.5 병렬화 실험 계획서
## MIG 분할 / 크로스프레임 파이프라인 / GPU 용량 스케일링

**작성일**: 2026-05-15  
**연구자**: Alpamayo-Korea 팀  
**플랫폼**: Jetson AGX Thor (Blackwell SM 11.0, LPDDR5X 273 GB/s, 128 GB unified)  
**모델**: Alpamayo 1.5 (Vision 1.15 GB + LM 16.44 GB + Action Expert 4.56 GB = 22.16 GB)

---

## 0. TL;DR — 먼저 읽을 것

> **원래 가설**: VLM에 MIG 인스턴스를 많이 배정하면 Vision/LM/Expert가 병렬로 빨라진다  
> **실제 정답**: **틀렸다.** 단일 프레임 안에서 세 모듈은 데이터 의존성으로 순차 실행됨.  
> MIG 분할은 단일 프레임 레이턴시를 줄이지 않는다.
>
> **진짜 기회**: **크로스프레임 파이프라이닝** — Frame N+1의 Vision 인코딩을  
> Frame N의 Decode와 겹쳐 실행하면 2~2.5× 처리량 향상 가능.  
> 이것이 이 계획서의 핵심 실험이다.

---

## 1. 연구 배경 및 동기

### 1-1. 현재 추론 파이프라인과 병목

```
[단일 프레임, 순차 실행]

Vision Enc.   642 ms  ─┐
LM Prefill   1369 ms   ├── 데이터 의존: 이전 단계 완료 후 시작
Decode       2013 ms   │
Flow          858 ms  ─┘
────────────────────────
Total        4882 ms (0.20 FPS)
```

현재 Alpamayo 1.5는 단일 CUDA 스트림에서 네 단계를 **완전 순차** 실행한다.
GPU는 어느 순간에도 한 모듈만 사용하며, 나머지는 idle 상태다.

### 1-2. 연구 질문

본 계획서는 다음 세 가지 질문을 검증한다:

```
Q1. Thor Blackwell에서 MIG로 모듈별 GPU 슬라이스를 배정하면
    단일 프레임 레이턴시가 줄어드는가?

Q2. 크로스프레임 파이프라이닝(Frame N+1 Vision ∥ Frame N Decode)이
    실효 처리량(FPS)을 높이는가?

Q3. GPU 가용 메모리/컴퓨트를 제한했을 때 성능이 선형적으로 감소하는가?
```

---

## 2. 선행 연구 검토

### 2-1. 직접 관련 논문

#### ActionFlow (arXiv:2512.20276, 2024) ★★★ 가장 중요
> "A Pipelined Action Acceleration for Vision Language Models on Edge"

**핵심 기여**: VLA 추론에서 크로스프레임 파이프라이닝을 최초로 구현.  
LM Decode(메모리 BW-bound)와 Frame N+1 Vision Prefill(compute-bound)의  
complementary 자원 프로파일을 이용해 **겹침 실행** 달성.

- Unified KV Ring Buffer: 프레임 간 KV state 공유
- 결과: OpenVLA-7B에서 **2.55× FPS** 향상, 레이턴시는 변경 없음
- 하드웨어: 엣지 GPU (Jetson-class 포함)

**우리와의 차이**: ActionFlow는 OpenVLA-7B 대상. Alpamayo 1.5는 Vision + LM + Flow  
3단계 구조 → Unified KV Ring Buffer 설계가 달라져야 함.

---

#### GR00T N1 (arXiv:2503.14734, NVIDIA 2025) ★★
> "An Open Foundation Model for Generalist Humanoid Robots"

**핵심 기여**: VLM backbone + Action Expert를 **병렬 토큰 디코딩**으로 분리.  
V-AEFusion 전략으로 LM 출력이 나오는 즉시 Action Expert를 시작.

- 결과: 7-DoF arm에서 **~2.5× 레이턴시 감소**
- Alpamayo 1.5와 유사한 "LM → Flow" 구조

**우리에게 적용 가능한 아이디어**: Decode 마지막 token이 생성되는 즉시  
Flow(Action Expert)를 시작하는 **non-blocking 연결** 구현.

---

#### DistServe / Splitwise / Sarathi-Serve (OSDI/ISCA 2024)

| 논문 | 핵심 아이디어 | 우리 적용 가능성 |
|---|---|---|
| DistServe (arXiv:2401.09670) | Prefill과 Decode를 별도 GPU에 분리 | 단일 보드라 적용 불가, 단 개념 참고 |
| Splitwise (ISCA 2024) | 이종 GPU로 단계 분리 (H100 prefill, A100 decode) | 동일 |
| Sarathi-Serve (arXiv:2403.02310) | Chunked prefill + decode 인터리빙 | **MIG 슬라이스로 모방 가능** |

---

#### MuxServe (arXiv:2404.02015, ICML 2024)

**핵심 기여**: 여러 LLM을 MIG + 시간 다중화로 동시 서빙.  
단일 모델 단일 요청 레이턴시는 개선 없지만, **멀티 태스크 처리량 1.8×**.

→ Alpamayo 1.5를 여러 시나리오에서 동시 추론할 때 적용 가능.

---

#### Hardware Compute Partitioning (ECRTS 2025, Bakita et al.)

**핵심 기여**: CUDA 12의 Green Contexts (GC)로 SM 레벨 격리.  
MIG보다 오버헤드 낮음, 메모리 격리는 없음. Jetson Orin Nano에서 검증.

→ MIG 대안으로 Green Contexts 실험 설계 시 참고.

---

### 2-2. 가장 중요한 선행 연구 공백

**Alpamayo-류 3단계 VLA(Vision + LM + Flow)에서 MIG 슬라이스 배정 최적화**를  
다룬 논문은 현재 없다. ActionFlow는 2단계(Vision+LM만, Flow 없음),  
GR00T N1은 병렬화 상세 구현을 공개하지 않음.

→ **이 실험의 novelty**: Thor Blackwell MIG + 3단계 VLA + 크로스프레임 파이프라인

---

## 3. 핵심 가설 (수정된 버전)

### 가설 H1 — MIG 단일 프레임 가속 ✗ (반증 예상)

```
"Vision, LM, Action Expert에 별도 MIG 슬라이스를 배정하면
 단일 프레임 엔드투엔드 레이턴시가 줄어든다"
```

**왜 기각 예상하는가**:
- 세 모듈은 데이터 의존성으로 반드시 순차 실행 (Vision output → LM input → Flow input)
- MIG는 공간적 격리이지, 데이터 의존성을 우회하지 않음
- 각 MIG 슬라이스가 전체 GPU보다 작은 자원을 받으므로 단계별로 오히려 느려짐

**검증 가치**: 직관과 달리 MIG가 왜 도움이 안 되는지 실증적으로 보여주는 것  
자체가 논문 contribution (부정적 결과도 기여).

---

### 가설 H2 — 크로스프레임 파이프라인 ✓ (부분 확인 예상)

```
"연속 프레임 스트림에서 Frame N+1 Vision을
 Frame N Decode와 겹쳐 실행하면 실효 FPS가 향상된다"
```

**근거**: ActionFlow가 OpenVLA-7B에서 2.55× 달성.  
우리 타임라인에서 이론적 가속:

```
순차:    642 + 1369 + 2013 + 858 = 4882 ms/frame → 0.20 FPS

파이프라인 (Vision을 Decode와 겹침):
Stage 1: Vision(642) + Prefill(1369) = 2011 ms
Stage 2: Decode(2013) ∥ Vision_next(642) → max(2013, 642) = 2013 ms
Stage 3: Flow(858) ∥ Prefill_next(1369) → max(858, 1369) = 1369 ms
실효 처리량: 프레임당 max(2011, 2013, 1369) ≈ 2013 ms → **0.50 FPS (2.4×)**
```

**제약 조건**:
- KV cache 관리: 프레임 간 KV state 공유 또는 분리 필요
- GPU 메모리 동시 사용: Vision(1.15 GB) + LM 활성화 중첩 시 메모리 여유 확인
- Alpamayo 1.5 모델 코드 수정 필요 (CUDA 스트림 분리)

---

### 가설 H3 — GPU 용량 성능 비선형 감소 ✓ (확인 예상)

```
"GPU 가용 메모리를 선형 감소시켜도 성능은 선형으로 감소하지 않는다.
 특정 임계치 이하에서 급격한 성능 하락(cliff)이 발생한다"
```

**이론적 근거**:
- Decode는 BW-bound, 용량이 아니라 대역폭이 병목
- 가중치(16.44 GB)가 GPU 메모리에 완전히 올라가 있으면 용량 감소는 성능 무관
- 가중치 일부가 페이징되는 순간 비선형 하락

---

## 4. 실험 설계

### [EXP-0] MIG 지원 확인 및 환경 구성 (사전 실험)

**목적**: Thor Blackwell iGPU에서 MIG 활성화 가능 여부 실증 확인

```bash
# MIG 지원 확인
sudo nvidia-smi --query-gpu=mig.mode.current --format=csv
sudo nvidia-smi mig -lgip   # GPU Instance Profile 목록

# MIG 활성화
sudo nvidia-smi -mig 1

# 사용 가능한 GI (GPU Instance) 구성 확인
sudo nvidia-smi mig -lgip
sudo nvidia-smi mig -lgi
```

**예상 결과**: Thor Blackwell iGPU가 MIG를 지원한다면  
최대 7개 인스턴스 (1g.Xgb 단위)로 분할 가능.

**실패 시**: MIG 미지원 → CUDA Green Contexts (CUDA 12.4+) 또는  
CUDA MPS로 대체 실험.

---

### [EXP-1] MIG 슬라이스별 단일 모듈 성능 기준선

**목적**: 각 모듈이 GPU 자원의 몇 %를 받을 때 현재 성능을 유지하는가?

**방법**:
1. MIG로 다양한 크기의 단일 인스턴스 생성
2. 각 인스턴스에서 Vision만 / LM만 / Flow만 단독 실행
3. 타이밍 측정 (CUDA Events)

```
실험 조건:
  GI_7g (전체 GPU, baseline):   Vision=642ms, LM=1369ms, Flow=858ms
  GI_4g (57% 자원):             각 측정
  GI_2g (28% 자원):             각 측정
  GI_1g (14% 자원):             각 측정
```

**측정 지표**:
- 각 MIG 크기 × 각 모듈 = GPU time (CUDA Events)
- Roofline 이동: 컴퓨트 제한 시 compute 비례, BW 제한 시 BW 비례

**예상 결과**:
- Vision (compute-bound): 자원 크기에 비례해 슬로우다운
- LM Prefill (compute-bound): 동일
- LM Decode (BW-bound): BW 비례. 하지만 MIG는 메모리 BW도 비례 할당  
  → Decode도 MIG 크기에 비례해 느려짐
- Flow (compute-bound): 자원 비례

**핵심 발견 예상**: 모든 모듈이 MIG 크기에 반비례해 느려짐  
→ H1 기각의 정량적 증거

---

### [EXP-2] MIG 기반 단일 프레임 병렬 실험 (H1 검증)

**목적**: Vision + LM + Flow에 별도 MIG 슬라이스 배정 시 레이턴시 변화

**MIG 구성 후보**:

```
Config A (균등 분할):
  MIG slice 1 (2g): Vision Encoder
  MIG slice 2 (4g): LM (VLM)
  MIG slice 3 (1g): Action Expert

Config B (LM 우선):
  MIG slice 1 (1g): Vision Encoder
  MIG slice 2 (5g): LM (VLM)
  MIG slice 3 (1g): Action Expert

Config C (baseline):
  전체 GPU (7g): 순차 실행
```

**실행 방법**:
```python
# 각 MIG 인스턴스에 별도 프로세스 할당
# CUDA_VISIBLE_DEVICES로 MIG UUID 지정
export CUDA_VISIBLE_DEVICES=MIG-UUID-1  # Vision 프로세스
export CUDA_VISIBLE_DEVICES=MIG-UUID-2  # LM 프로세스
```

**측정 지표**:
- 엔드투엔드 레이턴시 (wall clock)
- 각 모듈 GPU time

**예상 결과 (기각 방향)**:
```
Config C (baseline): 4882 ms ← 가장 빠를 것으로 예상
Config A (균등):     ~8000 ms 이상 (각 모듈이 작은 슬라이스에서 느려짐)
Config B (LM 우선):  ~6500 ms 이상 (LM은 빨라지나 Vision/Flow 너무 느림)
```

**논문에 쓸 수 있는 contribution**:
단일 프레임 레이턴시 관점에서 MIG 분할이 역효과임을 실증.  
"MIG는 멀티테넌시 처리량 최적화 도구이지, 단일 스트림 레이턴시 최적화 도구가 아니다."

---

### [EXP-3] 크로스프레임 파이프라인 구현 (H2 검증) ★ 핵심 실험

**목적**: ActionFlow를 Alpamayo 1.5 3단계 구조에 적용, FPS 향상 측정

**구현 구조**:

```
[순차 (현재)]
F0: [Vis][Pre][Dec][Flow]
F1:                      [Vis][Pre][Dec][Flow]
F2:                                          [Vis][Pre][Dec][Flow]
처리량: 1 / 4882ms = 0.20 FPS

[파이프라인 (목표)]
Stream A (Compute): [Vis_F0][Pre_F0]          [Vis_F1][Pre_F1]
Stream B (BW):                    [Dec_F0]              [Dec_F1]
Stream C (Flow):             [Flow_F0]              [Flow_F1]

파이프라인 단계 길이: max(Vis+Pre, Dec, Flow) = max(2011, 2013, 858) = 2013ms
처리량: 1 / 2013ms = 0.50 FPS → 2.4× 향상 이론치
```

**구현 방법**:
```python
# CUDA 스트림 2개 사용
stream_compute = torch.cuda.Stream()  # Vision + Prefill
stream_bw      = torch.cuda.Stream()  # Decode (BW-bound)

# Frame N Decode와 Frame N+1 Vision 동시 실행
with torch.cuda.stream(stream_bw):
    output_N = model.decode(kv_cache_N)          # Frame N decode

with torch.cuda.stream(stream_compute):
    vision_N1 = model.encode_vision(frames_N1)  # Frame N+1 vision

# 동기화
torch.cuda.current_stream().wait_stream(stream_bw)
torch.cuda.current_stream().wait_stream(stream_compute)
```

**KV Cache 관리 (핵심 기술적 도전)**:
- Frame N KV cache: Decode 중 stream_bw에서 접근
- Frame N+1 KV cache: Prefill 시 새로 생성
- 동시 접근 시 메모리 충돌 없음 확인 (별도 버퍼)

**메모리 사용량 확인**:
```
현재 단일 프레임 피크: ~28 GB (측정값)
파이프라인 시 추가 사용:
  Vision 활성화: ~2 GB
  추가 KV cache (N+1 프레임): ~3 GB
  필요 총량: ~33 GB
  Thor 가용: 128 GB → 충분
```

**측정 지표**:
- FPS (연속 10프레임 평균)
- 레이턴시 (첫 프레임 ~ 마지막 프레임)
- GPU 이용률 (두 스트림 동시 실행 확인)
- 메모리 사용량 피크

---

### [EXP-4] GPU 용량 스케일링 (H3 검증 — 비선형성 측정)

**목적**: 가용 GPU 메모리 제한 시 성능 변화 패턴 측정

**방법**: `torch.cuda.set_per_process_memory_fraction(fraction)` 로  
가용 메모리 제한 후 각 Phase 타이밍 측정

```
실험 조건 (가용 메모리 비율):
  100% (baseline): 128 GB
   80%:            102 GB
   60%:             77 GB   ← 가중치(22 GB) + KV(3 GB) + 활성화(5 GB) = ~30 GB 필요
   40%:             51 GB   ← 여전히 가중치 올라감
   25%:             32 GB   ← 임계치 근방
   20%:             26 GB   ← 가중치(22 GB) + 마진(4 GB) = 위험 구간
   15%:             19 GB   ← 가중치 전체 로드 불가 → 페이징 시작
```

**측정 지표**: 각 fraction에서 Vision/Prefill/Decode/Flow ms 및 BW

**예상 결과 (비선형 cliff 확인)**:

```
메모리 비율:  100%  80%  60%  40%  25%  20%  15%
Decode (ms):  2013 2015 2018 2020 2050 3000 15000+
                              ↑                 ↑
                         여전히 평탄           cliff
```

- **100% ~ 25%**: 가중치 전체 VRAM 상주 → Decode 시간 거의 불변
- **25% ~ 20%**: KV cache 압박 시작 → 소폭 증가
- **15% 이하**: 가중치 페이징 시작 → 지수적 증가

**논문 기여**:
"단일 배치 VLA 추론에서 GPU 메모리 용량은 성능에 비선형 영향.  
가중치 크기(22 GB) 이상의 용량이 보장되면 성능은 용량 무관."

---

### [EXP-5] MIG + 크로스프레임 조합 (종합 실험)

**목적**: MIG 2슬라이스 (Compute 전용 / BW 전용) + 크로스프레임 파이프라인 조합

```
슬라이스 배정:
  MIG_A (4g): Compute 집약 Phase (Vision + Prefill) — compute-bound
  MIG_B (3g): BW 집약 Phase (Decode + Flow)         — BW-bound
```

**가설**: BW-bound 단계는 MIG 축소 시 BW 비례 감소, Compute-bound 단계는  
Compute 비례 감소 → 균형 배정으로 전체 처리량 최대화 가능

**비교 조건**:
- Baseline: 전체 GPU 순차 (4882ms)
- EXP-2 최선 MIG 구성
- EXP-3 크로스프레임 (단일 GPU)
- **EXP-5**: MIG + 크로스프레임 조합

---

## 5. 실험 환경 구성

```bash
# Step 1: MIG 활성화
sudo nvidia-smi -mig 1
sudo nvidia-smi mig -lgip   # 사용 가능한 프로파일 확인

# Step 2: GPU 인스턴스 생성 (Config A 예시)
sudo nvidia-smi mig -cgi 2g.Xgb,4g.Ygb,1g.Zgb -C

# Step 3: UUID 확인
sudo nvidia-smi -L  # MIG UUID 목록

# Step 4: 프로세스별 MIG 지정
CUDA_VISIBLE_DEVICES=MIG-<UUID-LM> python3 run_lm.py &
CUDA_VISIBLE_DEVICES=MIG-<UUID-VIS> python3 run_vision.py &
```

### 의존성

```
torch >= 2.8.0 (소스 빌드, 현재 설치됨)
CUDA 13.0 (JetPack 7)
nvidia-smi with MIG support
```

---

## 6. 측정 지표 및 평가 기준

| 지표 | 측정 방법 | 목표 |
|---|---|---|
| 단일 프레임 레이턴시 (ms) | CUDA Events | baseline 대비 감소 % |
| 실효 FPS | 연속 10프레임 wall clock | > 0.35 FPS (75% 향상) |
| GPU 이용률 (%) | tegrastats GR3D_FREQ | 파이프라인 중 >80% |
| 메모리 피크 (GB) | torch.cuda.memory_stats | < 100 GB |
| 출력 품질 (waypoint RMSE) | 파이프라인 vs 순차 비교 | 동일 수준 |

---

## 7. 예상 결과 요약

| 실험 | 가설 | 예상 결과 | 기여 |
|---|---|---|---|
| EXP-1 (MIG 단독 스케일링) | BW-bound/Compute-bound별 다른 반응 | BW-bound는 MIG 크기 비례 감소 | Roofline 검증 |
| EXP-2 (MIG 병렬 배정) | H1 기각 | 레이턴시 증가 or 동일 | 부정적 결과 실증 |
| **EXP-3 (크로스프레임)** | **H2 확인** | **~2× FPS 향상** | **핵심 기여** |
| EXP-4 (용량 스케일링) | H3 확인 | 22 GB 이상이면 평탄, 이하 cliff | 설계 가이드라인 |
| EXP-5 (MIG + 파이프라인) | 조합 효과 | EXP-3 대비 소폭 추가 향상 or 동일 | 최적 구성 도출 |

---

## 8. 선행 연구와의 차별성 (Novelty)

| 선행 연구 | 한계 | 우리가 추가하는 것 |
|---|---|---|
| ActionFlow (2024) | OpenVLA-7B (2단계: Vision+LM), Flow 없음 | **3단계 VLA** (Vision+LM+Flow) 파이프라인 |
| DistServe / Splitwise | 다중 GPU 서버 환경 | **단일 엣지 SoC** (Jetson Thor) |
| MuxServe | 다중 모델 멀티테넌시 | **단일 모델** 내 단계별 파티셔닝 |
| GR00T N1 | 구현 상세 비공개 | **완전 공개 구현 + 측정** |
| ECRTS 2025 (GC) | 소형 모델, 범용 | **10B VLA 특화** |

**우리 연구의 포지션**:  
"Blackwell iGPU를 가진 Jetson AGX Thor에서 3단계 VLA(Alpamayo 1.5)의  
크로스프레임 파이프라이닝 및 MIG 분할 효과를 실증한 최초 연구"

---

## 9. 위험 요소 및 대안

| 위험 | 확률 | 실제 결과 | 대안 |
|---|---|---|---|
| Thor iGPU에서 MIG 미지원 | 20% | **❌ 실증 확인 (2026-05-15)** | CUDA MPS로 대체 |
| 크로스프레임 파이프라인 시 KV cache 관리 복잡도 | 높음 | 미측정 | ActionFlow KV Ring Buffer 코드 참고 |
| 파이프라인 출력 품질 저하 (stale KV state) | 중간 | 미측정 | 프레임 별 KV 완전 분리 옵션 유지 |
| MIG 슬라이스 간 통신 오버헤드 | 낮음 | **해당 없음 (MIG 미지원)** | 불필요 |

---

## ⚠️ EXP-0 실측 결과 업데이트 (2026-05-15)

### Thor Blackwell iGPU — MIG 하드웨어 지원 O, JetPack 7.1 소프트웨어 미구현

```
실험 일시: 2026-05-15 15:42 (Thor, JetPack 7.1, 드라이버 580.00)

시도한 명령어:
  sudo nvidia-smi -mig 1            → ✅ MIG 모드 플래그 활성화 성공
  sudo nvidia-smi -pm 1             → ✅ Persistence mode 활성화
  sudo nvidia-smi mig -lgip         → ❌ Failed to display GPU instance profiles: Unknown Error (RC=255)
```

**NVIDIA 공식 입장 (포럼 스태프 1차 발언)**:
- JetPack 7.0/7.1: MIG GI 파티셔닝 미지원 (확인됨)
- **JetPack 7.2 (2026년 6월 이전 출시 예정): MIG 최초 지원 예정**
- 소스: NVIDIA Developer Forums thread 367194, kayccc (NVIDIA 공식)

**하드웨어 상태**:
- Thor datasheet: "10 TPCs with MIG support" → 하드웨어 회로는 존재
- JetPack 7.1까지 NVML MIG 소프트웨어 스택이 DCE RPC 레이어에서 차단됨
- Fabric Manager 불필요 (NVLink 없는 단일 iGPU)
- 재부팅으로 해결 안 됨 (소프트웨어 미구현 문제)

**현재 실험 계획**:
- EXP-1, EXP-2 (MIG 기반): **JetPack 7.2 출시 대기** (6월 이전)
- EXP-3 (크로스프레임 파이프라인): **지금 실행 가능** → 우선 진행
- EXP-4 (GPU 용량 스케일링): **지금 실행 가능**
- MIG 대기 중 대안: **CUDA MPS** (SM 점유율 % 제한)

**논문 기여**:
"JetPack 7.1 기준 Thor iGPU는 MIG 하드웨어를 탑재하나  
 소프트웨어 스택이 미완성이며 JetPack 7.2에서 최초 지원 예정임을 실증."  
→ Jetson MIG 도입을 계획하는 연구자에게 실용적 타임라인 정보 제공

---

## 10. 실험 일정

| 주차 | 작업 |
|---|---|
| Week 1 | EXP-0: MIG 활성화 확인 및 환경 구성 |
| Week 1 | EXP-1: MIG 단독 스케일링 기준선 측정 |
| Week 2 | EXP-2: MIG 병렬 배정 실험 (H1 검증) |
| Week 3~4 | EXP-3: 크로스프레임 파이프라인 구현 및 측정 (핵심) |
| Week 5 | EXP-4: GPU 용량 스케일링 |
| Week 6 | EXP-5: 조합 실험 및 최종 분석 |

---

## 참고 문헌

1. ActionFlow: "A Pipelined Action Acceleration for Vision Language Models on Edge" (arXiv:2512.20276, 2024)
2. GR00T N1: "An Open Foundation Model for Generalist Humanoid Robots" (arXiv:2503.14734, NVIDIA 2025)
3. DistServe: "Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving" (arXiv:2401.09670, OSDI 2024)
4. Sarathi-Serve: "Taming Throughput-Latency Tradeoff in LLM Inference" (arXiv:2403.02310, OSDI 2024)
5. MuxServe: "Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving" (arXiv:2404.02015, ICML 2024)
6. Splitwise: "Efficient Generative LLM Inference Using Phase Splitting" (ISCA 2024)
7. Bakita et al.: "Hardware Compute Partitioning on NVIDIA GPUs for Composable Systems" (ECRTS 2025)
8. "Characterizing VLA Models across XPUs" (arXiv:2604.24447, 2025)
9. "Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference" (arXiv:2503.08311, 2025)

---

*데이터*: `profiling_results/260515_bw/allphase_bw.json` (기준선 측정값)  
*스크립트*: `scripts/profiling/260515_bw_allphase.py`

---

## 부록: 실험 스크립트 파일 목록

> 2026-05-15 작성 완료. Thor에서 바로 실행 가능.

| 실험 | 스크립트 | 출력 디렉터리 |
|---|---|---|
| EXP-0 (MIG 환경 확인) | `scripts/profiling/260515_exp0_mig_check.py` | `profiling_results/260515_exp0/` |
| EXP-1 (슬라이스 스케일링) | `scripts/profiling/260515_exp1_mig_scaling.py` | `profiling_results/260515_exp1/` |
| EXP-3 (크로스프레임 파이프라인) ★ | `scripts/profiling/260515_exp3_pipeline.py` | `profiling_results/260515_exp3/` |
| EXP-4 (용량 스케일링) | `scripts/profiling/260515_exp4_capacity_scaling.py` | `profiling_results/260515_exp4/` |
| 전체 순차 실행 | `scripts/profiling/260515_exp_runner.sh` | 위 전체 |

### 빠른 실행 (Thor)

```bash
# venv 활성화
source ~/alpamayo1.5/a1_5_venv/bin/activate

# 1) 환경 확인 (1~2분)
python3 ~/alpamayo1.5/scripts/profiling/260515_exp0_mig_check.py

# 2) 핵심 실험: 크로스프레임 파이프라인 (5~10분)
python3 ~/alpamayo1.5/scripts/profiling/260515_exp3_pipeline.py --frames 10

# 3) 전체 실험 일괄 실행 (20~30분)
bash ~/alpamayo1.5/scripts/profiling/260515_exp_runner.sh

# 4) 결과 Windows로 전송
scp -r ice401@100.95.177.101:~/alpamayo1.5/profiling_results/260515_exp* \
    /mnt/c/Users/nanay/Desktop/Alphamayo/profiling_results/
```

### MIG 활성화가 필요한 경우 (EXP-1 실제 MIG 모드)

```bash
# sudo로 MIG 활성화
sudo nvidia-smi -mig 1
sudo nvidia-smi mig -lgip    # 사용 가능한 프로파일 확인

# GI 생성 (예: 4개 균등 분할)
sudo nvidia-smi mig -cgi 19,19,19,19 -C
sudo nvidia-smi -L           # MIG UUID 확인

# 각 MIG 슬라이스에서 EXP-1 실행
CUDA_VISIBLE_DEVICES=MIG-<UUID> \
python3 ~/alpamayo1.5/scripts/profiling/260515_exp1_mig_scaling.py --real-mig
```
