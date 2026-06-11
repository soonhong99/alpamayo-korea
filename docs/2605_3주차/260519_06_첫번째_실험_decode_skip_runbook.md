# 실험 1: Decode Skip / Adaptive Decode

**작성**: 2026-05-19 | **플랫폼**: Jetson AGX Thor | **스크립트**: `scripts/exp1_decode_skip.py`

---

## 왜 이 실험을 하는가

### 출발점: "CoC는 항상 필요한가?"

Alpamayo 1.5의 추론 파이프라인은 네 단계로 구성된다.

```
Vision (714ms) → Prefill (1,472ms) → Decode (1,926ms) → Flow Matching (890ms)
                                          ↑
                           이 단계의 필요성을 검증한다
```

Decode 단계는 **Chain-of-Causation(CoC)** 텍스트를 자동회귀 방식으로 생성한다.  
현재 기준으로 16~19 토큰이 생성되며, 토큰당 110ms이므로 이 단계에만 **1,926ms**가 소요된다.  
전체 5,009ms 중 **38%** 를 차지한다.

여기에 암묵적인 가정이 있다: **"CoC 텍스트를 완전히 생성해야만 Flow Matching이 올바른 궤적을 출력한다."**

이 가정을 한 번도 실측으로 검증한 적이 없다.

### 아키텍처적 근거

Action Expert(Flow Matching)는 VLM의 마지막 **hidden state**를 conditioning으로 받는다.

```
[VLM]
  Prefill → hidden_state[last]
                 ↓
  Decode → token_1 → token_2 → ... → token_N → hidden_state[last_decode]
                                                         ↓
                                               [Action Expert (Flow)]
```

`max_generation_length=0`으로 설정하면 Decode가 실행되지 않고,  
**Prefill의 마지막 hidden state**가 그대로 Action Expert에 전달된다.

이것이 합리적인 궤적을 만들어낼 수 있는지는 이론으로 예측할 수 없다.  
실측만이 답을 준다.

### 실험이 중요한 이유

어떤 결과가 나와도 다음 실험 방향이 결정된다.

| 결과 | 의미 | 다음 행동 |
|---|---|---|
| N=0에서 PASS | CoC 없이도 단순 주행 가능 | Adaptive Decode 아키텍처 설계로 이동 |
| N=1~8에서 PASS | 품질 유지 최소 토큰 수 확정 | N × 57ms가 새 Decode 하한. EOS Sync 격리로 이동 |
| N≥13에서만 PASS | CoC가 Action Expert에 필수적 | CUDA Graph 집중 투자 근거 확정 |

하루 안에 CoC 기여도의 전체 그림이 나온다.

---

## 어떻게 실험을 진행하는가

### 실험 설계 원칙

**1. 같은 입력, 다른 토큰 수만 변경**

모든 조건에서 동일한 카메라 프레임과 egomotion을 사용해야 공정한 비교가 된다.  
고정 시드(`seed=42`)로 생성된 텐서를 전 조건에 동일하게 적용한다.

**2. baseline을 먼저 실행하여 기준 궤적 확보**

ADE/FDE를 계산하려면 비교 대상이 필요하다.  
N=16(현재 baseline) 조건을 먼저 10회 실행하고, 그 평균 궤적을 기준으로 삼는다.

**3. 각 조건 10회 반복**

단 1회 측정은 샘플링 노이즈에 취약하다.  
Alpamayo의 generation에는 temperature=0.6, top_p=0.98 샘플링이 적용되므로  
동일 입력에서도 궤적이 달라질 수 있다. 10회 평균으로 노이즈를 제거한다.

**4. CUDA Event 기반 타이밍**

CPU wall clock은 Python GIL, OS 스케줄링 영향을 받는다.  
`torch.cuda.Event(enable_timing=True)`로 GPU 기준 실측값을 사용한다.

### sweep 조건

| 조건 | max_generation_length | 이론 Decode (ms) | 이론 Total (ms) |
|---|---|---|---|
| Decode Skip | **0** | 0 | ~3,077 |
| 극소 | 1 | ~110 | ~3,187 |
| 최소 | 3 | ~330 | ~3,407 |
| 단축 | 5 | ~550 | ~3,627 |
| 중간 | 8 | ~880 | ~3,957 |
| 중간 | 10 | ~1,100 | ~4,177 |
| 중간 | 13 | ~1,430 | ~4,507 |
| **Baseline** | **16** | ~1,760 | **~5,037** |

이론값은 NSight 실측 110ms/token 기준. 실제 측정값은 실험 후 채워진다.

### 핵심 API 호출

이 실험 전체가 아래 파라미터 1개의 변화다.

```python
pred_xyz_t, pred_rot_t, extra = model.sample_trajectories_from_data_with_vlm_rollout(
    data=model_inputs,
    top_p=0.98,
    temperature=0.6,
    num_traj_samples=1,
    max_generation_length=N + 8,  # 안전망 (전체 토큰 상한)
    max_coc_tokens=N,             # ← 이 값만 0, 1, 3, 5, 8, 10, 13, 16으로 변경
    return_extra=True,
)
```

`max_coc_tokens=N`이면 `ForceEarlyEOS` LogitsProcessor가 N번째 스텝 이후  
`<|traj_future_start|>` (EOS) logit을 강제로 최대값으로 만들어 즉시 EOS를 생성한다.  
N=0이면 첫 번째 토큰부터 EOS → Prefill 마지막 hidden state만으로 Flow Matching이 실행된다.

**사전 조건**: Thor에서 `patch_alpamayo_coc.py`가 먼저 실행되어 있어야 한다.  
패치가 없으면 `max_coc_tokens` kwarg가 무시되고 `max_generation_length`만으로 동작한다.

---

## 품질 판정 기준

ADE/FDE는 N=16 baseline 궤적을 기준으로 계산한다.

| 지표 | 정의 | 통과 | 실패 |
|---|---|---|---|
| **ADE** | 64개 waypoint 전체 평균 L2 거리 | < 0.2m | > 0.5m |
| **FDE** | waypoint[63] (6.4초 후) 단독 L2 거리 | < 1.0m | > 3.0m |
| **wp[0] 물리 검증** | 첫 waypoint: x ∈ [0.5, 1.5]m, \|y\| < 0.3m | 범위 내 | 범위 밖 |

**PASS 조건**: ADE < 0.2m **AND** FDE < 1.0m **AND** wp[0] 물리 검증 통과.  
세 조건을 모두 만족해야 "해당 토큰 수면 품질 유지"로 판정한다.

---

## 결과 해석

### 시나리오 A: N=0에서 PASS

CoC 없이도 Prefill hidden state만으로 합리적인 궤적이 생성된다.

```
의미:
  - "항상 CoC가 필요하다"는 가정이 틀렸다
  - 장면 복잡도에 따라 CoC 길이를 동적으로 결정하는
    Adaptive Decode 아키텍처의 실증적 근거 확보
  - 코드 파라미터 1줄 변경으로 5,037ms → 3,077ms (1.63× 개선)

다음 실험: Exp 2 (인터-사이클 파이프라이닝)
```

### 시나리오 B: N=1~8에서 PASS, N=0 FAIL

품질 유지 최소 토큰 수 N이 확정된다.

```
의미:
  - Action Expert가 최소 N tokens의 CoC reasoning을 필요로 한다
  - N × 110ms가 새 Decode 현실 하한 (오버헤드 포함)
  - N × 57ms가 새 Decode 이론 하한 (CUDA Graph 적용 시)

다음 실험: Exp 7 (EOS Sync 격리) → cudaStreamSynchronize 53ms 제거 검증
```

### 시나리오 C: N≥13에서만 PASS

CoC가 Action Expert 품질에 구조적으로 필수적이다.

```
의미:
  - "모델을 다르게 실행"하는 접근은 여기서 한계
  - Decode 자체를 빠르게 만드는 방향으로 집중
  - CUDA Graph로 53ms/token 오버헤드 제거가 최우선

다음 실험: Exp 6 (CUDA Graph) + Exp 5 (torch.compile) 병행
```

---

## 알려진 실행 오류

### 모델 체크포인트 없음

```
FileNotFoundError: 체크포인트 디렉토리 없음:
/home/ice401/alpamayo1.5/checkpoints/alpamayo_base
```

모델이 아직 다운로드되지 않은 것이다. 모델 다운로드 후 재실행:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'nvidia/Alpamayo-1.5-10B',
    local_dir='/home/ice401/alpamayo1.5/checkpoints/alpamayo_base',
    token='$HF_TOKEN',
)
"
```

### `max_generation_length=0` 시 RuntimeError

```
RuntimeError: cannot generate 0 tokens
```

이 에러 자체가 중요한 데이터다. Decode loop이 아키텍처적으로 우회 불가능하다는 뜻이다.  
→ 시나리오 C로 직접 이동. CUDA Graph 투자 근거가 된다.

---

## 관련 문서

- 실험 계획서 전체: `docs/260519_99ms_달성위한_실험계획.md`
- 이 실험의 배경 수치: `docs/260519_nsight_decode_overhead_analysis.md`
- 기술 분석 (BF16 ceiling 계산): `docs/260519_99ms_달성위한_현기술분석.md`
