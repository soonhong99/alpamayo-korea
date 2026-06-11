# Shared Prefill + Early Exit 실험 분석
**날짜**: 2026-05-28  
**스크립트**: `scripts/inference/260528_shared_prefill.py`  
**결과 파일**: `profiling_results/260528_shared_prefill/results.json`  
**상태**: 완료

---

## 1. 실험 목적

Shared Prefill(B=1 prefill → KV 가방 B=N 복제 → B=N decode)와
Early Exit(조기종료) 두 기법을 결합해 throughput을 측정한다.

이전 실험(260527)에서는 MAX_DECODE_STEPS=80을 전부 실행해 decode가 ~14,700ms였다.  
이번 실험에서는 Early Exit을 추가해 decode를 실제 EOS 위치에서 종료한다.

---

## 2. 실험 설정

```python
MAX_DECODE_STEPS    = 80    # 안전망 (하드 상한)
EOS_CHECK_INTERVAL  = 4     # 매 4 스텝마다 found.all() 체크
N_SWEEP             = [1, 2, 4]
NUM_WARMUP          = 2
NUM_MEASURE         = 5
TEMPERATURE         = 0.6
TOP_P               = 0.98
```

KV 가방 스펙:
```
layers=36, kv_heads=8, head_dim=128, seq_max=3170+80
B=1 가방 크기: 445.8 MB
```

---

## 3. Early Exit 구현 (핵심 코드)

두 가지 메커니즘을 조합:

### 3a. GPU Mask (CPU sync 0회 추가)
```python
already_done = tracker.found.unsqueeze(1)           # [B, 1], GPU bool
eos_fill     = torch.full_like(cur, eos_id)
cur_in       = torch.where(already_done, eos_fill, cur)
```
EOS를 이미 만난 시퀀스의 `cur`를 EOS 토큰으로 대체.  
이후 스텝의 forward pass는 무해하게 진행되고, tracker는 `& ~self.found` 가드로 재기록을 방지.

### 3b. Periodic CPU Sync (매 4스텝, 총 ~5회)
```python
if step % EOS_CHECK_INTERVAL == 0 and tracker.found.all().item():
    exited_at = step
    break
```
모든 배치 항목이 EOS를 만났으면 즉시 루프 탈출.

**안전 보장**: `tracker.eos_steps`는 매 스텝 GPU에서 업데이트되므로 break 시점에  
이미 정확한 EOS 위치가 기록되어 있다. EOS를 놓칠 가능성 없음.

---

## 4. 측정 결과

### Phase A — Baseline (B=N prefill + B=N decode)

| N | prefill_ms | decode_ms | steps_avg | total_ms (±std) | tps |
|---|-----------|-----------|-----------|-----------------|-----|
| 1 | 4,610 | 2,067 | 19.0 | 7,440 ± 443 | 2.8 |
| 2 | 8,572 | 2,086 | ~17.3 | 10,872 ± 268 | 2.9 |
| 4 | 17,123 | 2,997 | ~14.8 | 20,563 ± 370 | 2.2 |

### Phase B — Shared Prefill (B=1 prefill → clone → B=N decode)

| N | prefill_ms | clone_ms | decode_ms | steps_avg | total_ms (±std) | tps | speedup |
|---|-----------|---------|-----------|-----------|-----------------|-----|---------|
| 1 | 4,492 | 6 | 2,072 | 19.0 | 6,481 ± 165 | 2.9 | 1.15× |
| 2 | 4,484 | 13 | ~2,620 | ~17.8 | 7,116 ± 332 | 5.5 | 1.53× |
| 4 | 4,487 | 24 | 3,015 | ~17.1 | 8,124 ± 300 | 8.2 | 2.53× |

---

## 5. 핵심 관찰

### 5.1 Early Exit 효과

| | 이전 실험 (80 steps 전부) | 이번 (Early Exit) | 감소 |
|--|--------------------------|-------------------|------|
| decode_ms (N=1) | ~14,700ms | 2,067ms | **7.1×** |
| decode_ms (N=4 B) | ~14,779ms | 3,015ms | **4.9×** |

Early Exit으로 decode loop이 step 17 또는 21에서 일관되게 종료.  
(check_interval=4이므로 check point: step 16 또는 step 20)

### 5.2 Shared Prefill — Prefill 선형성 검증

Phase A prefill은 N에 선형 비례:
```
N=1: 4,610ms → N=2: 8,572ms (× 1.86) → N=4: 17,123ms (× 3.71)
```

Phase B prefill은 N에 완전 불변:
```
N=1: 4,492ms, N=2: 4,484ms, N=4: 4,487ms
```
→ Shared Prefill이 설계대로 1회만 수행됨을 실측 확인.

### 5.3 Speedup 향상: 1.64× → 2.53×

Speedup 공식 (clone 무시):
$$\text{speedup} = \frac{N \cdot P + D}{P + D}$$

| 상황 | D(decode) | speedup 이론 (N=4, P=4490ms) |
|------|-----------|------------------------------|
| 이전 (80 steps) | 14,700ms | (4×4490+14700)/(4490+14700) = **1.70×** |
| 이번 (early exit) | 3,000ms | (4×4490+3000)/(4490+3000) = **2.80×** |

실측: 1.64× → 2.53×. 이론 예측 방향과 일치.

**결론**: decode가 짧아질수록 prefill 절약 효과가 전체 비중에서 더 커져서 speedup이 향상된다.  
decode → 0 수렴 시 이론 최댓값 = N (= 4×).

### 5.4 배치 Decode 스케일링 특성

Phase A decode time:
```
N=1: 2,067ms
N=2: 2,086ms  ← N=1과 거의 동일 (1% 증가만)
N=4: 2,997ms  ← N=1의 1.45×
```

Thor GPU가 batch decode에서 N=2까지는 완전히 공짜로 처리.  
이는 N을 더 키울수록 (N=8, 16) decode overhead 대비 prefill 절약이 더 커진다는 의미.

### 5.5 Clone 오버헤드 — 완전히 무시 가능

| N | clone_ms | total 대비 비중 | 등가 복사 속도 |
|---|---------|----------------|----------------|
| 1 | 6ms | 0.09% | - |
| 2 | 13ms | 0.18% | - |
| 4 | 24ms | 0.30% | ~74 GB/s |

N에 선형. 실험 오차 수준. 이후 모든 분석에서 무시.

### 5.6 Throughput (tps)

```
Phase B N=4: 8.2 t/s  vs  Phase A N=1: 2.8 t/s  →  2.93× tps 향상
Phase B N=4: 8.2 t/s  vs  Phase A N=4: 2.2 t/s  →  3.73× tps 향상
```

동일한 4개 trajectory 동시 생성 시 baseline 대비 **3.7× throughput**.

---

## 6. 남은 병목 분석

Phase B N=4 시간 분해:

```
total = 8,124ms (100%)
├── prefill : 4,487ms  (55.2%)  ← 주요 병목
├── decode  : 3,015ms  (37.1%)  ← 2차 병목
└── clone   :    24ms  ( 0.3%)  ← 무시 가능
```

prefill이 여전히 절반 이상을 차지. 다음 연구 단계:
- **Async Pipeline**: `cudaMemPrefetchAsync` + CUDA Stream 이중화로 레이어별 load/compute 중첩 → prefill 단축
- **N 확장 실험**: N=8, 16으로 GPU 포화도 측정 → decode 스케일링 한계 탐색

---

## 7. 이전 실험 대비 요약

| 기법 | N=4 speedup | N=4 tps | 비고 |
|------|------------|---------|------|
| 260527 (decode 80 steps) | 1.64× | ~3.5 t/s | Early Exit 없음 |
| **260528 (Early Exit)** | **2.53×** | **8.2 t/s** | 조기종료 + Shared Prefill 결합 |

---

## 8. 다음 단계

| 우선순위 | 실험 | 예상 효과 |
|---------|------|----------|
| 1 | Async Pipeline (cudaMemPrefetchAsync + CUDA Stream) | prefill 55% 병목 공격 |
| 2 | N 확장 (N=8, 16) | batch decode GPU 포화도 측정 |
| 3 | EOS_CHECK_INTERVAL 민감도 분석 (1, 2, 4, 8) | sync overhead vs exit latency tradeoff |
