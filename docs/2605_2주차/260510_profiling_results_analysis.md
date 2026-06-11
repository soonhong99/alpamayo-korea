# Alpamayo 1.5 프로파일링 결과 분석
**실험일**: 2026-05-10 | **보드**: Jetson AGX Thor (ice401@100.95.177.101)  
**실행**: `--warmup 2 --runs 4 --interval-ms 100`

---

## 1. 측정 결과 원시값

### 타이밍

| 구분 | 소요 시간 |
|------|-----------|
| Warmup 1 | 6,004 ms (cold, JIT compile 포함) |
| Warmup 2 | 5,059 ms |
| Run 1 | 5,183 ms |
| Run 2 | 4,846 ms |
| Run 3 | 4,861 ms |
| Run 4 | 4,844 ms |
| **평균 (run 1-4)** | **4,934 ms** |

### 단계별 GPU 활용률 및 메모리

| 단계 | SM 평균 | SM 최대 | GPU 메모리 평균 | GPU 메모리 피크 |
|------|---------|---------|----------------|----------------|
| Vision | 93.3% | 97.0% | 22.13 GB | 22.18 GB |
| Prefill | 96.0% | 98.0% | 22.29 GB | 22.50 GB |
| Decode | 94.8% | 98.0% | 22.25 GB | 22.26 GB |
| Flow | 94.4% | 95.0% | 22.49 GB | 22.55 GB |
| **전체 평균** | **92.7%** | **98.0%** | 22.27 GB | 22.55 GB |

---

## 2. 핵심 발견 3가지

### ① SM 활용률이 모든 단계에서 90%+

**예상했던 것**: Decode는 메모리 대역폭 병목(BW-bound) → SM 30~50%  
**실제 측정**: Decode 94.8% → 예상과 정반대

**이유: `GR3D_FREQ`는 SM compute 효율이 아니다**

`tegrastats`의 `GR3D_FREQ X%`는 GPU 3D 엔진이 클럭 주기 중 **무언가를 하고 있는 비율**입니다. DRAM에서 데이터를 기다리며 stall하는 시간도 "active"로 집계됩니다.

```
일반적인 SM utilization (nvml):
  "SM이 실제 computation을 수행한 시간 비율"
  → BW-bound decode: 30~50% (데이터 기다리는 시간이 많음)

GR3D_FREQ (tegrastats):
  "GPU warp scheduler가 active한 시간 비율"
  → BW-bound decode도 90%+ (DRAM 대기 중에도 warp 교체로 스케줄러는 active)
```

Decode 단계에서 GPU는 22 GB 가중치를 계속 DRAM에서 읽습니다. 읽는 동안 다른 warp가 실행되므로 warp scheduler는 항상 바쁩니다 → `GR3D_FREQ` 높음. 하지만 실제 compute 효율은 낮을 수 있습니다.

**결론**: `GR3D_FREQ`로는 compute-bound vs BW-bound 구분 불가.  
정밀 분석은 `nsys profile`의 `sm__throughput` + `l1tex__t_bytes` 메트릭 필요.

---

### ② KV Cache 실제 증가량: +0.37 GB (예상 +1.5 GB)

```
Vision  기저:  22.13 GB  (모델 가중치만)
Prefill 피크:  22.50 GB  (+0.37 GB)
Decode  평균:  22.25 GB
Flow    피크:  22.55 GB  (+0.42 GB, action expert activation)
```

예상(1,464 토큰 × 32 레이어 × bf16 ≈ 1.5 GB)보다 훨씬 작습니다.  
실제 입력 시퀀스가 짧거나, KV cache가 Decode 완료 후 즉시 해제되고 있음을 의미합니다.

---

### ③ 1회 추론 = 4.9초 (안정)

warmup 1 이후 run 1~4는 4.84~5.18초로 안정적입니다.  
이전 예상(8분/회)은 잘못됐습니다. 실제 decode 토큰 수가 예상보다 훨씬 적거나 early stopping이 적용됩니다.

---

## 3. 한계 및 다음 실험

### 현재 방법의 한계

| 한계 | 원인 | 해결책 |
|------|------|--------|
| compute-bound/BW-bound 구분 불가 | GR3D_FREQ의 의미 제한 | `nsys profile` 사용 |
| 단계별 소요 시간 불명 | 전체 시간만 측정 | CUDA Events per-phase 측정 |
| KV cache 증가량 불일치 | 샘플링 간격(100ms) > KV 증가 속도 | 10ms 간격 또는 hook 기반 측정 |

### 다음 실험: nsys로 정밀 분석

```bash
nsys profile \
    --trace=cuda,nvtx \
    --output=~/alpamayo1.5/profiling_results/260510_memory_utilization/nsys_run \
    python scripts/profiling/260510_profile_memory_utilization.py \
        --warmup 1 --runs 1
```

nsys에서 확인할 것:
- `sm__throughput` → 실제 SM compute 효율 (%)
- `l1tex__t_bytes` → SM당 DRAM 트래픽 → BW 병목 확인
- Decode vs Flow의 FLOPS/byte 비율 → BW-bound/compute-bound 판별

---

## 4. 논문에 쓸 수 있는 수치 (검증 완료)

| 수치 | 값 | 출처 |
|------|----|------|
| 모델 실제 파라미터 | 11.08 B | 실측 |
| 모델 크기 (bf16) | 22.16 GB | 실측 |
| 1회 추론 시간 | 4.93 s (avg, 4 runs) | 실측 |
| GPU 메모리 피크 | 22.55 GB | 실측 |
| Prefill KV Cache 증가 | +0.37 GB | 실측 |
| Decode 이론 하한 | 81.2 ms/step | 분석 (22.16 GB / 273 GB/s) |
| GR3D_FREQ (전체 평균) | 92.7% | 실측 (≠ SM compute util) |

> **주의**: GR3D_FREQ는 compute 효율이 아닙니다. 논문에서 "SM utilization" 대신  
> "GPU engine activity (tegrastats GR3D_FREQ)" 로 명시해야 합니다.
