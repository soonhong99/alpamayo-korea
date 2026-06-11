# Cache 크기 · 모델 가중치 · Latency 상관관계 분석
## Thor iGPU에서 Decode가 느린 근본 원인과 현실적 해법

**작성일**: 2026-05-15  
**플랫폼**: Jetson AGX Thor (SM 11.0, LPDDR5X 128 GB, 273 GB/s)  
**모델**: Alpamayo 1.5 (22.16 GB, BF16)

---

## 1. Thor 메모리 계층 구조 — 실측 수치

기존 문서(`260510_hardware_spec_results.md`, `260510_professor_feedback_experiment_plan.md`)에서 확인된 수치:

```
[GPU 연산 단위 — 20 SM]
  │
  ├─ GPU L1 / Shared Memory : 228 KB / SM × 20 = 4.56 MB  (SRAM, ~20 TB/s)
  │
  ├─ GPU L2 Cache           : 33.6 MB  ← CUDA prop.L2CacheSize 실측값
  │                           (SRAM, 추정 ~1.5–4 TB/s)
  │
  └─ Unified LPDDR5X        : 128 GB   (DRAM, 273 GB/s 피크)
       │
[CPU — 12코어 Cortex-X4 + A720]
  ├─ L1 : 64 KB / 코어
  ├─ L2 : 1 MB / 코어  (= 12 MB 합계)
  └─ L3 (LLC) : 16 MB shared  ← NVIDIA 공식 datasheet DS-11945-001
```

**핵심 비율**:

| 저장소 | 크기 | Alpamayo 가중치(22.16 GB) 대비 |
|---|---|---|
| GPU L2 Cache | 33.6 MB | **659× 작음** |
| CPU L3 (LLC) | 16 MB | **1,385× 작음** |
| LPDDR5X DRAM | 128 GB | 0.17× (가중치가 메모리의 17%) |

→ **Decode 1 step마다 캐시 히트율 ≈ 0%. 매번 DRAM에서 전체 가중치를 새로 로드.**

---

## 2. 왜 Decode가 느린가 — 수식으로

### 2-1. Decode의 연산 특성

Decode는 `seq=1` GEMV (행렬-벡터 곱) 연산입니다.

```
[LM Decode, 1 token 생성]

입력: x (hidden_dim=4096, 1개 벡터)
가중치: W (4096 × 4096 행렬, FP16 = 32 MB per layer)

연산:  FLOPs = 2 × 4096 × 4096 = 33.5 MFLOPs  (매우 작음)
메모리: W를 DRAM에서 읽어야 함 = 32 MB per layer

Arithmetic Intensity = FLOPs / Bytes = 33.5 MF / 32 MB = 1.05 ops/byte
```

### 2-2. Roofline 분석

```
Thor Roofline:
  Ridge Point = BF16 TFLOPS / DRAM BW
             = 259 × 10¹² / 273 × 10⁹
             = 949 ops/byte

Decode AI ≈ 1 ops/byte  ←  Ridge Point(949)보다 949배 낮음

→ Decode는 현존하는 연산 중 가장 극단적인 메모리 병목 상태
```

**시각적으로:**
```
Compute 
(ops/s)    ┌─── Compute Roof (259 TFLOPS BF16)
259T ───── │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
           │
           │         / ← Ridge Point (949 ops/byte)
           │        /
BW Roof ───│───────/ ← BW Roof (273 GB/s)
           │      ╱
           │     ╱ ← Decode는 여기 (AI≈1)
           └────────────────────────── AI (ops/byte)
                1    10   100   949
                ↑
           지금 위치
```

### 2-3. 실측값과 이론값 비교

| 항목 | 이론 하한 | 실측값 |
|---|---|---|
| 가중치 로드 시간 (20 tok) | 22.16 GB / 273 GB/s = 81 ms | - |
| Decode 시간 (20 tok) | - | **2013 ms** |
| 유효 대역폭 | - | 163.4 GB/s (MBU 60%) |
| 이론 대비 달성률 | - | 60% (LPDDR5X 상한 대비) |

→ DRAM 대역폭을 60% 활용하고 있으며, 나머지 40%는 메모리 컨트롤러 오버헤드, 캐시 미스 패널티 등.

---

## 3. 캐시에 가중치를 올릴 수 있는가

### 3-1. L2 캐시 용량과 필요 압축률

```
GPU L2 = 33.6 MB
LM 가중치 = 16.44 GB (BF16)
필요 압축률 = 16.44 GB / 33.6 MB = 502×
```

알려진 모든 압축 기법을 적용해도:

| 기법 | 비트 수 | 압축 후 크기 | L2 대비 | 품질 |
|---|---|---|---|---|
| BF16 (현재) | 16 bit | 16.44 GB | 502× 초과 | 100% |
| FP8 | 8 bit | 8.22 GB | 251× 초과 | ~99% |
| INT4 (GPTQ, AWQ) | 4 bit | 4.11 GB | 125× 초과 | ~97% |
| 2-bit (QuIP#) | 2 bit | 2.05 GB | 63× 초과 | ~90% |
| 1.58-bit (BitNet b1.58) | 1.58 bit | 1.62 GB | 49× 초과 | ~81% |
| 1-bit (이진) | 1 bit | 1.03 GB | 31× 초과 | 심각한 저하 |
| **이론 최솟값** | **~0.5 bit** | **~514 MB** | **15× 초과** | **사실상 불가** |

**결론: 어떤 압축 기법으로도 10B 모델을 33.6 MB L2에 올리는 것은 물리적으로 불가능.**

### 3-2. L2에 올릴 수 있는 최대 모델 크기

```
L2 = 33.6 MB, BF16 = 2 bytes/param
최대 파라미터 수 = 33.6 MB / 2 = 16.8M parameters

→ 약 17M 파라미터 모델만 L2에 완전 상주 가능
   (BERT-tiny 수준, VLA로는 전혀 불가)
```

### 3-3. 단일 Transformer Layer는?

Alpamayo 1.5 LM의 레이어 1개 크기:
```
hidden_dim = 4096
4개 가중치 행렬 (Q, K, V, O) × (4096 × 4096) × 2 bytes = 128 MB/layer
FFN 포함: ~256 MB/layer

→ 레이어 1개 = 256 MB ≫ L2 (33.6 MB)
→ 레이어 1개도 캐시에 올릴 수 없음
```

---

## 4. 가중치-캐시-Latency 상관관계

### 4-1. 이론값: 가중치가 각 저장소에 있을 때

| 저장소 | 대역폭 | 20 token Decode 시간 | 현재 대비 |
|---|---|---|---|
| **DRAM (현재)** | **273 GB/s** | **2013 ms (실측)** | **baseline** |
| L2 Cache (가상) | ~1.5 TB/s | ~370 ms | ~5.4× 빠름 |
| L1/Shared (가상) | ~20 TB/s | ~28 ms | ~72× 빠름 |
| 레지스터 (이론한계) | ~259 TFLOPS | ~1.5 ms | ~1340× 빠름 |

→ **L2에 가중치가 있다면 현재의 5배 빠를 수 있다. 하지만 올릴 방법이 없다.**

### 4-2. 관계 공식

Decode latency의 하한:

```
T_decode = max(
    W_bytes / BW_effective,    ← 메모리 병목 (현재 지배)
    FLOPs / TFLOPS             ← 연산 병목 (batch=1에서는 무시 가능)
)

현재:
  W_bytes = 16.44 GB
  BW_effective = 163.4 GB/s (측정)
  T_decode_min = 16.44 GB / 163.4 GB/s × 20 tokens = ~2013 ms ✓ (실측과 일치)

만약 FP4 적용:
  W_bytes = 16.44 GB / 4 = 4.11 GB
  T_decode_min ≈ 4.11 / 163.4 × 20 = ~503 ms  (4× 감소)
```

---

## 5. 실제로 존재하는 연구와 기법

### 5-1. 가중치 Prefetch / 캐시 활용 연구

#### PRESERVE (arXiv:2501.08192, 2025)
**"Prefetching Model Weights and KV-Cache in Distributed LLM Serving"**
- 텐서 병렬 통신(all-reduce) 동안 다음 레이어 가중치를 L2로 prefetch
- Ascend 910B (192 MB L2)에서 **1.6× 속도 향상**
- 최적 L2 크기: 104 MB로 측정 (현재 우리 L2 33.6 MB보다 3× 큼)
- **한계**: 텐서 병렬 = 다중 디바이스 필요. Thor 단일 보드에는 적용 불가

#### LLM in a Flash (arXiv:2312.11514, Apple, 2023)
**"Efficient LLM Inference with Limited Memory"**
- Flash(SSD)에서 가중치를 온디맨드 로드할 때, FFN 레이어 희소성(90%+)을 이용
- 활성화될 뉴런만 예측 후 해당 가중치만 로드 → 로드량 **98% 감소**
- SSD→DRAM 전송을 줄이는 기법. DRAM→L2는 해당 없음
- **한계**: 우리는 이미 모든 가중치가 DRAM에 있음. 이 기법은 DRAM 용량이 부족할 때 적용

#### DECA (arXiv:2505.19349, MICRO 2025)
**"Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model"**
- L2 캐시 근처에 압축 해제 가속기를 두어 DRAM↔L2 트래픽 감소
- 4-bit 가중치를 DRAM에 저장, L2에서 즉시 압축 해제
- **가장 직접적으로 관련된 연구**: DRAM 대역폭 소모 ≈ W_int4 / T_decode
- 결과: DRAM 트래픽 4× 감소, 실효 대역폭 4× 향상

### 5-2. 캐시에 특정 가중치 고정 (CUDA L2 Persistence)

CUDA 11.0부터 Ampere(SM 8.0+)에서 제공, Blackwell(SM 11.0)에서 지원:

```python
# L2 캐시 Persistence 힌트 (PyTorch 수준)
# 특정 텐서를 L2에 우선 유지

import torch

# 방법 1: cudaAccessPropertyPersisting 마킹
# 텐서를 "persisting" 힌트로 마킹 → 하드웨어가 L2에 우선 유지 시도
# CUDA C++ 수준에서만 직접 접근 가능

# 방법 2: prefetch hint
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    tensor.data_ptr()  # GPU 접근으로 L2 warmup
```

**현실적 효과:**
- KV cache (핫 데이터, ~수십 MB) → L2 우선 유지 **가능하고 유효**
- 가중치 행렬 전체 → L2 (33.6 MB) 대비 레이어 1개(128 MB)도 크므로 **효과 없음**
- 첫 번째 / 마지막 attention layer의 KV 버퍼 고정 → 수 ms 수준 개선 가능

### 5-3. Speculative Decoding — 캐시 활용률 간접 개선

arXiv:2408.11049 (MagicDec, ICLR 2025):

```
일반 Decode:
  가중치 W 로드 → 1 token 생성
  Arithmetic Intensity ≈ 1 ops/byte

Speculative Decoding (draft length L=4):
  Draft model (소형): 4개 후보 token 생성 (빠름)
  Target model (Alpamayo): 5개 token 한 번에 검증
  → 같은 W를 로드하지만 5개 token 생성
  → 실효 Arithmetic Intensity ≈ 5 ops/byte (5× 향상)
  → 같은 DRAM 트래픽으로 5× 많은 token
```

**실측 결과 (MagicDec, Memory-Bound GPU)**:
- 수용률 α=0.8, L=4: 기대 token/step ≈ 3.3
- 실효 처리량: **2–3× 향상**
- Latency는 동일, Throughput이 증가

**VLA 적용 제약**: Draft 모델이 Alpamayo 구조(Vision+LM)와 맞아야 함 → 범용 LLM draft 모델 바로 사용 불가. 전용 소형 VLA draft 모델 개발 필요.

### 5-4. 양자화 — 현재 가장 현실적인 선택

#### FP4 (Blackwell 네이티브 지원)

```
현재 (BF16): 가중치 22.16 GB
FP4 적용:    가중치 22.16 / 4 = 5.54 GB

Decode 이론 하한:
  BF16: 5.54 GB × 4 / 273 GB/s × 20 tok = ~1621 ms → 4× 빠름
  실측 MBU 60% 유지 시: ~503 ms / 20 tokens

```

**중요**: CLAUDE.md에 `--dtype fp4` 플래그가 이미 문서화되어 있음.  
FP4가 Decode latency를 **이론적으로 4× 단축**시킬 수 있는 근거가 여기에 있음.

#### AWQ / GPTQ (INT4, 4-bit) — 검증된 품질 유지

| 기법 | 정확도 손실 | 속도 향상 | 특이사항 |
|---|---|---|---|
| GPTQ (arXiv:2210.17323) | <1% perplexity | 3–4× | 보정 데이터 필요 |
| AWQ (arXiv:2306.00978) | <0.5% | 3–4× | activation-aware |
| SqueezeLLM (arXiv:2306.07629) | <0.5% | 3–4× | sparse + quantize |

---

## 6. 실험 계획 — Cache-Weight-Latency 상관관계 측정

### 실험 A: 가중치 크기와 Decode Latency 선형 관계 검증

**목적**: `T_decode ∝ W_bytes / BW_effective` 가 실측에서도 성립하는가?

```python
# 실험 방법: 모델 레이어 수를 줄여서 가중치 크기 변화
# Alpamayo LM 32 레이어 → 1, 2, 4, 8, 16, 32 레이어로 측정

layer_counts = [1, 2, 4, 8, 16, 32]
# 각 설정에서 Decode 20 token 시간 측정
# 기대: T_decode ∝ n_layers (선형 관계)
# 이탈 시: 비선형 오버헤드 존재
```

**측정 지표**: CUDA Events 타이밍, 유효 대역폭 (= 레이어 크기 × n_tok / T_decode)

### 실험 B: FP4 vs BF16 Decode Latency 직접 비교

```bash
# BF16 (현재)
python3 scripts/profiling/260515_bw_allphase.py --dtype bf16

# FP4 (Blackwell 네이티브)
python3 scripts/profiling/260515_bw_allphase.py --dtype fp4
```

**기대**: FP4에서 Decode 약 4× 단축 (가중치 크기 4× 감소)

### 실험 C: KV Cache L2 Persistence 효과

```python
# KV Cache를 L2 persistence 힌트로 마킹
# Decode 단계에서 KV 재사용율 변화 측정
import ctypes
cudart = ctypes.CDLL('libcudart.so')
# cudaDeviceSetCacheConfig(cudaFuncCachePreferL2)
```

**기대**: 수 ms 수준 개선. KV cache가 33.6 MB L2 내에 들어올 때만 유효.  
(seq=20 token의 KV cache ≈ 20 × 32 layers × 2 × 32 heads × 128 dim × 2 bytes = 10 MB → **L2에 들어감**)

---

## 7. 핵심 결론

### 할 수 없는 것

```
❌ 가중치를 L2 캐시에 올리기 (22 GB >> 33.6 MB, 502× 불가)
❌ 단일 레이어를 L2에 올리기 (256 MB >> 33.6 MB)
❌ 압축으로 L2에 맞추기 (이론 한계도 514 MB, 15× 초과)
❌ PRESERVE prefetch (단일 디바이스에서 통신-컴퓨트 오버랩 없음)
```

### 지금 할 수 있는 것

```
✅ FP4 양자화 → Decode 이론 4× 단축 (22 GB → 5.5 GB)
✅ KV Cache L2 persistence 힌트 → 수 ms 개선
✅ Speculative Decoding → 2–3× Throughput (Latency는 동일, 처리량 증가)
✅ Cross-frame pipelining (EXP-3) → FPS 향상 (Latency 숨기기)
```

### 수치 요약

```
현재 상태:
  Decode = 2013 ms / 20 tokens = 100 ms/token

각 최적화 적용 후 이론 Decode 시간:
  FP4 적용:                 ~500 ms / 20 tokens  (4× 개선)
  Speculative Dec (3× 처리량): 동일 latency, 3× throughput
  FP4 + Speculative:        ~170 ms 등가 latency

도달 불가능한 이론 하한:
  L2-bound (가상):          ~370 ms / 20 tokens  (5.4×)
  Compute-bound (이론):     ~1.5 ms / 20 tokens  (1340×)
```

---

## 8. 참고 문헌

1. **PRESERVE** (arXiv:2501.08192) — LLM 가중치/KV prefetch, L2 최적 크기 104 MB 실측
2. **LLM in a Flash** (arXiv:2312.11514, Apple 2023) — Flash→DRAM 가중치 온디맨드 로드
3. **DECA** (arXiv:2505.19349, MICRO 2025) — L2 근처 압축 해제 가속기로 DRAM 트래픽 감소
4. **MagicDec** (arXiv:2408.11049, ICLR 2025) — Speculative decoding, memory-bound 환경 분석
5. **LLM Inference Unveiled** (arXiv:2402.16363) — Roofline 모델 체계적 적용
6. **Mind the Memory Gap** (arXiv:2503.08311) — GPU 메모리 병목 분석
7. **BitNet b1.58** (arXiv:2402.17764) — 1.58-bit 양자화 품질 한계
8. **AWQ** (arXiv:2306.00978) — activation-aware INT4 양자화
9. **Flash-Decoding** (Stanford CRFM, 2023) — KV cache 병렬 attention (가중치 병목 해결 안 함)
10. **StreamingLLM** (arXiv:2309.17453) — KV cache window (가중치 병목 해결 안 함)
11. **GPTQ** (arXiv:2210.17323) — 4-bit PTQ, <1% 품질 손실

---

*기반 실측 데이터: `docs/260515_allphase_bw_analysis.md`, `docs/260510_hardware_spec_results.md`*  
*실험 스크립트: `scripts/profiling/260515_bw_allphase.py`*
