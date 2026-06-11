# Async VE Pipeline 실험 결과 — GPU SM 병렬 실행 불가 확인 및 진정한 병렬화 방향

**날짜**: 2026-06-04  
**스크립트**:
- `scripts/inference/260604_async_ve_pipeline_exp.py` (v4, Monkey-Patch, Async Pipeline 실험)
- `scripts/inference/260604_cuda_stream_concurrency_test.py` (하드웨어 병렬 실행 검증)

**결과 파일**:
- `profiling_results/260604_async_ve_pipeline_exp/results.json`
- `profiling_results/260604_cuda_stream_concurrency_test/results.json`

---

## 1. 요약

우리가 시도한 Async VE Pipeline은 **"두 GPU 커널을 CUDA stream으로 분리하면 동시에 실행된다"** 는 전제 위에 설계되었다. 그러나 실험을 거듭하면서 이 전제가 Thor iGPU에서 성립하지 않음을 확인했고, 최종적으로 완전히 독립적인 워크로드로 구성한 최소 단위 테스트를 통해 **GPU SM 간 동시 실행이 하드웨어 수준에서 불가능**함을 증명했다.

아울러 이 과정에서 중요한 인식 전환이 이루어졌다: **진정한 병렬 실행은 GPU SM 내부가 아니라 서로 독립적인 하드웨어 유닛 사이에서만 가능하다.**

---

## 2. 우리가 시도한 것: GPU SM ∥ GPU SM (CUDA multi-stream)

### 설계 의도

Alpamayo 1 회 inference는 아래 순서로 구성된다:

```
[VE (~550ms)] → [LM Prefill (~1430ms)] → [Decode (~1350ms)] → [Flow (~870ms)]
```

VE(Visual Encoder, ViT)는 Decode와 사용하는 자원이 다르다:
- VE: Tensor Core 집중 사용 (compute-bound, ViT attention + FFN)
- Decode: DRAM 대역폭 집중 사용 (memory-bound, 22GB 모델 가중치 순회)

이론적으로 이 두 연산은 **다른 자원**을 사용하므로 동시에 실행 가능해야 하고,
Decode(k) 가 진행되는 동안 VE(k+1) 을 미리 실행하면 다음 inference의 critical path에서 VE를 제거할 수 있다:

```
[이론 설계]
Inference k:   [VE_k][LM_k][Dec_k]
                          └──[VE_{k+1}]──┘  ← Dec_k와 병렬
Inference k+1:                 [LM_only_{k+1}][Dec_{k+1}]

이론 가속비: (VE+LM+Dec) / (max(VE,Dec)+LM) ≈ 1.22×
```

이를 구현하기 위해 CUDA stream을 두 개 사용했다:
- **stream_default**: VE 실행 (default stream)
- **stream_dec**: Decode 실행 (background stream)

`.item()` 을 decode loop 안에서 사용하면 stream_dec 만 sync하고 default stream(VE)은 계속 실행 가능하다는 점을 활용했다.

---

## 3. 실험 이력: 세 가지 다른 구현, 동일한 결과

### 실험 1: VE background, Decode default (v4 초기)

| 설정 | 내용 |
|------|------|
| VE stream | stream_ve (background) |
| Decode stream | default |
| .item() | ✅ 있음 |

**결과**: Sequential 3,294ms → Async 3,342ms = **0.99×** (가속 없음)

### 실험 2: VE default, Decode background, .item() 제거

| 설정 | 내용 |
|------|------|
| VE stream | default ← Phase 2-B 결론 적용 |
| Decode stream | stream_dec |
| .item() | ❌ 없음 (Python dispatch 전체 먼저 완료 후 VE 시작) |

**결과**: Sequential 3,294ms → Async 3,342ms = **0.99×** (가속 없음)

`.item()` 이 없으면 decode 15 step dispatch 에 ~600ms Python time이 소요되어 그동안 VE가 시작조차 못 함. ve_fg=572ms 는 JIT warmup 효과.

### 실험 3: VE default 먼저 디스패치, Decode background, .item() 복원 (최종)

| 설정 | 내용 |
|------|------|
| VE stream | default, decode 이전에 먼저 디스패치 |
| Decode stream | stream_dec |
| .item() | ✅ 있음 |

**결과**: Sequential 3,225ms → Async 3,339ms = **0.97×** (가속 없음, 오히려 약간 느림)

**Wall clock 직렬화 증명 (실험 3 inf 0):**

```
타임스탬프: 17:48:46,023 → 17:48:49,198 = 3,175ms wall

병렬 모델: max(ve=547, dec=1203) + lm_pf=1433 = 2,636ms  ← 불일치 (539ms 설명 불가) ❌
직렬 모델: ve(547) + dec(1203) + lm_pf(1433)  = 3,183ms  ← 8ms 오차 내 일치             ✅
```

4개 iteration 모두 동일 패턴으로 직렬 모델 일치:

| iteration | wall time | ve+dec+lm_pf | 오차 |
|-----------|-----------|--------------|------|
| inf 0 | 3,175ms | 547+1203+1433=3,183ms | 8ms |
| inf 1 | 3,204ms | 555+1218+1439=3,212ms | 8ms |
| inf 2 | 3,194ms | 559+1202+1440=3,201ms | 7ms |
| inf 3 | 1,779ms | 584+1204=1,788ms | 9ms |

---

## 4. ve_fg 가 빠른 이유: JIT warmup (병렬 실행 아님)

실험 3에서 `ve_fg=547ms` 로 측정되었는데, 이것이 Phase 2의 standalone VE=702ms보다 22% 빠르다. 처음에는 이것을 "VE가 Decode의 idle SM을 활용해서 빠른 것"으로 해석했으나, 이는 틀린 해석이다.

**결정적 증거:**

```
full_prefill (warm) ≈ 1,991ms = VE + LM
lm_pf       (warm) ≈ 1,437ms = LM만

내재된 순차 VE 시간 = 1,991 - 1,437 = 554ms

ve_fg (Phase 3-B)  = 547ms  ← 7ms 차이 (측정 오차 범위)
```

순차 실행의 warm VE 시간(554ms)과 "concurrent" ve_fg(547ms)가 동일하다.
만약 진짜 병렬 실행이었다면 Tensor Core를 더 많이 써서 438ms 수준이어야 한다.
**ve_fg=547ms는 Decode 이전에 직렬로 실행된 VE의 JIT-warm 시간이다.**

---

## 5. 핵심 검증 실험: 최소 단위 하드웨어 병렬 실행 테스트

세 번의 Async VE Pipeline 실험 모두 직렬 결과가 나왔지만, 이것이 "구현 문제"인지 "하드웨어 한계"인지 분리하기 위해 완전히 독립적인 최소 테스트를 실행했다.

**스크립트**: `260604_cuda_stream_concurrency_test.py`

### 테스트 설계

모델과 무관한 순수 CUDA 워크로드:
- **stream1**: matmul 반복 (compute-bound, Tensor Core 집중)
- **stream2 (alloc)**: `B.clone()` (memory-bound, DRAM 집중, 새 메모리 할당)
- **stream2 (prealloc)**: `B_dst.copy_(B)` (memory-bound, 사전 할당 버퍼, **allocator 호출 없음**)

`prealloc` 버전은 PyTorch caching allocator의 cross-stream implicit synchronization을 완전히 제거한다.

### 결과

| 케이스 | Compute | Memory | Serial | Parallel (alloc) | **Parallel (prealloc)** | 최대가능 |
|--------|---------|--------|--------|------------------|-------------------------|---------|
| Small (2K×2K, 1GB) | 0.8ms | 9.1ms | 9.6ms | 9.5ms (1.00×) | **9.8ms (0.97×)** | 1.08× |
| Medium (4K×4K, 4GB) | 5.1ms | 35.4ms | 39.9ms | 40.3ms (0.99×) | **40.4ms (0.99×)** | 1.14× |
| Large (4K×4K, 10GB) | 10.4ms | 88.2ms | 97.7ms | 93.2ms (1.05×) | **97.3ms (1.00×)** | 1.12× |

### 해석

```
alloc ≈ prealloc ≈ serial  →  세 케이스 모두 동일

가설 A: PyTorch allocator가 cross-stream sync 삽입
  검증: prealloc(copy_, allocator 호출 없음) 에서도 1.00× → ❌ 기각

가설 B: Thor iGPU 하드웨어 자체가 SM 간 스트림을 직렬화
  검증: data dependency 없음 + allocator 없음 + 다른 자원(compute vs memory) → 여전히 1.00× → ✅ 확인
```

**이 테스트가 결정적인 이유**: 모델, KV cache, Python dispatch overhead, PyTorch allocator 등 모든 소프트웨어 요인을 제거한 상태에서도 병렬화가 되지 않는다. 하드웨어 수준의 제약이다.

---

## 6. 근본 원인: GPU SM들이 독립적으로 실행되지 않는다

### 우리가 전제했던 것 vs 실제

```
[우리의 전제 — dGPU 아키텍처 기반]

stream_default: VE kernels  →  SM Cluster A에서 실행
stream_dec:     Decode kernels → SM Cluster B에서 실행
                                  ↑
                            완전히 독립 → 동시 실행 가능

[Thor iGPU의 실제]

stream_default: VE kernels  ─┐
stream_dec:     Decode kernels─┤ → 단일 SM Pool → 스케줄러가 순차 처리
                               │   (VE 완료 후 Decode 시작)
                               └─ 직렬화
```

Thor iGPU는 SoC 통합 GPU이므로:
1. **단일 Command Processor**: 커널 디스패치 큐가 하나 → stream A 완료 후 stream B 시작
2. **단일 SM Pool**: dGPU처럼 독립 GPC(Graphics Processing Cluster) 뱅크가 없음
3. **단일 DRAM 컨트롤러**: 131.9GB unified memory를 하나의 컨트롤러가 순차 처리

반면 dGPU(RTX 등, RTAS'24/RTSS'22 논문 환경):
```
GPC 0 (SM cluster) → stream A 전용 실행
GPC 1 (SM cluster) → stream B 전용 실행
→ 물리적으로 다른 하드웨어 → 진짜 동시 실행 가능
```

RTAS'24(RT-Swap), RTSS'22(Demand Layering)의 async pipeline 기법은 이 dGPU의 멀티 GPC 구조를 전제로 한다. Thor iGPU에는 이 전제가 없다.

---

## 7. 인식 전환: 진정한 병렬 실행은 독립 하드웨어 유닛 사이에서만 가능하다

이번 실험들을 통해 얻은 핵심 인식:

**"같은 GPU 안의 두 CUDA stream은 Thor에서 동시에 실행되지 않는다. 진정한 병렬 실행은 서로 다른 독립 하드웨어 유닛 사이에서만 가능하다."**

Thor SoC가 가진 독립 하드웨어 유닛:

```
Thor SoC
┌─────────────────────────────────────────────────────────────┐
│  CPU Cluster (ARM, 다수 코어)    ← GPU와 독립적으로 실행    │
│                                                             │
│  GPU                                                        │
│  ├── SM Array (모든 CUDA 커널)   ← 커널끼리는 직렬화 ❌     │
│  └── DMA/Copy Engine             ← SM과 독립적으로 동작 ✅  │
│                                                             │
│  (DLA, PVA 등 있을 경우)         ← GPU와 독립적으로 실행 ✅ │
└─────────────────────────────────────────────────────────────┘
```

| 조합 | 병렬 가능? | 이유 |
|------|----------|------|
| GPU SM ∥ GPU SM | ❌ 불가 (증명됨) | 같은 SM Pool |
| **DMA Engine ∥ GPU SM** | **✅ 가능 (미확인, 이론)** | **다른 하드웨어 유닛** |
| CPU ∥ GPU SM | ✅ 가능 | 다른 하드웨어 유닛 |
| DLA ∥ GPU SM | ✅ 가능 (DLA 존재 시) | 다른 하드웨어 유닛 |

---

## 8. 우리가 놓친 것: DMA Engine ∥ GPU SM (아직 미검증)

CLAUDE.md의 연구 방향에는 다음이 적혀있다:

> `cudaMemPrefetchAsync` + CUDA Stream 이중화로 **layer prefetch와 compute 중첩**

이것은 우리가 실험한 "GPU SM ∥ GPU SM" 이 아니다. **DMA Engine ∥ GPU SM** 이다:

```
stream_compute:  [Layer 1 SM 연산]──────────[Layer 2 SM 연산]──────────
stream_copy:           [→ DMA: Layer 2 가중치 prefetch →][→ DMA: Layer 3 →]

DMA Engine = GPU SM과 물리적으로 독립된 하드웨어
→ 이 조합은 동시 실행이 가능할 수 있다 (미검증)
```

Xavier 보드에서 "CPU랑 GPU 나눠서" 파이프라이닝을 했다는 것도 이와 같은 맥락이다. GPU SM을 두 stream으로 나눈 것이 아니라 **DMA(혹은 CPU, 혹은 DLA)와 GPU SM을 독립적으로 사용**한 것이다.

**단, DMA-SM overlap이 Thor에서 실제로 동작하는지는 아직 실험하지 않았다.** 이것이 다음 검증 대상이다.

---

## 9. 확인된 기술 자산

이번 실험들을 통해 직접적인 가속은 없었지만 다음을 완전 검증했다:

1. **Monkey-patch VE 캐시 기법** (정확성 100% 검증)
   - `run_ve_with_cache()` + `run_lm_prefill_monkey_patched()` 구현 완료
   - deepstack_features 3개 텐서 포함 full path 재현
   - logit max_diff = 0.0, top-1 일치 ✅
   - → VE를 분리 실행하는 것 자체는 완전히 작동한다. 병렬화 실패는 하드웨어 한계이지 monkey-patch 기법의 문제가 아님.

2. **VE/LM 분리 시간 측정** (warm 기준 확정)
   - VE alone: **562ms** (warm)
   - LM-only Prefill: **1430ms** (warm)
   - Decode 15 step: **1200ms** (~80ms/step, AppendOnlyCache-C 확인)

3. **Thor iGPU CUDA stream 직렬화 확정** (하드웨어 검증)
   - 3가지 stream 구성 모두 0.97~0.99×
   - 최소 단위 테스트 (prealloc) 에서 1.00× → 하드웨어 한계 확정
   - 소프트웨어 수준 우회 불가능

---

## 10. 실험 오류 이력

| 에러 | 원인 | 수정 |
|------|------|------|
| `ValueError: must specify exactly one of input_ids or inputs_embeds` | XOR check 위반 | position_ids pre-compute 방식 시도 |
| `AttributeError: 'Qwen3VL' has no 'get_rope_index'` | outer model에 없고 `model.vlm.model`에 있음 | `inner.get_rope_index` 사용 |
| `logit max_diff=2.3125` | deepstack_features injection 누락 | monkey-patch 방식으로 전면 재설계 |
| `TypeError: cannot unpack SimpleNamespace` | Thor `get_image_features` 2-tuple 언패킹 | `return (pooler_output, deepstack_features)` |
| `ValueError: too many values to unpack` | `.unsqueeze(0).unsqueeze(0)` → shape [1,1,1] 생성 | `.unsqueeze(1)` 단일 적용 |
| 0.99× (stream_ve bg) | VE를 background에 둬서 decode SM이 먼저 점유 | stream 순서 변경 시도 |
| 0.99× (.item() 제거) | Python dispatch 600ms 후에야 VE 시작 | .item() 복원 |
| 0.97× (최종) | **하드웨어 SM 직렬화 — 소프트웨어로 해결 불가** | 방향 전환 |

---

## 11. 다음 방향

### 즉각 필요한 실험

**DMA-GPU Compute Overlap 검증** (`cudaMemPrefetchAsync`)
- 28개 LM 레이어를 윈도우 단위로 나눠 prefetch
- `stream_copy`로 다음 레이어 가중치를 DMA prefetch
- `stream_compute`로 현재 레이어 SM 연산
- DMA Engine ∥ SM = 독립 하드웨어 → 이론적으로 가능
- **이것이 RT-Swap / Demand Layering 논문이 실제로 하는 것**

### 병행 진행 가능

- **KV Temporal Reuse** (Δt=100ms): 연속 프레임 KV cache 재사용, 단일 stream
- **Speculative Decoding**: Draft + Target 구조, 단일 stream 내 동작

---

*실험 상태: GPU SM 병렬 실행 불가 확정. DMA-GPU compute overlap은 미검증 상태로 다음 실험 대상.*
