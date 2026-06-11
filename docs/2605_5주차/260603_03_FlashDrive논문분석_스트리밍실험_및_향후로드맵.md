# FlashDrive 논문 분석 · 스트리밍 실험 결과 · 향후 로드맵

**날짜**: 2026-06-03  
**관련 실험**: `scripts/inference/260604_streaming_incremental_kv_feasibility.py`  
**관련 논문**: FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving (Zekai Li et al., UC San Diego / Princeton, 2025)

---

## 1. 우리 스트리밍 실험 결과 요약

### 1.1 Phase 1 — 토큰 구조 분석

Alpamayo 1.5의 입력 시퀀스(3,086 tokens)에서 각 카메라의 frame0(최오래된 프레임) 위치를 직접 추적했다.

```
Total tokens: 3,086
├── Text prefix: 28 tokens
├── Vision region: 2,983 tokens
│   ├── cam0_frame0: [29,  209)   ← 첫 카메라 첫 프레임
│   ├── cam0_frame1~3: [209, 776) 
│   ├── cam1_frame0: [776,  956)  ← gap = 567 tokens
│   ├── cam2_frame0: [1524, 1704) ← gap = 568 tokens
│   └── cam3_frame0: [2273, 2453) ← gap = 569 tokens
└── Ego suffix: 75 tokens
```

**결론**: camera-first(view-major) 레이아웃으로 인해 oldest frame(F0) 토큰이 시퀀스 내에서 완전히 **비연속적(non-contiguous)**. F0만 선택적으로 교체하는 incremental KV update는 불가능.

수식으로 표현하면:
$$K^l_p = W_K^l \cdot \text{LayerNorm}(X^l_p), \quad X^l_p \text{ depends on all } X^l_{q<p} \text{ via causal attn}$$

F0 위치 KV를 바꾸면 그 이후 모든 토큰의 K/V도 무효화됨 → **사실상 full re-prefill과 동일**.

---

### 1.2 Phase 2 — VE(Vision Encoder) 재사용 검증

t0→t1 (Δt=100ms) 에서 `t0_cam_k_frame1 == t1_cam_k_frame0` 검증 결과:

| 카메라 | pixel_values 완전 일치 | 타임스탬프 일치 |
|--------|----------------------|---------------|
| cam0   | ✅ EXACT             | ✅             |
| cam1   | ✅ EXACT             | ✅             |
| cam2   | ✅ EXACT             | ✅             |
| cam3   | ✅ EXACT             | ✅             |

**결론**: 4개 카메라 모두 pixel_values가 완전 동일. VE 재사용은 수치적으로 완벽하게 유효하다.  
16개 이미지 중 12개(75%)가 이전 step에서 재사용 가능 → **VE 728ms의 최대 75% = 546ms 절감 가능성**.

---

### 1.3 Phase 3 — Streaming Benchmark: **결정적 실패 확인**

| 모드 | Step | 총 latency | EOS | Decode steps |
|------|------|-----------|-----|-------------|
| FULL | 0 | 4,142ms | ✅ | 17 |
| FULL | 1–4 | 평균 4,203ms | ✅ 100% | 16–19 |
| EXPC | 0 (fresh) | 4,457ms | ✅ | 19 |
| **EXPC** | **1** | **5,852ms** | **❌** | **81 (MAX)** |
| **EXPC** | **2** | **6,234ms** | **❌** | **81 (MAX)** |
| **EXPC** | **3** | **7,133ms** | **❌** | **81 (MAX)** |
| **EXPC** | **4** | **7,452ms** | **❌** | **81 (MAX)** |

> EXPC가 FULL보다 **41% 느리다**. KV staleness가 단순 정확도 저하가 아니라 **EOS 생성을 완전히 막는다**.

**EOS 실패 원인 분석**:

단순 suffix-forward(Exp C) 방식은 t0의 vision KV(4프레임 윈도우: t−300ms~t+0ms) + t1의 ego suffix(t+100ms 상태)를 조합한다. 모델 관점에서 "비전이 t0를, ego가 t1을 말하는" 불일치가 발생한다. 이 불일치 컨텍스트에서 CoC reasoning이 수렴하지 못하고 MAX_STEPS까지 토큰을 생성한다.

**v3 실험(single-step)과 streaming의 차이**:
- v3: warmup 5회에서 suffix+decode path가 JIT 완전 컴파일됨 → 안정적
- streaming: EXPC path가 JIT cold → step1 첫 실행에서 숫자 미세 차이 → stochastic 발산
- 단, step2–4도 JIT 안정화 후 71ms/step이지만 여전히 EOS 실패 → **KV mismatch 효과도 실존**

---

## 2. FlashDrive 논문 분석

> FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving  
> Zekai Li, Yihao Liang, Hongfei Zhang (UC San Diego, Princeton)  
> OpenReview 2025 — https://z-lab.ai/projects/flashdrive

### 2.1 핵심 인사이트: "4단계 파이프라인, 4가지 다른 낭비"

FlashDrive가 발견한 핵심: Alpamayo의 추론은 하나의 병목이 아니라 **4가지 다른 성격의 낭비**가 직렬로 연결된 파이프라인이다.

```
[Encode] → [Prefill] → [Decode] → [Action]
   ↑            ↑          ↑          ↑
 공간 중복    시간 중복    순차 생성   과도한 반복
(75% 동일   (75% 동일    (자동회귀    (중간 단계는
 프레임 재처리) KV 재처리)  1토큰씩)    변화 없음)
```

각 단계에 맞는 해결책이 다르고, 이를 동시에 적용할 때 효과가 **곱셈적으로(multiplicatively) 중첩**된다.

---

### 2.2 최적화 기술 1: Streaming Inference — Encode+Prefill 동시 최적화

**비유**: 4컷 만화를 매번 처음부터 그리는 게 아니라, 3컷은 지난번 그림을 그대로 두고 가장 새로운 1컷만 새로 그리는 것.

**기술적 구현**:

1. **새 프레임을 마지막 프레임 위치에 삽입**  
   단순히 suffix만 교체(우리 Exp C)하는 게 아니라, 각 카메라 블록의 **마지막 프레임 위치에 새 프레임 토큰을 삽입**한다. Camera-first 레이아웃이 유지되면서 슬라이딩이 이루어진다.

   ```
   t0: [cam0_f0 | cam0_f1 | cam0_f2 | cam0_f3 | ... | cam3_f3 | ego_t0]
   t1: [cam0_f1 | cam0_f2 | cam0_f3 | cam0_f4 | ... | cam3_f4 | ego_t1]
                                               ↑                ↑
                                          KV 재사용           새로 계산
   ```

2. **Pre-RoPE Key Caching** (RoPE 이전 상태 저장)  
   **문제**: RoPE는 절대 위치를 인코딩하므로, 슬라이딩 윈도우에서 토큰 위치가 한 칸씩 앞당겨지면 이전에 저장한 K 벡터가 틀린 위치 정보를 담게 된다.

   ```
   [표준 캐싱 — 우리가 시도한 방식]
   K_cached = W_K × embedding + RoPE(position_t0)
   → t1에서 position이 달라지면 이 K는 틀린 값

   [FlashDrive Pre-RoPE 캐싱]
   K_raw = W_K × embedding          ← RoPE 적용 전 저장
   At t1: K = K_raw + RoPE(position_t1_corrected)  ← 올바른 위치로 on-the-fly 재계산
   ```

3. **Streaming Attention Mask**  
   새로 삽입된 프레임이 이전 프레임들에 attend하는 causality를 올바르게 유지하는 커스텀 마스크.

**결과 (RTX PRO 6000)**:
- Encode: 88ms → **12.5ms** (7.0×)
- Prefill: 177ms → **62ms** (2.9×)

---

### 2.3 최적화 기술 2: Speculative Reasoning — Decode 최적화

**비유**: 초안 작가(작은 2-layer 모델)가 8개 토큰을 한 번에 초안으로 제안하고, 검토자(10B 본 모델)가 한 번의 forward pass로 한꺼번에 검토하여 수락/거절.

**왜 자율주행 reasoning이 speculative decoding에 최적인가?**
- CoC(chain-of-causation) 토큰은 **고도로 구조화된 템플릿** (속도, 차선, 장애물 설명)
- 도메인이 좁아서 per-token entropy가 매우 낮음
- "차선 변경 결정" 토큰이 나오면 다음에 올 토큰이 이미 상당히 결정됨

**구현**: DFlash (Chen et al., 2026) — block size=8의 diffusion-based drafter, 2-layer. 평균 accept length = **5.6 tokens** (드래프트 8개 중 5.6개 수락).

**결과**: Decode 264ms → **61ms** (4.3×), 처리량 62 tok/s → **242 tok/s**

---

### 2.4 최적화 기술 3: Adaptive-Step Flow Matching — Action 최적화

**비유**: 10단계짜리 조각상 제작에서, 중간 5~8단계는 이전 단계와 거의 동일하니 이전 단계 결과를 그대로 재사용. 중요한 처음 3단계와 마지막 2단계만 실제로 작업.

**발견 (논문 Figure 5)**: 10-step flow matching에서 velocity field는 U자 패턴을 보임:
- Step 1~2: velocity 변화율 높음 (궤적 큰 그림 결정)
- Step 3~8: velocity 거의 상수 (중간 단계, 거의 변화 없음)
- Step 9~10: velocity 변화율 다시 높음 (물리적 궤적에 수렴)

→ 중간 단계의 velocity를 캐싱하여 재사용 → 10번 → 4번의 실제 forward pass

**결과**: Action 187ms → **46ms** (2.5×)

---

### 2.5 최적화 기술 4: System Optimizations — 모든 단계 기반 절감

**CUDA Graph**: 수백 개의 개별 kernel launch를 단일 GPU-side replay로 대체. Decode에 가장 효과적 (작은 kernel이 많기 때문).

**Kernel Fusion**: Q/K/V projection 3개 → 1개, gate/up projection 2개 → 1개로 융합. `max-autotune` 모드로 최적 구현 자동 선택.

**결과**: 716ms → **515ms** (1.39×, 28.1% 절감). 모델 계산량 변경 없이 오버헤드만 제거.

---

### 2.6 전체 성능 테이블 (RTX PRO 6000, 1 trajectory sample)

| 구성 | Encode | Prefill | Decode | Action | **Total** | ADE↓ | minADE6↓ |
|------|--------|---------|--------|--------|-----------|------|----------|
| Alpamayo 1.5 (baseline) | 88ms | 177ms | 264ms | 187ms | **716ms** | 1.721 | 0.767 |
| +System optimizations | 43ms | 192ms | 167ms | 113ms | **515ms** | - | - |
| +Streaming inference | 13ms | 62ms | 171ms | 116ms | **362ms** | 1.733 | 0.792 |
| +Speculative reasoning | 44ms | 198ms | 61ms | 114ms | **417ms** | 1.650 | 0.754 |
| +Adaptive flow matching | 44ms | 195ms | 170ms | 46ms | **454ms** | 1.566 | 0.818 |
| **All above (no quant)** | **12ms** | **62ms** | **61ms** | **46ms** | **181ms** | 1.561 | 0.855 |
| +W4A8 quantization | 13ms | 53ms | 48ms | 46ms | **159ms** | 1.568 | 0.844 |

**Jetson Thor**:

| 모델 | 1 trajectory sample |
|------|---------------------|
| Alpamayo 1.5 | 3,770ms |
| **FlashDrive (all)** | **943ms (4.0×)** |

---

### 2.7 Streaming Fine-tuning 필요성 (중요)

논문의 핵심 발견:
- Streaming KV cache(근사) → action accuracy 약 **0.3m ADE, 0.2m minADE 열화**
- CoC reasoning 토큰은 stale KV에 **robust** (주로 recent tokens에만 attend)
- Action Expert는 **전체 KV cache에 cross-attention** → 작은 mismatch도 증폭됨

해결책: **VLM backbone은 완전히 동결**, action expert만 fine-tuning  
방법: rollout-based teacher-forcing (L-1 step 스트리밍으로 KV 누적 후 마지막 step에서 action loss 계산)  
결과: 완전히 accuracy 복원 (1.73m ADE, 0.79m minADE6)

> 이것이 우리 실험에서 EXPC EOS 실패의 근본 원인과 연결된다: 단순 suffix replacement는 FlashDrive의 올바른 sliding window 삽입과 다르고, fine-tuning 없이는 action expert(및 EOS 생성 관련 레이어)가 불안정해진다.

---

## 3. "FlashDrive가 다 해놓은 건가?" — 우리의 포지션 분석

### 3.1 FlashDrive가 한 것

FlashDrive는 Alpamayo 1.5-10B에 대해 4가지 최적화를 적용하여 716ms → 159ms (4.5×), Thor에서 3,770ms → 943ms (4.0×)를 달성했다. **개념 증명(proof-of-concept)은 완료된 상태다.**

### 3.2 FlashDrive가 하지 않은 것 / 우리의 차별성

| 항목 | FlashDrive | 우리 상황 |
|------|-----------|----------|
| 양자화 | W4A8 포함 (159ms) | **불가 — 시스템 방법만 허용** |
| 코드 공개 여부 | 코드 비공개 | 직접 구현 필요 |
| 실험 하드웨어 | RTX PRO 6000 + Thor | **Thor 단독 타깃** |
| 양자화 제외 성능 | 181ms (RTX PRO 6000) | Thor: 추정 ~1,000ms |
| Fine-tuning 필요 | Streaming fine-tuning 필수 | 모델 수정 없는 경로 우선 탐색 |
| Rolling Trajectory | 언급 없음 | **CLAUDE.md 최종 목표** |

### 3.3 양자화 없이 우리가 달성 가능한 성능 추정

FlashDrive 결과 (no quant): 181ms on RTX PRO 6000  
Thor/RTX PRO 6000 속도 비율: 3,770 / 716 = **5.27×**

양자화 없이 FlashDrive 기법 모두 적용 시 Thor 추정:
```
181ms × 5.27 ≈ 954ms
```

우리 현재 기준선 (AOC): 4,366ms  
달성 가능 개선: **4.6× 개선** 예상

---

## 4. 우리 기술 적용 가능성 분류

### 4.1 ✅ 모델 수정 없이 즉시 적용 가능

| 기술 | 기대 효과 | 구현 방법 |
|------|---------|---------|
| Async VE Pipeline (CUDA Stream) | VE 728ms 제거 (critical path에서) | cudaStreams 이중화: decode(t_k)와 VE(t_{k+1}) 병렬 실행 |
| System Opt: CUDA Graph | Decode ~167ms/step → ~100ms | torch.compile(`cudagraphs`) — 단, CLAUDE.md에서 비호환 확인됨 → 수동 CUDA Graph capture |
| System Opt: Kernel Fusion | 1.39× overall | `torch.compile(mode='max-autotune')` — Triton API 비호환 문제 있음 |
| Adaptive-Step Flow Matching | Action 870ms → ~350ms | 10-step 중 중간 step velocity 캐싱, no parameter change |
| AppendOnlyCache-C (적용 완료) | Decode −24% (107ms→79ms/step) | ✅ 이미 적용됨 |

### 4.2 ⚠️ 추론 시간(inference-time) 수정 — 파라미터 변경 없음

| 기술 | 기대 효과 | 비고 |
|------|---------|------|
| Pre-RoPE Key Caching | Encode 7×, Prefill 3× | Qwen3-VL attention layer monkey-patch. 파라미터 변경 없음. Fine-tuning 없으면 accuracy 열화 허용 필요 |
| Streaming Attention Mask | 위와 함께 필요 | 새 프레임을 last-frame 위치에 삽입하는 로직 구현 |

### 4.3 ❌ 모델 수정 필요 (현재 방향에서 제외)

| 기술 | 이유 |
|------|------|
| Streaming Fine-tuning | Action expert 파라미터 업데이트 필요 |
| DFlash Speculative Decoding | 2-layer draft model 새로 학습(60k clips) 필요 |
| W4A8 Quantization | "양자화는 하면 안돼" 제약 |

---

## 5. 향후 로드맵

### Phase A: Async VE Pipeline (즉시 다음 실험)

**목표**: decode(t_k)와 VE(t_{k+1})를 CUDA Stream으로 병렬 실행  
**기대**: VE 728ms를 critical path에서 제거  
**구현 핵심**:

```python
# Stream 이중화 설계
stream_compute = torch.cuda.Stream()   # LM prefill + decode
stream_prefetch = torch.cuda.Stream()  # VE 실행

# Step k 실행 중:
with torch.cuda.stream(stream_compute):
    output_k = lm_decode(kv_cache_k)   # 현재 step decode

with torch.cuda.stream(stream_prefetch):
    ve_output_k1 = vision_encoder(frames_k1)  # 다음 step VE 미리 실행

torch.cuda.synchronize()  # 두 stream 완료 대기
```

**예상 결과**:
- 현재: VE(728ms) + LM(2,021ms) + Action(870ms) = 3,619ms (sequential)
- async 후: max(VE, LM) + Action = max(728, 2,021) + 870 = **2,891ms** → 약 720ms 절감

### Phase B: System Optimizations

**목표**: CUDA Graph + Kernel Fusion으로 각 stage overhead 제거  
**기대**: FlashDrive 기준 1.39× (현재 ~4,366ms → ~3,140ms)  
**주의**: CLAUDE.md에서 torch.compile 비호환 확인. 수동 `torch.cuda.CUDAGraph` capture 검토

### Phase C: Adaptive-Step Flow Matching

**목표**: Action Expert의 10-step → 4-step 실행 (중간 velocity 캐싱)  
**기대**: Action 870ms → ~350ms (2.5×)  
**구현**: step 3~8에서 v_t ≈ v_{t-1} 활용, cache 구조만 추가. 모델 파라미터 무변경.

### Phase D: Pre-RoPE Streaming (선택적)

**목표**: Qwen3-VL attention에 pre-RoPE key caching 적용  
**기대**: Encode ~728ms → ~182ms, Prefill ~1,423ms → ~460ms  
**조건**: 파라미터 변경 없음, accuracy 열화 허용(±0.3m ADE) 또는 추후 streaming fine-tuning  
**구현 방법**: `Qwen3VLSdpaAttention.forward` monkey-patch, K_raw 별도 저장, RoPE on-the-fly 재계산

### Phase E: Rolling Trajectory Pipeline (최종 목표)

**CLAUDE.md 최종 목표**: 단일 추론 latency가 100ms 미만은 물리적 불가 → 대신 **Rolling Trajectory 연속 스트리밍** 파이프라인으로 재정의.

```
시간 →
t=0ms:   [Inference_0 시작 ─────────────── 완료 t=X ms] → traj_0 출력
t=100ms: [Inference_1 시작 ─────────────── 완료 t=X+100ms] → traj_1 출력
t=200ms: [Inference_2 시작 ─────────────── 완료 t=X+200ms] → traj_2 출력

→ 매 100ms마다 새 trajectory 출력 (latency = X ms이지만 throughput = 10Hz)
```

Phase A~C를 통해 latency를 ~2,000ms 수준으로 낮춘 뒤, Rolling Trajectory로 10Hz 출력 달성.

---

## 6. 예상 성능 진화 (Thor, no quantization)

| 단계 | 기법 | 예상 Latency | 비고 |
|------|------|------------|------|
| 현재 (AOC 적용) | AppendOnlyCache-C | 4,366ms | 실측 |
| Phase A 완료 | +Async VE Pipeline | ~3,140ms | 720ms 절감 추정 |
| Phase B 완료 | +System Opt | ~2,260ms | 1.39× 추가 절감 |
| Phase C 완료 | +Adaptive Flow Matching | ~1,740ms | Action 2.5× |
| Phase D 완료 | +Pre-RoPE Streaming | ~1,000ms | Encode/Prefill 3× |
| Phase E | Rolling Trajectory | **10Hz throughput** | X ms latency, 매 100ms 출력 |

> **중요**: 이 수치는 RTX PRO 6000의 FlashDrive 결과를 Thor 배율로 환산한 추정값. 실측이 우선이며 상황에 따라 크게 달라질 수 있다.

---

## 7. 결론

**"FlashDrive가 다 해놓았나?"**  
개념 증명 측면에서는 그렇다. 그러나:

1. **코드 비공개** → 직접 구현해야 한다
2. **양자화 제약** → W4A8 없이 system + streaming + adaptive flow만으로 구현해야 한다 (~1,000ms 목표)
3. **Thor 단독 타깃** → 우리는 Jetson AGX Thor에서 실증해야 한다
4. **Rolling Trajectory** → FlashDrive는 단순 latency 감소, 우리는 10Hz 연속 스트리밍 아키텍처

우리 연구의 차별점은 **"양자화 없이, 모델 파라미터 수정 없이, 순수 시스템 파이프라이닝 기법만으로 Jetson AGX Thor에서 얼마나 달성 가능한가"**를 실증하는 것이다.

**다음 즉시 실험**: `scripts/inference/260604_async_ve_pipeline.py` — CUDA Stream 이중화로 VE(t+1)와 decode(t) 병렬 실행 실증.

---

## 참고

- FlashDrive paper: https://openreview.net/forum?id=kuZrNI5oZM
- FlashDrive project: https://z-lab.ai/projects/flashdrive
- DFlash (speculative decoding): arXiv:2602.06036
- 관련 실험 코드: `scripts/inference/260604_streaming_incremental_kv_feasibility.py`
- 스트리밍 실험 결과: `profiling_results/260604_streaming_incremental_kv_feasibility/results.json`
