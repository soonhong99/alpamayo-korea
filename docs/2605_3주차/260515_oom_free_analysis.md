# oom-free-alpamayo 기법 분석: 우리가 적용할 수 없는 이유

**분석 일시**: 2026-05-15  
**레포**: https://github.com/aveeslab/oom-free-alpamayo  
**결론 선요약**: 그들의 기법은 **메모리 용량(Capacity)** 문제를 해결한다. 우리 문제는 **메모리 대역폭(Bandwidth)** 문제다. 근본적으로 다른 두 문제이며, 그들의 기법을 Thor에 적용하면 아무 효과가 없다.

---

## 1. oom-free-alpamayo가 실제로 구현한 것

### 1.1 타겟 하드웨어

```
aveeslab 기준 환경:
  GPU: RTX 5070 Ti (VRAM 16 GB GDDR6X, dedicated)
       RTX 3080 Ti (VRAM 12 GB GDDR6X, dedicated)
  CPU: DDR5 64 GB (별도 시스템 메모리)
  연결: PCIe 4.0 x16 (이론 64 GB/s, 실측 H2D ≈ 20-28 GB/s)

모델 크기: Alpamayo-R1-10B = 21.52 GB (BF16)
문제:  21.52 GB > 16 GB  →  OOM, 실행 자체가 불가능
```

### 1.2 구현된 기법 (hook.py 분석)

```
alpamayo_memopt/hook.py: DoubleBufHook 클래스
alpamayo_memopt/profiler.py: measure_h2d_bandwidth_gbps() — PCIe H2D 대역폭 측정
alpamayo_memopt/predictor.py: C_total(k) = C_total(0) - k × slope_per_layer
scripts/infer.py: 3개 hook 인스턴스 (VLM, ViT, Expert)
```

**핵심 아키텍처: CPU Offloading + Ping-Pong Double Buffer**

```
GPU VRAM (16 GB):
  [고정 상주 영역]     - k개 VLM 레이어 항상 유지 (interleaved residency)
  [더블 버퍼 슬롯 A]  - 512 MB  ← 현재 실행 중인 레이어의 weight
  [더블 버퍼 슬롯 B]  - 512 MB  ← 다음 레이어 weight를 PCIe로 prefetch 중
  [ViT + activations + KV cache + overhead]

CPU RAM (64 GB):
  [나머지 모든 레이어 weight - 핀 메모리]
```

**실행 흐름:**

```
레이어 N 실행 시:
  슬롯 A: 레이어 N weight (이미 VRAM에 있음)
  슬롯 B: 레이어 N+1 weight를 CPU→GPU PCIe DMA 비동기 전송 중
                          ↑
                    별도 CUDA 프리페치 스트림
  
레이어 N 실행 완료 → CUDA event sync:
  compute stream: 슬롯 B ready 대기 (RAW sync)
  prefetch stream: 슬롯 A에 레이어 N+2 DMA 시작

→ 슬롯 A ↔ 슬롯 B 번갈아 사용 (ping-pong / 모듈로 연산)
```

**코드 핵심 로직 (hook.py 요약):**
```python
def _submit_dma(self, layer_idx, slot):
    # CPU pinned memory → GPU buffer slot (H2D DMA, 비동기)
    with torch.cuda.stream(self.prefetch_stream):
        self.gpu_buf[slot].copy_(self.cpu_pinned[layer_idx], non_blocking=True)
        self.dma_events[slot].record()

def register(self, module):
    # pre_hook: 이 레이어 실행 전 DMA 완료 대기 (RAW 동기화)
    #           GPU buffer → module.weight 포인터 rebind
    # post_hook: 실행 완료 후 CPU 포인터 restore
    #            다음다음 레이어 DMA 제출
```

**Interleaved Residency 최적화:**
```python
# predictor.py의 선형 모델
C_total(k) = C_total(0) - k * (R_decode * (C_DMA - C_EXE))

# k개 레이어를 항상 VRAM에 상주 → 해당 레이어의 DMA 시간 = 0
# k를 늘릴수록 latency 감소 (선형)
# k의 한계: VRAM 용량
```

### 1.3 실측 성능

| GPU | VRAM | 성능 | 의미 |
|---|---|---|---|
| RTX 5070 Ti | 16 GB | **4.09 s** | OOM → 실행 가능 (latency 증가 감수) |
| RTX 3080 Ti | 12 GB | **15.46 s** | OOM → 실행 가능 (더 많은 offloading) |
| 일반 GPU (24 GB+) | 24 GB | ~2-3 s | OOM 없음, 비교 기준 |

→ **그들의 목표: "돌아가지 않던 것을 돌아가게 한다."**  
→ 대가: latency 2-6× 증가 (기꺼이 감수)

---

## 2. Jetson AGX Thor의 상황

### 2.1 하드웨어 구조

```
Thor 메모리 아키텍처:
  ┌─────────────────────────────────┐
  │   LPDDR5X 128 GB (통합 메모리)   │
  │                                 │
  │   CPU 접근: ~80-100 GB/s         │
  │   GPU 접근: ~273 GB/s (이론)     │
  │              163 GB/s (실측)    │
  └─────────────────────────────────┘
         ↑
    CPU와 GPU가 같은 물리 메모리 공유
    PCIe 없음, 별도 VRAM 없음

모델 크기: 22 GB
사용 가능: 128 GB
용량 여유: 128 - 22 = 106 GB (충분히 남음)
```

### 2.2 우리의 실제 병목

```
Decode (seq=1) 한 token 생성 시:

레이어 1개 처리:
  1. Weight 로드: LPDDR5X → 레지스터/L2  (512 MB)
     시간 = 512 MB / 163 GB/s = 3.13 ms 중
            → 하나의 transformer layer forward: ~1.87 ms 측정값
  
  2. GEMV 계산 (seq=1):
     FLOPs = 2 × 512 × 8192 ≈ 8.4 MFLOPS
     시간 = 8.4M / 64,000 GFLOPS ≈ 0.00013 ms

Compute : Load = 0.00013 ms : 1.87 ms = 1 : 14,384
                                         ↑
                            Bandwidth-bound, 14,000배 차이
```

---

## 3. 핵심 차이: 왜 그들의 기법이 우리에게 무효인가

### 3.1 문제 자체가 다르다

```
               aveeslab (RTX GPU)          우리 (Jetson Thor)
               ─────────────────           ─────────────────
문제 종류:     메모리 용량 부족              메모리 대역폭 한계
               (Capacity)                  (Bandwidth)
               
증상:          OOM → 실행 불가             느림 → latency 목표 미달
               
목표:          실행 가능하게 만들기         더 빠르게 만들기
               
trade-off:     latency 희생하고 실행        latency를 줄여야 함
```

### 3.2 "CPU→GPU 전송"이 Thor에서 의미가 없는 이유

aveeslab의 double buffering이 효과를 내는 전제 조건:

```
전제 1: CPU 메모리와 GPU 메모리가 물리적으로 분리되어 있어야 함
  aveeslab: CPU DDR5  ─── PCIe 4.0 ───→  GPU GDDR6X  ✅
  Thor:     CPU/GPU 모두 LPDDR5X (같은 물리 메모리)       ❌

전제 2: CPU→GPU 전송에 사용하는 경로가 compute와 독립적이어야 함
  aveeslab: PCIe DMA 엔진 (compute 엔진과 완전 독립)      ✅
  Thor:     메모리 버스 1개를 CPU/GPU가 공유               ❌
            → "전송"을 해도 같은 LPDDR5X 버스 사용
            → compute와 bandwidth 경합 발생

전제 3: 전송 비용이 compute 시간으로 충분히 가려져야 함 (overlap 성립)
  aveeslab: seq=1 기준 layer compute ≈ 0.13 ms
            PCIe H2D (512 MB @ 20 GB/s) ≈ 25.6 ms
            → overlap 비율: 0.13/25.6 = 0.5% (사실상 거의 안 됨)
            → BUT: 상주 레이어 k 증가로 보완 + 실행 가능이 목적
  Thor:     layer compute ≈ 0.00013 ms
            LPDDR5X "전송" ≈ 1.87 ms
            → overlap 비율: 0.00013/1.87 = 0.007%  ← 사실상 0
            → "전송" 자체가 bandwidth 낭비 (이미 메모리에 있는데 복사하는 꼴)
```

### 3.3 Thor에서 그들의 기법을 구현하면 일어나는 일

```python
# Thor에 DoubleBufHook을 그대로 적용한다고 가정:

# 1. CPU pinned memory에 weight 복사
cpu_pinned = weight.cpu().pin_memory()
# → LPDDR5X의 일부 영역에 데이터 존재

# 2. GPU buffer로 "H2D 전송"
gpu_buf.copy_(cpu_pinned, non_blocking=True)
# → LPDDR5X의 한 영역 → 또다른 LPDDR5X 영역
# → 실제로는 메모리 내 복사 (device-to-device on unified memory)
# → 비용: 전송한 512 MB × 2 (read + write) = 1024 MB의 대역폭 소비
# → 원래는 512 MB만 읽으면 되는데, 2배의 대역폭 낭비!

# 3. 결과: 기존보다 latency 2배 증가
```

**요약: Thor에 그들의 기법을 적용하면 오히려 2배 느려진다.**

---

## 4. 정량적 비교

### 4.1 동일 기법 적용 시 예상 결과

| 항목 | aveeslab RTX 5070 Ti | Jetson AGX Thor |
|---|---|---|
| GPU 전용 VRAM | 16 GB GDDR6X | 없음 (통합) |
| CPU RAM | 64 GB DDR5 (별도) | 128 GB LPDDR5X (공유) |
| 모델 크기 | 21.52 GB | 22 GB |
| 용량 문제 | **있음** (21.52 > 16) | 없음 (22 << 128) |
| BW 병목 | PCIe 20 GB/s (오가는 데이터) | LPDDR5X 163 GB/s |
| Layer compute | ~0.13 ms (seq=1) | ~0.00013 ms (seq=1) |
| Layer DMA 시간 | ~25.6 ms (PCIe) | 1.87 ms (LPDDR5X) |
| Compute/DMA 비 | 1 : 197 | 1 : **14,384** |
| Double buffer 효과 | 용량 제약 극복 + 약간의 overlap | 없음 (BW 낭비만 발생) |
| 기법 적용 결과 | OOM → 4.09 s (실행 가능) | 1.87 ms → **~3.74 ms (2배 악화)** |

### 4.2 Roofline 관점

```
aveeslab (PCIe 시스템):
  Ridge Point = Compute Throughput / Memory BW
             = ~15,000 GFLOPS / 20 GB/s
             = 750 ops/byte
  
  Decode AI = 1 ops/byte (GEMV) → BW-bound
  
  하지만 그들의 목적은 "더 빠르게"가 아니라 "실행 가능하게"
  → Roofline 분석이 그들에게는 부차적 문제

Jetson Thor:
  Ridge Point = 64,000 GFLOPS / 163 GB/s = 392 ops/byte
  Decode AI = 1 ops/byte → BW-bound (392배 차이)
  
  우리의 목적: latency 감소 → BW 문제를 직접 해결해야 함
  → 데이터 이동을 늘리는 어떤 기법도 역효과
```

---

## 5. Double Buffering이 도움이 되는 경우 vs 안 되는 경우

### 5.1 도움이 되는 조건 (aveeslab이 해당)

```
Double Buffering이 유효하려면:
  compute_time(layer N) ≥ transfer_time(layer N+1)
  
aveeslab 기준:
  transfer_time = 512 MB / 20 GB/s = 25.6 ms (PCIe)
  
  compute_time이 25.6 ms 이상 되는 조건:
  batch_size × seq_len이 충분히 클 때
  
  decode seq=1: compute ≈ 0.13 ms → overlap ratio 0.5%
  prefill seq=512: compute ≈ 66 ms → overlap ratio 2.6× → 유리!
  
  → aveeslab 기법은 prefill에서 가장 효율적
  → decode에서는 그들도 k 상주 레이어에 의존
```

### 5.2 Thor에서의 crossover point

```
Thor에서 double buffering이 의미 있으려면:
  compute_time ≥ memory_load_time
  
  compute_time(seq=B) = 2 × hidden × B / GPU_TFLOPS
                      = 2 × 8192 × B / 64,000,000  ms
                      = B × 2.56 × 10^-7 s
  
  memory_load_time = 512 MB / 163 GB/s = 3.13 ms
  
  crossover: B × 2.56 × 10^-7 = 3.13 × 10^-3
             B = 3.13 × 10^-3 / 2.56 × 10^-7 = 12,227
  
  → batch_size ≥ 12,227이어야 double buffering이 의미 있음
  → 실시간 추론(batch=1)에서는 12,227배 조건 미달
```

---

## 6. Thor에서 실제로 효과 있는 최적화

aveeslab 기법이 안 된다면, 우리에게 맞는 방법은:

| 기법 | 원리 | 예상 효과 | 우선순위 |
|---|---|---|---|
| **FP4 quantization** | weight 크기 4× 감소 → BW 4× 감소 | 4× decode 속도 향상 | ★★★ 최우선 |
| **Cross-frame pipeline** | Frame N+1 Vision을 Frame N Decode 중 병렬 실행 | 전체 pipeline 1.70× | ★★★ |
| **KV cache L2 persistence** | 20토큰 KV(~10 MB) → L2(33.6 MB)에 고정 | KV re-read 제거 | ★★ |
| **Speculative decoding** (MagicDec) | Draft-verify 병렬화 | 2-3× | ★★ |
| **Prefill-Decode split** | Prefill batching으로 amortization | Prefill 효율 향상 | ★ |
| ~~Layer offloading (aveeslab 기법)~~ | ~~BW 낭비~~ | **역효과** | ✗ 사용 금지 |

---

## 7. 정리

```
aveeslab이 해결한 문제:
  "GPU VRAM이 모델보다 작다" → CPU에 분산 저장 + 필요할 때 PCIe로 가져옴
  결과: OOM 없이 실행 가능 (latency 증가는 감수)

우리의 문제:
  "LPDDR5X 대역폭이 모든 weight를 충분히 빠르게 읽기에 부족하다"
  → 이미 메모리에 있음, 더 이상 "가져올" 곳이 없음
  → 데이터를 덜 읽거나 (FP4), 읽는 동안 다른 작업을 하거나 (pipeline)

핵심 한 줄 요약:
  aveeslab = "접시에 음식을 올리는" 문제 (용량)
  우리 = "음식을 더 빨리 먹는" 문제 (속도)
  
  젓가락을 더 빨리 움직이는 방법이
  접시 크기를 늘리는 데 쓸 수 없듯이,
  접시 크기를 늘리는 방법이
  젓가락 속도를 높이는 데 쓸 수 없다.
```

---

## 참고: aveeslab 코드의 흥미로운 점 (Thor MIG 이후 활용 가능성)

JetPack 7.2에서 MIG가 지원되면, MIG 슬라이스마다 별도 VRAM 할당량이 생긴다.  
이 경우 각 슬라이스의 "effective VRAM"이 줄어들면 aveeslab 기법이 유효해질 수 있다.

```
MIG 슬라이스 예시 (JetPack 7.2 이후):
  1g.X slice → SM 1/10, 메모리 할당량 Y GB (Thor UMA 특성 상 미정)
  
  만약 Y < 22 GB 이면:
    → 슬라이스 내에서 capacity 문제 발생
    → 이때 aveeslab 기법 (CPU UMA ↔ GPU slice 간 이동) 유효
  
  하지만 Thor UMA 특성 상:
    - 물리 메모리는 여전히 같은 LPDDR5X
    - 슬라이스 이동은 메모리 내 복사 (BW 낭비)
    - MIG 메모리 할당이 소프트웨어 quota일 경우 단순 제한
    
→ JetPack 7.2 출시 후 MIG 프로파일 확인 필요
```
