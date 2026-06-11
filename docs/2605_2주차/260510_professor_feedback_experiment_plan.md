# 교수 피드백 기반 실험 계획서 v1.0
**작성일**: 2026-05-10  
**작성자**: Alpamayo-Korea 연구팀  
**대상**: Jetson AGX Thor 상의 Alpamayo 1.5 추론 프로파일링 2차 실험

---

## 0. 피드백 원문 및 분류

교수님 피드백을 주제별로 분류한다.

| # | 피드백 원문 | 분류 | 우선순위 |
|---|------------|------|---------|
| F1 | VRAM 요구? VRAM이 뭐지? 4GB까지 줄였다 24GB를 demand paging으로 | 메모리 계층 이해 | ★★★ |
| F2 | 정확히 shared 메모리 사용량이 언제 많아지고 언제 줄어드는지 시간에 따라서 | 시계열 메모리 측정 | ★★★ |
| F3 | GPU를 얼만큼 쓰는지 확인해보기 | GPU SM 활용률 | ★★★ |
| F4 | 모델이 여러개 돌아갈때 CPU 단독사용 GPU 단독사용 최대 가용 메모리 bandwidth | 대역폭 roofline | ★★ |
| F5 | 레지스터를 공유하면서? 하이퍼스레딩? 가상화된 CPU인가? | CPU 아키텍처 분석 | ★★ |
| F6 | GPU의 VRAM의 크기는 얼마인가? | 하드웨어 스펙 정리 | ★★ |
| F7 | SRAM VRAM의 차이는 무엇인가? | 메모리 개념 정리 | ★★ |
| F8 | CPU L3캐시 - 이게 중요하다 GPU VRAM이랑 / LLC 크기 파악 | LLC vs 모델 크기 | ★★★ |
| F9 | 이 크기를 넘어가면 너무 느려진다 (LLC 초과 시 성능 저하) | 캐시 병목 분석 | ★★★ |
| F10 | weight와 latency, response time 상관관계 파악 | 크기-레이턴시 실험 | ★★★ |

---

## 1. 배경 지식 정리 — 메모리 계층 구조

### 1.1 SRAM vs VRAM vs DRAM

교수님 질문 F6, F7에 대한 답으로, Thor의 메모리 계층을 정확히 정의한다.

```
메모리 유형 분류:

SRAM (Static RAM)
  ├─ 특성: 플립플롭 구조, 리프레시 불필요, 매우 빠름, 비쌈, 소용량
  ├─ 위치: CPU L1/L2/L3 캐시, GPU L1 캐시(shared memory), GPU L2 캐시
  ├─ 속도: ~수 TB/s (GPU L1), ~수백 GB/s (CPU L3)
  └─ Thor에서: CPU L2=256KB/core, GPU L2=50MB

DRAM (Dynamic RAM) = VRAM의 실체
  ├─ 특성: 커패시터 구조, 리프레시 필요, SRAM보다 느림, 싸고 대용량
  ├─ 종류: LPDDR5X (저전력), HBM2e (고대역폭 GPU용), GDDR6X (소비자 GPU)
  ├─ 속도: 273 GB/s (Thor LPDDR5X), 2,000 GB/s (A100 HBM2e)
  └─ Thor에서: 128GB LPDDR5X (CPU+GPU 공유, 물리적으로 같은 칩 옆에 위치)

"VRAM"의 정확한 의미:
  ├─ 원래: Video RAM, GPU 전용 메모리 (A100: 80GB HBM2e, RTX 4090: 24GB GDDR6X)
  ├─ Thor: VRAM이 별도로 존재하지 않음
  └─ Thor의 GPU는 128GB LPDDR5X를 CPU와 공유 → "Unified Memory"
```

### 1.2 Thor의 메모리 계층 전체 그림

```
[GPU SM 20개]
    │
    ├─ GPU L1 / Shared Memory: 48KB per SM × 20 = 960KB  (SRAM, ~20 TB/s)
    │
    ├─ GPU L2 Cache: ~50MB total  (SRAM, ~2-3 TB/s)
    │
    └─ Unified Memory (LPDDR5X): 128GB  (DRAM, 273 GB/s)
                │
[CPU 14 Cores (Cortex-A78AE)]
    │
    ├─ CPU L1: 64KB/core  (SRAM)
    ├─ CPU L2: 256KB/core  (SRAM)
    └─ CPU L3 (LLC): ~8MB shared  (SRAM, ~수백 GB/s)
```

**핵심 포인트**: Alpamayo 모델 가중치 = **22GB** → GPU L2(50MB) << CPU LLC(8MB) << 22GB  
→ 매 decode step마다 22GB를 273 GB/s LPDDR5X에서 읽어야 → **메모리 대역폭 병목 확정**

### 1.3 LLC(Last Level Cache)가 결정적인 이유 (F8, F9)

LLC = CPU의 L3 캐시 = GPU의 L2 캐시 중 가장 크고 마지막 캐시.  
이 크기를 넘어서는 데이터에 접근하면 DRAM(본 메모리)까지 가야 한다.

```
LLC 크기 < 작업 데이터 크기  →  DRAM 접근 필수  →  273 GB/s 병목

Thor LLC 계층:
  GPU L2:   ~50MB   ← 22GB 모델 대비 0.23%만 캐시 가능
  CPU L3:   ~8MB    ← 더욱 작음

결론: Decode 1 step마다 22GB 전체를 DRAM에서 읽음 (캐시 히트율 ≈ 0%)
이것이 decode latency = 22GB / 273 GB/s ≈ 81ms (이론 하한)의 근거
```

### 1.4 CUDA Demand Paging (F1)

교수님 질문: "4GB까지 줄였다 24GB를 demand paging으로"

```
전통적 GPU (A100):
  GPU VRAM 80GB  ←→  CPU DRAM
  추론 시 모델 전체(22GB)를 VRAM에 올려두고 사용

CUDA Demand Paging (Unified Memory 포함):
  GPU가 메모리 페이지를 "필요할 때만" 실제로 매핑
  
  - 할당만 하고 실제로 쓰지 않으면 물리 메모리를 차지하지 않음
  - GPU가 처음 접근하는 페이지 → Page Fault → OS가 physical mapping 수행
  - 이미 다른 곳에서 올려두었다면 → 재사용

실제 수치 (추정):
  "24GB 할당, 4GB만 demand paging으로 실제 사용"의 의미:
  → PyTorch activation buffer, KV cache 등 중간 텐서 할당 크기 vs 실제 사용 크기
  → 또는 nvidia-smi의 Used vs Reserved 차이

측정 실험 필요:
  nvmlDeviceGetMemoryInfo() → Used (실제 매핑된 페이지) vs Reserved (할당된 가상 공간)
```

---

## 2. 실험 A: GPU 메모리 사용량 시계열 측정 (F2, F3)

### 2.1 목적

추론 단계(Vision → Prefill → Decode → Flow)별로:
- GPU 메모리 사용량 (Used MB) 변화
- GPU SM 활용률 (%) 변화  
- 시간축에 정렬하여 "어느 단계에서 메모리가 증가/감소하는가" 가시화

### 2.2 측정 항목

| 항목 | API | 측정값 | 의미 |
|------|-----|--------|------|
| GPU Used Memory | `nvmlDeviceGetMemoryInfo().used` | MB | 실제 사용 중인 물리 메모리 |
| GPU Free Memory | `nvmlDeviceGetMemoryInfo().free` | MB | 사용 가능한 메모리 |
| SM Utilization | `nvmlDeviceGetUtilizationRates().gpu` | % | GPU 연산 코어가 얼마나 바쁜가 |
| Memory Utilization | `nvmlDeviceGetUtilizationRates().memory` | % | 메모리 인터페이스 사용률 |

### 2.3 측정 방법

```python
# scripts/profiling/profile_gpu_memory_timeline.py (신규 작성 예정)

import pynvml
import threading
import time
from collections import defaultdict

class GPUMemorySampler:
    """
    추론 단계별 GPU 메모리 + SM utilization 시계열 측정기.
    50ms 간격 샘플링, phase 마커로 단계 분리.
    """
    def __init__(self, device_idx=0, interval_ms=50):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        self.interval = interval_ms / 1000.0
        self.samples = []  # {'t_ms', 'used_mb', 'free_mb', 'sm_pct', 'mem_pct', 'phase'}
        self._running = False
        self._current_phase = "idle"
    
    def mark(self, phase_name):
        self._current_phase = phase_name
    
    def _sample_loop(self):
        t0 = time.perf_counter()
        while self._running:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.samples.append({
                't_ms': (time.perf_counter() - t0) * 1000,
                'used_mb': mem.used / 1e6,
                'free_mb': mem.free / 1e6,
                'sm_pct': util.gpu,
                'mem_pct': util.memory,
                'phase': self._current_phase,
            })
            time.sleep(self.interval)
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        self._thread.join()
```

### 2.4 기대 결과 형태

```
시간 (ms)     Used(GB)   SM%    Phase
─────────────────────────────────────────
0             23.1       0%     tokenize (GPU idle)
90            23.1       2%     vision start
800           23.8       85%    vision (peak SM 활용)
1500          23.2       90%    prefill
3400          23.0       65%    decode (BW-bound → SM 낮음)
5200          23.5       80%    flow_matching
5300          23.0       0%     done
```

**SM 활용률이 낮은 이유 (Decode)**: 메모리 대역폭 병목 → GPU 코어가 데이터를 기다리는 시간이 많음

### 2.5 시각화

- Panel 1: 시간 vs GPU Used Memory (GB)  → 단계별 메모리 증감 패턴
- Panel 2: 시간 vs SM Utilization (%)    → 각 단계의 연산 강도
- Panel 3: 시간 vs Memory Utilization (%) → 메모리 인터페이스 포화도
- 단계 경계선 + 색상 배경으로 Vision/Prefill/Decode/Flow 구분

---

## 3. 실험 B: CPU 아키텍처 분석 (F5)

### 3.1 목적

"레지스터 공유? 하이퍼스레딩? 가상화된 CPU?" 에 대한 실증적 답변

### 3.2 측정 방법

**Thor에서 실행할 명령어**:

```bash
# (1) CPU 기본 스펙
lscpu | grep -E "Architecture|CPU\(s\)|Thread|Core|Model name|Hypervisor|Virtualization"

# (2) 하이퍼스레딩 여부
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list
# 출력이 "0"이면 단일 스레드, "0,14"면 HT

# (3) 가상화 여부
systemd-detect-virt
# 출력: none(베어메탈), docker, kvm, lxc 등

# (4) GPU 레지스터 per SM
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# (5) LLC 크기
lscpu | grep "L3 cache"
# 또는
cat /sys/devices/system/cpu/cpu0/cache/index*/size
```

### 3.3 GPU 레지스터 공유 메커니즘

```
GPU SM 내부:
  ├─ 한 SM당 최대 65,536개 레지스터 (32-bit)
  ├─ 한 스레드 블록 = 32 threads (1 warp)
  ├─ 1 warp당 레지스터 = 65,536 / (동시 실행 warp 수)
  └─ 레지스터가 부족하면 → "register spilling" → DRAM 접근 → 느려짐

Alpamayo bfloat16 추론:
  - bfloat16 = 16-bit → 레지스터 효율 2배 (fp32 대비)
  - register spilling 최소화 → Thor SM 11.0에서 최적
```

### 3.4 예상 결론

| 질문 | 예상 답 | 근거 |
|------|---------|------|
| 하이퍼스레딩? | **없음** | ARM Cortex-A78AE는 SMT 미지원 (Intel x86과 달리) |
| 가상화된 CPU? | **베어메탈** | Jetson은 OS 위 직접 실행, hypervisor 없음 |
| GPU 레지스터 공유? | **있음** | CUDA warp간 SM 내 레지스터 풀 공유 |

---

## 4. 실험 C: 메모리 대역폭 Roofline 분석 (F4)

### 4.1 목적

"CPU 단독사용, GPU 단독사용 최대 가용 메모리 bandwidth" 측정  
→ 실제 추론이 이론 한계의 몇 %를 사용하는가

### 4.2 이론값

| 구성 | 이론 Peak BW | 비고 |
|------|-------------|------|
| CPU DRAM (LPDDR5X) | 273 GB/s | CPU+GPU 공유 |
| GPU (단독 사용 가정) | 273 GB/s | 동일 물리 메모리 |
| CPU + GPU 동시 | 273 GB/s (공유) | 총합 아님, 경쟁 발생 |
| GPU L2 BW | ~2,000 GB/s | 캐시 히트 시 |

### 4.3 실측 방법

```bash
# (1) 이론 피크 대역폭 측정 도구
# STREAM benchmark (CPU)
git clone https://github.com/jeffhammond/STREAM
gcc -O3 -fopenmp stream.c -o stream && ./stream

# (2) GPU bandwidth (bandwidthTest)
cd /usr/local/cuda/samples/1_Utilities/bandwidthTest
make && ./bandwidthTest

# (3) 실제 추론 시 대역폭
# 실측 BW = 모델 크기 / decode step latency
# = 22GB / 0.110s = 200 GB/s (실측)
# 이론 대비: 200 / 273 = 73.3%
```

### 4.4 Roofline 모델 그래프

```
성능 (GFLOPS/s)
│                    _____________________ Compute Roof (2,070 TFLOPS fp4)
│                   /
│                  /
│                 /  ← Roofline: 이 선 아래는 BW bound
│    ●Decode     /
│   (BW-bound)  /
│              /● Prefill (Compute-bound)
│             /
└─────────────────────────────── 산술 강도 (FLOP/Byte)
              Vision  Prefill  Decode  Flow
```

**결론 예측**: Decode는 산술 강도가 낮아 BW-bound 구간에 위치  
Prefill은 산술 강도가 높아 Compute-bound 구간에 위치

### 4.5 다중 모델 동시 실행 시나리오

```python
# 실험 설계: 모델 2개 동시 추론
# Process 1: Alpamayo inference (GPU heavy)
# Process 2: Tokenization (CPU heavy)

# 예상: GPU 메모리 대역폭 경쟁 → 단독 대비 1.3-1.8x 느려짐
# 측정: 1개 추론 latency vs 2개 동시 추론 latency 비교
```

---

## 5. 실험 D: 모델 가중치 크기 vs 레이턴시 상관관계 (F10)

### 5.1 목적

"weight와 latency, response time 상관관계 파악"  
→ 모델이 클수록 얼마나 느려지는가? 선형 관계인가?

### 5.2 실험 변수

```
독립변수 (Knob):
  A. 수치 정밀도 (Precision): fp32 → bf16 → fp8 → fp4
  B. 모델 크기 (레이어 수 직접 조작 불가 → precision으로 대리 측정)

종속변수 (Metric):
  - Decode step latency (ms/step)
  - Prefill latency (ms)
  - Flow Matching latency (ms)
  - Peak GPU memory (GB)
  - SM Utilization (%)
```

### 5.3 예상 결과 및 이론적 근거

```
이론: Decode latency ∝ 모델 크기 (BW-bound이므로)

Precision   모델 크기  이론 BW 필요   예상 latency
─────────────────────────────────────────────────
fp32        44 GB      44/273 = 161ms  ~2.0x bf16
bf16        22 GB      22/273 = 81ms   기준 (현재 110ms)
fp8         11 GB      11/273 = 40ms   ~0.5x bf16 예측
fp4          5.5 GB     5.5/273 = 20ms  ~0.25x bf16 예측
```

**실측값이 이론보다 1.35x 느린 이유**: compute overhead + CPU dispatch (24ms) 포함

### 5.4 실험 실행 방법

```bash
# bf16 (현재)
python scripts/profiling/profile_alpamayo.py --dtype bf16 --runs 4

# fp4 (Thor의 --dtype fp4 지원 시)
python scripts/profiling/profile_alpamayo.py --dtype fp4 --runs 4

# fp32 (메모리 허용 시, 44GB 필요 → Thor 128GB로 가능)
python scripts/profiling/profile_alpamayo.py --dtype fp32 --runs 4
```

### 5.5 시각화

- X축: 모델 가중치 크기 (GB)
- Y축: Decode step latency (ms/step)
- 선형 회귀선 + 이론 예측선 (BW-bound 기울기)
- 실측점과 이론선의 괴리 = compute overhead 정량화

---

## 6. 실험 E: LLC 크기 vs 모델 크기 임계점 분석 (F8, F9)

### 6.1 핵심 가설

> "LLC 크기를 초과하는 순간 성능이 급격히 저하된다"  
> → Alpamayo 22GB >> Thor LLC 8MB → 항상 LLC miss → DRAM 병목 고착

### 6.2 검증 방법

```python
# 소형 합성 모델로 "캐시 크기 vs 레이턴시" 변곡점 찾기
import torch

def measure_bw_vs_model_size(size_mb_list):
    results = []
    for size_mb in size_mb_list:
        # size_mb 크기의 텐서 생성
        n_elements = (size_mb * 1024 * 1024) // 2  # bf16 = 2 bytes
        W = torch.randn(n_elements, device='cuda', dtype=torch.bfloat16)
        x = torch.randn(1024, device='cuda', dtype=torch.bfloat16)
        
        # 반복 접근으로 bandwidth 측정
        N_ITER = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = (W * x[0]).sum()  # 전체 W를 읽는 연산
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        bw_gb_s = (size_mb / 1024 * N_ITER) / elapsed
        results.append({'size_mb': size_mb, 'bw_gb_s': bw_gb_s})
    return results

# LLC = ~50MB (GPU L2) 근방에서 BW 급감 예측
sizes = [1, 5, 10, 20, 50, 100, 500, 1000, 5000, 22000]
```

### 6.3 예상 BW vs 모델 크기 그래프

```
BW (GB/s)
2000 │▓▓▓▓ L2 캐시 히트 (< 50MB)
     │
 800 │         ▓▓▓ LLC 부분 히트 (50~200MB)
     │
 273 │─────────────────────────────── LPDDR5X 이론 피크
 200 │                    ▓▓▓▓▓▓▓▓▓▓▓ DRAM 접근 (> 200MB)
     │
   0 └────────────────────────────── 모델 크기 (MB)
     1    50  200        1000      22000
     ↑     ↑    ↑
    L2   변곡점  완전 DRAM-bound
```

---

## 7. 실험 파일 구성 계획

### 7.1 신규 작성할 스크립트

```
scripts/profiling/
  ├─ 260510_profile_gpu_memory_timeline.py   ← 실험 A: GPU 메모리+SM 시계열
  ├─ 260510_profile_bandwidth_roofline.py    ← 실험 C: BW roofline 측정
  ├─ 260510_profile_llc_cache_sweep.py       ← 실험 E: LLC 임계점 탐색
  └─ 260510_profile_precision_latency.py     ← 실험 D: 정밀도 vs 레이턴시

scripts/system/
  └─ 260510_cpu_arch_inspect.sh              ← 실험 B: CPU 아키텍처 분석
```

### 7.2 결과물 저장 경로

```
profiling_results/
  ├─ 260510_gpu_memory_timeline/
  │    ├─ memory_samples.json
  │    └─ figures/
  │         ├─ fig_memory_timeline.png
  │         └─ fig_sm_utilization.png
  ├─ 260510_bandwidth_roofline/
  │    ├─ bandwidth_results.json
  │    └─ figures/
  │         └─ fig_roofline.png
  ├─ 260510_llc_cache_sweep/
  │    └─ figures/
  │         └─ fig_llc_bandwidth.png
  └─ 260510_precision_latency/
       └─ figures/
            └─ fig_precision_latency.png
```

---

## 8. 실험 우선순위 및 일정

| 우선순위 | 실험 | 소요 예상 | 교수 피드백 대응 |
|---------|------|----------|----------------|
| 1 | 실험 A: GPU 메모리+SM 시계열 | 1일 | F2, F3 |
| 2 | 실험 D: Precision vs Latency | 0.5일 | F10 |
| 3 | 실험 E: LLC 임계점 | 0.5일 | F8, F9 |
| 4 | 실험 B: CPU 아키텍처 | 2시간 | F5 |
| 5 | 실험 C: BW Roofline | 1일 | F4 |

### 선행 조건

- [ ] Thor 보드 접속 가능 상태 확인
- [ ] pynvml 설치 확인: `python -c "import pynvml; pynvml.nvmlInit()"`
- [ ] fp4 dtype 지원 여부 확인: `python -c "import torch; print(torch.float4_e2m1fn)"`
- [ ] STREAM benchmark 빌드 가능 여부 확인

---

## 9. 교수님께 보고할 핵심 답변 초안

**Q. VRAM이 뭐지? demand paging으로 4GB?**  
A. Thor는 전통적 의미의 VRAM(GPU 전용 HBM)이 없습니다. 128GB LPDDR5X를 CPU와 GPU가 공유하는 Unified Memory 구조입니다. CUDA Demand Paging은 GPU가 메모리 페이지를 필요할 때만 물리적으로 매핑하는 메커니즘으로, 22GB 모델 전체가 상시 GPU L2(50MB)에 있는 것이 아닌 DRAM에서 필요 시 로드됩니다. 실험 A에서 실제 phase별 메모리 점유 변화를 측정하겠습니다.

**Q. LLC가 중요하다**  
A. Thor GPU L2 캐시 50MB << 모델 가중치 22GB이므로, decode 매 step에서 22GB를 반드시 LPDDR5X(273 GB/s)에서 읽어야 합니다. 이것이 decode step latency = 110ms의 근본 원인입니다. 실험 E에서 모델 크기 vs BW 그래프를 통해 LLC 임계점을 가시화하겠습니다.

**Q. GPU를 얼만큼 쓰는지**  
A. SM Utilization을 실험 A에서 pynvml로 측정합니다. 예상: decode 구간은 BW-bound이므로 SM 활용률이 낮음(~30-50%), vision/prefill은 compute-bound이므로 높음(~80-90%).

**Q. weight와 latency 상관관계**  
A. BW-bound 구간에서는 `latency ∝ weight size`가 선형 관계여야 합니다. fp32(44GB) vs bf16(22GB) vs fp4(5.5GB) 실험으로 이 선형성을 실증하겠습니다. 실험 D의 목표입니다.
