# Demand Layering 논문 분석
**논문**: "Demand Layering for Real-Time DNN Inference with Minimized Memory Usage"  
**학회**: RTSS 2022 (Real-Time Systems Symposium)  
**분석일**: 2026-05-24  
**중요도**: ★★★★☆ — Async pipeline의 이론적 기반, Layer 관리 기법

---

## 1. 논문 개요

### 한 줄 요약
OS의 **Demand Paging**에서 영감을 받아, DNN 레이어를 필요할 때만 GPU 메모리에 로드함으로써 최소 메모리 사용으로 실시간 추론을 실현.

### 핵심 아이디어: 운영체제 비유

| OS 개념 | Demand Layering 대응 |
|---------|---------------------|
| Virtual Memory | 전체 DNN 모델 가중치 (DRAM) |
| Physical Memory | GPU 메모리 (제한된 용량) |
| Page | DNN 레이어 |
| Page Fault → Page Load | Layer miss → Layer Swap In |
| TLB (빠른 주소 변환) | GPU L2 캐시 |
| Working Set | 현재 추론에 필요한 레이어들 |

→ **큰 모델을 작은 GPU 메모리에서 실행하는 것이 핵심 문제**

### RT-Swap과의 차이

| 항목 | Demand Layering (RTSS'22) | RT-Swap (RTAS'24) |
|------|--------------------------|-------------------|
| 주 목표 | **메모리 최소화** | **throughput 최대화** |
| pipeline | 기본적 sequential | **Async** (발전된 버전) |
| 이론 기반 | OS virtual memory | Async I/O + RT scheduling |
| 발표 연도 | 2022 | 2024 |

→ RT-Swap이 Demand Layering의 아이디어를 받아 비동기화 + real-time 분석 추가

---

## 2. 핵심 기여

### 2.1 Layer-Granular Memory Management

DNN 모델 = 레이어들의 순서 집합으로 추상화:
```
Model = [Layer_0, Layer_1, ..., Layer_N]
각 Layer_i = {weights, biases, activations}
```

**Demand Layering의 동작**:
1. 추론 시작: GPU 메모리에 아무 레이어도 없음
2. Layer_0 실행 요청 → GPU 메모리에 없음 → "Layer Fault"
3. Layer_0을 DRAM → GPU 메모리로 로드
4. Layer_0 실행 완료 → Layer_0 evict (또는 메모리 여유면 유지)
5. Layer_1 실행 요청 → 반복

### 2.2 Prefetching 최적화

순차 실행 패턴을 이용한 선제적 로드:
```
현재 Layer_i 실행 중 → 백그라운드에서 Layer_i+1 로드 (비동기)
→ Layer_i+1 실행 시 이미 메모리에 있음 → "fault" 없음
```

이것이 RT-Swap의 Async Pipeline으로 발전.

### 2.3 Real-Time Analysis

최악 실행 시간(WCET) 계산:
$$T_{DL} = \sum_{i} \left[ t_{load,i} \cdot \mathbb{1}[\text{cache miss}] + t_{compute,i} \right]$$

단, prefetching 시:
$$T_{DL,prefetch} = t_{load,0} + \sum_{i=0}^{N-2} \max(t_{load,i+1}, t_{compute,i}) + t_{compute,N-1}$$

---

## 3. 수학적 구조

### 3.1 Memory Footprint 최소화

**기존 방식 (모두 로드)**: GPU 메모리 필요량 = 전체 모델 크기
$$M_{traditional} = \sum_{i=0}^{N} m_i$$

**Demand Layering**: 동시에 필요한 최대 레이어 수
$$M_{demand} = \max_{t} \sum_{i \in active(t)} m_i \approx 2 \times m_{max\_layer}$$

(현재 레이어 + prefetch 레이어 = 2개만 동시에 메모리에 필요)

### 3.2 Swap Bandwidth 요구량

파이프라인이 지연 없으려면:
$$t_{load,i+1} \leq t_{compute,i}$$

즉:
$$\frac{m_{i+1}}{BW_{DMA}} \leq t_{compute,i}$$

요구 DMA 대역폭:
$$BW_{required} = \frac{m_{i+1}}{t_{compute,i}}$$

이것이 **Alpamayo 적용 가능성 분석의 핵심 수식**이다.

---

## 4. Alpamayo에 대입 계산

### 4.1 레이어별 요구 DMA 대역폭 계산

Alpamayo에서 각 레이어의 prefetch를 다음 레이어 compute 중에 완료하려면:

| 레이어 | 크기 | GPU compute time (추정) | 필요 BW |
|-------|------|------------------------|---------|
| MLP (gate/up/down) | 96 MB | ~2-5 ms (추정, 미측정) | 96MB/3ms = **32 GB/s** |
| Q/O Attention | 32 MB | ~0.5-2 ms (추정) | 32MB/1ms = **32 GB/s** |
| K/V Attention (GQA) | ~8 MB | ~0.3-1 ms (추정) | 8MB/0.5ms = **16 GB/s** |

→ 필요 DMA BW ≈ **32 GB/s** 수준  
→ GPU DMA engine BW: 수백 GB/s (DRAM BW의 일부)  
→ **이론상 파이프라인 완전 중첩 가능** — 하지만 이는 iGPU에서 GPU와 CPU가 같은 DRAM 공유라는 점을 무시한 계산

### 4.2 iGPU에서의 함정

iGPU에서 "DMA copy"는 사실:
```
DRAM[cpu_ptr] → DRAM[gpu_ptr]  ← 같은 DRAM 내의 이동!
```

Thor는 **통합 메모리**이므로 CPU와 GPU가 같은 물리 DRAM 사용.
`cudaMemcpyAsync`는 실제로는:
1. 메모리 페이지 매핑 변경 (zero-copy일 경우 진짜 복사 없음)
2. 또는 L2 캐시 warming만 발생

→ `cudaMemPrefetchAsync(ptr, size, GPU_device)`를 쓰면  
→ 해당 메모리 페이지를 GPU L2 캐시로 prefetch  
→ 이것이 실제 우리 시스템에서의 "Demand Layering" 구현

---

## 5. 핵심 한계 (분석자 평가)

### 5.1 논문이 인정한 한계
- Prefetch가 맞지 않으면 (비순차 모델) 효과 없음
- Small compute time → prefetch가 compute보다 느리면 파이프라인 stall

### 5.2 통합 메모리 환경 한계 (Alpamayo 적용 시)
1. **"Swap"이 실제로 BW를 소모함**: 통합 메모리에서도 `cudaMemPrefetchAsync`는 PCIe bus 같은 별도 엔진이 아닌 DRAM BW 소모
2. **Autoregressive decode의 순차성**: 각 decode step에서 Layer 0 ~ Layer N을 순차 실행 → prefetch 방향이 명확해 예측 prefetching 가능 ✅
3. **65 step 반복**: 같은 레이어 순서를 65번 반복 → prefetch 패턴을 학습/고정 가능 ✅

---

## 6. 우리 연구에서 취할 아이디어

### 6.1 Layer Object 추상화
```python
class AlpamayoLayer:
    def __init__(self, weights_ptr, size_mb):
        self.weights_ptr = weights_ptr  # DRAM 주소 (통합 메모리)
        self.size_bytes = size_mb * 1024 * 1024
        self.is_prefetched = False
    
    def prefetch_to_gpu(self, stream):
        """GPU L2 캐시로 prefetch"""
        torch.cuda.cudart().cudaMemPrefetchAsync(
            self.weights_ptr.data_ptr(),
            self.size_bytes,
            0,  # GPU device
            stream.cuda_stream
        )
        self.is_prefetched = True
    
    def forward(self, x, stream):
        with torch.cuda.stream(stream):
            return self.module(x)
```

### 6.2 Alpamayo Demand Layering 파이프라인

```python
class AlpamayoPipeline:
    def forward_one_decode_step(self, hidden_state):
        prefetch_stream = torch.cuda.Stream()
        compute_stream  = torch.cuda.Stream()
        
        # Layer 0 먼저 prefetch (pipeline 시작)
        layers[0].prefetch_to_gpu(prefetch_stream)
        torch.cuda.synchronize()
        
        for i, layer in enumerate(self.layers):
            # 다음 레이어 prefetch (동시 시작)
            if i + 1 < len(self.layers):
                layers[i+1].prefetch_to_gpu(prefetch_stream)
            
            # 현재 레이어 compute
            with torch.cuda.stream(compute_stream):
                hidden_state = layer.forward(hidden_state)
            
            # 둘 다 완료 대기
            torch.cuda.current_stream().wait_stream(compute_stream)
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        
        return hidden_state
```

---

## 7. RT-Swap vs Demand Layering: 우리 연구 적용 전략

```
Demand Layering (RTSS'22):
  → 아이디어: 레이어 단위 on-demand loading
  → 우리 적용: cudaMemPrefetchAsync per layer

RT-Swap (RTAS'24):
  → 아이디어: Async pipeline (DMA + compute 중첩)
  → 우리 적용: CUDA Stream 이중화

통합 전략 (우리 연구):
  Demand Layering의 layer 추상화 + RT-Swap의 async pipeline
  + Thor iGPU 특성 (통합 메모리, System Cache, L2 Persistent Residency)
  + Alpamayo VLA 모델의 65-step decode 구조
```
