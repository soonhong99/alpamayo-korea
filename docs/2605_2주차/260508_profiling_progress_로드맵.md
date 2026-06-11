# [로드맵] 최적화 계획 및 진행 현황

> **이 파일의 목적**: 무엇을 할 것인가 — 현재 상태 / 우선순위 / 구현 가이드  
> **연관 파일**: [결과](profiling_baseline_results.md) · [방법론](profiling_nsight_analysis.md) · [CPU/GPU 분석](profiling_stage_analysis.md)

---

## 현재 진행 상태

| 항목 | 상태 | 비고 |
|---|---|---|
| Thor에서 Alpamayo 1.5 실행 | ✅ 완료 | bfloat16, eager attention |
| NSight .nsys-rep 생성 | ✅ 완료 | 54MB, GUI에서 열 수 있음 |
| **단계별 프로파일링 (v2.0)** | **✅ 완료** | Vision/Prefill 100% 직접 측정 |
| 베이스라인 수치 확정 | ✅ 완료 | **5,004ms ± 172ms (n=8)** |
| FlashDrive 논문 분석 | ✅ 완료 | 6가지 기법 정리 |
| CPU/GPU 병렬성 분석 | ✅ 완료 | 토크나이징 CPU 위치 확인 |
| KV Cache Reuse 구현 | ⏳ 다음 단계 | — |
| CUDA Graphs 구현 | ⏳ 다음 단계 | — |
| 양자화 적용 | ⏳ 이후 단계 | — |

---

## 베이스라인 요약 (교수님 보고용)

```
플랫폼:    Jetson AGX Thor (SM 11.0, 128GB, CUDA 13.0)
모델:      Alpamayo 1.5 (11.079B, bfloat16)
베이스라인: 5,004ms ± 172ms (n=8)
100ms 목표 대비: 50× 초과

단계 분해:
  Vision Encoding    716ms  14.3%  (결정론적, CV 0.25%)
  LLM Prefill      1,466ms  29.3%  (결정론적, CV 0.26%)
  LLM Decode       1,920ms  38.4%  ← 주 병목 (확률적, CV 9%)
  Action Expert      902ms  18.0%  (가장 안정적, CV 0.09%)

CoC 토큰: 평균 17.5개, 109.7ms/토큰 (완벽한 선형)
이봉형 분포: 16토큰(4,844ms) vs 19토큰(5,164ms) 각 50%
```

---

## FlashDrive 6가지 기법 — 구현 우선순위

### 우선순위 1: CUDA Graphs (난이도 낮음, 즉시 효과)

**목적**: Decode 루프의 71,419개 kernel dispatch 오버헤드 제거  
**대상**: LLM Decode 단계 (1,920ms, 38.4%)  
**예상 효과**: ~88ms 단축 (Decode 중 CPU-GPU 교대 제거)

```python
# 구현 위치: alpamayo1_5.py 또는 패치 스크립트
import torch

# 1회 캡처
static_input = torch.zeros(1, 1, dtype=torch.long, device="cuda")
static_past  = create_static_kv_cache(max_len=2048)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model.vlm(
        input_ids=static_input,
        past_key_values=static_past,
        use_cache=True,
    )

# 매 decode 스텝: replay만
def decode_step(token_id, past_kv):
    static_input.copy_(token_id)
    g.replay()
    return static_output
```

**제약사항**:
- KV cache 크기가 사전 고정 필요 (dynamic shape 불가)
- stopping_criteria는 그래프 밖에서 체크

---

### 우선순위 2: Streaming KV Cache Reuse (교수님 지시, 핵심)

**목적**: 4프레임 중 3프레임의 Vision+Prefill 계산 재사용  
**대상**: Vision Encoding + LLM Prefill (2,182ms, 43.6%)  
**예상 효과**: 75% 감소 → 545ms (절감 1,637ms)

**원리**:
```
현재 (매 추론마다 전체 prefill):
  프레임 N:   [cam0_t0][cam0_t1][cam0_t2][cam0_t3] + [cam1~3] → KV 빌드
  프레임 N+1: [cam0_t1][cam0_t2][cam0_t3][cam0_t4] + [cam1~3] → KV 빌드  ← 75% 중복

Streaming KV Cache Reuse:
  프레임 N:   모두 prefill → KV cache 저장
  프레임 N+1: [cam0_t4] 새 프레임만 incremental prefill (75% 재사용)
```

**구현 방법 — HuggingFace DynamicCache 서브클래싱**:

```python
# streaming_kv_cache.py
from transformers import DynamicCache

class StreamingKVCache(DynamicCache):
    """
    4프레임 슬라이딩 윈도우 KV 캐시.
    매 추론마다 가장 오래된 프레임 토큰을 제거하고 새 프레임만 추가.
    """
    
    def __init__(self, frame_token_counts: list[int]):
        super().__init__()
        self._frame_token_counts = frame_token_counts  # 각 프레임의 토큰 수
        self._total_frames = len(frame_token_counts)
    
    def roll_frame(self, new_frame_tokens: int):
        """가장 오래된 프레임 제거, 새 프레임 슬롯 확보"""
        drop_tokens = self._frame_token_counts[0]
        # 각 레이어의 K, V에서 첫 drop_tokens개 제거
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx]   = self.key_cache[layer_idx][:, :, drop_tokens:]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, drop_tokens:]
        self._frame_token_counts = self._frame_token_counts[1:] + [new_frame_tokens]

# 사용:
kv_cache = StreamingKVCache(frame_token_counts=[...])  # 초기화
result = model.vlm.generate(
    input_ids=new_frame_tokens_only,   # 새 프레임만
    past_key_values=kv_cache,          # 이전 3프레임 KV 재사용
    ...
)
kv_cache.roll_frame(len(new_frame_tokens))
```

**구현 난이도**: 중간  
**참고**: FlashDrive 논문 Section 3.1 (Streaming KV Cache Reuse)

---

### 우선순위 3: W8A8 / INT8 부분 양자화

**목적**: 모든 GEMM 커널 (nvjet_tst 계열) 연산량 절반  
**대상**: 전 단계 (주로 Decode GEMM, 전체 GPU 시간의 ~55%)  
**예상 효과**: GEMM 커널 50% 단축 → 전체 ~28% 단축

```bash
# LLM Compressor (NVIDIA 공식 도구) 사용
pip install llmcompressor

python - << 'EOF'
from llmcompressor.transformers import oneshot
from compressed_tensors.quantization import QuantizationConfig

model = load_alpamayo()

oneshot(
    model=model.vlm,                    # VLM 부분만 양자화
    dataset="open_platypus",
    recipe="W8A8_dynamic_per_token",    # Weight INT8, Activation INT8
    max_seq_length=2048,
    num_calibration_samples=512,
)
model.save_pretrained("alpamayo_w8a8/")
EOF
```

**대안 — bitsandbytes (더 간단)**:
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)
model = Alpamayo1_5.from_pretrained(
    model_path,
    quantization_config=quantization_config,
)
```

---

### 우선순위 4: Adaptive Flow Matching

**목적**: Action Expert에서 속도장(velocity field) 변화 없으면 재계산 생략  
**대상**: Action Expert (902ms, 18.0%)  
**예상 효과**: 상황에 따라 50% 감소 → 451ms

**원리**:
```python
# flow_matching.py 수정
class AdaptiveFlowMatching:
    def __init__(self, threshold=0.01):
        self._prev_velocity = None
        self._threshold = threshold
    
    def sample(self, condition, ...):
        v_initial = self.compute_velocity(condition, t=0)
        
        # 이전 추론 대비 속도장 변화량 계산
        if self._prev_velocity is not None:
            delta = (v_initial - self._prev_velocity).norm()
            if delta < self._threshold:
                # 변화 없음 → 이전 ODE 결과 재사용
                return self._prev_trajectory
        
        # 변화 있음 → 전체 ODE 풀기
        trajectory = self._euler(condition, ...)
        self._prev_velocity = v_initial
        self._prev_trajectory = trajectory
        return trajectory
```

---

### 우선순위 5: Speculative Decode / DFlash

**목적**: CoC 토큰을 여러 개 동시에 병렬 생성  
**대상**: LLM Decode (1,920ms, 38.4%)  
**예상 효과**: ~3-4× 단축 → 480ms (별도 논문 arxiv:2602.06036)

**난이도**: 높음 — 별도 draft model 필요 또는 self-speculative 구현

---

## 누적 최적화 예상 시나리오

| 단계 | 추가 기법 | 예상 레이턴시 | 감소율 | 목표 대비 |
|---|---|---:|---:|---:|
| 베이스라인 | — | 5,004 ms | — | 50× |
| Step 1 | CUDA Graphs | 4,916 ms | 2% | 49× |
| Step 2 | Streaming KV Cache Reuse | 3,279 ms | 33% | 33× |
| Step 3 | W8A8 양자화 | 2,461 ms | 25% | 25× |
| Step 4 | Adaptive Flow Matching | 2,010 ms | 18% | 20× |
| Step 5 | Speculative Decode | ~670 ms | 67% | **7×** |
| Step 6 | FP4 + Kernel Fusion | ~335 ms | 50% | **3.4×** |

> **결론**: FlashDrive 기법 전체 적용 시 Thor에서 ~335ms 예상.  
> 100ms 달성은 FP4 양자화 + SM 11.0 전용 커스텀 커널 없이는 불가.  
> **현실적 단기 목표: 500ms 이하 (현재의 10×)**

---

## 즉시 실행 가능한 다음 단계

### Step A: CPU 사용률 실측 (이번 주)

```bash
# Thor에서 실행
tegrastats --interval 100 > profiling_results/tegrastats.log &
TPID=$!
python scripts/profiling/profile_alpamayo.py --warmup 3 --runs 8
kill $TPID

# 분석
python - << 'EOF'
import re
with open("profiling_results/tegrastats.log") as f:
    for line in f:
        m_cpu = re.search(r'CPU \[([^\]]+)\]', line)
        m_gpu = re.search(r'GR3D_FREQ (\d+)%', line)
        if m_cpu and m_gpu:
            cpus = [int(x.split('%')[0]) for x in m_cpu.group(1).split(',')]
            print(f"CPU avg: {sum(cpus)/len(cpus):.0f}%  GPU: {m_gpu.group(1)}%")
EOF
```

### Step B: CUDA Graphs 프로토타입 (이번 주)

```bash
cd ~/alpamayo1.5
python - << 'EOF'
import torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).cuda().eval()

# 단순 테스트: generate() 전후 시간 비교
# CUDA Graphs 없음
t0 = time.perf_counter()
out = model.vlm.generate(input_ids, max_new_tokens=20)
t1 = time.perf_counter()
print(f"Normal: {(t1-t0)*1000:.0f}ms")

# CUDA Graphs 있음 (generate 수준은 적용 어렵고, 단일 forward step에 적용)
EOF
```

### Step C: KV Cache Reuse 구현 시작 (다음 주)

```bash
# 1. 프레임당 토큰 수 확인
python - << 'EOF'
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

data = load_physical_aiavdataset("030c760c-ae38-49aa-9ad8-f5650a545d26", t0_us=5_100_000)
# ...
inputs = processor.apply_chat_template(messages, tokenize=True, ...)
print(f"Total input tokens: {inputs['input_ids'].shape[1]}")
# → 이 수치로 프레임당 토큰 수 역산 후 StreamingKVCache 설계
EOF
```

---

## FlashDrive 논문 참고 섹션

| 구현 대상 | 논문 섹션 | 핵심 알고리즘 |
|---|---|---|
| Streaming KV Cache Reuse | Section 3.1 | HuggingFace DynamicCache 서브클래스 |
| CUDA Graphs | Section 3.2 | `torch.cuda.CUDAGraph()` |
| Kernel Fusion | Section 3.3 | 커스텀 CUDA 커널 (Triton/CUTLASS) |
| DFlash Speculative | Section 3.4 + arxiv:2602.06036 | Draft model 또는 self-spec |
| Adaptive Flow Matching | Section 3.5 | velocity field delta threshold |
| ParoQuant (W4A8) | Section 3.6 | pairwise rotation + GPTQ |

**논문 링크**: https://openreview.net/pdf/a92c5f4b658a2a081b6924ca882edcf143741816.pdf
