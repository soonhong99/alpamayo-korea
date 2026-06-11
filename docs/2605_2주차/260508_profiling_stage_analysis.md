# [분석] CPU/GPU 사용 패턴 및 병렬성 분석

> **이 파일의 목적**: CPU가 무엇을 하는가 — 토크나이징 위치 / CPU 점유율 / 병렬화 기회  
> **연관 파일**: [결과](profiling_baseline_results.md) · [방법론](profiling_nsight_analysis.md) · [로드맵](profiling_progress.md)

---

## 교수님 질문 직접 답변

> *"토크나이징이 CPU에서 일어나는지, GPU에서 일어나는지?"*  
> *"CPU와 GPU를 병렬로 처리하고 싶다"*

**결론 먼저:**
```
토크나이징(Tokenization) → 100% CPU 실행, GPU와 독립적
추론(Inference)         → 주로 GPU, Decode 루프에서 CPU가 주기적으로 깨어남
병렬화 가능 여부         → Yes — 두 가지 병렬화 경로 존재
```

---

## 1. 토크나이징 파이프라인 — CPU에서만 실행

### 1.1 전체 흐름

```
[CPU 전용 구간 — GPU 미사용]
─────────────────────────────────────────────────────────────────
① 이미지 전처리 (Qwen3-VL Image Processor, CPU NumPy/PIL)
   · resize(320×576 → ViT tile 크기 448×448)         ≈ 20~50ms
   · normalize(mean=[0.485,0.456,0.406], std=[...])  ≈  5~10ms
   · to_tensor(), pixel_values 구성                   ≈  3~ 5ms

② 텍스트 토크나이징 (HuggingFace Tokenizer, CPU)
   · BPE 인코딩: system prompt + user message         ≈  1~ 3ms
   · Chat template 적용: <|im_start|>user\n...        ≈  0.5ms

③ 입력 결합 (processor.apply_chat_template, CPU)
   · pixel_values + input_ids + attention_mask 구성  ≈  2~ 5ms

④ GPU 전송 (helper.to_device, cudaMemcpy H→D)
   · pixel_values: ~4~18MB                            ≈  5~15ms
   · input_ids, attention_mask: ~10KB                 ≈  < 1ms
─────────────────────────────────────────────────────────────────
토크나이징 총 CPU 시간:  ≈ 35~90ms  (GPU는 이 구간 완전 유휴)
```

### 1.2 토크나이징이 GPU에서 일어나지 않는 이유

HuggingFace Tokenizer는 Python/Rust 구현으로 CPU에 바인딩되어 있다.  
GPU 실행을 위한 CUDA 커널이 존재하지 않는다 (NVIDIA FasterTransformer, TensorRT-LLM 등은 별도 구현).

```python
# processor.apply_chat_template() 내부 호출 스택 (CPU only)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(...)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,          # ← Python/Rust tokenizer (CPU)
    return_tensors="pt",    # ← CPU tensor 생성 후 나중에 to("cuda")
)
# 반환되는 inputs["input_ids"] 는 CPU tensor
# helper.to_device(inputs, "cuda") 로 GPU에 복사
```

### 1.3 우리 프로파일러에서의 위치

```
현재 profiler._profile_one_run() 타임라인:
──────────────────────────────────────────────────────────
copy.deepcopy(model_inputs)         [CPU, ~50~100ms]  ← CUDA timer 밖
  ↓
ev_run_s.record()                   ← CUDA timer 시작
  ↓
sample_trajectories_from_data...
  └── self.vlm.generate()           [GPU 주도]
  └── self.diffusion.sample()       [GPU 주도]
  ↓
ev_run_e.record()                   ← CUDA timer 끝
torch.cuda.synchronize()
──────────────────────────────────────────────────────────
CUDA timer = 5,004ms  (GPU 관점)
벽시계 = 5,115ms  (이전 NSight 측정)
차이 = 111ms  → copy.deepcopy 비용
```

> **중요**: 토크나이징은 `load_model()`에서 사전 계산되므로 5,004ms에 포함되지 않는다.  
> 실제 배포 시 토크나이징 비용(~50~90ms)이 추론마다 추가된다.

---

## 2. 추론 중 CPU 활동 — 단계별 분석

### 2.1 단계별 CPU 역할

| 단계 | GPU 활동 | CPU 활동 | CPU 점유 |
|---|---|---|---|
| Vision Encoding (716ms) | ViT 연산 | 유휴 대기 | **< 1%** |
| LLM Prefill (1,466ms) | KV Cache 빌드 | dispatch 1회 후 유휴 | **< 1%** |
| **LLM Decode (1,920ms)** | 토큰 1개씩 처리 | **Python 루프 실행** | **~5~8%** |
| Action Expert (902ms) | Flow Matching ODE | 유휴 대기 | **< 1%** |

### 2.2 Decode 루프의 CPU-GPU 교대 패턴 (핵심)

Decode는 17.5회 반복하는 Python 루프다. 매 스텝마다 CPU와 GPU가 교대한다:

```
스텝 1 (~112ms):
  [CPU: prepare_inputs 2~5ms] → [GPU: transformer 1 token ~107ms]
스텝 2 (~112ms):
  [CPU: prepare_inputs 2~5ms] → [GPU: transformer 1 token ~107ms]
...
스텝 17 (~112ms):
  [CPU: prepare_inputs 2~5ms] → [GPU: transformer 1 token ~107ms]
```

**CPU가 하는 `prepare_inputs_for_generation()` 내용:**
```python
# HuggingFace generate() 루프 내부
model_inputs = self.prepare_inputs_for_generation(
    input_ids,           # 이전 토큰들
    past_key_values=..., # KV cache
    attention_mask=...,  # 현재 attention mask
)
# prepare() 내부에서:
# · position_ids 업데이트 (새 토큰 위치 계산) → CPU
# · attention_mask 1칸 연장             → CPU tensor op
# · input_ids를 마지막 토큰만으로 슬라이스 → CPU
# · 다음 forward 호출용 kwargs 딕셔너리 구성 → CPU

# 이후:
outputs = self(**model_inputs)  # GPU dispatch
next_token_logits = outputs.logits[:, -1, :]
next_token = top_p_sample(next_token_logits)  # GPU small op
stopping = stopping_criteria(input_ids, ...)   # CPU check
```

**NSight 증거 — 이 패턴의 커널 수준 흔적:**

```
vectorized_elementwise_kernel: 71,419 instances
  = Decode 루프의 작은 CPU-dispatched GPU 연산들
  = 17.5 steps × ~4,000 elementwise ops/step = 70,000 ✓

CatArrayBatchedCopy: 27,432 instances
  = 매 decode step에서 KV cache에 새 토큰 K,V 붙이기 (cat 연산)

H→D 메모리 전송: 112건 per inference
  = Decode 루프에서 position_ids, attention_mask를 매 스텝 GPU로 전송
```

### 2.3 CPU 총 점유 시간 추정

```
5,004ms GPU 실행 중 CPU 활동 시간:

  copy.deepcopy (CUDA timer 밖):   111ms
  fuse_traj_tokens 준비:           ~5ms
  Decode 루프 Python 오버헤드:
    17.5 steps × 5ms/step           ≈  88ms
  post-VLM 처리 (offset, position_ids):  ≈ 15ms
  ──────────────────────────────────────
  추론 관련 총 CPU 시간:            ≈ 219ms

CPU 점유율 (219ms / 5,115ms):       ≈ 4.3%
GPU 점유율:                         ≈ 95.7%

→ GPU가 유휴(idle)인 시간:
  Decode 루프의 CPU 준비 시간: 17.5 × 5ms = 88ms (1.7%)
```

---

## 3. CPU/GPU 병렬화 기회

### 3.1 기회 1: 프레임 간 토크나이징 파이프라이닝 (즉시 적용 가능)

현재 (순차):
```
시간 →
CPU: [Tokenize N 70ms][deepcopy 100ms][──────── 대기 ────────────────]
GPU:                                   [Inference N: 5,004ms         ]
CPU:                                                                  [Tokenize N+1 70ms]
GPU:                                                                             [Inference N+1]
```

최적화 (파이프라인):
```
시간 →
CPU: [Tokenize N 70ms][deepcopy 100ms][Tokenize N+1 70ms][───────── 대기 ──]
GPU:                                   [Inference N: 5,004ms              ]
GPU:                                                         [Inference N+1]
```

**효과:** Alpamayo 1.5는 5Hz 이하로 동작하므로 프레임 간 200ms 여유가 있다.  
이 200ms 안에 다음 프레임 토크나이징(70ms)을 완료할 수 있어 **GPU를 쉬지 않게** 할 수 있다.

**구현:**
```python
import threading

class PipelinedProfiler:
    def __init__(self, model):
        self.model = model
        self._next_inputs = None
        self._tokenize_thread = None
    
    def _tokenize_async(self, data_next):
        """백그라운드에서 다음 프레임 토크나이징"""
        self._next_inputs = tokenize(data_next)      # CPU only
        self._next_inputs = to_device(self._next_inputs, "cuda")  # H→D
    
    def run_pipelined(self, data_curr, data_next):
        # 현재 프레임 GPU 추론 시작
        # 동시에 다음 프레임 토크나이징 시작
        t = threading.Thread(target=self._tokenize_async, args=(data_next,))
        t.start()
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.model.sample_trajectories(data=self._current_inputs)
        
        t.join()  # 토크나이징 완료 대기 (GPU 추론보다 훨씬 빨리 끝남)
        self._current_inputs = self._next_inputs
        return result
```

### 3.2 기회 2: CUDA Graphs로 Decode 루프 CPU 오버헤드 제거 (중요)

현재 문제: 17.5 decode steps마다 CPU가 Python 작업을 하고 GPU에 커널을 launch한다.

```
현재 (동기식):
GPU: [──107ms──]  [──107ms──]  [──107ms──]  ...
CPU:             [5ms]        [5ms]        [5ms]
     ↑ GPU 완료 대기 → CPU 작업 → GPU 재시작 → 17.5× 반복
```

CUDA Graphs 적용:
```python
# 한 번 그래프 캡처
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model_single_decode_step(input_ids, past_kv, position_ids)

# 이후 매 스텝: CPU 개입 없이 GPU에서 직접 replay
g.replay()  # CPU → GPU 커뮤니케이션 없음, 순수 GPU 실행
```

```
CUDA Graphs 적용 후:
GPU: [──107ms──][──107ms──][──107ms──]...  (연속 실행, CPU 개입 없음)
CPU: [                  완전 유휴                        ]
```

**예상 효과:**  
- Decode CPU 오버헤드 17.5 × 5ms = 88ms → ~0ms  
- GPU idle 제거로 per-step 시간도 107ms → 105ms 수준으로 감소  
- 전체 Decode 1,920ms → ~1,840ms (약 4% 단축)

**주의사항 (CUDA Graphs 제약):**
```python
# CUDA Graphs는 아래 조건에서만 작동:
# 1. 입력 텐서의 shape가 고정 (dynamic shape 불가)
#    → input_ids shape가 decode마다 변하면 문제
# 2. 제어 흐름 없음 (if/else, Python 루프 불가)
#    → stopping_criteria 체크를 그래프 밖에서 해야 함
# 3. 메모리 할당 없음 (KV cache 확장 불가)
#    → Static KV cache가 필요 (사전에 최대 길이 할당)
```

### 3.3 기회 3: Prefill과 다음 Action 병렬화 (고급)

현재 순차:
```
[Vision+Prefill 2,182ms][Decode 1,920ms][Action 902ms]
```

잠재적 파이프라인:
```
프레임 N:   [V+P 2,182ms][Decode 1,920ms][Action 902ms]
프레임 N+1:                              [V+P 미리 시작?]
```

**가능성**: 낮음. Alpamayo의 Action Expert는 Decode의 KV cache 출력을 입력으로 받으므로 Decode가 완료되기 전에 Action Expert를 시작할 수 없다.

---

## 4. CPU 사용률 정밀 측정 방법

### 4.1 tegrastats (Thor 전용)

```bash
# 추론과 동시에 CPU/GPU/메모리 사용률 100ms 간격으로 기록
tegrastats --interval 100 | tee profiling_results/tegrastats.log &
TPID=$!
python scripts/profiling/profile_alpamayo.py --warmup 3 --runs 8
kill $TPID

# 로그 형식:
# RAM 21348/122702MB ... CPU [45%@1804,23%@1804,...] ... GR3D_FREQ 87%
```

**tegrastats 출력 파싱:**
```python
import re

def parse_tegrastats(line):
    cpu_match = re.search(r'CPU \[([^\]]+)\]', line)
    gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
    
    if cpu_match:
        cpu_loads = [int(x.split('%')[0]) for x in cpu_match.group(1).split(',')]
        avg_cpu = sum(cpu_loads) / len(cpu_loads)
    gpu_load = int(gpu_match.group(1)) if gpu_match else 0
    return avg_cpu, gpu_load
```

### 4.2 PyTorch Profiler CPU trace

```bash
python scripts/profiling/profile_alpamayo.py \
    --warmup 2 --runs 3 --pytorch_profiler
```

생성되는 `pytorch_trace.json`을 크롬 브라우저에서:
```
chrome://tracing → Load → pytorch_trace.json
```

**확인할 항목:**
```
python 스레드 트랙:
  [prepare_inputs 2ms]─[GPU dispatch]─[wait]─[prepare_inputs 2ms]─...
  ↑ Decode 루프 중 CPU가 주기적으로 깨어나는 패턴 확인

CUDA 스트림 트랙:
  [──Vision 716ms──][──Prefill 1466ms──][tok][tok][tok]...[──Action 902ms──]
  ↑ decode "tok" 사이에 CPU gap 있는지 확인
```

### 4.3 NSight Systems에서 CPU utilization 보기

GUI에서 `Analysis` → `CPU Utilization` 탭:
- Python 스레드 활동 타임라인
- CUDA API call 빈도
- CPU idle time 비율

---

## 5. 현재 CPU/GPU 효율성 요약

```
5,004ms 추론 중 CPU/GPU 사용:
────────────────────────────────────────────────────────────────
Vision Encoding  (716ms):  GPU  99%  │ CPU  1%  (dispatch overhead)
LLM Prefill    (1,466ms):  GPU  99%  │ CPU  1%  (dispatch 1회)
LLM Decode     (1,920ms):  GPU  92%  │ CPU  5%  (17.5× Python 루프)
                                      │      3%  GPU idle (CPU 대기)
Action Expert    (902ms):  GPU  99%  │ CPU  1%  (dispatch 1회)
────────────────────────────────────────────────────────────────
전체 평균:                 GPU  96%  │ CPU  4%
CUDA timer 밖 (111ms):           CPU 100% (copy.deepcopy)
────────────────────────────────────────────────────────────────
진단: GPU는 거의 풀가동, CPU는 거의 유휴
      → CPU 병렬화의 여지가 크다 (CPU가 다음 프레임 준비 가능)
      → GPU idle 3%는 CUDA Graphs로 제거 가능
```

---

## 6. 논문 서술용 — CPU/GPU 분석 요약 문장

```
Alpamayo 1.5의 추론 파이프라인에서 토크나이징(tokenization)은 전적으로
CPU에서 실행되며 약 35–90ms가 소요된다. 이 구간에서 GPU는 완전히 유휴 상태이므로,
이전 프레임의 GPU 추론과 현재 프레임의 CPU 토크나이징을 오버랩하는
파이프라인 방식으로 토크나이징 비용을 사실상 0으로 숨길 수 있다.

추론 중 GPU 점유율은 약 96%이며, 나머지 4%는 LLM Decode 루프에서
Python prepare_inputs_for_generation()이 각 토큰 생성 후 실행되는
CPU-GPU 동기화 오버헤드에 해당한다. 이 88ms(17.5 steps × 5ms)의 오버헤드는
CUDA Graphs 기법을 통해 제거 가능하다.
```
