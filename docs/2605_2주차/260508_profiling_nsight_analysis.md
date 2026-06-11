# [방법론] 프로파일링 코드 설계 및 계측 원리

> **이 파일의 목적**: 어떻게 측정했는지 — 코드 설계 / 계측 원리 / NSight 사용법  
> **연관 파일**: [결과](profiling_baseline_results.md) · [CPU/GPU 분석](profiling_stage_analysis.md) · [로드맵](profiling_progress.md)

---

## 1. 프로파일링 코드 아키텍처 (`profile_alpamayo.py v2.0`)

### 1.1 왜 단순 타이머로는 측정이 불가능한가

Alpamayo 1.5 추론은 단일 함수 `sample_trajectories_from_data_with_vlm_rollout()` 안에 모든 단계가 숨어있다:

```python
def sample_trajectories_from_data_with_vlm_rollout(self, data, ...):
    # 내부에 4단계가 순차 실행됨
    ...
    vlm_outputs = self.vlm.generate(...)   # Vision + Prefill + Decode (불투명)
    ...
    sampled_action = self.diffusion.sample(...)  # Action Expert (불투명)
```

이 함수 밖에서 타이머를 감싸면 **전체 시간만** 나오고 단계 분리가 불가능하다 (v1의 실패 원인).  
단계 분리를 하려면 함수 **내부**에 계측 코드를 주입해야 한다.

### 1.2 3계층 계측 설계

라이브러리 소스코드를 직접 수정하지 않고 Python의 동적 기능으로 내부에 측정 프로브를 삽입한다.

```
┌─────────────────────────────────────────────────────────────────┐
│  sample_trajectories_from_data_with_vlm_rollout()               │
│                                                                 │
│  Layer 1: model.vlm.generate() monkey-patch                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ [ev_vlm_s]                                               │   │
│  │                                                          │   │
│  │  Layer 2: model.vlm forward_pre/post hook                │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ [ev_pre_s]                                         │  │   │
│  │  │                                                    │  │   │
│  │  │  Layer 3: model.vlm.model.visual forward hook      │  │   │
│  │  │  ┌──────────────────────────────────────────────┐  │  │   │
│  │  │  │ [ev_vis_s] --- ViT 실행 --- [ev_vis_e]       │  │  │   │
│  │  │  └──────────────────────────────────────────────┘  │  │   │
│  │  │                                                    │  │   │
│  │  │  [LLM attention layers (Prefill)]                  │  │   │
│  │  │ [ev_pre_e]                                         │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                                                          │   │
│  │  [decode steps × N: ev_pre_e ~ ev_vlm_e]                │   │
│  │ [ev_vlm_e]                                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Flow Matching: ev_vlm_e ~ ev_run_e]                           │
└─────────────────────────────────────────────────────────────────┘
```

**단계별 시간 계산:**

| 단계 | 계산식 | 측정 방식 |
|---|---|---|
| Vision Encoding | `ev_vis_e − ev_vis_s` | 직접 측정 |
| LLM Prefill (순수) | `(ev_pre_e − ev_pre_s) − Vision` | 직접 측정 |
| LLM Decode | `(ev_vlm_e − ev_vlm_s) − Prefill` | 유도 |
| Action Expert | `(ev_run_e − ev_run_s) − VLM` | 유도 |

---

## 2. Layer 1 — `model.vlm.generate()` Monkey-Patch

### 2.1 코드

```python
# AlpamayoStagePatch.attach() 내부

self._orig_generate = model.vlm.generate   # 원본 함수 저장
_patch = self

def _patched_generate(*args, **kwargs):
    _patch._new_events()           # 이번 런용 CUDA Event 생성
    _patch.ev_vlm_s.record()       # GPU stream에 start 마커 삽입
    nvtx.range_push("vlm_generate")
    
    result = _patch._orig_generate(*args, **kwargs)  # 원본 실행
    
    if _patch._decode_nvtx_live:   # decode NVTX 닫기
        nvtx.range_pop()
    _patch.ev_vlm_e.record()       # GPU stream에 end 마커 삽입
    nvtx.range_pop()               # "vlm_generate"
    _patch._inside_generate = False
    return result

model.vlm.generate = _patched_generate  # 교체
```

### 2.2 Monkey-Patch를 선택한 이유

| 방법 | 장점 | 단점 | 채택 여부 |
|---|---|---|---|
| 소스 직접 수정 | 정확 | 라이브러리 변경 → 버전 관리 복잡 | ❌ |
| `nn.Module.register_forward_hook` | 표준 API | `generate()`는 Module이 아님 | ❌ |
| **Monkey-Patch** | 소스 불변, 런타임 주입 | Python 레벨 오버헤드 (< 1μs) | ✅ |
| NVTX Python API | NSight 연동 | 단독으로는 시간 측정 불가 | 보조 사용 |

### 2.3 CUDA Event vs time.perf_counter

```
time.perf_counter():  CPU 벽시계
  ✗ Python GIL, OS 스케줄러 지터 포함
  ✗ GPU 비동기 실행과 동기화 필요
  ✗ synchronize() 호출 위치에 따라 값이 달라짐

torch.cuda.Event(enable_timing=True):  GPU stream 타임스탬프
  ✓ CUDA stream에 마커를 직접 삽입
  ✓ GPU 실행 순서 기준 (CPU 지터 없음)
  ✓ elapsed_time()이 두 마커 사이의 GPU 실행 시간을 정확히 반환
  ✓ 비동기 — 측정 자체가 추론에 영향을 주지 않음
```

**오차 수준:**
- CUDA Event: ± 0.5μs (하드웨어 클럭 기준)
- `time.perf_counter` + synchronize: ± 1~5ms (OS 스케줄러)

---

## 3. Layer 2 — `model.vlm` Forward Hook (Prefill/Decode 구분)

### 3.1 Prefill vs Decode 감지 원리

HuggingFace `generate()` 내부 루프:

```python
# generate() 내부 (HuggingFace 구현)
while not stopping_criteria(input_ids):
    model_inputs = self.prepare_inputs_for_generation(input_ids, ...)
    outputs = self(**model_inputs)   # ← 이 __call__에 hook이 발화
    next_token = sample(outputs.logits)
    input_ids = cat([input_ids, next_token])
```

**첫 번째 호출 (Prefill)**: `model_inputs`에 `pixel_values=<tensor>` 포함  
**이후 호출 (Decode)**: `model_inputs`에 `pixel_values=None` (이미 KV 캐시에 저장됨)

### 3.2 코드

```python
def _vlm_pre(module, args, kwargs):
    if not _patch._inside_generate:
        return
    
    pixel_values = kwargs.get("pixel_values", None)
    past_kv      = kwargs.get("past_key_values", None)
    
    # 감지 조건: pixel_values 있음 OR KV cache 비어있음 = Prefill
    is_prefill = (
        pixel_values is not None
        or past_kv is None
        or (hasattr(past_kv, "get_seq_length") and past_kv.get_seq_length() == 0)
    )
    
    if is_prefill and not _patch._prefill_done:
        _patch.ev_pre_s.record()
        nvtx.range_push("llm_prefill")
    elif not is_prefill:
        if not _patch._decode_nvtx_live:
            nvtx.range_push("llm_decode")
            _patch._decode_nvtx_live = True
        _patch._decode_step_count += 1

def _vlm_post(module, args, output):
    if not _patch._inside_generate or _patch._prefill_done:
        return
    _patch.ev_pre_e.record()
    nvtx.range_pop()              # "llm_prefill"
    _patch._prefill_done = True   # 이후 호출은 decode로 처리
```

### 3.3 `with_kwargs=True` 의 필요성

```python
# PyTorch 2.0+ API
model.vlm.register_forward_pre_hook(_vlm_pre, with_kwargs=True)
#                                              ↑
# 이 옵션 없으면 hook(module, args) → kwargs 접근 불가
# HuggingFace는 pixel_values를 keyword argument로 전달하므로
# with_kwargs=True 없이는 pixel_values 감지 자체가 불가능
```

---

## 4. Layer 3 — Vision Encoder Forward Hook

### 4.1 속성 경로 자동 탐색

Qwen 계열 VLM은 버전마다 Vision Encoder 속성 이름이 다르다:

```python
_VISUAL_PATHS = [
    "model.visual",       # Qwen2.5-VL 표준 (우리 모델 ← 이 경로로 발견됨)
    "visual",
    "model.vision_model",
    "vision_model",
    "vision_tower",
]

visual = _find_submodule(model.vlm, _VISUAL_PATHS)
# → model.vlm.model.visual 발견 확인
# [Patch] 등록 완료: {'visual_path': 'model.visual', ...}
```

### 4.2 Vision hook이 중요한 이유

Vision Encoding은 LLM Prefill의 첫 번째 forward pass **내부**에서 실행된다.  
Layer 2 hook만으로는 "Prefill 총 시간"만 알 수 있고, 그 안에서 ViT가 얼마나 쓰는지 모른다.

```
Layer 2 (Prefill hook):
  ev_pre_s ─────────────────────────────── ev_pre_e
            [vision enc.][LLM attention][LLM MLP]
                  ↑
            Layer 3 (Vision hook):
            ev_vis_s ── ev_vis_e
            
→ 순수 LLM Prefill = ev_pre_e − ev_pre_s − (ev_vis_e − ev_vis_s)
                   = 1,466ms − 716ms = 750ms
```

---

## 5. `torch.autocast` 필요성 — 버그와 수정

### 5.1 오류 발생 경로

```python
# flow_matching.py 내부 (_euler 메서드)
def _euler(self, ...):
    x = torch.randn(B, T, D)            # dtype=float32 (PyTorch 기본값)
    for t in timesteps:
        v = step_fn(x=x, t=t)           # x가 float32인 채로 호출
            → action_in_proj(x, t)
              → self.trunk(x)           # Linear 레이어
                → F.linear(x_float32, W_bfloat16)  # ← CRASH
```

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

### 5.2 수정

```python
# _profile_one_run() 내부
with torch.autocast("cuda", dtype=torch.bfloat16):   # ← 이 한 줄이 핵심
    pred_xyz, pred_rot, extra = \
        self.model.sample_trajectories_from_data_with_vlm_rollout(...)
```

`torch.autocast`는 linear, matmul, conv 등 주요 연산 직전에 입력을 자동으로 지정 dtype으로 캐스팅한다.  
이로써 `torch.randn()`이 float32를 반환해도 Linear에 들어가는 순간 bfloat16으로 변환된다.

---

## 6. NVTX 마커 계층 구조

NSight Systems에서 올바르게 보이는 NVTX 트리:

```
[measure_run_N]                         ← 프로파일러 루프
  [alpamayo_full_inference]             ← Layer 0: 전체 추론
    [vlm_generate]                      ← Layer 1: model.vlm.generate() 
      [llm_prefill]                     ← Layer 2: 첫 번째 vlm forward
        [vision_encoding]               ← Layer 3: ViT forward
      [llm_decode]                      ← Layer 2: 이후 vlm forward × N
    (unmapped gap = flow matching)      ← ev_vlm_e ~ ev_run_e
```

**Flow Matching에 NVTX 마커가 없는 이유**:  
`diffusion.sample()` 호출은 `sample_trajectories` 내부에 있고, 소스를 수정하지 않고는  
generate() 반환 후 diffusion.sample() 전에 마커를 삽입할 위치가 없다.  
대신 NSight에서 `[vlm_generate end] ~ [alpamayo_full_inference end]` 사이의 GPU 활동이 Flow Matching에 해당한다.

---

## 7. NSight Systems GUI 사용법 — 확인해야 할 지점

### 7.1 파일 열기

```bash
# Thor에서 로컬로 복사
scp ice401@100.95.177.101:~/alpamayo1.5/profiling_results/nsight_stage.nsys-rep \
    /mnt/c/Users/nanay/Desktop/

# GUI에서: File → Open → nsight_stage.nsys-rep
```

### 7.2 각 트랙에서 봐야 할 것

```
트랙                   확인 포인트
─────────────────────────────────────────────────────────────────
NVTX                   [vlm_generate] 안에 [llm_prefill][llm_decode] 보이는지
CUDA HW                초록색(커널)과 흰색(idle) 패턴 — idle이 많으면 CPU bottleneck
CPU Threads            Python 스레드가 언제 활동하는지 (Decode 루프 중 주기적 스파이크)
Memory                 H→D 전송 빈도 (Decode 루프마다 작은 전송이 반복되는지)
```

### 7.3 CLI 텍스트 요약 (Thor에서)

```bash
# NVTX 단계별 시간 — 가장 중요
nsys stats --report nvtx_sum --format csv \
    profiling_results/nsight_stage.nsys-rep

# CUDA 커널 요약
nsys stats --report cuda_gpu_kern_sum --format csv \
    profiling_results/nsight_stage.nsys-rep | head -20

# 메모리 전송 패턴
nsys stats --report cuda_gpu_mem_time_sum --format csv \
    profiling_results/nsight_stage.nsys-rep
```

---

## 8. 프로파일러 실행 방법

```bash
# 기본 실행 (워밍업 3회, 측정 8회)
cd ~/alpamayo1.5 && source a1_5_venv/bin/activate
python scripts/profiling/profile_alpamayo.py --warmup 3 --runs 8

# NSight 포함 (NVTX 타임라인 생성)
./scripts/profiling/run_stage_profile.sh

# PyTorch Profiler + Chrome trace (크롬에서 chrome://tracing으로 열기)
python scripts/profiling/profile_alpamayo.py --warmup 2 --runs 3 --pytorch_profiler
```

**출력 파일:**
```
profiling_results/
├── raw_timings.json      ← 런별 모든 타이밍 (JSON)
├── summary.json          ← 통계 (mean/std/p50/p95/p99)
├── stage_breakdown.csv   ← 단계별 CSV (matplotlib 입력용)
└── pytorch_trace.json    ← Chrome trace (--pytorch_profiler 옵션 시)
```
