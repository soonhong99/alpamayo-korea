# AGENTS.md — Project Context for AI Assistants

This file gives AI coding assistants (Codex, Copilot, etc.) the context needed to contribute effectively to this project.

---

## Project Summary

**★★★ 연구 방향 전환 (2026-05-24 교수님 미팅) ★★★**

**이전 방향 (폐기)**: Korean fine-tuning, RL post-training, 모델 수정  
**새 방향 (확정)**: **모델 수정 없이 Async Pipeline + iGPU 메모리 최적화로 Alpamayo throughput 극대화**

핵심 목표:
- RT-Swap (RTAS'24) + Demand Layering (RTSS'22) 아이디어를 Alpamayo에 적용
- `cudaMemPrefetchAsync` + CUDA Stream 이중화로 layer prefetch와 compute 중첩
- GPU L2 Persistent Residency로 KV Cache를 L2에 고정 (4.76× DRAM 대비 속도)
- 65-step autoregressive decode의 layer-level async pipeline 구현

참고 문서: `docs/260524_01_교수님_미팅_연구방향_전환.md`

---

**[구 요약 — 참고용으로 보존]**  
**Alpamayo-Korea** deployed NVIDIA's Alpamayo 1.5 on Jetson AGX Thor. The original Korean fine-tuning direction has been superseded; system-level optimization is now the focus.

---

## Architecture Overview

```
[Korean Datasets]          [AlpaSim]
  AI Hub                     |
  Kakao Mobility    -->  [Scenario Config YAMLs]
  42dot                      |
  ETRI              -->  [Baseline Eval: Alpamayo 1.5]
                             |
                    [RL Fine-tuning Pipeline]
                             |
                    [Alpamayo-Korea Checkpoint]
                             |
                    [Jetson AGX Thor Deployment]
                             |
                    [Korean Reasoning Output]
```

---

## Key Files to Understand First

1. `configs/alpasim_korea.yaml` — All Korean scenario simulation parameters
2. `scenarios/korea/*.yaml` — Individual scenario definitions (traffic patterns, agent behaviors)
3. `scripts/run_baseline_eval.sh` — How baseline evaluation works end-to-end
4. `evaluation/metrics.py` — Custom metrics for Korean scenario performance
5. `scripts/run_thor_inference.py` — Edge deployment entry point

---

## Model Details

- **Base**: Alpamayo 1.5 (`nvidia/Alpamayo-1.5-10B` on HuggingFace)
- **Backbone**: Cosmos Reason2 (8.2B) + Action Expert (2.3B)
- **Input**: Multi-camera video + egomotion history (+ navigation in 1.5)
- **Output**: 6.4s trajectory (64 waypoints) + Korean reasoning trace
- **License**: Non-commercial research only

### Fine-tuning approach
We use RL post-training following the Cosmos Cookbook recipes. The key addition is:
- Korean-language reasoning trace supervision
- Scenario-weighted sampling (overrepresent rare Korean edge cases)
- AlpaSim closed-loop feedback as reward signal

---

## Dataset Pipeline

### AI Hub (aihub.or.kr)
- Registration required → approval in 1–3 days
- Download via their API after approval
- Relevant datasets: `#188` (신호등·표지판), `#도로주행`
- Format: Image + JSON annotation (COCO-style bounding boxes)

### Kakao Mobility (ETRI AI 나눔)
- URL: `https://nanum.etri.re.kr`
- Free, no copyright restrictions
- 150K samples: 3D dynamic objects + 2D static objects
- Format: LiDAR point clouds + camera images + labels

### NVIDIA Physical AI AV Dataset
- HuggingFace: `nvidia/PhysicalAI-Autonomous-Vehicles`
- Requires accepting gated dataset license
- Used for: loading into AlpaSim as reconstructed scenes (NuRec format)
- Korean scenes are sparse — use as baseline comparison only

---

## AlpaSim Architecture (important for scenario writing)

AlpaSim uses a **microservice architecture via Docker Compose**:

```
Runtime (orchestrator)
  |-- Driver service       <- Alpamayo model lives here
  |-- Renderer service     <- Generates sensor data from scene
  |-- TrafficSim service   <- Controls other vehicles/agents
  |-- Controller service   <- Executes planned trajectory
  +-- Physics service      <- Vehicle dynamics
```

Each service communicates via **gRPC**. To add a Korean scenario:
1. Define agent behaviors in `scenarios/korea/*.yaml`
2. Register in `configs/alpasim_korea.yaml`
3. AlpaSim will reconstruct it using NuRec scene data

---

## Jetson Thor Deployment Notes

- **JetPack version**: 7 (Ubuntu 24.04 LTS, Linux kernel 6.8, CUDA 13.0, SM 11.0)
- **Inference stack**: TensorRT + NVIDIA AI stack
- **Key flag**: `--dtype bf16` (기본, 실측 ~4.8s/inference). `--dtype int4` (bitsandbytes NF4, KV cache BF16 유지, 즉시 사용 가능). 진짜 FP4 (2,070 TFLOPS)는 TensorRT-LLM 엔진 변환 필요 — 단순 dtype 플래그 불가 (2026-05-19 확인)
- **Latency target**: 전체 추론(prefill+decode) 기준 — 1단계 ~3,500ms, 최종 99ms
  - **현재 실측 (sdpa, BF16, N=1)**: 4,838ms = VE 728ms + LM Prefill 1,423ms + Decode 1,818ms (17steps×107ms) + Flow 870ms
  - ⚠️ 이전 기록 "6,554ms (prefill 4,487ms + decode 2,067ms)"는 eager+StaticCache 커스텀 경로 수치임 — 전체 파이프라인 기준이 아님
  - BF16 물리 하한 (대역폭 한계): decode 1step 최솟값 86ms (22GB÷231GB/s), 17step = 1,462ms
  - 1단계 (~3,500ms): torch.compile + KV Temporal Reuse (시스템만, BF16)
  - 2단계 (~1,500ms): Speculative Decoding + Inter-Inference Pipeline
  - 최종 (99ms): Rolling Trajectory 연속 스트리밍 파이프라인으로 재정의 (단일 추론 99ms는 물리적 불가)
  - ⚠️ 이전 기록 "≤100ms per inference step"은 오기(誤記) — per step이 아니라 전체 추론 기준임
- **Memory**: 128GB shared CPU/GPU — no OOM risk with 10B model
- **Korean output**: Pass `--lang ko` to `run_thor_inference.py`
- **Python**: 3.12.13 (via uv), venv at `~/alpamayo1.5/a1_5_venv/`

### PyTorch on Thor — Must Build from Source

**PyTorch 2.8.0은 Thor(CUDA 13.0 / SM 11.0 / aarch64)용 공식 바이너리가 없다.**
반드시 소스에서 빌드해야 한다. 전체 가이드: `docs/thor_pytorch_build.md`

빌드 요약:
```bash
git clone --recursive https://github.com/pytorch/pytorch ~/pytorch
cd ~/pytorch && git checkout v2.8.0
# CUDA 13.0 호환 패치 적용 (docs/thor_pytorch_build.md 참고)
USE_TENSORPIPE=0 USE_DISTRIBUTED=0 USE_MPI=0 \
  TORCH_CUDA_ARCH_LIST="11.0" MAX_JOBS=8 \
  python setup.py build
python setup.py develop
```

빌드 시간: ~6시간 (Thor 단독, MAX_JOBS=8)

### Known issues on Thor
- **attn_implementation 규칙** (2026-05-28 최종 확정):
  - **결론**: `attn_implementation` 미지정(sdpa 기본값) + DynamicCache(기본값)가 유일한 최선 조합.
  - **eager 사용 금지**: seq_len=3086에서 LM Prefill이 1,423ms→3,753ms (2.6× 느려짐). 어떤 경로에서도 쓰지 말 것.
  - **StaticCache 사용 금지**: sdpa+StaticCache 조합은 ValueError 없이 동작하지만, StaticCache가 float 4D attention bias를 생성 → Flash Attention 백엔드 강제 비활성화 → MemEfficientAttn으로 강등 → prefill 1,423ms→2,839ms (2× 느려짐). 전체 파이프라인 환산 시 4,838ms→6,674ms로 오히려 퇴보.
  - **attn 백엔드 3계층**: FlashAttention(최속, attn_mask=None or bool) > MemEfficientAttn(float mask 지원, ~2×) > Math(eager동등, 최저속)
  - ⚠️ 구 기록 "sdpa+StaticCache → ValueError (2026-05-19)"는 틀린 정보. ValueError는 없음. 느려서 사용 안 하는 것.
  - **3중 비교 요약** (LM+Decode 커스텀 경로만): eager+Static=6,554ms / sdpa+Static=5,076ms / sdpa+Dynamic(추정)=3,241ms
- **torch.compile 실험 결과 (2026-05-28 확정): ❌ 비호환**
  - `reduce-overhead` (Inductor 경로): Triton 3.7.0 ↔ PyTorch 2.8.0 소스 API 연쇄 불일치
    - `triton_key` 삭제됨, `cluster_dims` 삭제됨 등 근본적 API 재설계
    - aarch64용 구버전 Triton wheel 없어 다운그레이드 불가
  - `cudagraphs` 경로: Qwen3VL `_deepstack_process`의 dynamic boolean indexing(`visual_pos_masks`)이 CUDA Graph 캡처와 근본 비호환 → `capture_end()` crash
  - **결론**: 모델 아키텍처 제약으로 어떤 torch.compile 모드도 작동하지 않음. 재시도 불필요.
- First boot model loading takes ~3–4 min (22GB weights)
- Use `MIG` mode if running parallel experiments
- **pip list에 torch가 안 보여도** site-packages에 CPU wheel이 숨어있을 수 있음 → `python -c "import torch; print(torch.__file__)"` 로 확인
- CUDA 13.0은 CUB iterator API를 CCCL로 통합하면서 다수 제거 → PyTorch 2.8.0 소스 패치 필요 (상세 내용: `docs/thor_pytorch_build.md`)

---

## Coding Conventions

- **Python**: 3.10+, type hints everywhere, follow existing style with `flake8`
- **Config files**: YAML, use Hydra overrides where possible (matches AlpaSim convention)
- **Scripts**: Bash for setup/download, Python for everything model-related
- **Logging**: Use Python `logging` module, not print statements
- **Docstrings**: Google-style
- **Tests**: Add pytest tests in `tests/` for any new evaluation metrics

---

## What NOT to do

- Do not commit model weights or dataset files (they're in `.gitignore`)
- Do not hardcode HF tokens — always read from `HF_TOKEN` env var
- Do not modify `configs/alpasim_base.yaml` — extend in `alpasim_korea.yaml`
- Do not push large video files — use Git LFS or link to external storage

---

## Research Context

This project positions as the **first public Korean road adaptation of a reasoning-based VLA model**. The audience includes:
- Korean AV startups (Rainbow Robotics, 42dot, Kakao Mobility)
- Global AV companies entering Korea (Pony.ai KR, WeRide KR)
- Academic reviewers (IROS, ICRA, ICCV workshops)

The novelty claim: **"First closed-loop benchmark of Alpamayo on Korean long-tail scenarios + first Korean-language reasoning trace deployment on Jetson AGX Thor edge hardware."**

---

## Git Commit & Push Guidelines

이 프로젝트에서 작업할 때는 아래 기준에 따라 커밋과 푸시를 권장한다.

### 즉시 커밋해야 하는 경우
- 시나리오 YAML 파일(`scenarios/korea/*.yaml`) 신규 추가 또는 수정 완료 시
- 평가 메트릭(`evaluation/metrics.py`, `evaluation/reasoning_eval.py`) 변경 시
- 스크립트(`scripts/`) 실제 실행 검증 완료 후
- 실험 결과(`evaluation/results/*.json`) 새로 생성됐을 때
- 이 AGENTS.md 파일에 변경사항이 생겼을 때 — **반드시 커밋+푸시**

### 커밋 메시지 형식
```
<type>: <short description>

# type 목록:
# feat     - 새 시나리오, 새 스크립트, 새 기능
# eval     - 실험 결과, 벤치마크 업데이트
# fix      - 버그 수정
# docs     - 문서 업데이트 (README, AGENTS.md 포함)
# config   - YAML 설정 변경
# chore    - 환경 셋업, 의존성 변경
```

예시:
```
feat: add narrow_alleyway scenario with pedestrian emergence events
eval: baseline results on horizontal_traffic_light (completion 40%)
docs: update AGENTS.md with commit guidelines
```

### 푸시 주기
- 하루 작업 마칠 때 그날 커밋들을 한 번에 push
- 실험 결과가 나왔을 때 즉시 push (재현 가능성 보장)
- **AGENTS.md 변경 시 항상 즉시 push** — AI 어시스턴트가 최신 컨텍스트를 읽어야 하기 때문

### 커밋하지 말아야 하는 것
- 모델 가중치 (`checkpoints/`, `*.safetensors`, `*.bin`)
- 데이터셋 파일 (`data/*/raw/`, `data/*/images/`)
- `.env`, `HF_TOKEN` 등 크리덴셜
- AlpaSim 런타임 로그 (`wizard_logs/`, `*.log`)

---

## Open Questions / TODO

See GitHub Issues for current open questions. Key unresolved items:
1. How to align AI Hub annotation format with AlpaSim NuRec format
2. Best RL reward shaping for Korean jaywalking scenarios
3. Whether to submit to ICRA 2026 or a dedicated AV workshop
