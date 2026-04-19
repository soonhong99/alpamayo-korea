# CLAUDE.md — Project Context for AI Assistants

This file gives AI coding assistants (Claude, Copilot, etc.) the context needed to contribute effectively to this project.

---

## Project Summary

**Alpamayo-Korea** fine-tunes NVIDIA's Alpamayo 1.5 (a 10B-parameter VLA model for autonomous driving) on Korean road scenarios and deploys it on NVIDIA Jetson AGX Thor for real-time edge inference with Korean-language reasoning traces.

The core insight: Alpamayo is trained on global data but Korean road environments have distinct characteristics (horizontal traffic lights, bus-only lanes, narrow alleyways, riding culture) that cause systematic model failures. This project bridges that gap.

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

- **JetPack version**: 7 (Ubuntu 24.04 LTS, Linux kernel 6.8)
- **Inference stack**: TensorRT + NVIDIA AI stack
- **Key flag**: Use `--dtype fp4` for 2,070 TFLOPS mode
- **Latency target**: ≤100ms per inference step
- **Memory**: 128GB shared CPU/GPU — no OOM risk with 10B model
- **Korean output**: Pass `--lang ko` to `run_thor_inference.py`

### Known issues on Thor
- Flash Attention 2 requires nvcc — use `attn_implementation="sdpa"` as fallback if needed
- First boot model loading takes ~3–4 min (22GB weights)
- Use `MIG` mode if running parallel experiments

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
- 이 CLAUDE.md 파일에 변경사항이 생겼을 때 — **반드시 커밋+푸시**

### 커밋 메시지 형식
```
<type>: <short description>

# type 목록:
# feat     - 새 시나리오, 새 스크립트, 새 기능
# eval     - 실험 결과, 벤치마크 업데이트
# fix      - 버그 수정
# docs     - 문서 업데이트 (README, CLAUDE.md 포함)
# config   - YAML 설정 변경
# chore    - 환경 셋업, 의존성 변경
```

예시:
```
feat: add narrow_alleyway scenario with pedestrian emergence events
eval: baseline results on horizontal_traffic_light (completion 40%)
docs: update CLAUDE.md with commit guidelines
```

### 푸시 주기
- 하루 작업 마칠 때 그날 커밋들을 한 번에 push
- 실험 결과가 나왔을 때 즉시 push (재현 가능성 보장)
- **CLAUDE.md 변경 시 항상 즉시 push** — AI 어시스턴트가 최신 컨텍스트를 읽어야 하기 때문

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
