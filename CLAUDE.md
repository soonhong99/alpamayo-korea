# CLAUDE.md — Project Context for AI Assistants

This file gives AI coding assistants (Claude, Copilot, etc.) the context needed to contribute effectively to this project.

---

## Project Summary

**★★★ 연구 방향 전환 (2026-05-24 교수님 미팅) ★★★**

**이전 방향 (폐기)**: Korean fine-tuning, RL post-training, 모델 수정  
**새 방향 (확정)**: **모델 수정 없이 Async Pipeline + iGPU 메모리 최적화로 Alpamayo throughput 극대화**

핵심 목표:
- **모델 수정 없이** `cudaMemPrefetchAsync` + CUDA Stream으로 layer prefetch ↔ compute 중첩
- GPU L2 Persistent Residency로 KV Cache를 L2에 고정 (4.76× DRAM 대비 속도)
- LM autoregressive decode의 layer-level async pipeline 구현 (EOS 기준 가변 step)

**★ 교수님이 관심있는 핵심 논문 (2026-06-07 피드백 확정)**:
- **llm.npu (ASPLOS 2025)** — RT-Swap보다 이 논문이 핵심
- 핵심 원리: **"의존성을 지키면서 각 자원의 유휴 시간(bubble)을 제거하는 스케줄링"**
  - NPU bubble 37% → 0.7% 달성 (out-of-order subgraph execution)
  - Thor 매핑: SM compute ↔ DMA prefetch (inter-layer), CPU work ↔ GPU compute (inter-inference)
- Thor 제약: **단일 SM 풀** → intra-layer SM병렬화(Level 0) ❌, inter-layer DMA-SM 중첩(Level 1) ✅
- **AI 어시스턴트 주의**: RT-Swap 중심으로 분석하지 말 것. llm.npu bubble elimination이 이 연구의 핵심 프레임.

**★ DRAM 대역폭 측정 결과 (2026-06-09 확정, Thor SM11.0)**:
- VE 35% / LM Prefill 55% / LM Decode 89% / Flow 88% (각각 LPDDR5X 231 GB/s 기준)
- Prefill: Layer당 ~79ms, FlashAttention SRAM-bound(39.5%), GEMM DRAM-bound(70%)
  → Layer prefetch 가능: 여유 BW 140 GB/s, 다음 레이어(~0.84 GB) 10ms에 prefetch 완료 ✅
- Decode: Step당 79.1ms, GEMV 89% BW 포화, 여유 BW ~25 GB/s
  → Layer prefetch 불가: 0.84 GB / 25 GB/s = 33ms >> layer 4.4ms ❌ (inter-inference 전략 필요)
- SM 메트릭: GB10B(SM 11.0)에서 `sm__active_cycles` 미존재 → `smsp__cycles_active.sum / smsp__cycles_elapsed.sum` 사용 (smsp_occupancy, warp 점유율, NOT compute utilization)

**★ 양자화는 이 연구의 범위 밖** — 시스템 스케줄링/파이프라이닝만 다룬다. AI 어시스턴트는 절대 양자화를 제안하거나 언급하지 말 것.

**★ UMIC 컴파일 엔진 (2026-06-10 착수) — 전용 repo로 분리됨**:
- **코드/실험 repo: `https://github.com/soonhong99/umic` (private), 로컬: `C:\Users\nanay\Desktop\umic`** — 컴파일 엔진 빌드·실험은 모두 그쪽에서 진행. 이 repo에는 연구 일지(docs/)만 남김
- 설계서: `docs/2606_1주차/260610_01_UMIC_iGPU_전용_컴파일엔진_설계서.md` (3계층 IR, measurement-guided compilation, 닫힌 패턴 집합 ~10개)
- **Triton 3.7.0 직접 `@triton.jit`은 SM 11.0에서 정상 동작** (2026-06-10 확정) — 죽은 건 torch.compile→Inductor 경로뿐. 커스텀 커널은 직접 Triton으로 작성
- M1 실측 (P5 gate_silu_mul fusion, `umic.integrate.fuse_mlps` — 체크포인트·모델 소스 무수정 forward 교체):
  **Prefill DRAM 232.0→148.1 GB (−36.2%), wall-clock 1,423→1,030ms (−27.6%)**, VE/Flow/Decode 회귀 없음
  - P5 단독 예측 −22.6GB를 크게 초과 — L2 경합 완화의 연쇄 효과 (마이크로벤치에 안 보이는 시스템 효과)
  - decode(seq=1)는 `FUSE_MIN_ROWS=64` 미만 시 eager 디스패치 (GEMV에 fusion은 손해)
- 모델 버전 불가지론 원칙: 패턴 매칭은 클래스가 아닌 구조(duck-typing) — Alpamayo 2.0 공개 대응 가능해야 함

**★ 측정 규칙 (2026-06-11 확정, 모든 성능 실험에 적용)**:
1. **측정 전 `sudo jetson_clocks` 필수** — memory-bound 단계(decode GEMV)는 SM 사용률이 낮아 DVFS 거버너가 GPU/EMC 클럭을 안 올림. 같은 코드가 거버너 107ms vs 고정 70-79ms. 10Hz 실시간 배포에도 클럭 고정 전제
2. **steady-state는 warmup 포함 5+ run 후 판정** — 클럭 고정 상태에서도 run 0→4에 걸쳐 allocator/페이지 워밍으로 모든 stage가 계단식 하락 (VE 427→305ms, decode 102→70ms)
3. **decode KV는 contiguous가 정답** (클럭 고정 조건). "view가 낫다"는 2026-06-10 기록은 거버너 변수 미통제 상태의 오판으로 철회됨
- AppendOnlyCache-C 79.1ms/step(2026-05-31)는 클럭 고정 조건에서 재현 확인됨
- **공식 벤치마크 (2026-06-11, 동일 조건: 클럭 고정·6-run steady)**: eager 3,846ms vs **UMIC 2,701ms (−29.8%)**, 단계별 VE −42.7% / Prefill −46.1% / Decode 70.0ms/step (−10.5%) / Flow −37.7%
  - ⚠ 구 기준선 4,838ms는 거버너+cold 조건 — 이 기준 대비 발표한 −48.7% 등은 부풀려진 수치로 철회. 클럭 고정 조건에서 eager DynamicCache decode는 이미 78.2ms/step (cat 비용 미미 — decode 개선 귀속은 융합+CUDA Graph)

참고 문서: `docs/260524_01_교수님_미팅_연구방향_전환.md`, `docs/2606_1주차/260607_03_llm_npu_to_thor_파이프라인_스케줄링_번역.md`

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
- **추론 주기 규칙 ★ 절대 고정**: Alpamayo는 **10Hz = 100ms 간격**으로 연속 추론한다.
  - Alpamayo 존재 이유: long-tail 상황(갑작스러운 보행자, 역주행차 등)에 대한 CoC(Chain of Causation) 인과추론으로 즉각 대응
  - 0.1초 사이에도 장면이 결정적으로 달라질 수 있음 → **KV Temporal Reuse 실험의 유효 Δt는 100ms 단 하나**
  - Δt=300/500/1000ms 측정은 학문적 참고용일 뿐 — 실제 시스템에서 1초 전 KV를 재사용하는 것은 시스템 설계 원칙 위반
  - **AI 어시스턴트 주의**: 실험 설계나 분석 시 "Δt를 늘려도 괜찮다"는 추론을 하지 말 것. NVIDIA가 정한 100ms 기준을 항상 따른다.
- **Latency target**: 전체 추론(prefill+decode) 기준 — 1단계 ~3,500ms, 최종 99ms
  - **현재 실측 (sdpa, BF16, N=1)**: 4,838ms = VE 728ms + LM Prefill 1,423ms + Decode 1,818ms (17steps×107ms) + Flow 870ms
  - **AppendOnlyCache-C 적용 후 (2026-05-31 확정)**: Decode 1,345ms (17steps×79.1ms, warm 기준) → 전체 ≈4,366ms (-9.9%)
    - decode steady-state: **79.1ms/step** (첫 실행 시 9 step JIT warmup 약 109ms, 이후 79ms)
    - decode mean (warmup 포함): 81.3ms/step (-24.3% vs DynamicCache)
    - prefill도 1,981ms로 감소 (-509ms, 원인: pre-alloc page mapping 선처리 효과)
  - ⚠️ 이전 기록 "6,554ms (prefill 4,487ms + decode 2,067ms)"는 eager+StaticCache 커스텀 경로 수치임 — 전체 파이프라인 기준이 아님
  - BF16 실질 하한 (L2 재사용 포함): decode 1step 최솟값 **79ms** (22GB 중 일부 L2 hit으로 실효 DRAM 접근 감소)
  - BF16 이론 하한 (대역폭만): 86ms (22GB÷231GB/s), 17step = 1,462ms
  - 1단계 (~3,500ms): AppendOnlyCache-C + KV Temporal Reuse (시스템만, BF16)
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
- **KV 캐시 구현 규칙** (2026-05-31 최종 확정):
  - **결론**: `AppendOnlyCache (force_contiguous)` + sdpa 기본값이 현재 최선. DynamicCache 대비 -24.3% (107ms→81ms/step).
  - 구현 위치: `scripts/inference/260531_appendonly_cache_exp.py` → `AppendOnlyCache` 클래스
  - 핵심 원리: DynamicCache 상속(FlashAttn 유지) + pre-alloc 버퍼 + in-place write + `.contiguous()` 출력
  - **eager 사용 금지**: seq_len=3086에서 LM Prefill이 1,423ms→3,753ms (2.6× 느려짐). 어떤 경로에서도 쓰지 말 것.
  - **StaticCache 주의 사항 (2026-05-31 업데이트)**:
    - 구 transformers: StaticCache → float 4D mask → MemEfficientAttn → 2× 느림 (구 기록)
    - **현재 transformers (Thor 설치본)**: `_update_causal_mask` 삭제됨 → `create_causal_mask` 도입 → StaticCache도 FlashAttn 작동, 102.5ms/step
    - 그러나 StaticCache는 full [B,H,MAX_LEN,D] 버퍼를 attention에 전달 (유효 토큰 n개 + zeros 포함) → AppendOnlyCache-C(79ms)보다 느림
    - ⚠️ 구 기록 "sdpa+StaticCache → MemEfficientAttn 2×"는 구 transformers 버전 기준임 — 현재 버전에서는 틀린 정보
  - **attn 백엔드 3계층**: FlashAttention(최속, attn_mask=None or bool) > MemEfficientAttn(float mask 지원, ~2×) > Math(eager동등, 최저속)
  - **4종 캐시 비교 요약** (2026-05-31 실측, sdpa, 80step):
    - AppendOnlyCache-C: **79.1ms/step** (steady-state), 81.3ms (mean) ← 최선
    - AppendOnlyCache-B (non-contiguous): 100.0ms/step
    - StaticCache: 102.5ms/step (FlashAttn, 현재 transformers)
    - DynamicCache: 107.4ms/step ← 종전 기준선
  - **`_update_causal_mask` 유무**: 현재 Thor transformers에는 이 메서드가 **없음**. monkey-patch 불가. `create_causal_mask` + `sdpa_mask()` 사용.
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
