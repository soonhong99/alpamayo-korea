# Alpamayo-Korea: Korean Road Scenario Adaptation on NVIDIA Jetson Thor

> **Fine-tuning NVIDIA Alpamayo 1.5 for Korean long-tail driving scenarios with real-time edge inference on Jetson AGX Thor (128 GB)**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![NVIDIA Alpamayo](https://img.shields.io/badge/Model-Alpamayo%201.5-76B900.svg)](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
[![AlpaSim](https://img.shields.io/badge/Sim-AlpaSim-76B900.svg)](https://github.com/NVlabs/alpasim)
[![Thor](https://img.shields.io/badge/Hardware-Jetson%20AGX%20Thor-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)

---

## Why This Project Exists

NVIDIA's Alpamayo 1.5 is trained on 1,727 hours of driving data from 25 countries and 2,500+ cities — but **Korean road scenarios are severely underrepresented**.

Korean driving environments present unique long-tail challenges that global AV models systematically fail on:

| Scenario | Global Models | This Project |
|---|---|---|
| Horizontal traffic lights (Korean standard) | Misclassified | Fine-tuned |
| Bus-only lane enforcement at intersections | No reasoning | Reasoning trace |
| Jaywalking in high-density urban areas | Inconsistent | Scenario-specific |
| Reverse-direction motorcycles (illegal but common) | Not handled | Edge-case trained |
| Narrow alleyways (골목길) with sudden pedestrians | Missing | Coverage added |

This project:
1. **Benchmarks** Alpamayo 1.5 on Korean-specific scenarios via AlpaSim
2. **Fine-tunes** using Korean open-source datasets (AI Hub, Kakao Mobility, 42dot)
3. **Deploys** the fine-tuned model on **NVIDIA Jetson AGX Thor** for real-time edge inference
4. **Outputs reasoning traces in Korean** — enabling local AV companies to audit model decisions

**Why Thor matters:** Alpamayo 1.5 is a 10B-parameter model requiring 24GB+ VRAM and fast memory bandwidth. Thor's 128GB LPDDR5X and 2,070 FP4 TFLOPS make it the only edge platform capable of running this model in real time (≤100ms latency) without cloud dependency.

---

## Project Structure

```
alpamayo-korea/
│
├── README.md                    # This file
├── CLAUDE.md                    # AI assistant context file
├── LICENSE                      # Apache 2.0
├── .gitignore
│
├── docs/
│   ├── setup.md                 # Full environment setup guide
│   ├── datasets.md              # Korean dataset sources & access instructions
│   ├── scenarios.md             # Korean scenario taxonomy & rationale
│   ├── finetuning.md            # Fine-tuning methodology
│   ├── thor_deployment.md       # Jetson AGX Thor deployment guide
│   └── benchmark_results.md    # Benchmark: baseline vs fine-tuned
│
├── configs/
│   ├── alpasim_base.yaml        # AlpaSim baseline config
│   ├── alpasim_korea.yaml       # AlpaSim Korean scenario config
│   └── finetune_config.yaml     # Fine-tuning hyperparameters
│
├── scenarios/
│   └── korea/
│       ├── horizontal_traffic_light.yaml
│       ├── bus_only_lane.yaml
│       ├── narrow_alleyway.yaml
│       ├── reverse_motorcycle.yaml
│       ├── jaywalking_dense.yaml
│       └── README.md            # Scenario design rationale
│
├── scripts/
│   ├── setup_env.sh             # One-command environment setup
│   ├── download_datasets.sh     # Korean dataset download helper
│   ├── run_baseline_eval.sh     # Run Alpamayo baseline on Korean scenarios
│   ├── run_finetuning.sh        # Launch fine-tuning pipeline
│   ├── run_thor_inference.py    # Real-time inference on Jetson Thor
│   └── benchmark_compare.py    # Baseline vs fine-tuned comparison
│
├── data/
│   ├── README.md                # Data directory structure guide
│   ├── aihub/                   # AI Hub dataset (not tracked in git)
│   ├── kakao/                   # Kakao Mobility dataset
│   ├── 42dot/                   # 42dot open dataset
│   └── nvidia_physicalai/       # NVIDIA Physical AI AV dataset subset
│
├── evaluation/
│   ├── metrics.py               # Custom Korean scenario evaluation metrics
│   ├── reasoning_eval.py        # Korean reasoning trace evaluation
│   └── results/                 # Benchmark output JSONs
│
└── assets/
    ├── demo_baseline.mp4        # Baseline model on Korean roads (TBD)
    └── demo_finetuned.mp4       # Fine-tuned model comparison (TBD)
```

---

## Quick Start

### Prerequisites
- NVIDIA GPU ≥ 24GB VRAM (RTX 3090 / A100 for development)
- Ubuntu 22.04 or 24.04
- Docker + Docker Compose v2
- CUDA 12.x with `nvcc`
- Python 3.10+
- HuggingFace account with access to:
  - [`nvidia/Alpamayo-1.5-10B`](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
  - [`nvidia/Cosmos-Reason2-8B`](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
  - [`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

### Step 1 — Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/alpamayo-korea.git
cd alpamayo-korea

# One-command setup (installs uv, creates venv, compiles protos)
bash scripts/setup_env.sh
```

### Step 2 — Download Alpamayo + AlpaSim

```bash
# Set your HuggingFace token
export HF_TOKEN="your_hf_token_here"

# Download model weights (22GB — takes ~5 min on 100MB/s)
huggingface-cli login
huggingface-cli download nvidia/Alpamayo-1.5-10B
huggingface-cli download nvidia/Cosmos-Reason2-8B

# Clone AlpaSim
git clone https://github.com/NVlabs/alpasim.git
cd alpasim && bash scripts/setup.sh && cd ..
```

### Step 3 — Run Baseline Evaluation on Korean Scenarios

```bash
# Download a sample Korean scene from NVIDIA Physical AI dataset
bash scripts/download_datasets.sh --source nvidia_sample

# Run AlpaSim with baseline Alpamayo on Korean scenario set
bash scripts/run_baseline_eval.sh \
  --scenario scenarios/korea/ \
  --output evaluation/results/baseline/
```

### Step 4 — Fine-tune on Korean Data

```bash
# Download Korean datasets (AI Hub + Kakao Mobility)
bash scripts/download_datasets.sh --source aihub kakao

# Launch fine-tuning
bash scripts/run_finetuning.sh \
  --config configs/finetune_config.yaml \
  --data data/aihub/ data/kakao/
```

### Step 5 — Deploy on Jetson AGX Thor

```bash
# On the Thor board (JetPack 7)
python scripts/run_thor_inference.py \
  --model checkpoints/alpamayo_korea_v1/ \
  --lang ko \
  --scenario realtime_camera
```

---

## Benchmark Results (Baseline vs Fine-tuned)

*Results will be updated as experiments progress.*

| Metric | Alpamayo 1.5 (Baseline) | Alpamayo-Korea v1 | Delta |
|---|---|---|---|
| L2 displacement error (Korean scenarios) | TBD | TBD | TBD |
| Collision rate (narrow alleyway) | TBD | TBD | TBD |
| Reasoning trace coherence (Korean) | TBD | TBD | TBD |
| Inference latency (Thor, FP4) | TBD | TBD | TBD |

---

## Korean Datasets Used

| Dataset | Source | Size | License | Access |
|---|---|---|---|---|
| 도로주행 영상 (신호등·표지판) | [AI Hub](https://aihub.or.kr) | 1.9M images | Open (신청 필요) | [Link](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=188) |
| 자율주행 AI 학습 데이터셋 | [Kakao Mobility / ETRI AI 나눔](https://nanum.etri.re.kr) | 150K samples | Free, no copyright | [Link](https://nanum.etri.re.kr) |
| 42dot Open Dataset | [42dot](https://42dot.ai/akit) | Multi-sensor | Free (신청 필요) | [Link](https://42dot.ai) |
| Physical AI AV (NVIDIA) | [HuggingFace](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | 1,727 hrs | Non-commercial | [Link](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |
| ETRI 주행궤적 데이터 | [공공데이터포털](https://data.go.kr) | CSV trajectories | Open | [Link](https://www.data.go.kr/data/15041797/fileData.do) |

---

## Key Technical Stack

| Component | Technology |
|---|---|
| Base Model | Alpamayo 1.5 (10B params, Cosmos Reason2 backbone) |
| Simulator | AlpaSim (NVlabs, microservice, Docker Compose) |
| Fine-tuning | RL post-training (Cosmos Cookbook recipes) |
| Edge Hardware | NVIDIA Jetson AGX Thor (128GB, 2,070 FP4 TFLOPS) |
| Reasoning Output | Korean-language Chain-of-Causation traces |
| Inference Stack | JetPack 7, TensorRT, NVIDIA AI stack |

---

## Roadmap

- [x] Project scaffolding & documentation
- [ ] AlpaSim setup + baseline run on 5 Korean scenarios
- [ ] Korean dataset download & preprocessing pipeline
- [ ] Baseline benchmark (Alpamayo 1.5 on Korean scenarios)
- [ ] Fine-tuning experiment 1: horizontal traffic light adaptation
- [ ] Fine-tuning experiment 2: narrow alleyway + jaywalking
- [ ] Full Korean scenario benchmark (baseline vs fine-tuned)
- [ ] Thor deployment & latency benchmarking
- [ ] Korean reasoning trace evaluation
- [ ] arXiv preprint submission

---

## Why This Is Commercially Relevant

South Korea is explicitly targeted by global AV companies (Pony.ai, WeRide, DeepRoute.ai) as a key expansion market due to its aging population, driver shortage, and progressive AV regulation. The Korean government is expanding autonomous driving pilot zones from 332km to 5,224km of highway. **A localized reasoning model is a prerequisite for safe deployment in Korean urban environments.**

This project directly enables:
- Korean AV startups needing a local reasoning baseline
- Global AV companies (Pony.ai KR, Hyundai Avride) seeking Korean road adaptation
- Government validation of AV safety reasoning in Korean urban environments

---

## Citation

```bibtex
@software{alpamayo_korea_2025,
  author    = {[Your Name]},
  title     = {Alpamayo-Korea: Korean Road Scenario Adaptation on NVIDIA Jetson Thor},
  year      = {2025},
  url       = {https://github.com/YOUR_USERNAME/alpamayo-korea}
}
```

Upstream citation:
```bibtex
@article{nvidia2025alpamayo,
  title   = {Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
  author  = {NVIDIA et al.},
  year    = {2025},
  journal = {arXiv preprint arXiv:2511.00088}
}
```

---

## Contact

Questions, collaboration proposals, or consulting inquiries:
- Email: your.email@example.com
- LinkedIn: your-profile
