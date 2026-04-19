# Environment Setup Guide

Complete step-by-step guide to set up the Alpamayo-Korea development environment.

---

## Hardware Requirements

### Development Machine (minimum)
| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA RTX 3090 (24GB) | A100 80GB |
| RAM | 32GB | 64GB |
| Storage | 500GB SSD | 1TB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 24.04 |
| CUDA | 12.x | 12.4 |

### Deployment Target
| Component | Spec |
|---|---|
| Board | NVIDIA Jetson AGX Thor |
| GPU Memory | 128GB LPDDR5X (unified) |
| AI Compute | 2,070 FP4 TFLOPS |
| OS | JetPack 7 (Ubuntu 24.04, Linux 6.8) |
| Power | 40–130W configurable |

---

## Step 0 — HuggingFace Account Setup

You need access to three gated HuggingFace resources:

1. **Alpamayo 1.5 model**: https://huggingface.co/nvidia/Alpamayo-1.5-10B
2. **Cosmos Reason2 backbone**: https://huggingface.co/nvidia/Cosmos-Reason2-8B
3. **Physical AI AV dataset**: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

Go to each URL, click "Access repository", and accept the license agreement.
Then create a HuggingFace token at: https://huggingface.co/settings/tokens

```bash
export HF_TOKEN="hf_your_token_here"
# Add to ~/.bashrc or ~/.zshrc to persist
```

---

## Step 1 — System Dependencies

```bash
# Ubuntu 22.04/24.04
sudo apt update && sudo apt install -y \
  git \
  curl \
  build-essential \
  libssl-dev \
  python3.10 \
  python3.10-dev \
  python3-pip \
  docker.io \
  docker-compose-plugin \
  nvidia-container-toolkit

# Verify NVIDIA setup
nvidia-smi
nvcc --version
docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 2 — Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/alpamayo-korea.git
cd alpamayo-korea

# One-command setup
bash scripts/setup_env.sh
source .venv/bin/activate
```

---

## Step 3 — Clone AlpaSim

```bash
git clone https://github.com/NVlabs/alpasim.git
cd alpasim

# Install alpasim_wizard CLI
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync

# One-time setup (compile protos, download sample)
export HF_TOKEN="your_token"
bash scripts/setup.sh

cd ..
```

---

## Step 4 — Download Alpamayo 1.5

```bash
huggingface-cli login  # paste HF token

# Download model weights (~22GB, ~5 min on 100MB/s)
huggingface-cli download nvidia/Alpamayo-1.5-10B
huggingface-cli download nvidia/Cosmos-Reason2-8B
```

---

## Step 5 — Verify Installation

```bash
# Test Alpamayo inference
cd alpasim
python -c "
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
import torch
model = Alpamayo1_5.from_pretrained(
    'nvidia/Alpamayo-1.5-10B',
    dtype=torch.bfloat16,
    attn_implementation='sdpa'
).to('cuda')
print('Alpamayo 1.5 loaded successfully')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
cd ..

# Test AlpaSim with default driver
cd alpasim
uv run alpasim_wizard deploy=local topology=1gpu driver=vavam wizard.log_dir=/tmp/test_run
cd ..
echo "AlpaSim working"
```

---

## Step 6 — Run First Korean Scenario

```bash
# Run baseline evaluation on Korean horizontal traffic light scenario
bash scripts/run_baseline_eval.sh \
  --scenario scenarios/korea/horizontal_traffic_light.yaml \
  --output evaluation/results/baseline/
```

---

## Jetson AGX Thor Setup (Deployment)

### 1. Flash JetPack 7
Download NVIDIA SDK Manager: https://developer.nvidia.com/sdk-manager
Flash JetPack 7 (Ubuntu 24.04) to the Thor board.

### 2. Install dependencies on Thor

```bash
# On the Thor board
sudo apt update && sudo apt install -y python3.10 python3-pip git
pip3 install huggingface_hub transformers accelerate torch

# Note: On Thor, use SDPA (no nvcc needed for flash-attn)
pip3 install torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 3. Copy model to Thor

```bash
# From development machine
scp -r ~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B \
    jetson@THOR_IP:/home/jetson/.cache/huggingface/hub/

scp -r checkpoints/alpamayo_korea_v1/ \
    jetson@THOR_IP:/home/jetson/alpamayo-korea/checkpoints/
```

### 4. Run inference on Thor

```bash
# On the Thor board
cd alpamayo-korea
python scripts/run_thor_inference.py \
  --model checkpoints/alpamayo_korea_v1/ \
  --lang ko \
  --dtype fp4 \
  --attn sdpa \
  --save_traces
```

---

## Troubleshooting

**flash-attn build fails:**
```bash
# Option A: Use SDPA fallback
python scripts/run_thor_inference.py --attn sdpa

# Option B: Ensure nvcc is on PATH
which nvcc
export PATH=/usr/local/cuda/bin:$PATH
pip install flash-attn --no-build-isolation
```

**CUDA out of memory:**
```bash
# Use FP4 or BF16 instead of FP16
python scripts/run_thor_inference.py --dtype bf16

# Or reduce batch size in finetune_config.yaml
```

**AlpaSim Docker fails:**
```bash
# Ensure nvidia-container-toolkit is installed
sudo systemctl restart docker
docker run --gpus all nvidia/cuda:12.4.0-base nvidia-smi
```

**HuggingFace download slow:**
```bash
# Use HF_HUB_ENABLE_HF_TRANSFER for faster downloads
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download nvidia/Alpamayo-1.5-10B
```
