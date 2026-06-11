# Jetson AGX Thor Deployment Guide

How to deploy Alpamayo-Korea on NVIDIA Jetson AGX Thor for real-time edge inference.

---

## Why Thor?

Alpamayo 1.5 is a 10B-parameter model. Running it in real time requires:
- **≥24GB VRAM** for model weights (FP4: ~11GB, BF16: ~22GB)
- **High memory bandwidth** for video input processing (4–7 cameras at 1080p)
- **<100ms inference latency** for safe autonomous driving

The Jetson AGX Thor provides:
- **128GB LPDDR5X** shared memory (GPU + CPU)
- **2,070 FP4 TFLOPS** peak throughput
- **900 GB/s** memory bandwidth
- **JetPack 7** (Ubuntu 24.04, CUDA 12.x, TensorRT 10.x)

No other edge platform matches this for a 10B-parameter model.

---

## Hardware Setup

### 1. Flash JetPack 7

Download NVIDIA SDK Manager on a host Ubuntu machine:
https://developer.nvidia.com/sdk-manager

Connect Thor via USB-C, boot into recovery mode (hold FORCE RECOVERY button while powering on), then flash via SDK Manager.

Target: **JetPack 7.0** (Ubuntu 24.04 LTS, Linux 6.8, CUDA 12.x)

### 2. First Boot Configuration

```bash
# On Thor (after flash, via SSH or HDMI terminal)
sudo apt update && sudo apt upgrade -y

# Enable max performance mode
sudo nvpmodel -m 0        # MAXN mode (130W)
sudo jetson_clocks        # lock clocks to max

# Verify GPU
nvidia-smi
# Should show: Jetson AGX Thor, 128GB

# Verify CUDA
nvcc --version
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Software Setup on Thor

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/alpamayo-korea.git
cd alpamayo-korea

# Install Python dependencies (use pip3, no uv needed on Thor)
pip3 install torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip3 install huggingface_hub transformers accelerate numpy

# Note: Flash Attention 2 may require nvcc on Thor
# Use SDPA fallback if build fails:
pip3 install flash-attn --no-build-isolation || echo "Will use SDPA fallback"
```

---

## Model Transfer to Thor

### Option A: SCP from dev machine

```bash
# From development machine
scp -r ~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B \
    jetson@THOR_IP:/home/jetson/.cache/huggingface/hub/

# Transfer fine-tuned checkpoint
scp -r checkpoints/alpamayo_korea_v1/ \
    jetson@THOR_IP:/home/jetson/alpamayo-korea/checkpoints/
```

### Option B: Download directly on Thor

```bash
# On Thor (requires internet connection)
export HF_TOKEN="your_token"
export HF_HUB_ENABLE_HF_TRANSFER=1
pip3 install hf_transfer
huggingface-cli download nvidia/Alpamayo-1.5-10B
```

---

## Running Inference

### Baseline model (Korean reasoning)

```bash
python scripts/run_thor_inference.py \
  --model nvidia/Alpamayo-1.5-10B \
  --lang ko \
  --dtype fp4 \
  --attn sdpa \
  --latency_target_ms 100 \
  --save_traces
```

### Fine-tuned Alpamayo-Korea

```bash
python scripts/run_thor_inference.py \
  --model checkpoints/alpamayo_korea_v1/ \
  --lang ko \
  --dtype fp4 \
  --attn sdpa \
  --latency_target_ms 100 \
  --save_traces \
  --output_dir evaluation/results/thor_inference/
```

---

## Expected Performance

| Mode | Model Size | Latency (avg) | Latency (P95) |
|---|---|---|---|
| FP4 | ~11GB | ~60–80ms | ~90ms |
| BF16 | ~22GB | ~120–150ms | ~180ms |
| FP16 | ~22GB | ~130–160ms | ~190ms |

FP4 mode meets the ≤100ms target. BF16 is safer if numerical precision matters for evaluation.

---

## MIG Mode (Multi-Instance GPU)

For running parallel experiments on Thor:

```bash
# Enable MIG mode
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb -C

# Run two inference instances simultaneously
python scripts/run_thor_inference.py --model baseline/ &
python scripts/run_thor_inference.py --model finetuned/ &
wait
```

---

## Known Issues on Thor

| Issue | Cause | Fix |
|---|---|---|
| Flash Attention 2 build fail | nvcc path issue | Use `--attn sdpa` |
| First load takes 3–4 min | 22GB model weights loading | Expected; subsequent loads use cache |
| OOM at BF16 + batch>1 | 22GB model near 128GB limit | Use `--dtype fp4` or reduce video seq_len |
| Python torch import slow | JetPack 7 first boot | Reboot once; cache warms up |

---

## Monitoring on Thor

```bash
# GPU utilization + memory
watch -n 1 nvidia-smi

# Thermal
watch -n 2 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Power draw
sudo tegrastats --interval 1000
```

---

## Latency Results Log Format

`run_thor_inference.py --save_traces` produces:

`evaluation/results/thor_inference/reasoning_traces.jsonl`:
```json
{"iteration": 1, "latency_ms": 72.3, "trajectory_next_5": [[...]], "reasoning_trace": "{\"상황\": \"...\", ...}", "timestamp": 1735000000.0}
```

`evaluation/results/thor_inference/latency_summary.json`:
```json
{"total_iterations": 1000, "avg_latency_ms": 68.4, "p50_latency_ms": 67.1, "p95_latency_ms": 88.2, "p99_latency_ms": 94.7, "target_ms": 100.0, "target_met_pct": 98.7}
```
