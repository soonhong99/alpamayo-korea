#!/usr/bin/env bash
# setup_env.sh — One-command environment setup for alpamayo-korea
# Usage: bash scripts/setup_env.sh
set -euo pipefail

echo "=========================================="
echo "  Alpamayo-Korea Environment Setup"
echo "=========================================="

# ── 1. Check CUDA ──────────────────────────────
echo "[1/7] Checking CUDA..."
if ! command -v nvcc &>/dev/null; then
  echo "ERROR: nvcc not found. Install CUDA Toolkit 12.x first."
  echo "  → https://developer.nvidia.com/cuda-downloads"
  exit 1
fi
CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "  ✓ CUDA $CUDA_VER found"

# ── 2. Check GPU VRAM ──────────────────────────
echo "[2/7] Checking GPU VRAM..."
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 24000 ]; then
  echo "WARNING: GPU has ${VRAM}MB VRAM. Alpamayo requires ≥24GB."
  echo "  Fine-tuning will fail. Evaluation may still work with reduced batch."
else
  echo "  ✓ GPU VRAM: ${VRAM}MB (sufficient)"
fi

# ── 3. Install uv (fast Python package manager) ─
echo "[3/7] Installing uv..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "  ✓ uv installed"
else
  echo "  ✓ uv already installed"
fi

# ── 4. Create virtual environment ─────────────
echo "[4/7] Creating Python venv..."
uv venv .venv --python 3.10
source .venv/bin/activate
echo "  ✓ venv created at .venv/"

# ── 5. Install Python dependencies ────────────
echo "[5/7] Installing dependencies..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install \
  huggingface_hub \
  transformers \
  accelerate \
  pyyaml \
  numpy \
  opencv-python \
  matplotlib \
  tqdm \
  pytest \
  loguru

# Try flash-attn (requires nvcc)
echo "  Installing flash-attn (may take a few minutes)..."
uv pip install flash-attn --no-build-isolation || {
  echo "  WARNING: flash-attn build failed. Will use SDPA fallback."
  echo "  To fix: ensure nvcc is on PATH and retry."
}
echo "  ✓ dependencies installed"

# ── 6. Clone AlpaSim ──────────────────────────
echo "[6/7] Cloning AlpaSim..."
if [ ! -d "alpasim" ]; then
  git clone https://github.com/NVlabs/alpasim.git
  echo "  ✓ AlpaSim cloned"
else
  echo "  ✓ AlpaSim already present"
fi

cd alpasim
# AlpaSim setup (compiles protos, installs wizard CLI)
uv run python -m pip install -e . --quiet 2>/dev/null || true
cd ..
echo "  ✓ AlpaSim setup done"

# ── 7. Check Docker ────────────────────────────
echo "[7/7] Checking Docker..."
if ! command -v docker &>/dev/null; then
  echo "WARNING: Docker not found. AlpaSim requires Docker Compose v2."
  echo "  → https://docs.docker.com/engine/install/"
else
  DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ',')
  echo "  ✓ Docker $DOCKER_VER found"
  if ! docker compose version &>/dev/null; then
    echo "WARNING: 'docker compose' (v2) not found. Install Docker Compose v2."
  else
    echo "  ✓ Docker Compose v2 found"
  fi
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Set HF token:  export HF_TOKEN='your_token'"
echo "  2. Download model: bash scripts/download_datasets.sh --source nvidia_sample"
echo "  3. Run baseline:   bash scripts/run_baseline_eval.sh"
echo "=========================================="
