#!/usr/bin/env bash
# run_finetuning.sh — Launch Alpamayo-Korea RL post-training
# Usage:
#   bash scripts/run_finetuning.sh
#   bash scripts/run_finetuning.sh --config configs/finetune_config.yaml --data data/aihub/ data/kakao/
set -euo pipefail

CONFIG="configs/finetune_config.yaml"
DATA_DIRS=()
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)       CONFIG="$2";       shift 2 ;;
    --data)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        DATA_DIRS+=("$1"); shift
      done
      ;;
    --resume)       RESUME_FROM="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  Alpamayo-Korea — RL Fine-tuning"
echo "============================================"
echo "  Config: $CONFIG"
echo "  Data:   ${DATA_DIRS[*]:-'(from config)'}"
echo "============================================"

# Prerequisites
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set."
  exit 1
fi

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Check data exists
for d in "${DATA_DIRS[@]:-}"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: Data directory not found: $d"
    echo "  Run: bash scripts/download_datasets.sh --source aihub kakao"
    exit 1
  fi
done

mkdir -p checkpoints/alpamayo_korea_v1 logs/finetune

echo ""
echo "Starting RL post-training..."
echo "  Following NVIDIA Cosmos Cookbook recipes."
echo "  This will take several hours on a single A100."
echo ""

# Build data args
DATA_ARGS=""
for d in "${DATA_DIRS[@]:-}"; do
  DATA_ARGS="$DATA_ARGS training.data.additional_paths=[\"$d\"]"
done

RESUME_ARG=""
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_ARG="training.resume_from_checkpoint=$RESUME_FROM"
fi

python -m alpamayo_korea.train \
  --config "$CONFIG" \
  $DATA_ARGS \
  $RESUME_ARG \
  2>&1 | tee logs/finetune/run_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Fine-tuning complete."
echo "Checkpoint saved to: checkpoints/alpamayo_korea_v1/"
echo ""
echo "Next: run evaluation"
echo "  bash scripts/run_baseline_eval.sh --model checkpoints/alpamayo_korea_v1/ --output evaluation/results/finetuned/"
