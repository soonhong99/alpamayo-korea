#!/usr/bin/env bash
# run_baseline_eval.sh — Run Alpamayo 1.5 baseline on Korean scenarios
# Usage:
#   bash scripts/run_baseline_eval.sh
#   bash scripts/run_baseline_eval.sh --scenario scenarios/korea/horizontal_traffic_light.yaml
#   bash scripts/run_baseline_eval.sh --scenario scenarios/korea/ --output evaluation/results/baseline/
set -euo pipefail

SCENARIO_PATH="scenarios/korea/"
OUTPUT_DIR="evaluation/results/baseline/"
MODEL="nvidia/Alpamayo-1.5-10B"
N_ROLLOUTS=10
SAVE_VIDEOS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --scenario)   SCENARIO_PATH="$2"; shift 2 ;;
    --output)     OUTPUT_DIR="$2";    shift 2 ;;
    --model)      MODEL="$2";         shift 2 ;;
    --rollouts)   N_ROLLOUTS="$2";    shift 2 ;;
    --save-videos) SAVE_VIDEOS=true;  shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  Alpamayo-Korea — Baseline Evaluation"
echo "============================================"
echo "  Model:    $MODEL"
echo "  Scenario: $SCENARIO_PATH"
echo "  Output:   $OUTPUT_DIR"
echo "  Rollouts: $N_ROLLOUTS per scenario"
echo "============================================"

# Check HF token
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set."
  echo "  Run: export HF_TOKEN='your_token'"
  exit 1
fi

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

mkdir -p "$OUTPUT_DIR"

# Check AlpaSim is present
if [[ ! -d "alpasim" ]]; then
  echo "ERROR: alpasim/ directory not found."
  echo "  Run: git clone https://github.com/NVlabs/alpasim.git"
  exit 1
fi

echo ""
echo "Running AlpaSim with Alpamayo 1.5 baseline..."
echo ""

cd alpasim

uv run alpasim_wizard \
  deploy=local \
  topology=1gpu \
  driver=alpamayo_configs \
  driver.model_name_or_path="$MODEL" \
  driver.dtype=bf16 \
  driver.attn_implementation=sdpa \
  wizard.scenario_path="../${SCENARIO_PATH}" \
  wizard.n_rollouts="$N_ROLLOUTS" \
  wizard.output_dir="../${OUTPUT_DIR}" \
  wizard.save_videos="$SAVE_VIDEOS" \
  wizard.language=ko \
  wizard.log_dir="/tmp/alpamayo_korea_baseline"

cd ..

echo ""
echo "Baseline evaluation complete."
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To compare baseline vs fine-tuned, run:"
echo "  python scripts/benchmark_compare.py \\"
echo "    --baseline $OUTPUT_DIR \\"
echo "    --finetuned evaluation/results/finetuned/"
