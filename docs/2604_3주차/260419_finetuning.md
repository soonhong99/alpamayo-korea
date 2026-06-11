# Fine-tuning Methodology

How we adapt Alpamayo 1.5 to Korean road scenarios using RL post-training.

---

## Overview

We follow NVIDIA's **Cosmos Cookbook** RL post-training recipe, extended with:
1. Korean-language reasoning trace supervision
2. Scenario-weighted sampling for Korean edge cases
3. AlpaSim closed-loop feedback as the primary reward signal

---

## Why RL Post-training?

Alpamayo 1.5 was trained using a two-phase approach:
- **Phase 1**: Supervised pre-training on trajectory prediction
- **Phase 2**: RL post-training to improve chain-of-thought reasoning quality

We apply a **Phase 2 continuation** focused on Korean scenarios. This is preferable to full fine-tuning because:
- Base perception capabilities are preserved
- Only the reasoning and trajectory refinement layers are updated
- Much lower compute requirement (~10K steps vs millions)

---

## Training Setup

### Hardware
| Environment | Config |
|---|---|
| Development | 1x A100 80GB or 4x RTX 3090 |
| Recommended | 4x A100 80GB (multi-GPU via DDP) |
| Minimum | 1x RTX 3090 24GB (reduced batch) |

### Key Hyperparameters
See `configs/finetune_config.yaml` for full config.

```yaml
training:
  method: rl_post_training
  total_steps: 10000
  batch_size: 4              # per GPU
  learning_rate: 1.0e-5
  dtype: bf16
```

---

## Reward Function

The reward has four components:

### 1. Trajectory L2 (weight: 1.0)
Standard displacement error against ground truth or preferred trajectory.

### 2. Collision Penalty (weight: -10.0)
Binary: any collision in the AlpaSim rollout receives a large negative reward.

### 3. Scenario Completion Bonus (weight: +5.0)
Binary: ego successfully completes the scenario criteria (stop at red, yield to pedestrian, etc.).

### 4. Korean Reasoning Coherence (weight: 0.5)
Scored by `evaluation/metrics.py::reasoning_coherence_score()`:
- Korean text present: +0.3
- Valid JSON structure with required keys: +0.4
- Scenario-specific keywords present: +0.3

---

## Data Pipeline

```
Phase 1: NVIDIA Physical AI AV NuRec scenes
  → AlpaSim reconstructed scenes
  → Baseline Alpamayo 1.5 rollouts
  → Measure baseline performance

Phase 2: Korean datasets preprocessing
  AI Hub #188  → convert to AlpaSim NuRec-compatible format
  Kakao Mobility → extract LiDAR + camera pairs
  42dot          → align multi-camera to Alpamayo 7-cam format

Phase 3: RL training loop
  for each step:
    sample scenario (weighted by finetune_config.yaml weights)
    run AlpaSim rollout with current model
    compute reward (trajectory + collision + completion + reasoning)
    update model weights via PPO/GRPO
    log to wandb + JSONL
```

---

## Scenario Weights

Rare scenarios are oversampled to prevent the model from ignoring them:

| Scenario | Weight | Reason |
|---|---|---|
| horizontal_traffic_light | 2.0x | Highest baseline failure rate |
| bus_only_lane | 1.5x | Moderate failure, complex reasoning needed |
| narrow_alleyway | 2.0x | High failure rate, safety critical |
| reverse_motorcycle | 3.0x | Rarest event, needs most coverage |
| jaywalking_dense | 1.5x | Common but model sometimes handles OK |

---

## Korean Reasoning Supervision

The key innovation over vanilla RL post-training: we supervise the **language output** to be in Korean with scenario-specific content.

The model is prompted with:
```
현재 주행 상황을 분석하고, 결정한 행동과 그 이유를 한국어로 설명하세요.
형식: {"상황": "...", "판단": "...", "행동": "...", "안전근거": "..."}
```

Required fields:
- `상황`: Current situation description
- `판단`: Model's assessment/judgment
- `행동`: Chosen action
- `안전근거`: Safety rationale

---

## Running Fine-tuning

```bash
# 1. Download Korean datasets first
bash scripts/download_datasets.sh --source aihub kakao

# 2. Verify AlpaSim is running
cd alpasim && uv run alpasim_wizard deploy=local topology=1gpu driver=vavam wizard.log_dir=/tmp/test && cd ..

# 3. Launch fine-tuning
bash scripts/run_finetuning.sh \
  --config configs/finetune_config.yaml \
  --data data/aihub/ data/kakao/

# 4. Monitor training
tail -f logs/finetune/run_*.log
```

---

## Expected Training Timeline

| Phase | Duration | Compute |
|---|---|---|
| Data preprocessing | 2–4 hours | CPU |
| First 1K steps (validation) | ~3 hours | 1x A100 |
| Full 10K steps | ~30 hours | 1x A100 |
| Full 10K steps | ~8 hours | 4x A100 |

---

## Checkpoints

Checkpoints are saved every 500 steps to `checkpoints/alpamayo_korea_v1/`.
Only the last 3 checkpoints are kept (configurable in `finetune_config.yaml`).

To resume from a checkpoint:
```bash
bash scripts/run_finetuning.sh --resume checkpoints/alpamayo_korea_v1/step_5000/
```
