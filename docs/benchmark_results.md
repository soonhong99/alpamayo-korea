# Benchmark Results

Comparison of Alpamayo 1.5 (baseline) vs Alpamayo-Korea (fine-tuned) on Korean long-tail scenarios.

*This document will be updated as experiments are completed.*

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Base model | nvidia/Alpamayo-1.5-10B |
| Fine-tuned model | alpamayo_korea_v1 (this repo) |
| Simulator | AlpaSim (NVlabs) |
| Rollouts per scenario | 10 |
| Evaluation hardware | Development: A100 80GB |
| Inference hardware | Jetson AGX Thor (128GB) |
| Language | Korean (한국어) |

---

## Results Table

### Scenario: Horizontal Traffic Light (가로형 신호등)

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Scenario completion rate | TBD | TBD | TBD |
| Red light detection latency (s) | TBD | TBD | TBD |
| Stop accuracy (m from line) | TBD | TBD | TBD |
| Reasoning mentions horizontal | TBD | TBD | TBD |
| Collision rate | TBD | TBD | TBD |

### Scenario: Bus-Only Lane (버스전용차로)

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Scenario completion rate | TBD | TBD | TBD |
| Lane legality compliance | TBD | TBD | TBD |
| Reasoning mentions enforcement | TBD | TBD | TBD |
| Collision rate | TBD | TBD | TBD |

### Scenario: Narrow Alleyway (골목길)

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Scenario completion rate | TBD | TBD | TBD |
| Peak speed in alleyway (m/s) | TBD | TBD | TBD |
| Pedestrian yield count (/2) | TBD | TBD | TBD |
| Minimum clearance to wall (m) | TBD | TBD | TBD |
| Collision rate | TBD | TBD | TBD |

### Scenario: Reverse Motorcycle (역주행 오토바이)

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Scenario completion rate | TBD | TBD | TBD |
| Detection latency (s) | TBD | TBD | TBD |
| Min clearance to motorcycle (m) | TBD | TBD | TBD |
| Reasoning flags anomaly | TBD | TBD | TBD |
| Collision rate | TBD | TBD | TBD |

### Scenario: Jaywalking Dense (고밀도 무단횡단)

| Metric | Baseline | Fine-tuned | Delta |
|---|---|---|---|
| Scenario completion rate | TBD | TBD | TBD |
| Jaywalker detection latency (s) | TBD | TBD | TBD |
| Min clearance to pedestrian (m) | TBD | TBD | TBD |
| Reasoning explains yield | TBD | TBD | TBD |
| Collision rate | TBD | TBD | TBD |

---

## Thor Inference Latency

| Mode | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Target Met (%) |
|---|---|---|---|---|---|
| Baseline FP4 | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned FP4 | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned BF16 | TBD | TBD | TBD | TBD | TBD |

Target: ≤100ms (P95)

---

## Korean Reasoning Quality

| Scenario | Baseline score | Fine-tuned score | Improvement |
|---|---|---|---|
| horizontal_traffic_light | TBD | TBD | TBD |
| bus_only_lane | TBD | TBD | TBD |
| narrow_alleyway | TBD | TBD | TBD |
| reverse_motorcycle | TBD | TBD | TBD |
| jaywalking_dense | TBD | TBD | TBD |

Scoring methodology: `evaluation/metrics.py::reasoning_coherence_score()`
- 0.0: empty or non-Korean
- 0.5: Korean text, no scenario keywords
- 1.0: Korean + valid JSON + all required keywords

---

## How to Reproduce

```bash
# 1. Run baseline evaluation
bash scripts/run_baseline_eval.sh \
  --scenario scenarios/korea/ \
  --output evaluation/results/baseline/

# 2. Run fine-tuned evaluation
bash scripts/run_baseline_eval.sh \
  --model checkpoints/alpamayo_korea_v1/ \
  --scenario scenarios/korea/ \
  --output evaluation/results/finetuned/

# 3. Generate comparison report
python scripts/benchmark_compare.py \
  --baseline evaluation/results/baseline/ \
  --finetuned evaluation/results/finetuned/ \
  --output evaluation/results/comparison/

# 4. Evaluate reasoning traces
python evaluation/reasoning_eval.py \
  --traces evaluation/results/thor_inference/reasoning_traces.jsonl \
  --scenario horizontal_traffic_light \
  --output evaluation/results/reasoning_eval.json
```
