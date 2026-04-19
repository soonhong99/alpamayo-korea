"""
reasoning_eval.py
Evaluate Korean-language reasoning traces from Alpamayo-Korea.

Reads JSONL files produced by run_thor_inference.py (--save_traces)
and scores each trace for language quality, structural validity,
and scenario-specific content.

Usage:
    python evaluation/reasoning_eval.py \
        --traces evaluation/results/thor_inference/reasoning_traces.jsonl \
        --scenario horizontal_traffic_light \
        --output evaluation/results/reasoning_eval.json
"""

import argparse
import json
import logging
from pathlib import Path

from evaluation.metrics import reasoning_coherence_score, SCENARIO_KEYWORDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_traces(path: Path) -> list[dict]:
    traces = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def evaluate_traces(traces: list[dict], scenario_id: str) -> dict:
    """Score all traces and return aggregated stats."""
    keywords = SCENARIO_KEYWORDS.get(scenario_id, [])
    scores = []
    keyword_hits: dict[str, int] = {kw: 0 for kw in keywords}
    empty_count = 0
    valid_json_count = 0

    for entry in traces:
        trace = entry.get("reasoning_trace") or ""

        if not trace:
            empty_count += 1
            scores.append(0.0)
            continue

        score = reasoning_coherence_score(trace, scenario_id, keywords)
        scores.append(score)

        # Count keyword hits
        for kw in keywords:
            if kw in trace:
                keyword_hits[kw] += 1

        # Count valid JSON
        try:
            parsed = json.loads(trace)
            if {"상황", "판단", "행동", "안전근거"}.issubset(parsed.keys()):
                valid_json_count += 1
        except (json.JSONDecodeError, TypeError):
            pass

    n = len(traces)
    if n == 0:
        return {"error": "No traces found"}

    import statistics
    result = {
        "scenario_id": scenario_id,
        "n_traces": n,
        "mean_score": round(statistics.mean(scores), 4),
        "median_score": round(statistics.median(scores), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "empty_trace_rate": round(empty_count / n, 4),
        "valid_json_rate": round(valid_json_count / n, 4),
        "keyword_hit_rates": {
            kw: round(count / n, 4) for kw, count in keyword_hits.items()
        },
    }

    logger.info(f"Scenario: {scenario_id} | n={n}")
    logger.info(f"  Mean score:     {result['mean_score']:.4f}")
    logger.info(f"  Valid JSON:     {result['valid_json_rate']:.1%}")
    logger.info(f"  Empty traces:   {result['empty_trace_rate']:.1%}")
    logger.info(f"  Keyword hits:   {result['keyword_hit_rates']}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Korean reasoning trace evaluator")
    parser.add_argument("--traces",   type=str, required=True,
                        help="Path to reasoning_traces.jsonl")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=list(SCENARIO_KEYWORDS.keys()),
                        help="Scenario ID for keyword matching")
    parser.add_argument("--output",   type=str,
                        default="evaluation/results/reasoning_eval.json")
    args = parser.parse_args()

    traces_path = Path(args.traces)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading traces from: {traces_path}")
    traces = load_traces(traces_path)
    logger.info(f"Loaded {len(traces)} traces")

    result = evaluate_traces(traces, args.scenario)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
