"""
benchmark_compare.py
Compare baseline Alpamayo 1.5 vs Alpamayo-Korea fine-tuned model.

Usage:
    python scripts/benchmark_compare.py \
        --baseline evaluation/results/baseline/ \
        --finetuned evaluation/results/finetuned/ \
        --output evaluation/results/comparison/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCENARIOS = [
    "horizontal_traffic_light",
    "bus_only_lane",
    "narrow_alleyway",
    "reverse_motorcycle",
    "jaywalking_dense",
]

METRICS = [
    "l2_displacement_error",
    "collision_rate",
    "scenario_completion_rate",
    "reasoning_coherence_score",
    "time_to_collision_min",
]


def load_results(results_dir: Path) -> dict[str, Any]:
    """Load evaluation results from a directory of JSON files."""
    results = {}
    for scenario in SCENARIOS:
        result_file = results_dir / f"{scenario}.json"
        if result_file.exists():
            with open(result_file) as f:
                results[scenario] = json.load(f)
        else:
            logger.warning(f"Missing result file: {result_file}")
    return results


def compute_delta(baseline_val: float, finetuned_val: float, lower_is_better: bool) -> str:
    """Compute delta with direction indicator."""
    delta = finetuned_val - baseline_val
    if lower_is_better:
        indicator = "BETTER" if delta < 0 else ("WORSE" if delta > 0 else "SAME")
    else:
        indicator = "BETTER" if delta > 0 else ("WORSE" if delta < 0 else "SAME")
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f} ({indicator})"


def print_comparison_table(baseline: dict, finetuned: dict) -> dict:
    """Print side-by-side comparison and return summary dict."""
    summary = {}

    print("\n" + "=" * 80)
    print("  BENCHMARK: Alpamayo 1.5 (Baseline) vs Alpamayo-Korea (Fine-tuned)")
    print("=" * 80)

    lower_is_better = {
        "l2_displacement_error": True,
        "collision_rate": True,
        "scenario_completion_rate": False,
        "reasoning_coherence_score": False,
        "time_to_collision_min": False,
    }

    for scenario in SCENARIOS:
        b = baseline.get(scenario, {})
        f = finetuned.get(scenario, {})
        if not b and not f:
            continue

        print(f"\n  Scenario: {scenario}")
        print(f"  {'Metric':<35} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>20}")
        print("  " + "-" * 79)

        scenario_summary = {}
        for metric in METRICS:
            b_val = b.get("metrics", {}).get(metric)
            f_val = f.get("metrics", {}).get(metric)
            if b_val is None and f_val is None:
                continue

            b_str = f"{b_val:.4f}" if b_val is not None else "N/A"
            f_str = f"{f_val:.4f}" if f_val is not None else "N/A"

            if b_val is not None and f_val is not None:
                delta_str = compute_delta(b_val, f_val, lower_is_better.get(metric, True))
                scenario_summary[metric] = {"baseline": b_val, "finetuned": f_val}
            else:
                delta_str = "N/A"

            print(f"  {metric:<35} {b_str:>12} {f_str:>12} {delta_str:>20}")

        summary[scenario] = scenario_summary

    print("\n" + "=" * 80)
    return summary


def compute_aggregate(summary: dict) -> dict:
    """Compute aggregate metrics across all scenarios."""
    aggregate: dict[str, list] = {m: [] for m in METRICS}

    for scenario_data in summary.values():
        for metric, vals in scenario_data.items():
            if metric in aggregate:
                aggregate[metric].append((vals["baseline"], vals["finetuned"]))

    print("\n  AGGREGATE (mean across all scenarios)")
    print(f"  {'Metric':<35} {'Baseline':>12} {'Fine-tuned':>12} {'Improvement':>12}")
    print("  " + "-" * 71)

    agg_summary = {}
    for metric, pairs in aggregate.items():
        if not pairs:
            continue
        b_mean = np.mean([p[0] for p in pairs])
        f_mean = np.mean([p[1] for p in pairs])
        improvement = ((b_mean - f_mean) / b_mean * 100) if b_mean != 0 else 0.0
        print(f"  {metric:<35} {b_mean:>12.4f} {f_mean:>12.4f} {improvement:>+11.1f}%")
        agg_summary[metric] = {"baseline_mean": b_mean, "finetuned_mean": f_mean, "improvement_pct": improvement}

    print("=" * 80 + "\n")
    return agg_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark comparison tool")
    parser.add_argument("--baseline",  type=str, required=True)
    parser.add_argument("--finetuned", type=str, required=True)
    parser.add_argument("--output",    type=str, default="evaluation/results/comparison/")
    args = parser.parse_args()

    baseline_dir  = Path(args.baseline)
    finetuned_dir = Path(args.finetuned)
    output_dir    = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading baseline results from:  {baseline_dir}")
    logger.info(f"Loading fine-tuned results from: {finetuned_dir}")

    baseline  = load_results(baseline_dir)
    finetuned = load_results(finetuned_dir)

    summary   = print_comparison_table(baseline, finetuned)
    aggregate = compute_aggregate(summary)

    full_report = {
        "per_scenario": summary,
        "aggregate": aggregate,
        "baseline_dir": str(baseline_dir),
        "finetuned_dir": str(finetuned_dir),
    }

    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
