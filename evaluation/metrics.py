"""
metrics.py
Custom evaluation metrics for Korean scenario benchmarking.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    scenario_id: str
    rollout_id: int
    success: bool
    failure_reason: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    reasoning_trace: Optional[str] = None


def l2_displacement_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    horizon_steps: int = 64,
) -> float:
    """Average L2 displacement error over the prediction horizon.

    Args:
        predicted:     (T, 2) or (T, 3) array of predicted waypoints (x, y[, z])
        ground_truth:  same shape as predicted
        horizon_steps: number of steps to evaluate (default: 64 = 6.4s)
    """
    assert predicted.shape == ground_truth.shape, "Shape mismatch"
    T = min(horizon_steps, predicted.shape[0])
    diff = predicted[:T, :2] - ground_truth[:T, :2]  # only x,y
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def collision_rate(results: list[ScenarioResult]) -> float:
    """Fraction of rollouts with at least one collision."""
    if not results:
        return 0.0
    collisions = sum(
        1 for r in results
        if r.failure_reason and "collision" in r.failure_reason.lower()
    )
    return collisions / len(results)


def scenario_completion_rate(results: list[ScenarioResult]) -> float:
    """Fraction of rollouts where ego completed the scenario successfully."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def time_to_collision_min(ttc_values: list[float]) -> float:
    """Minimum time-to-collision observed across a rollout (safety margin)."""
    if not ttc_values:
        return float("inf")
    return float(np.min(ttc_values))


def reasoning_coherence_score(
    reasoning_trace: str,
    scenario_id: str,
    required_keywords: Optional[list[str]] = None,
) -> float:
    """Score Korean reasoning trace quality.

    Scoring:
    - 0.0: empty or non-Korean trace
    - 0.5: Korean text present, no scenario-specific keywords
    - 1.0: Korean text + all required keywords present + valid JSON structure
    """
    if not reasoning_trace or not reasoning_trace.strip():
        return 0.0

    score = 0.0

    # Check Korean characters present
    korean_chars = sum(1 for c in reasoning_trace if "\uAC00" <= c <= "\uD7A3")
    if korean_chars < 5:
        return 0.0
    score += 0.3

    # Check JSON structure
    import json
    try:
        parsed = json.loads(reasoning_trace)
        required_keys = {"상황", "판단", "행동", "안전근거"}
        if required_keys.issubset(parsed.keys()):
            score += 0.4
        else:
            score += 0.2
    except (json.JSONDecodeError, TypeError):
        score += 0.1

    # Check scenario-specific keywords
    if required_keywords:
        found = sum(1 for kw in required_keywords if kw in reasoning_trace)
        score += 0.3 * (found / len(required_keywords))

    return min(score, 1.0)


SCENARIO_KEYWORDS: dict[str, list[str]] = {
    "horizontal_traffic_light": ["가로형", "수평", "신호등", "정지선"],
    "bus_only_lane":            ["버스전용", "차선", "시간", "위반"],
    "narrow_alleyway":          ["골목", "좁은", "보행자", "서행"],
    "reverse_motorcycle":       ["역주행", "오토바이", "충돌위험", "회피"],
    "jaywalking_dense":         ["무단횡단", "보행자", "감속", "안전거리"],
}


def evaluate_scenario(
    scenario_id: str,
    results: list[ScenarioResult],
    predicted_trajectories: Optional[list[np.ndarray]] = None,
    gt_trajectories: Optional[list[np.ndarray]] = None,
) -> dict:
    """Compute all metrics for a scenario and return as a dict."""
    keywords = SCENARIO_KEYWORDS.get(scenario_id, [])

    metrics: dict = {
        "collision_rate": collision_rate(results),
        "scenario_completion_rate": scenario_completion_rate(results),
    }

    if predicted_trajectories and gt_trajectories:
        l2_scores = [
            l2_displacement_error(pred, gt)
            for pred, gt in zip(predicted_trajectories, gt_trajectories)
        ]
        metrics["l2_displacement_error"] = float(np.mean(l2_scores))

    reasoning_scores = [
        reasoning_coherence_score(r.reasoning_trace or "", scenario_id, keywords)
        for r in results
    ]
    metrics["reasoning_coherence_score"] = float(np.mean(reasoning_scores))

    n = len(results)
    logger.info(
        f"[{scenario_id}] n={n} | "
        f"completion={metrics['scenario_completion_rate']:.2%} | "
        f"collision={metrics['collision_rate']:.2%} | "
        f"reasoning={metrics['reasoning_coherence_score']:.2f}"
    )

    return {"scenario_id": scenario_id, "n_rollouts": n, "metrics": metrics}
