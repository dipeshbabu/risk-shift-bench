"""Model comparison utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from risk_shift_bench.dataset import DecisionRecord
from risk_shift_bench.policy import action_probabilities, clear_policy_cache
from risk_shift_bench.risk_models import RiskModel


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    records: int
    accuracy: float
    nll: float
    mean_probability_of_observed_action: float
    brier_score: float
    calibration_error: float


def evaluate(records: Iterable[DecisionRecord], model: RiskModel, max_depth: int = 1) -> EvaluationResult:
    clear_policy_cache()
    total = 0
    correct = 0
    nll = 0.0
    prob_sum = 0.0
    brier_sum = 0.0
    calibration_bins: list[list[float]] = [[] for _ in range(10)]
    for record in records:
        probs = action_probabilities(record.to_state(), model, max_depth=max_depth)
        predicted = max(probs, key=probs.get)
        observed_prob = max(probs[record.action_taken], 1e-12)
        correct += int(predicted == record.action_taken)
        nll -= math.log(observed_prob)
        prob_sum += observed_prob
        p_hit = probs["hit"]
        y_hit = 1.0 if record.action_taken == "hit" else 0.0
        brier_sum += (p_hit - y_hit) ** 2
        bin_idx = min(9, int(p_hit * 10))
        calibration_bins[bin_idx].append(y_hit - p_hit)
        total += 1

    if total == 0:
        return EvaluationResult(model.name, 0, 0.0, float("inf"), 0.0, 0.0, 0.0)
    calibration_error = 0.0
    for bucket in calibration_bins:
        if not bucket:
            continue
        calibration_error += (len(bucket) / total) * abs(sum(bucket) / len(bucket))
    return EvaluationResult(
        model_name=model.name,
        records=total,
        accuracy=correct / total,
        nll=nll,
        mean_probability_of_observed_action=prob_sum / total,
        brier_score=brier_sum / total,
        calibration_error=calibration_error,
    )
