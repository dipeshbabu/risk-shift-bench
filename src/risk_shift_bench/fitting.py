"""Likelihood-based fitting for personalized risk models."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from risk_shift_bench.dataset import DecisionRecord
from risk_shift_bench.policy import action_probabilities, clear_policy_cache
from risk_shift_bench.risk_models import ProspectUtilityModel, RiskModel, StateDependentProspectModel


@dataclass(frozen=True)
class FitResult:
    model_name: str
    params: dict[str, float]
    nll: float
    records: int


def negative_log_likelihood(records: Iterable[DecisionRecord], model: RiskModel, max_depth: int = 1) -> float:
    total = 0.0
    count = 0
    clear_policy_cache()
    for record in records:
        probs = action_probabilities(record.to_state(), model, max_depth=max_depth)
        total -= math.log(max(probs[record.action_taken], 1e-12))
        count += 1
    if count == 0:
        return float("inf")
    return total


def fit_static_prospect(records: list[DecisionRecord], max_depth: int = 1) -> FitResult:
    """Fit a static prospect model with a compact deterministic grid search."""

    candidates = []
    for alpha, loss_aversion, temperature in itertools.product(
        (0.55, 0.75, 1.0, 1.25),
        (1.0, 1.5, 2.25, 3.5),
        (0.3, 0.7, 1.2, 2.0),
    ):
        model = ProspectUtilityModel(alpha=alpha, loss_aversion=loss_aversion, temperature=temperature)
        candidates.append((negative_log_likelihood(records, model, max_depth=max_depth), model))

    best_nll, best_model = min(candidates, key=lambda item: item[0])
    params = {
        "alpha": best_model.alpha,
        "loss_aversion": best_model.loss_aversion,
        "temperature": best_model.temperature,
    }
    return FitResult(best_model.name, params, best_nll, len(records))


def fit_state_dependent_prospect(records: list[DecisionRecord], max_depth: int = 1) -> FitResult:
    """Fit a small state-dependent model.

    This is intentionally modest: the goal is a reproducible first baseline
    for the research pipeline, not a high-capacity neural model.
    """

    candidates = []
    for alpha, loss_aversion, temperature in itertools.product(
        (0.75, 1.0),
        (1.5, 2.25),
        (0.7, 1.2),
    ):
        for alpha_loss_streak_weight in (-0.35, 0.0, 0.35):
            for lambda_near_ruin_weight in (0.0, 1.0):
                for lambda_loss_streak_weight in (0.0, 0.25):
                    model = StateDependentProspectModel(
                        alpha=alpha,
                        loss_aversion=loss_aversion,
                        temperature=temperature,
                        alpha_loss_streak_weight=alpha_loss_streak_weight,
                        lambda_near_ruin_weight=lambda_near_ruin_weight,
                        lambda_loss_streak_weight=lambda_loss_streak_weight,
                    )
                    candidates.append((negative_log_likelihood(records, model, max_depth=max_depth), model))

    best_nll, best_model = min(candidates, key=lambda item: item[0])
    return FitResult(best_model.name, asdict(best_model), best_nll, len(records))


def fit_by_subject(
    records: Iterable[DecisionRecord],
    fitter: Callable[[list[DecisionRecord]], FitResult] = fit_static_prospect,
) -> dict[str, FitResult]:
    by_subject: dict[str, list[DecisionRecord]] = {}
    for record in records:
        by_subject.setdefault(record.subject_id, []).append(record)
    return {subject_id: fitter(subject_records) for subject_id, subject_records in by_subject.items()}
