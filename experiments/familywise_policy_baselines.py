"""Familywise-valid baselines for RiskShiftBench v2 development.

The fixed-sample tests and alpha-spending confidence sequences in this module
use only bounded paired score differences. They provide transparent reference
points for the more powerful betting-mixture router.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, exp, log, pi, sqrt
from statistics import fmean

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    RouteDecision,
)


def fixed_sample_hoeffding_p_value(
    observations: list[float] | tuple[float, ...],
    *,
    null_mean: float,
    lower: float,
    upper: float,
) -> float:
    """One-sided bounded-mean p-value for a fixed sample size."""

    if not observations:
        raise ValueError("at least one observation is required")
    if lower >= upper:
        raise ValueError("observation bounds must be ordered")
    if any(not lower <= value <= upper for value in observations):
        raise ValueError("observation lies outside the declared bounds")
    advantage = max(fmean(observations) - null_mean, 0.0)
    width = upper - lower
    return min(1.0, exp(-2.0 * len(observations) * advantage**2 / width**2))


def exact_sign_p_value(
    observations: list[float] | tuple[float, ...],
    *,
    null_value: float = 0.0,
) -> float:
    """Exact one-sided sign-test p-value after discarding ties."""

    nonzero = [value for value in observations if value != null_value]
    if not nonzero:
        return 1.0
    positive = sum(value > null_value for value in nonzero)
    return sum(comb(len(nonzero), count) for count in range(positive, len(nonzero) + 1)) / (
        2 ** len(nonzero)
    )


def bonferroni_rejections(
    p_values: dict[str, float], familywise_alpha: float = 0.05
) -> set[str]:
    if not p_values:
        return set()
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one")
    if any(not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("p-values must lie between zero and one")
    local_alpha = familywise_alpha / len(p_values)
    return {task for task, p_value in p_values.items() if p_value <= local_alpha}


def holm_rejections(
    p_values: dict[str, float], familywise_alpha: float = 0.05
) -> set[str]:
    if not p_values:
        return set()
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    if any(not 0.0 <= value <= 1.0 for _task, value in ordered):
        raise ValueError("p-values must lie between zero and one")
    rejected: set[str] = set()
    family_size = len(ordered)
    for index, (task, p_value) in enumerate(ordered):
        if p_value > familywise_alpha / (family_size - index):
            break
        rejected.add(task)
    return rejected


def fixed_sample_rejections(
    observations: dict[str, list[float] | tuple[float, ...]],
    *,
    test: str,
    correction: str,
    familywise_alpha: float,
    null_mean: float,
    lower: float,
    upper: float,
) -> set[str]:
    if test == "hoeffding":
        p_values = {
            task: fixed_sample_hoeffding_p_value(
                values,
                null_mean=null_mean,
                lower=lower,
                upper=upper,
            )
            for task, values in observations.items()
        }
    elif test == "sign":
        p_values = {
            task: exact_sign_p_value(values, null_value=null_mean)
            for task, values in observations.items()
        }
    else:
        raise ValueError("test must be 'hoeffding' or 'sign'")

    if correction == "bonferroni":
        return bonferroni_rejections(p_values, familywise_alpha)
    if correction == "holm":
        return holm_rejections(p_values, familywise_alpha)
    raise ValueError("correction must be 'bonferroni' or 'holm'")


def alpha_spending_radius(
    observations: int,
    *,
    alpha: float,
    width: float,
) -> float:
    """One-sided time-uniform Hoeffding radius using 1/n^2 spending."""

    if observations <= 0:
        return float("inf")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if width <= 0.0:
        raise ValueError("width must be positive")
    spending_probability = 6.0 * alpha / (pi**2 * observations**2)
    return width * sqrt(log(1.0 / spending_probability) / (2.0 * observations))


def task_resolution_bound(
    gap: float,
    *,
    alpha: float,
    width: float,
    search_limit: int = 10_000_000,
) -> int:
    """First n for which simultaneous coverage forces a correct decision.

    On the confidence-sequence coverage event, the empirical mean can differ
    from the true mean by one radius. The decision bound uses another radius,
    so ``2 * radius < gap`` is sufficient.
    """

    if gap <= 0.0:
        raise ValueError("gap must be positive")
    if search_limit <= 0:
        raise ValueError("search_limit must be positive")
    for observations in range(1, search_limit + 1):
        if 2.0 * alpha_spending_radius(
            observations, alpha=alpha, width=width
        ) < gap:
            return observations
    raise RuntimeError("resolution bound exceeds search_limit")


@dataclass(frozen=True)
class ConfidenceSequenceEvidence:
    task: str
    decision: RouteDecision
    observations: int
    mean_difference: float
    lower_confidence_bound: float
    upper_confidence_bound: float


class AlphaSpendingFamilywiseRouter:
    """Successive-elimination router with explicit time-uniform bounds."""

    def __init__(self, plan: AnytimeFamilywisePlan):
        self.plan = plan
        self._counts = {task: 0 for task in plan.task_names}
        self._sums = {task: 0.0 for task in plan.task_names}
        self._decisions = {
            task: RouteDecision.UNDECIDED for task in plan.task_names
        }

    def _validate_task(self, task: str) -> None:
        if task not in self._counts:
            raise KeyError(f"unknown proposal task: {task}")

    def evidence(self, task: str) -> ConfidenceSequenceEvidence:
        self._validate_task(task)
        count = self._counts[task]
        mean = self._sums[task] / count if count else 0.0
        lower_radius = alpha_spending_radius(
            count,
            alpha=self.plan.acceptance_alpha(task),
            width=self.plan.observation_width,
        )
        upper_radius = alpha_spending_radius(
            count,
            alpha=self.plan.futility_alpha(task),
            width=self.plan.observation_width,
        )
        return ConfidenceSequenceEvidence(
            task=task,
            decision=self._decisions[task],
            observations=count,
            mean_difference=mean,
            lower_confidence_bound=mean - lower_radius,
            upper_confidence_bound=mean + upper_radius,
        )

    def update(self, task: str, paired_score_difference: float) -> ConfidenceSequenceEvidence:
        self._validate_task(task)
        if self._decisions[task] is not RouteDecision.UNDECIDED:
            raise RuntimeError(f"task {task} already has a terminal route decision")
        value = float(paired_score_difference)
        if not self.plan.observation_lower <= value <= self.plan.observation_upper:
            raise ValueError("observation lies outside the declared bounds")
        self._counts[task] += 1
        self._sums[task] += value
        evidence = self.evidence(task)
        if (
            evidence.observations >= self.plan.minimum_observations
            and evidence.lower_confidence_bound > self.plan.effect_margin
        ):
            self._decisions[task] = RouteDecision.ACCEPT_CANDIDATE
        elif (
            evidence.observations >= self.plan.minimum_observations
            and evidence.upper_confidence_bound < self.plan.effect_margin
        ):
            self._decisions[task] = RouteDecision.REJECT_TO_FALLBACK
        elif evidence.observations >= self.plan.maximum_observations_per_task:
            self._decisions[task] = RouteDecision.BUDGET_EXHAUSTED
        return self.evidence(task)

    def next_task(self) -> str | None:
        unresolved = [
            task
            for task in self.plan.task_names
            if self._decisions[task] is RouteDecision.UNDECIDED
        ]
        if not unresolved:
            return None

        def priority(task: str) -> tuple[float, str]:
            evidence = self.evidence(task)
            width = (
                evidence.upper_confidence_bound
                - evidence.lower_confidence_bound
            )
            return (-width, task)

        return min(unresolved, key=priority)

    def decisions(self) -> dict[str, ConfidenceSequenceEvidence]:
        return {task: self.evidence(task) for task in self.plan.task_names}

    def total_observations(self) -> int:
        return sum(self._counts.values())

