"""Conformal regret-controlled routing over a fixed policy library.

The predictor is fitted on one task split and calibrated on a disjoint task
split.  Calibration uses the maximum policy-wise overprediction on each task,
so the resulting one-sided correction accounts for choosing among delegates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, exp, sqrt
from typing import Callable, Generic, TypeVar


TaskT = TypeVar("TaskT")


@dataclass(frozen=True)
class RouterParams:
    k: int = 5
    temperature: float = 0.75
    alpha: float = 0.10
    margin: float = 0.0
    min_fit_evidence: int = 3
    min_calibration_tasks: int = 5
    screen_min_mean_advantage: float = 0.0
    max_screened_candidates: int = 1
    fallback_policy: str = "learned_mixture_searched"


@dataclass(frozen=True)
class RoutingProfile:
    task: str
    features: tuple[float, ...]
    policy_scores: dict[str, float]


@dataclass(frozen=True)
class Prediction:
    policy: str
    predicted_advantage: float
    lower_bound: float
    support_radius: float
    effective_n: float
    neighbor_min_advantage: float
    neighbor_max_advantage: float


@dataclass(frozen=True)
class RoutingDecision:
    selected_policy: str
    promoted: bool
    prediction: Prediction | None
    reason: str


@dataclass(frozen=True)
class CalibrationReport:
    alpha: float
    n_calibration_tasks: int
    conformal_correction: float
    support_radius_limit: float
    calibration_residuals: tuple[float, ...]
    calibration_support_radii: tuple[float, ...]
    candidate_library: tuple[str, ...]
    candidate_policies: tuple[str, ...]
    fit_mean_advantages: tuple[tuple[str, float], ...]


def feature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def finite_sample_upper_quantile(values: list[float], alpha: float) -> float:
    """Return the split-conformal upper quantile using the higher order statistic."""

    if not values:
        raise ValueError("a conformal quantile requires at least one value")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    ordered = sorted(float(value) for value in values)
    rank = ceil((len(ordered) + 1) * (1.0 - alpha))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def build_profiles(
    tasks: list[TaskT],
    scores_by_task: dict[str, dict[str, float]],
    feature_fn: Callable[[TaskT], tuple[float, ...]],
) -> list[RoutingProfile]:
    return [
        RoutingProfile(
            task=task.name,
            features=feature_fn(task),
            policy_scores=dict(scores_by_task[task.name]),
        )
        for task in tasks
        if task.name in scores_by_task
    ]


class ConformalAdvantageRouter(Generic[TaskT]):
    """Select a delegate only when its simultaneous calibrated bound is positive."""

    def __init__(
        self,
        fit_profiles: list[RoutingProfile],
        calibration_profiles: list[RoutingProfile],
        candidate_policies: tuple[str, ...],
        params: RouterParams,
        feature_fn: Callable[[TaskT], tuple[float, ...]],
    ) -> None:
        if not fit_profiles:
            raise ValueError("the router requires fit profiles")
        if not calibration_profiles:
            raise ValueError("the router requires a disjoint calibration split")
        if params.fallback_policy in candidate_policies:
            raise ValueError("candidate_policies must exclude the fallback")
        if not candidate_policies:
            raise ValueError("the router requires at least one candidate policy")
        overlap = {profile.task for profile in fit_profiles} & {
            profile.task for profile in calibration_profiles
        }
        if overlap:
            raise ValueError(f"fit and calibration tasks overlap: {sorted(overlap)}")
        self.fit_profiles = list(fit_profiles)
        self.calibration_profiles = list(calibration_profiles)
        self.candidate_library = tuple(candidate_policies)
        self.params = params
        self.feature_fn = feature_fn
        self.fit_mean_advantages = self._fit_mean_advantages()
        screened = [
            policy
            for policy in self.candidate_library
            if self.fit_mean_advantages[policy] > self.params.screen_min_mean_advantage
        ]
        screened.sort(key=lambda policy: (-self.fit_mean_advantages[policy], policy))
        self.candidate_policies = tuple(screened[: self.params.max_screened_candidates])
        if not self.candidate_policies:
            raise ValueError("no candidate policy passed the fit-only advantage screen")
        self.calibration = self._calibrate()

    def _fit_mean_advantages(self) -> dict[str, float]:
        output = {}
        for policy in self.candidate_library:
            advantages = [
                profile.policy_scores[policy] - profile.policy_scores[self.params.fallback_policy]
                for profile in self.fit_profiles
                if policy in profile.policy_scores
                and self.params.fallback_policy in profile.policy_scores
            ]
            if len(advantages) < self.params.min_fit_evidence:
                output[policy] = float("-inf")
            else:
                output[policy] = sum(advantages) / len(advantages)
        return output

    def _raw_prediction(self, features: tuple[float, ...], policy: str) -> tuple[float, float, float, float, float] | None:
        neighbors = []
        for profile in self.fit_profiles:
            scores = profile.policy_scores
            if policy not in scores or self.params.fallback_policy not in scores:
                continue
            distance = feature_distance(features, profile.features)
            advantage = scores[policy] - scores[self.params.fallback_policy]
            neighbors.append((distance, advantage))
        neighbors.sort(key=lambda item: item[0])
        neighbors = neighbors[: max(1, self.params.k)]
        if len(neighbors) < self.params.min_fit_evidence:
            return None
        weights = [
            exp(-distance / max(self.params.temperature, 1e-12))
            for distance, _advantage in neighbors
        ]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return None
        advantages = [advantage for _distance, advantage in neighbors]
        predicted = sum(weight * advantage for weight, advantage in zip(weights, advantages)) / total_weight
        effective_n = total_weight * total_weight / max(sum(weight * weight for weight in weights), 1e-12)
        return predicted, neighbors[-1][0], effective_n, min(advantages), max(advantages)

    def _calibrate(self) -> CalibrationReport:
        residuals = []
        support_radii = []
        for profile in self.calibration_profiles:
            if self.params.fallback_policy not in profile.policy_scores:
                raise ValueError(f"fallback score missing on calibration task {profile.task}")
            task_residuals = []
            task_support = []
            for policy in self.candidate_policies:
                if policy not in profile.policy_scores:
                    raise ValueError(f"candidate {policy} missing on calibration task {profile.task}")
                raw = self._raw_prediction(profile.features, policy)
                if raw is None:
                    raise ValueError(f"insufficient fit evidence for {policy} on calibration task {profile.task}")
                predicted, support_radius, _effective_n, _minimum, _maximum = raw
                observed = profile.policy_scores[policy] - profile.policy_scores[self.params.fallback_policy]
                task_residuals.append(predicted - observed)
                task_support.append(support_radius)
            residuals.append(max(task_residuals))
            support_radii.append(max(task_support))
        if len(residuals) < self.params.min_calibration_tasks:
            raise ValueError(
                f"router requires {self.params.min_calibration_tasks} calibration tasks; "
                f"found {len(residuals)}"
            )
        correction = max(0.0, finite_sample_upper_quantile(residuals, self.params.alpha))
        # Refuse extrapolation beyond every radius seen on the calibration split.
        support_limit = max(support_radii)
        return CalibrationReport(
            alpha=self.params.alpha,
            n_calibration_tasks=len(residuals),
            conformal_correction=correction,
            support_radius_limit=support_limit,
            calibration_residuals=tuple(residuals),
            calibration_support_radii=tuple(support_radii),
            candidate_library=self.candidate_library,
            candidate_policies=self.candidate_policies,
            fit_mean_advantages=tuple(sorted(self.fit_mean_advantages.items())),
        )

    def predictions(self, task: TaskT) -> list[Prediction]:
        features = self.feature_fn(task)
        predictions = []
        for policy in self.candidate_policies:
            raw = self._raw_prediction(features, policy)
            if raw is None:
                continue
            predicted, support_radius, effective_n, minimum, maximum = raw
            lower_bound = predicted - self.calibration.conformal_correction
            predictions.append(
                Prediction(
                    policy=policy,
                    predicted_advantage=predicted,
                    lower_bound=lower_bound,
                    support_radius=support_radius,
                    effective_n=effective_n,
                    neighbor_min_advantage=minimum,
                    neighbor_max_advantage=maximum,
                )
            )
        return predictions

    def decision(self, task: TaskT) -> RoutingDecision:
        supported = [
            prediction
            for prediction in self.predictions(task)
            if prediction.support_radius <= self.calibration.support_radius_limit
        ]
        if not supported:
            return RoutingDecision(
                selected_policy=self.params.fallback_policy,
                promoted=False,
                prediction=None,
                reason="outside_calibration_support",
            )
        best = max(supported, key=lambda item: (item.lower_bound, item.predicted_advantage, item.policy))
        if best.lower_bound <= self.params.margin:
            return RoutingDecision(
                selected_policy=self.params.fallback_policy,
                promoted=False,
                prediction=best,
                reason="calibrated_bound_below_margin",
            )
        return RoutingDecision(
            selected_policy=best.policy,
            promoted=True,
            prediction=best,
            reason="positive_simultaneous_lower_bound",
        )

    def proposal(self, task: TaskT) -> RoutingDecision:
        """Return a high-recall fit-only proposal for independent pilot verification."""

        supported = [
            prediction
            for prediction in self.predictions(task)
            if prediction.support_radius <= self.calibration.support_radius_limit
        ]
        if not supported:
            return RoutingDecision(
                selected_policy=self.params.fallback_policy,
                promoted=False,
                prediction=None,
                reason="outside_calibration_support",
            )
        best = max(supported, key=lambda item: (item.predicted_advantage, item.policy))
        if best.predicted_advantage <= self.params.margin:
            return RoutingDecision(
                selected_policy=self.params.fallback_policy,
                promoted=False,
                prediction=best,
                reason="fit_prediction_below_margin",
            )
        return RoutingDecision(
            selected_policy=best.policy,
            promoted=True,
            prediction=best,
            reason="positive_fit_prediction_pending_pilot_verification",
        )

    def report_dict(self) -> dict:
        return {
            "params": asdict(self.params),
            "calibration": asdict(self.calibration),
            "guarantee_scope": (
                "If calibration tasks and a future task are exchangeable, the fixed fit predictor and "
                "maximum-over-policy residual give marginal simultaneous one-sided coverage at 1-alpha."
            ),
        }
