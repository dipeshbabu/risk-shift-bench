"""Lower-confidence delegate selection for portfolio tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, sqrt

from risk_preference_inference.portfolio_benchmark import (
    PortfolioPolicy,
    downside_mass,
    expected_return,
    hidden_return_stats,
    portfolio_policy_lookup,
    visible_return_distribution,
)
from risk_preference_inference.portfolio_envs import PortfolioTask


@dataclass(frozen=True)
class PortfolioLCBParams:
    k: int = 5
    temperature: float = 0.75
    lcb_scale: float = 1.5
    margin: float = 4.0
    min_evidence: int = 2
    fallback_policy: str = "learned_mixture_searched"


@dataclass(frozen=True)
class PortfolioLCBProfile:
    task: str
    features: tuple[float, ...]
    policy_scores: dict[str, float]


@dataclass(frozen=True)
class PortfolioLCBSearchResult:
    params: PortfolioLCBParams
    selection_score: float
    train_profiles: list[dict]
    validation_summaries: list[dict]
    candidate_scores: list[dict]


def task_features(task: PortfolioTask) -> tuple[float, ...]:
    returns = visible_return_distribution(task)
    hidden_min, hidden_max, hidden_span = hidden_return_stats(task)
    return (
        task.periods / 60.0,
        task.initial_capital / 1200.0,
        task.ruin_capital / max(task.initial_capital, 1.0),
        (task.target_capital - task.initial_capital) / max(task.initial_capital, 1.0),
        task.drawdown_limit,
        expected_return(returns) / 0.04,
        downside_mass(returns),
        1.0 if task.episode_regimes is not None else 0.0,
        hidden_min / 0.04,
        hidden_max / 0.04,
        hidden_span / 0.08,
    )


def feature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def profiles_from_scores(
    tasks: list[PortfolioTask],
    scores_by_task: dict[str, dict[str, float]],
) -> list[PortfolioLCBProfile]:
    return [
        PortfolioLCBProfile(task=task.name, features=task_features(task), policy_scores=scores_by_task[task.name])
        for task in tasks
        if task.name in scores_by_task
    ]


class PortfolioLCBSelectorPolicy(PortfolioPolicy):
    name = "robust_searched_fallback_lcb"

    def __init__(
        self,
        profiles: list[PortfolioLCBProfile],
        params: PortfolioLCBParams,
        policies: dict[str, PortfolioPolicy] | None = None,
        name: str = "robust_searched_fallback_lcb",
    ) -> None:
        if not profiles:
            raise ValueError("portfolio LCB selector requires at least one profile")
        self.profiles = profiles
        self.params = params
        self.policies = policies or portfolio_policy_lookup()
        self.name = name
        self._selection_cache: dict[str, str] = {}

    def _policy_lcb(self, features: tuple[float, ...], policy_name: str) -> tuple[float, float, int]:
        neighbors = []
        for profile in self.profiles:
            scores = profile.policy_scores
            if policy_name not in scores or self.params.fallback_policy not in scores:
                continue
            distance = feature_distance(features, profile.features)
            delta = scores[policy_name] - scores[self.params.fallback_policy]
            neighbors.append((distance, delta))
        neighbors = sorted(neighbors, key=lambda item: item[0])[: max(1, self.params.k)]
        if len(neighbors) < self.params.min_evidence:
            return float("-inf"), float("-inf"), len(neighbors)
        weights = [exp(-distance / max(self.params.temperature, 1e-9)) for distance, _delta in neighbors]
        total_weight = sum(weights)
        deltas = [delta for _distance, delta in neighbors]
        mean_delta = sum(weight * delta for weight, delta in zip(weights, deltas)) / max(total_weight, 1e-12)
        mean_square = sum(weight * delta * delta for weight, delta in zip(weights, deltas)) / max(total_weight, 1e-12)
        variance = max(0.0, mean_square - mean_delta * mean_delta)
        effective_n = total_weight * total_weight / max(sum(weight * weight for weight in weights), 1e-12)
        lower_bound = mean_delta - self.params.lcb_scale * sqrt(variance / max(effective_n, 1.0))
        return lower_bound, mean_delta, len(neighbors)

    def selected_policy_name(self, task: PortfolioTask) -> str:
        if task.name in self._selection_cache:
            return self._selection_cache[task.name]
        features = task_features(task)
        best_policy = self.params.fallback_policy
        best_lcb = self.params.margin
        for policy_name in sorted(self.policies):
            if policy_name == self.params.fallback_policy:
                continue
            lower_bound, _mean_delta, _count = self._policy_lcb(features, policy_name)
            if lower_bound > best_lcb:
                best_policy = policy_name
                best_lcb = lower_bound
        self._selection_cache[task.name] = best_policy
        return best_policy

    def allocation(self, state, task: PortfolioTask) -> float:
        return self.policies[self.selected_policy_name(task)].allocation(state, task)


def cross_validated_score(
    tasks: list[PortfolioTask],
    scores_by_task: dict[str, dict[str, float]],
    params: PortfolioLCBParams,
) -> tuple[float, list[dict]]:
    task_by_name = {task.name: task for task in tasks}
    profiles = profiles_from_scores(tasks, scores_by_task)
    rows = []
    scores = []
    for heldout in profiles:
        train_profiles = [profile for profile in profiles if profile.task != heldout.task]
        if not train_profiles or heldout.task not in task_by_name:
            continue
        policy = PortfolioLCBSelectorPolicy(train_profiles, params)
        selected_policy = policy.selected_policy_name(task_by_name[heldout.task])
        if selected_policy not in heldout.policy_scores:
            selected_policy = params.fallback_policy
        selected_score = heldout.policy_scores[selected_policy]
        fallback_score = heldout.policy_scores[params.fallback_policy]
        rows.append(
            {
                "task": heldout.task,
                "selected_policy": selected_policy,
                "selected_score": selected_score,
                "fallback_score": fallback_score,
                "delta_vs_fallback": selected_score - fallback_score,
            }
        )
        scores.append(selected_score)
    return sum(scores) / len(scores) if scores else float("-inf"), rows


def risk_adjusted_score(validation_score: float, validation_rows: list[dict], promotion_loss_weight: float, worst_loss_weight: float) -> float:
    losses = [
        max(0.0, -row["delta_vs_fallback"])
        for row in validation_rows
        if row["selected_policy"] != row.get("fallback_policy", "")
    ]
    if not losses:
        return validation_score
    return validation_score - promotion_loss_weight * (sum(losses) / len(losses)) - worst_loss_weight * max(losses)


def candidate_params(fallback_policy: str = "learned_mixture_searched", smoke: bool = False) -> list[PortfolioLCBParams]:
    candidates = []
    for k in (3, 5, 7):
        for temperature in (0.35, 0.75, 1.25):
            for lcb_scale in (0.5, 1.0, 1.5, 2.0):
                for margin in (0.0, 2.0, 4.0, 6.0):
                    for min_evidence in (2, 3):
                        if min_evidence <= k:
                            candidates.append(
                                PortfolioLCBParams(
                                    k=k,
                                    temperature=temperature,
                                    lcb_scale=lcb_scale,
                                    margin=margin,
                                    min_evidence=min_evidence,
                                    fallback_policy=fallback_policy,
                                )
                            )
    return candidates[:2] if smoke else candidates


def search_portfolio_lcb_selector(
    tasks: list[PortfolioTask],
    scores_by_task: dict[str, dict[str, float]],
    fallback_policy: str = "learned_mixture_searched",
    robust_selection: bool = True,
    promotion_loss_weight: float = 4.0,
    worst_loss_weight: float = 2.0,
    smoke: bool = False,
) -> PortfolioLCBSearchResult:
    results = []
    candidate_rows = []
    for index, params in enumerate(candidate_params(fallback_policy=fallback_policy, smoke=smoke)):
        validation_score, validation_rows = cross_validated_score(tasks, scores_by_task, params)
        selection_score = (
            risk_adjusted_score(validation_score, validation_rows, promotion_loss_weight, worst_loss_weight)
            if robust_selection
            else validation_score
        )
        row = {"candidate": index, "validation_score": validation_score, "selection_score": selection_score, **asdict(params)}
        candidate_rows.append(row)
        results.append(
            PortfolioLCBSearchResult(
                params=params,
                selection_score=selection_score,
                train_profiles=[asdict(profile) for profile in profiles_from_scores(tasks, scores_by_task)],
                validation_summaries=validation_rows,
                candidate_scores=list(candidate_rows),
            )
        )
    if not results:
        raise RuntimeError("no portfolio LCB candidates were evaluated")
    best_score = max(result.selection_score for result in results)
    eligible = [result for result in results if result.selection_score >= best_score - 1.0]
    guarded = [result for result in eligible if result.params.margin >= 2.0 and result.params.lcb_scale >= 1.0]
    return max(guarded or eligible, key=lambda result: result.selection_score)


def policy_from_scores(
    tasks: list[PortfolioTask],
    scores_by_task: dict[str, dict[str, float]],
    params: PortfolioLCBParams,
    name: str = "robust_searched_fallback_lcb",
) -> PortfolioLCBSelectorPolicy:
    return PortfolioLCBSelectorPolicy(profiles_from_scores(tasks, scores_by_task), params=params, name=name)
