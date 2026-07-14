"""Lower-confidence task-feature selector over policy delegates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, sqrt

from risk_shift_bench.envs import STANDARD_DECK, RiskTask
from risk_shift_bench.family_selector import family_candidate_lookup, family_candidate_policies
from risk_shift_bench.policies import BenchmarkPolicy


@dataclass(frozen=True)
class LCBSelectorParams:
    k: int = 5
    temperature: float = 0.50
    lcb_scale: float = 1.0
    margin: float = 0.0
    min_evidence: int = 2
    hidden_weight: float = 1.0
    shift_weight: float = 1.0
    bankroll_weight: float = 1.0
    horizon_weight: float = 1.0
    fallback_policy: str = "signed_regime_learned_ensemble"
    comparison_policies: tuple[str, ...] = ()


@dataclass(frozen=True)
class LCBSelectorProfile:
    task: str
    features: tuple[float, ...]
    policy_scores: dict[str, float]


@dataclass(frozen=True)
class LCBSearchResult:
    params: LCBSelectorParams
    validation_score: float
    train_profiles: list[dict]
    validation_summaries: list[dict]
    candidate_scores: list[dict]


def card_mean(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(card * prob for card, prob in card_probs)


def high_card_mass(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(prob for card, prob in card_probs if card >= 10)


def low_card_mass(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(prob for card, prob in card_probs if card <= 6)


def hidden_shift_stats(task: RiskTask) -> tuple[float, float, float]:
    if task.episode_card_regimes is None:
        return 0.0, 0.0, 0.0
    standard_mean = card_mean(STANDARD_DECK)
    shifts = [card_mean(card_probs) - standard_mean for card_probs, _prob in task.episode_card_regimes]
    return min(shifts), max(shifts), max(shifts) - min(shifts)


def task_features(task: RiskTask, params: LCBSelectorParams) -> tuple[float, ...]:
    standard_mean = card_mean(STANDARD_DECK)
    mean_shift = card_mean(task.card_probs) - standard_mean
    hidden_min, hidden_max, hidden_span = hidden_shift_stats(task)
    bankroll_bet_ratio = task.initial_bankroll / max(task.bet, 1.0)
    target_gap = (task.target_bankroll - task.initial_bankroll) / max(task.initial_bankroll, 1.0)
    hidden = 1.0 if task.episode_card_regimes is not None else 0.0
    return (
        params.horizon_weight * task.rounds / 60.0,
        params.bankroll_weight * bankroll_bet_ratio / 30.0,
        params.bankroll_weight * task.ruin_bankroll / max(task.initial_bankroll, 1.0),
        target_gap,
        task.drawdown_limit,
        params.shift_weight * mean_shift / 3.0,
        params.shift_weight * high_card_mass(task.card_probs),
        params.shift_weight * low_card_mass(task.card_probs),
        params.hidden_weight * hidden,
        params.hidden_weight * hidden_min / 3.0,
        params.hidden_weight * hidden_max / 3.0,
        params.hidden_weight * hidden_span / 4.0,
    )


def feature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def profiles_from_scores(
    tasks: list[RiskTask],
    scores_by_task: dict[str, dict[str, float]],
    params: LCBSelectorParams,
) -> list[LCBSelectorProfile]:
    profiles = []
    for task in tasks:
        if task.name in scores_by_task:
            profiles.append(
                LCBSelectorProfile(
                    task=task.name,
                    features=task_features(task, params),
                    policy_scores=scores_by_task[task.name],
                )
            )
    return profiles


class LowerConfidenceSelectorPolicy(BenchmarkPolicy):
    name = "lower_confidence_selector"

    def __init__(
        self,
        profiles: list[LCBSelectorProfile],
        params: LCBSelectorParams,
        policies: dict[str, BenchmarkPolicy] | None = None,
        name: str = "lower_confidence_selector",
    ) -> None:
        if not profiles:
            raise ValueError("lower-confidence selector requires at least one profile")
        self.profiles = profiles
        self.params = params
        self.policies = policies or family_candidate_lookup()
        self.name = name
        self._selection_cache: dict[str, str] = {}

    def _policy_lcb(
        self,
        task_features_: tuple[float, ...],
        policy_name: str,
        baseline_policy: str,
    ) -> tuple[float, float, int]:
        neighbors = []
        for profile in self.profiles:
            scores = profile.policy_scores
            if policy_name not in scores or baseline_policy not in scores:
                continue
            distance = feature_distance(task_features_, profile.features)
            delta = scores[policy_name] - scores[baseline_policy]
            neighbors.append((distance, delta))
        neighbors = sorted(neighbors, key=lambda item: item[0])[: max(1, self.params.k)]
        if len(neighbors) < self.params.min_evidence:
            return float("-inf"), float("-inf"), len(neighbors)
        weights = [exp(-distance / max(self.params.temperature, 1e-9)) for distance, _delta in neighbors]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return float("-inf"), float("-inf"), len(neighbors)
        deltas = [delta for _distance, delta in neighbors]
        mean_delta = sum(weight * delta for weight, delta in zip(weights, deltas)) / total_weight
        mean_square = sum(weight * delta * delta for weight, delta in zip(weights, deltas)) / total_weight
        variance = max(0.0, mean_square - mean_delta * mean_delta)
        effective_n = total_weight * total_weight / max(sum(weight * weight for weight in weights), 1e-12)
        lower_bound = mean_delta - self.params.lcb_scale * sqrt(variance / max(effective_n, 1.0))
        return lower_bound, mean_delta, len(neighbors)

    def selected_policy_name(self, task: RiskTask) -> str:
        if task.name in self._selection_cache:
            return self._selection_cache[task.name]
        features = task_features(task, self.params)
        best_policy = self.params.fallback_policy
        best_lcb = self.params.margin
        comparison_policies = self.params.comparison_policies or (self.params.fallback_policy,)
        for policy_name in sorted(self.policies):
            if policy_name == self.params.fallback_policy:
                continue
            lower_bounds = []
            for baseline_policy in comparison_policies:
                if baseline_policy == policy_name:
                    continue
                lower_bound, _mean_delta, _count = self._policy_lcb(features, policy_name, baseline_policy)
                lower_bounds.append(lower_bound)
            if not lower_bounds:
                continue
            robust_lower_bound = min(lower_bounds)
            if robust_lower_bound > best_lcb:
                best_lcb = robust_lower_bound
                best_policy = policy_name
        self._selection_cache[task.name] = best_policy
        return best_policy

    def action_probabilities(
        self,
        state,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        delegate = self.policies[self.selected_policy_name(task)]
        return delegate.action_probabilities(
            state,
            task=task,
            rounds_remaining=rounds_remaining,
            hand_depth=hand_depth,
            peak_bankroll=peak_bankroll,
        )


def cross_validated_score(
    tasks: list[RiskTask],
    scores_by_task: dict[str, dict[str, float]],
    params: LCBSelectorParams,
) -> tuple[float, list[dict]]:
    task_by_name = {task.name: task for task in tasks}
    profiles = profiles_from_scores(tasks, scores_by_task, params)
    rows = []
    scores = []
    for heldout in profiles:
        if heldout.task not in task_by_name:
            continue
        train_profiles = [profile for profile in profiles if profile.task != heldout.task]
        if not train_profiles:
            continue
        policy = LowerConfidenceSelectorPolicy(train_profiles, params)
        selected_policy = policy.selected_policy_name(task_by_name[heldout.task])
        if selected_policy not in heldout.policy_scores:
            selected_policy = params.fallback_policy
        selected_score = heldout.policy_scores[selected_policy]
        best_policy, best_score = max(heldout.policy_scores.items(), key=lambda item: (item[1], item[0]))
        fallback_score = heldout.policy_scores[params.fallback_policy]
        rows.append(
            {
                "task": heldout.task,
                "selected_policy": selected_policy,
                "selected_score": selected_score,
                "fallback_score": fallback_score,
                "delta_vs_fallback": selected_score - fallback_score,
                "best_policy": best_policy,
                "best_score": best_score,
                "regret": best_score - selected_score,
            }
        )
        scores.append(selected_score)
    return sum(scores) / len(scores) if scores else float("-inf"), rows


def risk_adjusted_validation_score(
    validation_score: float,
    validation_rows: list[dict],
    promotion_loss_weight: float = 1.0,
    worst_loss_weight: float = 0.25,
) -> float:
    losses = [
        max(0.0, -row["delta_vs_fallback"])
        for row in validation_rows
        if row["selected_policy"] != row.get("fallback_policy", "")
    ]
    if not losses:
        return validation_score
    mean_loss = sum(losses) / len(losses)
    worst_loss = max(losses)
    return validation_score - promotion_loss_weight * mean_loss - worst_loss_weight * worst_loss


def candidate_params(
    smoke: bool = False,
    fallback_policy: str = "signed_regime_learned_ensemble",
    comparison_policies: tuple[str, ...] = (),
) -> list[LCBSelectorParams]:
    candidates = []
    for k in (3, 5, 7):
        for temperature in (0.35, 0.50, 0.75):
            for lcb_scale in (0.5, 1.0, 1.5):
                for margin in (0.0, 1.0, 2.0, 4.0, 6.0):
                    candidates.append(
                        LCBSelectorParams(
                            k=k,
                            temperature=temperature,
                            lcb_scale=lcb_scale,
                            margin=margin,
                            min_evidence=2,
                            fallback_policy=fallback_policy,
                            comparison_policies=comparison_policies,
                        )
                    )
    candidates.extend(
        [
            LCBSelectorParams(
                k=5,
                temperature=0.50,
                lcb_scale=1.0,
                margin=1.0,
                hidden_weight=1.5,
                fallback_policy=fallback_policy,
                comparison_policies=comparison_policies,
            ),
            LCBSelectorParams(
                k=5,
                temperature=0.50,
                lcb_scale=1.0,
                margin=1.0,
                shift_weight=1.5,
                fallback_policy=fallback_policy,
                comparison_policies=comparison_policies,
            ),
            LCBSelectorParams(
                k=5,
                temperature=0.50,
                lcb_scale=1.0,
                margin=1.0,
                bankroll_weight=1.5,
                fallback_policy=fallback_policy,
                comparison_policies=comparison_policies,
            ),
            LCBSelectorParams(
                k=7,
                temperature=0.75,
                lcb_scale=1.5,
                margin=2.0,
                min_evidence=3,
                fallback_policy=fallback_policy,
                comparison_policies=comparison_policies,
            ),
            LCBSelectorParams(
                k=5,
                temperature=0.35,
                lcb_scale=1.5,
                margin=2.0,
                min_evidence=4,
                fallback_policy=fallback_policy,
                comparison_policies=comparison_policies,
            ),
            LCBSelectorParams(
                k=7,
                temperature=0.35,
                lcb_scale=2.0,
                margin=2.0,
                min_evidence=5,
                fallback_policy=fallback_policy,
                comparison_policies=comparison_policies,
            ),
        ]
    )
    return candidates[:2] if smoke else candidates


def search_lcb_selector(
    tasks: list[RiskTask],
    scores_by_task: dict[str, dict[str, float]],
    smoke: bool = False,
    robust_selection: bool = False,
    promotion_loss_weight: float = 1.0,
    worst_loss_weight: float = 0.25,
    fallback_policy: str = "signed_regime_learned_ensemble",
    comparison_policies: tuple[str, ...] = (),
) -> LCBSearchResult:
    results = []
    candidate_rows = []
    for index, params in enumerate(
        candidate_params(
            smoke=smoke,
            fallback_policy=fallback_policy,
            comparison_policies=comparison_policies,
        )
    ):
        validation_score, validation_rows = cross_validated_score(tasks, scores_by_task, params)
        selection_score = (
            risk_adjusted_validation_score(
                validation_score,
                validation_rows,
                promotion_loss_weight=promotion_loss_weight,
                worst_loss_weight=worst_loss_weight,
            )
            if robust_selection
            else validation_score
        )
        row = {
            "candidate": index,
            "validation_score": validation_score,
            "selection_score": selection_score,
            **asdict(params),
        }
        candidate_rows.append(row)
        profiles = profiles_from_scores(tasks, scores_by_task, params)
        results.append(
            LCBSearchResult(
                params=params,
                validation_score=selection_score,
                train_profiles=[asdict(profile) for profile in profiles],
                validation_summaries=validation_rows,
                candidate_scores=list(candidate_rows),
            )
        )
    if not results:
        raise RuntimeError("no lower-confidence selector candidates were evaluated")
    best_score = max(result.validation_score for result in results)
    eligible = [result for result in results if result.validation_score >= best_score - 2.5]
    guarded = [result for result in eligible if result.params.lcb_scale >= 1.0 and result.params.margin >= 1.0]
    best = max(guarded or eligible, key=lambda result: result.validation_score)
    return LCBSearchResult(
        params=best.params,
        validation_score=best.validation_score,
        train_profiles=best.train_profiles,
        validation_summaries=best.validation_summaries,
        candidate_scores=candidate_rows,
    )


def policy_from_scores(
    tasks: list[RiskTask],
    scores_by_task: dict[str, dict[str, float]],
    params: LCBSelectorParams,
    name: str = "lower_confidence_selector",
) -> LowerConfidenceSelectorPolicy:
    return LowerConfidenceSelectorPolicy(
        profiles=profiles_from_scores(tasks, scores_by_task, params),
        params=params,
        name=name,
    )
