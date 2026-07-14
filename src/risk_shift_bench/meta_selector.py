"""Task-feature meta-selection over strong incumbent policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, sqrt

from risk_shift_bench.adaptive_search import summary_score
from risk_shift_bench.benchmark import BenchmarkSummary, run_benchmark
from risk_shift_bench.envs import STANDARD_DECK, RiskTask
from risk_shift_bench.objectives import EntropicObjective, MeanObjective, OCEObjective
from risk_shift_bench.policies import BenchmarkPolicy, RegimeAdaptivePolicy, StaticObjectivePolicy
from risk_shift_bench.policy_registry import (
    learned_mixture_policy,
    searched_learned_mixture_policy,
    signed_regime_learned_policy,
    state_adaptive_utility_policy,
)


@dataclass(frozen=True)
class MetaSelectorParams:
    k: int = 5
    temperature: float = 0.30
    hidden_weight: float = 1.0
    shift_weight: float = 1.0
    bankroll_weight: float = 1.0
    horizon_weight: float = 1.0
    margin: float = 0.0
    instability_penalty: float = 0.0
    pairwise_regret_penalty: float = 0.0
    min_agreement: float = 0.0
    fallback_policy: str = "signed_regime_learned_ensemble"


@dataclass(frozen=True)
class MetaSelectorProfile:
    task: str
    features: tuple[float, ...]
    policy_scores: dict[str, float]
    policy_advantages: dict[str, float]


@dataclass(frozen=True)
class MetaSelectorSearchResult:
    params: MetaSelectorParams
    validation_score: float
    train_profiles: list[dict]
    validation_summaries: list[dict]
    candidate_scores: list[dict]


CONSERVATIVE_SELECTION_TOLERANCE = 2.5


def card_mean(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(card * prob for card, prob in card_probs)


def high_card_mass(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(prob for card, prob in card_probs if card >= 10)


def low_card_mass(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(prob for card, prob in card_probs if card <= 6)


def regime_shift_stats(task: RiskTask) -> tuple[float, float, float]:
    if task.episode_card_regimes is None:
        return 0.0, 0.0, 0.0
    standard_mean = card_mean(STANDARD_DECK)
    shifts = [card_mean(card_probs) - standard_mean for card_probs, _prob in task.episode_card_regimes]
    return min(shifts), max(shifts), max(shifts) - min(shifts)


def task_features(task: RiskTask, params: MetaSelectorParams) -> tuple[float, ...]:
    standard_mean = card_mean(STANDARD_DECK)
    mean_shift = card_mean(task.card_probs) - standard_mean
    hidden_min_shift, hidden_max_shift, hidden_span = regime_shift_stats(task)
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
        params.hidden_weight * hidden_min_shift / 3.0,
        params.hidden_weight * hidden_max_shift / 3.0,
        params.hidden_weight * hidden_span / 4.0,
    )


def feature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def meta_candidate_policies() -> list[BenchmarkPolicy]:
    return [
        StaticObjectivePolicy(MeanObjective(), name="expected_value"),
        StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="fixed_entropic_0.025"),
        StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="fixed_oce_3"),
        state_adaptive_utility_policy(name="adaptive_utility_default"),
        learned_mixture_policy(name="learned_mixture_default"),
        searched_learned_mixture_policy(),
        RegimeAdaptivePolicy(),
        signed_regime_learned_policy(),
    ]


def meta_candidate_lookup() -> dict[str, BenchmarkPolicy]:
    return {policy.name: policy for policy in meta_candidate_policies()}


class AdvantageKnnMetaPolicy(BenchmarkPolicy):
    name = "advantage_knn_meta_selector"

    def __init__(
        self,
        profiles: list[MetaSelectorProfile],
        params: MetaSelectorParams,
        policies: dict[str, BenchmarkPolicy] | None = None,
        name: str = "advantage_knn_meta_selector",
    ) -> None:
        if not profiles:
            raise ValueError("meta-selector requires at least one profile")
        self.profiles = profiles
        self.params = params
        self.policies = policies or meta_candidate_lookup()
        self.name = name

    def selected_policy_name(self, task: RiskTask) -> str:
        features = task_features(task, self.params)
        neighbors = sorted(
            ((feature_distance(features, profile.features), profile) for profile in self.profiles),
            key=lambda item: (item[0], item[1].task),
        )[: max(1, self.params.k)]
        estimates = {policy_name: 0.0 for policy_name in self.policies}
        squared_estimates = {policy_name: 0.0 for policy_name in self.policies}
        pairwise_delta_sums = {policy_name: 0.0 for policy_name in self.policies}
        pairwise_delta_square_sums = {policy_name: 0.0 for policy_name in self.policies}
        winner_votes = {policy_name: 0.0 for policy_name in self.policies}
        total_weight = 0.0
        for distance, profile in neighbors:
            weight = exp(-distance / max(self.params.temperature, 1e-9))
            total_weight += weight
            neighbor_winner = max(profile.policy_advantages.items(), key=lambda item: (item[1], item[0]))[0]
            winner_votes[neighbor_winner] = winner_votes.get(neighbor_winner, 0.0) + weight
            fallback_advantage = profile.policy_advantages.get(self.params.fallback_policy, 0.0)
            for policy_name in estimates:
                advantage = profile.policy_advantages.get(policy_name, 0.0)
                estimates[policy_name] += weight * advantage
                squared_estimates[policy_name] += weight * advantage * advantage
                delta = advantage - fallback_advantage
                pairwise_delta_sums[policy_name] += weight * delta
                pairwise_delta_square_sums[policy_name] += weight * delta * delta
        if total_weight <= 0.0:
            return self.params.fallback_policy
        estimates = {policy_name: score / total_weight for policy_name, score in estimates.items()}
        if self.params.instability_penalty > 0.0:
            penalized_estimates = {}
            for policy_name, estimate in estimates.items():
                mean_square = squared_estimates[policy_name] / total_weight
                variance = max(0.0, mean_square - estimate * estimate)
                penalized_estimates[policy_name] = estimate - self.params.instability_penalty * sqrt(variance)
        else:
            penalized_estimates = estimates
        best_policy, best_score = max(penalized_estimates.items(), key=lambda item: (item[1], item[0]))
        fallback_score = penalized_estimates.get(self.params.fallback_policy, float("-inf"))
        agreement = winner_votes.get(best_policy, 0.0) / total_weight
        if best_policy != self.params.fallback_policy and agreement < self.params.min_agreement:
            return self.params.fallback_policy
        if best_policy != self.params.fallback_policy and self.params.pairwise_regret_penalty > 0.0:
            mean_delta = pairwise_delta_sums[best_policy] / total_weight
            mean_square_delta = pairwise_delta_square_sums[best_policy] / total_weight
            variance_delta = max(0.0, mean_square_delta - mean_delta * mean_delta)
            lower_confidence_delta = mean_delta - self.params.pairwise_regret_penalty * sqrt(variance_delta)
            if lower_confidence_delta < self.params.margin:
                return self.params.fallback_policy
        if best_policy != self.params.fallback_policy and best_score - fallback_score < self.params.margin:
            return self.params.fallback_policy
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


def build_profiles(
    tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    params: MetaSelectorParams,
) -> tuple[list[MetaSelectorProfile], list[dict]]:
    scores_by_task, raw_rows = collect_policy_scores(
        tasks=tasks,
        seeds=seeds,
        episodes=episodes,
        hand_depth=hand_depth,
    )
    return profiles_from_scores(tasks=tasks, scores_by_task=scores_by_task, params=params), raw_rows


def collect_policy_scores(
    tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    policies = meta_candidate_policies()
    summaries_by_task_policy: dict[tuple[str, str], list[BenchmarkSummary]] = {}
    raw_rows: list[dict] = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(tasks=tasks, policies=policies, episodes=episodes, seed=seed, hand_depth=hand_depth)
        for summary in summaries:
            summaries_by_task_policy.setdefault((summary.task, summary.policy), []).append(summary)
            row = asdict(summary)
            row["seed"] = seed
            row["score"] = summary_score(summary)
            raw_rows.append(row)

    scores_by_task: dict[str, dict[str, float]] = {}
    for (task_name, policy_name), summaries in summaries_by_task_policy.items():
        scores_by_task.setdefault(task_name, {})[policy_name] = sum(summary_score(summary) for summary in summaries) / len(summaries)
    return scores_by_task, raw_rows


def profiles_from_scores(
    tasks: list[RiskTask],
    scores_by_task: dict[str, dict[str, float]],
    params: MetaSelectorParams,
) -> list[MetaSelectorProfile]:
    task_by_name = {task.name: task for task in tasks}
    profiles = []
    for task_name, policy_scores in sorted(scores_by_task.items()):
        task_mean = sum(policy_scores.values()) / len(policy_scores)
        advantages = {policy_name: score - task_mean for policy_name, score in policy_scores.items()}
        profiles.append(
            MetaSelectorProfile(
                task=task_name,
                features=task_features(task_by_name[task_name], params),
                policy_scores=policy_scores,
                policy_advantages=advantages,
            )
        )
    return profiles


def evaluate_meta_selector(
    tasks: list[RiskTask],
    profiles: list[MetaSelectorProfile],
    params: MetaSelectorParams,
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    name: str = "advantage_knn_meta_selector",
) -> tuple[float, list[dict]]:
    policy = AdvantageKnnMetaPolicy(profiles=profiles, params=params, name=name)
    rows = []
    scores = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
        for summary in summaries:
            row = asdict(summary)
            row["seed"] = seed
            row["score"] = summary_score(summary)
            row["selected_policy"] = policy.selected_policy_name(next(task for task in tasks if task.name == summary.task))
            rows.append(row)
            scores.append(row["score"])
    return sum(scores) / len(scores) if scores else float("-inf"), rows


def cross_validated_profile_score(
    tasks: list[RiskTask],
    profiles: list[MetaSelectorProfile],
    params: MetaSelectorParams,
) -> tuple[float, list[dict]]:
    task_by_name = {task.name: task for task in tasks}
    rows = []
    scores = []
    for heldout in profiles:
        train_profiles = [profile for profile in profiles if profile.task != heldout.task]
        if not train_profiles:
            continue
        policy = AdvantageKnnMetaPolicy(profiles=train_profiles, params=params, name="advantage_knn_meta_selector_cv")
        selected_policy = policy.selected_policy_name(task_by_name[heldout.task])
        selected_score = heldout.policy_scores[selected_policy]
        best_policy, best_score = max(heldout.policy_scores.items(), key=lambda item: (item[1], item[0]))
        row = {
            "task": heldout.task,
            "selected_policy": selected_policy,
            "selected_score": selected_score,
            "best_policy": best_policy,
            "best_score": best_score,
            "regret": best_score - selected_score,
        }
        rows.append(row)
        scores.append(selected_score)
    return sum(scores) / len(scores) if scores else float("-inf"), rows


def meta_selector_candidate_params(smoke: bool = False) -> list[MetaSelectorParams]:
    candidates = [
        MetaSelectorParams(k=1, temperature=0.20, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=3, temperature=0.30, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=5, temperature=0.40, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=3, temperature=0.30, hidden_weight=1.5, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=3, temperature=0.30, shift_weight=1.5, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=3, temperature=0.30, bankroll_weight=1.5, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=5, temperature=0.50, margin=2.0, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=5, temperature=0.50, margin=4.0, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=5, temperature=0.50, margin=4.0, min_agreement=0.45, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(k=5, temperature=0.50, margin=6.0, min_agreement=0.45, fallback_policy="signed_regime_learned_ensemble"),
        MetaSelectorParams(
            k=5,
            temperature=0.50,
            margin=4.0,
            instability_penalty=0.25,
            fallback_policy="signed_regime_learned_ensemble",
        ),
        MetaSelectorParams(
            k=5,
            temperature=0.50,
            margin=4.0,
            instability_penalty=0.50,
            fallback_policy="signed_regime_learned_ensemble",
        ),
        MetaSelectorParams(
            k=5,
            temperature=0.50,
            margin=6.0,
            instability_penalty=0.25,
            min_agreement=0.45,
            fallback_policy="signed_regime_learned_ensemble",
        ),
        MetaSelectorParams(
            k=5,
            temperature=0.50,
            margin=2.0,
            pairwise_regret_penalty=0.50,
            min_agreement=0.35,
            fallback_policy="signed_regime_learned_ensemble",
        ),
        MetaSelectorParams(
            k=5,
            temperature=0.50,
            margin=4.0,
            pairwise_regret_penalty=0.75,
            min_agreement=0.45,
            fallback_policy="signed_regime_learned_ensemble",
        ),
        MetaSelectorParams(
            k=7,
            temperature=0.65,
            margin=2.0,
            pairwise_regret_penalty=1.00,
            min_agreement=0.35,
            fallback_policy="signed_regime_learned_ensemble",
        ),
        MetaSelectorParams(
            k=7,
            temperature=0.65,
            margin=4.0,
            pairwise_regret_penalty=1.00,
            min_agreement=0.45,
            fallback_policy="signed_regime_learned_ensemble",
        ),
        MetaSelectorParams(k=3, temperature=0.30, fallback_policy="learned_mixture_searched"),
        MetaSelectorParams(k=5, temperature=0.50, margin=2.0, fallback_policy="learned_mixture_searched"),
        MetaSelectorParams(k=3, temperature=0.30, fallback_policy="adaptive_utility_default"),
        MetaSelectorParams(k=5, temperature=0.50, margin=2.0, fallback_policy="adaptive_utility_default"),
    ]
    return candidates[:2] if smoke else candidates


def select_meta_search_result(
    indexed_results: list[tuple[int, MetaSelectorSearchResult]],
    tolerance: float = CONSERVATIVE_SELECTION_TOLERANCE,
) -> MetaSelectorSearchResult:
    """Prefer regret-guarded signed-regime candidates near the best CV score."""
    if not indexed_results:
        raise RuntimeError("no meta-selector candidates were evaluated")
    best_score = max(result.validation_score for _index, result in indexed_results)
    eligible = [
        (index, result)
        for index, result in indexed_results
        if result.validation_score >= best_score - tolerance
    ]
    guarded = [
        (index, result)
        for index, result in eligible
        if result.params.fallback_policy == "signed_regime_learned_ensemble"
        and result.params.pairwise_regret_penalty > 0.0
    ]
    candidates = guarded or eligible
    return max(candidates, key=lambda item: (item[1].validation_score, -item[0]))[1]


def search_meta_selector_cv(
    tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    smoke: bool = False,
) -> MetaSelectorSearchResult:
    results: list[tuple[int, MetaSelectorSearchResult]] = []
    candidate_scores = []
    scores_by_task, _train_rows = collect_policy_scores(
        tasks=tasks,
        seeds=seeds,
        episodes=episodes,
        hand_depth=hand_depth,
    )
    for index, params in enumerate(meta_selector_candidate_params(smoke=smoke)):
        profiles = profiles_from_scores(tasks=tasks, scores_by_task=scores_by_task, params=params)
        validation_score, validation_rows = cross_validated_profile_score(tasks=tasks, profiles=profiles, params=params)
        candidate_row = {"candidate": index, "validation_score": validation_score, **asdict(params)}
        candidate_scores.append(candidate_row)
        result = MetaSelectorSearchResult(
            params=params,
            validation_score=validation_score,
            train_profiles=[asdict(profile) for profile in profiles],
            validation_summaries=validation_rows,
            candidate_scores=list(candidate_scores),
        )
        results.append((index, result))
    best = select_meta_search_result(results)
    return MetaSelectorSearchResult(
        params=best.params,
        validation_score=best.validation_score,
        train_profiles=best.train_profiles,
        validation_summaries=best.validation_summaries,
        candidate_scores=candidate_scores,
    )


def search_meta_selector(
    train_tasks: list[RiskTask],
    validation_tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    smoke: bool = False,
) -> MetaSelectorSearchResult:
    results: list[tuple[int, MetaSelectorSearchResult]] = []
    candidate_scores = []
    for index, params in enumerate(meta_selector_candidate_params(smoke=smoke)):
        profiles, _train_rows = build_profiles(
            tasks=train_tasks,
            seeds=seeds,
            episodes=episodes,
            hand_depth=hand_depth,
            params=params,
        )
        validation_score, validation_rows = evaluate_meta_selector(
            tasks=validation_tasks,
            profiles=profiles,
            params=params,
            seeds=seeds,
            episodes=episodes,
            hand_depth=hand_depth,
        )
        candidate_row = {"candidate": index, "validation_score": validation_score, **asdict(params)}
        candidate_scores.append(candidate_row)
        result = MetaSelectorSearchResult(
            params=params,
            validation_score=validation_score,
            train_profiles=[asdict(profile) for profile in profiles],
            validation_summaries=validation_rows,
            candidate_scores=list(candidate_scores),
        )
        results.append((index, result))
    best = select_meta_search_result(results)
    return MetaSelectorSearchResult(
        params=best.params,
        validation_score=best.validation_score,
        train_profiles=best.train_profiles,
        validation_summaries=best.validation_summaries,
        candidate_scores=candidate_scores,
    )
