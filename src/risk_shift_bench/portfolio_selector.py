"""Task-feature policy portfolio selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, sqrt

from risk_shift_bench.adaptive_search import summary_score
from risk_shift_bench.benchmark import BenchmarkSummary, run_benchmark
from risk_shift_bench.envs import STANDARD_DECK, RiskTask
from risk_shift_bench.multiseed import multiseed_policies
from risk_shift_bench.policies import BenchmarkPolicy


@dataclass(frozen=True)
class PortfolioProfile:
    task: str
    features: tuple[float, ...]
    policy: str
    score: float


@dataclass(frozen=True)
class PortfolioSelectorParams:
    k: int = 3
    temperature: float = 0.35
    uncertainty_weight: float = 1.0
    bankroll_weight: float = 1.0
    shift_weight: float = 1.0


@dataclass(frozen=True)
class PortfolioSearchResult:
    params: PortfolioSelectorParams
    validation_score: float
    train_profiles: list[dict]
    validation_summaries: list[dict]


def card_mean(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(card * prob for card, prob in card_probs)


def high_card_mass(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(prob for card, prob in card_probs if card >= 10)


def regime_span(task: RiskTask) -> float:
    if task.episode_card_regimes is None:
        return 0.0
    standard_mean = card_mean(STANDARD_DECK)
    shifts = [card_mean(card_probs) - standard_mean for card_probs, _prob in task.episode_card_regimes]
    return max(shifts) - min(shifts)


def task_features(task: RiskTask, params: PortfolioSelectorParams | None = None) -> tuple[float, ...]:
    weights = params or PortfolioSelectorParams()
    standard_mean = card_mean(STANDARD_DECK)
    mean_shift = card_mean(task.card_probs) - standard_mean
    target_gap = (task.target_bankroll - task.initial_bankroll) / max(task.initial_bankroll, 1.0)
    bankroll_ratio = task.initial_bankroll / max(task.bet, 1.0)
    hidden = 1.0 if task.episode_card_regimes is not None else 0.0
    return (
        task.rounds / 60.0,
        weights.bankroll_weight * bankroll_ratio / 25.0,
        weights.bankroll_weight * task.ruin_bankroll / max(task.initial_bankroll, 1.0),
        target_gap,
        task.drawdown_limit,
        weights.shift_weight * mean_shift / 3.0,
        weights.shift_weight * high_card_mass(task.card_probs),
        weights.uncertainty_weight * hidden,
        weights.uncertainty_weight * regime_span(task) / 4.0,
    )


def profile_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def policy_lookup() -> dict[str, BenchmarkPolicy]:
    return {policy.name: policy for policy in multiseed_policies()}


class TaskFeaturePortfolioPolicy(BenchmarkPolicy):
    name = "task_feature_portfolio"

    def __init__(
        self,
        profiles: list[PortfolioProfile],
        params: PortfolioSelectorParams,
        policies: dict[str, BenchmarkPolicy] | None = None,
        name: str = "task_feature_portfolio",
    ) -> None:
        if not profiles:
            raise ValueError("portfolio selector requires at least one profile")
        self.profiles = profiles
        self.params = params
        self.policies = policies or policy_lookup()
        self.name = name

    def selected_policy_name(self, task: RiskTask) -> str:
        features = task_features(task, self.params)
        neighbors = sorted(
            ((profile_distance(features, profile.features), profile) for profile in self.profiles),
            key=lambda item: (item[0], -item[1].score, item[1].policy),
        )[: max(1, self.params.k)]
        votes: dict[str, float] = {}
        for distance, profile in neighbors:
            weight = exp(-distance / max(self.params.temperature, 1e-9)) * max(profile.score, 1.0)
            votes[profile.policy] = votes.get(profile.policy, 0.0) + weight
        return max(votes, key=votes.get)

    def action_probabilities(
        self,
        state,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        selected = self.selected_policy_name(task)
        delegate = self.policies[selected]
        return delegate.action_probabilities(
            state,
            task=task,
            rounds_remaining=rounds_remaining,
            hand_depth=hand_depth,
            peak_bankroll=peak_bankroll,
        )


def best_policy_profiles(
    tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    params: PortfolioSelectorParams,
) -> tuple[list[PortfolioProfile], list[dict]]:
    policies = multiseed_policies()
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

    by_task: dict[str, list[tuple[str, float]]] = {}
    for (task_name, policy_name), summaries in summaries_by_task_policy.items():
        score = sum(summary_score(summary) for summary in summaries) / len(summaries)
        by_task.setdefault(task_name, []).append((policy_name, score))

    task_by_name = {task.name: task for task in tasks}
    profiles = []
    for task_name, policy_scores in sorted(by_task.items()):
        best_policy, best_score = max(policy_scores, key=lambda item: (item[1], item[0]))
        task = task_by_name[task_name]
        profiles.append(
            PortfolioProfile(
                task=task_name,
                features=task_features(task, params),
                policy=best_policy,
                score=best_score,
            )
        )
    return profiles, raw_rows


def evaluate_portfolio(
    tasks: list[RiskTask],
    profiles: list[PortfolioProfile],
    params: PortfolioSelectorParams,
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    name: str = "task_feature_portfolio",
) -> tuple[float, list[dict]]:
    policy = TaskFeaturePortfolioPolicy(profiles=profiles, params=params, name=name)
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


def portfolio_candidate_params(smoke: bool = False) -> list[PortfolioSelectorParams]:
    candidates = [
        PortfolioSelectorParams(k=1, temperature=0.25, uncertainty_weight=1.0, bankroll_weight=1.0, shift_weight=1.0),
        PortfolioSelectorParams(k=3, temperature=0.35, uncertainty_weight=1.0, bankroll_weight=1.0, shift_weight=1.0),
        PortfolioSelectorParams(k=5, temperature=0.50, uncertainty_weight=1.0, bankroll_weight=1.0, shift_weight=1.0),
        PortfolioSelectorParams(k=3, temperature=0.35, uncertainty_weight=1.5, bankroll_weight=1.0, shift_weight=1.0),
        PortfolioSelectorParams(k=3, temperature=0.35, uncertainty_weight=1.0, bankroll_weight=1.5, shift_weight=1.0),
        PortfolioSelectorParams(k=3, temperature=0.35, uncertainty_weight=1.0, bankroll_weight=1.0, shift_weight=1.5),
        PortfolioSelectorParams(k=5, temperature=0.35, uncertainty_weight=1.5, bankroll_weight=1.0, shift_weight=1.5),
        PortfolioSelectorParams(k=1, temperature=0.20, uncertainty_weight=1.5, bankroll_weight=1.5, shift_weight=1.5),
    ]
    return candidates[:2] if smoke else candidates


def search_portfolio_selector(
    train_tasks: list[RiskTask],
    validation_tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    smoke: bool = False,
) -> PortfolioSearchResult:
    best: PortfolioSearchResult | None = None
    for params in portfolio_candidate_params(smoke=smoke):
        profiles, _train_rows = best_policy_profiles(
            tasks=train_tasks,
            seeds=seeds,
            episodes=episodes,
            hand_depth=hand_depth,
            params=params,
        )
        validation_score, validation_rows = evaluate_portfolio(
            tasks=validation_tasks,
            profiles=profiles,
            params=params,
            seeds=seeds,
            episodes=episodes,
            hand_depth=hand_depth,
        )
        result = PortfolioSearchResult(
            params=params,
            validation_score=validation_score,
            train_profiles=[asdict(profile) for profile in profiles],
            validation_summaries=validation_rows,
        )
        if best is None or result.validation_score > best.validation_score:
            best = result
    if best is None:
        raise RuntimeError("no portfolio candidates were evaluated")
    return best
