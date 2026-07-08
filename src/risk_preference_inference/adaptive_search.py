"""Search and held-out evaluation for adaptive risk schedules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from itertools import product

from risk_preference_inference.benchmark import BenchmarkSummary, run_benchmark
from risk_preference_inference.envs import RiskTask
from risk_preference_inference.policy_registry import adaptive_cvar_policy, state_adaptive_utility_policy, strong_baseline_grid


@dataclass(frozen=True)
class AdaptiveParams:
    min_alpha: float
    max_alpha: float
    ruin_zone_ratio: float
    safe_zone_ratio: float
    ruin_penalty: float
    target_bonus: float


@dataclass(frozen=True)
class SearchResult:
    params: AdaptiveParams
    train_score: float
    test_score: float
    train_summaries: list[dict]
    test_summaries: list[dict]


@dataclass(frozen=True)
class AdaptiveUtilityParams:
    low_bankroll_ratio: float
    safe_bankroll_ratio: float
    drawdown_trigger: float
    target_window: float
    terminal_window: int
    cvar_alpha: float
    entropic_eta: float
    risk_weight: float
    ruin_penalty: float
    drawdown_penalty: float
    target_bonus: float
    target_excess_weight: float


@dataclass(frozen=True)
class UtilitySearchResult:
    params: AdaptiveUtilityParams
    train_score: float
    test_score: float
    train_summaries: list[dict]
    test_summaries: list[dict]


def candidate_params(smoke: bool = False) -> list[AdaptiveParams]:
    if smoke:
        return [
            AdaptiveParams(0.05, 0.5, 0.6, 1.2, 100.0, 50.0),
            AdaptiveParams(0.05, 0.75, 0.6, 1.25, 250.0, 100.0),
        ]
    return [
        AdaptiveParams(*values)
        for values in product(
            (0.01, 0.05, 0.1),
            (0.5, 0.75, 1.0),
            (0.5, 0.65, 0.8),
            (1.1, 1.25, 1.4),
            (100.0, 250.0, 500.0),
            (50.0, 150.0, 300.0),
        )
        if values[0] <= values[1] and values[2] < values[3]
    ]


def utility_candidate_params(smoke: bool = False) -> list[AdaptiveUtilityParams]:
    if smoke:
        return [
            AdaptiveUtilityParams(0.55, 1.15, 0.12, 0.35, 8, 0.2, 0.01, 0.35, 400.0, 0.35, 180.0, 0.15),
            AdaptiveUtilityParams(0.45, 1.05, 0.16, 0.45, 10, 0.25, 0.005, 0.20, 250.0, 0.20, 250.0, 0.25),
        ]
    return [
        AdaptiveUtilityParams(*values)
        for values in product(
            (0.45, 0.55),
            (1.05, 1.20),
            (0.10, 0.16),
            (0.35, 0.55),
            (6, 10),
            (0.15, 0.30),
            (0.001, 0.01),
            (0.15, 0.35),
            (100.0, 400.0),
            (0.10, 0.40),
            (150.0, 350.0),
            (0.05, 0.25),
        )
        if values[0] < values[1]
    ]


def summary_score(summary: BenchmarkSummary) -> float:
    return (
        summary.mean_final_bankroll
        + 0.5 * summary.cvar_5_final_bankroll
        + 150.0 * summary.target_probability
        - 500.0 * summary.ruin_probability
        - 0.25 * summary.mean_max_drawdown
    )


def aggregate_score(summaries: list[BenchmarkSummary]) -> float:
    if not summaries:
        return float("-inf")
    return sum(summary_score(summary) for summary in summaries) / len(summaries)


def policy_score_report(summaries: list[dict]) -> dict:
    by_policy: dict[str, list[BenchmarkSummary]] = {}
    by_task: dict[str, list[BenchmarkSummary]] = {}
    for summary_dict in summaries:
        summary = BenchmarkSummary(**summary_dict)
        by_policy.setdefault(summary.policy, []).append(summary)
        by_task.setdefault(summary.task, []).append(summary)
    policy_scores = {
        policy: aggregate_score(policy_summaries)
        for policy, policy_summaries in sorted(by_policy.items())
    }
    oracle_by_task = {
        task: max(task_summaries, key=summary_score).policy
        for task, task_summaries in sorted(by_task.items())
    }
    oracle_score = sum(summary_score(max(task_summaries, key=summary_score)) for task_summaries in by_task.values()) / max(len(by_task), 1)
    return {
        "policy_scores": policy_scores,
        "oracle_by_task": oracle_by_task,
        "oracle_score": oracle_score,
    }


def evaluate_params(
    params: AdaptiveParams,
    tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
) -> tuple[float, list[BenchmarkSummary]]:
    policy = adaptive_cvar_policy(**asdict(params))
    _, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
    return aggregate_score(summaries), summaries


def evaluate_utility_params(
    params: AdaptiveUtilityParams,
    tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
) -> tuple[float, list[BenchmarkSummary]]:
    policy = state_adaptive_utility_policy(**asdict(params))
    _, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
    return aggregate_score(summaries), summaries


def search_adaptive_policy(
    train_tasks: list[RiskTask],
    test_tasks: list[RiskTask],
    episodes: int = 100,
    seed: int = 0,
    hand_depth: int = 3,
    smoke: bool = False,
    max_candidates: int | None = None,
) -> SearchResult:
    best_params: AdaptiveParams | None = None
    best_train_score = float("-inf")
    best_train_summaries: list[BenchmarkSummary] = []
    candidates = candidate_params(smoke=smoke)
    if max_candidates is not None and len(candidates) > max_candidates:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates = candidates[:max_candidates]
    for idx, params in enumerate(candidates):
        score, summaries = evaluate_params(params, train_tasks, episodes, seed + idx * 1000, hand_depth)
        if score > best_train_score:
            best_train_score = score
            best_params = params
            best_train_summaries = summaries

    if best_params is None:
        raise RuntimeError("no adaptive candidates were evaluated")
    test_score, test_summaries = evaluate_params(best_params, test_tasks, episodes, seed + 99_000, hand_depth)
    return SearchResult(
        params=best_params,
        train_score=best_train_score,
        test_score=test_score,
        train_summaries=[asdict(summary) for summary in best_train_summaries],
        test_summaries=[asdict(summary) for summary in test_summaries],
    )


def search_adaptive_utility_policy(
    train_tasks: list[RiskTask],
    test_tasks: list[RiskTask],
    episodes: int = 100,
    seed: int = 0,
    hand_depth: int = 3,
    smoke: bool = False,
    max_candidates: int | None = None,
) -> UtilitySearchResult:
    best_params: AdaptiveUtilityParams | None = None
    best_train_score = float("-inf")
    best_train_summaries: list[BenchmarkSummary] = []
    candidates = utility_candidate_params(smoke=smoke)
    if max_candidates is not None and len(candidates) > max_candidates:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates = candidates[:max_candidates]
    for idx, params in enumerate(candidates):
        score, summaries = evaluate_utility_params(params, train_tasks, episodes, seed + idx * 3000, hand_depth)
        if score > best_train_score:
            best_train_score = score
            best_params = params
            best_train_summaries = summaries

    if best_params is None:
        raise RuntimeError("no adaptive utility candidates were evaluated")
    test_score, test_summaries = evaluate_utility_params(best_params, test_tasks, episodes, seed + 199_000, hand_depth)
    return UtilitySearchResult(
        params=best_params,
        train_score=best_train_score,
        test_score=test_score,
        train_summaries=[asdict(summary) for summary in best_train_summaries],
        test_summaries=[asdict(summary) for summary in test_summaries],
    )


def evaluate_strong_baselines(
    tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
) -> list[dict]:
    _, summaries = run_benchmark(
        tasks=tasks,
        policies=strong_baseline_grid(),
        episodes=episodes,
        seed=seed,
        hand_depth=hand_depth,
    )
    return [asdict(summary) for summary in summaries]
