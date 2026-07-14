"""Search a linear adaptive CVaR schedule."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from itertools import product

from risk_shift_bench.adaptive_search import aggregate_score
from risk_shift_bench.benchmark import BenchmarkSummary, run_benchmark
from risk_shift_bench.envs import RiskTask
from risk_shift_bench.policy_registry import learned_adaptive_cvar_policy


@dataclass(frozen=True)
class LinearScheduleParams:
    intercept: float
    bankroll_weight: float
    drawdown_weight: float
    target_gap_weight: float
    ruin_penalty: float
    target_bonus: float


@dataclass(frozen=True)
class LearnedSearchResult:
    params: LinearScheduleParams
    train_score: float
    test_score: float
    train_summaries: list[dict]
    test_summaries: list[dict]


def linear_candidates(smoke: bool = False) -> list[LinearScheduleParams]:
    if smoke:
        return [
            LinearScheduleParams(0.25, 0.25, -0.25, 0.10, 250.0, 100.0),
            LinearScheduleParams(0.15, 0.45, -0.40, 0.25, 500.0, 150.0),
        ]
    return [
        LinearScheduleParams(*values)
        for values in product(
            (0.10, 0.20, 0.35, 0.50),
            (0.0, 0.25, 0.50, 0.75),
            (-0.75, -0.50, -0.25, 0.0),
            (0.0, 0.10, 0.25, 0.50),
            (100.0, 250.0, 500.0),
            (50.0, 150.0, 300.0),
        )
    ]


def evaluate_linear_params(
    params: LinearScheduleParams,
    tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
) -> tuple[float, list[BenchmarkSummary]]:
    policy = learned_adaptive_cvar_policy(**asdict(params))
    _, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
    return aggregate_score(summaries), summaries


def search_learned_adaptive_policy(
    train_tasks: list[RiskTask],
    test_tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
    smoke: bool = False,
    max_candidates: int | None = None,
) -> LearnedSearchResult:
    best_params = None
    best_score = float("-inf")
    best_summaries: list[BenchmarkSummary] = []
    candidates = linear_candidates(smoke=smoke)
    if max_candidates is not None and len(candidates) > max_candidates:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates = candidates[:max_candidates]
    for idx, params in enumerate(candidates):
        score, summaries = evaluate_linear_params(params, train_tasks, episodes, seed + idx * 2000, hand_depth)
        if score > best_score:
            best_params = params
            best_score = score
            best_summaries = summaries
    if best_params is None:
        raise RuntimeError("no learned adaptive candidates were evaluated")
    test_score, test_summaries = evaluate_linear_params(best_params, test_tasks, episodes, seed + 700_000, hand_depth)
    return LearnedSearchResult(
        params=best_params,
        train_score=best_score,
        test_score=test_score,
        train_summaries=[asdict(summary) for summary in best_summaries],
        test_summaries=[asdict(summary) for summary in test_summaries],
    )
