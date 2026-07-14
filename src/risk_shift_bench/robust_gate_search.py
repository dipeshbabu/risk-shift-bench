"""Development-only search for robust signed-regime gate variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

from risk_shift_bench.adaptive_search import summary_score
from risk_shift_bench.benchmark import BenchmarkSummary, run_benchmark
from risk_shift_bench.envs import RiskTask
from risk_shift_bench.objectives import EntropicObjective, MeanObjective, OCEObjective
from risk_shift_bench.policies import BasicStrategyPolicy, BenchmarkPolicy, SignedRegimeAdaptivePolicy, StaticObjectivePolicy
from risk_shift_bench.policy_registry import learned_mixture_policy, searched_learned_mixture_policy, target_branch_searched_policy


@dataclass(frozen=True)
class RobustGateParams:
    severe_ruin_bet_ratio: float
    near_ruin_bet_ratio: float
    near_ruin_delegate: str
    hidden_drawdown_delegate: str
    hidden_long_delegate: str
    long_shift_drawdown_delegate: str
    high_shift_delegate: str
    extreme_low_shift_delegate: str


@dataclass(frozen=True)
class RobustGateSearchResult:
    params: RobustGateParams
    selection_score: float
    mean_score: float
    std_score: float
    min_score: float
    train_summaries: list[dict]
    validation_score: float | None = None
    validation_mean_score: float | None = None
    validation_std_score: float | None = None
    validation_min_score: float | None = None
    validation_summaries: list[dict] | None = None


def _delegate(kind: str, name: str) -> BenchmarkPolicy:
    if kind == "basic":
        return BasicStrategyPolicy(name=name)
    if kind == "mean":
        return StaticObjectivePolicy(MeanObjective(), name=name)
    if kind == "oce":
        return StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=name)
    if kind == "entropic":
        return StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name=name)
    if kind == "mixture":
        return learned_mixture_policy(name=name)
    if kind == "searched_mixture":
        return searched_learned_mixture_policy(name=name)
    raise ValueError(f"unknown delegate kind: {kind}")


def robust_gate_policy(params: RobustGateParams, name: str = "learned_robust_gate_dev") -> BenchmarkPolicy:
    return SignedRegimeAdaptivePolicy(
        name=name,
        severe_ruin_bet_ratio=params.severe_ruin_bet_ratio,
        near_ruin_bet_ratio=params.near_ruin_bet_ratio,
        mean_delegate=learned_mixture_policy(name=f"{name}_mean_mixture"),
        severe_ruin_delegate=BasicStrategyPolicy(name=f"{name}_severe_ruin_basic"),
        hidden_drawdown_delegate=_delegate(params.hidden_drawdown_delegate, f"{name}_hidden_drawdown_{params.hidden_drawdown_delegate}"),
        hidden_long_delegate=_delegate(params.hidden_long_delegate, f"{name}_hidden_long_{params.hidden_long_delegate}"),
        near_ruin_delegate=_delegate(params.near_ruin_delegate, f"{name}_near_ruin_{params.near_ruin_delegate}"),
        short_target_delegate=BasicStrategyPolicy(name=f"{name}_short_target_basic"),
        long_drawdown_delegate=BasicStrategyPolicy(name=f"{name}_long_drawdown_basic"),
        long_shift_drawdown_delegate=_delegate(params.long_shift_drawdown_delegate, f"{name}_long_shift_drawdown_{params.long_shift_drawdown_delegate}"),
        hidden_tail_delegate=BasicStrategyPolicy(name=f"{name}_hidden_tail_basic"),
        ruin_delegate=StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=f"{name}_ruin_oce"),
        low_shift_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name=f"{name}_low_entropic"),
        extreme_low_shift_delegate=_delegate(params.extreme_low_shift_delegate, f"{name}_extreme_low_{params.extreme_low_shift_delegate}"),
        target_delegate=target_branch_searched_policy(name=f"{name}_target_delegate"),
        drawdown_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.01), name=f"{name}_drawdown_entropic"),
        high_shift_delegate=_delegate(params.high_shift_delegate, f"{name}_high_shift_{params.high_shift_delegate}"),
    )


def robust_gate_candidate_params(smoke: bool = False) -> list[RobustGateParams]:
    candidates = [
        RobustGateParams(8.0, 8.0, "basic", "oce", "mean", "mean", "mixture", "searched_mixture"),
        RobustGateParams(6.0, 8.0, "oce", "oce", "mean", "mean", "mixture", "searched_mixture"),
        RobustGateParams(8.0, 10.0, "basic", "oce", "mean", "mean", "mixture", "searched_mixture"),
        RobustGateParams(8.0, 8.0, "basic", "mean", "mean", "mean", "mixture", "searched_mixture"),
        RobustGateParams(8.0, 8.0, "basic", "oce", "mean", "basic", "mixture", "searched_mixture"),
        RobustGateParams(8.0, 8.0, "basic", "oce", "mean", "mean", "entropic", "searched_mixture"),
        RobustGateParams(8.0, 8.0, "basic", "oce", "mean", "mean", "mixture", "mixture"),
        RobustGateParams(10.0, 10.0, "basic", "oce", "mean", "mean", "mixture", "searched_mixture"),
    ]
    return candidates[:2] if smoke else candidates


def _cell_scores(summaries: list[BenchmarkSummary]) -> list[float]:
    return [summary_score(summary) for summary in summaries]


def robust_selection_score(scores: list[float]) -> tuple[float, float, float, float]:
    if not scores:
        return float("-inf"), float("-inf"), 0.0, float("-inf")
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / max(len(scores) - 1, 1)
    std_score = sqrt(variance)
    min_score = min(scores)
    selection_score = mean_score - 0.10 * std_score + 0.05 * min_score
    return selection_score, mean_score, std_score, min_score


def evaluate_candidate(
    params: RobustGateParams,
    tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
) -> tuple[float, float, float, float, list[BenchmarkSummary]]:
    policy = robust_gate_policy(params, name="learned_robust_gate_candidate")
    all_summaries: list[BenchmarkSummary] = []
    all_scores: list[float] = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
        all_summaries.extend(summaries)
        all_scores.extend(_cell_scores(summaries))
    selection_score, mean_score, std_score, min_score = robust_selection_score(all_scores)
    return selection_score, mean_score, std_score, min_score, all_summaries


def search_robust_gate(
    train_tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    max_candidates: int | None = None,
    smoke: bool = False,
    validation_tasks: list[RiskTask] | None = None,
) -> RobustGateSearchResult:
    candidates = robust_gate_candidate_params(smoke=smoke)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    best: RobustGateSearchResult | None = None
    for params in candidates:
        selection_score, mean_score, std_score, min_score, summaries = evaluate_candidate(
            params=params,
            tasks=train_tasks,
            seeds=seeds,
            episodes=episodes,
            hand_depth=hand_depth,
        )
        validation_score = None
        validation_mean_score = None
        validation_std_score = None
        validation_min_score = None
        validation_summaries = None
        ranking_score = selection_score
        if validation_tasks:
            (
                validation_score,
                validation_mean_score,
                validation_std_score,
                validation_min_score,
                raw_validation_summaries,
            ) = evaluate_candidate(
                params=params,
                tasks=validation_tasks,
                seeds=seeds,
                episodes=episodes,
                hand_depth=hand_depth,
            )
            validation_summaries = [asdict(summary) for summary in raw_validation_summaries]
            ranking_score = validation_score
        result = RobustGateSearchResult(
            params=params,
            selection_score=selection_score,
            mean_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            train_summaries=[asdict(summary) for summary in summaries],
            validation_score=validation_score,
            validation_mean_score=validation_mean_score,
            validation_std_score=validation_std_score,
            validation_min_score=validation_min_score,
            validation_summaries=validation_summaries,
        )
        best_score = best.validation_score if best is not None and best.validation_score is not None else (best.selection_score if best is not None else None)
        if best is None or ranking_score > best_score:
            best = result
    if best is None:
        raise RuntimeError("no robust gate candidates were evaluated")
    return best
