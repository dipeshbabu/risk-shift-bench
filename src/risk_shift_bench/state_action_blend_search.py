"""Validation search for state/action blend policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from risk_shift_bench.adaptive_search import summary_score
from risk_shift_bench.benchmark import BenchmarkSummary, run_benchmark
from risk_shift_bench.objectives import EntropicObjective, OCEObjective
from risk_shift_bench.policies import BasicStrategyPolicy, BenchmarkPolicy, StateActionBlendPolicy, StaticObjectivePolicy
from risk_shift_bench.policy_registry import (
    learned_mixture_policy,
    signed_regime_learned_policy,
    target_branch_searched_policy,
)
from risk_shift_bench.envs import RiskTask


@dataclass(frozen=True)
class StateActionBlendParams:
    base_weight: float = 1.0
    signed_weight: float = 0.35
    risk_weight: float = 1.25
    drawdown_weight: float = 0.8
    target_weight: float = 0.7
    basic_weight: float = 0.8
    uncertainty_weight: float = 0.5


@dataclass(frozen=True)
class StateActionBlendSearchResult:
    params: StateActionBlendParams
    validation_score: float
    validation_summaries: list[dict]
    candidate_scores: list[dict]


def state_action_blend_from_params(
    params: StateActionBlendParams,
    name: str = "validated_state_action_blend",
) -> StateActionBlendPolicy:
    return StateActionBlendPolicy(
        name=name,
        mean_delegate=learned_mixture_policy(name=f"{name}_mean_mixture"),
        risk_delegate=StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=f"{name}_risk_oce"),
        drawdown_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name=f"{name}_drawdown_entropic"),
        target_delegate=target_branch_searched_policy(name=f"{name}_target_delegate"),
        basic_delegate=BasicStrategyPolicy(name=f"{name}_basic"),
        signed_delegate=signed_regime_learned_policy(name=f"{name}_signed_delegate"),
        base_weight=params.base_weight,
        signed_weight=params.signed_weight,
        risk_weight=params.risk_weight,
        drawdown_weight=params.drawdown_weight,
        target_weight=params.target_weight,
        basic_weight=params.basic_weight,
        uncertainty_weight=params.uncertainty_weight,
    )


def state_action_blend_candidates(smoke: bool = False) -> list[StateActionBlendParams]:
    candidates = [
        StateActionBlendParams(),
        StateActionBlendParams(base_weight=1.0, signed_weight=0.0, risk_weight=0.0, drawdown_weight=0.0, target_weight=0.0, basic_weight=0.0, uncertainty_weight=0.0),
        StateActionBlendParams(base_weight=0.35, signed_weight=1.50, risk_weight=0.0, drawdown_weight=0.0, target_weight=0.0, basic_weight=0.15, uncertainty_weight=0.0),
        StateActionBlendParams(base_weight=0.65, signed_weight=1.00, risk_weight=0.35, drawdown_weight=0.15, target_weight=0.15, basic_weight=0.15, uncertainty_weight=0.10),
        StateActionBlendParams(base_weight=1.25, signed_weight=0.25, risk_weight=0.35, drawdown_weight=0.10, target_weight=0.10, basic_weight=0.05, uncertainty_weight=0.05),
        StateActionBlendParams(base_weight=0.85, signed_weight=0.35, risk_weight=0.0, drawdown_weight=0.0, target_weight=1.25, basic_weight=0.0, uncertainty_weight=0.0),
        StateActionBlendParams(base_weight=0.75, signed_weight=0.25, risk_weight=1.50, drawdown_weight=0.75, target_weight=0.0, basic_weight=0.35, uncertainty_weight=0.75),
        StateActionBlendParams(base_weight=0.50, signed_weight=0.65, risk_weight=0.75, drawdown_weight=0.75, target_weight=0.75, basic_weight=0.25, uncertainty_weight=0.35),
        StateActionBlendParams(base_weight=1.50, signed_weight=0.10, risk_weight=0.10, drawdown_weight=0.0, target_weight=0.0, basic_weight=0.0, uncertainty_weight=0.0),
        StateActionBlendParams(base_weight=0.90, signed_weight=0.50, risk_weight=0.25, drawdown_weight=0.25, target_weight=0.50, basic_weight=0.10, uncertainty_weight=0.15),
        StateActionBlendParams(base_weight=0.25, signed_weight=0.25, risk_weight=0.0, drawdown_weight=0.0, target_weight=0.0, basic_weight=1.50, uncertainty_weight=0.50),
        StateActionBlendParams(base_weight=0.60, signed_weight=0.20, risk_weight=0.10, drawdown_weight=0.10, target_weight=1.50, basic_weight=0.0, uncertainty_weight=0.0),
    ]
    return candidates[:3] if smoke else candidates


def evaluate_blend_policy(
    tasks: list[RiskTask],
    policy: BenchmarkPolicy,
    seeds: list[int],
    episodes: int,
    hand_depth: int,
) -> tuple[float, list[dict]]:
    rows = []
    scores = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
        for summary in summaries:
            row = asdict(summary)
            row["seed"] = seed
            row["score"] = summary_score(summary)
            rows.append(row)
            scores.append(row["score"])
    return sum(scores) / len(scores) if scores else float("-inf"), rows


def search_state_action_blend(
    validation_tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    smoke: bool = False,
) -> StateActionBlendSearchResult:
    best: StateActionBlendSearchResult | None = None
    candidate_scores = []
    for index, params in enumerate(state_action_blend_candidates(smoke=smoke)):
        policy = state_action_blend_from_params(params, name=f"state_action_blend_candidate_{index}")
        score, rows = evaluate_blend_policy(
            tasks=validation_tasks,
            policy=policy,
            seeds=seeds,
            episodes=episodes,
            hand_depth=hand_depth,
        )
        candidate_row = {"candidate": index, "validation_score": score, **asdict(params)}
        candidate_scores.append(candidate_row)
        result = StateActionBlendSearchResult(
            params=params,
            validation_score=score,
            validation_summaries=rows,
            candidate_scores=list(candidate_scores),
        )
        if best is None or result.validation_score > best.validation_score:
            best = result
    if best is None:
        raise RuntimeError("no state/action blend candidates were evaluated")
    return StateActionBlendSearchResult(
        params=best.params,
        validation_score=best.validation_score,
        validation_summaries=best.validation_summaries,
        candidate_scores=candidate_scores,
    )
