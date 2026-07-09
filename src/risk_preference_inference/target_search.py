"""Search target-branch delegates for the signed regime policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from itertools import product

from risk_preference_inference.adaptive_search import summary_score
from risk_preference_inference.benchmark import BenchmarkSummary, run_benchmark
from risk_preference_inference.envs import RiskTask
from risk_preference_inference.objectives import EntropicObjective, MeanObjective, OCEObjective, TargetSeekingObjective
from risk_preference_inference.policies import BasicStrategyPolicy, BenchmarkPolicy, SignedRegimeAdaptivePolicy, StaticObjectivePolicy
from risk_preference_inference.policy_registry import (
    learned_mixture_policy,
    searched_learned_mixture_policy,
    target_branch_searched_policy,
)


@dataclass(frozen=True)
class TargetBranchParams:
    target_gap_weight: float
    terminal_weight: float
    terminal_window: int
    cvar_alpha: float
    entropic_eta: float
    entropic_weight: float
    cvar_weight: float
    oce_weight: float
    target_bonus: float
    target_excess_weight: float


@dataclass(frozen=True)
class TargetSearchResult:
    params: TargetBranchParams
    train_score: float
    test_score: float
    benchmark_score: float
    promotion_gate: "PromotionGateResult"
    train_summaries: list[dict]
    test_summaries: list[dict]
    benchmark_summaries: list[dict]


@dataclass(frozen=True)
class PromotionGateResult:
    accepted: bool
    min_delta: float
    target_family_candidate_score: float
    target_family_incumbent_score: float
    target_family_delta: float
    benchmark_target_candidate_score: float
    benchmark_target_incumbent_score: float
    benchmark_target_delta: float
    signed_ensemble_candidate_score: float
    signed_ensemble_incumbent_score: float
    signed_ensemble_delta: float
    failed_checks: tuple[str, ...]


def target_branch_candidate_policy(params: TargetBranchParams, name: str = "target_branch_candidate") -> BenchmarkPolicy:
    return learned_mixture_policy(
        risk_intercept=0.0,
        bankroll_weight=0.25,
        drawdown_weight=0.5,
        deck_shift_weight=0.5,
        target_intercept=0.0,
        target_gap_weight=params.target_gap_weight,
        terminal_weight=params.terminal_weight,
        terminal_window=params.terminal_window,
        cvar_alpha=params.cvar_alpha,
        entropic_eta=params.entropic_eta,
        oce_penalty=3.0,
        entropic_weight=params.entropic_weight,
        cvar_weight=params.cvar_weight,
        oce_weight=params.oce_weight,
        deck_entropic_weight=1.25,
        ruin_penalty=250.0,
        drawdown_penalty=0.1,
        target_bonus=params.target_bonus,
        target_excess_weight=params.target_excess_weight,
        name=name,
    )


def signed_policy_with_target_delegate(target_delegate: BenchmarkPolicy, name: str) -> BenchmarkPolicy:
    return SignedRegimeAdaptivePolicy(
        name=name,
        ruin_delegate=StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=f"{name}_ruin_oce"),
        low_shift_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name=f"{name}_low_entropic"),
        target_delegate=target_delegate,
        drawdown_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.01), name=f"{name}_drawdown_entropic"),
        high_shift_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name=f"{name}_high_entropic"),
    )


def target_candidate_params(smoke: bool = False) -> list[TargetBranchParams]:
    current = TargetBranchParams(1.25, 0.5, 16, 0.10, 0.005, 0.25, 0.10, 0.15, 350.0, 0.50)
    previous = TargetBranchParams(0.75, 0.25, 8, 0.15, 0.01, 0.35, 0.05, 0.15, 350.0, 0.15)
    if smoke:
        return [
            current,
            previous,
        ]
    candidates = [
        TargetBranchParams(*values)
        for values in product(
            (0.5, 0.75, 1.0, 1.25),
            (0.0, 0.25, 0.5, 0.75),
            (6, 8, 12, 16),
            (0.10, 0.15, 0.25),
            (0.005, 0.01, 0.025),
            (0.15, 0.25, 0.35),
            (0.0, 0.05, 0.10),
            (0.05, 0.10, 0.15),
            (250.0, 350.0, 500.0, 700.0),
            (0.05, 0.15, 0.30, 0.50),
        )
    ]
    return [current, previous] + [params for params in candidates if params not in {current, previous}]


def target_summary_score(summary: BenchmarkSummary) -> float:
    return (
        summary.mean_final_bankroll
        + 0.5 * summary.cvar_5_final_bankroll
        + 500.0 * summary.target_probability
        - 500.0 * summary.ruin_probability
        - 0.25 * summary.mean_max_drawdown
    )


def aggregate_target_score(summaries: list[BenchmarkSummary]) -> float:
    if not summaries:
        return float("-inf")
    return sum(target_summary_score(summary) for summary in summaries) / len(summaries)


def aggregate_paper_score(summaries: list[BenchmarkSummary]) -> float:
    if not summaries:
        return float("-inf")
    return sum(summary_score(summary) for summary in summaries) / len(summaries)


def score_policy(
    policy: BenchmarkPolicy,
    tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
    score_fn,
) -> tuple[float, list[BenchmarkSummary]]:
    _episodes, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
    if not summaries:
        return float("-inf"), summaries
    return sum(score_fn(summary) for summary in summaries) / len(summaries), summaries


def evaluate_target_policy(
    policy: BenchmarkPolicy,
    tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
) -> tuple[float, list[BenchmarkSummary]]:
    _episodes, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
    return aggregate_target_score(summaries), summaries


def evaluate_promotion_gate(
    candidate_target_policy: BenchmarkPolicy,
    test_tasks: list[RiskTask],
    benchmark_tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
    min_delta: float = 0.0,
) -> PromotionGateResult:
    incumbent_target = searched_learned_mixture_policy(name="target_branch_incumbent")
    benchmark_target_tasks = [task for task in benchmark_tasks if task.name == "RiskBlackjack-Target-v0"]
    if not benchmark_target_tasks:
        raise ValueError("benchmark_tasks must include RiskBlackjack-Target-v0")

    target_family_candidate_score, _candidate_test = score_policy(
        candidate_target_policy,
        test_tasks,
        episodes,
        seed,
        hand_depth,
        target_summary_score,
    )
    target_family_incumbent_score, _incumbent_test = score_policy(
        incumbent_target,
        test_tasks,
        episodes,
        seed,
        hand_depth,
        target_summary_score,
    )
    benchmark_target_candidate_score, _candidate_target = score_policy(
        candidate_target_policy,
        benchmark_target_tasks,
        episodes,
        seed + 101,
        hand_depth,
        target_summary_score,
    )
    benchmark_target_incumbent_score, _incumbent_target = score_policy(
        incumbent_target,
        benchmark_target_tasks,
        episodes,
        seed + 101,
        hand_depth,
        target_summary_score,
    )

    candidate_signed = signed_policy_with_target_delegate(candidate_target_policy, name="signed_candidate_target")
    incumbent_signed = signed_policy_with_target_delegate(incumbent_target, name="signed_incumbent_target")
    signed_ensemble_candidate_score, _candidate_signed = score_policy(
        candidate_signed,
        benchmark_tasks,
        episodes,
        seed + 202,
        hand_depth,
        summary_score,
    )
    signed_ensemble_incumbent_score, _incumbent_signed = score_policy(
        incumbent_signed,
        benchmark_tasks,
        episodes,
        seed + 202,
        hand_depth,
        summary_score,
    )

    target_family_delta = target_family_candidate_score - target_family_incumbent_score
    benchmark_target_delta = benchmark_target_candidate_score - benchmark_target_incumbent_score
    signed_ensemble_delta = signed_ensemble_candidate_score - signed_ensemble_incumbent_score
    failed_checks = []
    if target_family_delta <= min_delta:
        failed_checks.append("target_family")
    if benchmark_target_delta <= min_delta:
        failed_checks.append("benchmark_target")
    if signed_ensemble_delta <= min_delta:
        failed_checks.append("signed_ensemble")

    return PromotionGateResult(
        accepted=not failed_checks,
        min_delta=min_delta,
        target_family_candidate_score=target_family_candidate_score,
        target_family_incumbent_score=target_family_incumbent_score,
        target_family_delta=target_family_delta,
        benchmark_target_candidate_score=benchmark_target_candidate_score,
        benchmark_target_incumbent_score=benchmark_target_incumbent_score,
        benchmark_target_delta=benchmark_target_delta,
        signed_ensemble_candidate_score=signed_ensemble_candidate_score,
        signed_ensemble_incumbent_score=signed_ensemble_incumbent_score,
        signed_ensemble_delta=signed_ensemble_delta,
        failed_checks=tuple(failed_checks),
    )


def target_baseline_policies() -> list[BenchmarkPolicy]:
    return [
        BasicStrategyPolicy(),
        StaticObjectivePolicy(MeanObjective(), name="expected_value"),
        StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="fixed_entropic_0.025"),
        StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="fixed_oce_3"),
        StaticObjectivePolicy(TargetSeekingObjective(MeanObjective(), target_bonus=300.0), name="target_mean_300"),
        StaticObjectivePolicy(TargetSeekingObjective(MeanObjective(), target_bonus=600.0), name="target_mean_600"),
        searched_learned_mixture_policy(),
        target_branch_searched_policy(name="target_branch_promoted"),
    ]


def evaluate_target_baselines(
    tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
) -> list[dict]:
    _episodes, summaries = run_benchmark(
        tasks=tasks,
        policies=target_baseline_policies(),
        episodes=episodes,
        seed=seed,
        hand_depth=hand_depth,
    )
    return [asdict(summary) for summary in summaries]


def target_score_report(summaries: list[dict]) -> dict:
    by_policy: dict[str, list[BenchmarkSummary]] = {}
    by_task: dict[str, list[BenchmarkSummary]] = {}
    for row in summaries:
        summary = BenchmarkSummary(**row)
        by_policy.setdefault(summary.policy, []).append(summary)
        by_task.setdefault(summary.task, []).append(summary)
    return {
        "policy_target_scores": {
            policy: aggregate_target_score(policy_summaries)
            for policy, policy_summaries in sorted(by_policy.items())
        },
        "policy_paper_scores": {
            policy: aggregate_paper_score(policy_summaries)
            for policy, policy_summaries in sorted(by_policy.items())
        },
        "oracle_by_task": {
            task: max(task_summaries, key=target_summary_score).policy
            for task, task_summaries in sorted(by_task.items())
        },
    }


def search_target_branch_policy(
    train_tasks: list[RiskTask],
    test_tasks: list[RiskTask],
    benchmark_tasks: list[RiskTask],
    episodes: int = 100,
    seed: int = 0,
    hand_depth: int = 1,
    smoke: bool = False,
    max_candidates: int | None = None,
    selection_seeds: int = 1,
    promotion_min_delta: float = 0.0,
) -> TargetSearchResult:
    candidates = target_candidate_params(smoke=smoke)
    if max_candidates is not None and len(candidates) > max_candidates:
        current = candidates[0]
        pool = candidates[1:]
        rng = random.Random(seed)
        rng.shuffle(pool)
        candidates = [current] + pool[: max(max_candidates - 1, 0)]

    best_params: TargetBranchParams | None = None
    best_train_score = float("-inf")
    best_train_summaries: list[BenchmarkSummary] = []
    for idx, params in enumerate(candidates):
        policy = target_branch_candidate_policy(params)
        repeated_scores = []
        selected_summaries: list[BenchmarkSummary] = []
        for repeat_idx in range(max(selection_seeds, 1)):
            score, summaries = evaluate_target_policy(
                policy,
                train_tasks,
                episodes,
                seed + idx * 7000 + repeat_idx * 503,
                hand_depth,
            )
            repeated_scores.append(score)
            if repeat_idx == 0:
                selected_summaries = summaries
        score = sum(repeated_scores) / len(repeated_scores)
        if score > best_train_score:
            best_train_score = score
            best_params = params
            best_train_summaries = selected_summaries

    if best_params is None:
        raise RuntimeError("no target branch candidates were evaluated")

    best_policy = target_branch_candidate_policy(best_params, name="target_branch_searched")
    test_score, test_summaries = evaluate_target_policy(best_policy, test_tasks, episodes, seed + 701_000, hand_depth)
    _episodes, benchmark_summaries = run_benchmark(
        tasks=benchmark_tasks,
        policies=[best_policy],
        episodes=episodes,
        seed=seed + 702_000,
        hand_depth=hand_depth,
    )
    promotion_gate = evaluate_promotion_gate(
        candidate_target_policy=best_policy,
        test_tasks=test_tasks,
        benchmark_tasks=benchmark_tasks,
        episodes=episodes,
        seed=seed + 703_000,
        hand_depth=hand_depth,
        min_delta=promotion_min_delta,
    )
    return TargetSearchResult(
        params=best_params,
        train_score=best_train_score,
        test_score=test_score,
        benchmark_score=aggregate_paper_score(benchmark_summaries),
        promotion_gate=promotion_gate,
        train_summaries=[asdict(summary) for summary in best_train_summaries],
        test_summaries=[asdict(summary) for summary in test_summaries],
        benchmark_summaries=[asdict(summary) for summary in benchmark_summaries],
    )
