"""Ablation policies and score tables for the benchmark."""

from __future__ import annotations

from dataclasses import asdict

from risk_preference_inference.adaptive_search import summary_score
from risk_preference_inference.benchmark import BenchmarkSummary, run_benchmark
from risk_preference_inference.objectives import EntropicObjective, MeanObjective, OCEObjective
from risk_preference_inference.policies import BenchmarkPolicy, RegimeAdaptivePolicy, StaticObjectivePolicy
from risk_preference_inference.policy_registry import adaptive_cvar_policy, state_adaptive_utility_policy
from risk_preference_inference.envs import RiskTask


def ablation_policies() -> list[BenchmarkPolicy]:
    return [
        StaticObjectivePolicy(MeanObjective(), name="expected_value"),
        StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="fixed_entropic_0.025"),
        StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="fixed_oce_3"),
        adaptive_cvar_policy(name="naive_adaptive_cvar"),
        state_adaptive_utility_policy(name="no_regime_switch_adaptive_utility"),
        RegimeAdaptivePolicy(name="regime_full"),
        RegimeAdaptivePolicy(name="ablate_deck_shift", enable_deck_shift=False),
        RegimeAdaptivePolicy(name="ablate_ruin_branch", enable_ruin=False),
        RegimeAdaptivePolicy(name="ablate_drawdown_branch", enable_drawdown=False),
        RegimeAdaptivePolicy(name="ablate_target_branch", enable_target=False),
        RegimeAdaptivePolicy(name="ablate_target_gate", require_target_regime=False),
    ]


def aggregate_policy_scores(summaries: list[BenchmarkSummary]) -> list[dict]:
    by_policy: dict[str, list[BenchmarkSummary]] = {}
    for summary in summaries:
        by_policy.setdefault(summary.policy, []).append(summary)

    rows = []
    for policy, policy_summaries in sorted(by_policy.items()):
        score = sum(summary_score(summary) for summary in policy_summaries) / len(policy_summaries)
        rows.append(
            {
                "policy": policy,
                "aggregate_score": score,
                "mean_final_bankroll": sum(summary.mean_final_bankroll for summary in policy_summaries) / len(policy_summaries),
                "cvar_5_final_bankroll": sum(summary.cvar_5_final_bankroll for summary in policy_summaries) / len(policy_summaries),
                "ruin_probability": sum(summary.ruin_probability for summary in policy_summaries) / len(policy_summaries),
                "target_probability": sum(summary.target_probability for summary in policy_summaries) / len(policy_summaries),
                "mean_max_drawdown": sum(summary.mean_max_drawdown for summary in policy_summaries) / len(policy_summaries),
            }
        )
    rows.sort(key=lambda row: row["aggregate_score"], reverse=True)
    best = rows[0]["aggregate_score"] if rows else 0.0
    for row in rows:
        row["score_gap_to_best"] = best - row["aggregate_score"]
    return rows


def task_policy_scores(summaries: list[BenchmarkSummary]) -> list[dict]:
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["score"] = summary_score(summary)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["task"], -row["score"], row["policy"]))


def run_ablation_study(
    tasks: list[RiskTask],
    episodes: int,
    seed: int,
    hand_depth: int,
) -> tuple[list[BenchmarkSummary], list[dict], list[dict]]:
    _episodes, summaries = run_benchmark(
        tasks=tasks,
        policies=ablation_policies(),
        episodes=episodes,
        seed=seed,
        hand_depth=hand_depth,
    )
    return summaries, aggregate_policy_scores(summaries), task_policy_scores(summaries)
