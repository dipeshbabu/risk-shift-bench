"""Multi-seed policy evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict
from math import erf, sqrt

from risk_preference_inference.adaptive_search import summary_score
from risk_preference_inference.benchmark import BenchmarkSummary, run_benchmark
from risk_preference_inference.envs import RiskTask
from risk_preference_inference.objectives import EntropicObjective, MeanObjective, OCEObjective
from risk_preference_inference.policies import BasicStrategyPolicy, BenchmarkPolicy, RegimeAdaptivePolicy, StaticObjectivePolicy
from risk_preference_inference.policy_registry import (
    adaptive_cvar_policy,
    searched_learned_mixture_policy,
    signed_regime_learned_policy,
    state_adaptive_utility_policy,
)


def multiseed_policies() -> list[BenchmarkPolicy]:
    return [
        BasicStrategyPolicy(),
        StaticObjectivePolicy(MeanObjective(), name="expected_value"),
        StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="fixed_entropic_0.025"),
        StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="fixed_oce_3"),
        adaptive_cvar_policy(name="naive_adaptive_cvar"),
        state_adaptive_utility_policy(name="adaptive_utility_default"),
        searched_learned_mixture_policy(),
        RegimeAdaptivePolicy(),
        signed_regime_learned_policy(),
    ]


def summarize_seed(seed: int, summaries: list[BenchmarkSummary]) -> list[dict]:
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["seed"] = seed
        row["score"] = summary_score(summary)
        rows.append(row)
    return rows


def aggregate_seed_scores(rows: list[dict]) -> list[dict]:
    by_policy: dict[str, list[float]] = {}
    by_policy_task: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row["score"])
        by_policy_task.setdefault((row["task"], row["policy"]), []).append(row["score"])

    output = []
    for policy, scores in sorted(by_policy.items()):
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / max(len(scores) - 1, 1)
        output.append(
            {
                "scope": "all_tasks",
                "task": "ALL",
                "policy": policy,
                "n": len(scores),
                "mean_score": mean_score,
                "std_score": variance**0.5,
            }
        )

    for (task, policy), scores in sorted(by_policy_task.items()):
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / max(len(scores) - 1, 1)
        output.append(
            {
                "scope": "task",
                "task": task,
                "policy": policy,
                "n": len(scores),
                "mean_score": mean_score,
                "std_score": variance**0.5,
            }
        )
    return output


def paired_policy_deltas(rows: list[dict], reference_policy: str = "learned_mixture_searched") -> list[dict]:
    cells: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        cells.setdefault((row["task"], row["seed"]), {})[row["policy"]] = row["score"]

    policies = sorted({row["policy"] for row in rows if row["policy"] != reference_policy})
    output = []
    for policy in policies:
        deltas = []
        for scores in cells.values():
            if reference_policy in scores and policy in scores:
                deltas.append(scores[reference_policy] - scores[policy])
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        variance = sum((delta - mean_delta) ** 2 for delta in deltas) / max(len(deltas) - 1, 1)
        std_delta = variance**0.5
        stderr_delta = std_delta / sqrt(len(deltas)) if deltas else 0.0
        t_stat = mean_delta / stderr_delta if stderr_delta > 0 else 0.0
        normal_approx_p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
        output.append(
            {
                "reference_policy": reference_policy,
                "baseline_policy": policy,
                "n_pairs": len(deltas),
                "mean_delta": mean_delta,
                "std_delta": std_delta,
                "stderr_delta": stderr_delta,
                "t_stat": t_stat,
                "normal_approx_p": normal_approx_p,
                "win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
            }
        )
    return sorted(output, key=lambda row: row["mean_delta"], reverse=True)


def run_multiseed_evaluation(
    tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    reference_policy: str = "signed_regime_learned_ensemble",
) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(
            tasks=tasks,
            policies=multiseed_policies(),
            episodes=episodes,
            seed=seed,
            hand_depth=hand_depth,
        )
        rows.extend(summarize_seed(seed, summaries))
    return rows, aggregate_seed_scores(rows), paired_policy_deltas(rows, reference_policy=reference_policy)
