"""Synthetic calibration and efficiency checks for the v2 anytime router.

The simulator generates bounded two-point paired differences with exactly the
requested conditional means.  It is development evidence only and never reads
external confirmation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from math import sqrt
from pathlib import Path

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
    RouteDecision,
)


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    task_means: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class SyntheticTrialResult:
    false_accept: bool
    false_accept_count: int
    true_accept_count: int
    positive_task_count: int
    rejected_count: int
    unresolved_count: int
    total_observations: int
    expected_equal_task_improvement: float


def default_scenarios() -> dict[str, SyntheticScenario]:
    global_null = SyntheticScenario(
        name="global_null",
        task_means=tuple((f"null_{index:02d}", 0.0) for index in range(23)),
    )
    sparse = SyntheticScenario(
        name="sparse_positive",
        task_means=(
            *[(f"null_{index:02d}", 0.0) for index in range(20)],
            ("positive_20", 0.2),
            ("positive_40", 0.4),
            ("positive_60", 0.6),
        ),
    )
    mixed = SyntheticScenario(
        name="mixed_effects",
        task_means=(
            *[(f"negative_{index:02d}", -0.4) for index in range(8)],
            *[(f"null_{index:02d}", 0.0) for index in range(8)],
            *[(f"small_positive_{index:02d}", 0.15) for index in range(4)],
            ("medium_positive_00", 0.35),
            ("medium_positive_01", 0.35),
            ("large_positive_00", 0.6),
        ),
    )
    return {scenario.name: scenario for scenario in (global_null, sparse, mixed)}


def bounded_two_point_observation(
    mean: float,
    *,
    lower: float,
    upper: float,
    rng: random.Random,
) -> float:
    if not lower <= mean <= upper:
        raise ValueError("synthetic mean must lie inside the observation bounds")
    upper_probability = (mean - lower) / (upper - lower)
    return upper if rng.random() < upper_probability else lower


def task_stream_seed(trial_seed: int, task: str) -> int:
    encoded = f"{trial_seed}:{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def run_synthetic_trial(
    scenario: SyntheticScenario,
    *,
    strategy: str,
    seed: int,
    familywise_alpha: float,
    effect_margin: float,
    e_process_method: str,
    maximum_observations_per_task: int,
    global_observation_budget: int,
    forced_initial_observations: int,
) -> SyntheticTrialResult:
    if global_observation_budget <= 0:
        raise ValueError("global_observation_budget must be positive")
    means = dict(scenario.task_means)
    plan = AnytimeFamilywisePlan(
        task_names=tuple(sorted(means)),
        familywise_alpha=familywise_alpha,
        futility_familywise_alpha=familywise_alpha,
        effect_margin=effect_margin,
        e_process_method=e_process_method,
        maximum_observations_per_task=maximum_observations_per_task,
    )
    router = AnytimeFamilywiseRouter(plan)
    task_rngs = {
        task: random.Random(task_stream_seed(seed, task)) for task in plan.task_names
    }
    while router.total_observations() < global_observation_budget:
        task = router.next_task(
            strategy=strategy,
            forced_initial_observations=forced_initial_observations,
        )
        if task is None:
            break
        observation = bounded_two_point_observation(
            means[task],
            lower=plan.observation_lower,
            upper=plan.observation_upper,
            rng=task_rngs[task],
        )
        router.update(task, observation)

    decisions = router.decisions()
    accepted = {
        task
        for task, evidence in decisions.items()
        if evidence.decision is RouteDecision.ACCEPT_CANDIDATE
    }
    false_accepted = {
        task for task in accepted if means[task] <= plan.effect_margin
    }
    true_accepted = {
        task for task in accepted if means[task] > plan.effect_margin
    }
    positive_tasks = {
        task for task, mean in means.items() if mean > plan.effect_margin
    }
    rejected = sum(
        evidence.decision is RouteDecision.REJECT_TO_FALLBACK
        for evidence in decisions.values()
    )
    unresolved = sum(
        evidence.decision
        in {RouteDecision.UNDECIDED, RouteDecision.BUDGET_EXHAUSTED}
        for evidence in decisions.values()
    )
    expected_improvement = sum(means[task] for task in accepted) / len(means)
    return SyntheticTrialResult(
        false_accept=bool(false_accepted),
        false_accept_count=len(false_accepted),
        true_accept_count=len(true_accepted),
        positive_task_count=len(positive_tasks),
        rejected_count=rejected,
        unresolved_count=unresolved,
        total_observations=router.total_observations(),
        expected_equal_task_improvement=expected_improvement,
    )


def _run_indexed_synthetic_trial(
    trial_index: int,
    *,
    scenario: SyntheticScenario,
    strategy: str,
    seed: int,
    familywise_alpha: float,
    effect_margin: float,
    e_process_method: str,
    maximum_observations_per_task: int,
    global_observation_budget: int,
    forced_initial_observations: int,
) -> SyntheticTrialResult:
    return run_synthetic_trial(
        scenario,
        strategy=strategy,
        seed=seed + trial_index,
        familywise_alpha=familywise_alpha,
        effect_margin=effect_margin,
        e_process_method=e_process_method,
        maximum_observations_per_task=maximum_observations_per_task,
        global_observation_budget=global_observation_budget,
        forced_initial_observations=forced_initial_observations,
    )


def collect_synthetic_trials(
    scenario: SyntheticScenario,
    *,
    strategy: str,
    trials: int,
    seed: int,
    familywise_alpha: float,
    effect_margin: float,
    e_process_method: str,
    maximum_observations_per_task: int,
    global_observation_budget: int,
    forced_initial_observations: int,
    workers: int,
) -> list[SyntheticTrialResult]:
    """Run deterministic trials serially or in spawned worker processes."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    worker = partial(
        _run_indexed_synthetic_trial,
        scenario=scenario,
        strategy=strategy,
        seed=seed,
        familywise_alpha=familywise_alpha,
        effect_margin=effect_margin,
        e_process_method=e_process_method,
        maximum_observations_per_task=maximum_observations_per_task,
        global_observation_budget=global_observation_budget,
        forced_initial_observations=forced_initial_observations,
    )
    if workers == 1:
        return [worker(trial_index) for trial_index in range(trials)]
    chunksize = max(1, trials // (workers * 20))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, range(trials), chunksize=chunksize))


def wilson_interval(successes: int, trials: int, z_value: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials")
    proportion = successes / trials
    denominator = 1.0 + z_value * z_value / trials
    center = (
        proportion + z_value * z_value / (2.0 * trials)
    ) / denominator
    radius = (
        z_value
        * sqrt(
            proportion * (1.0 - proportion) / trials
            + z_value * z_value / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def summarize_trials(
    scenario: SyntheticScenario,
    *,
    strategy: str,
    trials: int,
    seed: int,
    familywise_alpha: float = 0.05,
    effect_margin: float = 0.0,
    e_process_method: str = "betting_mixture",
    maximum_observations_per_task: int = 100,
    global_observation_budget: int = 2_300,
    forced_initial_observations: int = 2,
    workers: int = 1,
) -> dict:
    if trials <= 0:
        raise ValueError("trials must be positive")
    results = collect_synthetic_trials(
        scenario,
        strategy=strategy,
        trials=trials,
        seed=seed,
        familywise_alpha=familywise_alpha,
        effect_margin=effect_margin,
        e_process_method=e_process_method,
        maximum_observations_per_task=maximum_observations_per_task,
        global_observation_budget=global_observation_budget,
        forced_initial_observations=forced_initial_observations,
        workers=workers,
    )
    false_accept_families = sum(result.false_accept for result in results)
    interval = wilson_interval(false_accept_families, trials)
    positive_count = results[0].positive_task_count
    return {
        "scope": "Synthetic development calibration; no confirmation artifact is read.",
        "scenario": scenario.name,
        "strategy": strategy,
        "task_means": dict(scenario.task_means),
        "trials": trials,
        "seed": seed,
        "familywise_alpha": familywise_alpha,
        "effect_margin": effect_margin,
        "e_process_method": e_process_method,
        "maximum_observations_per_task": maximum_observations_per_task,
        "global_observation_budget": global_observation_budget,
        "forced_initial_observations": forced_initial_observations,
        "workers": workers,
        "familywise_false_accept_rate": false_accept_families / trials,
        "familywise_false_accept_wilson_95_ci": list(interval),
        "mean_false_accept_count": sum(
            result.false_accept_count for result in results
        )
        / trials,
        "mean_true_accept_count": sum(
            result.true_accept_count for result in results
        )
        / trials,
        "positive_task_count": positive_count,
        "mean_positive_acceptance_rate": (
            sum(result.true_accept_count for result in results)
            / (trials * positive_count)
            if positive_count
            else None
        ),
        "mean_rejected_count": sum(result.rejected_count for result in results)
        / trials,
        "mean_unresolved_count": sum(
            result.unresolved_count for result in results
        )
        / trials,
        "mean_total_observations": sum(
            result.total_observations for result in results
        )
        / trials,
        "mean_expected_equal_task_improvement": sum(
            result.expected_equal_task_improvement for result in results
        )
        / trials,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=(*default_scenarios(), "all"),
        default="all",
    )
    parser.add_argument(
        "--strategy", choices=("uniform", "resolution", "both"), default="both"
    )
    parser.add_argument("--trials", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2_026_071_900)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--effect-margin", type=float, default=0.0)
    parser.add_argument(
        "--e-process-method",
        choices=("hoeffding_mixture", "betting_mixture", "both"),
        default="both",
    )
    parser.add_argument("--maximum-observations-per-task", type=int, default=100)
    parser.add_argument("--global-observation-budget", type=int, default=2_300)
    parser.add_argument("--forced-initial-observations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = default_scenarios()
    selected_scenarios = (
        list(scenarios.values())
        if args.scenario == "all"
        else [scenarios[args.scenario]]
    )
    strategies = (
        ("uniform", "resolution")
        if args.strategy == "both"
        else (args.strategy,)
    )
    methods = (
        ("hoeffding_mixture", "betting_mixture")
        if args.e_process_method == "both"
        else (args.e_process_method,)
    )
    summaries = [
        summarize_trials(
            scenario,
            strategy=strategy,
            trials=args.trials,
            seed=args.seed,
            familywise_alpha=args.familywise_alpha,
            effect_margin=args.effect_margin,
            e_process_method=method,
            maximum_observations_per_task=args.maximum_observations_per_task,
            global_observation_budget=args.global_observation_budget,
            forced_initial_observations=args.forced_initial_observations,
            workers=args.workers,
        )
        for scenario in selected_scenarios
        for strategy in strategies
        for method in methods
    ]
    payload = {
        "design": "riskshiftbench-v2-anytime-familywise-synthetic-calibration",
        "summaries": summaries,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
