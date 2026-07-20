"""Nonbinding empirical check of the certified betting resolution quotas."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

from experiments.anytime_familywise_calibration import (
    bounded_two_point_observation,
    task_stream_seed,
    wilson_interval,
)
from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
    RouteDecision,
)
from experiments.frontier_v2_statistical_hash import (
    statistical_implementation_sha256,
)


RESOLUTION_BOUND_FILE = "resolution_bound_nonbinding_1000_current.json"


def implementation_sha256() -> str:
    path = Path(__file__)
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def separated_scenario() -> tuple[tuple[str, float, float], ...]:
    """Return (task, true mean, planning gap) with both null directions."""

    return (
        *[(f"negative_large_{index:02d}", -0.35, 0.35) for index in range(8)],
        *[(f"negative_small_{index:02d}", -0.10, 0.10) for index in range(4)],
        *[(f"positive_small_{index:02d}", 0.10, 0.10) for index in range(8)],
        ("positive_medium_00", 0.30, 0.30),
        ("positive_medium_01", 0.30, 0.30),
        ("positive_large_00", 0.60, 0.60),
    )


def _plan(
    scenario: tuple[tuple[str, float, float], ...],
    *,
    maximum_observations_per_task: int,
) -> AnytimeFamilywisePlan:
    return AnytimeFamilywisePlan(
        task_names=tuple(sorted(task for task, _mean, _gap in scenario)),
        familywise_alpha=0.05,
        futility_familywise_alpha=0.05,
        effect_margin=0.0,
        observation_lower=-1.0,
        observation_upper=1.0,
        maximum_observations_per_task=maximum_observations_per_task,
        e_process_method="betting_mixture",
        planning_effect_gaps=tuple(
            (task, gap) for task, _mean, gap in scenario
        ),
        resolution_familywise_beta=0.05,
    )


def required_targets(
    scenario: tuple[tuple[str, float, float], ...],
) -> tuple:
    provisional = _plan(scenario, maximum_observations_per_task=1_000_000)
    return AnytimeFamilywiseRouter(provisional).certified_sample_targets()


def run_trial(
    scenario: tuple[tuple[str, float, float], ...],
    *,
    seed: int,
) -> dict:
    targets = required_targets(scenario)
    maximum = max(target.required_observations for target in targets)
    plan = _plan(scenario, maximum_observations_per_task=maximum)
    router = AnytimeFamilywiseRouter(plan)
    means = {task: mean for task, mean, _gap in scenario}
    rngs = {
        task: random.Random(task_stream_seed(seed, task)) for task in plan.task_names
    }
    targets_by_task = {target.task: target for target in targets}

    for observation_index in range(maximum):
        for task in plan.task_names:
            if router.evidence(task).decision is not RouteDecision.UNDECIDED:
                continue
            if observation_index >= targets_by_task[task].required_observations:
                continue
            router.update(
                task,
                bounded_two_point_observation(
                    means[task],
                    lower=plan.observation_lower,
                    upper=plan.observation_upper,
                    rng=rngs[task],
                ),
            )

    unresolved = []
    incorrect = []
    for task, evidence in router.decisions().items():
        if evidence.decision in {
            RouteDecision.UNDECIDED,
            RouteDecision.BUDGET_EXHAUSTED,
        }:
            unresolved.append(task)
            continue
        expected = (
            RouteDecision.ACCEPT_CANDIDATE
            if means[task] > 0.0
            else RouteDecision.REJECT_TO_FALLBACK
        )
        if evidence.decision is not expected:
            incorrect.append(task)
    budget = sum(target.required_observations for target in targets)
    return {
        "unresolved": unresolved,
        "incorrect": incorrect,
        "observations": router.total_observations(),
        "quota_budget": budget,
        "quota_budget_respected": router.total_observations() <= budget,
    }


def run_check(
    *,
    trials: int,
    seed: int,
    workers: int,
) -> dict:
    if trials <= 0 or workers <= 0:
        raise ValueError("trials and workers must be positive")
    scenario = separated_scenario()
    target_records = required_targets(scenario)
    trial_seeds = [seed + index for index in range(trials)]
    worker = partial(run_trial, scenario)
    if workers == 1:
        results = [worker(seed=trial_seed) for trial_seed in trial_seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(worker, trial_seeds, chunksize=1))
    unresolved_families = sum(bool(result["unresolved"]) for result in results)
    incorrect_families = sum(bool(result["incorrect"]) for result in results)
    combined_families = sum(
        bool(result["unresolved"] or result["incorrect"]) for result in results
    )
    return {
        "design": "riskshiftbench-frontier-v2-nonbinding-resolution-check-v1",
        "scope": "Synthetic development diagnostic; no external outcome is read.",
        "implementation_sha256": implementation_sha256(),
        "statistical_implementation_sha256": statistical_implementation_sha256(),
        "trials": trials,
        "seed": seed,
        "task_count": len(scenario),
        "familywise_alpha": 0.05,
        "resolution_familywise_beta": 0.05,
        "scenario": [
            {"task": task, "true_mean": mean, "planning_effect_gap": gap}
            for task, mean, gap in scenario
        ],
        "certified_sample_targets": [asdict(target) for target in target_records],
        "maximum_required_observations": max(
            target.required_observations for target in target_records
        ),
        "global_quota_budget": sum(
            target.required_observations for target in target_records
        ),
        "all_targets_nonbinding": not any(
            target.clipped_by_task_cap for target in target_records
        ),
        "all_trials_respected_quota_budget": all(
            result["quota_budget_respected"] for result in results
        ),
        "unresolved_family_count": unresolved_families,
        "unresolved_family_rate": unresolved_families / trials,
        "unresolved_family_wilson_95_ci": list(
            wilson_interval(unresolved_families, trials)
        ),
        "incorrect_family_count": incorrect_families,
        "incorrect_family_rate": incorrect_families / trials,
        "incorrect_family_wilson_95_ci": list(
            wilson_interval(incorrect_families, trials)
        ),
        "combined_failure_family_count": combined_families,
        "combined_failure_family_rate": combined_families / trials,
        "combined_failure_family_wilson_95_ci": list(
            wilson_interval(combined_families, trials)
        ),
        "mean_observations": sum(result["observations"] for result in results)
        / trials,
        "maximum_observations": max(result["observations"] for result in results),
    }


def audit_resolution_bound_check(payload: dict) -> dict:
    if payload.get("design") != "riskshiftbench-frontier-v2-nonbinding-resolution-check-v1":
        raise RuntimeError("unexpected nonbinding resolution-check design")
    if payload.get("implementation_sha256") != implementation_sha256():
        raise RuntimeError("nonbinding resolution-check implementation changed")
    if payload.get("statistical_implementation_sha256") != statistical_implementation_sha256():
        raise RuntimeError("statistical implementation changed after resolution check")
    if int(payload.get("trials", -1)) < 1_000 or int(payload.get("task_count", -1)) != 23:
        raise RuntimeError("nonbinding resolution check is undersized")
    if payload.get("all_targets_nonbinding") is not True:
        raise RuntimeError("certified sample targets remain clipped")
    if payload.get("all_trials_respected_quota_budget") is not True:
        raise RuntimeError("a resolution trial exceeded the certified quota budget")
    combined_interval = payload.get("combined_failure_family_wilson_95_ci")
    if (
        not isinstance(combined_interval, list)
        or len(combined_interval) != 2
        or not 0.0 <= float(combined_interval[0]) <= float(combined_interval[1]) <= 0.15
    ):
        raise RuntimeError("nonbinding combined-failure diagnostic exceeds 0.15")
    return {
        "trials": int(payload["trials"]),
        "task_count": int(payload["task_count"]),
        "global_quota_budget": int(payload["global_quota_budget"]),
        "maximum_required_observations": int(
            payload["maximum_required_observations"]
        ),
        "unresolved_family_rate": float(payload["unresolved_family_rate"]),
        "incorrect_family_rate": float(payload["incorrect_family_rate"]),
        "combined_failure_family_rate": float(
            payload["combined_failure_family_rate"]
        ),
        "combined_failure_family_wilson_95_ci": [
            float(value) for value in combined_interval
        ],
        "all_targets_nonbinding": True,
        "all_trials_respected_quota_budget": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2_026_072_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/frontier_v2_development") / RESOLUTION_BOUND_FILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_check(trials=args.trials, seed=args.seed, workers=args.workers)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
