from __future__ import annotations

from experiments.frontier_v2_resolution_bound_check import (
    required_targets,
    run_check,
    run_trial,
    separated_scenario,
)


def test_nonbinding_scenario_has_23_unclipped_certified_targets() -> None:
    targets = required_targets(separated_scenario())
    assert len(targets) == 23
    assert not any(target.clipped_by_task_cap for target in targets)
    assert max(target.required_observations for target in targets) == 5127


def test_resolution_trial_never_exceeds_sum_of_certified_quotas() -> None:
    result = run_trial(separated_scenario(), seed=7)
    assert result["quota_budget_respected"] is True
    assert result["observations"] <= result["quota_budget"]


def test_parallel_resolution_check_executes_picklable_trials() -> None:
    result = run_check(trials=2, seed=9, workers=2)
    assert result["trials"] == 2
    assert result["all_trials_respected_quota_budget"] is True
