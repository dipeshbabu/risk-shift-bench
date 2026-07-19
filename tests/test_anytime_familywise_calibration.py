from __future__ import annotations

import random

import pytest

from experiments.anytime_familywise_calibration import (
    SyntheticScenario,
    bounded_two_point_observation,
    collect_synthetic_trials,
    run_synthetic_trial,
    summarize_trials,
    task_stream_seed,
    wilson_interval,
)


def test_bounded_two_point_observation_respects_degenerate_endpoints() -> None:
    rng = random.Random(0)
    assert bounded_two_point_observation(-1.0, lower=-1.0, upper=1.0, rng=rng) == -1.0
    assert bounded_two_point_observation(1.0, lower=-1.0, upper=1.0, rng=rng) == 1.0


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(5, 100)
    assert lower < 0.05 < upper


def test_task_stream_seed_is_stable_and_task_specific() -> None:
    assert task_stream_seed(7, "a") == task_stream_seed(7, "a")
    assert task_stream_seed(7, "a") != task_stream_seed(7, "b")


def test_trial_budget_is_never_exceeded() -> None:
    scenario = SyntheticScenario(
        name="small",
        task_means=(("a", -0.5), ("b", 0.0), ("c", 0.5)),
    )
    result = run_synthetic_trial(
        scenario,
        strategy="resolution",
        seed=4,
        familywise_alpha=0.05,
        effect_margin=0.0,
        e_process_method="betting_mixture",
        maximum_observations_per_task=20,
        global_observation_budget=30,
        forced_initial_observations=1,
    )
    assert result.total_observations <= 30
    assert result.positive_task_count == 1


def test_summary_is_deterministic_for_fixed_seed() -> None:
    scenario = SyntheticScenario(
        name="small",
        task_means=(("a", -0.5), ("b", 0.0), ("c", 0.5)),
    )
    first = summarize_trials(
        scenario,
        strategy="uniform",
        trials=5,
        seed=9,
        maximum_observations_per_task=10,
        global_observation_budget=30,
    )
    second = summarize_trials(
        scenario,
        strategy="uniform",
        trials=5,
        seed=9,
        maximum_observations_per_task=10,
        global_observation_budget=30,
    )
    assert first == second
    assert first["trials"] == 5


def test_trial_collection_matches_direct_indexing() -> None:
    scenario = SyntheticScenario(
        name="small",
        task_means=(("a", -0.5), ("b", 0.0), ("c", 0.5)),
    )
    results = collect_synthetic_trials(
        scenario,
        strategy="uniform",
        trials=3,
        seed=14,
        familywise_alpha=0.05,
        effect_margin=0.0,
        e_process_method="betting_mixture",
        maximum_observations_per_task=10,
        global_observation_budget=20,
        forced_initial_observations=1,
        workers=1,
    )
    direct = [
        run_synthetic_trial(
            scenario,
            strategy="uniform",
            seed=14 + index,
            familywise_alpha=0.05,
            effect_margin=0.0,
            e_process_method="betting_mixture",
            maximum_observations_per_task=10,
            global_observation_budget=20,
            forced_initial_observations=1,
        )
        for index in range(3)
    ]
    assert results == direct


def test_invalid_synthetic_mean_is_rejected() -> None:
    with pytest.raises(ValueError, match="inside"):
        bounded_two_point_observation(
            1.1,
            lower=-1.0,
            upper=1.0,
            rng=random.Random(0),
        )


def test_certified_summary_reports_clipped_and_unclipped_quotas() -> None:
    scenario = SyntheticScenario(
        name="planned",
        task_means=(("hard", 0.2), ("easy", 0.8)),
        planning_effect_gaps=(("hard", 0.2), ("easy", 0.8)),
    )
    summary = summarize_trials(
        scenario,
        strategy="certified",
        trials=1,
        seed=3,
        maximum_observations_per_task=200,
        global_observation_budget=100,
    )
    assert len(summary["certified_sample_targets"]) == 2
    assert summary["certified_targets_clipped_by_task_cap"] >= 1
