from __future__ import annotations

from pathlib import Path

import pytest

from experiments.frontier_v2_omnisafe import (
    PADDED_OBSERVATION_SIZE,
    SUPPORTED_BASELINES,
    expected_omnisafe_checkpoints,
    omnisafe_custom_config,
    pooled_safety_task_index,
)


def test_safe_baseline_coverage_is_complete() -> None:
    assert len(SUPPORTED_BASELINES) == 4
    assert {baseline.algorithm for baseline in SUPPORTED_BASELINES.values()} == {
        "PPO-Lagrangian",
        "constrained policy optimization",
    }
    assert PADDED_OBSERVATION_SIZE == {
        "safety_gymnasium_point_goal": 60,
        "safety_gymnasium_point_button": 76,
    }


def test_omnisafe_checkpoint_schedule_matches_fifty_thousand_steps() -> None:
    checkpoints = expected_omnisafe_checkpoints(1_000_000)
    assert len(checkpoints) == 20
    assert checkpoints[0] == (5, 50_000)
    assert checkpoints[-1] == (100, 1_000_000)


def test_pool_schedule_is_balanced() -> None:
    assert [pooled_safety_task_index(50_000_001, index) for index in range(8)] == [
        1,
        2,
        3,
        0,
        1,
        2,
        3,
        0,
    ]


def test_cost_limit_is_in_algorithm_specific_location(tmp_path: Path) -> None:
    ppo = omnisafe_custom_config(
        "PPO-Lagrangian",
        training_seed=1,
        total_steps=50_000,
        log_directory=tmp_path,
        device="cpu",
    )
    cpo = omnisafe_custom_config(
        "constrained policy optimization",
        training_seed=1,
        total_steps=50_000,
        log_directory=tmp_path,
        device="cpu",
    )
    assert ppo["lagrange_cfgs"]["cost_limit"] == 25.0
    assert "cost_limit" not in ppo["algo_cfgs"]
    assert cpo["algo_cfgs"]["cost_limit"] == 25.0


def test_invalid_checkpoint_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        expected_omnisafe_checkpoints(55_000)
