from __future__ import annotations

import pytest

from experiments.frontier_v2_double_dqn import (
    ACTION_COUNT,
    MAX_IMAGE_SIZE,
    SUPPORTED_BASELINES,
    epsilon_at_step,
    pooled_task_index,
)


def test_double_dqn_covers_all_declared_discrete_deep_references() -> None:
    assert set(SUPPORTED_BASELINES) == {
        "or_gym_online_knapsack",
        "minigrid_dynamic_obstacles",
        "minigrid_lava_crossing",
    }
    assert MAX_IMAGE_SIZE == {
        "minigrid_dynamic_obstacles": 16,
        "minigrid_lava_crossing": 11,
    }
    assert ACTION_COUNT == {
        "minigrid_dynamic_obstacles": 3,
        "minigrid_lava_crossing": 7,
        "or_gym_online_knapsack": 2,
    }


def test_epsilon_schedule_is_frozen_and_bounded() -> None:
    assert epsilon_at_step(0, 1_000_000) == 1.0
    assert epsilon_at_step(250_000, 1_000_000) == pytest.approx(0.525)
    assert epsilon_at_step(500_000, 1_000_000) == pytest.approx(0.05)
    assert epsilon_at_step(1_000_000, 1_000_000) == pytest.approx(0.05)


def test_pooled_task_schedule_is_balanced_and_seeded() -> None:
    assert [pooled_task_index(50_000_001, episode, 4) for episode in range(8)] == [
        1,
        2,
        3,
        0,
        1,
        2,
        3,
        0,
    ]


def test_runner_source_contains_double_dqn_and_replay_audit() -> None:
    source = __import__(
        "experiments.frontier_v2_double_dqn", fromlist=["ignored"]
    ).__file__
    content = open(source, encoding="utf-8").read()
    assert "next_actions = online_next.argmax" in content
    assert "target(next_obs_batch).gather" in content
    assert "calibration_replay_exact" in content


@pytest.mark.parametrize("arguments", [(-1, 10), (0, 0)])
def test_invalid_epsilon_schedules_are_rejected(arguments: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        epsilon_at_step(*arguments)
