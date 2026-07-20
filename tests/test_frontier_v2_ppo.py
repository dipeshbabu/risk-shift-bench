from __future__ import annotations

import pytest

from experiments.frontier_v2_ppo import (
    INVENTORY_ACTION_BINS,
    INVENTORY_INPUT_SIZE,
    NUM_ENVS,
    RECURRENCE_STEPS,
    ROLLOUT_STEPS,
    ROLLOUT_TIME_STEPS,
    SUPPORTED_BASELINES,
    inventory_bin_to_action,
)


def test_ppo_covers_inventory_and_both_minigrid_domains() -> None:
    assert set(SUPPORTED_BASELINES) == {
        "or_gym_inventory_management",
        "minigrid_dynamic_obstacles",
        "minigrid_lava_crossing",
    }
    assert ROLLOUT_STEPS == 250
    assert NUM_ENVS == 10
    assert ROLLOUT_TIME_STEPS == 25
    assert RECURRENCE_STEPS == 5
    assert INVENTORY_INPUT_SIZE == 39
    assert INVENTORY_ACTION_BINS == 11


def test_inventory_action_bins_span_each_supply_capacity() -> None:
    assert inventory_bin_to_action((0, 5, 10), (100, 90, 80)) == (0, 45, 80)


def test_invalid_inventory_action_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="three stages"):
        inventory_bin_to_action((0, 1), (100, 90, 80))
