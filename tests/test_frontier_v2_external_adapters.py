from __future__ import annotations

import math

import pytest

from experiments.frontier_v2_external_adapters import (
    V2EpisodeOutcome,
    assert_development_execution,
    bounded_score,
    image_navigation_action,
    inventory_base_stock_levels,
    inventory_reorder_action,
    run_v2_development_task,
    safety_navigation_action,
    summarize_v2_outcomes,
    transformed_value_iteration_action_table,
)
from experiments.frontier_v2_external_design import domain_tasks


def test_bounded_score_clips_and_rejects_nonfinite_values() -> None:
    assert bounded_score(-2.0) == 0.0
    assert bounded_score(0.4) == 0.4
    assert bounded_score(3.0) == 1.0
    with pytest.raises(ValueError, match="finite"):
        bounded_score(math.nan)


def test_confirmation_execution_is_refused() -> None:
    assert_development_execution(domain_tasks("gymnasium_taxi", "development")[0])
    with pytest.raises(RuntimeError, match="prohibited"):
        assert_development_execution(domain_tasks("gymnasium_taxi", "confirmation")[0])


def test_dispatch_refuses_confirmation_before_source_access(tmp_path) -> None:
    task = domain_tasks("or_gym_inventory_management", "confirmation")[0]
    with pytest.raises(RuntimeError, match="prohibited"):
        run_v2_development_task(task, "base_stock_nominal", 1, 0, tmp_path)


def test_value_iteration_chooses_better_terminal_action() -> None:
    transitions = {
        0: {
            0: [(1.0, 1, 0.0, False)],
            1: [(1.0, 1, 1.0, True)],
        },
        1: {
            0: [(1.0, 1, 0.0, True)],
            1: [(1.0, 1, 0.0, True)],
        },
    }
    table = transformed_value_iteration_action_table(transitions, gamma=0.99)
    assert table[0] == 1


def _empty_image(width: int, height: int):
    return [[[1, 0, 0] for _y in range(height)] for _x in range(width)]


def test_image_navigation_turns_then_moves_toward_goal() -> None:
    image = _empty_image(5, 5)
    image[1][1] = [10, 0, 0]
    image[3][1] = [8, 1, 0]
    assert image_navigation_action(image, "image_shortest_path") == 2

    image[1][1] = [10, 0, 1]
    assert image_navigation_action(image, "image_shortest_path") == 0


def test_image_navigation_avoids_encoded_hazards() -> None:
    image = _empty_image(5, 5)
    image[1][1] = [10, 0, 0]
    image[3][1] = [8, 1, 0]
    image[2][1] = [9, 0, 0]
    assert image_navigation_action(image, "image_lava_clearance") in {0, 1}


def test_inventory_policy_levels_and_actions_are_ordered() -> None:
    lean = inventory_base_stock_levels(
        "base_stock_lean", demand_mean=20.0, lead_time_scale=1.0
    )
    nominal = inventory_base_stock_levels(
        "base_stock_nominal", demand_mean=20.0, lead_time_scale=1.0
    )
    buffered = inventory_base_stock_levels(
        "base_stock_buffered", demand_mean=20.0, lead_time_scale=1.0
    )
    assert all(left < middle < right for left, middle, right in zip(lean, nominal, buffered, strict=True))
    action = inventory_reorder_action(
        "base_stock_nominal",
        demand_mean=20.0,
        lead_time_scale=1.0,
        inventory_position=(40.0, 80.0, 120.0),
        supply_capacity=(100.0, 90.0, 80.0),
    )
    assert action == (10.0, 0.0, 0.0)


def test_safety_navigation_avoids_target_obstacle() -> None:
    observation = {
        "goal_lidar": [1.0, 0.0, 0.0, 0.0],
        "hazards_lidar": [1.0, 0.0, 0.0, 0.0],
    }
    greedy = safety_navigation_action("goal_greedy", observation)
    cautious = safety_navigation_action("hazard_aware_strict", observation)
    assert greedy == [1.0, 0.0]
    assert cautious[0] == 0.25
    assert cautious[1] != 0.0


def test_outcome_summary_preserves_score_bounds() -> None:
    rows = [
        V2EpisodeOutcome(
            domain="domain",
            task="task",
            policy="policy",
            episode=index,
            seed=index,
            score=score,
            raw_utility=score,
            raw_return=score,
            cost=0.0,
            failure=False,
            steps=1,
            successes=1,
        )
        for index, score in enumerate((0.25, 0.75))
    ]
    summary = summarize_v2_outcomes(rows)
    assert summary["mean_score"] == 0.5
    assert summary["minimum_score"] == 0.25
    assert summary["maximum_score"] == 0.75
