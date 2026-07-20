from __future__ import annotations

import numpy as np

from experiments.frontier_v2_nonlearned_baselines import (
    finite_horizon_score_oracle,
    fractional_knapsack_upper_bound,
    newsvendor_base_stock_levels,
    poisson_quantile,
)


def test_fractional_knapsack_upper_bound_uses_the_best_fractions() -> None:
    assert fractional_knapsack_upper_bound(
        [4.0, 6.0],
        [8.0, 6.0],
        capacity=7.0,
    ) == 11.0


def test_finite_horizon_oracle_maximizes_terminal_bounded_score() -> None:
    transitions = {
        0: {
            0: [(1.0, 1, 0.0, True)],
            1: [(1.0, 1, 1.0, True)],
        },
        1: {
            0: [(1.0, 1, 0.0, True)],
            1: [(1.0, 1, 0.0, True)],
        },
    }
    actions = finite_horizon_score_oracle(
        transitions,
        domain="gymnasium_frozenlake",
        horizon=2,
    )
    assert int(actions[0, 0, 0]) == 1


def test_poisson_quantile_is_the_smallest_integer_reaching_probability() -> None:
    assert poisson_quantile(0.5, 0.0) == 0
    assert poisson_quantile(0.5, 1.0) == 1


def test_newsvendor_levels_are_economic_lead_time_quantiles() -> None:
    class Environment:
        lead_time = np.asarray([1, 2, 4])
        unit_price = np.asarray([2.0, 1.5, 1.0, 0.75])
        unit_cost = np.asarray([1.5, 1.0, 0.75, 0.5])
        demand_cost = np.asarray([0.10, 0.075, 0.05, 0.025])
        holding_cost = np.asarray([0.15, 0.10, 0.05, 0.0])

    levels, critical_ratios = newsvendor_base_stock_levels(
        Environment(), demand_mean=10.0
    )
    assert levels[0] < levels[1] < levels[2]
    assert all(0.0 < ratio < 1.0 for ratio in critical_ratios)
