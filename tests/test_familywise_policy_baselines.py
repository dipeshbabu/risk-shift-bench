from __future__ import annotations

import pytest

from experiments.anytime_familywise_router import AnytimeFamilywisePlan, RouteDecision
from experiments.familywise_policy_baselines import (
    AlphaSpendingFamilywiseRouter,
    alpha_spending_radius,
    bonferroni_rejections,
    exact_sign_p_value,
    fixed_sample_hoeffding_p_value,
    fixed_sample_rejections,
    holm_rejections,
    task_resolution_bound,
)


def test_fixed_sample_hoeffding_p_value_orders_clear_effects() -> None:
    null_p = fixed_sample_hoeffding_p_value(
        [0.0] * 20,
        null_mean=0.0,
        lower=-1.0,
        upper=1.0,
    )
    positive_p = fixed_sample_hoeffding_p_value(
        [1.0] * 20,
        null_mean=0.0,
        lower=-1.0,
        upper=1.0,
    )
    assert null_p == 1.0
    assert positive_p < null_p


def test_exact_sign_p_value_matches_unanimous_probability() -> None:
    assert exact_sign_p_value([1.0] * 9) == pytest.approx(1.0 / 512.0)
    assert exact_sign_p_value([0.0] * 9) == 1.0


def test_bonferroni_and_holm_rejections() -> None:
    p_values = {"a": 0.001, "b": 0.02, "c": 0.5}
    assert bonferroni_rejections(p_values, 0.05) == {"a"}
    assert holm_rejections(p_values, 0.05) == {"a", "b"}


def test_fixed_sample_dispatch() -> None:
    observations = {"a": [1.0] * 20, "b": [-1.0] * 20}
    rejected = fixed_sample_rejections(
        observations,
        test="hoeffding",
        correction="bonferroni",
        familywise_alpha=0.05,
        null_mean=0.0,
        lower=-1.0,
        upper=1.0,
    )
    assert rejected == {"a"}


def test_alpha_spending_radius_shrinks_and_resolution_bound_separates() -> None:
    assert alpha_spending_radius(100, alpha=0.05, width=2.0) < alpha_spending_radius(
        10, alpha=0.05, width=2.0
    )
    small_gap = task_resolution_bound(0.2, alpha=0.05, width=2.0)
    large_gap = task_resolution_bound(0.5, alpha=0.05, width=2.0)
    assert large_gap < small_gap


def test_alpha_spending_router_accepts_and_rejects_clear_streams() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("positive", "negative"),
        maximum_observations_per_task=2_000,
    )
    router = AlphaSpendingFamilywiseRouter(plan)
    while router.evidence("positive").decision is RouteDecision.UNDECIDED:
        router.update("positive", 1.0)
    while router.evidence("negative").decision is RouteDecision.UNDECIDED:
        router.update("negative", -1.0)
    assert router.evidence("positive").decision is RouteDecision.ACCEPT_CANDIDATE
    assert router.evidence("negative").decision is RouteDecision.REJECT_TO_FALLBACK


def test_alpha_spending_race_samples_widest_interval() -> None:
    plan = AnytimeFamilywisePlan(task_names=("a", "b"))
    router = AlphaSpendingFamilywiseRouter(plan)
    assert router.next_task() == "a"
    router.update("a", 0.0)
    assert router.next_task() == "b"

