from __future__ import annotations

import math

import pytest

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
    RouteDecision,
)


def test_uniform_alpha_weights_sum_to_familywise_level() -> None:
    plan = AnytimeFamilywisePlan(task_names=("a", "b", "c"))
    assert sum(plan.acceptance_alpha(task) for task in plan.task_names) == pytest.approx(
        plan.familywise_alpha
    )


def test_development_weights_are_normalized_without_changing_familywise_level() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("a", "b"),
        task_weights=(("a", 3.0), ("b", 1.0)),
    )
    assert plan.acceptance_alpha("a") == pytest.approx(0.0375)
    assert plan.acceptance_alpha("b") == pytest.approx(0.0125)
    assert sum(plan.acceptance_alpha(task) for task in plan.task_names) == pytest.approx(
        0.05
    )


def test_positive_bounded_stream_crosses_acceptance_boundary() -> None:
    plan = AnytimeFamilywisePlan(task_names=("positive",), maximum_observations_per_task=200)
    router = AnytimeFamilywiseRouter(plan)
    evidence = None
    for _ in range(200):
        evidence = router.update("positive", 1.0)
        if evidence.decision is not RouteDecision.UNDECIDED:
            break
    assert evidence is not None
    assert evidence.decision is RouteDecision.ACCEPT_CANDIDATE
    assert evidence.acceptance_log_e >= evidence.acceptance_log_threshold


@pytest.mark.parametrize("method", ["hoeffding_mixture", "betting_mixture"])
def test_each_e_process_method_accepts_clear_positive_stream(method: str) -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("positive",),
        maximum_observations_per_task=200,
        e_process_method=method,
    )
    router = AnytimeFamilywiseRouter(plan)
    while router.evidence("positive").decision is RouteDecision.UNDECIDED:
        router.update("positive", 1.0)
    assert router.evidence("positive").decision is RouteDecision.ACCEPT_CANDIDATE


def test_negative_bounded_stream_crosses_futility_boundary() -> None:
    plan = AnytimeFamilywisePlan(task_names=("negative",), maximum_observations_per_task=200)
    router = AnytimeFamilywiseRouter(plan)
    evidence = None
    for _ in range(200):
        evidence = router.update("negative", -1.0)
        if evidence.decision is not RouteDecision.UNDECIDED:
            break
    assert evidence is not None
    assert evidence.decision is RouteDecision.REJECT_TO_FALLBACK
    assert evidence.futility_log_e >= evidence.futility_log_threshold


def test_null_stream_does_not_gain_evidence_from_time_alone() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("null",),
        maximum_observations_per_task=20,
    )
    router = AnytimeFamilywiseRouter(plan)
    evidence = None
    for _ in range(20):
        evidence = router.update("null", 0.0)
    assert evidence is not None
    assert evidence.decision is RouteDecision.BUDGET_EXHAUSTED
    assert evidence.acceptance_log_e < 0.0
    assert evidence.futility_log_e < 0.0


def test_resolution_allocation_prioritizes_clear_task_after_forced_sampling() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("ambiguous", "clear"),
        maximum_observations_per_task=200,
    )
    router = AnytimeFamilywiseRouter(plan)
    router.update("ambiguous", 0.01)
    router.update("clear", 0.9)
    assert (
        router.next_task(strategy="resolution", forced_initial_observations=1)
        == "clear"
    )


def test_uniform_allocation_uses_sample_count_then_name() -> None:
    plan = AnytimeFamilywisePlan(task_names=("b", "a"))
    router = AnytimeFamilywiseRouter(plan)
    assert router.next_task(strategy="uniform") == "a"
    router.update("a", 0.0)
    assert router.next_task(strategy="uniform") == "b"


def test_observation_bounds_are_enforced() -> None:
    router = AnytimeFamilywiseRouter(AnytimeFamilywisePlan(task_names=("a",)))
    with pytest.raises(ValueError, match="outside"):
        router.update("a", 1.01)
    with pytest.raises(ValueError, match="finite"):
        router.update("a", math.nan)


def test_terminal_task_cannot_be_sampled_again() -> None:
    router = AnytimeFamilywiseRouter(
        AnytimeFamilywisePlan(task_names=("a",), maximum_observations_per_task=200)
    )
    while router.evidence("a").decision is RouteDecision.UNDECIDED:
        router.update("a", 1.0)
    with pytest.raises(RuntimeError, match="terminal"):
        router.update("a", 1.0)


def test_invalid_weight_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly"):
        AnytimeFamilywisePlan(
            task_names=("a", "b"),
            task_weights=(("a", 1.0),),
        )
