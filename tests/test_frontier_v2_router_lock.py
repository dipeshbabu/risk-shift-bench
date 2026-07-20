from __future__ import annotations

import pytest

from experiments.frontier_v2_external_design import DOMAIN_SPECS, domain_tasks
from experiments.frontier_v2_router_lock import (
    MAXIMUM_PLANNING_GAP,
    MINIMUM_PLANNING_GAP,
    paired_policy_effect,
    planning_gap,
    select_domain_candidate,
)


def _payload(task: str, domain: str, effects: dict[str, float]) -> dict:
    spec = DOMAIN_SPECS[domain]
    seeds = [101, 102]
    outcomes = {
        spec.fallback_policy: [
            {"seed": seed, "score": 0.4} for seed in seeds
        ]
    }
    for candidate, effect in effects.items():
        outcomes[candidate] = [
            {"seed": seed, "score": 0.4 + effect} for seed in seeds
        ]
    return {"task": task, "domain": domain, "outcomes": outcomes}


def test_paired_policy_effect_requires_common_seeds() -> None:
    payload = {
        "outcomes": {
            "candidate": [{"seed": 1, "score": 0.7}],
            "fallback": [{"seed": 2, "score": 0.4}],
        }
    }
    with pytest.raises(RuntimeError, match="common seeds"):
        paired_policy_effect(payload, "candidate", "fallback")


def test_select_domain_candidate_uses_equal_task_development_mean() -> None:
    domain = "gymnasium_frozenlake"
    candidates = DOMAIN_SPECS[domain].candidate_policies
    payloads = [
        _payload(
            task.name,
            domain,
            {candidates[0]: 0.10, candidates[1]: 0.05},
        )
        for task in domain_tasks(domain, "development")
    ]
    selected, diagnostics = select_domain_candidate(domain, payloads)
    assert selected == candidates[0]
    assert diagnostics[candidates[0]]["equal_task_mean_effect"] == pytest.approx(0.10)


def test_candidate_tie_break_follows_frozen_candidate_order() -> None:
    domain = "gymnasium_taxi"
    candidates = DOMAIN_SPECS[domain].candidate_policies
    payloads = [
        _payload(
            task.name,
            domain,
            {candidates[0]: 0.02, candidates[1]: 0.02},
        )
        for task in domain_tasks(domain, "development")
    ]
    selected, _diagnostics = select_domain_candidate(domain, payloads)
    assert selected == candidates[0]


@pytest.mark.parametrize(
    ("effect", "expected"),
    [
        (0.0, MINIMUM_PLANNING_GAP),
        (0.12, 0.12),
        (-0.20, 0.20),
        (0.90, MAXIMUM_PLANNING_GAP),
    ],
)
def test_planning_gap_is_frozen_bounded_absolute_transform(
    effect: float,
    expected: float,
) -> None:
    assert planning_gap(effect) == pytest.approx(expected)
