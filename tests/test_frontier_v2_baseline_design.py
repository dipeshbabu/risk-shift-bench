from __future__ import annotations

from experiments.frontier_v2_baseline_design import (
    BASELINE_SOURCE_LOCKS,
    COMPETITIVE_BASELINES,
    TRAINING_SEEDS,
    baseline_design_summary,
    validate_baseline_design,
)
from experiments.frontier_v2_external_design import DOMAIN_SPECS


def test_baseline_design_covers_every_declared_domain_reference() -> None:
    validate_baseline_design()
    for domain, spec in DOMAIN_SPECS.items():
        observed = {
            baseline.name
            for baseline in COMPETITIVE_BASELINES
            if baseline.domain == domain
        }
        assert observed == set(spec.competitive_baselines)


def test_learned_baselines_use_five_frozen_seeds_and_real_budgets() -> None:
    learned = [
        baseline
        for baseline in COMPETITIVE_BASELINES
        if baseline.kind == "learned_policy"
    ]
    assert learned
    assert all(baseline.training_seeds == TRAINING_SEEDS for baseline in learned)
    assert all(baseline.training_steps_per_seed >= 500_000 for baseline in learned)
    assert all(baseline.checkpoint_interval_steps == 50_000 for baseline in learned)


def test_safe_references_have_a_frozen_cost_limit() -> None:
    safe = [
        baseline
        for baseline in COMPETITIVE_BASELINES
        if baseline.implementation_source == "omnisafe"
    ]
    assert len(safe) == 4
    assert all(baseline.safety_cost_limit == 25.0 for baseline in safe)


def test_external_baseline_sources_are_commit_locked() -> None:
    assert set(BASELINE_SOURCE_LOCKS) == {
        "omnisafe",
        "rl_starter_files",
        "cleanrl",
    }
    assert all(len(lock.commit) == 40 for lock in BASELINE_SOURCE_LOCKS.values())


def test_baseline_design_summary_is_machine_hashable() -> None:
    summary = baseline_design_summary()
    assert summary["training_task_split"] == "development"
    assert summary["checkpoint_selection_split"] == "calibration"
    assert len(summary["internal_implementation_sha256"]) == 64
    assert len(summary["design_sha256"]) == 64
