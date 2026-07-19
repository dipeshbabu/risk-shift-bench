from __future__ import annotations

from dataclasses import asdict

import pytest

from experiments.frontier_v2_external_design import (
    CODEBASE_LOCKS,
    DOMAINS,
    DOMAIN_SPECS,
    SPLITS,
    all_tasks,
    design_summary,
    domain_tasks,
    task_manifest_sha256,
    validate_design,
)


def test_breadth_requirements_and_bounded_differences() -> None:
    validate_design()
    assert len(DOMAINS) == 9
    assert {spec.codebase for spec in DOMAIN_SPECS.values()} == set(CODEBASE_LOCKS)
    assert sum(
        spec.minimum_observation_coordinates >= 32
        for spec in DOMAIN_SPECS.values()
    ) >= 2
    for spec in DOMAIN_SPECS.values():
        assert (spec.score_lower, spec.score_upper) == (0.0, 1.0)
        assert (spec.paired_difference_lower, spec.paired_difference_upper) == (
            -1.0,
            1.0,
        )


def test_every_domain_has_four_disjoint_tasks_per_split() -> None:
    names = set()
    signatures = {domain: set() for domain in DOMAINS}
    for split in SPLITS:
        assert len(all_tasks(split)) == 4 * len(DOMAINS)
        for domain in DOMAINS:
            tasks = domain_tasks(domain, split)
            assert len(tasks) == 4
            for task in tasks:
                assert task.name not in names
                names.add(task.name)
                signature = (task.environment_id, task.parameters)
                assert signature not in signatures[domain]
                signatures[domain].add(signature)


def test_manifest_hash_is_stable_and_order_sensitive() -> None:
    tasks = all_tasks("development")
    assert task_manifest_sha256(tasks) == task_manifest_sha256(tasks)
    assert task_manifest_sha256(tasks) != task_manifest_sha256(list(reversed(tasks)))


def test_design_summary_contains_no_outcomes_or_execution_authority() -> None:
    summary = design_summary()
    assert summary["domain_count"] == 9
    assert summary["codebase_count"] == 4
    assert summary["confirmation_execution"] == "prohibited_before_external_registration"
    assert "outcomes" not in summary
    assert summary["splits"]["confirmation"]["task_count"] == 36
    assert set(summary["domains"]) == set(DOMAINS)


def test_serialized_tasks_do_not_hide_mutable_parameters() -> None:
    for task in all_tasks("confirmation"):
        payload = asdict(task)
        assert isinstance(payload["parameters"], tuple)
        assert tuple(sorted(payload["parameters"])) == payload["parameters"]


def test_unknown_domain_or_split_is_rejected() -> None:
    with pytest.raises(KeyError):
        domain_tasks("not_a_domain", "development")
    with pytest.raises(KeyError):
        domain_tasks(DOMAINS[0], "not_a_split")
