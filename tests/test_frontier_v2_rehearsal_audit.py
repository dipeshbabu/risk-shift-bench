from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.frontier_v2_external_design import (
    CODEBASE_LOCKS,
    DOMAIN_SPECS,
    all_tasks,
    canonical_episode_seed_base,
    domain_tasks,
    expected_episode_seeds,
    outcome_implementation_sha256,
    task_manifest_sha256,
    task_sha256,
)
from experiments.frontier_v2_rehearsal_audit import (
    audit_rehearsal_files,
    audit_rehearsal_payload,
    audit_split_coverage_payloads,
)


def sample_payload(task=None, *, episodes: int = 2) -> dict:
    if task is None:
        task = domain_tasks("gymnasium_taxi", "development")[0]
    domain = task.domain
    spec = DOMAIN_SPECS[domain]
    lock = CODEBASE_LOCKS[spec.codebase]
    policies = (spec.fallback_policy, *spec.candidate_policies)
    seed_base = canonical_episode_seed_base(task)
    seeds = expected_episode_seeds(task, episodes=episodes, seed_base=seed_base)
    return {
        "design": "riskshiftbench-frontier-v2-development-task-v1",
        "scope": "Development only.",
        "domain": domain,
        "task": task.name,
        "split": task.split,
        "task_definition": asdict(task),
        "task_sha256": task_sha256(task),
        "split_manifest_sha256": task_manifest_sha256(all_tasks(task.split)),
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "source_audit": {
            "codebase": spec.codebase,
            "expected_commit": lock.commit,
            "observed_commit": lock.commit,
            "clean": True,
        },
        "codebase_lock": asdict(lock),
        "score_rule": spec.score_rule,
        "score_bounds": [spec.score_lower, spec.score_upper],
        "episodes_per_policy": episodes,
        "seed_base": seed_base,
        "canonical_seed_base": seed_base,
        "canonical_seed_schedule": True,
        "determinism_verified": True,
        "runtime_seconds": {
            "collection": 1.0,
            "determinism_verification": 1.0,
            "total": 2.0,
        },
        "summaries": {
            policy: {
                "domain": domain,
                "task": task.name,
                "policy": policy,
                "episodes": episodes,
                "mean_score": 0.5,
                "minimum_score": 0.5,
                "maximum_score": 0.5,
                "mean_raw_utility": 0.0,
                "failure_probability": 0.0,
                "mean_cost": 0.0,
                "mean_steps": 1.0,
            }
            for policy in policies
        },
        "outcomes": {
            policy: [
                {
                    "domain": domain,
                    "task": task.name,
                    "policy": policy,
                    "episode": episode,
                    "seed": seeds[episode],
                    "score": 0.5,
                    "raw_utility": 0.0,
                    "raw_return": 0.0,
                    "cost": 0.0,
                    "failure": False,
                    "steps": 1,
                    "successes": 0,
                }
                for episode in range(episodes)
            ]
            for policy in policies
        },
    }


def test_rehearsal_payload_checks_policy_library_and_pairing() -> None:
    record = audit_rehearsal_payload(sample_payload())
    assert record["domain"] == "gymnasium_taxi"
    assert record["common_random_numbers"] is True
    assert record["episodes_per_policy"] == 2


def test_rehearsal_payload_rejects_score_bound_violation() -> None:
    payload = sample_payload()
    first_policy = next(iter(payload["outcomes"]))
    payload["outcomes"][first_policy][0]["score"] = 1.01
    with pytest.raises(RuntimeError, match="bound"):
        audit_rehearsal_payload(payload)


def test_rehearsal_payload_rejects_unpaired_seeds() -> None:
    payload = sample_payload()
    second_policy = list(payload["outcomes"])[1]
    payload["outcomes"][second_policy][0]["seed"] = 999
    with pytest.raises(RuntimeError, match="seed"):
        audit_rehearsal_payload(payload)


def test_rehearsal_payload_rejects_stale_task_hash() -> None:
    payload = sample_payload()
    payload["task_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="task hash"):
        audit_rehearsal_payload(payload)


def test_rehearsal_payload_rejects_stale_implementation_hash() -> None:
    payload = sample_payload()
    payload["outcome_implementation_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="implementation hash"):
        audit_rehearsal_payload(payload)


def test_rehearsal_files_reject_duplicate_domains(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(sample_payload()), encoding="utf-8")
    second.write_text(json.dumps(copy.deepcopy(sample_payload())), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_rehearsal_files((first, second))


def test_full_split_audit_requires_exact_canonical_coverage() -> None:
    payloads = [sample_payload(task, episodes=1) for task in all_tasks("development")]
    audit = audit_split_coverage_payloads(
        payloads,
        split="development",
        expected_episodes_per_policy=1,
    )
    assert audit["complete_split"] is True
    assert audit["domain_count"] == 9
    assert audit["task_count"] == 36
    assert audit["total_episode_rows"] == 108


def test_full_split_audit_rejects_missing_task() -> None:
    payloads = [sample_payload(task, episodes=1) for task in all_tasks("calibration")]
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        audit_split_coverage_payloads(payloads[:-1], split="calibration")


def test_full_split_audit_rejects_custom_seed_schedule() -> None:
    payloads = [sample_payload(task, episodes=1) for task in all_tasks("development")]
    payloads[0]["canonical_seed_schedule"] = False
    with pytest.raises(RuntimeError, match="canonical"):
        audit_split_coverage_payloads(payloads, split="development")
