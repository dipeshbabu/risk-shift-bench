from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.frontier_v2_external_design import DOMAIN_SPECS, domain_tasks
from experiments.frontier_v2_rehearsal_audit import (
    audit_rehearsal_files,
    audit_rehearsal_payload,
)


def sample_payload() -> dict:
    domain = "gymnasium_taxi"
    spec = DOMAIN_SPECS[domain]
    task = domain_tasks(domain, "development")[0]
    policies = (spec.fallback_policy, *spec.candidate_policies)
    return {
        "design": "riskshiftbench-frontier-v2-development-smoke",
        "scope": "Development only.",
        "domain": domain,
        "task": task.name,
        "split": task.split,
        "determinism_verified": True,
        "summaries": {},
        "outcomes": {
            policy: [
                {
                    "domain": domain,
                    "task": task.name,
                    "policy": policy,
                    "episode": episode,
                    "seed": 100 + episode,
                    "score": 0.5,
                }
                for episode in range(2)
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
    with pytest.raises(RuntimeError, match="pairing"):
        audit_rehearsal_payload(payload)


def test_rehearsal_files_reject_duplicate_domains(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(sample_payload()), encoding="utf-8")
    second.write_text(json.dumps(copy.deepcopy(sample_payload())), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_rehearsal_files((first, second))
