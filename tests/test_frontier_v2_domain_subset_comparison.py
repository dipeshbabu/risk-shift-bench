from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.frontier_v2_domain_subset_comparison import (
    compare_task_subset,
    task_policy_matrix,
)
from experiments.frontier_v2_external_design import DOMAIN_SPECS


def payload(task: str, offset: float, *, domain: str = "gymnasium_taxi") -> dict:
    spec = DOMAIN_SPECS[domain]
    policies = (spec.fallback_policy, *spec.candidate_policies)
    return {
        "domain": domain,
        "task": task,
        "split": "development",
        "outcomes": {
            policy: [{"score": offset + 0.1 * index}]
            for index, policy in enumerate(policies)
        },
    }


def test_task_policy_matrix_is_rectangular() -> None:
    domain, matrix = task_policy_matrix([payload("a", 0.1), payload("b", 0.2)])
    assert domain == "gymnasium_taxi"
    assert set(matrix) == {"a", "b"}
    assert set(matrix["a"]) == {
        DOMAIN_SPECS[domain].fallback_policy,
        *DOMAIN_SPECS[domain].candidate_policies,
    }


def test_task_policy_matrix_rejects_mixed_domains() -> None:
    with pytest.raises(RuntimeError, match="mix"):
        task_policy_matrix(
            [payload("a", 0.1), payload("b", 0.2, domain="gymnasium_cliffwalking")]
        )


def test_comparison_runs_on_serialized_development_payloads(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"task_{index}.json"
        path.write_text(json.dumps(payload(f"task_{index}", 0.1 * index)), encoding="utf-8")
        paths.append(path)
    result = compare_task_subset(tuple(paths), subset_size=2)
    assert result["domain"] == "gymnasium_taxi"
    assert result["task_count"] == 3
    assert len(result["subset_result"]["selected_tasks"]) == 2
