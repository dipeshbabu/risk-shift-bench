from __future__ import annotations

import pytest

from experiments.frontier_v2_statistical_readiness import _audit_null_payload


def _summary() -> dict:
    return {
        "e_process_method": "betting_mixture",
        "strategy": "certified",
        "scenario": "global_null",
        "trials": 10_000,
        "familywise_alpha": 0.05,
        "task_means": {"null": 0.0},
        "familywise_false_accept_rate": 0.01,
        "familywise_false_accept_wilson_95_ci": [0.008, 0.012],
    }


def test_null_audit_requires_current_method_and_conservative_interval() -> None:
    audited = _audit_null_payload(
        {"summaries": [_summary()]},
        expected_methods={("betting_mixture", "certified")},
    )
    assert audited[0]["trials"] == 10_000


def test_null_audit_rejects_interval_above_familywise_level() -> None:
    summary = _summary()
    summary["familywise_false_accept_wilson_95_ci"] = [0.04, 0.06]
    with pytest.raises(RuntimeError, match="exceeds"):
        _audit_null_payload(
            {"summaries": [summary]},
            expected_methods={("betting_mixture", "certified")},
        )
