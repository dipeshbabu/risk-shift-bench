from __future__ import annotations

from experiments.frontier_v2_proof_certificate import (
    audit_proof_certificate,
    build_proof_certificate,
)


def test_proof_certificate_checks_exact_null_paths_and_alpha_sum() -> None:
    payload = build_proof_certificate(maximum_horizon=8)
    audit = audit_proof_certificate(payload)
    assert audit["family_task_count"] == 36
    assert audit["task_acceptance_alpha_sum"] == 0.05
    assert set(audit["methods"]) == {"betting_mixture", "predictable_betting"}
    assert all(
        method["maximum_expected_stopped_acceptance_e"] <= 1.0 + 1e-12
        for method in audit["methods"].values()
    )
