from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from experiments.frontier_v2_baseline_audit import (
    audit_baseline_manifest,
    select_checkpoint,
)
from experiments.frontier_v2_baseline_design import (
    COMPETITIVE_BASELINES,
    baseline_design_summary,
)
from experiments.frontier_v2_external_design import all_tasks, task_manifest_sha256


def q_learning_spec():
    return next(
        baseline
        for baseline in COMPETITIVE_BASELINES
        if baseline.identifier
        == "gymnasium_cliffwalking:q_learning_reference"
    )


def sample_manifest() -> dict:
    baseline = q_learning_spec()
    design = baseline_design_summary()
    runs = []
    for seed in baseline.training_seeds:
        checkpoints = []
        for step in range(
            baseline.checkpoint_interval_steps,
            baseline.training_steps_per_seed + 1,
            baseline.checkpoint_interval_steps,
        ):
            checkpoints.append(
                {
                    "step": step,
                    "checkpoint_path": f"seed-{seed}/step-{step}.bin",
                    "checkpoint_sha256": f"{seed + step:064x}",
                    "calibration_equal_task_mean_score": (
                        step / baseline.training_steps_per_seed
                    ),
                    "calibration_equal_task_mean_cost": 0.0,
                }
            )
        runs.append(
            {
                "training_seed": seed,
                "training_steps": baseline.training_steps_per_seed,
                "checkpoints": checkpoints,
                "selected_checkpoint_sha256": checkpoints[-1][
                    "checkpoint_sha256"
                ],
            }
        )
    return {
        "protocol_id": "riskshiftbench-frontier-v2-baseline-checkpoints-v1",
        "baseline_design_sha256": design["design_sha256"],
        "baseline_implementation_sha256": design[
            "internal_implementation_sha256"
        ],
        "baseline_identifier": baseline.identifier,
        "baseline_spec": asdict(baseline),
        "development_manifest_sha256": task_manifest_sha256(
            all_tasks("development")
        ),
        "calibration_manifest_sha256": task_manifest_sha256(
            all_tasks("calibration")
        ),
        "source_lock": {"name": "riskshiftbench_internal"},
        "runs": runs,
    }


def test_baseline_manifest_audits_all_runs_and_selection() -> None:
    payload = json.loads(json.dumps(sample_manifest()))
    audit = audit_baseline_manifest(payload)
    assert audit["training_seed_count"] == 5
    assert audit["training_steps_per_seed"] == 500_000
    assert audit["selection_rule_verified"] is True
    assert all(
        record["selected_step"] == 500_000
        for record in audit["selected_checkpoints"]
    )


def test_baseline_manifest_rejects_cherry_picked_checkpoint() -> None:
    payload = sample_manifest()
    payload["runs"][0]["selected_checkpoint_sha256"] = payload["runs"][0][
        "checkpoints"
    ][0]["checkpoint_sha256"]
    with pytest.raises(RuntimeError, match="selection rule"):
        audit_baseline_manifest(payload)


def test_baseline_manifest_rejects_missing_training_seed() -> None:
    payload = sample_manifest()
    payload["runs"] = payload["runs"][:-1]
    with pytest.raises(RuntimeError, match="every frozen training seed"):
        audit_baseline_manifest(payload)


def test_safe_selection_prefers_feasible_score_then_earlier_step() -> None:
    checkpoints = [
        {
            "step": 50_000,
            "calibration_equal_task_mean_score": 0.7,
            "calibration_equal_task_mean_cost": 20.0,
        },
        {
            "step": 100_000,
            "calibration_equal_task_mean_score": 0.9,
            "calibration_equal_task_mean_cost": 30.0,
        },
        {
            "step": 150_000,
            "calibration_equal_task_mean_score": 0.7,
            "calibration_equal_task_mean_cost": 15.0,
        },
    ]
    selected = select_checkpoint(checkpoints, cost_limit=25.0)
    assert selected["step"] == 50_000


def test_baseline_manifest_rejects_any_confirmation_reference() -> None:
    payload = sample_manifest()
    payload["notes"] = "looked at confirmation results"
    with pytest.raises(RuntimeError, match="confirmation"):
        audit_baseline_manifest(payload)
