from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.frontier_v2_competitive_evaluation import (
    _score_records_summary,
    selected_checkpoint_records,
)


def test_selected_checkpoint_records_verify_unique_physical_files(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("checkpoint\n", encoding="utf-8")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "runs": [
            {
                "training_seed": 7,
                "selected_checkpoint_sha256": digest,
                "checkpoints": [
                    {
                        "step": 100,
                        "checkpoint_path": "checkpoint.json",
                        "checkpoint_sha256": digest,
                    }
                ],
            }
        ]
    }
    selected = selected_checkpoint_records(manifest, tmp_path)
    assert selected[0]["training_seed"] == 7
    assert selected[0]["checkpoint_path"] == checkpoint


def test_competitive_summary_uses_equal_task_weight() -> None:
    records = [
        {"task": "a", "score": 0.0, "cost": 2.0},
        {"task": "a", "score": 1.0, "cost": 0.0},
        {"task": "b", "score": 0.25, "cost": 1.0},
    ]
    summary = _score_records_summary(records)
    assert summary["equal_task_mean_score"] == pytest.approx(0.375)
    assert summary["equal_task_mean_cost"] == pytest.approx(1.0)
