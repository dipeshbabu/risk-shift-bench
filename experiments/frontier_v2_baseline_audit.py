"""Audit learned competitive-baseline checkpoints before confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from experiments.frontier_v2_baseline_design import (
    BASELINE_SOURCE_LOCKS,
    COMPETITIVE_BASELINES,
    baseline_design_summary,
)
from experiments.frontier_v2_baseline_runner_hash import (
    runner_implementation_files,
    runner_implementation_sha256,
)
from experiments.frontier_v2_external_design import (
    all_tasks,
    canonical_sha256,
    task_manifest_sha256,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_rank(checkpoint: dict, *, cost_limit: float | None) -> tuple:
    score = float(checkpoint["calibration_equal_task_mean_score"])
    cost = float(checkpoint["calibration_equal_task_mean_cost"])
    step = int(checkpoint["step"])
    if cost_limit is None:
        return (score, -step)
    if cost <= cost_limit:
        return (1, score, -step)
    return (0, -cost, score, -step)


def select_checkpoint(checkpoints: list[dict], *, cost_limit: float | None) -> dict:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    return max(checkpoints, key=lambda item: _checkpoint_rank(item, cost_limit=cost_limit))


def _baseline(identifier: str):
    matches = [
        baseline
        for baseline in COMPETITIVE_BASELINES
        if baseline.identifier == identifier
    ]
    if len(matches) != 1 or matches[0].kind != "learned_policy":
        raise RuntimeError(f"unknown learned baseline identifier: {identifier}")
    return matches[0]


def _audit_checkpoint(checkpoint: dict, *, expected_step: int) -> None:
    if int(checkpoint.get("step", -1)) != expected_step:
        raise RuntimeError("checkpoint steps do not match the frozen interval")
    path = PurePosixPath(str(checkpoint.get("checkpoint_path", "")))
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise RuntimeError("checkpoint path must be a nonempty relative POSIX path")
    checksum = str(checkpoint.get("checkpoint_sha256", ""))
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise RuntimeError("checkpoint SHA-256 is invalid")
    score = float(checkpoint.get("calibration_equal_task_mean_score", float("nan")))
    cost = float(checkpoint.get("calibration_equal_task_mean_cost", float("nan")))
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError("checkpoint calibration score is invalid")
    if not math.isfinite(cost) or cost < 0.0:
        raise RuntimeError("checkpoint calibration cost is invalid")


def audit_baseline_manifest(
    payload: dict,
    *,
    checkpoint_root: Path | None = None,
) -> dict:
    if payload.get("protocol_id") != "riskshiftbench-frontier-v2-baseline-checkpoints-v1":
        raise RuntimeError("unexpected baseline-checkpoint protocol identifier")
    design = baseline_design_summary()
    if payload.get("baseline_design_sha256") != design["design_sha256"]:
        raise RuntimeError("competitive-baseline design hash changed")
    if (
        payload.get("baseline_implementation_sha256")
        != design["internal_implementation_sha256"]
    ):
        raise RuntimeError("competitive-baseline implementation hash changed")
    baseline = _baseline(str(payload.get("baseline_identifier", "")))
    if canonical_sha256(payload.get("baseline_spec")) != canonical_sha256(
        asdict(baseline)
    ):
        raise RuntimeError("serialized baseline specification changed")
    if payload.get("development_manifest_sha256") != task_manifest_sha256(
        all_tasks("development")
    ):
        raise RuntimeError("development task manifest changed")
    if payload.get("calibration_manifest_sha256") != task_manifest_sha256(
        all_tasks("calibration")
    ):
        raise RuntimeError("calibration task manifest changed")
    if "confirmation" in json.dumps(payload, sort_keys=True).lower():
        raise RuntimeError("baseline training manifest must not contain confirmation data")

    if baseline.implementation_source in BASELINE_SOURCE_LOCKS:
        expected_lock = asdict(BASELINE_SOURCE_LOCKS[baseline.implementation_source])
        if payload.get("source_lock") != expected_lock:
            raise RuntimeError("external baseline source lock changed")
        expected_files = list(runner_implementation_files(baseline.algorithm))
        if payload.get("runner_implementation_files") != expected_files:
            raise RuntimeError("external baseline runner file set changed")
        if payload.get("runner_implementation_sha256") != runner_implementation_sha256(
            baseline.algorithm
        ):
            raise RuntimeError("external baseline runner implementation changed")
    elif payload.get("source_lock") != {"name": "riskshiftbench_internal"}:
        raise RuntimeError("internal baseline source marker changed")

    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != len(baseline.training_seeds):
        raise RuntimeError("baseline manifest does not contain every frozen training seed")
    if {int(run.get("training_seed", -1)) for run in runs} != set(
        baseline.training_seeds
    ):
        raise RuntimeError("baseline training seeds changed")

    selected_records = []
    expected_steps = list(
        range(
            baseline.checkpoint_interval_steps,
            baseline.training_steps_per_seed + 1,
            baseline.checkpoint_interval_steps,
        )
    )
    for run in runs:
        if int(run.get("training_steps", -1)) != baseline.training_steps_per_seed:
            raise RuntimeError("baseline training budget changed")
        checkpoints = run.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != len(expected_steps):
            raise RuntimeError("baseline checkpoint schedule is incomplete")
        for checkpoint, expected_step in zip(
            checkpoints, expected_steps, strict=True
        ):
            _audit_checkpoint(checkpoint, expected_step=expected_step)
            if checkpoint_root is not None:
                path = checkpoint_root / PurePosixPath(checkpoint["checkpoint_path"])
                if not path.is_file() or file_sha256(path) != checkpoint["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint file hash mismatch: {path}")
        selected = select_checkpoint(
            checkpoints,
            cost_limit=baseline.safety_cost_limit,
        )
        if run.get("selected_checkpoint_sha256") != selected["checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint violates the frozen selection rule")
        selected_records.append(
            {
                "training_seed": int(run["training_seed"]),
                "selected_step": int(selected["step"]),
                "selected_checkpoint_sha256": selected["checkpoint_sha256"],
                "calibration_equal_task_mean_score": float(
                    selected["calibration_equal_task_mean_score"]
                ),
                "calibration_equal_task_mean_cost": float(
                    selected["calibration_equal_task_mean_cost"]
                ),
            }
        )
    return {
        "design": "riskshiftbench-frontier-v2-baseline-checkpoint-audit-v1",
        "scope": "Development training and calibration selection only.",
        "baseline_identifier": baseline.identifier,
        "training_seed_count": len(selected_records),
        "training_steps_per_seed": baseline.training_steps_per_seed,
        "baseline_implementation_sha256": design[
            "internal_implementation_sha256"
        ],
        "runner_implementation_sha256": payload.get(
            "runner_implementation_sha256"
        ),
        "checkpoint_files_verified": checkpoint_root is not None,
        "selection_rule_verified": True,
        "selected_checkpoints": sorted(
            selected_records, key=lambda record: record["training_seed"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_baseline_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        checkpoint_root=args.checkpoint_root,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
