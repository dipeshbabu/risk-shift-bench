"""Machine-readable registration-readiness audit for RiskShiftBench v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.frontier_v2_baseline_audit import audit_baseline_manifest
from experiments.frontier_v2_baseline_design import COMPETITIVE_BASELINES
from experiments.frontier_v2_baseline_source_audit import audit_baseline_source_suite
from experiments.frontier_v2_external_design import outcome_implementation_sha256
from experiments.frontier_v2_rehearsal_audit import (
    audit_rehearsal_payload,
    audit_split_coverage_payloads,
)
from experiments.frontier_v2_source_audit import audit_source_suite


LEARNED_BASELINES = tuple(
    baseline for baseline in COMPETITIVE_BASELINES if baseline.kind == "learned_policy"
)


def _audit_rehearsal_directory(path: Path, split: str) -> dict:
    if not path.is_dir():
        raise RuntimeError(f"rehearsal directory is missing: {path}")
    payloads = []
    for artifact in sorted(path.glob("RSBv2-*.json")):
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        audit_rehearsal_payload(payload)
        payloads.append(payload)
    return audit_split_coverage_payloads(
        payloads,
        split=split,
        expected_episodes_per_policy=1,
    )


def registration_readiness(
    *,
    environment_source_root: Path,
    baseline_source_root: Path,
    rehearsal_root: Path,
    baseline_root: Path,
) -> dict:
    checks = []
    missing = []

    environment_sources = audit_source_suite(environment_source_root)
    checks.append(
        {
            "gate": "environment_source_audit",
            "passed": environment_sources["codebase_count"] == 4,
            "codebase_count": environment_sources["codebase_count"],
        }
    )
    baseline_sources = audit_baseline_source_suite(baseline_source_root)
    checks.append(
        {
            "gate": "baseline_source_audit",
            "passed": baseline_sources["source_count"] == 3,
            "source_count": baseline_sources["source_count"],
        }
    )

    rehearsal_directories = {
        "development": rehearsal_root / "development_portable_1ep",
        "calibration": rehearsal_root / "calibration_portable_1ep",
    }
    for split, directory in rehearsal_directories.items():
        try:
            audit = _audit_rehearsal_directory(directory, split)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            missing.append(f"{split} rehearsal: {error}")
            checks.append({"gate": f"{split}_rehearsal", "passed": False})
        else:
            checks.append(
                {
                    "gate": f"{split}_rehearsal",
                    "passed": True,
                    "task_count": audit["task_count"],
                    "row_count": audit["total_episode_rows"],
                    "outcome_implementation_sha256": audit[
                        "outcome_implementation_sha256"
                    ],
                }
            )

    audited_baselines = []
    for baseline in LEARNED_BASELINES:
        manifest = baseline_root / baseline.domain / baseline.name / "manifest.json"
        if not manifest.is_file():
            missing.append(f"learned baseline: {baseline.identifier}")
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            audit = audit_baseline_manifest(payload, checkpoint_root=baseline_root)
            replay = payload.get("selected_checkpoint_replay_audit")
            if baseline.implementation_source != "riskshiftbench_internal" and (
                not isinstance(replay, dict)
                or replay.get("calibration_replay_exact") is not True
                or int(replay.get("checkpoint_count", -1))
                != len(baseline.training_seeds)
            ):
                raise RuntimeError("selected-checkpoint replay audit is incomplete")
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            missing.append(f"invalid learned baseline {baseline.identifier}: {error}")
        else:
            audited_baselines.append(
                {
                    "baseline_identifier": baseline.identifier,
                    "training_seed_count": audit["training_seed_count"],
                    "training_steps_per_seed": audit["training_steps_per_seed"],
                    "checkpoint_files_verified": audit["checkpoint_files_verified"],
                    "selected_checkpoint_replay_exact": baseline.implementation_source
                    == "riskshiftbench_internal"
                    or replay["calibration_replay_exact"],
                }
            )
    checks.append(
        {
            "gate": "learned_competitive_baselines",
            "passed": len(audited_baselines) == len(LEARNED_BASELINES),
            "audited": len(audited_baselines),
            "required": len(LEARNED_BASELINES),
        }
    )

    ready = all(check["passed"] for check in checks) and not missing
    return {
        "design": "riskshiftbench-frontier-v2-registration-readiness-v1",
        "ready_for_registration": ready,
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "checks": checks,
        "learned_baselines": audited_baselines,
        "missing_or_invalid": missing,
        "confirmation_execution_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment-source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument(
        "--baseline-source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baseline_sources"),
    )
    parser.add_argument(
        "--rehearsal-root",
        type=Path,
        default=Path("artifacts/frontier_v2_full_rehearsal"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baselines"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = registration_readiness(
        environment_source_root=args.environment_source_root,
        baseline_source_root=args.baseline_source_root,
        rehearsal_root=args.rehearsal_root,
        baseline_root=args.baseline_root,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
