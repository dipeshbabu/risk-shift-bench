"""Machine-readable registration-readiness audit for RiskShiftBench v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.frontier_v2_baseline_audit import audit_baseline_manifest
from experiments.frontier_v2_baseline_design import COMPETITIVE_BASELINES
from experiments.frontier_v2_baseline_source_audit import audit_baseline_source_suite
from experiments.frontier_v2_external_design import outcome_implementation_sha256
from experiments.frontier_v2_nonlearned_baselines import (
    NONLEARNED_BASELINES,
    audit_nonlearned_manifest,
)
from experiments.frontier_v2_rehearsal_audit import (
    audit_rehearsal_payload,
    audit_split_coverage_payloads,
)
from experiments.frontier_v2_router_lock import audit_router_lock
from experiments.frontier_v2_source_audit import audit_source_suite
from experiments.frontier_v2_statistical_readiness import (
    audit_statistical_readiness,
)


LEARNED_BASELINES = tuple(
    baseline for baseline in COMPETITIVE_BASELINES if baseline.kind == "learned_policy"
)


def _audit_rehearsal_directory(
    path: Path,
    split: str,
    *,
    expected_episodes_per_policy: int,
) -> dict:
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
        expected_episodes_per_policy=expected_episodes_per_policy,
    )


def registration_readiness(
    *,
    environment_source_root: Path,
    baseline_source_root: Path,
    rehearsal_root: Path,
    baseline_root: Path,
    statistical_root: Path = Path("artifacts/frontier_v2_development"),
    router_lock_path: Path = Path("artifacts/frontier_v2_router_lock/router_lock.json"),
    v1_development_root: Path = Path("artifacts/external_development_v1"),
    v1_router_root: Path = Path("artifacts/external_router_lock_v1"),
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

    try:
        statistical_audit = audit_statistical_readiness(statistical_root)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        missing.append(f"statistical calibration: {error}")
        checks.append({"gate": "statistical_calibration", "passed": False})
        statistical_audit = None
    else:
        checks.append(
            {
                "gate": "statistical_calibration",
                "passed": True,
                "primary_null_families": statistical_audit["primary_null"][0][
                    "trials"
                ],
                "predictable_null_families": sum(
                    item["trials"] for item in statistical_audit["predictable_null"]
                ),
                "paired_method_trials": statistical_audit[
                    "paired_method_comparison"
                ]["trials"],
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
        "development": rehearsal_root / "development_portable_1ep_episode_lifetime",
        "calibration": rehearsal_root / "calibration_portable_1ep_episode_lifetime",
    }
    for split, directory in rehearsal_directories.items():
        try:
            audit = _audit_rehearsal_directory(
                directory,
                split,
                expected_episodes_per_policy=1,
            )
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

    sized_rehearsal_directories = {
        "development": rehearsal_root / "development_portable_20ep_episode_lifetime",
        "calibration": rehearsal_root / "calibration_portable_20ep_episode_lifetime",
    }
    for split, directory in sized_rehearsal_directories.items():
        try:
            audit = _audit_rehearsal_directory(
                directory,
                split,
                expected_episodes_per_policy=20,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            missing.append(f"sized {split} suite: {error}")
            checks.append({"gate": f"sized_{split}_suite", "passed": False})
        else:
            checks.append(
                {
                    "gate": f"sized_{split}_suite",
                    "passed": True,
                    "task_count": audit["task_count"],
                    "row_count": audit["total_episode_rows"],
                    "episodes_per_policy": 20,
                    "outcome_implementation_sha256": audit[
                        "outcome_implementation_sha256"
                    ],
                }
            )

    try:
        router_audit = audit_router_lock(
            router_lock_path,
            development_root=sized_rehearsal_directories["development"],
            calibration_root=sized_rehearsal_directories["calibration"],
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        missing.append(f"router lock: {error}")
        checks.append({"gate": "outcome_free_router_lock", "passed": False})
        router_audit = None
    else:
        checks.append(
            {
                "gate": "outcome_free_router_lock",
                "passed": True,
                "proposal_family_size": router_audit["proposal_family_size"],
                "paired_observation_budget": router_audit[
                    "paired_observation_budget"
                ],
                "router_lock_canonical_sha256": router_audit[
                    "router_lock_canonical_sha256"
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

    audited_nonlearned = []
    for baseline in NONLEARNED_BASELINES:
        manifest = baseline_root / baseline.domain / baseline.name / "manifest.json"
        if not manifest.is_file():
            missing.append(f"nonlearned baseline: {baseline.identifier}")
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            audit = audit_nonlearned_manifest(
                payload,
                source_root=environment_source_root,
                v1_development_root=v1_development_root,
                v1_router_root=v1_router_root,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            missing.append(f"invalid nonlearned baseline {baseline.identifier}: {error}")
        else:
            audited_nonlearned.append(audit)
    checks.append(
        {
            "gate": "nonlearned_competitive_baselines",
            "passed": len(audited_nonlearned) == len(NONLEARNED_BASELINES),
            "audited": len(audited_nonlearned),
            "required": len(NONLEARNED_BASELINES),
        }
    )

    ready = all(check["passed"] for check in checks) and not missing
    return {
        "design": "riskshiftbench-frontier-v2-registration-readiness-v1",
        "ready_for_registration": ready,
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "checks": checks,
        "learned_baselines": audited_baselines,
        "nonlearned_baselines": audited_nonlearned,
        "statistical_calibration": statistical_audit,
        "router_lock": router_audit,
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
    parser.add_argument(
        "--statistical-root",
        type=Path,
        default=Path("artifacts/frontier_v2_development"),
    )
    parser.add_argument(
        "--router-lock",
        type=Path,
        default=Path("artifacts/frontier_v2_router_lock/router_lock.json"),
    )
    parser.add_argument(
        "--v1-development-root",
        type=Path,
        default=Path("artifacts/external_development_v1"),
    )
    parser.add_argument(
        "--v1-router-root",
        type=Path,
        default=Path("artifacts/external_router_lock_v1"),
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
        statistical_root=args.statistical_root,
        router_lock_path=args.router_lock,
        v1_development_root=args.v1_development_root,
        v1_router_root=args.v1_router_root,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
