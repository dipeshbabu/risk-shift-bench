"""Freeze development-only proposals for the external confirmation tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from experiments.conformal_router import ConformalAdvantageRouter, RouterParams, build_profiles
from experiments.external_study_design import (
    DOMAINS,
    POLICY_LIBRARIES,
    domain_tasks,
    task_manifest_sha256,
)


def sha256_file(path: Path) -> str:
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty proposal table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_map(path: Path, expected_tasks: list[str], policies: tuple[str, ...]) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        task_scores = output.setdefault(row["task"], {})
        if row["policy"] in task_scores:
            raise RuntimeError(
                f"duplicate aggregate score for {row['task']} / {row['policy']} in {path}"
            )
        task_scores[row["policy"]] = float(row["score"])
    if set(output) != set(expected_tasks):
        missing = sorted(set(expected_tasks) - set(output))
        extra = sorted(set(output) - set(expected_tasks))
        raise RuntimeError(f"task coverage changed for {path}: missing={missing}, extra={extra}")
    for task, values in output.items():
        if set(values) != set(policies):
            raise RuntimeError(f"policy coverage changed for {task}: {sorted(values)}")
    return output


def build_domain(
    domain: str,
    artifact_root: Path,
    out_dir: Path,
) -> tuple[list[dict], dict]:
    library = POLICY_LIBRARIES[domain]
    policies = (library.fallback, *library.candidates)
    fit_tasks = domain_tasks(domain, "development")
    calibration_tasks = domain_tasks(domain, "calibration")
    confirmation_tasks = domain_tasks(domain, "confirmation")
    fit_path = artifact_root / domain / "development" / "aggregate_scores.csv"
    calibration_path = artifact_root / domain / "calibration" / "aggregate_scores.csv"
    fit_scores = score_map(fit_path, [task.name for task in fit_tasks], policies)
    calibration_scores = score_map(
        calibration_path,
        [task.name for task in calibration_tasks],
        policies,
    )
    params = RouterParams(
        k=5,
        temperature=0.75,
        alpha=0.10,
        margin=0.0,
        min_fit_evidence=3,
        min_calibration_tasks=5,
        screen_min_mean_advantage=0.0,
        max_screened_candidates=1,
        fallback_policy=library.fallback,
    )
    fit_profiles = build_profiles(fit_tasks, fit_scores, lambda task: task.features)
    calibration_profiles = build_profiles(
        calibration_tasks,
        calibration_scores,
        lambda task: task.features,
    )
    try:
        router = ConformalAdvantageRouter(
            fit_profiles=fit_profiles,
            calibration_profiles=calibration_profiles,
            candidate_policies=library.candidates,
            params=params,
            feature_fn=lambda task: task.features,
        )
    except ValueError as error:
        if "no candidate policy passed" not in str(error):
            raise
        proposals = [
            {
                "domain": domain,
                "task": task.name,
                "proposal_active": False,
                "fallback_policy": library.fallback,
                "candidate_policy": "",
                "predicted_advantage": "",
                "support_radius": "",
                "reason": "no_positive_mean_development_candidate",
            }
            for task in confirmation_tasks
        ]
        report = {
            "domain": domain,
            "status": "no_positive_mean_development_candidate",
            "params": asdict(params),
            "fit_aggregate_sha256": sha256_file(fit_path),
            "calibration_aggregate_sha256": sha256_file(calibration_path),
            "confirmation_task_manifest_sha256": task_manifest_sha256(confirmation_tasks),
            "proposal_count": 0,
        }
    else:
        proposals = []
        for task in confirmation_tasks:
            decision = router.proposal(task)
            prediction = decision.prediction
            proposals.append(
                {
                    "domain": domain,
                    "task": task.name,
                    "proposal_active": decision.promoted,
                    "fallback_policy": library.fallback,
                    "candidate_policy": (
                        decision.selected_policy if decision.promoted else router.candidate_policies[0]
                    ),
                    "predicted_advantage": (
                        "" if prediction is None else prediction.predicted_advantage
                    ),
                    "support_radius": "" if prediction is None else prediction.support_radius,
                    "reason": decision.reason,
                }
            )
        report = {
            "domain": domain,
            "status": "proposals_frozen_from_development_and_calibration",
            "router": router.report_dict(),
            "fit_aggregate_sha256": sha256_file(fit_path),
            "calibration_aggregate_sha256": sha256_file(calibration_path),
            "confirmation_task_manifest_sha256": task_manifest_sha256(confirmation_tasks),
            "proposal_count": sum(bool(row["proposal_active"]) for row in proposals),
        }
    root = out_dir / domain
    write_csv(root / "proposals.csv", proposals)
    (root / "router_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return proposals, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/external_development_v1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/external_router_lock_v1"),
    )
    args = parser.parse_args()

    all_proposals = []
    reports = {}
    for domain in DOMAINS:
        proposals, report = build_domain(domain, args.artifact_root, args.out_dir)
        all_proposals.extend(proposals)
        reports[domain] = report
    write_csv(args.out_dir / "all_proposals.csv", all_proposals)
    proposal_family = [row for row in all_proposals if bool(row["proposal_active"])]
    summary = {
        "scope": "Frozen before any external confirmation reset.",
        "development_artifact_root": str(args.artifact_root),
        "domains": reports,
        "confirmation_task_count": len(all_proposals),
        "proposal_family_size": len(proposal_family),
        "proposal_tasks": [row["task"] for row in proposal_family],
        "all_proposals_sha256": sha256_file(args.out_dir / "all_proposals.csv"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"external_router_summary={args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
