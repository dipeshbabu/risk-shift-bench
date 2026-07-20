"""Diagnose whether v2 task policy libraries create meaningful contrasts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from experiments.frontier_v2_external_design import DOMAIN_SPECS
from experiments.frontier_v2_rehearsal_audit import (
    audit_rehearsal_payload,
    audit_split_coverage_payloads,
)


TRAJECTORY_FIELDS = (
    "score",
    "raw_utility",
    "raw_return",
    "cost",
    "failure",
    "steps",
    "successes",
)


def paired_policy_contrast(candidate_rows: list[dict], fallback_rows: list[dict]) -> dict:
    if not candidate_rows or len(candidate_rows) != len(fallback_rows):
        raise ValueError("paired policy rows must be nonempty and equal in length")
    candidate_keys = [
        (int(row["episode"]), int(row["seed"])) for row in candidate_rows
    ]
    fallback_keys = [
        (int(row["episode"]), int(row["seed"])) for row in fallback_rows
    ]
    if candidate_keys != fallback_keys:
        raise ValueError("candidate and fallback rows are not paired by episode and seed")
    score_differences = [
        float(candidate["score"]) - float(fallback["score"])
        for candidate, fallback in zip(
            candidate_rows, fallback_rows, strict=True
        )
    ]
    trajectory_differences = [
        any(candidate[field] != fallback[field] for field in TRAJECTORY_FIELDS)
        for candidate, fallback in zip(
            candidate_rows, fallback_rows, strict=True
        )
    ]
    return {
        "episodes": len(score_differences),
        "mean_paired_score_difference": fmean(score_differences),
        "mean_absolute_paired_score_difference": fmean(
            abs(value) for value in score_differences
        ),
        "nonzero_score_difference_fraction": fmean(
            float(value != 0.0) for value in score_differences
        ),
        "trajectory_difference_fraction": fmean(
            float(value) for value in trajectory_differences
        ),
    }


def task_policy_contrast(payload: dict) -> dict:
    record = audit_rehearsal_payload(payload)
    spec = DOMAIN_SPECS[record["domain"]]
    fallback_rows = payload["outcomes"][spec.fallback_policy]
    contrasts = {
        candidate: paired_policy_contrast(
            payload["outcomes"][candidate], fallback_rows
        )
        for candidate in spec.candidate_policies
    }
    return {
        "domain": record["domain"],
        "task": record["task"],
        "split": record["split"],
        "fallback_policy": spec.fallback_policy,
        "candidate_contrasts": contrasts,
        "any_observed_score_contrast": any(
            contrast["nonzero_score_difference_fraction"] > 0.0
            for contrast in contrasts.values()
        ),
        "any_observed_trajectory_contrast": any(
            contrast["trajectory_difference_fraction"] > 0.0
            for contrast in contrasts.values()
        ),
    }


def audit_policy_contrast_split(payloads: list[dict], *, split: str) -> dict:
    coverage = audit_split_coverage_payloads(payloads, split=split)
    records = [task_policy_contrast(payload) for payload in payloads]
    domains = {}
    for domain in DOMAIN_SPECS:
        domain_records = [record for record in records if record["domain"] == domain]
        candidate_contrasts = [
            contrast
            for record in domain_records
            for contrast in record["candidate_contrasts"].values()
        ]
        domains[domain] = {
            "task_count": len(domain_records),
            "tasks_with_observed_score_contrast": sum(
                record["any_observed_score_contrast"] for record in domain_records
            ),
            "tasks_with_observed_trajectory_contrast": sum(
                record["any_observed_trajectory_contrast"] for record in domain_records
            ),
            "mean_absolute_paired_score_difference": fmean(
                contrast["mean_absolute_paired_score_difference"]
                for contrast in candidate_contrasts
            ),
            "maximum_absolute_mean_paired_score_difference": max(
                abs(contrast["mean_paired_score_difference"])
                for contrast in candidate_contrasts
            ),
        }
    flagged = [
        domain
        for domain, summary in domains.items()
        if summary["tasks_with_observed_trajectory_contrast"] == 0
    ]
    if any(
        not math.isfinite(summary["mean_absolute_paired_score_difference"])
        for summary in domains.values()
    ):
        raise RuntimeError("nonfinite policy-contrast summary")
    return {
        "design": "riskshiftbench-frontier-v2-policy-contrast-audit-v1",
        "scope": "Outcome-eligible diagnostic only; no confirmation artifact is read.",
        "split": split,
        "split_manifest_sha256": coverage["split_manifest_sha256"],
        "episodes_per_policy": coverage["episodes_per_policy"],
        "domain_count": len(domains),
        "task_count": len(records),
        "domains": domains,
        "domains_without_observed_trajectory_contrast": flagged,
        "all_domains_have_observed_trajectory_contrast": not flagged,
        "interpretation": (
            "Observed contrasts are a necessary policy-library diagnostic, not a power "
            "or utility claim. A zero contrast can reflect a weak library or too few "
            "episodes and must be resolved before the confirmation design is frozen."
        ),
        "records": sorted(records, key=lambda record: record["task"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--split",
        choices=("development", "calibration"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_policy_contrast_split(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.paths],
        split=args.split,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
