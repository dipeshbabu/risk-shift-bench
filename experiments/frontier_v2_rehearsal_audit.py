"""Validate outcome-eligible v2 adapter rehearsal artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.frontier_v2_external_design import DOMAIN_SPECS


def audit_rehearsal_payload(payload: dict) -> dict:
    if payload.get("design") != "riskshiftbench-frontier-v2-development-smoke":
        raise RuntimeError("unexpected rehearsal design identifier")
    domain = payload.get("domain")
    if domain not in DOMAIN_SPECS:
        raise RuntimeError(f"unknown rehearsal domain: {domain}")
    if payload.get("split") not in {"development", "calibration"}:
        raise RuntimeError("rehearsal artifact is not outcome-eligible")
    if "confirmation" in str(payload.get("task", "")):
        raise RuntimeError("confirmation task found in a rehearsal artifact")
    if payload.get("determinism_verified") is not True:
        raise RuntimeError("rehearsal artifact lacks a successful deterministic rerun")

    spec = DOMAIN_SPECS[domain]
    expected_policies = {
        spec.fallback_policy,
        *spec.candidate_policies,
    }
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != expected_policies:
        raise RuntimeError(f"policy library is incomplete for {domain}")

    reference_seeds = None
    reference_episodes = None
    observed_scores = []
    for policy in sorted(outcomes):
        rows = outcomes[policy]
        if not rows:
            raise RuntimeError(f"empty rehearsal outcome for {domain}/{policy}")
        seeds = [int(row["seed"]) for row in rows]
        episodes = [int(row["episode"]) for row in rows]
        if len(set(seeds)) != len(seeds):
            raise RuntimeError(f"duplicate episode seed for {domain}/{policy}")
        if reference_seeds is None:
            reference_seeds = seeds
            reference_episodes = episodes
        elif seeds != reference_seeds or episodes != reference_episodes:
            raise RuntimeError(f"common-random-number pairing failed for {domain}")
        for row in rows:
            if row["domain"] != domain or row["task"] != payload["task"]:
                raise RuntimeError(f"row metadata mismatch for {domain}/{policy}")
            if row["policy"] != policy:
                raise RuntimeError(f"row policy mismatch for {domain}/{policy}")
            score = float(row["score"])
            if not spec.score_lower <= score <= spec.score_upper:
                raise RuntimeError(f"score bound violation for {domain}/{policy}")
            observed_scores.append(score)
    return {
        "domain": domain,
        "task": payload["task"],
        "split": payload["split"],
        "policy_count": len(expected_policies),
        "episodes_per_policy": len(reference_seeds),
        "common_random_numbers": True,
        "determinism_verified": True,
        "minimum_score": min(observed_scores),
        "maximum_score": max(observed_scores),
        "score_bounds": [spec.score_lower, spec.score_upper],
    }


def audit_rehearsal_files(paths: tuple[Path, ...]) -> dict:
    if not paths:
        raise ValueError("at least one rehearsal artifact is required")
    records = []
    seen_domains = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = audit_rehearsal_payload(payload)
        if record["domain"] in seen_domains:
            raise RuntimeError(f"duplicate rehearsal domain: {record['domain']}")
        seen_domains.add(record["domain"])
        records.append(record)
    return {
        "design": "riskshiftbench-frontier-v2-adapter-rehearsal-audit",
        "scope": "Development/calibration artifacts only; confirmation is prohibited.",
        "domain_count": len(records),
        "complete_nine_domain_smoke": seen_domains == set(DOMAIN_SPECS),
        "all_scores_bounded": True,
        "all_common_random_numbers": True,
        "all_determinism_verified": True,
        "records": sorted(records, key=lambda record: record["domain"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_rehearsal_files(tuple(args.paths))
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
