"""Validate outcome-eligible v2 adapter rehearsal artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from experiments.frontier_v2_external_design import (
    CODEBASE_LOCKS,
    DOMAIN_SPECS,
    all_tasks,
    canonical_episode_seed_base,
    canonical_sha256,
    domain_tasks,
    expected_episode_seeds,
    task_manifest_sha256,
    task_sha256,
)


TASK_DESIGN = "riskshiftbench-frontier-v2-development-task-v1"


def _expected_task(payload: dict):
    domain = payload.get("domain")
    if domain not in DOMAIN_SPECS:
        raise RuntimeError(f"unknown rehearsal domain: {domain}")
    split = payload.get("split")
    if split not in {"development", "calibration"}:
        raise RuntimeError("rehearsal artifact is not outcome-eligible")
    matches = [
        task
        for task in domain_tasks(domain, split)
        if task.name == payload.get("task")
    ]
    if len(matches) != 1:
        raise RuntimeError("task is not an exact member of the frozen split manifest")
    return matches[0]


def _audit_provenance(payload: dict, task) -> None:
    if payload.get("task_sha256") != task_sha256(task):
        raise RuntimeError("task hash does not match the frozen task definition")
    if canonical_sha256(payload.get("task_definition")) != task_sha256(task):
        raise RuntimeError("serialized task definition does not match its frozen hash")
    expected_manifest = task_manifest_sha256(all_tasks(task.split))
    if payload.get("split_manifest_sha256") != expected_manifest:
        raise RuntimeError("split manifest hash does not match the frozen design")

    spec = DOMAIN_SPECS[task.domain]
    lock = CODEBASE_LOCKS[spec.codebase]
    if canonical_sha256(payload.get("codebase_lock")) != canonical_sha256(asdict(lock)):
        raise RuntimeError("serialized codebase lock does not match the frozen design")
    source_audit = payload.get("source_audit", {})
    if (
        source_audit.get("codebase") != spec.codebase
        or source_audit.get("expected_commit") != lock.commit
        or source_audit.get("observed_commit") != lock.commit
        or source_audit.get("clean") is not True
    ):
        raise RuntimeError("source audit does not prove a clean frozen upstream commit")
    if payload.get("score_rule") != spec.score_rule:
        raise RuntimeError("score rule does not match the frozen domain specification")
    if payload.get("score_bounds") != [spec.score_lower, spec.score_upper]:
        raise RuntimeError("score bounds do not match the frozen domain specification")


def _audit_runtime(payload: dict) -> float:
    runtime = payload.get("runtime_seconds", {})
    values = {
        name: float(runtime.get(name, float("nan")))
        for name in ("collection", "determinism_verification", "total")
    }
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise RuntimeError("runtime accounting is missing or invalid")
    if not math.isclose(
        values["total"],
        values["collection"] + values["determinism_verification"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError("runtime total does not match its recorded components")
    return values["total"]


def audit_rehearsal_payload(payload: dict) -> dict:
    if payload.get("design") != TASK_DESIGN:
        raise RuntimeError("unexpected rehearsal design identifier")
    task = _expected_task(payload)
    domain = task.domain
    _audit_provenance(payload, task)
    if payload.get("determinism_verified") is not True:
        raise RuntimeError("rehearsal artifact lacks a successful deterministic rerun")
    runtime_seconds = _audit_runtime(payload)

    spec = DOMAIN_SPECS[domain]
    expected_policies = {
        spec.fallback_policy,
        *spec.candidate_policies,
    }
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != expected_policies:
        raise RuntimeError(f"policy library is incomplete for {domain}")

    episode_count = int(payload.get("episodes_per_policy", 0))
    seed_base = int(payload.get("seed_base", -1))
    expected_seeds = list(
        expected_episode_seeds(task, episodes=episode_count, seed_base=seed_base)
    )
    expected_canonical_seed_base = canonical_episode_seed_base(task)
    canonical_seed_base = int(payload.get("canonical_seed_base", -1))
    canonical_schedule = (
        payload.get("canonical_seed_schedule") is True
        and canonical_seed_base == expected_canonical_seed_base
        and seed_base == canonical_seed_base
    )
    reference_seeds = None
    observed_scores = []
    for policy in sorted(outcomes):
        rows = outcomes[policy]
        if not rows:
            raise RuntimeError(f"empty rehearsal outcome for {domain}/{policy}")
        seeds = [int(row["seed"]) for row in rows]
        episodes = [int(row["episode"]) for row in rows]
        if len(set(seeds)) != len(seeds):
            raise RuntimeError(f"duplicate episode seed for {domain}/{policy}")
        if episodes != list(range(episode_count)):
            raise RuntimeError(f"episode indices do not match the frozen schedule for {domain}")
        if seeds != expected_seeds:
            raise RuntimeError(f"episode seeds do not match the recorded seed base for {domain}")
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
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
        summary = payload.get("summaries", {}).get(policy, {})
        if (
            summary.get("domain") != domain
            or summary.get("task") != task.name
            or summary.get("policy") != policy
            or int(summary.get("episodes", 0)) != episode_count
            or not math.isclose(
                float(summary.get("mean_score", float("nan"))),
                fmean(float(row["score"]) for row in rows),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"outcome summary does not match rows for {domain}/{policy}")
    return {
        "domain": domain,
        "task": task.name,
        "split": task.split,
        "task_sha256": task_sha256(task),
        "split_manifest_sha256": task_manifest_sha256(all_tasks(task.split)),
        "source_commit": CODEBASE_LOCKS[spec.codebase].commit,
        "policy_count": len(expected_policies),
        "episodes_per_policy": len(reference_seeds),
        "common_random_numbers": True,
        "canonical_seed_schedule": canonical_schedule,
        "determinism_verified": True,
        "runtime_seconds": runtime_seconds,
        "minimum_score": min(observed_scores),
        "maximum_score": max(observed_scores),
        "score_bounds": [spec.score_lower, spec.score_upper],
    }


def audit_split_coverage_payloads(
    payloads: list[dict],
    *,
    split: str,
    expected_episodes_per_policy: int | None = None,
) -> dict:
    """Require exact, provenance-checked coverage of one frozen 36-task split."""

    if split not in {"development", "calibration"}:
        raise ValueError("full coverage is restricted to outcome-eligible splits")
    expected_tasks = {task.name: task for task in all_tasks(split)}
    records = []
    seen_tasks = set()
    for payload in payloads:
        record = audit_rehearsal_payload(payload)
        if record["split"] != split:
            raise RuntimeError("full-coverage audit cannot mix data splits")
        if record["task"] in seen_tasks:
            raise RuntimeError(f"duplicate task artifact: {record['task']}")
        seen_tasks.add(record["task"])
        records.append(record)
    if seen_tasks != set(expected_tasks):
        missing = sorted(set(expected_tasks) - seen_tasks)
        extra = sorted(seen_tasks - set(expected_tasks))
        raise RuntimeError(f"split coverage mismatch; missing={missing}, extra={extra}")
    episode_counts = {record["episodes_per_policy"] for record in records}
    if len(episode_counts) != 1:
        raise RuntimeError("episode counts differ across split tasks")
    episode_count = next(iter(episode_counts))
    if (
        expected_episodes_per_policy is not None
        and episode_count != expected_episodes_per_policy
    ):
        raise RuntimeError("episode count differs from the required split design")
    if not all(record["canonical_seed_schedule"] for record in records):
        raise RuntimeError("full split coverage requires canonical disjoint seed schedules")
    domain_task_counts = {
        domain: sum(record["domain"] == domain for record in records)
        for domain in DOMAIN_SPECS
    }
    if set(domain_task_counts.values()) != {4}:
        raise RuntimeError("full split coverage requires four tasks per domain")
    return {
        "design": "riskshiftbench-frontier-v2-full-split-audit-v1",
        "scope": "Development/calibration only; confirmation is prohibited.",
        "split": split,
        "split_manifest_sha256": task_manifest_sha256(all_tasks(split)),
        "complete_split": True,
        "domain_count": len(domain_task_counts),
        "task_count": len(records),
        "domain_task_counts": domain_task_counts,
        "policies_per_task": 3,
        "episodes_per_policy": episode_count,
        "total_episode_rows": sum(
            record["policy_count"] * record["episodes_per_policy"]
            for record in records
        ),
        "total_runtime_seconds": sum(record["runtime_seconds"] for record in records),
        "all_scores_bounded": True,
        "all_common_random_numbers": True,
        "all_canonical_seed_schedules": True,
        "all_determinism_verified": True,
        "records": sorted(records, key=lambda record: record["task"]),
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
    parser.add_argument("--full-split", choices=("development", "calibration"))
    parser.add_argument("--expected-episodes-per-policy", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_split:
        payload = audit_split_coverage_payloads(
            [json.loads(path.read_text(encoding="utf-8")) for path in args.paths],
            split=args.full_split,
            expected_episodes_per_policy=args.expected_episodes_per_policy,
        )
    else:
        payload = audit_rehearsal_files(tuple(args.paths))
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
