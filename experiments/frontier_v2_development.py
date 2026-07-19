"""Run one outcome-eligible v2 development/calibration task and policy library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.frontier_v2_external_adapters import (
    outcome_rows,
    run_v2_development_task,
    summarize_v2_outcomes,
)
from experiments.frontier_v2_external_design import DOMAIN_SPECS, domain_tasks


SUPPORTED_DOMAINS = tuple(DOMAIN_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=SUPPORTED_DOMAINS, required=True)
    parser.add_argument("--split", choices=("development", "calibration"), default="development")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--policy", default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = domain_tasks(args.domain, args.split)
    if not 0 <= args.task_index < len(tasks):
        raise ValueError("task-index is outside the frozen task list")
    task = tasks[args.task_index]
    library = DOMAIN_SPECS[args.domain]
    policies = (
        (library.fallback_policy, *library.candidate_policies)
        if args.policy == "all"
        else (args.policy,)
    )
    outcomes = {
        policy: run_v2_development_task(
            task,
            policy,
            args.episodes,
            args.seed_base,
            args.source_root,
        )
        for policy in policies
    }
    if args.verify_determinism:
        repeated = {
            policy: run_v2_development_task(
                task,
                policy,
                args.episodes,
                args.seed_base,
                args.source_root,
            )
            for policy in policies
        }
        if repeated != outcomes:
            raise RuntimeError("determinism verification failed for the task policy library")
    payload = {
        "design": "riskshiftbench-frontier-v2-development-smoke",
        "scope": "Development/calibration only; confirmation execution is prohibited.",
        "task": task.name,
        "domain": task.domain,
        "split": task.split,
        "determinism_verified": args.verify_determinism,
        "summaries": {
            policy: summarize_v2_outcomes(rows)
            for policy, rows in outcomes.items()
        },
        "outcomes": {
            policy: outcome_rows(rows) for policy, rows in outcomes.items()
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
