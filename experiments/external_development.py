"""Run outcome-eligible development or calibration tasks for the external study."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from experiments.external_domain_adapters import (
    outcome_rows,
    run_external_task,
    summarize_outcomes,
)
from experiments.external_study_design import (
    DOMAINS,
    ENVIRONMENT_LOCKS,
    POLICY_LIBRARIES,
    domain_tasks,
    task_manifest_sha256,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--split", choices=("development", "calibration"), default="development")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--environment-source", type=Path, required=True)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/external_development_v1"))
    args = parser.parse_args()

    tasks = domain_tasks(args.domain, args.split)
    if args.task_limit is not None:
        if args.task_limit <= 0:
            raise ValueError("task-limit must be positive")
        tasks = tasks[: args.task_limit]
    library = POLICY_LIBRARIES[args.domain]
    policies = (library.fallback, *library.candidates)
    episodes = []
    aggregates = []
    for task_index, task in enumerate(tasks):
        for policy in policies:
            rows = run_external_task(
                task=task,
                policy=policy,
                episodes=args.episodes,
                seed_base=args.seed_base + task_index * 1_000_000,
                environment_source=args.environment_source,
            )
            episodes.extend(outcome_rows(rows))
            aggregates.append(summarize_outcomes(rows))

    root = args.out_dir / args.domain / args.split
    write_csv(root / "episodes.csv", episodes)
    write_csv(root / "aggregate_scores.csv", aggregates)
    summary = {
        "scope": "Development/calibration only; no external confirmation task was run.",
        "domain": args.domain,
        "split": args.split,
        "environment_lock": asdict(ENVIRONMENT_LOCKS[args.domain]),
        "task_count": len(tasks),
        "full_split_task_manifest_sha256": task_manifest_sha256(
            domain_tasks(args.domain, args.split)
        ),
        "executed_task_names": [task.name for task in tasks],
        "policies": policies,
        "episodes_per_task_policy": args.episodes,
        "seed_base": args.seed_base,
        "verified_environment_source_commit": ENVIRONMENT_LOCKS[args.domain].commit,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"external_development_summary={root / 'summary.json'}")


if __name__ == "__main__":
    main()
