"""Generate development and calibration score caches for inventory control."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from experiments.inventory_domain import (
    inventory_calibration_tasks,
    inventory_development_tasks,
    inventory_policy_grid,
    run_inventory_benchmark,
)
from risk_shift_bench.adaptive_search import summary_score
from risk_shift_bench.reporting import write_json


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("inventory_dev", "inventory_calibration"), required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out-dir", default="artifacts/inventory_benchmark")
    args = parser.parse_args()

    tasks = (
        inventory_development_tasks()
        if args.suite == "inventory_dev"
        else inventory_calibration_tasks()
    )
    seeds = parse_seeds(args.seeds)
    policies = inventory_policy_grid()
    rows = []
    for seed in seeds:
        _episodes, summaries = run_inventory_benchmark(
            tasks=tasks,
            policies=policies,
            episodes=args.episodes,
            seed=seed,
        )
        for summary in summaries:
            row = asdict(summary)
            rows.append(
                {
                    "task": summary.task,
                    "seed": seed,
                    "policy": summary.policy,
                    "score": summary_score(summary),
                    **{key: value for key, value in row.items() if key not in {"task", "policy"}},
                }
            )
    aggregate = []
    cells: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        cells.setdefault((row["task"], row["policy"]), []).append(float(row["score"]))
    for (task, policy), values in sorted(cells.items()):
        aggregate.append(
            {
                "scope": "task",
                "task": task,
                "policy": policy,
                "n": len(values),
                "mean_score": sum(values) / len(values),
            }
        )

    suite_dir = Path(args.out_dir) / args.suite
    if (suite_dir / "summary.json").exists():
        raise RuntimeError(f"refusing to overwrite inventory cache: {suite_dir}")
    write_csv(suite_dir / "seed_task_scores.csv", rows)
    write_csv(suite_dir / "aggregate_scores.csv", aggregate)
    write_json(
        suite_dir / "summary.json",
        {
            "suite": args.suite,
            "seeds": seeds,
            "episodes": args.episodes,
            "tasks": [asdict(task) for task in tasks],
            "policies": [policy.name for policy in policies],
            "aggregate_scores": aggregate,
        },
    )
    print(f"suite={args.suite}")
    print(f"tasks={len(tasks)}")
    print(f"rows={len(rows)}")
    print(f"output={suite_dir}")


if __name__ == "__main__":
    main()
