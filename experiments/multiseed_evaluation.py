"""Run multi-seed policy evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from risk_preference_inference.config import load_benchmark_config
from risk_preference_inference.envs import benchmark_tasks
from risk_preference_inference.multiseed import aggregate_seed_scores, paired_policy_deltas, run_multiseed_evaluation
from risk_preference_inference.reporting import write_json


def parse_csv_value(value: str) -> str | int | float:
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [{key: parse_csv_value(value) for key, value in row.items()} for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_tasks(names: list[str] | None) -> list:
    tasks = benchmark_tasks()
    if not names:
        return tasks
    requested = set(names)
    selected = [task for task in tasks if task.name in requested]
    missing = requested - {task.name for task in selected}
    if missing:
        raise ValueError(f"Unknown tasks: {sorted(missing)}")
    return selected


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_full.json")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--hand-depth", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--out-dir", default="artifacts/multiseed")
    parser.add_argument("--input-scores", default=None)
    parser.add_argument("--reference-policy", default="signed_regime_learned_ensemble")
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    seeds = parse_seeds(args.seeds)
    episodes = args.episodes if args.episodes is not None else config.episodes
    hand_depth = args.hand_depth if args.hand_depth is not None else config.hand_depth
    tasks = select_tasks(args.tasks or (list(config.tasks) if config.tasks is not None else None))

    if args.input_scores:
        rows = read_csv(Path(args.input_scores))
        aggregate = aggregate_seed_scores(rows)
        paired_deltas = paired_policy_deltas(rows, reference_policy=args.reference_policy)
    else:
        rows, aggregate, paired_deltas = run_multiseed_evaluation(
            tasks=tasks,
            seeds=seeds,
            episodes=episodes,
            hand_depth=hand_depth,
            reference_policy=args.reference_policy,
        )

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "seed_task_scores.csv", rows)
    write_csv(out_dir / "aggregate_scores.csv", aggregate)
    write_csv(out_dir / "paired_deltas.csv", paired_deltas)
    write_json(
        out_dir / "summary.json",
        {
            "config": asdict(config),
            "seeds": seeds,
            "episodes": episodes,
            "hand_depth": hand_depth,
            "tasks": [task.name for task in tasks],
            "seed_task_scores": rows,
            "aggregate_scores": aggregate,
            "paired_deltas": paired_deltas,
        },
    )

    for row in aggregate:
        if row["scope"] == "all_tasks":
            print(f"{row['policy']} | mean_score={row['mean_score']:.3f} std={row['std_score']:.3f}")
    for row in paired_deltas:
        print(
            f"{row['reference_policy']} - {row['baseline_policy']} | "
            f"delta={row['mean_delta']:.3f} se={row['stderr_delta']:.3f} "
            f"win_rate={row['win_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
