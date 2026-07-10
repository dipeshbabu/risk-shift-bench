"""Run policy ablations for the adaptive benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from risk_preference_inference.ablations import run_ablation_study
from risk_preference_inference.config import load_benchmark_config
from risk_preference_inference.envs import benchmark_suite_names, benchmark_tasks
from risk_preference_inference.reporting import write_json, write_summary_csv


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_tasks(names: list[str] | None, suite: str) -> list:
    tasks = benchmark_tasks(suite)
    if not names:
        return tasks
    requested = set(names)
    selected = [task for task in tasks if task.name in requested]
    missing = requested - {task.name for task in selected}
    if missing:
        raise ValueError(f"Unknown tasks: {sorted(missing)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_full.json")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hand-depth", type=int, default=None)
    parser.add_argument("--suite", choices=benchmark_suite_names(), default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--out-dir", default="artifacts/ablations")
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    episodes = args.episodes if args.episodes is not None else config.episodes
    seed = args.seed if args.seed is not None else config.seed + 900_000
    hand_depth = args.hand_depth if args.hand_depth is not None else config.hand_depth
    suite = args.suite or config.suite
    tasks = select_tasks(args.tasks or (list(config.tasks) if config.tasks is not None else None), suite)

    summaries, aggregate_scores, task_scores = run_ablation_study(
        tasks=tasks,
        episodes=episodes,
        seed=seed,
        hand_depth=hand_depth,
    )

    out_dir = Path(args.out_dir)
    write_summary_csv(out_dir / "summary.csv", summaries)
    write_csv(out_dir / "aggregate_scores.csv", aggregate_scores)
    write_csv(out_dir / "task_scores.csv", task_scores)
    write_json(
        out_dir / "summary.json",
        {
            "config": asdict(config),
            "episodes": episodes,
            "seed": seed,
            "hand_depth": hand_depth,
            "suite": suite,
            "tasks": [task.name for task in tasks],
            "summaries": [asdict(summary) for summary in summaries],
            "aggregate_scores": aggregate_scores,
            "task_scores": task_scores,
        },
    )

    for row in aggregate_scores:
        print(
            f"{row['policy']} | score={row['aggregate_score']:.3f} "
            f"gap={row['score_gap_to_best']:.3f} "
            f"mean={row['mean_final_bankroll']:.2f}"
        )


if __name__ == "__main__":
    main()
