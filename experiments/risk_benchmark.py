"""Run the adaptive risk-objective benchmark suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from risk_preference_inference.benchmark import run_benchmark
from risk_preference_inference.config import load_benchmark_config
from risk_preference_inference.envs import benchmark_suite_names, benchmark_tasks
from risk_preference_inference.policy_registry import core_policies, strong_baseline_grid
from risk_preference_inference.reporting import write_episode_jsonl, write_json, write_summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-depth", type=int, default=4)
    parser.add_argument("--suite", choices=benchmark_suite_names(), default="standard")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--policy-set", choices=("core", "strong"), default="core")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.config:
        config = load_benchmark_config(args.config)
        args.episodes = config.episodes
        args.seed = config.seed
        args.hand_depth = config.hand_depth
        args.suite = config.suite
        args.tasks = list(config.tasks) if config.tasks is not None else None
        args.policy_set = config.policy_set
        if args.out_dir is None:
            args.out_dir = config.out_dir

    tasks = benchmark_tasks(args.suite)
    if args.tasks:
        requested = set(args.tasks)
        tasks = [task for task in tasks if task.name in requested]
        missing = requested - {task.name for task in tasks}
        if missing:
            raise ValueError(f"Unknown tasks: {sorted(missing)}")

    policies = core_policies() if args.policy_set == "core" else strong_baseline_grid()
    episodes, summaries = run_benchmark(
        tasks=tasks,
        policies=policies,
        episodes=args.episodes,
        seed=args.seed,
        hand_depth=args.hand_depth,
    )

    out_dir = Path(args.out_dir or "artifacts/risk_benchmark")
    write_episode_jsonl(out_dir / "episodes.jsonl", episodes)
    write_summary_csv(out_dir / "summary.csv", summaries)
    write_json(
        out_dir / "summary.json",
        {
            "episodes": args.episodes,
            "seed": args.seed,
            "hand_depth": args.hand_depth,
            "suite": args.suite,
            "tasks": [task.name for task in tasks],
            "summaries": [asdict(summary) for summary in summaries],
        },
    )

    for summary in summaries:
        print(
            f"{summary.task} | {summary.policy} | "
            f"mean={summary.mean_final_bankroll:.2f} "
            f"cvar5={summary.cvar_5_final_bankroll:.2f} "
            f"ruin={summary.ruin_probability:.3f} "
            f"target={summary.target_probability:.3f}"
        )


if __name__ == "__main__":
    main()
