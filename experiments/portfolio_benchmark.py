"""Run portfolio allocation benchmark suites."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from risk_shift_bench.adaptive_search import summary_score
from risk_shift_bench.multiseed import aggregate_seed_scores, paired_policy_deltas
from risk_shift_bench.portfolio_benchmark import portfolio_policy_grid, run_portfolio_benchmark
from risk_shift_bench.portfolio_envs import portfolio_suite_names, portfolio_tasks
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


def summarize_seed(seed: int, summaries) -> list[dict]:
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["seed"] = seed
        row["score"] = summary_score(summary)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--suite", choices=portfolio_suite_names(), default="portfolio_dev")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out-dir", default="artifacts/portfolio_benchmark")
    parser.add_argument("--reference-policy", default="learned_mixture_searched")
    args = parser.parse_args()

    if args.config:
        with Path(args.config).open(encoding="utf-8") as file:
            config = json.load(file)
        args.suite = config.get("suite", args.suite)
        args.seeds = config.get("seeds", args.seeds)
        args.episodes = int(config.get("episodes", args.episodes))
        args.out_dir = config.get("out_dir", args.out_dir)
        args.reference_policy = config.get("reference_policy", args.reference_policy)

    tasks = portfolio_tasks(args.suite)
    policies = portfolio_policy_grid()
    rows = []
    episodes_out = []
    for seed in parse_seeds(args.seeds):
        episodes, summaries = run_portfolio_benchmark(tasks=tasks, policies=policies, episodes=args.episodes, seed=seed)
        episodes_out.extend(asdict(episode) for episode in episodes)
        rows.extend(summarize_seed(seed, summaries))
    aggregate = aggregate_seed_scores(rows)
    paired = paired_policy_deltas(rows, reference_policy=args.reference_policy)

    out_dir = Path(args.out_dir) / args.suite
    write_csv(out_dir / "episodes.csv", episodes_out)
    write_csv(out_dir / "seed_task_scores.csv", rows)
    write_csv(out_dir / "aggregate_scores.csv", aggregate)
    write_csv(out_dir / "paired_deltas.csv", paired)
    write_json(
        out_dir / "summary.json",
        {
            "suite": args.suite,
            "seeds": parse_seeds(args.seeds),
            "episodes": args.episodes,
            "tasks": [task.name for task in tasks],
            "policies": [policy.name for policy in policies],
            "reference_policy": args.reference_policy,
            "protocol_summary": [row for row in aggregate if row["scope"] == "all_tasks"],
        },
    )
    for row in aggregate:
        if row["scope"] == "all_tasks":
            print(f"{args.suite} | {row['policy']} | mean_score={row['mean_score']:.3f}")


if __name__ == "__main__":
    main()
