"""Search target-branch delegates on held-out target-family tasks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from risk_preference_inference.envs import benchmark_tasks, target_family_split
from risk_preference_inference.reporting import write_json
from risk_preference_inference.target_search import (
    evaluate_target_baselines,
    search_target_branch_policy,
    target_score_report,
)


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
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-depth", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--selection-seeds", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/target_branch_search")
    args = parser.parse_args()

    train_tasks, test_tasks = target_family_split()
    bench_tasks = benchmark_tasks()
    result = search_target_branch_policy(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        benchmark_tasks=bench_tasks,
        episodes=args.episodes,
        seed=args.seed,
        hand_depth=args.hand_depth,
        smoke=args.smoke,
        max_candidates=args.max_candidates,
        selection_seeds=args.selection_seeds,
    )
    baseline_test = evaluate_target_baselines(
        tasks=test_tasks,
        episodes=args.episodes,
        seed=args.seed + 900_000,
        hand_depth=args.hand_depth,
    )
    baseline_benchmark = evaluate_target_baselines(
        tasks=bench_tasks,
        episodes=args.episodes,
        seed=args.seed + 901_000,
        hand_depth=args.hand_depth,
    )

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "train_summaries.csv", result.train_summaries)
    write_csv(out_dir / "test_summaries.csv", result.test_summaries)
    write_csv(out_dir / "benchmark_summaries.csv", result.benchmark_summaries)
    write_csv(out_dir / "baseline_test_summaries.csv", baseline_test)
    write_csv(out_dir / "baseline_benchmark_summaries.csv", baseline_benchmark)

    payload = {
        "config": {
            "episodes": args.episodes,
            "seed": args.seed,
            "hand_depth": args.hand_depth,
            "max_candidates": args.max_candidates,
            "selection_seeds": args.selection_seeds,
            "smoke": args.smoke,
            "train_tasks": [task.name for task in train_tasks],
            "test_tasks": [task.name for task in test_tasks],
            "benchmark_tasks": [task.name for task in bench_tasks],
        },
        "best_target_branch": asdict(result),
        "target_test_score_report": target_score_report(baseline_test + result.test_summaries),
        "benchmark_score_report": target_score_report(baseline_benchmark + result.benchmark_summaries),
    }
    write_json(out_dir / "summary.json", payload)

    print(f"best_train_target_score={result.train_score:.3f}")
    print(f"best_test_target_score={result.test_score:.3f}")
    print(f"best_benchmark_paper_score={result.benchmark_score:.3f}")
    print(f"best_params={asdict(result.params)}")
    print("target_test_scores=" + str(payload["target_test_score_report"]["policy_target_scores"]))
    print("benchmark_paper_scores=" + str(payload["benchmark_score_report"]["policy_paper_scores"]))


if __name__ == "__main__":
    main()
