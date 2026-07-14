"""Search and evaluate validation-selected state/action blend policies."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from risk_shift_bench.benchmark import run_benchmark
from risk_shift_bench.envs import benchmark_tasks
from risk_shift_bench.multiseed import aggregate_seed_scores, multiseed_policies, paired_policy_deltas, summarize_seed
from risk_shift_bench.reporting import write_json
from risk_shift_bench.state_action_blend_search import (
    search_state_action_blend,
    state_action_blend_from_params,
)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_splits(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_dev_tasks(validation_count: int) -> tuple[list, list]:
    tasks = benchmark_tasks("frontier_dev")
    if validation_count <= 0:
        raise ValueError("state/action blend search requires a positive dev validation count")
    if validation_count >= len(tasks):
        raise ValueError("validation count must be smaller than the frontier_dev task count")
    return tasks[:-validation_count], tasks[-validation_count:]


def evaluate_split(
    split: str,
    selected_policy,
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    out_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    tasks = benchmark_tasks(split)
    policies = [*multiseed_policies(), selected_policy]
    rows = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(tasks=tasks, policies=policies, episodes=episodes, seed=seed, hand_depth=hand_depth)
        rows.extend(summarize_seed(seed, summaries))

    aggregate = aggregate_seed_scores(rows)
    paired = paired_policy_deltas(rows, reference_policy=selected_policy.name)
    split_dir = out_dir / split
    write_csv(split_dir / "seed_task_scores.csv", rows)
    write_csv(split_dir / "aggregate_scores.csv", aggregate)
    write_csv(split_dir / "paired_deltas.csv", paired)
    write_json(
        split_dir / "summary.json",
        {
            "split": split,
            "seeds": seeds,
            "episodes": episodes,
            "hand_depth": hand_depth,
            "tasks": [task.name for task in tasks],
            "aggregate_scores": aggregate,
            "paired_deltas": paired,
        },
    )
    return rows, aggregate, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-seeds", default="0")
    parser.add_argument("--eval-seeds", default="0,1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--hand-depth", type=int, default=1)
    parser.add_argument("--dev-validation-count", type=int, default=4)
    parser.add_argument("--eval-splits", default="frontier_holdout,frontier_audit")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/state_action_blend_search")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    selection_seeds = parse_seeds(args.selection_seeds)
    eval_seeds = parse_seeds(args.eval_seeds)
    train_tasks, validation_tasks = split_dev_tasks(args.dev_validation_count)
    search_result = search_state_action_blend(
        validation_tasks=validation_tasks,
        seeds=selection_seeds,
        episodes=args.episodes,
        hand_depth=args.hand_depth,
        smoke=args.smoke,
    )
    selected_policy = state_action_blend_from_params(search_result.params, name="validated_state_action_blend")

    write_csv(out_dir / "candidate_scores.csv", search_result.candidate_scores)
    write_csv(out_dir / "dev_validation_scores.csv", search_result.validation_summaries)
    write_json(
        out_dir / "search_summary.json",
        {
            "selection_seeds": selection_seeds,
            "episodes": args.episodes,
            "hand_depth": args.hand_depth,
            "dev_train_tasks": [task.name for task in train_tasks],
            "dev_validation_tasks": [task.name for task in validation_tasks],
            "selected_params": asdict(search_result.params),
            "validation_score": search_result.validation_score,
            "candidate_scores": search_result.candidate_scores,
        },
    )

    protocol_rows = []
    for split in parse_splits(args.eval_splits):
        _rows, aggregate, _paired = evaluate_split(
            split=split,
            selected_policy=selected_policy,
            seeds=eval_seeds,
            episodes=args.episodes,
            hand_depth=args.hand_depth,
            out_dir=out_dir,
        )
        for row in aggregate:
            if row["scope"] == "all_tasks":
                protocol_rows.append({"split": split, **row})
    write_csv(out_dir / "protocol_summary.csv", protocol_rows)
    write_json(
        out_dir / "summary.json",
        {
            "selection_seeds": selection_seeds,
            "eval_seeds": eval_seeds,
            "episodes": args.episodes,
            "hand_depth": args.hand_depth,
            "selected_params": asdict(search_result.params),
            "validation_score": search_result.validation_score,
            "protocol_summary": protocol_rows,
        },
    )

    print(f"selected_params={asdict(search_result.params)}")
    print(f"validation_score={search_result.validation_score:.3f}")
    for split in parse_splits(args.eval_splits):
        print(split)
        split_rows = [row for row in protocol_rows if row["split"] == split]
        for row in sorted(split_rows, key=lambda item: item["mean_score"], reverse=True)[:6]:
            print(f"  {row['policy']} | mean_score={row['mean_score']:.3f} std={row['std_score']:.3f}")


if __name__ == "__main__":
    main()
