"""Search and evaluate task-regime switches between incumbent policies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from risk_shift_bench.benchmark import run_benchmark
from risk_shift_bench.envs import benchmark_tasks
from risk_shift_bench.incumbent_switch import IncumbentSwitchParams, incumbent_switch_policy, search_incumbent_switch
from risk_shift_bench.multiseed import aggregate_seed_scores, multiseed_policies, paired_policy_deltas, summarize_seed
from risk_shift_bench.reporting import write_json


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


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def split_selection_tasks(selection_suite: str, validation_count: int) -> tuple[list, list]:
    tasks = benchmark_tasks(selection_suite)
    if selection_suite != "frontier_dev":
        if validation_count <= 0 or validation_count >= len(tasks):
            return [], tasks
        return tasks[:-validation_count], tasks[-validation_count:]
    if validation_count <= 0:
        raise ValueError("incumbent switch search requires a positive dev validation count")
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
    selected_rows = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(tasks=tasks, policies=policies, episodes=episodes, seed=seed, hand_depth=hand_depth)
        rows.extend(summarize_seed(seed, summaries))
    for task in tasks:
        selected_rows.append({"task": task.name, "selected_policy": selected_policy.selected_policy_name(task)})

    aggregate = aggregate_seed_scores(rows)
    paired = paired_policy_deltas(rows, reference_policy=selected_policy.name)
    split_dir = out_dir / split
    write_csv(split_dir / "seed_task_scores.csv", rows)
    write_csv(split_dir / "aggregate_scores.csv", aggregate)
    write_csv(split_dir / "paired_deltas.csv", paired)
    write_csv(split_dir / "selected_policies.csv", selected_rows)
    write_json(
        split_dir / "summary.json",
        {
            "split": split,
            "seeds": seeds,
            "episodes": episodes,
            "hand_depth": hand_depth,
            "tasks": [task.name for task in tasks],
            "selected_policies": selected_rows,
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
    parser.add_argument("--selection-suite", default="frontier_dev")
    parser.add_argument("--dev-validation-count", type=int, default=4)
    parser.add_argument("--eval-splits", default="frontier_holdout,frontier_audit")
    parser.add_argument("--selected-search-summary", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/incumbent_switch")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    selection_seeds = parse_seeds(args.selection_seeds)
    eval_seeds = parse_seeds(args.eval_seeds)
    if args.selected_search_summary:
        frozen_summary = read_json(args.selected_search_summary)
        selected_params = frozen_summary.get("selected_params")
        if not isinstance(selected_params, dict):
            raise ValueError("selected search summary must contain a selected_params object")
        params = IncumbentSwitchParams(**selected_params)
        validation_score = frozen_summary.get("validation_score")
        search_metadata = {
            "selected_search_summary": args.selected_search_summary,
            "selected_params": asdict(params),
            "validation_score": validation_score,
            "frozen": True,
        }
        write_json(out_dir / "search_summary.json", search_metadata)
    else:
        train_tasks, validation_tasks = split_selection_tasks(args.selection_suite, args.dev_validation_count)
        search_result = search_incumbent_switch(
            validation_tasks=validation_tasks,
            seeds=selection_seeds,
            episodes=args.episodes,
            hand_depth=args.hand_depth,
            smoke=args.smoke,
        )
        params = search_result.params
        validation_score = search_result.validation_score
        write_csv(out_dir / "candidate_scores.csv", search_result.candidate_scores)
        write_csv(out_dir / "dev_validation_scores.csv", search_result.validation_summaries)
        search_metadata = {
            "selection_seeds": selection_seeds,
            "episodes": args.episodes,
            "hand_depth": args.hand_depth,
            "selection_suite": args.selection_suite,
            "dev_train_tasks": [task.name for task in train_tasks],
            "dev_validation_tasks": [task.name for task in validation_tasks],
            "selected_params": asdict(search_result.params),
            "validation_score": search_result.validation_score,
            "candidate_scores": search_result.candidate_scores,
            "frozen": False,
        }
        write_json(out_dir / "search_summary.json", search_metadata)
    selected_policy = incumbent_switch_policy(params, name="validated_incumbent_switch")

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
            "selected_params": asdict(params),
            "validation_score": validation_score,
            "selected_search_summary": args.selected_search_summary,
            "protocol_summary": protocol_rows,
        },
    )

    print(f"selected_params={asdict(params)}")
    if isinstance(validation_score, (int, float)):
        print(f"validation_score={validation_score:.3f}")
    else:
        print(f"validation_score={validation_score}")
    for split in parse_splits(args.eval_splits):
        print(split)
        split_rows = [row for row in protocol_rows if row["split"] == split]
        for row in sorted(split_rows, key=lambda item: item["mean_score"], reverse=True)[:6]:
            print(f"  {row['policy']} | mean_score={row['mean_score']:.3f} std={row['std_score']:.3f}")


if __name__ == "__main__":
    main()
