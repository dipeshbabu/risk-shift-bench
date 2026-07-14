"""Train and evaluate cached task-family promotion selectors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from risk_shift_bench.envs import benchmark_tasks
from risk_shift_bench.family_selector import (
    FamilyPromotionPolicy,
    learn_family_promotions,
    task_family,
)
from risk_shift_bench.benchmark import run_benchmark
from risk_shift_bench.multiseed import aggregate_seed_scores, paired_policy_deltas, summarize_seed
from risk_shift_bench.family_selector import family_candidate_lookup
from risk_shift_bench.policy_registry import signed_regime_learned_policy
from risk_shift_bench.reporting import write_json


def parse_suites(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def load_selection_scores(path: Path) -> dict[str, dict[str, list[float]]]:
    scores: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            scores.setdefault(row["task"], {}).setdefault(row["policy"], []).append(float(row["score"]))
    return scores


def load_aggregate_scores(path: Path) -> dict[str, dict[str, list[float]]]:
    scores: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] == "task":
                scores.setdefault(row["task"], {}).setdefault(row["policy"], []).append(float(row["mean_score"]))
    return scores


def merge_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        loaded = load_selection_scores(path) if path.name == "selection_train_scores.csv" else load_aggregate_scores(path)
        for task, policy_scores in loaded.items():
            for policy, values in policy_scores.items():
                merged.setdefault(task, {}).setdefault(policy, []).extend(values)
    return {
        task: {policy: sum(values) / len(values) for policy, values in policy_scores.items()}
        for task, policy_scores in merged.items()
    }


def evaluate_from_cache(split: str, policy: FamilyPromotionPolicy, score_cache: Path, out_dir: Path):
    task_scores: dict[str, dict[str, float]] = {}
    with score_cache.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] == "task":
                task_scores.setdefault(row["task"], {})[row["policy"]] = float(row["mean_score"])

    rows = []
    selected = []
    selector_scores = []
    fallback_scores = []
    for task in benchmark_tasks(split):
        family = task_family(task)
        selected_policy = policy.selected_policy_name(task)
        if selected_policy not in task_scores[task.name]:
            selected_policy = policy.params.fallback_policy
        selector_score = task_scores[task.name][selected_policy]
        fallback_score = task_scores[task.name][policy.params.fallback_policy]
        selector_scores.append(selector_score)
        fallback_scores.append(fallback_score)
        selected.append(
            {
                "task": task.name,
                "family": family,
                "selected_policy": selected_policy,
                "cached_mean_score": selector_score,
                "fallback_score": fallback_score,
                "delta_vs_fallback": selector_score - fallback_score,
            }
        )
        rows.append(
            {
                "scope": "task",
                "task": task.name,
                "policy": policy.name,
                "n": 1,
                "mean_score": selector_score,
                "std_score": 0.0,
            }
        )
        rows.append(
            {
                "scope": "task",
                "task": task.name,
                "policy": policy.params.fallback_policy,
                "n": 1,
                "mean_score": fallback_score,
                "std_score": 0.0,
            }
        )

    aggregate = [
        {
            "scope": "all_tasks",
            "task": "ALL",
            "policy": policy.name,
            "n": len(selector_scores),
            "mean_score": sum(selector_scores) / len(selector_scores),
            "std_score": 0.0,
        },
        {
            "scope": "all_tasks",
            "task": "ALL",
            "policy": policy.params.fallback_policy,
            "n": len(fallback_scores),
            "mean_score": sum(fallback_scores) / len(fallback_scores),
            "std_score": 0.0,
        },
        *rows,
    ]
    split_dir = out_dir / split
    write_csv(split_dir / "aggregate_scores.csv", aggregate)
    write_csv(split_dir / "selected_policies.csv", selected)
    write_json(
        split_dir / "summary.json",
        {
            "split": split,
            "score_cache": str(score_cache),
            "cached": True,
            "selected_policies": selected,
            "aggregate_scores": aggregate,
            "note": "Cached deterministic evaluation from prior per-task policy scores; not a fresh simulator rerun.",
        },
    )
    return aggregate, selected


def evaluate_fresh(
    split: str,
    policy: FamilyPromotionPolicy,
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    out_dir: Path,
):
    tasks = benchmark_tasks(split)
    signed_policy = signed_regime_learned_policy()
    policies = [signed_policy, policy]
    if policy.params.fallback_policy != signed_policy.name:
        fallback_lookup = family_candidate_lookup()
        if policy.params.fallback_policy not in fallback_lookup:
            raise ValueError(f"unknown fallback policy: {policy.params.fallback_policy}")
        policies.append(fallback_lookup[policy.params.fallback_policy])
    rows = []
    selected = [
        {
            "task": task.name,
            "family": task_family(task),
            "selected_policy": policy.selected_policy_name(task),
        }
        for task in tasks
    ]
    for seed in seeds:
        _episodes, summaries = run_benchmark(
            tasks=tasks,
            policies=policies,
            episodes=episodes,
            seed=seed,
            hand_depth=hand_depth,
        )
        rows.extend(summarize_seed(seed, summaries))

    aggregate = aggregate_seed_scores(rows)
    paired = paired_policy_deltas(rows, reference_policy=policy.name)
    split_dir = out_dir / split
    write_csv(split_dir / "seed_task_scores.csv", rows)
    write_csv(split_dir / "aggregate_scores.csv", aggregate)
    write_csv(split_dir / "paired_deltas.csv", paired)
    write_csv(split_dir / "selected_policies.csv", selected)
    write_json(
        split_dir / "summary.json",
        {
            "split": split,
            "cached": False,
            "seeds": seeds,
            "episodes": episodes,
            "hand_depth": hand_depth,
            "selected_policies": selected,
            "aggregate_scores": aggregate,
            "paired_deltas": paired,
        },
    )
    return aggregate, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-suites", default="frontier_dev,frontier_holdout,frontier_audit,frontier_final_audit,frontier_blind_audit")
    parser.add_argument("--score-cache", action="append", required=True)
    parser.add_argument("--eval-split", default="frontier_confirmation_audit")
    parser.add_argument("--eval-score-cache")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--hand-depth", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=2.0)
    parser.add_argument("--min-sparse-evidence", type=int, default=2)
    parser.add_argument("--fallback-policy", default="signed_regime_learned_ensemble")
    parser.add_argument("--allow-negative-sparse-evidence", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/family_selector_cached")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_tasks = [task for suite in parse_suites(args.train_suites) for task in benchmark_tasks(suite)]
    scores_by_task = merge_scores([Path(path) for path in args.score_cache])
    params = learn_family_promotions(
        tasks=train_tasks,
        scores_by_task=scores_by_task,
        fallback_policy=args.fallback_policy,
        min_delta=args.min_delta,
        min_sparse_evidence=args.min_sparse_evidence,
        require_nonnegative_sparse_evidence=not args.allow_negative_sparse_evidence,
    )
    policy = FamilyPromotionPolicy(params=params)

    family_rows = []
    for task in train_tasks:
        if task.name not in scores_by_task or params.fallback_policy not in scores_by_task[task.name]:
            continue
        family = task_family(task)
        selected_policy = params.family_delegates.get(family, params.fallback_policy)
        if selected_policy not in scores_by_task[task.name]:
            selected_policy = params.fallback_policy
        family_rows.append(
            {
                "task": task.name,
                "family": family,
                "selected_policy": selected_policy,
                "train_score": scores_by_task[task.name][selected_policy],
                "fallback_score": scores_by_task[task.name][params.fallback_policy],
                "delta_vs_fallback": scores_by_task[task.name][selected_policy] - scores_by_task[task.name][params.fallback_policy],
            }
        )

    if args.eval_score_cache:
        aggregate, selected = evaluate_from_cache(
            split=args.eval_split,
            policy=policy,
            score_cache=Path(args.eval_score_cache),
            out_dir=out_dir,
        )
    else:
        aggregate, selected = evaluate_fresh(
            split=args.eval_split,
            policy=policy,
            seeds=parse_seeds(args.eval_seeds),
            episodes=args.episodes,
            hand_depth=args.hand_depth,
            out_dir=out_dir,
        )
    write_csv(out_dir / "train_family_scores.csv", family_rows)
    write_json(
        out_dir / "summary.json",
        {
            "train_suites": parse_suites(args.train_suites),
            "score_cache": args.score_cache,
            "eval_split": args.eval_split,
            "eval_score_cache": args.eval_score_cache,
            "eval_seeds": parse_seeds(args.eval_seeds),
            "episodes": args.episodes,
            "hand_depth": args.hand_depth,
            "min_sparse_evidence": args.min_sparse_evidence,
            "fallback_policy": args.fallback_policy,
            "allow_negative_sparse_evidence": args.allow_negative_sparse_evidence,
            "params": asdict(params),
            "train_family_scores": family_rows,
            "selected_policies": selected,
            "protocol_summary": [row for row in aggregate if row["scope"] == "all_tasks"],
        },
    )

    print(f"params={asdict(params)}")
    for row in aggregate:
        if row["scope"] == "all_tasks":
            print(f"{args.eval_split} | {row['policy']} | mean_score={row['mean_score']:.3f}")


if __name__ == "__main__":
    main()
