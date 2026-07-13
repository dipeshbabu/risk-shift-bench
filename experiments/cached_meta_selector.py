"""Evaluate meta-selector variants from cached selection scores."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from risk_preference_inference.benchmark import run_benchmark
from risk_preference_inference.envs import benchmark_tasks
from risk_preference_inference.meta_selector import (
    AdvantageKnnMetaPolicy,
    cross_validated_profile_score,
    meta_selector_candidate_params,
    profiles_from_scores,
    select_meta_search_result,
    MetaSelectorSearchResult,
)
from risk_preference_inference.multiseed import aggregate_seed_scores, paired_policy_deltas, summarize_seed
from risk_preference_inference.policy_registry import signed_regime_learned_policy
from risk_preference_inference.reporting import write_json


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


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            grouped.setdefault((row["task"], row["policy"]), []).append(float(row["score"]))
    scores: dict[str, dict[str, float]] = {}
    for (task, policy), values in grouped.items():
        scores.setdefault(task, {})[policy] = sum(values) / len(values)
    return scores


def select_params(tasks, scores_by_task, smoke: bool = False):
    results = []
    candidate_scores = []
    for index, params in enumerate(meta_selector_candidate_params(smoke=smoke)):
        profiles = profiles_from_scores(tasks=tasks, scores_by_task=scores_by_task, params=params)
        validation_score, validation_rows = cross_validated_profile_score(tasks=tasks, profiles=profiles, params=params)
        candidate_row = {"candidate": index, "validation_score": validation_score, **asdict(params)}
        candidate_scores.append(candidate_row)
        result = MetaSelectorSearchResult(
            params=params,
            validation_score=validation_score,
            train_profiles=[asdict(profile) for profile in profiles],
            validation_summaries=validation_rows,
            candidate_scores=list(candidate_scores),
        )
        results.append((index, result))
    best = select_meta_search_result(results)
    profiles = profiles_from_scores(tasks=tasks, scores_by_task=scores_by_task, params=best.params)
    return (best.validation_score, best.params, profiles, best.validation_summaries), candidate_scores


def evaluate_split(split: str, selector_policy, seeds: list[int], episodes: int, hand_depth: int, out_dir: Path):
    tasks = benchmark_tasks(split)
    signed_policy = signed_regime_learned_policy()
    rows = []
    selected_rows = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(
            tasks=tasks,
            policies=[signed_policy, selector_policy],
            episodes=episodes,
            seed=seed,
            hand_depth=hand_depth,
        )
        rows.extend(summarize_seed(seed, summaries))
    for task in tasks:
        selected_rows.append({"task": task.name, "selected_policy": selector_policy.selected_policy_name(task)})

    aggregate = aggregate_seed_scores(rows)
    paired = paired_policy_deltas(rows, reference_policy=selector_policy.name)
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
    return aggregate


def evaluate_split_from_cache(split: str, selector_policy, score_cache: Path, out_dir: Path):
    tasks = benchmark_tasks(split)
    task_scores: dict[str, dict[str, float]] = {}
    with score_cache.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] == "task":
                task_scores.setdefault(row["task"], {})[row["policy"]] = float(row["mean_score"])

    rows = []
    selected_rows = []
    for task in tasks:
        selected_policy = selector_policy.selected_policy_name(task)
        selector_score = task_scores[task.name][selected_policy]
        signed_score = task_scores[task.name]["signed_regime_learned_ensemble"]
        selected_rows.append(
            {
                "task": task.name,
                "selected_policy": selected_policy,
                "cached_mean_score": selector_score,
                "signed_score": signed_score,
                "delta_vs_signed": selector_score - signed_score,
            }
        )
        rows.append(
            {
                "scope": "task",
                "task": task.name,
                "policy": selector_policy.name,
                "n": 1,
                "mean_score": selector_score,
                "std_score": 0.0,
            }
        )
        rows.append(
            {
                "scope": "task",
                "task": task.name,
                "policy": "signed_regime_learned_ensemble",
                "n": 1,
                "mean_score": signed_score,
                "std_score": 0.0,
            }
        )

    selector_scores = [row["cached_mean_score"] for row in selected_rows]
    signed_scores = [row["signed_score"] for row in selected_rows]
    aggregate = [
        {
            "scope": "all_tasks",
            "task": "ALL",
            "policy": selector_policy.name,
            "n": len(selector_scores),
            "mean_score": sum(selector_scores) / len(selector_scores),
            "std_score": 0.0,
        },
        {
            "scope": "all_tasks",
            "task": "ALL",
            "policy": "signed_regime_learned_ensemble",
            "n": len(signed_scores),
            "mean_score": sum(signed_scores) / len(signed_scores),
            "std_score": 0.0,
        },
        *rows,
    ]
    split_dir = out_dir / split
    write_csv(split_dir / "aggregate_scores.csv", aggregate)
    write_csv(split_dir / "selected_policies.csv", selected_rows)
    write_json(
        split_dir / "summary.json",
        {
            "split": split,
            "score_cache": str(score_cache),
            "cached": True,
            "tasks": [task.name for task in tasks],
            "selected_policies": selected_rows,
            "aggregate_scores": aggregate,
            "note": "Cached deterministic evaluation from prior per-task policy scores; not a fresh simulator rerun.",
        },
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-score-cache", required=True)
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--hand-depth", type=int, default=1)
    parser.add_argument("--train-suite", default="frontier_dev")
    parser.add_argument("--validation-suite", default="frontier_holdout")
    parser.add_argument("--eval-splits", default="frontier_confirmation_audit")
    parser.add_argument("--eval-score-cache")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/cached_meta_selector")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_tasks = benchmark_tasks(args.train_suite)
    validation_tasks = benchmark_tasks(args.validation_suite)
    final_train_tasks = [*train_tasks, *validation_tasks]
    scores_by_task = load_scores(Path(args.selection_score_cache))
    missing = sorted(task.name for task in final_train_tasks if task.name not in scores_by_task)
    if missing:
        raise ValueError(f"selection cache is missing tasks: {missing}")

    (validation_score, params, profiles, validation_rows), candidate_scores = select_params(
        tasks=final_train_tasks,
        scores_by_task=scores_by_task,
        smoke=args.smoke,
    )
    selector_policy = AdvantageKnnMetaPolicy(
        profiles=profiles,
        params=params,
        name="regret_guard_meta_selector",
    )

    write_csv(out_dir / "candidate_scores.csv", candidate_scores)
    write_csv(out_dir / "validation_scores.csv", validation_rows)
    write_csv(out_dir / "meta_profiles.csv", [asdict(profile) for profile in profiles])
    write_json(
        out_dir / "search_summary.json",
        {
            "selection_score_cache": args.selection_score_cache,
            "episodes": args.episodes,
            "hand_depth": args.hand_depth,
            "train_suite": args.train_suite,
            "validation_suite": args.validation_suite,
            "selected_params": asdict(params),
            "validation_score": validation_score,
            "candidate_scores": candidate_scores,
        },
    )

    protocol_rows = []
    for split in parse_splits(args.eval_splits):
        if args.eval_score_cache:
            aggregate = evaluate_split_from_cache(
                split=split,
                selector_policy=selector_policy,
                score_cache=Path(args.eval_score_cache),
                out_dir=out_dir,
            )
        else:
            aggregate = evaluate_split(
                split=split,
                selector_policy=selector_policy,
                seeds=parse_seeds(args.eval_seeds),
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
            "eval_seeds": parse_seeds(args.eval_seeds),
            "episodes": args.episodes,
            "hand_depth": args.hand_depth,
            "eval_score_cache": args.eval_score_cache,
            "selected_params": asdict(params),
            "validation_score": validation_score,
            "protocol_summary": protocol_rows,
        },
    )

    print(f"selected_params={asdict(params)}")
    print(f"validation_score={validation_score:.3f}")
    for row in sorted(protocol_rows, key=lambda item: item["mean_score"], reverse=True):
        print(f"{row['split']} | {row['policy']} | mean_score={row['mean_score']:.3f} std={row['std_score']:.3f}")


if __name__ == "__main__":
    main()
