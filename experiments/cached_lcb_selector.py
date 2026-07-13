"""Train and evaluate lower-confidence selectors from cached policy scores."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from risk_preference_inference.benchmark import run_benchmark
from risk_preference_inference.envs import benchmark_tasks
from risk_preference_inference.family_selector import family_candidate_lookup
from risk_preference_inference.lcb_selector import policy_from_scores, search_lcb_selector
from risk_preference_inference.multiseed import aggregate_seed_scores, paired_policy_deltas, summarize_seed
from risk_preference_inference.policy_registry import signed_regime_learned_policy
from risk_preference_inference.reporting import write_json


def parse_suites(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_policy_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


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


def evaluate_from_cache(split: str, policy, score_cache: Path, out_dir: Path):
    task_scores: dict[str, dict[str, float]] = {}
    with score_cache.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] == "task":
                task_scores.setdefault(row["task"], {})[row["policy"]] = float(row["mean_score"])

    selected = []
    rows = []
    selector_scores = []
    fallback_scores = []
    for task in benchmark_tasks(split):
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
                "selected_policy": selected_policy,
                "cached_mean_score": selector_score,
                "fallback_score": fallback_score,
                "delta_vs_fallback": selector_score - fallback_score,
            }
        )
        rows.extend(
            [
                {
                    "scope": "task",
                    "task": task.name,
                    "policy": policy.name,
                    "n": 1,
                    "mean_score": selector_score,
                    "std_score": 0.0,
                },
                {
                    "scope": "task",
                    "task": task.name,
                    "policy": policy.params.fallback_policy,
                    "n": 1,
                    "mean_score": fallback_score,
                    "std_score": 0.0,
                },
            ]
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
            "cached": True,
            "score_cache": str(score_cache),
            "selected_policies": selected,
            "aggregate_scores": aggregate,
        },
    )
    return aggregate, selected


def evaluate_fresh(
    split: str,
    policy,
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    out_dir: Path,
    include_delegate_policies: bool = False,
    extra_policy_names: list[str] | None = None,
):
    tasks = benchmark_tasks(split)
    rows = []
    selected = [{"task": task.name, "selected_policy": policy.selected_policy_name(task)} for task in tasks]
    delegate_lookup = family_candidate_lookup()
    eval_policies = [signed_regime_learned_policy(), policy]
    seen = {eval_policy.name for eval_policy in eval_policies}
    fallback_policy = getattr(policy.params, "fallback_policy", "signed_regime_learned_ensemble")
    if fallback_policy not in seen:
        if fallback_policy not in delegate_lookup:
            raise ValueError(f"unknown fallback policy: {fallback_policy}")
        eval_policies.append(delegate_lookup[fallback_policy])
        seen.add(fallback_policy)
    if include_delegate_policies:
        for candidate in delegate_lookup.values():
            if candidate.name not in seen:
                eval_policies.append(candidate)
                seen.add(candidate.name)
    for policy_name in extra_policy_names or []:
        if policy_name not in delegate_lookup:
            raise ValueError(f"unknown extra policy: {policy_name}")
        if policy_name not in seen:
            eval_policies.append(delegate_lookup[policy_name])
            seen.add(policy_name)
    for seed in seeds:
        _episodes, summaries = run_benchmark(
            tasks=tasks,
            policies=eval_policies,
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
    parser.add_argument("--robust-selection", action="store_true")
    parser.add_argument("--fallback-policy", default="signed_regime_learned_ensemble")
    parser.add_argument("--comparison-policies")
    parser.add_argument("--promotion-loss-weight", type=float, default=1.0)
    parser.add_argument("--worst-loss-weight", type=float, default=0.25)
    parser.add_argument("--include-delegate-policies", action="store_true")
    parser.add_argument("--extra-policy-names")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/lcb_selector_cached")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_tasks = [task for suite in parse_suites(args.train_suites) for task in benchmark_tasks(suite)]
    scores_by_task = merge_scores([Path(path) for path in args.score_cache])
    search_result = search_lcb_selector(
        train_tasks,
        scores_by_task,
        smoke=args.smoke,
        robust_selection=args.robust_selection,
        promotion_loss_weight=args.promotion_loss_weight,
        worst_loss_weight=args.worst_loss_weight,
        fallback_policy=args.fallback_policy,
        comparison_policies=tuple(parse_policy_names(args.comparison_policies)),
    )
    policy = policy_from_scores(
        tasks=train_tasks,
        scores_by_task=scores_by_task,
        params=search_result.params,
        name="lower_confidence_selector",
    )

    write_csv(out_dir / "candidate_scores.csv", search_result.candidate_scores)
    write_csv(out_dir / "validation_scores.csv", search_result.validation_summaries)
    write_csv(out_dir / "profiles.csv", search_result.train_profiles)
    write_json(
        out_dir / "search_summary.json",
        {
            "train_suites": parse_suites(args.train_suites),
            "score_cache": args.score_cache,
            "robust_selection": args.robust_selection,
            "fallback_policy": args.fallback_policy,
            "comparison_policies": parse_policy_names(args.comparison_policies),
            "promotion_loss_weight": args.promotion_loss_weight,
            "worst_loss_weight": args.worst_loss_weight,
            "selected_params": asdict(search_result.params),
            "selection_score": search_result.validation_score,
            "candidate_scores": search_result.candidate_scores,
        },
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
            include_delegate_policies=args.include_delegate_policies,
            extra_policy_names=parse_policy_names(args.extra_policy_names),
        )
    write_json(
        out_dir / "summary.json",
        {
            "selected_params": asdict(search_result.params),
            "selection_score": search_result.validation_score,
            "robust_selection": args.robust_selection,
            "fallback_policy": args.fallback_policy,
            "comparison_policies": parse_policy_names(args.comparison_policies),
            "promotion_loss_weight": args.promotion_loss_weight,
            "worst_loss_weight": args.worst_loss_weight,
            "eval_split": args.eval_split,
            "eval_score_cache": args.eval_score_cache,
            "eval_seeds": parse_seeds(args.eval_seeds),
            "episodes": args.episodes,
            "hand_depth": args.hand_depth,
            "include_delegate_policies": args.include_delegate_policies,
            "extra_policy_names": parse_policy_names(args.extra_policy_names),
            "selected_policies": selected,
            "protocol_summary": [row for row in aggregate if row["scope"] == "all_tasks"],
        },
    )
    print(f"selected_params={asdict(search_result.params)}")
    print(f"selection_score={search_result.validation_score:.3f}")
    for row in aggregate:
        if row["scope"] == "all_tasks":
            print(f"{args.eval_split} | {row['policy']} | mean_score={row['mean_score']:.3f}")


if __name__ == "__main__":
    main()
