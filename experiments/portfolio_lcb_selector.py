"""Train and evaluate portfolio lower-confidence selectors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from risk_shift_bench.adaptive_search import summary_score
from risk_shift_bench.multiseed import aggregate_seed_scores, paired_policy_deltas
from risk_shift_bench.portfolio_benchmark import portfolio_policy_lookup, run_portfolio_benchmark
from risk_shift_bench.portfolio_envs import portfolio_tasks
from risk_shift_bench.portfolio_lcb_selector import policy_from_scores, search_portfolio_lcb_selector
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


def load_scores(path: Path) -> dict[str, dict[str, list[float]]]:
    scores: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("scope") == "task":
                scores.setdefault(row["task"], {}).setdefault(row["policy"], []).append(float(row["mean_score"]))
    return scores


def merge_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        for task, policy_scores in load_scores(path).items():
            for policy, values in policy_scores.items():
                merged.setdefault(task, {}).setdefault(policy, []).extend(values)
    return {
        task: {policy: sum(values) / len(values) for policy, values in policy_scores.items()}
        for task, policy_scores in merged.items()
    }


def summarize_seed(seed: int, summaries) -> list[dict]:
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["seed"] = seed
        row["score"] = summary_score(summary)
        rows.append(row)
    return rows


def evaluate_fresh(split: str, policy, seeds: list[int], episodes: int, out_dir: Path):
    tasks = portfolio_tasks(split)
    lookup = portfolio_policy_lookup()
    eval_policies = [lookup["learned_mixture_searched"], lookup["signed_regime_learned_ensemble"], policy]
    rows = []
    selected = [{"task": task.name, "selected_policy": policy.selected_policy_name(task)} for task in tasks]
    for seed in seeds:
        _episodes, summaries = run_portfolio_benchmark(tasks=tasks, policies=eval_policies, episodes=episodes, seed=seed)
        rows.extend(summarize_seed(seed, summaries))
    aggregate = aggregate_seed_scores(rows)
    paired = paired_policy_deltas(rows, reference_policy=policy.name)
    split_dir = out_dir / split
    write_csv(split_dir / "seed_task_scores.csv", rows)
    write_csv(split_dir / "aggregate_scores.csv", aggregate)
    write_csv(split_dir / "paired_deltas.csv", paired)
    write_csv(split_dir / "selected_policies.csv", selected)
    write_json(split_dir / "summary.json", {"split": split, "selected_policies": selected, "aggregate_scores": aggregate, "paired_deltas": paired})
    return aggregate, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-suites", default="portfolio_dev,portfolio_holdout,portfolio_audit")
    parser.add_argument("--score-cache", action="append", required=True)
    parser.add_argument("--eval-split", default="portfolio_confirmation")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--fallback-policy", default="learned_mixture_searched")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/portfolio_lcb_selector")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_tasks = [task for suite in parse_suites(args.train_suites) for task in portfolio_tasks(suite)]
    scores_by_task = merge_scores([Path(path) for path in args.score_cache])
    search_result = search_portfolio_lcb_selector(
        tasks=train_tasks,
        scores_by_task=scores_by_task,
        fallback_policy=args.fallback_policy,
        smoke=args.smoke,
    )
    policy = policy_from_scores(train_tasks, scores_by_task, search_result.params)

    write_csv(out_dir / "candidate_scores.csv", search_result.candidate_scores)
    write_csv(out_dir / "validation_scores.csv", search_result.validation_summaries)
    write_csv(out_dir / "profiles.csv", search_result.train_profiles)
    aggregate, selected = evaluate_fresh(args.eval_split, policy, parse_seeds(args.eval_seeds), args.episodes, out_dir)
    write_json(
        out_dir / "summary.json",
        {
            "train_suites": parse_suites(args.train_suites),
            "score_cache": args.score_cache,
            "selected_params": asdict(search_result.params),
            "selection_score": search_result.selection_score,
            "eval_split": args.eval_split,
            "eval_seeds": parse_seeds(args.eval_seeds),
            "episodes": args.episodes,
            "selected_policies": selected,
            "protocol_summary": [row for row in aggregate if row["scope"] == "all_tasks"],
        },
    )
    print(f"selected_params={asdict(search_result.params)}")
    print(f"selection_score={search_result.selection_score:.3f}")
    for row in aggregate:
        if row["scope"] == "all_tasks":
            print(f"{args.eval_split} | {row['policy']} | mean_score={row['mean_score']:.3f}")


if __name__ == "__main__":
    main()
