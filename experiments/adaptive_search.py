"""Search adaptive risk schedules and evaluate held-out tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from risk_preference_inference.adaptive_search import (
    evaluate_strong_baselines,
    policy_score_report,
    search_adaptive_policy,
    search_adaptive_utility_policy,
    search_learned_mixture_policy,
)
from risk_preference_inference.config import load_adaptive_search_config
from risk_preference_inference.envs import available_benchmark_tasks
from risk_preference_inference.learned_adaptive_search import search_learned_adaptive_policy
from risk_preference_inference.reporting import write_json


def select_tasks(names: tuple[str, ...]) -> list:
    all_tasks = {task.name: task for task in available_benchmark_tasks()}
    missing = set(names) - set(all_tasks)
    if missing:
        raise ValueError(f"Unknown tasks: {sorted(missing)}")
    return [all_tasks[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adaptive_search_smoke.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_adaptive_search_config(args.config)
    train_tasks = select_tasks(config.train_tasks)
    test_tasks = select_tasks(config.test_tasks)
    smoke = args.smoke or "smoke" in Path(args.config).stem

    result = search_adaptive_policy(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        episodes=config.episodes,
        seed=config.seed,
        hand_depth=config.hand_depth,
        smoke=smoke,
        max_candidates=config.max_candidates,
    )
    learned_result = search_learned_adaptive_policy(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        episodes=config.episodes,
        seed=config.seed + 250_000,
        hand_depth=config.hand_depth,
        smoke=smoke,
        max_candidates=config.max_candidates,
    )
    utility_result = search_adaptive_utility_policy(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        episodes=config.episodes,
        seed=config.seed + 375_000,
        hand_depth=config.hand_depth,
        smoke=smoke,
        max_candidates=config.max_candidates,
    )
    mixture_result = search_learned_mixture_policy(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        episodes=config.episodes,
        seed=config.seed + 425_000,
        hand_depth=config.hand_depth,
        smoke=smoke,
        max_candidates=config.max_candidates,
    )
    baseline_test = evaluate_strong_baselines(
        tasks=test_tasks,
        episodes=config.episodes,
        seed=config.seed + 500_000,
        hand_depth=config.hand_depth,
    )
    all_test_summaries = (
        baseline_test
        + result.test_summaries
        + learned_result.test_summaries
        + utility_result.test_summaries
        + mixture_result.test_summaries
    )

    payload = {
        "config": asdict(config),
        "best_adaptive": asdict(result),
        "best_learned_adaptive": asdict(learned_result),
        "best_adaptive_utility": asdict(utility_result),
        "best_learned_mixture": asdict(mixture_result),
        "baseline_test_summaries": baseline_test,
        "test_score_report": policy_score_report(all_test_summaries),
    }
    out_dir = args.out_dir or config.out_dir
    write_json(Path(out_dir) / "summary.json", payload)
    print(f"best_adaptive_train_score={result.train_score:.3f}")
    print(f"best_adaptive_test_score={result.test_score:.3f}")
    print(f"best_params={asdict(result.params)}")
    print(f"best_learned_test_score={learned_result.test_score:.3f}")
    print(f"best_learned_params={asdict(learned_result.params)}")
    print(f"best_adaptive_utility_test_score={utility_result.test_score:.3f}")
    print(f"best_adaptive_utility_params={asdict(utility_result.params)}")
    print(f"best_learned_mixture_test_score={mixture_result.test_score:.3f}")
    print(f"best_learned_mixture_params={asdict(mixture_result.params)}")


if __name__ == "__main__":
    main()
