"""Validate paper experiment configuration and entry-point readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from risk_preference_inference.adaptive_search import candidate_params
from risk_preference_inference.config import load_adaptive_search_config, load_benchmark_config
from risk_preference_inference.envs import benchmark_tasks
from risk_preference_inference.learned_adaptive_search import linear_candidates
from risk_preference_inference.policy_registry import core_policies, strong_baseline_grid


def validate_task_names(names: tuple[str, ...] | None) -> None:
    if names is None:
        return
    available = {task.name for task in benchmark_tasks()}
    missing = set(names) - available
    if missing:
        raise ValueError(f"Unknown task names: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default="configs/benchmark_full.json")
    parser.add_argument("--adaptive-config", default="configs/adaptive_search_full.json")
    args = parser.parse_args()

    benchmark_config = load_benchmark_config(args.benchmark_config)
    adaptive_config = load_adaptive_search_config(args.adaptive_config)
    validate_task_names(benchmark_config.tasks)
    validate_task_names(adaptive_config.train_tasks)
    validate_task_names(adaptive_config.test_tasks)

    policy_count = len(core_policies()) if benchmark_config.policy_set == "core" else len(strong_baseline_grid())
    task_count = len(benchmark_config.tasks) if benchmark_config.tasks is not None else len(benchmark_tasks())
    benchmark_episodes = policy_count * task_count * benchmark_config.episodes

    adaptive_candidates = len(candidate_params(smoke=False))
    learned_candidates = len(linear_candidates(smoke=False))
    adaptive_evaluated = min(adaptive_candidates, adaptive_config.max_candidates or adaptive_candidates)
    learned_evaluated = min(learned_candidates, adaptive_config.max_candidates or learned_candidates)

    print("pipeline validation ok")
    print(f"benchmark_tasks={task_count}")
    print(f"benchmark_policies={policy_count}")
    print(f"benchmark_episode_rollouts={benchmark_episodes}")
    print(f"adaptive_candidates_total={adaptive_candidates}")
    print(f"learned_candidates_total={learned_candidates}")
    print(f"adaptive_candidates_evaluated={adaptive_evaluated}")
    print(f"learned_candidates_evaluated={learned_evaluated}")
    print(f"benchmark_out={benchmark_config.out_dir}")
    print(f"adaptive_out={adaptive_config.out_dir}")


if __name__ == "__main__":
    main()

