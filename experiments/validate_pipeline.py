"""Validate paper experiment configuration and entry-point readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from risk_shift_bench.adaptive_search import candidate_params, mixture_candidate_params, utility_candidate_params
from risk_shift_bench.ablations import ablation_policies
from risk_shift_bench.config import load_adaptive_search_config, load_benchmark_config
from risk_shift_bench.envs import available_benchmark_tasks, benchmark_suite_names, benchmark_tasks
from risk_shift_bench.learned_adaptive_search import linear_candidates
from risk_shift_bench.policy_registry import core_policies, strong_baseline_grid


def validate_suite_name(suite: str) -> None:
    if suite not in benchmark_suite_names():
        raise ValueError(f"Unknown benchmark suite: {suite}")


def validate_task_names(names: tuple[str, ...] | None, suite: str | None = None) -> None:
    if names is None:
        return
    available_tasks = benchmark_tasks(suite) if suite is not None else available_benchmark_tasks()
    available = {task.name for task in available_tasks}
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
    validate_suite_name(benchmark_config.suite)
    validate_task_names(benchmark_config.tasks, suite=benchmark_config.suite)
    validate_task_names(adaptive_config.train_tasks)
    validate_task_names(adaptive_config.test_tasks)

    policy_count = len(core_policies()) if benchmark_config.policy_set == "core" else len(strong_baseline_grid())
    task_count = len(benchmark_config.tasks) if benchmark_config.tasks is not None else len(benchmark_tasks(benchmark_config.suite))
    benchmark_episodes = policy_count * task_count * benchmark_config.episodes
    ablation_rollouts = len(ablation_policies()) * task_count * benchmark_config.episodes

    adaptive_candidates = len(candidate_params(smoke=False))
    utility_candidates = len(utility_candidate_params(smoke=False))
    mixture_candidates = len(mixture_candidate_params(smoke=False))
    learned_candidates = len(linear_candidates(smoke=False))
    adaptive_evaluated = min(adaptive_candidates, adaptive_config.max_candidates or adaptive_candidates)
    utility_evaluated = min(utility_candidates, adaptive_config.max_candidates or utility_candidates)
    mixture_evaluated = min(mixture_candidates, adaptive_config.max_candidates or mixture_candidates)
    learned_evaluated = min(learned_candidates, adaptive_config.max_candidates or learned_candidates)

    print("pipeline validation ok")
    print(f"benchmark_suite={benchmark_config.suite}")
    print(f"benchmark_tasks={task_count}")
    print(f"benchmark_policies={policy_count}")
    print(f"benchmark_episode_rollouts={benchmark_episodes}")
    print(f"ablation_policies={len(ablation_policies())}")
    print(f"ablation_episode_rollouts={ablation_rollouts}")
    print(f"adaptive_candidates_total={adaptive_candidates}")
    print(f"adaptive_utility_candidates_total={utility_candidates}")
    print(f"learned_mixture_candidates_total={mixture_candidates}")
    print(f"learned_candidates_total={learned_candidates}")
    print(f"adaptive_candidates_evaluated={adaptive_evaluated}")
    print(f"adaptive_utility_candidates_evaluated={utility_evaluated}")
    print(f"learned_mixture_candidates_evaluated={mixture_evaluated}")
    print(f"learned_candidates_evaluated={learned_evaluated}")
    print(f"benchmark_out={benchmark_config.out_dir}")
    print(f"adaptive_out={adaptive_config.out_dir}")


if __name__ == "__main__":
    main()
