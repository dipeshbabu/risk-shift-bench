"""Compute exact small-horizon final-bankroll distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

from risk_preference_inference.envs import benchmark_tasks
from risk_preference_inference.multiround_distributions import final_bankroll_distribution
from risk_preference_inference.objectives import cvar_lower, mean, probability_at_or_above, probability_at_or_below
from risk_preference_inference.policy_registry import core_policies
from risk_preference_inference.reporting import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="RiskBlackjack-Mean-v0")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--hand-depth", type=int, default=1)
    parser.add_argument("--out-dir", default="artifacts/multiround_exact")
    args = parser.parse_args()

    tasks = {task.name: task for task in benchmark_tasks()}
    if args.task not in tasks:
        raise ValueError(f"Unknown task: {args.task}")
    task = tasks[args.task]
    rows = []
    for policy in core_policies():
        distribution = final_bankroll_distribution(task, policy, rounds=args.rounds, hand_depth=args.hand_depth, grid=task.bet)
        rows.append(
            {
                "task": task.name,
                "policy": policy.name,
                "rounds": args.rounds,
                "mean_final_bankroll": mean(distribution),
                "cvar_5_final_bankroll": cvar_lower(distribution, 0.05),
                "ruin_probability": probability_at_or_below(distribution, task.ruin_bankroll),
                "target_probability": probability_at_or_above(distribution, task.target_bankroll),
                "support_size": len(distribution),
                "distribution": distribution,
            }
        )
    write_json(Path(args.out_dir) / "summary.json", rows)
    print(f"wrote {Path(args.out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()

