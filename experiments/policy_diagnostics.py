"""Export policy action maps and adaptive-risk schedules."""

from __future__ import annotations

import argparse
from pathlib import Path

from risk_preference_inference.diagnostics import action_map, adaptive_alpha_curve, rows_as_dicts
from risk_preference_inference.envs import benchmark_tasks
from risk_preference_inference.policy_registry import adaptive_cvar_policy, core_policies
from risk_preference_inference.reporting import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="RiskBlackjack-RuinConstraint-v0")
    parser.add_argument("--out-dir", default="artifacts/policy_diagnostics")
    parser.add_argument("--hand-depth", type=int, default=1)
    args = parser.parse_args()

    tasks = {task.name: task for task in benchmark_tasks()}
    if args.task not in tasks:
        raise ValueError(f"Unknown task: {args.task}")
    task = tasks[args.task]
    policies = core_policies()
    adaptive = adaptive_cvar_policy(name="adaptive_cvar")
    maps = []
    for policy in policies:
        maps.extend(rows_as_dicts(action_map(policy, task, hand_depth=args.hand_depth)))
    payload = {
        "task": task.name,
        "action_map": maps,
        "adaptive_alpha": rows_as_dicts(adaptive_alpha_curve(adaptive, task)),
    }
    write_json(Path(args.out_dir) / "diagnostics.json", payload)
    print(f"wrote {Path(args.out_dir) / 'diagnostics.json'}")


if __name__ == "__main__":
    main()

