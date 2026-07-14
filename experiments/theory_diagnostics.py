"""Export horizon reversal diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from risk_shift_bench.blackjack import DecisionState
from risk_shift_bench.envs import benchmark_tasks
from risk_shift_bench.policy_registry import core_policies
from risk_shift_bench.reporting import write_json
from risk_shift_bench.theory import horizon_action_table, horizon_reversals, rows_as_dicts


def diagnostic_states(task_name: str) -> list[DecisionState]:
    return [
        DecisionState((10, 6), 10, current_bankroll=240.0, initial_bankroll=500.0, target_bankroll=650.0),
        DecisionState((10, 2), 6, current_bankroll=500.0, initial_bankroll=500.0, target_bankroll=650.0),
        DecisionState((11, 7), 9, current_bankroll=620.0, initial_bankroll=500.0, target_bankroll=650.0),
        DecisionState((9, 7), 10, current_bankroll=420.0, initial_bankroll=500.0, target_bankroll=650.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="RiskBlackjack-RuinConstraint-v0")
    parser.add_argument("--out-dir", default="artifacts/theory_diagnostics")
    parser.add_argument("--hand-depth", type=int, default=1)
    args = parser.parse_args()

    tasks = {task.name: task for task in benchmark_tasks()}
    task = tasks[args.task]
    action_rows = []
    reversal_rows = []
    for policy in core_policies():
        rows = horizon_action_table(task, policy, diagnostic_states(task.name), hand_depth=args.hand_depth)
        action_rows.extend(rows)
        reversal_rows.extend(horizon_reversals(rows, 1, 10))
    payload = {
        "task": task.name,
        "horizon_actions": rows_as_dicts(action_rows),
        "reversals": rows_as_dicts(reversal_rows),
    }
    write_json(Path(args.out_dir) / "diagnostics.json", payload)
    print(f"wrote {Path(args.out_dir) / 'diagnostics.json'}")


if __name__ == "__main__":
    main()

