"""Build a task-policy matrix and run the RPOSST-inspired subset baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from experiments.frontier_v2_external_design import DOMAIN_SPECS
from experiments.robust_test_subset_baseline import select_robust_test_subset


def task_policy_matrix(payloads: list[dict]) -> tuple[str, dict[str, dict[str, float]]]:
    if not payloads:
        raise ValueError("at least one development payload is required")
    domain = payloads[0].get("domain")
    if domain not in DOMAIN_SPECS:
        raise RuntimeError(f"unknown domain: {domain}")
    spec = DOMAIN_SPECS[domain]
    expected_policies = {spec.fallback_policy, *spec.candidate_policies}
    matrix = {}
    for payload in payloads:
        if payload.get("domain") != domain:
            raise RuntimeError("task-policy matrix cannot mix domains")
        if payload.get("split") not in {"development", "calibration"}:
            raise RuntimeError("matrix payload is not outcome-eligible")
        task = payload.get("task")
        if task in matrix:
            raise RuntimeError(f"duplicate task payload: {task}")
        outcomes = payload.get("outcomes", {})
        if set(outcomes) != expected_policies:
            raise RuntimeError(f"policy library is incomplete for task {task}")
        matrix[task] = {
            policy: fmean(float(row["score"]) for row in rows)
            for policy, rows in outcomes.items()
        }
    return domain, matrix


def compare_task_subset(paths: tuple[Path, ...], *, subset_size: int) -> dict:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    domain, matrix = task_policy_matrix(payloads)
    result = select_robust_test_subset(matrix, subset_size=subset_size)
    return {
        "design": "riskshiftbench-frontier-v2-rposst-inspired-subset-comparison",
        "scope": "Development/calibration task scores only; no confirmation artifact is read.",
        "domain": domain,
        "task_count": len(matrix),
        "policy_count": len(next(iter(matrix.values()))),
        "subset_size": subset_size,
        "result_matrix": matrix,
        "subset_result": asdict(result),
        "guarantee": (
            "Deterministic minimax approximation baseline only; this implementation "
            "does not inherit RPOSST's k-of-N theorem."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--subset-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare_task_subset(tuple(args.paths), subset_size=args.subset_size)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
