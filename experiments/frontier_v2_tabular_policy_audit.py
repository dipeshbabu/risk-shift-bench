"""Inspect structural diversity in v2 tabular policy libraries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.frontier_v2_external_adapters import (
    _activate_verified_source,
    _gymnasium_policy_table,
)
from experiments.frontier_v2_external_design import DOMAIN_SPECS, domain_tasks
from experiments.frontier_v2_source_audit import SOURCE_DIRECTORIES


TABULAR_DOMAINS = tuple(
    domain
    for domain, spec in DOMAIN_SPECS.items()
    if spec.codebase == "gymnasium"
)


def compare_action_tables(
    tables: dict[str, dict[int, int]],
    *,
    fallback_policy: str,
) -> dict:
    if fallback_policy not in tables or len(tables) < 2:
        raise ValueError("action tables must contain a fallback and a candidate")
    state_sets = {frozenset(table) for table in tables.values()}
    if len(state_sets) != 1:
        raise ValueError("action tables do not cover the same state space")
    states = sorted(next(iter(state_sets)))
    if not states:
        raise ValueError("action tables must be nonempty")
    fallback = tables[fallback_policy]
    contrasts = {}
    for policy, table in tables.items():
        if policy == fallback_policy:
            continue
        disagreements = sum(table[state] != fallback[state] for state in states)
        contrasts[policy] = {
            "state_count": len(states),
            "action_disagreement_count": disagreements,
            "action_disagreement_fraction": disagreements / len(states),
        }
    return {
        "state_count": len(states),
        "candidate_contrasts": contrasts,
        "any_candidate_differs_from_fallback": any(
            contrast["action_disagreement_count"] > 0
            for contrast in contrasts.values()
        ),
    }


def audit_tabular_policy_library(split: str, source_root: Path) -> dict:
    if split not in {"development", "calibration"}:
        raise ValueError("tabular policy audit is restricted to outcome-eligible splits")
    _activate_verified_source(
        source_root / SOURCE_DIRECTORIES["gymnasium"],
        "gymnasium",
    )
    records = []
    for domain in TABULAR_DOMAINS:
        spec = DOMAIN_SPECS[domain]
        for task in domain_tasks(domain, split):
            policies = (spec.fallback_policy, *spec.candidate_policies)
            tables = {
                policy: _gymnasium_policy_table(task, policy)[0]
                for policy in policies
            }
            records.append(
                {
                    "domain": domain,
                    "task": task.name,
                    "split": split,
                    **compare_action_tables(
                        tables,
                        fallback_policy=spec.fallback_policy,
                    ),
                }
            )
    domains = {
        domain: {
            "task_count": sum(record["domain"] == domain for record in records),
            "tasks_with_structural_policy_contrast": sum(
                record["domain"] == domain
                and record["any_candidate_differs_from_fallback"]
                for record in records
            ),
        }
        for domain in TABULAR_DOMAINS
    }
    return {
        "design": "riskshiftbench-frontier-v2-tabular-policy-audit-v1",
        "scope": (
            "Development/calibration planning models only; no confirmation task or "
            "episode outcome is accessed."
        ),
        "split": split,
        "domain_count": len(domains),
        "task_count": len(records),
        "domains": domains,
        "all_tasks_have_structural_policy_contrast": all(
            record["any_candidate_differs_from_fallback"] for record in records
        ),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("development", "calibration"),
        required=True,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_tabular_policy_library(args.split, args.source_root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
