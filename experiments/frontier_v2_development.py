"""Run one outcome-eligible v2 development/calibration task and policy library."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from experiments.frontier_v2_external_adapters import (
    outcome_rows,
    run_v2_development_task,
    summarize_v2_outcomes,
)
from experiments.frontier_v2_external_design import (
    CODEBASE_LOCKS,
    DOMAIN_SPECS,
    V2ExternalTask,
    all_tasks,
    canonical_episode_seed_base,
    domain_tasks,
    task_manifest_sha256,
    task_sha256,
)
from experiments.frontier_v2_source_audit import (
    SOURCE_DIRECTORIES,
    audit_codebase_source,
)


SUPPORTED_DOMAINS = tuple(DOMAIN_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=SUPPORTED_DOMAINS, required=True)
    parser.add_argument("--split", choices=("development", "calibration"), default="development")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--policy", default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run_development_task_payload(
    task: V2ExternalTask,
    *,
    policy: str = "all",
    episodes: int = 20,
    seed_base: int | None = None,
    verify_determinism: bool = False,
    source_root: Path = Path("artifacts/frontier_v2_sources"),
) -> dict:
    if task.split not in {"development", "calibration"}:
        raise RuntimeError("only development and calibration tasks are outcome-eligible")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    library = DOMAIN_SPECS[task.domain]
    policies = (
        (library.fallback_policy, *library.candidate_policies)
        if policy == "all"
        else (policy,)
    )
    allowed_policies = {
        library.fallback_policy,
        *library.candidate_policies,
    }
    if not set(policies) <= allowed_policies:
        raise KeyError(policy)
    canonical_seed_base = canonical_episode_seed_base(task)
    selected_seed_base = canonical_seed_base if seed_base is None else seed_base
    source_audit = audit_codebase_source(
        source_root / SOURCE_DIRECTORIES[library.codebase],
        library.codebase,
    )
    source_audit_payload = asdict(source_audit)
    source_audit_payload["source"] = SOURCE_DIRECTORIES[library.codebase]

    collection_started = perf_counter()
    outcomes = {
        selected_policy: run_v2_development_task(
            task,
            selected_policy,
            episodes,
            selected_seed_base,
            source_root,
        )
        for selected_policy in policies
    }
    collection_runtime = perf_counter() - collection_started

    verification_runtime = 0.0
    if verify_determinism:
        verification_started = perf_counter()
        repeated = {
            selected_policy: run_v2_development_task(
                task,
                selected_policy,
                episodes,
                selected_seed_base,
                source_root,
            )
            for selected_policy in policies
        }
        verification_runtime = perf_counter() - verification_started
        if repeated != outcomes:
            raise RuntimeError("determinism verification failed for the task policy library")
    return {
        "design": "riskshiftbench-frontier-v2-development-task-v1",
        "scope": "Development/calibration only; confirmation execution is prohibited.",
        "task": task.name,
        "domain": task.domain,
        "split": task.split,
        "task_definition": asdict(task),
        "task_sha256": task_sha256(task),
        "split_manifest_sha256": task_manifest_sha256(all_tasks(task.split)),
        "source_audit": source_audit_payload,
        "codebase_lock": asdict(CODEBASE_LOCKS[library.codebase]),
        "score_rule": library.score_rule,
        "score_bounds": [library.score_lower, library.score_upper],
        "episodes_per_policy": episodes,
        "seed_base": selected_seed_base,
        "canonical_seed_base": canonical_seed_base,
        "canonical_seed_schedule": selected_seed_base == canonical_seed_base,
        "determinism_verified": verify_determinism,
        "runtime_seconds": {
            "collection": collection_runtime,
            "determinism_verification": verification_runtime,
            "total": collection_runtime + verification_runtime,
        },
        "summaries": {
            selected_policy: summarize_v2_outcomes(rows)
            for selected_policy, rows in outcomes.items()
        },
        "outcomes": {
            selected_policy: outcome_rows(rows)
            for selected_policy, rows in outcomes.items()
        },
    }


def main() -> None:
    args = parse_args()
    tasks = domain_tasks(args.domain, args.split)
    if not 0 <= args.task_index < len(tasks):
        raise ValueError("task-index is outside the frozen task list")
    payload = run_development_task_payload(
        tasks[args.task_index],
        policy=args.policy,
        episodes=args.episodes,
        seed_base=args.seed_base,
        verify_determinism=args.verify_determinism,
        source_root=args.source_root,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    if not args.quiet:
        print(rendered, end="")


if __name__ == "__main__":
    main()
