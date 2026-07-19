"""Build and finalize the registration payload for external confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from experiments.external_budget_baselines import fixed_budget_comparison
from experiments.external_familywise_verifier import FamilywisePilotPlan
from experiments.external_study_design import (
    DOMAINS,
    ENVIRONMENT_LOCKS,
    POLICY_LIBRARIES,
    RUNTIME_DEPENDENCIES,
    canonical_sha256,
    domain_tasks,
    task_manifest_sha256,
)


LOCKED_EXTERNAL_FILES = (
    "experiments/conformal_router.py",
    "experiments/pilot_verifier.py",
    "experiments/external_study_design.py",
    "experiments/external_familywise_verifier.py",
    "experiments/external_budget_baselines.py",
    "experiments/external_domain_adapters.py",
    "experiments/external_development.py",
    "experiments/external_router_build.py",
    "experiments/external_protocol_lock.py",
    "experiments/external_confirmation_evaluation.py",
)


def sha256_file(path: Path) -> str:
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def source_manifest() -> list[dict[str, str]]:
    return [
        {"path": path, "sha256": sha256_file(Path(path))}
        for path in LOCKED_EXTERNAL_FILES
    ]


def truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_locked_design(
    development_root: Path,
    router_root: Path,
    evaluation_root: Path,
) -> dict:
    router_summary_path = router_root / "summary.json"
    router_summary = json.loads(router_summary_path.read_text(encoding="utf-8"))
    proposals_path = router_root / "all_proposals.csv"
    proposal_rows = read_csv(proposals_path)
    active = [row for row in proposal_rows if truthy(row["proposal_active"])]
    if not active:
        raise RuntimeError("external protocol requires at least one frozen proposal")
    expected_tasks = {
        task.name
        for domain in DOMAINS
        for task in domain_tasks(domain, "confirmation")
    }
    if len(proposal_rows) != len(expected_tasks) or {
        row["task"] for row in proposal_rows
    } != expected_tasks:
        raise RuntimeError("proposal table does not cover the complete external confirmation suite")
    task_index = {
        task.name: task
        for domain in DOMAINS
        for task in domain_tasks(domain, "confirmation")
    }
    for row in proposal_rows:
        task = task_index[row["task"]]
        if row["domain"] != task.domain:
            raise RuntimeError(f"proposal domain changed for {task.name}")
        library = POLICY_LIBRARIES[task.domain]
        if row["fallback_policy"] != library.fallback:
            raise RuntimeError(f"proposal fallback changed for {task.name}")
        if row["candidate_policy"] and row["candidate_policy"] not in library.candidates:
            raise RuntimeError(f"unknown proposal candidate for {task.name}")
        if truthy(row["proposal_active"]) and not row["candidate_policy"]:
            raise RuntimeError(f"active proposal has no candidate for {task.name}")
    if int(router_summary["confirmation_task_count"]) != len(proposal_rows):
        raise RuntimeError("router summary confirmation-task count is inconsistent")
    if int(router_summary["proposal_family_size"]) != len(active):
        raise RuntimeError("router summary proposal-family size is inconsistent")
    if router_summary["all_proposals_sha256"] != sha256_file(proposals_path):
        raise RuntimeError("router summary proposal hash is inconsistent")
    plan = FamilywisePilotPlan(
        proposal_family_size=len(active),
        episodes_per_batch=20,
    )
    budget = fixed_budget_comparison(
        all_tasks=sorted(expected_tasks),
        proposal_tasks=[row["task"] for row in active],
        plan=plan,
    )
    candidate_policy_by_task = {
        row["task"]: (
            row["candidate_policy"]
            or POLICY_LIBRARIES[row["domain"]].candidates[0]
        )
        for row in proposal_rows
    }
    development_artifacts = []
    for domain in DOMAINS:
        for split in ("development", "calibration"):
            expected_split_tasks = domain_tasks(domain, split)
            summary_path = development_root / domain / split / "summary.json"
            development_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if development_summary["domain"] != domain or development_summary["split"] != split:
                raise RuntimeError(f"development summary identity changed for {domain}/{split}")
            if development_summary["environment_lock"] != asdict(ENVIRONMENT_LOCKS[domain]):
                raise RuntimeError(f"development environment lock changed for {domain}/{split}")
            if development_summary["full_split_task_manifest_sha256"] != task_manifest_sha256(
                expected_split_tasks
            ):
                raise RuntimeError(f"development task manifest changed for {domain}/{split}")
            if development_summary["executed_task_names"] != [
                task.name for task in expected_split_tasks
            ]:
                raise RuntimeError(f"development task coverage changed for {domain}/{split}")
            expected_policies = [
                POLICY_LIBRARIES[domain].fallback,
                *POLICY_LIBRARIES[domain].candidates,
            ]
            if development_summary["policies"] != expected_policies:
                raise RuntimeError(f"development policy coverage changed for {domain}/{split}")
            for filename in ("episodes.csv", "aggregate_scores.csv", "summary.json"):
                path = development_root / domain / split / filename
                development_artifacts.append(
                    {"path": path.as_posix(), "sha256": sha256_file(path)}
                )
    design = {
        "protocol_id": "riskshiftbench-external-confirmation-v1",
        "purpose": (
            "Independent confirmation of familywise pilot-verified routing on "
            "environment code maintained outside RiskShiftBench."
        ),
        "environment_locks": {
            domain: asdict(ENVIRONMENT_LOCKS[domain]) for domain in DOMAINS
        },
        "runtime_dependencies": {
            domain: dict(RUNTIME_DEPENDENCIES[domain]) for domain in DOMAINS
        },
        "task_suites": {
            domain: {
                split: {
                    "task_count": len(domain_tasks(domain, split)),
                    "task_manifest_sha256": task_manifest_sha256(
                        domain_tasks(domain, split)
                    ),
                }
                for split in ("development", "calibration", "confirmation")
            }
            for domain in DOMAINS
        },
        "policy_libraries": {
            domain: asdict(POLICY_LIBRARIES[domain]) for domain in DOMAINS
        },
        "development_artifacts": development_artifacts,
        "router_lock": {
            "summary_path": router_summary_path.as_posix(),
            "summary_sha256": sha256_file(router_summary_path),
            "proposal_path": proposals_path.as_posix(),
            "proposal_sha256": sha256_file(proposals_path),
            "proposal_family_size": len(active),
            "proposal_tasks": [row["task"] for row in active],
            "candidate_policy_by_task": candidate_policy_by_task,
            "domain_reports": router_summary["domains"],
        },
        "pilot": {
            "familywise_alpha": plan.familywise_alpha,
            "multiplicity": "Bonferroni over the complete frozen proposal family",
            "local_alpha": plan.local_alpha,
            "required_unanimous_batches": plan.required_unanimous_batches,
            "batch_indices": list(range(plan.required_unanimous_batches)),
            "episodes_per_batch": plan.episodes_per_batch,
            "seed_base": 100_000_000,
            "task_seed_stride": 1_000_000,
            "batch_seed_stride": plan.episodes_per_batch,
            "common_random_numbers": True,
            "tie_rule": "A zero batch advantage forces rejection.",
            "minimum_mean_advantage": plan.min_mean_advantage,
        },
        "cost_matched_baselines": {
            **budget,
            "reporting_rule": (
                "Report fallback-only, candidate-everywhere, fit-only, familywise "
                "pilot verification, and the outcome-blind random-task allocation "
                "with the identical candidate-plus-fallback pilot episode budget. "
                "The thin uniform-all-task allocation is retained as an accounting "
                "diagnostic but is not treated as a viable gate."
            ),
        },
        "evaluation": {
            "seed_indices": [0, 1, 2, 3, 4],
            "episodes_per_task_policy_seed": 20,
            "seed_base": 500_000_000,
            "seed_index_stride": 100_000_000,
            "task_seed_stride": 1_000_000,
            "common_random_numbers": True,
            "output_root": evaluation_root.as_posix(),
            "pilot_seed_disjointness": "Pilot and final seed bases are disjoint.",
        },
        "primary_analysis": {
            "estimand": (
                "Equal-domain mean relative score improvement of the familywise "
                "pilot-verified router over each fixed domain fallback."
            ),
            "relative_improvement_formula": (
                "(route_score - fallback_score) / max(abs(fallback_score), 1e-12)"
            ),
            "primary_contrast": "familywise_pilot versus fallback_only",
            "secondary_contrasts": (
                "Candidate-everywhere, fit-only, and outcome-blind random-pilot "
                "routes are diagnostic cost/allocation comparisons."
            ),
            "task_score": "mean episode utility plus 0.5 times lower-tail 5% utility",
            "fixed_domain_interval": (
                "Resample tasks independently within each domain, then give each "
                "domain equal weight; 10000 percentile bootstrap replicates."
            ),
            "domain_resampling_interval": (
                "Resample the three named external domains and tasks within sampled "
                "domains; report separately because three domains give coarse support."
            ),
            "randomization": (
                "Task-level sign flips nested within domain with equal domain weight; "
                "100000 randomizations."
            ),
            "confirmatory_scope": (
                "Only the external suites contribute confirmatory evidence. Earlier "
                "RiskShiftBench suites remain descriptive."
            ),
        },
        "score_sensitivity": {
            "status": "frozen_before_confirmation",
            "frozenlake_failure_penalty": [20.0, 35.0, 50.0],
            "frozenlake_step_penalty": [0.05, 0.10, 0.20],
            "knapsack_early_exhaustion_penalty": [50.0, 100.0, 200.0],
            "knapsack_unused_capacity_penalty": [0.025, 0.05, 0.10],
            "safety_goal_bonus": [2.5, 5.0, 10.0],
            "safety_no_goal_penalty": [1.0, 2.0, 4.0],
            "route_rule": "Hold the deployed routes fixed for every sensitivity value.",
        },
        "source_manifest": source_manifest(),
        "reporting_commitment": (
            "Report every fixed task, proposal, pilot outcome, route, baseline, and "
            "score variant regardless of direction. Do not revise the method on these suites."
        ),
    }
    return design


def write_registration_draft(
    development_root: Path,
    router_root: Path,
    evaluation_root: Path,
    locked_design_path: Path,
    protocol_path: Path,
) -> None:
    if locked_design_path.exists() or protocol_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing locked design or registration draft"
        )
    design = build_locked_design(development_root, router_root, evaluation_root)
    canonical_design_hash = canonical_sha256(design)
    locked_design_path.parent.mkdir(parents=True, exist_ok=True)
    locked_design_path.write_bytes(json_bytes(design))
    design_hash = sha256_bytes(locked_design_path)
    wrapper = {
        "status": "awaiting_external_registration",
        "locked_design_path": locked_design_path.as_posix(),
        "locked_design_sha256": design_hash,
        "locked_design_canonical_sha256": canonical_design_hash,
        "registration": {
            "provider": None,
            "url": None,
            "registered_at": None,
        },
        "execution_guard": (
            "Pilot and final commands refuse this draft. Register the locked design "
            "externally, then finalize the wrapper with the immutable URL and timestamp."
        ),
    }
    protocol_path.write_bytes(json_bytes(wrapper))
    print(f"locked_design={locked_design_path}")
    print(f"locked_design_sha256={design_hash}")
    print(f"registration_draft={protocol_path}")


def finalize_registration(
    draft_path: Path,
    output_path: Path,
    provider: str,
    url: str,
    registered_at: str,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite registered protocol: {output_path}")
    if not provider.strip():
        raise ValueError("registration provider cannot be blank")
    if not url.startswith("https://"):
        raise ValueError("registration URL must be an immutable HTTPS URL")
    timestamp = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("registration timestamp must include a timezone")
    wrapper = json.loads(draft_path.read_text(encoding="utf-8"))
    if wrapper.get("status") != "awaiting_external_registration":
        raise RuntimeError("only an unregistered draft can be finalized")
    design_path = Path(wrapper["locked_design_path"])
    design = json.loads(design_path.read_text(encoding="utf-8"))
    observed_file_hash = sha256_bytes(design_path)
    if observed_file_hash != wrapper["locked_design_sha256"]:
        raise RuntimeError("locked design bytes changed after draft creation")
    observed_canonical_hash = canonical_sha256(design)
    if observed_canonical_hash != wrapper["locked_design_canonical_sha256"]:
        raise RuntimeError("locked design changed after draft creation")
    wrapper["status"] = "externally_registered_locked"
    wrapper["registration"] = {
        "provider": provider,
        "url": url,
        "registered_at": registered_at,
        "registered_design_sha256": wrapper["locked_design_sha256"],
    }
    wrapper["execution_guard"] = "Registration fields present; runtime still validates every hash."
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(json_bytes(wrapper))
    print(f"registered_protocol={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, default=Path("artifacts/external_development_v1"))
    parser.add_argument("--router-root", type=Path, default=Path("artifacts/external_router_lock_v1"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/external_confirmation_v1"))
    parser.add_argument(
        "--locked-design",
        type=Path,
        default=Path("configs/external_confirmation_locked_design_v1.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/external_confirmation_protocol_v1.registration-draft.json"),
    )
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--provider")
    parser.add_argument("--url")
    parser.add_argument("--registered-at")
    parser.add_argument(
        "--registered-output",
        type=Path,
        default=Path("configs/external_confirmation_protocol_v1.registered.json"),
    )
    args = parser.parse_args()
    if args.finalize:
        if not all((args.provider, args.url, args.registered_at)):
            raise ValueError("finalization requires --provider, --url, and --registered-at")
        finalize_registration(
            args.protocol,
            args.registered_output,
            args.provider,
            args.url,
            args.registered_at,
        )
    else:
        write_registration_draft(
            development_root=args.development_root,
            router_root=args.router_root,
            evaluation_root=args.evaluation_root,
            locked_design_path=args.locked_design,
            protocol_path=args.protocol,
        )


if __name__ == "__main__":
    main()
