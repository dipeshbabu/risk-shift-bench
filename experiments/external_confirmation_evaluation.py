"""Guarded pilot and final evaluation for registered external confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from experiments.external_domain_adapters import (
    lower_tail_mean,
    run_external_task,
)
from experiments.external_familywise_verifier import (
    FamilywisePilotPlan,
    verify_familywise_promotion,
)
from experiments.external_study_design import (
    DOMAINS,
    POLICY_LIBRARIES,
    canonical_sha256,
    domain_tasks,
    task_manifest_sha256,
)


def sha256_file(path: Path) -> str:
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_once(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked outcome file: {path}")
    if not rows:
        raise ValueError(f"cannot write empty outcome file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json_once(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked result file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validate_hash_record(record: dict[str, str]) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise RuntimeError(f"locked file is missing: {path}")
    observed = sha256_file(path)
    if observed != record["sha256"]:
        raise RuntimeError(f"locked file changed: {path}")


def validate_protocol(
    protocol_path: Path,
    *,
    require_registration: bool,
) -> tuple[dict, dict]:
    """Validate the wrapper, registration state, task manifests, and every file hash."""

    wrapper = json.loads(protocol_path.read_text(encoding="utf-8"))
    design_path = Path(wrapper["locked_design_path"])
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if sha256_bytes(design_path) != wrapper["locked_design_sha256"]:
        raise RuntimeError("locked external design bytes do not match its wrapper")
    if canonical_sha256(design) != wrapper["locked_design_canonical_sha256"]:
        raise RuntimeError("locked external design hash does not match its wrapper")
    if require_registration:
        if wrapper.get("status") != "externally_registered_locked":
            raise RuntimeError(
                "confirmation execution is blocked until the design is externally registered"
            )
        registration = wrapper.get("registration", {})
        required = ("provider", "url", "registered_at", "registered_design_sha256")
        if any(not registration.get(field) for field in required):
            raise RuntimeError("external registration metadata is incomplete")
        if not str(registration["url"]).startswith("https://"):
            raise RuntimeError("external registration URL must use HTTPS")
        if registration["registered_design_sha256"] != wrapper["locked_design_sha256"]:
            raise RuntimeError("registered design hash does not match the execution wrapper")
    for record in design["source_manifest"]:
        _validate_hash_record(record)
    for record in design["development_artifacts"]:
        _validate_hash_record(record)
    router = design["router_lock"]
    _validate_hash_record(
        {"path": router["summary_path"], "sha256": router["summary_sha256"]}
    )
    _validate_hash_record(
        {"path": router["proposal_path"], "sha256": router["proposal_sha256"]}
    )
    for domain in DOMAINS:
        tasks = domain_tasks(domain, "confirmation")
        task_lock = design["task_suites"][domain]["confirmation"]
        if len(tasks) != task_lock["task_count"]:
            raise RuntimeError(f"confirmation task count changed for {domain}")
        if task_manifest_sha256(tasks) != task_lock["task_manifest_sha256"]:
            raise RuntimeError(f"confirmation task manifest changed for {domain}")
    proposal_rows = read_csv(Path(router["proposal_path"]))
    expected_tasks = {
        task.name for domain in DOMAINS for task in domain_tasks(domain, "confirmation")
    }
    if len(proposal_rows) != len(expected_tasks) or {
        row["task"] for row in proposal_rows
    } != expected_tasks:
        raise RuntimeError("locked proposal table no longer covers the confirmation suite")
    if set(router["candidate_policy_by_task"]) != expected_tasks:
        raise RuntimeError("locked candidate-policy map no longer covers the suite")
    active = [row for row in proposal_rows if _truthy(row["proposal_active"])]
    if len(active) != int(router["proposal_family_size"]):
        raise RuntimeError("locked proposal-family size no longer matches the proposal table")
    for row in proposal_rows:
        expected_candidate = row["candidate_policy"] or POLICY_LIBRARIES[
            row["domain"]
        ].candidates[0]
        if router["candidate_policy_by_task"][row["task"]] != expected_candidate:
            raise RuntimeError(f"locked candidate-policy map changed for {row['task']}")
    return wrapper, design


def _task_index() -> dict[str, int]:
    tasks = [task for domain in DOMAINS for task in domain_tasks(domain, "confirmation")]
    return {task.name: index for index, task in enumerate(tasks)}


def _proposal_rows(design: dict) -> list[dict[str, str]]:
    return read_csv(Path(design["router_lock"]["proposal_path"]))


def _truthy(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value.strip().lower() == "true"


def _allocation_tasks(design: dict, mode: str) -> set[str]:
    baselines = design["cost_matched_baselines"]
    key = "proposal_focused" if mode == "proposal" else "outcome_blind_random_tasks"
    allocation = baselines[key]
    if mode == "random":
        allocation = allocation["allocation"]
    return {row["task"] for row in allocation if int(row["batches"]) > 0}


def _batch_path(design: dict, mode: str, domain: str, batch_index: int) -> Path:
    root = Path(design["evaluation"]["output_root"])
    return root / f"pilot_{mode}" / domain / f"batch_{batch_index:02d}.csv"


def _run_pair(
    task,
    candidate: str,
    fallback: str,
    episodes: int,
    seed_base: int,
    environment_source: Path,
) -> list[dict]:
    rows = []
    for role, policy in (("candidate", candidate), ("fallback", fallback)):
        outcomes = run_external_task(
            task,
            policy,
            episodes=episodes,
            seed_base=seed_base,
            environment_source=environment_source,
        )
        rows.extend({"role": role, **asdict(outcome)} for outcome in outcomes)
    return rows


def run_pilot_batch(
    design: dict,
    mode: str,
    domain: str,
    batch_index: int,
    environment_source: Path,
) -> Path:
    if mode not in {"proposal", "random"}:
        raise ValueError("pilot mode must be proposal or random")
    pilot = design["pilot"]
    required = int(pilot["required_unanimous_batches"])
    if not 0 <= batch_index < required:
        raise ValueError(f"batch index must lie between 0 and {required - 1}")
    selected = _allocation_tasks(design, mode)
    candidates = design["router_lock"]["candidate_policy_by_task"]
    indices = _task_index()
    rows = []
    for task in domain_tasks(domain, "confirmation"):
        if task.name not in selected:
            continue
        seed_base = (
            int(pilot["seed_base"])
            + indices[task.name] * int(pilot["task_seed_stride"])
            + batch_index * int(pilot["batch_seed_stride"])
        )
        rows.extend(
            _run_pair(
                task,
                candidates[task.name],
                POLICY_LIBRARIES[domain].fallback,
                int(pilot["episodes_per_batch"]),
                seed_base,
                environment_source,
            )
        )
    if not rows:
        raise RuntimeError(f"no {mode} pilot tasks were allocated to {domain}")
    path = _batch_path(design, mode, domain, batch_index)
    write_csv_once(path, rows)
    return path


def _score(rows: list[dict[str, str]]) -> float:
    utilities = [float(row["utility"]) for row in rows]
    if not utilities:
        raise RuntimeError("missing locked pilot outcomes")
    return mean(utilities) + 0.5 * lower_tail_mean(utilities)


def _gate_mode(design: dict, mode: str) -> list[dict]:
    selected = _allocation_tasks(design, mode)
    pilot = design["pilot"]
    plan = FamilywisePilotPlan(
        proposal_family_size=len(selected),
        familywise_alpha=float(pilot["familywise_alpha"]),
        episodes_per_batch=int(pilot["episodes_per_batch"]),
        min_mean_advantage=float(pilot["minimum_mean_advantage"]),
    )
    if plan.required_unanimous_batches != int(pilot["required_unanimous_batches"]):
        raise RuntimeError("locked familywise batch count is internally inconsistent")
    candidates_by_task = design["router_lock"]["candidate_policy_by_task"]
    task_indices = _task_index()
    rows = []
    for domain in DOMAINS:
        domain_selected = sorted(
            task.name
            for task in domain_tasks(domain, "confirmation")
            if task.name in selected
        )
        if not domain_selected:
            continue
        advantages = {task: [] for task in domain_selected}
        for batch_index in range(plan.required_unanimous_batches):
            batch_rows = read_csv(_batch_path(design, mode, domain, batch_index))
            expected_rows = 2 * plan.episodes_per_batch * len(domain_selected)
            if len(batch_rows) != expected_rows:
                raise RuntimeError(f"unexpected row count in {mode} pilot batch for {domain}")
            if any(
                row["domain"] != domain or row["task"] not in advantages
                for row in batch_rows
            ):
                raise RuntimeError(f"unexpected task in {mode} pilot batch for {domain}")
            for task in domain_selected:
                candidate = [
                    row for row in batch_rows if row["task"] == task and row["role"] == "candidate"
                ]
                fallback = [
                    row for row in batch_rows if row["task"] == task and row["role"] == "fallback"
                ]
                expected = plan.episodes_per_batch
                if len(candidate) != expected or len(fallback) != expected:
                    raise RuntimeError(f"incomplete {mode} pilot batch for {task}")
                expected_seed_base = (
                    int(pilot["seed_base"])
                    + task_indices[task] * int(pilot["task_seed_stride"])
                    + batch_index * int(pilot["batch_seed_stride"])
                )
                expected_seeds = list(range(expected_seed_base, expected_seed_base + expected))
                candidate_seeds = [int(row["seed"]) for row in candidate]
                fallback_seeds = [int(row["seed"]) for row in fallback]
                if candidate_seeds != expected_seeds or fallback_seeds != expected_seeds:
                    raise RuntimeError(f"common-random-number pairing changed for {task}")
                if any(row["policy"] != candidates_by_task[task] for row in candidate):
                    raise RuntimeError(f"candidate policy changed in pilot batch for {task}")
                fallback_policy = POLICY_LIBRARIES[domain].fallback
                if any(row["policy"] != fallback_policy for row in fallback):
                    raise RuntimeError(f"fallback policy changed in pilot batch for {task}")
                advantages[task].append(_score(candidate) - _score(fallback))
        for task, values in advantages.items():
            result = verify_familywise_promotion(values, plan)
            rows.append(
                {
                    "mode": mode,
                    "task": task,
                    **asdict(result),
                    "batch_advantages": json.dumps(values, separators=(",", ":")),
                }
            )
    return rows


def _gate_signature(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "mode": row["mode"],
                "task": row["task"],
                "accepted": _truthy(row["accepted"]),
                "mean_advantage": float(row["mean_advantage"]),
                "sign_test_p": float(row["sign_test_p"]),
                "positive_batches": int(row["positive_batches"]),
                "negative_batches": int(row["negative_batches"]),
                "zero_batches": int(row["zero_batches"]),
                "reason": row["reason"],
                "batch_advantages": [
                    float(value) for value in json.loads(row["batch_advantages"])
                ],
            }
        )
    return sorted(normalized, key=lambda row: (row["mode"], row["task"]))


def lock_gates(design: dict) -> Path:
    rows = _gate_mode(design, "proposal") + _gate_mode(design, "random")
    path = Path(design["evaluation"]["output_root"]) / "gate_decisions.csv"
    write_csv_once(path, rows)
    return path


def run_final_seed(
    design: dict,
    domain: str,
    seed_index: int,
    environment_source: Path,
) -> Path:
    evaluation = design["evaluation"]
    if seed_index not in evaluation["seed_indices"]:
        raise ValueError(f"seed index is not locked: {seed_index}")
    candidates = design["router_lock"]["candidate_policy_by_task"]
    indices = _task_index()
    rows = []
    for task in domain_tasks(domain, "confirmation"):
        seed_base = (
            int(evaluation["seed_base"])
            + seed_index * int(evaluation["seed_index_stride"])
            + indices[task.name] * int(evaluation["task_seed_stride"])
        )
        rows.extend(
            _run_pair(
                task,
                candidates[task.name],
                POLICY_LIBRARIES[domain].fallback,
                int(evaluation["episodes_per_task_policy_seed"]),
                seed_base,
                environment_source,
            )
        )
    path = (
        Path(evaluation["output_root"])
        / "final"
        / domain
        / f"seed_{seed_index:02d}.csv"
    )
    write_csv_once(path, rows)
    return path


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _equal_domain_mean(effects: dict[str, list[float]]) -> float:
    return mean(mean(effects[domain]) for domain in DOMAINS)


def _inference(effects: dict[str, list[float]], seed: int) -> dict:
    observed = _equal_domain_mean(effects)
    rng = random.Random(seed)
    fixed = []
    resampled_domains = []
    for _replicate in range(10_000):
        fixed_draw = {
            domain: [rng.choice(effects[domain]) for _ in effects[domain]]
            for domain in DOMAINS
        }
        fixed.append(_equal_domain_mean(fixed_draw))
        domain_draw = [rng.choice(DOMAINS) for _ in DOMAINS]
        resampled_domains.append(
            mean(
                mean(rng.choice(effects[domain]) for _ in effects[domain])
                for domain in domain_draw
            )
        )
    extreme = 0
    for _replicate in range(100_000):
        randomized = {
            domain: [value * rng.choice((-1.0, 1.0)) for value in effects[domain]]
            for domain in DOMAINS
        }
        extreme += _equal_domain_mean(randomized) >= observed
    return {
        "equal_domain_mean_relative_improvement": observed,
        "fixed_domain_bootstrap_95_ci": [_quantile(fixed, 0.025), _quantile(fixed, 0.975)],
        "domain_resampling_bootstrap_95_ci": [
            _quantile(resampled_domains, 0.025),
            _quantile(resampled_domains, 0.975),
        ],
        "one_sided_sign_flip_p": (extreme + 1) / 100_001,
    }


def _sensitivity_variants(design: dict, domain: str) -> list[dict[str, float]]:
    grid = design["score_sensitivity"]
    if domain == "gymnasium_frozenlake":
        keys = ("frozenlake_failure_penalty", "frozenlake_step_penalty")
    elif domain == "or_gym_online_knapsack":
        keys = ("knapsack_early_exhaustion_penalty", "knapsack_unused_capacity_penalty")
    elif domain == "safety_gymnasium_point_goal":
        keys = ("safety_goal_bonus", "safety_no_goal_penalty")
    else:
        raise KeyError(domain)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[key] for key in keys))]


def _variant_utility(row: dict[str, str], task, parameters: dict[str, float]) -> float:
    domain = task.domain
    if domain == "gymnasium_frozenlake":
        return (
            100.0 * int(row["successes"])
            - parameters["frozenlake_failure_penalty"] * float(_truthy(row["failure"]))
            - parameters["frozenlake_step_penalty"] * int(row["steps"])
        )
    if domain == "or_gym_online_knapsack":
        return (
            float(row["raw_return"])
            - parameters["knapsack_early_exhaustion_penalty"] * float(_truthy(row["failure"]))
            - parameters["knapsack_unused_capacity_penalty"] * float(row["resource_residual"])
        )
    if domain == "safety_gymnasium_point_goal":
        cost_weight = float(task.parameter_dict()["cost_weight"])
        return (
            float(row["raw_return"])
            + parameters["safety_goal_bonus"] * int(row["successes"])
            - cost_weight * float(row["cost"])
            - parameters["safety_no_goal_penalty"] * float(_truthy(row["failure"]))
        )
    raise KeyError(domain)


def _score_values(values: list[float]) -> float:
    return mean(values) + 0.5 * lower_tail_mean(values)


def _score_sensitivity(
    design: dict,
    all_final: list[dict[str, str]],
    routes: dict[str, set[str]],
) -> dict:
    task_map = {
        task.name: task for domain in DOMAINS for task in domain_tasks(domain, "confirmation")
    }
    report = {}
    for strategy, candidate_tasks in routes.items():
        if strategy == "fallback_only":
            continue
        report[strategy] = {}
        for domain in DOMAINS:
            variants = []
            domain_tasks_locked = domain_tasks(domain, "confirmation")
            for parameters in _sensitivity_variants(design, domain):
                relative_effects = []
                for task in domain_tasks_locked:
                    role_scores = {}
                    for role in ("candidate", "fallback"):
                        values = [
                            _variant_utility(row, task_map[task.name], parameters)
                            for row in all_final
                            if row["task"] == task.name and row["role"] == role
                        ]
                        role_scores[role] = _score_values(values)
                    routed = role_scores["candidate"] if task.name in candidate_tasks else role_scores["fallback"]
                    relative_effects.append(
                        (routed - role_scores["fallback"])
                        / max(abs(role_scores["fallback"]), 1e-12)
                    )
                variants.append(
                    {
                        "parameters": parameters,
                        "mean_relative_improvement": mean(relative_effects),
                    }
                )
            estimates = [row["mean_relative_improvement"] for row in variants]
            report[strategy][domain] = {
                "route_held_fixed": True,
                "minimum": min(estimates),
                "maximum": max(estimates),
                "variants": variants,
            }
    return report


def combine_results(design: dict) -> tuple[Path, Path]:
    root = Path(design["evaluation"]["output_root"])
    gate_path = root / "gate_decisions.csv"
    gate_rows = read_csv(gate_path)
    recomputed_gate_rows = _gate_mode(design, "proposal") + _gate_mode(design, "random")
    if _gate_signature(gate_rows) != _gate_signature(recomputed_gate_rows):
        raise RuntimeError("locked gate decisions no longer match the pilot outcomes")
    gate_rows = recomputed_gate_rows
    accepted = {
        mode: {row["task"] for row in gate_rows if row["mode"] == mode and row["accepted"]}
        for mode in ("proposal", "random")
    }
    proposal_rows = _proposal_rows(design)
    fit_tasks = {row["task"] for row in proposal_rows if _truthy(row["proposal_active"])}
    all_final = []
    final_paths = []
    for domain in DOMAINS:
        for seed_index in design["evaluation"]["seed_indices"]:
            final_path = root / "final" / domain / f"seed_{seed_index:02d}.csv"
            final_paths.append(final_path)
            all_final.extend(read_csv(final_path))
    task_scores = {}
    expected_final_rows = (
        len(design["evaluation"]["seed_indices"])
        * int(design["evaluation"]["episodes_per_task_policy_seed"])
    )
    for domain in DOMAINS:
        for task in domain_tasks(domain, "confirmation"):
            role_rows = {
                role: [
                    row
                    for row in all_final
                    if row["task"] == task.name and row["role"] == role
                ]
                for role in ("candidate", "fallback")
            }
            if any(len(rows) != expected_final_rows for rows in role_rows.values()):
                raise RuntimeError(f"incomplete final evaluation for {task.name}")
            candidate_seeds = sorted(row["seed"] for row in role_rows["candidate"])
            fallback_seeds = sorted(row["seed"] for row in role_rows["fallback"])
            if candidate_seeds != fallback_seeds:
                raise RuntimeError(
                    f"final common-random-number pairing changed for {task.name}"
                )
            if len(set(candidate_seeds)) != expected_final_rows:
                raise RuntimeError(f"duplicate final seeds for {task.name}")
            task_scores[task.name] = {
                role: _score(role_rows[role])
                for role in ("candidate", "fallback")
            }
    routes = {
        "fallback_only": set(),
        "candidate_everywhere": set(task_scores),
        "fit_only": fit_tasks,
        "familywise_pilot": accepted["proposal"],
        "outcome_blind_random_pilot": accepted["random"],
    }
    task_rows = []
    effects_by_strategy = {
        strategy: {domain: [] for domain in DOMAINS}
        for strategy in routes
        if strategy != "fallback_only"
    }
    task_to_domain = {
        task.name: domain for domain in DOMAINS for task in domain_tasks(domain, "confirmation")
    }
    for strategy, candidate_tasks in routes.items():
        for task, scores in task_scores.items():
            route_score = scores["candidate"] if task in candidate_tasks else scores["fallback"]
            denominator = max(abs(scores["fallback"]), 1e-12)
            relative = (route_score - scores["fallback"]) / denominator
            task_rows.append(
                {
                    "strategy": strategy,
                    "domain": task_to_domain[task],
                    "task": task,
                    "selected_role": "candidate" if task in candidate_tasks else "fallback",
                    "fallback_score": scores["fallback"],
                    "candidate_score": scores["candidate"],
                    "route_score": route_score,
                    "relative_improvement": relative,
                }
            )
            if strategy != "fallback_only":
                effects_by_strategy[strategy][task_to_domain[task]].append(relative)
    summary = {
        "protocol_id": design["protocol_id"],
        "confirmatory_scope": design["primary_analysis"]["confirmatory_scope"],
        "strategies": {
            strategy: _inference(effects, seed=71_000 + index)
            for index, (strategy, effects) in enumerate(effects_by_strategy.items())
        },
        "score_sensitivity": _score_sensitivity(design, all_final, routes),
        "input_manifest": [
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in (gate_path, *final_paths)
        ],
    }
    task_path = root / "combined_task_results.csv"
    summary_path = root / "combined_summary.json"
    write_csv_once(task_path, task_rows)
    write_json_once(summary_path, summary)
    return task_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/external_confirmation_protocol_v1.registration-draft.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("dry-run")
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--mode", choices=("proposal", "random"), required=True)
    pilot.add_argument("--domain", choices=DOMAINS, required=True)
    pilot.add_argument("--batch-index", type=int, required=True)
    pilot.add_argument("--environment-source", type=Path, required=True)
    commands.add_parser("lock-gates")
    final = commands.add_parser("final")
    final.add_argument("--domain", choices=DOMAINS, required=True)
    final.add_argument("--seed-index", type=int, required=True)
    final.add_argument("--environment-source", type=Path, required=True)
    commands.add_parser("combine")
    args = parser.parse_args()
    require_registration = args.command != "dry-run"
    wrapper, design = validate_protocol(
        args.protocol,
        require_registration=require_registration,
    )
    if args.command == "dry-run":
        print(f"protocol_status={wrapper['status']}")
        print("protocol_hashes_valid=true")
        allowed = wrapper["status"] == "externally_registered_locked"
        print(f"confirmation_execution_allowed={str(allowed).lower()}")
    elif args.command == "pilot":
        print(
            f"pilot_output={run_pilot_batch(design, args.mode, args.domain, args.batch_index, args.environment_source)}"
        )
    elif args.command == "lock-gates":
        print(f"gate_output={lock_gates(design)}")
    elif args.command == "final":
        print(
            f"final_output={run_final_seed(design, args.domain, args.seed_index, args.environment_source)}"
        )
    elif args.command == "combine":
        task_path, summary_path = combine_results(design)
        print(f"combined_tasks={task_path}")
        print(f"combined_summary={summary_path}")


if __name__ == "__main__":
    main()
