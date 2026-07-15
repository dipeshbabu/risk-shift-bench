"""Hash-locked pilot and evaluation pipeline for three-domain routing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from experiments.pilot_verified_runtime import (
    DOMAINS,
    domain_policy_lookup,
    domain_router,
    domain_tasks,
    mapped_policy,
    proposal_rows,
    run_domain,
)
from experiments.pilot_verifier import PilotGateParams, verify_promotion
from risk_shift_bench.reporting import write_json
from risk_shift_bench.statistics import paired_score_report


REFERENCE_POLICY = "pilot_verified_router"
LOCKED_EXPERIMENT_FILES = (
    "experiments/conformal_router.py",
    "experiments/pilot_verifier.py",
    "experiments/frontier_v4_tasks.py",
    "experiments/inventory_domain.py",
    "experiments/frontier_router_builders.py",
    "experiments/pilot_verified_runtime.py",
    "experiments/pilot_verified_evaluation.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_manifest_sha256() -> str:
    paths = sorted(Path("src/risk_shift_bench").rglob("*.py"))
    paths.extend(Path(path) for path in LOCKED_EXPERIMENT_FILES)
    entries = [f"{path.as_posix()}:{sha256_file(path)}" for path in paths]
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def task_manifest_sha256(tasks) -> str:
    return canonical_sha256([asdict(task) for task in tasks])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_and_validate_protocol(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        protocol = json.load(file)
    observed_source = source_manifest_sha256()
    expected_source = protocol["implementation"]["source_manifest_sha256"]
    if observed_source != expected_source:
        raise RuntimeError(
            f"source manifest changed: expected {expected_source}, found {observed_source}"
        )
    for item in protocol["training_caches"]:
        observed = sha256_file(Path(item["path"]))
        if observed != item["sha256"]:
            raise RuntimeError(
                f"training cache changed for {item['path']}: expected {item['sha256']}, found {observed}"
            )
    for domain in DOMAINS:
        tasks = domain_tasks(domain)
        suite = protocol["domains"][domain]
        if len(tasks) != suite["task_count"]:
            raise RuntimeError(f"task count changed for {domain}")
        observed_manifest = task_manifest_sha256(tasks)
        if observed_manifest != suite["task_manifest_sha256"]:
            raise RuntimeError(f"task manifest changed for {domain}")
        router = domain_router(domain)
        routing = protocol["routing"][domain]
        if router.params.fallback_policy != routing["fallback_policy"]:
            raise RuntimeError(f"fallback changed for {domain}")
        if list(router.candidate_policies) != [routing["candidate_policy"]]:
            raise RuntimeError(f"fit-only candidate changed for {domain}")
        if canonical_sha256(router.report_dict()) != routing["router_report_sha256"]:
            raise RuntimeError(f"router report changed for {domain}")
        if canonical_sha256(proposal_rows(domain)) != routing["proposal_manifest_sha256"]:
            raise RuntimeError(f"proposal manifest changed for {domain}")
    return protocol


def output_dir(protocol: dict) -> Path:
    return Path(protocol["evaluation"]["out_dir"])


def run_pilot_batch(protocol_path: Path, protocol: dict, domain: str, batch: int) -> None:
    locked_batches = [int(value) for value in protocol["pilot"]["batches"]]
    if batch not in locked_batches:
        raise RuntimeError(f"pilot batch {batch} is not locked")
    path = output_dir(protocol) / "pilot" / domain / f"batch_{batch}.csv"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite pilot checkpoint: {path}")
    tasks = domain_tasks(domain)
    tasks_by_name = {task.name: task for task in tasks}
    proposals = proposal_rows(domain)
    policies = domain_policy_lookup(domain)
    episodes = int(protocol["pilot"]["episodes_per_batch"])
    hand_depth = int(protocol["evaluation"]["blackjack_hand_depth"])
    base_seed = int(protocol["pilot"]["seed_base"])
    rows = []
    for task_index, proposal in enumerate(proposals):
        row = dict(proposal)
        row["batch"] = batch
        if not proposal["proposal_active"]:
            row.update(
                {
                    "candidate_score": "",
                    "fallback_score": "",
                    "pilot_advantage": "",
                }
            )
            rows.append(row)
            continue
        candidate = proposal["candidate_policy"]
        fallback = proposal["fallback_policy"]
        seed = base_seed + batch * 1_000_000 + task_index * 10_000
        scores = run_domain(
            domain=domain,
            tasks=[tasks_by_name[proposal["task"]]],
            policies=[policies[candidate], policies[fallback]],
            episodes=episodes,
            seed=seed,
            hand_depth=hand_depth,
        )
        values = {score["policy"]: float(score["score"]) for score in scores}
        row.update(
            {
                "candidate_score": values[candidate],
                "fallback_score": values[fallback],
                "pilot_advantage": values[candidate] - values[fallback],
            }
        )
        rows.append(row)
    write_csv(path, rows)
    write_json(
        path.with_suffix(".json"),
        {
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "domain": domain,
            "batch": batch,
            "episodes_per_proposed_task": episodes,
            "checkpoint_sha256": sha256_file(path),
            "row_count": len(rows),
        },
    )
    print(f"pilot_checkpoint={path}")


def lock_gates(protocol_path: Path, protocol: dict, domain: str) -> None:
    gate_path = output_dir(protocol) / "gates" / f"{domain}.csv"
    if gate_path.exists():
        raise RuntimeError(f"refusing to overwrite pilot gates: {gate_path}")
    batches = [int(value) for value in protocol["pilot"]["batches"]]
    by_task: dict[str, list[dict]] = {}
    for batch in batches:
        path = output_dir(protocol) / "pilot" / domain / f"batch_{batch}.csv"
        if not path.exists():
            raise RuntimeError(f"missing pilot checkpoint: {path}")
        for row in read_csv(path):
            by_task.setdefault(row["task"], []).append(row)
    params = PilotGateParams(
        alpha=float(protocol["pilot"]["sign_test_alpha"]),
        min_mean_advantage=float(protocol["pilot"]["min_mean_advantage"]),
        min_nonzero_batches=int(protocol["pilot"]["min_nonzero_batches"]),
    )
    gates = []
    for proposal in proposal_rows(domain):
        task_rows = by_task.get(proposal["task"], [])
        if len(task_rows) != len(batches):
            raise RuntimeError(f"pilot batch coverage changed for {proposal['task']}")
        if not proposal["proposal_active"]:
            gates.append(
                {
                    **proposal,
                    "gate_accepted": False,
                    "selected_policy": proposal["fallback_policy"],
                    "pilot_mean_advantage": 0.0,
                    "sign_test_p": 1.0,
                    "positive_batches": 0,
                    "negative_batches": 0,
                    "zero_batches": 0,
                    "gate_reason": "no_fit_only_proposal",
                }
            )
            continue
        advantages = [float(row["pilot_advantage"]) for row in task_rows]
        result = verify_promotion(advantages, params)
        gates.append(
            {
                **proposal,
                "gate_accepted": result.accepted,
                "selected_policy": (
                    proposal["candidate_policy"] if result.accepted else proposal["fallback_policy"]
                ),
                "pilot_mean_advantage": result.mean_advantage,
                "sign_test_p": result.sign_test_p,
                "positive_batches": result.positive_batches,
                "negative_batches": result.negative_batches,
                "zero_batches": result.zero_batches,
                "gate_reason": result.reason,
            }
        )
    write_csv(gate_path, gates)
    write_json(
        gate_path.with_suffix(".json"),
        {
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "domain": domain,
            "pilot_gate_params": asdict(params),
            "gate_count": len(gates),
            "accepted_count": sum(bool(row["gate_accepted"]) for row in gates),
            "gates_sha256": sha256_file(gate_path),
        },
    )
    print(f"gates={gate_path}")
    print(f"accepted={sum(bool(row['gate_accepted']) for row in gates)}")


def run_evaluation_seed(protocol_path: Path, protocol: dict, domain: str, seed: int) -> None:
    locked_seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    if seed not in locked_seeds:
        raise RuntimeError(f"evaluation seed {seed} is not locked")
    checkpoint = output_dir(protocol) / "evaluation" / domain / f"seed_{seed}.csv"
    if checkpoint.exists():
        raise RuntimeError(f"refusing to overwrite evaluation checkpoint: {checkpoint}")
    gate_path = output_dir(protocol) / "gates" / f"{domain}.csv"
    if not gate_path.exists():
        raise RuntimeError(f"missing locked pilot gates: {gate_path}")
    gates = read_csv(gate_path)
    delegates = {row["task"]: row["selected_policy"] for row in gates}
    router = domain_router(domain)
    fallback = router.params.fallback_policy
    candidate = router.candidate_policies[0]
    lookup = domain_policy_lookup(domain)
    policies = [mapped_policy(domain, delegates), lookup[fallback], lookup[candidate]]
    rows = run_domain(
        domain=domain,
        tasks=domain_tasks(domain),
        policies=policies,
        episodes=int(protocol["evaluation"]["episodes_per_task_seed"]),
        seed=seed,
        hand_depth=int(protocol["evaluation"]["blackjack_hand_depth"]),
    )
    write_csv(checkpoint, rows)
    write_json(
        checkpoint.with_suffix(".json"),
        {
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "domain": domain,
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "row_count": len(rows),
        },
    )
    print(f"evaluation_checkpoint={checkpoint}")


def mean_scores(rows: list[dict]) -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        cells.setdefault((row["task"], row["policy"]), []).append(float(row["score"]))
    return {cell: sum(values) / len(values) for cell, values in cells.items()}


def combine_domain(protocol_path: Path, protocol: dict, domain: str) -> None:
    combined_path = output_dir(protocol) / "evaluation" / domain / "combined.csv"
    if combined_path.exists():
        raise RuntimeError(f"refusing to overwrite combined evaluation: {combined_path}")
    seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    tasks = domain_tasks(domain)
    router = domain_router(domain)
    fallback = router.params.fallback_policy
    candidate = router.candidate_policies[0]
    expected_policies = {REFERENCE_POLICY, fallback, candidate}
    rows = []
    for seed in seeds:
        checkpoint = output_dir(protocol) / "evaluation" / domain / f"seed_{seed}.csv"
        if not checkpoint.exists():
            raise RuntimeError(f"missing evaluation checkpoint: {checkpoint}")
        seed_rows = read_csv(checkpoint)
        if {row["task"] for row in seed_rows} != {task.name for task in tasks}:
            raise RuntimeError(f"task coverage changed in {checkpoint}")
        if {row["policy"] for row in seed_rows} != expected_policies:
            raise RuntimeError(f"policy coverage changed in {checkpoint}")
        rows.extend(seed_rows)
    task_means = mean_scores(rows)
    gates = {row["task"]: row for row in read_csv(output_dir(protocol) / "gates" / f"{domain}.csv")}
    effects = []
    for task in tasks:
        reference_score = task_means[(task.name, REFERENCE_POLICY)]
        fallback_score = task_means[(task.name, fallback)]
        candidate_score = task_means[(task.name, candidate)]
        gate = gates[task.name]
        proposal_score = candidate_score if gate["proposal_active"] == "True" else fallback_score
        effects.append(
            {
                "domain": domain,
                "task": task.name,
                "gate_accepted": gate["gate_accepted"] == "True",
                "selected_policy": gate["selected_policy"],
                "router_score": reference_score,
                "fallback_score": fallback_score,
                "candidate_score": candidate_score,
                "router_delta": reference_score - fallback_score,
                "relative_router_delta": (reference_score - fallback_score) / max(abs(fallback_score), 1.0),
                "fit_only_proposal_delta": proposal_score - fallback_score,
            }
        )
    report_kwargs = {
        "rows": rows,
        "reference_policy": REFERENCE_POLICY,
        "score_field": "score",
        "unit": "task",
        "bootstrap_samples": int(protocol["inference"]["bootstrap_samples"]),
        "randomization_samples": int(protocol["inference"]["randomization_samples"]),
        "seed": int(protocol["inference"]["random_seed"]),
    }
    primary = paired_score_report(baseline_policy=fallback, **report_kwargs)
    secondary = paired_score_report(baseline_policy=candidate, **report_kwargs)
    write_csv(combined_path, rows)
    write_csv(combined_path.with_name("task_effects.csv"), effects)
    summary = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "domain": domain,
        "fallback_policy": fallback,
        "candidate_policy": candidate,
        "primary_task_level_inference": primary,
        "secondary_task_level_inference": secondary,
        "accepted_promotions": sum(effect["gate_accepted"] for effect in effects),
        "harmful_promotions": sum(
            effect["gate_accepted"] and effect["router_delta"] < 0.0 for effect in effects
        ),
        "mean_fit_only_proposal_delta": sum(effect["fit_only_proposal_delta"] for effect in effects) / len(effects),
        "task_effects": effects,
    }
    write_json(combined_path.with_name("summary.json"), summary)
    print(f"domain_summary={combined_path.with_name('summary.json')}")
    print(f"primary={primary}")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = int(round(probability * (len(ordered) - 1)))
    return ordered[min(max(index, 0), len(ordered) - 1)]


def combine_all(protocol_path: Path, protocol: dict) -> None:
    final_path = output_dir(protocol) / "summary.json"
    if final_path.exists():
        raise RuntimeError(f"refusing to overwrite final summary: {final_path}")
    domain_summaries = []
    effects_by_domain = {}
    for domain in DOMAINS:
        path = output_dir(protocol) / "evaluation" / domain / "summary.json"
        if not path.exists():
            raise RuntimeError(f"missing domain summary: {path}")
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
        domain_summaries.append(summary)
        effects_by_domain[domain] = [
            float(effect["relative_router_delta"]) for effect in summary["task_effects"]
        ]
    domain_means = {
        domain: sum(values) / len(values) for domain, values in effects_by_domain.items()
    }
    global_effect = sum(domain_means.values()) / len(domain_means)
    rng = random.Random(int(protocol["inference"]["random_seed"]))
    bootstrap_values = []
    for _ in range(int(protocol["inference"]["bootstrap_samples"])):
        sampled_domain_means = []
        for values in effects_by_domain.values():
            sampled = [values[rng.randrange(len(values))] for _ in values]
            sampled_domain_means.append(sum(sampled) / len(sampled))
        bootstrap_values.append(sum(sampled_domain_means) / len(sampled_domain_means))
    all_nonzero = [
        value
        for values in effects_by_domain.values()
        for value in values
        if abs(value) > 1e-15
    ]
    observed_abs = abs(global_effect)
    exceedances = 0
    samples = int(protocol["inference"]["randomization_samples"])
    for _ in range(samples):
        randomized_domain_means = []
        for values in effects_by_domain.values():
            randomized = [value if rng.random() < 0.5 else -value for value in values]
            randomized_domain_means.append(sum(randomized) / len(randomized))
        statistic = abs(sum(randomized_domain_means) / len(randomized_domain_means))
        exceedances += statistic >= observed_abs
    global_report = {
        "estimand": "Equal-domain mean relative score improvement over the fallback",
        "mean_relative_delta": global_effect,
        "percent_improvement": 100.0 * global_effect,
        "bootstrap_ci_low": percentile(bootstrap_values, 0.025),
        "bootstrap_ci_high": percentile(bootstrap_values, 0.975),
        "sign_flip_p": (exceedances + 1) / (samples + 1),
        "nonzero_task_effects": len(all_nonzero),
        "domain_mean_relative_deltas": domain_means,
    }
    final = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "primary_cross_domain_inference": global_report,
        "domain_summaries": domain_summaries,
        "total_accepted_promotions": sum(summary["accepted_promotions"] for summary in domain_summaries),
        "total_harmful_promotions": sum(summary["harmful_promotions"] for summary in domain_summaries),
        "retuned_after_confirmation": False,
    }
    write_json(final_path, final)
    print(f"final_summary={final_path}")
    print(f"primary_cross_domain_inference={global_report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/frontier_pilot_verified_protocol.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", choices=DOMAINS)
    parser.add_argument("--pilot-batch", type=int)
    parser.add_argument("--lock-gates", choices=DOMAINS)
    parser.add_argument("--eval", choices=DOMAINS)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--combine-domain", choices=DOMAINS)
    parser.add_argument("--combine-all", action="store_true")
    args = parser.parse_args()
    actions = [
        args.dry_run,
        args.pilot is not None,
        args.lock_gates is not None,
        args.eval is not None,
        args.combine_domain is not None,
        args.combine_all,
    ]
    if sum(bool(action) for action in actions) != 1:
        parser.error("choose exactly one pipeline action")
    if args.pilot is not None and args.pilot_batch is None:
        parser.error("--pilot requires --pilot-batch")
    if args.eval is not None and args.eval_seed is None:
        parser.error("--eval requires --eval-seed")

    protocol_path = Path(args.protocol)
    protocol = load_and_validate_protocol(protocol_path)
    print(f"protocol_sha256={sha256_file(protocol_path)}")
    if args.dry_run:
        for domain in DOMAINS:
            router = domain_router(domain)
            active = sum(row["proposal_active"] for row in proposal_rows(domain))
            print(
                f"{domain}: candidate={router.candidate_policies[0]} "
                f"proposals={active}/{len(domain_tasks(domain))}"
            )
        print("dry_run_complete=true")
    elif args.pilot is not None:
        run_pilot_batch(protocol_path, protocol, args.pilot, args.pilot_batch)
    elif args.lock_gates is not None:
        lock_gates(protocol_path, protocol, args.lock_gates)
    elif args.eval is not None:
        run_evaluation_seed(protocol_path, protocol, args.eval, args.eval_seed)
    elif args.combine_domain is not None:
        combine_domain(protocol_path, protocol, args.combine_domain)
    else:
        combine_all(protocol_path, protocol)


if __name__ == "__main__":
    main()
