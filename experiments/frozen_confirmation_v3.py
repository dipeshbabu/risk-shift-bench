"""Run the locked third confirmation suite without re-searching the selector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from risk_shift_bench.benchmark import run_benchmark
from risk_shift_bench.envs import RiskTask, benchmark_tasks
from risk_shift_bench.family_selector import family_candidate_lookup
from risk_shift_bench.lcb_selector import LCBSelectorParams, policy_from_scores
from risk_shift_bench.multiseed import aggregate_seed_scores, paired_policy_deltas, summarize_seed
from risk_shift_bench.reporting import write_json
from risk_shift_bench.statistics import paired_score_report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest_sha256() -> str:
    paths = sorted(Path("src/risk_shift_bench").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    entries = []
    workspace = Path.cwd().resolve()
    for path in paths:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(workspace).as_posix()
        except ValueError:
            label = resolved.as_posix()
        entries.append(f"{label}:{sha256_file(resolved)}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def task_manifest_sha256(tasks: list[RiskTask]) -> str:
    payload = json.dumps(
        [asdict(task) for task in tasks],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_selection_scores(path: Path) -> dict[str, dict[str, list[float]]]:
    scores: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            scores.setdefault(row["task"], {}).setdefault(row["policy"], []).append(float(row["score"]))
    return scores


def load_aggregate_scores(path: Path) -> dict[str, dict[str, list[float]]]:
    scores: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] == "task":
                scores.setdefault(row["task"], {}).setdefault(row["policy"], []).append(float(row["mean_score"]))
    return scores


def merge_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        loaded = load_selection_scores(path) if path.name == "selection_train_scores.csv" else load_aggregate_scores(path)
        for task, policy_scores in loaded.items():
            for policy, values in policy_scores.items():
                merged.setdefault(task, {}).setdefault(policy, []).extend(values)
    return {
        task: {policy: sum(values) / len(values) for policy, values in policy_scores.items()}
        for task, policy_scores in merged.items()
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_validate_protocol(path: Path) -> tuple[dict, list[RiskTask], list[Path]]:
    with path.open(encoding="utf-8") as file:
        protocol = json.load(file)

    observed_source_manifest = source_manifest_sha256()
    expected_source_manifest = protocol["implementation"]["source_manifest_sha256"].lower()
    if observed_source_manifest != expected_source_manifest:
        raise RuntimeError(
            "implementation source manifest changed: "
            f"expected {expected_source_manifest}, found {observed_source_manifest}"
        )

    tasks = benchmark_tasks(protocol["suite"]["name"])
    expected_count = int(protocol["suite"]["task_count"])
    if len(tasks) != expected_count:
        raise RuntimeError(f"suite task count changed: expected {expected_count}, found {len(tasks)}")
    observed_manifest = task_manifest_sha256(tasks)
    expected_manifest = protocol["suite"]["task_manifest_sha256"].lower()
    if observed_manifest != expected_manifest:
        raise RuntimeError(
            "suite manifest hash changed: "
            f"expected {expected_manifest}, found {observed_manifest}"
        )

    frozen_summary = Path(protocol["method"]["search_summary"]["path"])
    expected_summary_hash = protocol["method"]["search_summary"]["sha256"].lower()
    observed_summary_hash = sha256_file(frozen_summary)
    if observed_summary_hash != expected_summary_hash:
        raise RuntimeError(
            "frozen search summary hash changed: "
            f"expected {expected_summary_hash}, found {observed_summary_hash}"
        )
    with frozen_summary.open(encoding="utf-8") as file:
        frozen_search = json.load(file)
    if frozen_search["selected_params"] != protocol["method"]["selected_params"]:
        raise RuntimeError("protocol parameters do not match the frozen search summary")

    cache_paths = []
    for cache in protocol["method"]["score_caches"]:
        cache_path = Path(cache["path"])
        observed_hash = sha256_file(cache_path)
        if observed_hash != cache["sha256"].lower():
            raise RuntimeError(
                f"score-cache hash changed for {cache_path}: "
                f"expected {cache['sha256'].lower()}, found {observed_hash}"
            )
        cache_paths.append(cache_path)
    return protocol, tasks, cache_paths


def task_deltas(rows: list[dict], reference_policy: str, baselines: list[str]) -> list[dict]:
    cells: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        cells.setdefault((row["task"], int(row["seed"])), {})[row["policy"]] = float(row["score"])
    by_task: dict[str, dict[str, list[float]]] = {}
    for (task, _seed), values in cells.items():
        if reference_policy not in values:
            continue
        for baseline in baselines:
            if baseline in values:
                by_task.setdefault(task, {}).setdefault(baseline, []).append(
                    values[reference_policy] - values[baseline]
                )
    output = []
    for task, baseline_deltas in sorted(by_task.items()):
        parts = task.removeprefix("RiskBlackjack-ConfirmV3-").removesuffix("-v0").rsplit("-", 1)
        row = {
            "task": task,
            "regime": parts[0],
            "risk_profile": parts[1],
        }
        for baseline, deltas in baseline_deltas.items():
            row[f"mean_delta_vs_{baseline}"] = sum(deltas) / len(deltas)
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/frontier_confirmation_v3_protocol.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.seed is not None and args.combine:
        parser.error("--seed and --combine are mutually exclusive")

    protocol_path = Path(args.protocol)
    protocol, tasks, cache_paths = load_and_validate_protocol(protocol_path)
    train_tasks = [
        task
        for suite in protocol["method"]["train_suites"]
        for task in benchmark_tasks(suite)
    ]
    scores_by_task = merge_scores(cache_paths)
    raw_params = dict(protocol["method"]["selected_params"])
    raw_params["comparison_policies"] = tuple(raw_params["comparison_policies"])
    params = LCBSelectorParams(**raw_params)
    selector_name = protocol["evaluation"]["reference_policy"]
    selector = policy_from_scores(
        tasks=train_tasks,
        scores_by_task=scores_by_task,
        params=params,
        name=selector_name,
    )
    selected = [
        {
            "task": task.name,
            "selected_policy": selector.selected_policy_name(task),
        }
        for task in tasks
    ]
    print(f"protocol_sha256={sha256_file(protocol_path)}", flush=True)
    print(f"source_manifest_sha256={source_manifest_sha256()}", flush=True)
    print(f"suite_manifest_sha256={task_manifest_sha256(tasks)}", flush=True)
    print(f"selection_counts={dict(sorted(Counter(row['selected_policy'] for row in selected).items()))}", flush=True)
    if args.dry_run:
        print("dry_run_complete=true", flush=True)
        return

    output_dir = Path(protocol["evaluation"]["out_dir"])
    result_paths = (
        output_dir / "seed_task_scores.csv",
        output_dir / "summary.json",
    )
    if any(path.exists() for path in result_paths):
        raise RuntimeError(f"refusing to overwrite an existing confirmation result in {output_dir}")

    policy_names = protocol["evaluation"]["policies"]
    locked_seeds = [int(seed) for seed in protocol["evaluation"]["seeds"]]
    if args.seed is not None:
        seed = int(args.seed)
        if seed not in locked_seeds:
            raise RuntimeError(f"seed {seed} is not in the locked protocol")
        checkpoint = output_dir / f"seed_{seed}_task_scores.csv"
        if checkpoint.exists():
            raise RuntimeError(f"refusing to overwrite existing seed checkpoint: {checkpoint}")
        candidate_lookup = family_candidate_lookup()
        eval_policies = [selector]
        for policy_name in policy_names:
            if policy_name == selector_name:
                continue
            if policy_name not in candidate_lookup:
                raise RuntimeError(f"unknown locked evaluation policy: {policy_name}")
            eval_policies.append(candidate_lookup[policy_name])
        print(f"starting_seed={seed}", flush=True)
        _episodes, summaries = run_benchmark(
            tasks=tasks,
            policies=eval_policies,
            episodes=int(protocol["evaluation"]["episodes_per_task_seed"]),
            seed=seed,
            hand_depth=int(protocol["evaluation"]["hand_depth"]),
        )
        seed_rows = summarize_seed(seed, summaries)
        write_csv(checkpoint, seed_rows)
        write_json(
            output_dir / f"seed_{seed}_metadata.json",
            {
                "seed": seed,
                "protocol_sha256": sha256_file(protocol_path),
                "source_manifest_sha256": source_manifest_sha256(),
                "suite_manifest_sha256": task_manifest_sha256(tasks),
                "row_count": len(seed_rows),
                "checkpoint_sha256": sha256_file(checkpoint),
            },
        )
        print(f"completed_seed={seed}", flush=True)
        print(f"checkpoint={checkpoint}", flush=True)
        return

    if not args.combine:
        parser.error("choose --dry-run, --seed SEED, or --combine")

    expected_tasks = {task.name for task in tasks}
    expected_policies = set(policy_names)
    expected_rows_per_seed = len(expected_tasks) * len(expected_policies)
    rows = []
    for seed in locked_seeds:
        checkpoint = output_dir / f"seed_{seed}_task_scores.csv"
        if not checkpoint.exists():
            raise RuntimeError(f"missing locked seed checkpoint: {checkpoint}")
        with checkpoint.open(encoding="utf-8", newline="") as file:
            seed_rows = list(csv.DictReader(file))
        if len(seed_rows) != expected_rows_per_seed:
            raise RuntimeError(
                f"checkpoint {checkpoint} has {len(seed_rows)} rows; "
                f"expected {expected_rows_per_seed}"
            )
        if {row["task"] for row in seed_rows} != expected_tasks:
            raise RuntimeError(f"task coverage changed in checkpoint: {checkpoint}")
        if {row["policy"] for row in seed_rows} != expected_policies:
            raise RuntimeError(f"policy coverage changed in checkpoint: {checkpoint}")
        if {int(row["seed"]) for row in seed_rows} != {seed}:
            raise RuntimeError(f"seed label changed in checkpoint: {checkpoint}")
        for row in seed_rows:
            row["seed"] = int(row["seed"])
            row["score"] = float(row["score"])
        rows.extend(seed_rows)

    primary_baseline = protocol["inference"]["primary_baseline"]
    secondary_baseline = protocol["inference"]["secondary_baseline"]
    report_kwargs = {
        "rows": rows,
        "reference_policy": selector_name,
        "score_field": protocol["inference"]["score_field"],
        "bootstrap_samples": int(protocol["inference"]["bootstrap_samples"]),
        "randomization_samples": int(protocol["inference"]["randomization_samples"]),
        "seed": int(protocol["inference"]["random_seed"]),
        "unit": protocol["inference"]["unit"],
    }
    primary_report = paired_score_report(
        baseline_policy=primary_baseline,
        **report_kwargs,
    )
    secondary_report = paired_score_report(
        baseline_policy=secondary_baseline,
        **report_kwargs,
    )
    cell_sensitivity = [
        paired_score_report(
            rows=rows,
            reference_policy=selector_name,
            baseline_policy=baseline,
            score_field=protocol["inference"]["score_field"],
            bootstrap_samples=int(protocol["inference"]["bootstrap_samples"]),
            randomization_samples=int(protocol["inference"]["randomization_samples"]),
            seed=int(protocol["inference"]["random_seed"]),
            unit="task_seed",
        )
        for baseline in (primary_baseline, secondary_baseline)
    ]
    aggregate = aggregate_seed_scores(rows)
    conventional_paired = paired_policy_deltas(rows, reference_policy=selector_name)
    per_task = task_deltas(rows, selector_name, [primary_baseline, secondary_baseline])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "seed_task_scores.csv", rows)
    write_csv(output_dir / "aggregate_scores.csv", aggregate)
    write_csv(output_dir / "selected_policies.csv", selected)
    write_csv(output_dir / "task_mean_deltas.csv", per_task)
    write_csv(output_dir / "paired_deltas_conventional.csv", conventional_paired)
    write_csv(output_dir / "task_level_inference.csv", [primary_report, secondary_report])
    write_csv(output_dir / "task_seed_sensitivity.csv", cell_sensitivity)
    summary = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "suite_manifest_sha256": task_manifest_sha256(tasks),
        "selected_policies": selected,
        "aggregate_scores": aggregate,
        "primary_task_level_inference": primary_report,
        "secondary_task_level_inference": secondary_report,
        "task_seed_sensitivity": cell_sensitivity,
        "retuned_after_confirmation": False,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"primary_task_level_inference={primary_report}", flush=True)
    print(f"secondary_task_level_inference={secondary_report}", flush=True)
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
