"""Diagnose promotion calibration and oracle headroom on an inspected suite.

This script is exploratory by construction.  It reads an already evaluated
suite and must not be used to relabel that suite as confirmatory evidence for a
subsequently revised selector.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from math import exp, sqrt
from pathlib import Path

from risk_shift_bench.envs import benchmark_tasks
from risk_shift_bench.lcb_selector import (
    LCBSelectorParams,
    feature_distance,
    policy_from_scores,
    task_features,
)
from risk_shift_bench.reporting import write_json

from experiments.frozen_confirmation_v3 import merge_scores


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def task_policy_means(path: Path) -> dict[str, dict[str, float]]:
    cells: dict[tuple[str, str], list[float]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            cells.setdefault((row["task"], row["policy"]), []).append(float(row["score"]))
    means: dict[str, dict[str, float]] = {}
    for (task, policy), values in cells.items():
        means.setdefault(task, {})[policy] = sum(values) / len(values)
    return means


def neighbor_diagnostics(selector, task, policy_name: str, baseline: str) -> dict:
    features = task_features(task, selector.params)
    neighbors = []
    for profile in selector.profiles:
        if policy_name not in profile.policy_scores or baseline not in profile.policy_scores:
            continue
        distance = feature_distance(features, profile.features)
        delta = profile.policy_scores[policy_name] - profile.policy_scores[baseline]
        neighbors.append((distance, delta, profile.task))
    neighbors.sort(key=lambda item: item[0])
    neighbors = neighbors[: max(1, selector.params.k)]
    if not neighbors:
        return {
            "predicted_lcb": float("-inf"),
            "predicted_mean_delta": float("-inf"),
            "support_radius": float("inf"),
            "neighbor_min_delta": float("-inf"),
            "neighbor_max_delta": float("-inf"),
            "neighbor_positive_fraction": 0.0,
            "effective_n": 0.0,
            "neighbors": "",
        }
    weights = [
        exp(-distance / max(selector.params.temperature, 1e-9))
        for distance, _delta, _task in neighbors
    ]
    total_weight = sum(weights)
    deltas = [delta for _distance, delta, _task in neighbors]
    mean_delta = sum(weight * delta for weight, delta in zip(weights, deltas)) / max(total_weight, 1e-12)
    mean_square = sum(weight * delta * delta for weight, delta in zip(weights, deltas)) / max(total_weight, 1e-12)
    variance = max(0.0, mean_square - mean_delta * mean_delta)
    effective_n = total_weight * total_weight / max(sum(weight * weight for weight in weights), 1e-12)
    lower_bound = mean_delta - selector.params.lcb_scale * sqrt(variance / max(effective_n, 1.0))
    return {
        "predicted_lcb": lower_bound,
        "predicted_mean_delta": mean_delta,
        "support_radius": neighbors[-1][0],
        "neighbor_min_delta": min(deltas),
        "neighbor_max_delta": max(deltas),
        "neighbor_positive_fraction": sum(delta > 0.0 for delta in deltas) / len(deltas),
        "effective_n": effective_n,
        "neighbors": ";".join(f"{name}|d={distance:.4f}|delta={delta:.4f}" for distance, delta, name in neighbors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/frontier_confirmation_v3_protocol.json")
    parser.add_argument(
        "--results",
        default="artifacts/frontier_confirmation_v3_frozen_5seed_100ep_v1/seed_task_scores.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/frontier_confirmation_v3_failure_analysis_v1",
    )
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    with protocol_path.open(encoding="utf-8") as file:
        protocol = json.load(file)
    train_tasks = [
        task
        for suite in protocol["method"]["train_suites"]
        for task in benchmark_tasks(suite)
    ]
    cache_paths = [Path(item["path"]) for item in protocol["method"]["score_caches"]]
    scores_by_task = merge_scores(cache_paths)
    raw_params = dict(protocol["method"]["selected_params"])
    raw_params["comparison_policies"] = tuple(raw_params["comparison_policies"])
    params = LCBSelectorParams(**raw_params)
    selector_name = protocol["evaluation"]["reference_policy"]
    selector = policy_from_scores(train_tasks, scores_by_task, params, name=selector_name)

    observed = task_policy_means(Path(args.results))
    fallback = params.fallback_policy
    rows = []
    for task in benchmark_tasks(protocol["suite"]["name"]):
        selected = selector.selected_policy_name(task)
        values = observed[task.name]
        selected_score = values[selector_name]
        fallback_score = values[fallback]
        # The suite measured both locked baselines on every task and measured
        # the selected delegate through the selector trajectory.
        measured_candidates = {
            fallback: fallback_score,
            protocol["inference"]["secondary_baseline"]: values[protocol["inference"]["secondary_baseline"]],
            selected: selected_score,
        }
        oracle_policy, oracle_score = max(measured_candidates.items(), key=lambda item: (item[1], item[0]))
        diagnostic = (
            neighbor_diagnostics(selector, task, selected, fallback)
            if selected != fallback
            else {
                "predicted_lcb": 0.0,
                "predicted_mean_delta": 0.0,
                "support_radius": 0.0,
                "neighbor_min_delta": 0.0,
                "neighbor_max_delta": 0.0,
                "neighbor_positive_fraction": 1.0,
                "effective_n": 0.0,
                "neighbors": "",
            }
        )
        rows.append(
            {
                "task": task.name,
                "selected_policy": selected,
                "promoted": selected != fallback,
                **diagnostic,
                "realized_delta_vs_fallback": selected_score - fallback_score,
                "restricted_oracle_policy": oracle_policy,
                "restricted_oracle_score": oracle_score,
                "restricted_oracle_gap": oracle_score - selected_score,
            }
        )

    promoted = [row for row in rows if row["promoted"]]
    harmful = [row for row in promoted if row["realized_delta_vs_fallback"] < 0.0]
    helpful = [row for row in promoted if row["realized_delta_vs_fallback"] > 0.0]
    oracle_gaps = [float(row["restricted_oracle_gap"]) for row in rows]
    summary = {
        "status": "exploratory_post_outcome_analysis",
        "protocol": str(protocol_path),
        "selector_params": asdict(params),
        "n_tasks": len(rows),
        "n_promotions": len(promoted),
        "n_helpful_promotions": len(helpful),
        "n_harmful_promotions": len(harmful),
        "promotion_success_rate": len(helpful) / len(promoted) if promoted else 0.0,
        "mean_promotion_delta": (
            sum(float(row["realized_delta_vs_fallback"]) for row in promoted) / len(promoted)
            if promoted
            else 0.0
        ),
        "worst_promotion_delta": min(
            (float(row["realized_delta_vs_fallback"]) for row in promoted),
            default=0.0,
        ),
        "mean_restricted_oracle_gap": sum(oracle_gaps) / len(oracle_gaps),
        "restricted_oracle_policy_counts": dict(Counter(row["restricted_oracle_policy"] for row in rows)),
        "interpretation": (
            "Exploratory only. The restricted oracle includes the two policies measured on every task "
            "and the selected delegate; it is not an oracle over the full policy library."
        ),
    }
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "route_diagnostics.csv", rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
