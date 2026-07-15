"""Fit the conformal router on historical splits and audit inspected suites."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from experiments.conformal_router import ConformalAdvantageRouter, RouterParams, build_profiles
from experiments.frozen_confirmation_v3 import merge_scores as merge_blackjack_scores
from risk_shift_bench.envs import benchmark_tasks
from risk_shift_bench.lcb_selector import LCBSelectorParams, task_features as blackjack_task_features
from risk_shift_bench.portfolio_envs import portfolio_tasks
from risk_shift_bench.portfolio_lcb_selector import task_features as portfolio_task_features
from risk_shift_bench.reporting import write_json


BLACKJACK_FIT_SUITES = ("frontier_dev", "frontier_holdout", "frontier_audit")
BLACKJACK_CALIBRATION_SUITES = ("frontier_final_audit", "frontier_blind_audit")
BLACKJACK_CANDIDATES = (
    "adaptive_utility_default",
    "expected_value",
    "fixed_entropic_0.025",
    "fixed_oce_3",
    "learned_mixture_default",
    "regime_adaptive_ensemble",
    "signed_regime_learned_ensemble",
)
PORTFOLIO_FIT_SUITES = ("portfolio_dev",)
PORTFOLIO_CALIBRATION_SUITES = ("portfolio_holdout", "portfolio_audit")
PORTFOLIO_CANDIDATES = (
    "adaptive_utility_default",
    "balanced_50",
    "cash_only",
    "expected_value",
    "fixed_entropic_0.025",
    "fixed_oce_3",
    "signed_regime_learned_ensemble",
    "target_seeking_mean",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_aggregate_scores(path: Path) -> dict[str, dict[str, list[float]]]:
    scores: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] == "task":
                scores.setdefault(row["task"], {}).setdefault(row["policy"], []).append(float(row["mean_score"]))
    return scores


def merge_aggregate_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        for task, policy_scores in load_aggregate_scores(path).items():
            for policy, values in policy_scores.items():
                merged.setdefault(task, {}).setdefault(policy, []).extend(values)
    return {
        task: {policy: sum(values) / len(values) for policy, values in policy_scores.items()}
        for task, policy_scores in merged.items()
    }


def decision_row(task, router) -> dict:
    decision = router.decision(task)
    prediction = decision.prediction
    proposal = router.proposal(task)
    return {
        "task": task.name,
        "selected_policy": decision.selected_policy,
        "promoted": decision.promoted,
        "reason": decision.reason,
        "proposed_policy": proposal.selected_policy,
        "proposal_requires_pilot": proposal.promoted,
        "proposal_reason": proposal.reason,
        "predicted_advantage": prediction.predicted_advantage if prediction else "",
        "simultaneous_lower_bound": prediction.lower_bound if prediction else "",
        "support_radius": prediction.support_radius if prediction else "",
        "support_radius_limit": router.calibration.support_radius_limit,
        "conformal_correction": router.calibration.conformal_correction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/conformal_router_historical_analysis_v1")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    with Path("configs/frontier_confirmation_v3_protocol.json").open(encoding="utf-8") as file:
        protocol = json.load(file)
    blackjack_scores = merge_blackjack_scores(
        [Path(item["path"]) for item in protocol["method"]["score_caches"]]
    )
    blackjack_feature_params = LCBSelectorParams()

    def blackjack_feature_fn(task):
        return blackjack_task_features(task, blackjack_feature_params)
    blackjack_fit_tasks = [task for suite in BLACKJACK_FIT_SUITES for task in benchmark_tasks(suite)]
    blackjack_calibration_tasks = [
        task for suite in BLACKJACK_CALIBRATION_SUITES for task in benchmark_tasks(suite)
    ]
    blackjack_router = ConformalAdvantageRouter(
        fit_profiles=build_profiles(blackjack_fit_tasks, blackjack_scores, blackjack_feature_fn),
        calibration_profiles=build_profiles(
            blackjack_calibration_tasks,
            blackjack_scores,
            blackjack_feature_fn,
        ),
        candidate_policies=BLACKJACK_CANDIDATES,
        params=RouterParams(),
        feature_fn=blackjack_feature_fn,
    )
    blackjack_rows = [
        decision_row(task, blackjack_router)
        for task in benchmark_tasks("frontier_confirmation_audit_v3")
    ]

    portfolio_paths = [
        Path("artifacts/portfolio_benchmark") / suite / "aggregate_scores.csv"
        for suite in (*PORTFOLIO_FIT_SUITES, *PORTFOLIO_CALIBRATION_SUITES)
    ]
    portfolio_scores = merge_aggregate_scores(portfolio_paths)
    portfolio_fit_tasks = [task for suite in PORTFOLIO_FIT_SUITES for task in portfolio_tasks(suite)]
    portfolio_calibration_tasks = [
        task for suite in PORTFOLIO_CALIBRATION_SUITES for task in portfolio_tasks(suite)
    ]
    portfolio_router = ConformalAdvantageRouter(
        fit_profiles=build_profiles(portfolio_fit_tasks, portfolio_scores, portfolio_task_features),
        calibration_profiles=build_profiles(
            portfolio_calibration_tasks,
            portfolio_scores,
            portfolio_task_features,
        ),
        candidate_policies=PORTFOLIO_CANDIDATES,
        params=RouterParams(),
        feature_fn=portfolio_task_features,
    )
    portfolio_rows = [
        decision_row(task, portfolio_router)
        for task in portfolio_tasks("portfolio_confirmation")
    ]

    summary = {
        "status": "exploratory_post_outcome_method_development",
        "blackjack": {
            "fit_suites": BLACKJACK_FIT_SUITES,
            "calibration_suites": BLACKJACK_CALIBRATION_SUITES,
            "router": blackjack_router.report_dict(),
            "inspected_suite": "frontier_confirmation_audit_v3",
            "selection_counts": dict(Counter(row["selected_policy"] for row in blackjack_rows)),
        },
        "portfolio": {
            "fit_suites": PORTFOLIO_FIT_SUITES,
            "calibration_suites": PORTFOLIO_CALIBRATION_SUITES,
            "router": portfolio_router.report_dict(),
            "inspected_suite": "portfolio_confirmation",
            "selection_counts": dict(Counter(row["selected_policy"] for row in portfolio_rows)),
        },
        "confirmatory_status": (
            "Neither inspected suite is confirmatory for this method. New tasks and a pre-outcome lock are required."
        ),
    }
    write_csv(out_dir / "blackjack_v3_decisions.csv", blackjack_rows)
    write_csv(out_dir / "portfolio_confirmation_decisions.csv", portfolio_rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
