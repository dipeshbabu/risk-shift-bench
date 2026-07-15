"""Build the three fixed-domain routers used by the frontier extension."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.conformal_router import ConformalAdvantageRouter, RouterParams, build_profiles
from experiments.conformal_router_analysis import (
    BLACKJACK_CALIBRATION_SUITES,
    BLACKJACK_CANDIDATES,
    BLACKJACK_FIT_SUITES,
    PORTFOLIO_CALIBRATION_SUITES,
    PORTFOLIO_CANDIDATES,
    PORTFOLIO_FIT_SUITES,
)
from experiments.frozen_confirmation_v3 import merge_scores as merge_blackjack_scores
from experiments.inventory_domain import (
    inventory_calibration_tasks,
    inventory_development_tasks,
    inventory_policy_grid,
    inventory_task_features,
)
from risk_shift_bench.envs import benchmark_tasks
from risk_shift_bench.lcb_selector import LCBSelectorParams, task_features as blackjack_features
from risk_shift_bench.portfolio_envs import portfolio_tasks
from risk_shift_bench.portfolio_lcb_selector import task_features as portfolio_features


INVENTORY_CANDIDATES = tuple(
    policy.name for policy in inventory_policy_grid() if policy.name != "adaptive_base_stock"
)


def load_aggregate_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row["scope"] != "task":
                    continue
                merged.setdefault(row["task"], {}).setdefault(row["policy"], []).append(
                    float(row["mean_score"])
                )
    return {
        task: {policy: sum(values) / len(values) for policy, values in policies.items()}
        for task, policies in merged.items()
    }


def build_blackjack_router(params: RouterParams = RouterParams()):
    with Path("configs/frontier_confirmation_v3_protocol.json").open(encoding="utf-8") as file:
        protocol = json.load(file)
    scores = merge_blackjack_scores(
        [Path(item["path"]) for item in protocol["method"]["score_caches"]]
    )
    feature_params = LCBSelectorParams()

    def feature_fn(task):
        return blackjack_features(task, feature_params)

    fit_tasks = [task for suite in BLACKJACK_FIT_SUITES for task in benchmark_tasks(suite)]
    calibration_tasks = [
        task for suite in BLACKJACK_CALIBRATION_SUITES for task in benchmark_tasks(suite)
    ]
    return ConformalAdvantageRouter(
        fit_profiles=build_profiles(fit_tasks, scores, feature_fn),
        calibration_profiles=build_profiles(calibration_tasks, scores, feature_fn),
        candidate_policies=BLACKJACK_CANDIDATES,
        params=params,
        feature_fn=feature_fn,
    )


def build_portfolio_router(params: RouterParams = RouterParams()):
    suites = (*PORTFOLIO_FIT_SUITES, *PORTFOLIO_CALIBRATION_SUITES)
    scores = load_aggregate_scores(
        [Path("artifacts/portfolio_benchmark") / suite / "aggregate_scores.csv" for suite in suites]
    )
    fit_tasks = [task for suite in PORTFOLIO_FIT_SUITES for task in portfolio_tasks(suite)]
    calibration_tasks = [
        task for suite in PORTFOLIO_CALIBRATION_SUITES for task in portfolio_tasks(suite)
    ]
    return ConformalAdvantageRouter(
        fit_profiles=build_profiles(fit_tasks, scores, portfolio_features),
        calibration_profiles=build_profiles(calibration_tasks, scores, portfolio_features),
        candidate_policies=PORTFOLIO_CANDIDATES,
        params=params,
        feature_fn=portfolio_features,
    )


def build_inventory_router(params: RouterParams | None = None):
    inventory_params = params or RouterParams(fallback_policy="adaptive_base_stock")
    scores = load_aggregate_scores(
        [
            Path("artifacts/inventory_benchmark/inventory_dev/aggregate_scores.csv"),
            Path("artifacts/inventory_benchmark/inventory_calibration/aggregate_scores.csv"),
        ]
    )
    return ConformalAdvantageRouter(
        fit_profiles=build_profiles(inventory_development_tasks(), scores, inventory_task_features),
        calibration_profiles=build_profiles(
            inventory_calibration_tasks(),
            scores,
            inventory_task_features,
        ),
        candidate_policies=INVENTORY_CANDIDATES,
        params=inventory_params,
        feature_fn=inventory_task_features,
    )
