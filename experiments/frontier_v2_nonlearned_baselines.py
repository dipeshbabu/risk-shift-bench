"""Execute and audit the nonlearned competitive references for v2.

The references in this module use calibration tasks only.  Confirmation tasks
remain ineligible.  In particular, the knapsack reference is a clairvoyant
upper bound, not a deployable policy, and the v1 router is transferred without
refitting or access to any v2 outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any

from experiments.conformal_router import (
    ConformalAdvantageRouter,
    RouterParams,
    build_profiles,
)
from experiments.external_study_design import (
    POLICY_LIBRARIES as V1_POLICY_LIBRARIES,
    domain_tasks as v1_domain_tasks,
)
from experiments.frontier_v2_baseline_design import (
    COMPETITIVE_BASELINES,
    baseline_design_summary,
)
from experiments.frontier_v2_external_adapters import (
    V2EpisodeOutcome,
    _activate_verified_source,
    bounded_score,
    outcome_rows,
    run_v2_development_task,
    summarize_v2_outcomes,
)
from experiments.frontier_v2_external_design import (
    CODEBASE_LOCKS,
    V2ExternalTask,
    all_tasks,
    canonical_episode_seed_base,
    canonical_sha256,
    domain_tasks,
    expected_episode_seeds,
    task_manifest_sha256,
    task_sha256,
)
from experiments.frontier_v2_source_audit import (
    SOURCE_DIRECTORIES,
    audit_codebase_source,
)


NONLEARNED_BASELINES = tuple(
    baseline
    for baseline in COMPETITIVE_BASELINES
    if baseline.kind != "learned_policy"
)
NONLEARNED_IMPLEMENTATION_FILES = (
    "experiments/frontier_v2_nonlearned_baselines.py",
    "experiments/conformal_router.py",
    "experiments/external_study_design.py",
)
V1_FROZENLAKE_INPUTS = (
    "gymnasium_frozenlake/development/aggregate_scores.csv",
    "gymnasium_frozenlake/calibration/aggregate_scores.csv",
)
V1_ROUTER_REPORT = "gymnasium_frozenlake/router_report.json"
EPISODES_PER_CALIBRATION_TASK = 100


def file_sha256(path: Path, *, canonical_newlines: bool = False) -> str:
    content = path.read_bytes()
    if canonical_newlines:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def nonlearned_implementation_sha256(
    repository_root: Path | None = None,
) -> str:
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root
    )
    digest = hashlib.sha256()
    for relative in NONLEARNED_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"nonlearned implementation file is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def _baseline(identifier: str):
    matches = [
        baseline
        for baseline in NONLEARNED_BASELINES
        if baseline.identifier == identifier
    ]
    if len(matches) != 1:
        raise RuntimeError(f"unknown nonlearned baseline identifier: {identifier}")
    return matches[0]


def _tabular_environment(task: V2ExternalTask):
    import gymnasium as gym

    parameters = task.parameter_dict()
    kwargs: dict[str, Any]
    if task.domain == "gymnasium_frozenlake":
        from gymnasium.envs.toy_text.frozen_lake import generate_random_map

        description = generate_random_map(
            size=int(parameters["map_size"]),
            p=float(parameters["frozen_probability"]),
            seed=int(parameters["map_seed"]),
        )
        kwargs = {
            "desc": description,
            "is_slippery": bool(parameters["is_slippery"]),
            "success_rate": float(parameters["success_rate"]),
        }
    elif task.domain == "gymnasium_cliffwalking":
        kwargs = {"is_slippery": bool(parameters["is_slippery"])}
    elif task.domain == "gymnasium_taxi":
        kwargs = {
            "is_rainy": bool(parameters["is_rainy"]),
            "rainy_probability": float(parameters["rainy_probability"]),
            "fickle_passenger": bool(parameters["fickle_passenger"]),
            "fickle_probability": float(parameters["fickle_probability"]),
        }
    else:
        raise KeyError(task.domain)
    return gym.make(
        task.environment_id,
        **kwargs,
        max_episode_steps=int(parameters["max_steps"]),
    )


def _transition_cost(domain: str, reward: float) -> int:
    return int(
        (domain == "gymnasium_cliffwalking" and reward <= -100.0)
        or (domain == "gymnasium_taxi" and reward <= -10.0)
    )


def _transition_success(domain: str, reward: float, terminated: bool) -> bool:
    if domain == "gymnasium_frozenlake":
        return reward > 0.0
    if domain == "gymnasium_cliffwalking":
        return terminated
    if domain == "gymnasium_taxi":
        return reward >= 20.0
    raise KeyError(domain)


def finite_horizon_score_oracle(
    transitions,
    *,
    domain: str,
    horizon: int,
):
    """Return an optimal time/state/cost policy for the frozen bounded score."""

    import numpy as np

    if horizon <= 0:
        raise ValueError("oracle horizon must be positive")
    state_count = len(transitions)
    if set(int(state) for state in transitions) != set(range(state_count)):
        raise ValueError("tabular oracle requires contiguous integer states")
    cost_cap = 10
    next_values = np.zeros((state_count, cost_cap + 1), dtype=np.float64)
    actions = np.zeros((horizon, state_count, cost_cap + 1), dtype=np.int16)
    cost_states = np.arange(cost_cap + 1, dtype=np.int16)
    for time_index in range(horizon - 1, -1, -1):
        current_values = np.zeros_like(next_values)
        for state in range(state_count):
            state_actions = transitions[state]
            best_values = np.full(cost_cap + 1, -math.inf, dtype=np.float64)
            best_actions = np.zeros(cost_cap + 1, dtype=np.int16)
            for action in sorted(state_actions):
                action_values = np.zeros(cost_cap + 1, dtype=np.float64)
                for probability, next_state, reward, terminated in state_actions[action]:
                    numeric_reward = float(reward)
                    next_costs = np.minimum(
                        cost_cap,
                        cost_states + _transition_cost(domain, numeric_reward),
                    )
                    if terminated:
                        success = _transition_success(
                            domain, numeric_reward, bool(terminated)
                        )
                        continuation = np.clip(
                            float(success)
                            * (1.0 - 0.5 * (time_index + 1) / horizon)
                            - 0.1 * next_costs,
                            0.0,
                            1.0,
                        )
                    elif time_index + 1 >= horizon:
                        continuation = np.zeros(cost_cap + 1, dtype=np.float64)
                    else:
                        continuation = next_values[int(next_state), next_costs]
                    action_values += float(probability) * continuation
                improved = action_values > best_values + 1e-15
                best_values[improved] = action_values[improved]
                best_actions[improved] = int(action)
            current_values[state] = best_values
            actions[time_index, state] = best_actions
        next_values = current_values
    return actions


@lru_cache(maxsize=12)
def _cached_tabular_oracle_actions(task: V2ExternalTask, source_root: str):
    source = Path(source_root) / SOURCE_DIRECTORIES["gymnasium"]
    _activate_verified_source(source, "gymnasium")
    environment = _tabular_environment(task)
    try:
        return finite_horizon_score_oracle(
            environment.unwrapped.P,
            domain=task.domain,
            horizon=int(task.parameter_dict()["max_steps"]),
        )
    finally:
        environment.close()


def _run_tabular_oracle(
    task: V2ExternalTask,
    *,
    episodes: int,
    seed_base: int,
    source_root: Path,
) -> tuple[list[V2EpisodeOutcome], dict]:
    source = source_root / SOURCE_DIRECTORIES["gymnasium"]
    _activate_verified_source(source, "gymnasium")
    environment = _tabular_environment(task)
    horizon = int(task.parameter_dict()["max_steps"])
    try:
        action_table = _cached_tabular_oracle_actions(
            task,
            str(source_root.resolve()),
        )
        rows = []
        seeds = expected_episode_seeds(
            task, episodes=episodes, seed_base=seed_base
        )
        for episode, seed in enumerate(seeds):
            observation, _info = environment.reset(seed=seed)
            raw_return = 0.0
            cost = 0
            success = False
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = int(
                    action_table[steps, int(observation), min(10, cost)]
                )
                observation, reward, terminated, truncated, _info = environment.step(
                    action
                )
                numeric_reward = float(reward)
                raw_return += numeric_reward
                cost += _transition_cost(task.domain, numeric_reward)
                success = success or _transition_success(
                    task.domain, numeric_reward, bool(terminated)
                )
                steps += 1
            score = bounded_score(
                float(success) * (1.0 - 0.5 * steps / horizon) - 0.1 * cost
            )
            rows.append(
                V2EpisodeOutcome(
                    domain=task.domain,
                    task=task.name,
                    policy="tabular_oracle",
                    episode=episode,
                    seed=seed,
                    score=score,
                    raw_utility=raw_return,
                    raw_return=raw_return,
                    cost=float(cost),
                    failure=not success,
                    steps=steps,
                    successes=int(success),
                )
            )
    finally:
        environment.close()
    return rows, {
        "reference_type": "finite_horizon_true_model_score_oracle",
        "horizon": horizon,
        "cost_state_cap": 10,
        "tie_break": "smallest action index",
    }


def fractional_knapsack_upper_bound(
    item_weights: list[float],
    item_values: list[float],
    *,
    capacity: float,
) -> float:
    if capacity < 0.0 or len(item_weights) != len(item_values):
        raise ValueError("invalid fractional-knapsack instance")
    if any(weight <= 0.0 for weight in item_weights):
        raise ValueError("fractional-knapsack weights must be positive")
    remaining = float(capacity)
    optimum = 0.0
    items = sorted(
        zip(item_weights, item_values, strict=True),
        key=lambda item: (item[1] / item[0], item[1], -item[0]),
        reverse=True,
    )
    for weight, value in items:
        if remaining <= 0.0:
            break
        fraction = min(1.0, remaining / float(weight))
        optimum += fraction * float(value)
        remaining -= fraction * float(weight)
    return optimum


def _run_fractional_knapsack(
    task: V2ExternalTask,
    *,
    episodes: int,
    seed_base: int,
    source_root: Path,
) -> tuple[list[V2EpisodeOutcome], dict]:
    source = source_root / SOURCE_DIRECTORIES["or_gym"]
    _activate_verified_source(source, "or_gym")
    import numpy as np

    from experiments.external_domain_adapters import _knapsack_catalog
    from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv

    parameters = task.parameter_dict()
    capacity = int(parameters["capacity"])
    horizon = int(parameters["horizon"])
    weights, values = _knapsack_catalog(task)
    normalizer = capacity * max(
        value / weight for weight, value in zip(weights, values, strict=True)
    )
    rows = []
    sequence_hashes = []
    seeds = expected_episode_seeds(task, episodes=episodes, seed_base=seed_base)
    for episode, seed in enumerate(seeds):
        np.random.seed(seed)
        environment = OnlineKnapsackEnv()
        environment.max_weight = capacity
        environment.step_limit = horizon
        environment.item_weights = np.asarray(weights, dtype=np.int32)
        environment.item_values = np.asarray(values, dtype=np.int32)
        environment.item_limits_init = np.ones(len(weights), dtype=np.int32)
        environment.item_probs = (
            environment.item_limits_init / environment.item_limits_init.sum()
        )
        np.random.seed(seed)
        environment.reset()
        episode_weights = []
        episode_values = []
        item_indices = []
        done = False
        steps = 0
        while not done and steps < horizon:
            item = int(environment.current_item)
            item_indices.append(item)
            episode_weights.append(float(environment.item_weights[item]))
            episode_values.append(float(environment.item_values[item]))
            _observation, _reward, done, _info = environment.step(0)
            steps += 1
        optimum = fractional_knapsack_upper_bound(
            episode_weights,
            episode_values,
            capacity=capacity,
        )
        sequence_hashes.append(canonical_sha256(item_indices))
        rows.append(
            V2EpisodeOutcome(
                domain=task.domain,
                task=task.name,
                policy="fractional_oracle",
                episode=episode,
                seed=seed,
                score=bounded_score(optimum / normalizer),
                raw_utility=optimum,
                raw_return=optimum,
                cost=0.0,
                failure=False,
                steps=steps,
                successes=0,
            )
        )
    return rows, {
        "reference_type": "clairvoyant_episode_fractional_upper_bound",
        "deployable_policy": False,
        "normalizer": normalizer,
        "episode_item_sequence_sha256": sequence_hashes,
    }


def poisson_quantile(probability: float, mean: float) -> int:
    if not 0.0 < probability < 1.0 or mean < 0.0:
        raise ValueError("invalid Poisson quantile arguments")
    if mean == 0.0:
        return 0
    mass = math.exp(-mean)
    cumulative = mass
    value = 0
    while cumulative < probability:
        value += 1
        mass *= mean / value
        cumulative += mass
        if value > 100_000:
            raise RuntimeError("Poisson quantile failed to converge")
    return value


def newsvendor_base_stock_levels(environment, *, demand_mean: float) -> tuple[int, ...]:
    levels = []
    service_levels = []
    for index, lead_time in enumerate(environment.lead_time):
        margin = max(
            0.0,
            float(environment.unit_price[index])
            - float(environment.unit_cost[index]),
        )
        shortage = float(environment.demand_cost[index])
        holding = float(environment.holding_cost[index])
        critical_ratio = (margin + shortage) / (
            margin + shortage + holding
        )
        service_levels.append(critical_ratio)
        exposure_mean = demand_mean * (int(lead_time) + 1)
        levels.append(poisson_quantile(critical_ratio, exposure_mean))
    if any(left > right for left, right in zip(levels, levels[1:], strict=False)):
        raise RuntimeError("newsvendor echelon levels must be nondecreasing")
    return tuple(levels), tuple(service_levels)


def _run_newsvendor(
    task: V2ExternalTask,
    *,
    episodes: int,
    seed_base: int,
    source_root: Path,
) -> tuple[list[V2EpisodeOutcome], dict]:
    source = source_root / SOURCE_DIRECTORIES["or_gym"]
    _activate_verified_source(source, "or_gym")
    import numpy as np
    from or_gym.envs.supply_chain.inventory_management import (
        InvManagementBacklogEnv,
        InvManagementLostSalesEnv,
    )

    parameters = task.parameter_dict()
    environment_class = (
        InvManagementBacklogEnv
        if bool(parameters["backlog"])
        else InvManagementLostSalesEnv
    )
    demand_mean = float(parameters["demand_mean"])
    lead_times = [
        max(1, round(base * float(parameters["lead_time_scale"])))
        for base in (3, 5, 10)
    ]
    rows = []
    levels_seen = None
    service_levels_seen = None
    seeds = expected_episode_seeds(task, episodes=episodes, seed_base=seed_base)
    for episode, seed in enumerate(seeds):
        np.random.seed(seed)
        environment = environment_class(
            periods=int(parameters["periods"]),
            dist=1,
            dist_param={"mu": demand_mean},
            seed_int=seed,
            L=lead_times,
        )
        environment.reset()
        levels, service_levels = newsvendor_base_stock_levels(
            environment, demand_mean=demand_mean
        )
        levels_seen = levels
        service_levels_seen = service_levels
        raw_return = 0.0
        done = False
        steps = 0
        while not done:
            environment._update_base_stock_policy_state()
            inventory_position = environment.state
            action = tuple(
                max(0.0, min(float(capacity), level - float(position)))
                for level, position, capacity in zip(
                    levels,
                    inventory_position,
                    environment.supply_capacity,
                    strict=True,
                )
            )
            _observation, reward, done, _info = environment.step(
                np.asarray(action, dtype=np.int32)
            )
            raw_return += float(reward)
            steps += 1
        unmet = float(environment.B.sum() + environment.LS.sum())
        lower = float(parameters["profit_lower"])
        upper = float(parameters["profit_upper"])
        rows.append(
            V2EpisodeOutcome(
                domain=task.domain,
                task=task.name,
                policy="newsvendor_base_stock",
                episode=episode,
                seed=seed,
                score=bounded_score((raw_return - lower) / (upper - lower)),
                raw_utility=raw_return,
                raw_return=raw_return,
                cost=unmet,
                failure=raw_return < 0.0,
                steps=steps,
                successes=0,
            )
        )
    return rows, {
        "reference_type": "lead_time_poisson_newsvendor_base_stock",
        "base_stock_levels": list(levels_seen or ()),
        "critical_ratios": list(service_levels_seen or ()),
        "lead_times": lead_times,
    }


def _read_v1_scores(
    path: Path,
    *,
    expected_tasks: list[str],
    policies: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        scores.setdefault(row["task"], {})[row["policy"]] = float(row["score"])
    if set(scores) != set(expected_tasks):
        raise RuntimeError("v1 router input task coverage changed")
    if any(set(task_scores) != set(policies) for task_scores in scores.values()):
        raise RuntimeError("v1 router input policy coverage changed")
    return scores


def _build_v1_frozenlake_router(
    *,
    v1_development_root: Path,
    v1_router_root: Path,
):
    domain = "gymnasium_frozenlake"
    library = V1_POLICY_LIBRARIES[domain]
    policies = (library.fallback, *library.candidates)
    fit_tasks = v1_domain_tasks(domain, "development")
    calibration_tasks = v1_domain_tasks(domain, "calibration")
    fit_path = v1_development_root / V1_FROZENLAKE_INPUTS[0]
    calibration_path = v1_development_root / V1_FROZENLAKE_INPUTS[1]
    fit_scores = _read_v1_scores(
        fit_path,
        expected_tasks=[task.name for task in fit_tasks],
        policies=policies,
    )
    calibration_scores = _read_v1_scores(
        calibration_path,
        expected_tasks=[task.name for task in calibration_tasks],
        policies=policies,
    )
    params = RouterParams(
        k=5,
        temperature=0.75,
        alpha=0.10,
        margin=0.0,
        min_fit_evidence=3,
        min_calibration_tasks=5,
        screen_min_mean_advantage=0.0,
        max_screened_candidates=1,
        fallback_policy=library.fallback,
    )
    router = ConformalAdvantageRouter(
        fit_profiles=build_profiles(fit_tasks, fit_scores, lambda task: task.features),
        calibration_profiles=build_profiles(
            calibration_tasks,
            calibration_scores,
            lambda task: task.features,
        ),
        candidate_policies=library.candidates,
        params=params,
        feature_fn=lambda task: task.features,
    )
    report_path = v1_router_root / V1_ROUTER_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if canonical_sha256(router.report_dict()) != canonical_sha256(report["router"]):
        raise RuntimeError("reconstructed v1 FrozenLake router changed")
    inputs = [
        {
            "role": "v1_development_or_calibration_scores",
            "path": relative,
            "sha256": file_sha256(
                v1_development_root / relative, canonical_newlines=True
            ),
        }
        for relative in V1_FROZENLAKE_INPUTS
    ]
    inputs.append(
        {
            "role": "v1_frozen_router_report",
            "path": V1_ROUTER_REPORT,
            "sha256": file_sha256(report_path, canonical_newlines=True),
        }
    )
    return router, inputs


def _run_v1_fixed_router(
    task: V2ExternalTask,
    *,
    episodes: int,
    seed_base: int,
    source_root: Path,
    router,
) -> tuple[list[V2EpisodeOutcome], dict]:
    parameters = task.parameter_dict()
    v1_features = (
        int(parameters["map_size"]) / 8.0,
        float(bool(parameters["is_slippery"])),
        float(parameters["success_rate"]),
    )
    decision = router.proposal(replace(task, features=v1_features))
    selected_policy = decision.selected_policy
    rows = run_v2_development_task(
        task,
        selected_policy,
        episodes,
        seed_base,
        source_root,
    )
    relabeled = [replace(row, policy="v1_fixed_router") for row in rows]
    prediction = decision.prediction
    return relabeled, {
        "reference_type": "frozen_v1_fit_router_transfer",
        "refit_on_v2": False,
        "v2_outcomes_used_for_routing": False,
        "v1_feature_schema": [
            "map_size_over_8",
            "is_slippery",
            "success_rate",
        ],
        "v1_features": list(v1_features),
        "selected_policy": selected_policy,
        "promoted": decision.promoted,
        "reason": decision.reason,
        "predicted_advantage": (
            None if prediction is None else prediction.predicted_advantage
        ),
        "support_radius": (
            None if prediction is None else prediction.support_radius
        ),
    }


def _run_reference(
    baseline,
    task: V2ExternalTask,
    *,
    episodes: int,
    seed_base: int,
    source_root: Path,
    v1_router,
) -> tuple[list[V2EpisodeOutcome], dict]:
    if baseline.name == "tabular_oracle":
        return _run_tabular_oracle(
            task,
            episodes=episodes,
            seed_base=seed_base,
            source_root=source_root,
        )
    if baseline.name == "fractional_oracle":
        return _run_fractional_knapsack(
            task,
            episodes=episodes,
            seed_base=seed_base,
            source_root=source_root,
        )
    if baseline.name == "newsvendor_base_stock":
        return _run_newsvendor(
            task,
            episodes=episodes,
            seed_base=seed_base,
            source_root=source_root,
        )
    if baseline.name == "v1_fixed_router":
        if v1_router is None:
            raise RuntimeError("v1 router was not initialized")
        return _run_v1_fixed_router(
            task,
            episodes=episodes,
            seed_base=seed_base,
            source_root=source_root,
            router=v1_router,
        )
    raise KeyError(baseline.identifier)


def evaluate_nonlearned_baseline(
    identifier: str,
    *,
    output_root: Path,
    source_root: Path,
    v1_development_root: Path,
    v1_router_root: Path,
    episodes_per_task: int = EPISODES_PER_CALIBRATION_TASK,
) -> dict:
    baseline = _baseline(identifier)
    if episodes_per_task != EPISODES_PER_CALIBRATION_TASK:
        raise ValueError("nonlearned calibration episode count is frozen at 100")
    codebase = CODEBASE_LOCKS[
        {
            "gymnasium_frozenlake": "gymnasium",
            "gymnasium_cliffwalking": "gymnasium",
            "gymnasium_taxi": "gymnasium",
            "or_gym_online_knapsack": "or_gym",
            "or_gym_inventory_management": "or_gym",
        }[baseline.domain]
    ]
    source_directory = source_root / SOURCE_DIRECTORIES[codebase.name]
    source_audit = asdict(audit_codebase_source(source_directory, codebase.name))
    source_audit["source"] = SOURCE_DIRECTORIES[codebase.name]
    v1_router = None
    v1_inputs = []
    if baseline.name == "v1_fixed_router":
        v1_router, v1_inputs = _build_v1_frozenlake_router(
            v1_development_root=v1_development_root,
            v1_router_root=v1_router_root,
        )
    task_records = []
    for task in domain_tasks(baseline.domain, "calibration"):
        seed_base = canonical_episode_seed_base(task)
        rows, metadata = _run_reference(
            baseline,
            task,
            episodes=episodes_per_task,
            seed_base=seed_base,
            source_root=source_root,
            v1_router=v1_router,
        )
        repeated_rows, repeated_metadata = _run_reference(
            baseline,
            task,
            episodes=episodes_per_task,
            seed_base=seed_base,
            source_root=source_root,
            v1_router=v1_router,
        )
        if rows != repeated_rows or metadata != repeated_metadata:
            raise RuntimeError(f"nonlearned replay changed for {task.name}")
        serialized_rows = outcome_rows(rows)
        task_records.append(
            {
                "task": task.name,
                "task_sha256": task_sha256(task),
                "seed_base": seed_base,
                "canonical_seed_schedule": True,
                "summary": summarize_v2_outcomes(rows),
                "reference_metadata": metadata,
                "outcomes": serialized_rows,
                "outcome_sha256": canonical_sha256(serialized_rows),
                "deterministic_replay_exact": True,
            }
        )
    design = baseline_design_summary()
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-nonlearned-reference-v1",
        "scope": "Calibration execution only; no v2 confirmation task was reset.",
        "baseline_design_sha256": design["design_sha256"],
        "nonlearned_implementation_files": list(NONLEARNED_IMPLEMENTATION_FILES),
        "nonlearned_implementation_sha256": nonlearned_implementation_sha256(),
        "baseline_identifier": baseline.identifier,
        "baseline_spec": asdict(baseline),
        "calibration_manifest_sha256": task_manifest_sha256(
            all_tasks("calibration")
        ),
        "episodes_per_task": episodes_per_task,
        "environment_codebase_lock": asdict(codebase),
        "environment_source_audit": source_audit,
        "upstream_v1_inputs": v1_inputs,
        "tasks": task_records,
    }
    manifest = output_root / baseline.domain / baseline.name / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def audit_nonlearned_manifest(
    payload: dict,
    *,
    source_root: Path | None = None,
    v1_development_root: Path | None = None,
    v1_router_root: Path | None = None,
) -> dict:
    if payload.get("protocol_id") != "riskshiftbench-frontier-v2-nonlearned-reference-v1":
        raise RuntimeError("unexpected nonlearned-reference protocol")
    baseline = _baseline(str(payload.get("baseline_identifier", "")))
    design = baseline_design_summary()
    if payload.get("baseline_design_sha256") != design["design_sha256"]:
        raise RuntimeError("nonlearned baseline design hash changed")
    if canonical_sha256(payload.get("baseline_spec")) != canonical_sha256(
        asdict(baseline)
    ):
        raise RuntimeError("nonlearned baseline specification changed")
    if payload.get("nonlearned_implementation_files") != list(
        NONLEARNED_IMPLEMENTATION_FILES
    ):
        raise RuntimeError("nonlearned implementation file set changed")
    if payload.get("nonlearned_implementation_sha256") != nonlearned_implementation_sha256():
        raise RuntimeError("nonlearned implementation hash changed")
    if payload.get("calibration_manifest_sha256") != task_manifest_sha256(
        all_tasks("calibration")
    ):
        raise RuntimeError("nonlearned calibration manifest changed")
    if int(payload.get("episodes_per_task", -1)) != EPISODES_PER_CALIBRATION_TASK:
        raise RuntimeError("nonlearned episode count changed")
    expected_tasks = domain_tasks(baseline.domain, "calibration")
    records = payload.get("tasks")
    if not isinstance(records, list) or len(records) != len(expected_tasks):
        raise RuntimeError("nonlearned task coverage is incomplete")
    by_name = {record.get("task"): record for record in records}
    if set(by_name) != {task.name for task in expected_tasks}:
        raise RuntimeError("nonlearned task names changed")
    row_count = 0
    for task in expected_tasks:
        record = by_name[task.name]
        if record.get("task_sha256") != task_sha256(task):
            raise RuntimeError("nonlearned task hash changed")
        if record.get("seed_base") != canonical_episode_seed_base(task):
            raise RuntimeError("nonlearned seed base changed")
        if record.get("canonical_seed_schedule") is not True:
            raise RuntimeError("nonlearned seed schedule is not canonical")
        if record.get("deterministic_replay_exact") is not True:
            raise RuntimeError("nonlearned deterministic replay is incomplete")
        rows = record.get("outcomes")
        if not isinstance(rows, list) or len(rows) != EPISODES_PER_CALIBRATION_TASK:
            raise RuntimeError("nonlearned episode coverage is incomplete")
        if record.get("outcome_sha256") != canonical_sha256(rows):
            raise RuntimeError("nonlearned outcome digest changed")
        expected_seeds = expected_episode_seeds(
            task,
            episodes=EPISODES_PER_CALIBRATION_TASK,
            seed_base=canonical_episode_seed_base(task),
        )
        if tuple(int(row["seed"]) for row in rows) != expected_seeds:
            raise RuntimeError("nonlearned episode seeds changed")
        if any(
            row.get("task") != task.name
            or row.get("domain") != baseline.domain
            or row.get("policy") != baseline.name
            or not 0.0 <= float(row.get("score", math.nan)) <= 1.0
            for row in rows
        ):
            raise RuntimeError("nonlearned outcome row is invalid")
        scores = [float(row["score"]) for row in rows]
        if not math.isclose(
            float(record["summary"]["mean_score"]),
            fmean(scores),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("nonlearned task summary changed")
        row_count += len(rows)
    codebase_name = str(payload["environment_codebase_lock"]["name"])
    if canonical_sha256(payload["environment_codebase_lock"]) != canonical_sha256(
        asdict(CODEBASE_LOCKS[codebase_name])
    ):
        raise RuntimeError("nonlearned codebase lock changed")
    if source_root is not None:
        current = asdict(
            audit_codebase_source(
                source_root / SOURCE_DIRECTORIES[codebase_name], codebase_name
            )
        )
        current["source"] = SOURCE_DIRECTORIES[codebase_name]
        if canonical_sha256(payload.get("environment_source_audit")) != canonical_sha256(
            current
        ):
            raise RuntimeError("nonlearned environment source audit changed")
    v1_inputs = payload.get("upstream_v1_inputs")
    if baseline.name == "v1_fixed_router":
        if not isinstance(v1_inputs, list) or len(v1_inputs) != 3:
            raise RuntimeError("v1 router inputs are incomplete")
        if v1_development_root is not None and v1_router_root is not None:
            for item in v1_inputs:
                relative = PurePosixPath(str(item["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("v1 router input path is invalid")
                root = (
                    v1_router_root
                    if item["role"] == "v1_frozen_router_report"
                    else v1_development_root
                )
                path = root / relative
                if not path.is_file() or file_sha256(
                    path, canonical_newlines=True
                ) != item["sha256"]:
                    raise RuntimeError("v1 router input hash changed")
    elif v1_inputs != []:
        raise RuntimeError("unexpected v1 inputs on analytic reference")
    return {
        "baseline_identifier": baseline.identifier,
        "task_count": len(expected_tasks),
        "episode_rows": row_count,
        "deterministic_replay_exact": True,
        "source_tree_verified": source_root is not None,
        "v1_inputs_verified": baseline.name != "v1_fixed_router"
        or (v1_development_root is not None and v1_router_root is not None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        choices=tuple(baseline.identifier for baseline in NONLEARNED_BASELINES),
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baselines"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument(
        "--v1-development-root",
        type=Path,
        default=Path("artifacts/external_development_v1"),
    )
    parser.add_argument(
        "--v1-router-root",
        type=Path,
        default=Path("artifacts/external_router_lock_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_nonlearned_baseline(
        args.baseline,
        output_root=args.output_root,
        source_root=args.source_root,
        v1_development_root=args.v1_development_root,
        v1_router_root=args.v1_router_root,
    )
    audit = audit_nonlearned_manifest(
        payload,
        source_root=args.source_root,
        v1_development_root=args.v1_development_root,
        v1_router_root=args.v1_router_root,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
