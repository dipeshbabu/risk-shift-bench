"""Adapters for independently maintained external environment implementations.

Imports are intentionally lazy because the three pinned environments require
different Python and Gymnasium versions.  Each domain is executed in its own
isolated environment while producing the same episode schema.
"""

from __future__ import annotations

import importlib.metadata
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from experiments.external_study_design import (
    ENVIRONMENT_LOCKS,
    RUNTIME_DEPENDENCIES,
    ExternalTask,
)


@dataclass(frozen=True)
class ExternalEpisodeOutcome:
    domain: str
    task: str
    policy: str
    episode: int
    seed: int
    utility: float
    raw_return: float
    cost: float
    failure: bool
    steps: int
    successes: int
    resource_residual: float


def lower_tail_mean(values: list[float], fraction: float = 0.05) -> float:
    if not values:
        raise ValueError("lower_tail_mean requires values")
    count = max(1, math.ceil(len(values) * fraction))
    return mean(sorted(values)[:count])


def summarize_outcomes(outcomes: list[ExternalEpisodeOutcome]) -> dict:
    if not outcomes:
        raise ValueError("cannot summarize empty external outcomes")
    utilities = [row.utility for row in outcomes]
    return {
        "domain": outcomes[0].domain,
        "task": outcomes[0].task,
        "policy": outcomes[0].policy,
        "episodes": len(outcomes),
        "mean_utility": mean(utilities),
        "cvar_5_utility": lower_tail_mean(utilities),
        "failure_probability": mean(float(row.failure) for row in outcomes),
        "mean_cost": mean(row.cost for row in outcomes),
        "mean_steps": mean(row.steps for row in outcomes),
        "mean_successes": mean(row.successes for row in outcomes),
        "score": mean(utilities) + 0.5 * lower_tail_mean(utilities),
    }


def outcome_rows(outcomes: list[ExternalEpisodeOutcome]) -> list[dict]:
    return [asdict(row) for row in outcomes]


def _require_distribution(distribution: str, expected_version: str) -> None:
    observed = importlib.metadata.version(distribution)
    if observed != expected_version:
        raise RuntimeError(
            f"{distribution} version changed: expected {expected_version}, found {observed}"
        )


def _require_python(expected_version: str) -> None:
    observed = f"{sys.version_info.major}.{sys.version_info.minor}"
    if observed != expected_version:
        raise RuntimeError(
            f"Python version changed: expected {expected_version}, found {observed}"
        )


def _activate_verified_source(source: Path, domain: str, package_directory: str) -> None:
    lock = ENVIRONMENT_LOCKS[domain]
    if not (source / package_directory).is_dir():
        raise RuntimeError(f"external source tree is incomplete: {source}")
    observed = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed != lock.commit:
        raise RuntimeError(
            f"external source commit changed for {domain}: "
            f"expected {lock.commit}, found {observed}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            f"external source checkout is dirty for {domain}; use a clean checkout of {lock.commit}"
        )
    _require_python(lock.python)
    _require_distribution(lock.distribution, lock.version)
    for distribution, version in RUNTIME_DEPENDENCIES[domain]:
        _require_distribution(distribution, version)
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def _value_iteration_action_table(environment, gamma: float) -> dict[int, int]:
    transitions = environment.unwrapped.P
    values = {int(state): 0.0 for state in transitions}
    for _iteration in range(10_000):
        updated = {}
        for state, actions in transitions.items():
            action_values = []
            for action in sorted(actions):
                action_values.append(
                    sum(
                        probability
                        * (reward + (0.0 if terminated else gamma * values[int(next_state)]))
                        for probability, next_state, reward, terminated in actions[action]
                    )
                )
            updated[int(state)] = max(action_values)
        if max(abs(updated[state] - values[state]) for state in values) < 1e-12:
            values = updated
            break
        values = updated
    policy = {}
    for state, actions in transitions.items():
        scored = []
        for action in sorted(actions):
            value = sum(
                probability
                * (reward + (0.0 if terminated else gamma * values[int(next_state)]))
                for probability, next_state, reward, terminated in actions[action]
            )
            scored.append((value, -int(action), int(action)))
        policy[int(state)] = max(scored)[2]
    return policy


def _frozenlake_policy(task: ExternalTask, policy: str):
    import gymnasium as gym

    parameters = task.parameter_dict()
    schedules = {
        "nominal_value_iteration": ((1.0, 0.0, 0.0), 0.99),
        "hazard_averse_value_iteration": ((1.0, -2.0, -0.005), 0.99),
        "short_path_value_iteration": ((1.0, -0.5, -0.02), 0.95),
    }
    reward_schedule, gamma = schedules[policy]
    planning = gym.make(
        task.environment_id,
        map_name=str(parameters["map_name"]),
        is_slippery=bool(parameters["is_slippery"]),
        success_rate=float(parameters["success_rate"]),
        reward_schedule=reward_schedule,
    )
    try:
        return _value_iteration_action_table(planning, gamma)
    finally:
        planning.close()


def run_frozenlake(
    task: ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    environment_source: Path,
) -> list[ExternalEpisodeOutcome]:
    _activate_verified_source(
        environment_source,
        "gymnasium_frozenlake",
        "gymnasium",
    )
    import gymnasium as gym

    parameters = task.parameter_dict()
    action_table = _frozenlake_policy(task, policy)
    environment = gym.make(
        task.environment_id,
        map_name=str(parameters["map_name"]),
        is_slippery=bool(parameters["is_slippery"]),
        success_rate=float(parameters["success_rate"]),
        max_episode_steps=int(parameters["max_steps"]),
    )
    rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            observation, _info = environment.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                observation, reward, terminated, truncated, _info = environment.step(
                    action_table[int(observation)]
                )
                total_reward += float(reward)
                steps += 1
            success = int(total_reward > 0.0)
            failure = not bool(success)
            utility = 100.0 * success - 35.0 * failure - 0.1 * steps
            rows.append(
                ExternalEpisodeOutcome(
                    domain=task.domain,
                    task=task.name,
                    policy=policy,
                    episode=episode,
                    seed=seed,
                    utility=utility,
                    raw_return=total_reward,
                    cost=float(failure),
                    failure=failure,
                    steps=steps,
                    successes=success,
                    resource_residual=0.0,
                )
            )
    finally:
        environment.close()
    return rows


def _knapsack_catalog(task: ExternalTask) -> tuple[list[int], list[int]]:
    parameters = task.parameter_dict()
    rng = random.Random(int(parameters["catalog_seed"]))
    regime = str(parameters["item_regime"])
    weights = []
    values = []
    for _index in range(int(parameters["catalog_size"])):
        if regime == "balanced":
            weight = rng.randint(5, 30)
            value = max(1, int(1.45 * weight + rng.randint(-8, 18)))
        elif regime == "bulky":
            weight = rng.randint(20, 60)
            value = max(1, int(1.75 * weight + rng.randint(-20, 25)))
        elif regime == "volatile":
            if rng.random() < 0.2:
                weight = rng.randint(30, 70)
                value = rng.randint(90, 180)
            else:
                weight = rng.randint(3, 35)
                value = rng.randint(1, 65)
        else:
            raise KeyError(regime)
        weights.append(weight)
        values.append(value)
    return weights, values


def knapsack_action(policy: str, state, capacity: int, horizon: int, step: int) -> int:
    current_weight, _item, item_weight, item_value = [float(value) for value in state]
    if current_weight + item_weight > capacity:
        return 0
    ratio = item_value / max(item_weight, 1.0)
    if policy == "ratio_threshold_1_25":
        threshold = 1.25
    elif policy == "ratio_threshold_2_0":
        threshold = 2.0
    elif policy == "dynamic_reserve":
        fill = current_weight / max(capacity, 1)
        remaining = (horizon - step) / max(horizon, 1)
        threshold = 1.05 + 1.2 * fill + 0.35 * remaining
    else:
        raise KeyError(policy)
    return int(ratio >= threshold)


def run_knapsack(
    task: ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    environment_source: Path,
) -> list[ExternalEpisodeOutcome]:
    _activate_verified_source(
        environment_source,
        "or_gym_online_knapsack",
        "or_gym",
    )
    import numpy as np
    from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv

    parameters = task.parameter_dict()
    capacity = int(parameters["capacity"])
    horizon = int(parameters["horizon"])
    weights, values = _knapsack_catalog(task)
    rows = []
    for episode in range(episodes):
        seed = seed_base + episode
        np.random.seed(seed)
        environment = OnlineKnapsackEnv()
        environment.max_weight = capacity
        environment.step_limit = horizon
        environment.item_weights = np.asarray(weights, dtype=np.int32)
        environment.item_values = np.asarray(values, dtype=np.int32)
        environment.item_limits_init = np.ones(len(weights), dtype=np.int32)
        environment.item_probs = environment.item_limits_init / environment.item_limits_init.sum()
        np.random.seed(seed)
        observation = environment.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < horizon:
            action = knapsack_action(
                policy,
                observation["state"],
                capacity=capacity,
                horizon=horizon,
                step=steps,
            )
            observation, reward, done, _info = environment.step(action)
            total_reward += float(reward)
            steps += 1
        early_exhaustion = done and steps < horizon
        unused_capacity = max(0.0, capacity - float(environment.current_weight))
        utility = total_reward - 100.0 * early_exhaustion - 0.05 * unused_capacity
        rows.append(
            ExternalEpisodeOutcome(
                domain=task.domain,
                task=task.name,
                policy=policy,
                episode=episode,
                seed=seed,
                utility=utility,
                raw_return=total_reward,
                cost=float(early_exhaustion),
                failure=bool(early_exhaustion),
                steps=steps,
                successes=0,
                resource_residual=unused_capacity,
            )
        )
    return rows


def safety_point_action(policy: str, observation: dict) -> list[float]:
    goal = observation["goal_lidar"]
    hazards = observation.get("hazards_lidar")
    vases = observation.get("vases_lidar")
    bins = len(goal)
    goal_index = max(range(bins), key=lambda index: (float(goal[index]), -index))
    chosen_index = goal_index
    drive = 1.0
    if policy != "goal_greedy":
        threshold = {
            "hazard_aware_moderate": 0.45,
            "hazard_aware_strict": 0.25,
        }[policy]
        obstacle = [0.0] * bins
        for index in range(bins):
            hazard_value = 0.0 if hazards is None else float(hazards[index])
            vase_value = 0.0 if vases is None else float(vases[index])
            obstacle[index] = max(hazard_value, vase_value)
        target_pressure = max(
            obstacle[(goal_index + offset) % bins]
            for offset in (-1, 0, 1)
        )
        if target_pressure >= threshold:
            chosen_index = max(
                range(bins),
                key=lambda index: (
                    float(goal[index]) - 1.5 * obstacle[index],
                    -abs(index - goal_index),
                    -index,
                ),
            )
            drive = 0.25
    angle = 2.0 * math.pi * chosen_index / bins
    if angle > math.pi:
        angle -= 2.0 * math.pi
    turn = max(-1.0, min(1.0, angle / (math.pi / 2.0)))
    if abs(angle) > math.pi / 2.0:
        drive = min(drive, 0.2)
    return [drive, turn]


def run_safety_point_goal(
    task: ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    environment_source: Path,
) -> list[ExternalEpisodeOutcome]:
    _activate_verified_source(
        environment_source,
        "safety_gymnasium_point_goal",
        "safety_gymnasium",
    )
    import safety_gymnasium
    from gymnasium.spaces.utils import unflatten

    parameters = task.parameter_dict()
    max_steps = int(parameters["max_steps"])
    cost_weight = float(parameters["cost_weight"])
    layout_seed_base = int(parameters["layout_seed_base"])
    rows = []
    environment = safety_gymnasium.make(task.environment_id)
    try:
        for episode in range(episodes):
            seed = layout_seed_base + seed_base + episode
            observation, _info = environment.reset(seed=seed)
            total_reward = 0.0
            total_cost = 0.0
            successes = 0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated) and steps < max_steps:
                structured = unflatten(environment.task.obs_info.obs_space_dict, observation)
                action = safety_point_action(policy, structured)
                observation, reward, cost, terminated, truncated, info = environment.step(action)
                total_reward += float(reward)
                total_cost += float(cost)
                successes += int(bool(info.get("goal_met", False)))
                steps += 1
            failure = successes == 0
            utility = total_reward + 5.0 * successes - cost_weight * total_cost - 2.0 * failure
            rows.append(
                ExternalEpisodeOutcome(
                    domain=task.domain,
                    task=task.name,
                    policy=policy,
                    episode=episode,
                    seed=seed,
                    utility=utility,
                    raw_return=total_reward,
                    cost=total_cost,
                    failure=failure,
                    steps=steps,
                    successes=successes,
                    resource_residual=0.0,
                )
            )
    finally:
        environment.close()
    return rows


def run_external_task(
    task: ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    environment_source: Path | None = None,
) -> list[ExternalEpisodeOutcome]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if environment_source is None:
        raise RuntimeError(
            "external execution requires --environment-source at the pinned commit"
        )
    if task.domain == "gymnasium_frozenlake":
        return run_frozenlake(task, policy, episodes, seed_base, environment_source)
    if task.domain == "or_gym_online_knapsack":
        return run_knapsack(task, policy, episodes, seed_base, environment_source)
    if task.domain == "safety_gymnasium_point_goal":
        return run_safety_point_goal(task, policy, episodes, seed_base, environment_source)
    raise KeyError(task.domain)
