"""Executable development adapters for the RiskShiftBench v2 external suite.

Each adapter emits a score in [0, 1], so every paired candidate-minus-fallback
difference is deterministically bounded in [-1, 1]. This module refuses all
confirmation tasks; a separately registered wrapper will be required later.
"""

from __future__ import annotations

import heapq
import importlib.metadata
import math
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from experiments.frontier_v2_external_design import (
    CODEBASE_LOCKS,
    DOMAIN_SPECS,
    V2ExternalTask,
    expected_episode_seeds,
)
from experiments.frontier_v2_source_audit import SOURCE_DIRECTORIES, audit_codebase_source


@dataclass(frozen=True)
class V2EpisodeOutcome:
    domain: str
    task: str
    policy: str
    episode: int
    seed: int
    score: float
    raw_utility: float
    raw_return: float
    cost: float
    failure: bool
    steps: int
    successes: int


def bounded_score(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("score input must be finite")
    return max(0.0, min(1.0, numeric))


def summarize_v2_outcomes(outcomes: list[V2EpisodeOutcome]) -> dict:
    if not outcomes:
        raise ValueError("cannot summarize empty v2 outcomes")
    scores = [outcome.score for outcome in outcomes]
    return {
        "domain": outcomes[0].domain,
        "task": outcomes[0].task,
        "policy": outcomes[0].policy,
        "episodes": len(outcomes),
        "mean_score": fmean(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "mean_raw_utility": fmean(outcome.raw_utility for outcome in outcomes),
        "failure_probability": fmean(float(outcome.failure) for outcome in outcomes),
        "mean_cost": fmean(outcome.cost for outcome in outcomes),
        "mean_steps": fmean(outcome.steps for outcome in outcomes),
    }


def outcome_rows(outcomes: list[V2EpisodeOutcome]) -> list[dict]:
    return [asdict(outcome) for outcome in outcomes]


def assert_development_execution(task: V2ExternalTask) -> None:
    if task.split not in {"development", "calibration"}:
        raise RuntimeError(
            "v2 confirmation execution is prohibited until an externally registered "
            "design and wrapper exist"
        )


@lru_cache(maxsize=len(CODEBASE_LOCKS))
def _activate_verified_source(source: Path, codebase: str) -> None:
    lock = CODEBASE_LOCKS[codebase]
    audit_codebase_source(source, codebase)
    observed_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if observed_python != lock.python:
        raise RuntimeError(
            f"Python version changed for {codebase}: expected {lock.python}, "
            f"found {observed_python}"
        )
    observed_distribution = importlib.metadata.version(lock.distribution)
    if observed_distribution != lock.version:
        raise RuntimeError(
            f"{lock.distribution} version changed: expected {lock.version}, "
            f"found {observed_distribution}"
        )
    for distribution, expected_version in lock.runtime_dependencies:
        observed_version = importlib.metadata.version(distribution)
        if observed_version != expected_version:
            raise RuntimeError(
                f"{distribution} version changed for {codebase}: expected "
                f"{expected_version}, found {observed_version}"
            )
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def transformed_value_iteration_action_table(
    transitions,
    *,
    gamma: float,
    cliff_multiplier: float = 1.0,
    step_multiplier: float = 1.0,
    next_state_penalties: dict[int, float] | None = None,
) -> dict[int, int]:
    """Solve a finite tabular MDP with a frozen reward transformation."""

    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if cliff_multiplier <= 0.0 or step_multiplier <= 0.0:
        raise ValueError("reward multipliers must be positive")
    penalties = next_state_penalties or {}
    if any(state not in transitions for state in penalties):
        raise ValueError("next-state penalty references an unknown state")
    if any(not math.isfinite(value) or value < 0.0 for value in penalties.values()):
        raise ValueError("next-state penalties must be finite and nonnegative")

    def transformed_reward(reward: float) -> float:
        if reward <= -100.0:
            return cliff_multiplier * reward
        if reward < 0.0:
            return step_multiplier * reward
        return reward

    values = {int(state): 0.0 for state in transitions}
    for _iteration in range(20_000):
        updated = {}
        for state, actions in transitions.items():
            updated[int(state)] = max(
                sum(
                    probability
                    * (
                        transformed_reward(float(reward))
                        - penalties.get(int(next_state), 0.0)
                        + (0.0 if terminated else gamma * values[int(next_state)])
                    )
                    for probability, next_state, reward, terminated in outcomes
                )
                for outcomes in actions.values()
            )
        if max(abs(updated[state] - values[state]) for state in values) < 1e-12:
            values = updated
            break
        values = updated
    table = {}
    for state, actions in transitions.items():
        scored = []
        for action, outcomes in actions.items():
            value = sum(
                probability
                * (
                    transformed_reward(float(reward))
                    - penalties.get(int(next_state), 0.0)
                    + (0.0 if terminated else gamma * values[int(next_state)])
                )
                for probability, next_state, reward, terminated in outcomes
            )
            scored.append((value, -int(action), int(action)))
        table[int(state)] = max(scored)[2]
    return table


def _gymnasium_policy_table(task: V2ExternalTask, policy: str):
    import gymnasium as gym

    parameters = task.parameter_dict()
    if task.domain == "gymnasium_frozenlake":
        from gymnasium.envs.toy_text.frozen_lake import generate_random_map

        actual_slippery = bool(parameters["is_slippery"])
        description = generate_random_map(
            size=int(parameters["map_size"]),
            p=float(parameters["frozen_probability"]),
            seed=int(parameters["map_seed"]),
        )
        schedules = {
            "nominal_value_iteration": ((1.0, 0.0, 0.0), 0.99, False),
            "hazard_averse_value_iteration": (
                (1.0, -2.0, -0.005),
                0.99,
                actual_slippery,
            ),
            "short_path_value_iteration": (
                (1.0, -0.5, -0.02),
                0.95,
                actual_slippery,
            ),
        }
        reward_schedule, gamma, planning_slippery = schedules[policy]
        planning = gym.make(
            task.environment_id,
            desc=description,
            is_slippery=planning_slippery,
            success_rate=float(parameters["success_rate"]),
            reward_schedule=reward_schedule,
        )
        try:
            return (
                transformed_value_iteration_action_table(
                    planning.unwrapped.P, gamma=gamma
                ),
                description,
            )
        finally:
            planning.close()

    if task.domain == "gymnasium_cliffwalking":
        actual_slippery = bool(parameters["is_slippery"])
        settings = {
            "nominal_value_iteration": (0.99, 1.0, 1.0, 0.0, False),
            "cliff_averse_value_iteration": (
                0.995,
                3.0,
                1.0,
                5.0,
                actual_slippery,
            ),
            "fast_value_iteration": (0.90, 1.0, 1.0, 0.0, actual_slippery),
        }
        (
            gamma,
            cliff_multiplier,
            step_multiplier,
            proximity_penalty,
            planning_slippery,
        ) = settings[policy]
        planning = gym.make(
            task.environment_id,
            is_slippery=planning_slippery,
        )
        next_state_penalties = {
            state: proximity_penalty
            for state in planning.unwrapped.P
            if state // 12 == 2 and 1 <= state % 12 <= 10
        }
        try:
            return (
                transformed_value_iteration_action_table(
                    planning.unwrapped.P,
                    gamma=gamma,
                    cliff_multiplier=cliff_multiplier,
                    step_multiplier=step_multiplier,
                    next_state_penalties=next_state_penalties,
                ),
                None,
            )
        finally:
            planning.close()

    if task.domain == "gymnasium_taxi":
        actual_rain = bool(parameters["is_rainy"])
        if policy == "dry_value_iteration":
            rainy = False
            gamma = 0.99
            step_multiplier = 1.0
        elif policy == "rain_robust_value_iteration":
            rainy = True
            gamma = 0.99
            step_multiplier = 1.0
        elif policy == "delay_averse_value_iteration":
            rainy = actual_rain
            gamma = 0.92
            step_multiplier = 1.5
        else:
            raise KeyError(policy)
        planning = gym.make(
            task.environment_id,
            is_rainy=rainy,
            rainy_probability=float(parameters["rainy_probability"]),
            fickle_passenger=False,
        )
        try:
            return (
                transformed_value_iteration_action_table(
                    planning.unwrapped.P,
                    gamma=gamma,
                    step_multiplier=step_multiplier,
                ),
                None,
            )
        finally:
            planning.close()
    raise KeyError(task.domain)


def run_gymnasium_task(
    task: V2ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    source: Path,
) -> list[V2EpisodeOutcome]:
    assert_development_execution(task)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if policy not in {
        DOMAIN_SPECS[task.domain].fallback_policy,
        *DOMAIN_SPECS[task.domain].candidate_policies,
    }:
        raise KeyError(policy)
    _activate_verified_source(source, "gymnasium")
    import gymnasium as gym

    parameters = task.parameter_dict()
    action_table, frozenlake_description = _gymnasium_policy_table(task, policy)
    kwargs: dict[str, object] = {}
    if task.domain == "gymnasium_frozenlake":
        kwargs = {
            "desc": frozenlake_description,
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
    environment = gym.make(
        task.environment_id,
        **kwargs,
        max_episode_steps=int(parameters["max_steps"]),
    )
    rows = []
    try:
        seeds = expected_episode_seeds(
            task, episodes=episodes, seed_base=seed_base
        )
        for episode, seed in enumerate(seeds):
            observation, _info = environment.reset(seed=seed)
            raw_return = 0.0
            steps = 0
            cliff_falls = 0
            illegal_actions = 0
            success = False
            terminated = truncated = False
            while not (terminated or truncated):
                observation, reward, terminated, truncated, _info = environment.step(
                    action_table[int(observation)]
                )
                numeric_reward = float(reward)
                raw_return += numeric_reward
                steps += 1
                cliff_falls += int(task.domain == "gymnasium_cliffwalking" and numeric_reward <= -100.0)
                illegal_actions += int(task.domain == "gymnasium_taxi" and numeric_reward <= -10.0)
                if task.domain == "gymnasium_frozenlake":
                    success = success or numeric_reward > 0.0
                elif task.domain == "gymnasium_cliffwalking":
                    success = success or bool(terminated)
                elif task.domain == "gymnasium_taxi":
                    success = success or numeric_reward >= 20.0
            max_steps = int(parameters["max_steps"])
            cost = float(cliff_falls + illegal_actions)
            score = bounded_score(
                float(success) * (1.0 - 0.5 * steps / max_steps) - 0.1 * cost
            )
            rows.append(
                V2EpisodeOutcome(
                    domain=task.domain,
                    task=task.name,
                    policy=policy,
                    episode=episode,
                    seed=seed,
                    score=score,
                    raw_utility=raw_return,
                    raw_return=raw_return,
                    cost=cost,
                    failure=not success,
                    steps=steps,
                    successes=int(success),
                )
            )
    finally:
        environment.close()
    return rows


def image_navigation_action(image, policy: str) -> int:
    """Choose a MiniGrid left/right/forward action from a full image encoding."""

    clearance_by_policy = {
        "image_shortest_path": 0,
        "image_clearance_one": 1,
        "image_clearance_two": 2,
        "image_lava_clearance": 1,
        "image_conservative_turn": 2,
    }
    try:
        clearance = clearance_by_policy[policy]
    except KeyError as error:
        raise KeyError(policy) from error
    width = len(image)
    if width == 0:
        raise ValueError("image must be nonempty")
    height = len(image[0])
    agent = goal = None
    hazards = set()
    walls = set()
    direction = 0
    for x in range(width):
        if len(image[x]) != height:
            raise ValueError("image rows must have equal height")
        for y in range(height):
            object_index = int(image[x][y][0])
            if object_index == 10:
                agent = (x, y)
                direction = int(image[x][y][2])
            elif object_index == 8:
                goal = (x, y)
            elif object_index == 2:
                walls.add((x, y))
            elif object_index in {6, 9}:
                hazards.add((x, y))
    if agent is None or goal is None:
        raise ValueError("image must encode exactly an agent and a goal")

    def risk_cost(cell: tuple[int, int]) -> float:
        if clearance <= 0 or not hazards:
            return 0.0
        distance = min(
            abs(cell[0] - hazard[0]) + abs(cell[1] - hazard[1])
            for hazard in hazards
        )
        return 5.0 * max(clearance + 1 - distance, 0)

    distances = {agent: 0.0}
    first_step: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, agent)]
    while queue:
        distance, cell = heapq.heappop(queue)
        if distance != distances[cell]:
            continue
        if cell == goal:
            break
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if neighbor in walls or neighbor in hazards:
                continue
            candidate_distance = distance + 1.0 + risk_cost(neighbor)
            if candidate_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate_distance
                first_step[neighbor] = neighbor if cell == agent else first_step[cell]
                heapq.heappush(queue, (candidate_distance, neighbor))
    if goal not in distances:
        return 0
    next_cell = first_step[goal]
    delta = (next_cell[0] - agent[0], next_cell[1] - agent[1])
    desired_direction = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[delta]
    turn = (desired_direction - direction) % 4
    if turn == 0:
        return 2
    if turn == 1:
        return 1
    return 0


def run_minigrid_task(
    task: V2ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    source: Path,
) -> list[V2EpisodeOutcome]:
    assert_development_execution(task)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if policy not in {
        DOMAIN_SPECS[task.domain].fallback_policy,
        *DOMAIN_SPECS[task.domain].candidate_policies,
    }:
        raise KeyError(policy)
    _activate_verified_source(source, "minigrid")
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.wrappers import FullyObsWrapper

    parameters = task.parameter_dict()
    kwargs: dict[str, object]
    if task.domain == "minigrid_dynamic_obstacles":
        kwargs = {
            "size": int(parameters["size"]),
            "n_obstacles": int(parameters["n_obstacles"]),
            "agent_start_pos": None
            if bool(parameters["agent_start_random"])
            else (1, 1),
            "max_steps": int(parameters["max_steps"]),
        }
    elif task.domain == "minigrid_lava_crossing":
        kwargs = {
            "size": int(parameters["size"]),
            "num_crossings": int(parameters["num_crossings"]),
            "max_steps": int(parameters["max_steps"]),
        }
    else:
        raise KeyError(task.domain)
    environment = FullyObsWrapper(gym.make(task.environment_id, **kwargs))
    rows = []
    try:
        seeds = expected_episode_seeds(
            task, episodes=episodes, seed_base=seed_base
        )
        for episode, seed in enumerate(seeds):
            observation, _info = environment.reset(seed=seed)
            image = observation["image"]
            if image.size < DOMAIN_SPECS[task.domain].minimum_observation_coordinates:
                raise RuntimeError("MiniGrid observation is below the frozen dimension floor")
            raw_return = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = image_navigation_action(observation["image"], policy)
                observation, reward, terminated, truncated, _info = environment.step(action)
                raw_return += float(reward)
                steps += 1
            success = raw_return > 0.0
            collision = raw_return < 0.0
            score = bounded_score(raw_return)
            rows.append(
                V2EpisodeOutcome(
                    domain=task.domain,
                    task=task.name,
                    policy=policy,
                    episode=episode,
                    seed=seed,
                    score=score,
                    raw_utility=raw_return,
                    raw_return=raw_return,
                    cost=float(collision),
                    failure=not success,
                    steps=steps,
                    successes=int(success),
                )
            )
    finally:
        environment.close()
    return rows


def inventory_base_stock_levels(
    policy: str,
    *,
    demand_mean: float,
    lead_time_scale: float,
) -> tuple[float, float, float]:
    multiplier = {
        "base_stock_lean": 0.80,
        "base_stock_nominal": 1.00,
        "base_stock_buffered": 1.25,
    }
    try:
        policy_multiplier = multiplier[policy]
    except KeyError as error:
        raise KeyError(policy) from error
    base = demand_mean * policy_multiplier
    return (
        base * (1.0 + 1.5 * lead_time_scale),
        base * (1.0 + 2.5 * lead_time_scale),
        base * (1.0 + 4.0 * lead_time_scale),
    )


def inventory_reorder_action(
    policy: str,
    *,
    demand_mean: float,
    lead_time_scale: float,
    inventory_position,
    supply_capacity,
) -> tuple[float, ...]:
    levels = inventory_base_stock_levels(
        policy,
        demand_mean=demand_mean,
        lead_time_scale=lead_time_scale,
    )
    if len(inventory_position) != len(levels) or len(supply_capacity) != len(levels):
        raise ValueError("inventory vectors must match the three frozen stages")
    return tuple(
        max(0.0, min(float(capacity), level - float(position)))
        for level, position, capacity in zip(
            levels, inventory_position, supply_capacity, strict=True
        )
    )


def run_or_gym_task(
    task: V2ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    source: Path,
) -> list[V2EpisodeOutcome]:
    assert_development_execution(task)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if policy not in {
        DOMAIN_SPECS[task.domain].fallback_policy,
        *DOMAIN_SPECS[task.domain].candidate_policies,
    }:
        raise KeyError(policy)
    _activate_verified_source(source, "or_gym")
    import numpy as np

    parameters = task.parameter_dict()
    rows = []
    if task.domain == "or_gym_online_knapsack":
        from experiments.external_domain_adapters import (
            _knapsack_catalog,
            knapsack_action,
        )
        from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv

        capacity = int(parameters["capacity"])
        horizon = int(parameters["horizon"])
        weights, values = _knapsack_catalog(task)
        fractional_upper_bound = capacity * max(
            value / weight for weight, value in zip(weights, values, strict=True)
        )
        seeds = expected_episode_seeds(
            task, episodes=episodes, seed_base=seed_base
        )
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
            observation = environment.reset()
            raw_return = 0.0
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
                raw_return += float(reward)
                steps += 1
            early_exhaustion = done and steps < horizon
            unused_capacity = max(0.0, capacity - float(environment.current_weight))
            raw_utility = (
                raw_return - 100.0 * early_exhaustion - 0.05 * unused_capacity
            )
            rows.append(
                V2EpisodeOutcome(
                    domain=task.domain,
                    task=task.name,
                    policy=policy,
                    episode=episode,
                    seed=seed,
                    score=bounded_score(raw_return / fractional_upper_bound),
                    raw_utility=raw_utility,
                    raw_return=raw_return,
                    cost=float(early_exhaustion),
                    failure=bool(early_exhaustion),
                    steps=steps,
                    successes=0,
                )
            )
        return rows

    if task.domain == "or_gym_inventory_management":
        from or_gym.envs.supply_chain.inventory_management import (
            InvManagementBacklogEnv,
            InvManagementLostSalesEnv,
        )

        environment_class = (
            InvManagementBacklogEnv
            if bool(parameters["backlog"])
            else InvManagementLostSalesEnv
        )
        demand_mean = float(parameters["demand_mean"])
        lead_time_scale = float(parameters["lead_time_scale"])
        lead_times = [
            max(1, round(base * lead_time_scale)) for base in (3, 5, 10)
        ]
        seeds = expected_episode_seeds(
            task, episodes=episodes, seed_base=seed_base
        )
        for episode, seed in enumerate(seeds):
            np.random.seed(seed)
            environment = environment_class(
                periods=int(parameters["periods"]),
                dist=1,
                dist_param={"mu": demand_mean},
                seed_int=seed,
                L=lead_times,
            )
            observation = environment.reset()
            raw_return = 0.0
            done = False
            steps = 0
            while not done:
                environment._update_base_stock_policy_state()
                action = inventory_reorder_action(
                    policy,
                    demand_mean=demand_mean,
                    lead_time_scale=lead_time_scale,
                    inventory_position=environment.state,
                    supply_capacity=environment.supply_capacity,
                )
                observation, reward, done, _info = environment.step(
                    np.asarray(action, dtype=np.int32)
                )
                raw_return += float(reward)
                steps += 1
            del observation
            unmet = float(environment.B.sum() + environment.LS.sum())
            lower = float(parameters["profit_lower"])
            upper = float(parameters["profit_upper"])
            rows.append(
                V2EpisodeOutcome(
                    domain=task.domain,
                    task=task.name,
                    policy=policy,
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
        return rows
    raise KeyError(task.domain)


def safety_navigation_action(policy: str, observation: dict) -> list[float]:
    target = observation["goal_lidar"]
    bins = len(target)
    target_index = max(range(bins), key=lambda index: (float(target[index]), -index))
    thresholds = {
        "goal_greedy": None,
        "button_greedy": None,
        "hazard_aware_moderate": 0.45,
        "button_hazard_aware_moderate": 0.45,
        "hazard_aware_strict": 0.25,
        "button_hazard_aware_strict": 0.25,
    }
    try:
        threshold = thresholds[policy]
    except KeyError as error:
        raise KeyError(policy) from error
    obstacle = [0.0] * bins
    for key in ("hazards_lidar", "vases_lidar", "gremlins_lidar"):
        values = observation.get(key)
        if values is not None:
            for index in range(bins):
                obstacle[index] = max(obstacle[index], float(values[index]))
    chosen_index = target_index
    drive = 1.0
    if threshold is not None:
        target_pressure = max(
            obstacle[(target_index + offset) % bins] for offset in (-1, 0, 1)
        )
        if target_pressure >= threshold:
            chosen_index = max(
                range(bins),
                key=lambda index: (
                    float(target[index]) - 1.5 * obstacle[index],
                    -abs(index - target_index),
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


def run_safety_gymnasium_task(
    task: V2ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    source: Path,
) -> list[V2EpisodeOutcome]:
    assert_development_execution(task)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if policy not in {
        DOMAIN_SPECS[task.domain].fallback_policy,
        *DOMAIN_SPECS[task.domain].candidate_policies,
    }:
        raise KeyError(policy)
    _activate_verified_source(source, "safety_gymnasium")
    import safety_gymnasium
    from gymnasium.spaces.utils import unflatten

    parameters = task.parameter_dict()
    max_steps = int(parameters["max_steps"])
    cost_weight = float(parameters["cost_weight"])
    rows = []
    environment = safety_gymnasium.make(task.environment_id)
    try:
        seeds = expected_episode_seeds(
            task, episodes=episodes, seed_base=seed_base
        )
        for episode, seed in enumerate(seeds):
            observation, _info = environment.reset(seed=seed)
            raw_return = 0.0
            total_cost = 0.0
            successes = 0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated) and steps < max_steps:
                structured = unflatten(
                    environment.task.obs_info.obs_space_dict, observation
                )
                action = safety_navigation_action(policy, structured)
                observation, reward, cost, terminated, truncated, info = (
                    environment.step(action)
                )
                raw_return += float(reward)
                total_cost += float(cost)
                successes += int(bool(info.get("goal_met", False)))
                steps += 1
            failure = successes == 0
            raw_utility = (
                raw_return
                + 5.0 * successes
                - cost_weight * total_cost
                - 2.0 * failure
            )
            score = 0.5 + 0.5 * math.tanh(raw_utility / 25.0)
            rows.append(
                V2EpisodeOutcome(
                    domain=task.domain,
                    task=task.name,
                    policy=policy,
                    episode=episode,
                    seed=seed,
                    score=bounded_score(score),
                    raw_utility=raw_utility,
                    raw_return=raw_return,
                    cost=total_cost,
                    failure=failure,
                    steps=steps,
                    successes=successes,
                )
            )
    finally:
        environment.close()
    return rows


def run_v2_development_task(
    task: V2ExternalTask,
    policy: str,
    episodes: int,
    seed_base: int,
    source_root: Path,
) -> list[V2EpisodeOutcome]:
    """Dispatch an outcome-eligible task to its isolated-codebase adapter."""

    assert_development_execution(task)
    codebase = DOMAIN_SPECS[task.domain].codebase
    source = source_root / SOURCE_DIRECTORIES[codebase]
    if codebase == "gymnasium":
        return run_gymnasium_task(task, policy, episodes, seed_base, source)
    if codebase == "minigrid":
        return run_minigrid_task(task, policy, episodes, seed_base, source)
    if codebase == "or_gym":
        return run_or_gym_task(task, policy, episodes, seed_base, source)
    if codebase == "safety_gymnasium":
        return run_safety_gymnasium_task(task, policy, episodes, seed_base, source)
    raise KeyError(codebase)
