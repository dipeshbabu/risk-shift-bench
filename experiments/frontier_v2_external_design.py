"""Outcome-free breadth design for the RiskShiftBench v2 external study.

This module is separate from the completed v1 external design. Importing it
does not instantiate or reset an environment. Confirmation tasks are declared
for prospective hashing but are not execution-eligible before registration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import TypeAlias


ParameterValue: TypeAlias = str | int | float | bool | tuple[int, ...] | tuple[float, ...]
SPLITS = ("development", "calibration", "confirmation")


@dataclass(frozen=True)
class V2CodebaseLock:
    name: str
    distribution: str
    version: str
    repository: str
    commit: str
    license: str
    python: str
    runtime_dependencies: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class V2DomainSpec:
    name: str
    codebase: str
    observation_kind: str
    minimum_observation_coordinates: int
    score_rule: str
    score_lower: float
    score_upper: float
    fallback_policy: str
    candidate_policies: tuple[str, ...]
    competitive_baselines: tuple[str, ...]

    @property
    def paired_difference_lower(self) -> float:
        return self.score_lower - self.score_upper

    @property
    def paired_difference_upper(self) -> float:
        return self.score_upper - self.score_lower


@dataclass(frozen=True)
class V2ExternalTask:
    name: str
    domain: str
    split: str
    environment_id: str
    parameters: tuple[tuple[str, ParameterValue], ...]
    features: tuple[float, ...]

    def parameter_dict(self) -> dict[str, ParameterValue]:
        return dict(self.parameters)


CODEBASE_LOCKS = {
    "gymnasium": V2CodebaseLock(
        name="gymnasium",
        distribution="gymnasium",
        version="1.3.0",
        repository="https://github.com/Farama-Foundation/Gymnasium",
        commit="53bf3e9a884783eb72ad3fc8b15780914c97c3e1",
        license="MIT",
        python="3.12",
        runtime_dependencies=(("numpy", "2.3.1"),),
    ),
    "or_gym": V2CodebaseLock(
        name="or_gym",
        distribution="or-gym",
        version="0.5.0",
        repository="https://github.com/hubbs5/or-gym",
        commit="0b18d16e569e2db70e83f09e867b53bdb4b87298",
        license="MIT",
        python="3.10",
        runtime_dependencies=(
            ("gym", "0.26.2"),
            ("numpy", "1.26.4"),
            ("scipy", "1.14.1"),
            ("matplotlib", "3.9.2"),
            ("pandas", "2.2.3"),
            ("networkx", "3.4.2"),
        ),
    ),
    "safety_gymnasium": V2CodebaseLock(
        name="safety_gymnasium",
        distribution="safety-gymnasium",
        version="1.2.0",
        repository="https://github.com/PKU-Alignment/safety-gymnasium",
        commit="98231340a4c5b223c8d111fa9597d81836ce09b4",
        license="Apache-2.0",
        python="3.10",
        runtime_dependencies=(
            ("gymnasium", "0.28.1"),
            ("gymnasium-robotics", "1.2.2"),
            ("mujoco", "2.3.3"),
            ("numpy", "1.23.5"),
            ("pygame", "2.1.0"),
            ("xmltodict", "0.14.2"),
            ("PyYAML", "6.0.2"),
            ("imageio", "2.37.3"),
        ),
    ),
    "minigrid": V2CodebaseLock(
        name="minigrid",
        distribution="minigrid",
        version="3.1.0",
        repository="https://github.com/Farama-Foundation/Minigrid",
        commit="90928729376741a41222a257911343b97103b548",
        license="MIT",
        python="3.12",
        runtime_dependencies=(
            ("gymnasium", "1.3.0"),
            ("numpy", "2.3.1"),
            ("pygame-ce", "2.5.5"),
        ),
    ),
}


DOMAIN_SPECS = {
    "gymnasium_frozenlake": V2DomainSpec(
        name="gymnasium_frozenlake",
        codebase="gymnasium",
        observation_kind="discrete_state",
        minimum_observation_coordinates=1,
        score_rule="bounded_success_time_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="nominal_value_iteration",
        candidate_policies=(
            "hazard_averse_value_iteration",
            "short_path_value_iteration",
        ),
        competitive_baselines=("tabular_oracle", "v1_fixed_router"),
    ),
    "gymnasium_cliffwalking": V2DomainSpec(
        name="gymnasium_cliffwalking",
        codebase="gymnasium",
        observation_kind="discrete_state",
        minimum_observation_coordinates=1,
        score_rule="bounded_safe_completion_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="nominal_value_iteration",
        candidate_policies=("cliff_averse_value_iteration", "fast_value_iteration"),
        competitive_baselines=("tabular_oracle", "q_learning_reference"),
    ),
    "gymnasium_taxi": V2DomainSpec(
        name="gymnasium_taxi",
        codebase="gymnasium",
        observation_kind="discrete_state",
        minimum_observation_coordinates=1,
        score_rule="bounded_delivery_efficiency_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="dry_value_iteration",
        candidate_policies=("rain_robust_value_iteration", "delay_averse_value_iteration"),
        competitive_baselines=("tabular_oracle", "q_learning_reference"),
    ),
    "or_gym_online_knapsack": V2DomainSpec(
        name="or_gym_online_knapsack",
        codebase="or_gym",
        observation_kind="structured_vector",
        minimum_observation_coordinates=4,
        score_rule="bounded_fractional_upper_bound_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="ratio_threshold_1_25",
        candidate_policies=("dynamic_reserve", "ratio_threshold_2_0"),
        competitive_baselines=("fractional_oracle", "rl_policy_reference"),
    ),
    "or_gym_inventory_management": V2DomainSpec(
        name="or_gym_inventory_management",
        codebase="or_gym",
        observation_kind="pipeline_inventory_vector",
        minimum_observation_coordinates=12,
        score_rule="clipped_profit_affine_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="base_stock_nominal",
        candidate_policies=("base_stock_lean", "base_stock_buffered"),
        competitive_baselines=("newsvendor_base_stock", "ppo_reference"),
    ),
    "safety_gymnasium_point_goal": V2DomainSpec(
        name="safety_gymnasium_point_goal",
        codebase="safety_gymnasium",
        observation_kind="flattened_lidar_proprioception",
        minimum_observation_coordinates=32,
        score_rule="bounded_safe_goal_utility_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="goal_greedy",
        candidate_policies=("hazard_aware_moderate", "hazard_aware_strict"),
        competitive_baselines=("ppo_lagrangian_reference", "cpo_reference"),
    ),
    "safety_gymnasium_point_button": V2DomainSpec(
        name="safety_gymnasium_point_button",
        codebase="safety_gymnasium",
        observation_kind="flattened_lidar_proprioception",
        minimum_observation_coordinates=32,
        score_rule="bounded_safe_button_utility_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="button_greedy",
        candidate_policies=(
            "button_hazard_aware_moderate",
            "button_hazard_aware_strict",
        ),
        competitive_baselines=("ppo_lagrangian_reference", "cpo_reference"),
    ),
    "minigrid_dynamic_obstacles": V2DomainSpec(
        name="minigrid_dynamic_obstacles",
        codebase="minigrid",
        observation_kind="fully_observable_image",
        minimum_observation_coordinates=75,
        score_rule="bounded_navigation_safety_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="image_shortest_path",
        candidate_policies=("image_clearance_one", "image_clearance_two"),
        competitive_baselines=("dqn_reference", "ppo_reference"),
    ),
    "minigrid_lava_crossing": V2DomainSpec(
        name="minigrid_lava_crossing",
        codebase="minigrid",
        observation_kind="fully_observable_image",
        minimum_observation_coordinates=243,
        score_rule="bounded_navigation_safety_v1",
        score_lower=0.0,
        score_upper=1.0,
        fallback_policy="image_shortest_path",
        candidate_policies=("image_lava_clearance", "image_conservative_turn"),
        competitive_baselines=("dqn_reference", "ppo_reference"),
    ),
}

DOMAINS = tuple(DOMAIN_SPECS)

STREAM_SEED_BASES = {
    ("development", "evaluation"): 10_000_000,
    ("calibration", "evaluation"): 20_000_000,
    ("confirmation", "pilot"): 30_000_000,
    ("confirmation", "final"): 40_000_000,
}
DOMAIN_SEED_STRIDE = 100_000
TASK_SEED_STRIDE = 10_000
MAX_EPISODES_PER_TASK_STREAM = TASK_SEED_STRIDE
OUTCOME_IMPLEMENTATION_FILES = (
    "experiments/frontier_v2_external_design.py",
    "experiments/frontier_v2_external_adapters.py",
    "experiments/frontier_v2_development.py",
    "experiments/frontier_v2_source_audit.py",
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def outcome_implementation_sha256(repository_root: Path | None = None) -> str:
    """Hash every repository file that defines a task outcome artifact."""

    root = Path(__file__).resolve().parents[1] if repository_root is None else repository_root
    digest = hashlib.sha256()
    for relative in OUTCOME_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"outcome implementation file is missing: {path}")
        canonical_content = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_content)
        digest.update(b"\0")
    return digest.hexdigest()


def _task(
    *,
    domain: str,
    split: str,
    index: int,
    environment_id: str,
    parameters: dict[str, ParameterValue],
    features: tuple[float, ...],
) -> V2ExternalTask:
    label = domain.replace("gymnasium_", "").replace("or_gym_", "").replace("_", "-")
    return V2ExternalTask(
        name=f"RSBv2-{label}-{split}-{index:02d}-v0",
        domain=domain,
        split=split,
        environment_id=environment_id,
        parameters=tuple(sorted(parameters.items())),
        features=features,
    )


def _split_seed_base(split: str, domain_offset: int) -> int:
    split_offset = {"development": 110_000, "calibration": 210_000, "confirmation": 310_000}
    try:
        return split_offset[split] + domain_offset
    except KeyError as error:
        raise KeyError(split) from error


def frozenlake_tasks(split: str) -> list[V2ExternalTask]:
    base = _split_seed_base(split, 1_000)
    frozen_probabilities = {
        "development": (0.78, 0.82, 0.86, 0.90),
        "calibration": (0.76, 0.80, 0.84, 0.88),
        "confirmation": (0.74, 0.79, 0.83, 0.87),
    }[split]
    tasks = []
    for index, (size, slippery, probability) in enumerate(
        zip((5, 6, 7, 8), (False, True, False, True), frozen_probabilities, strict=True)
    ):
        tasks.append(
            _task(
                domain="gymnasium_frozenlake",
                split=split,
                index=index,
                environment_id="FrozenLake-v1",
                parameters={
                    "map_seed": base + index,
                    "map_size": size,
                    "frozen_probability": probability,
                    "is_slippery": slippery,
                    "success_rate": 0.65,
                    "max_steps": 4 * size * size,
                },
                features=(size / 8.0, float(slippery), probability),
            )
        )
    return tasks


def cliffwalking_tasks(split: str) -> list[V2ExternalTask]:
    horizons = {
        "development": (80, 120, 160, 200),
        "calibration": (90, 130, 170, 210),
        "confirmation": (100, 140, 180, 220),
    }[split]
    return [
        _task(
            domain="gymnasium_cliffwalking",
            split=split,
            index=index,
            environment_id="CliffWalking-v1",
            parameters={"is_slippery": index % 2 == 1, "max_steps": horizon},
            features=(float(index % 2 == 1), horizon / 220.0),
        )
        for index, horizon in enumerate(horizons)
    ]


def taxi_tasks(split: str) -> list[V2ExternalTask]:
    rain = {
        "development": (0.55, 0.70, 0.82, 0.92),
        "calibration": (0.58, 0.73, 0.84, 0.94),
        "confirmation": (0.61, 0.76, 0.86, 0.96),
    }[split]
    fickle = {
        "development": (0.10, 0.20, 0.35, 0.50),
        "calibration": (0.12, 0.24, 0.38, 0.54),
        "confirmation": (0.14, 0.28, 0.42, 0.58),
    }[split]
    return [
        _task(
            domain="gymnasium_taxi",
            split=split,
            index=index,
            environment_id="Taxi-v4",
            parameters={
                "is_rainy": index > 0,
                "rainy_probability": rain[index],
                "fickle_passenger": index >= 2,
                "fickle_probability": fickle[index],
                "max_steps": 200,
            },
            features=(float(index > 0), rain[index], float(index >= 2), fickle[index]),
        )
        for index in range(4)
    ]


def knapsack_tasks(split: str) -> list[V2ExternalTask]:
    base = _split_seed_base(split, 4_000)
    capacities = {
        "development": (160, 220, 280, 340),
        "calibration": (175, 235, 295, 355),
        "confirmation": (190, 250, 310, 370),
    }[split]
    regimes = ("balanced", "bulky", "volatile", "balanced")
    return [
        _task(
            domain="or_gym_online_knapsack",
            split=split,
            index=index,
            environment_id="Knapsack-v3",
            parameters={
                "capacity": capacity,
                "horizon": 35 + 10 * index,
                "catalog_seed": base + index,
                "item_regime": regimes[index],
                "catalog_size": 200,
            },
            features=(capacity / 400.0, (35 + 10 * index) / 65.0, index / 3.0),
        )
        for index, capacity in enumerate(capacities)
    ]


def inventory_tasks(split: str) -> list[V2ExternalTask]:
    base = _split_seed_base(split, 5_000)
    demand_means = {
        "development": (12.0, 18.0, 24.0, 30.0),
        "calibration": (14.0, 20.0, 26.0, 32.0),
        "confirmation": (16.0, 22.0, 28.0, 34.0),
    }[split]
    lead_scales = (0.6, 0.8, 1.0, 1.2)
    return [
        _task(
            domain="or_gym_inventory_management",
            split=split,
            index=index,
            environment_id="InvManagement-v0" if index % 2 == 0 else "InvManagement-v1",
            parameters={
                "periods": 30,
                "demand_mean": demand_mean,
                "lead_time_scale": lead_scales[index],
                "backlog": index % 2 == 0,
                "demand_seed": base + index,
                "profit_lower": -4_000.0,
                "profit_upper": 4_000.0,
            },
            features=(
                demand_mean / 35.0,
                lead_scales[index],
                float(index % 2 == 0),
            ),
        )
        for index, demand_mean in enumerate(demand_means)
    ]


def _safety_tasks(split: str, *, button: bool) -> list[V2ExternalTask]:
    offset = 7_000 if button else 6_000
    base = _split_seed_base(split, offset)
    cost_weights = {
        "development": (0.75, 1.5, 3.0, 6.0),
        "calibration": (1.0, 2.0, 4.0, 8.0),
        "confirmation": (1.25, 2.5, 5.0, 10.0),
    }[split]
    domain = (
        "safety_gymnasium_point_button"
        if button
        else "safety_gymnasium_point_goal"
    )
    task_label = "Button" if button else "Goal"
    levels = (0, 1, 2, 2)
    return [
        _task(
            domain=domain,
            split=split,
            index=index,
            environment_id=f"SafetyPoint{task_label}{levels[index]}-v0",
            parameters={
                "level": levels[index],
                "cost_weight": cost_weights[index],
                "layout_seed_base": base + index * 10_000,
                "max_steps": 500,
            },
            features=(levels[index] / 2.0, cost_weights[index] / 10.0),
        )
        for index in range(4)
    ]


def minigrid_dynamic_tasks(split: str) -> list[V2ExternalTask]:
    base = _split_seed_base(split, 8_000)
    sizes = (5, 6, 8, 16)
    obstacles = (2, 3, 4, 8)
    return [
        _task(
            domain="minigrid_dynamic_obstacles",
            split=split,
            index=index,
            environment_id=f"MiniGrid-Dynamic-Obstacles-{size}x{size}-v0",
            parameters={
                "size": size,
                "n_obstacles": obstacles[index],
                "agent_start_random": index % 2 == 1,
                "layout_seed_base": base + index * 10_000,
                "max_steps": 4 * size * size,
            },
            features=(size / 16.0, obstacles[index] / 8.0, float(index % 2 == 1)),
        )
        for index, size in enumerate(sizes)
    ]


def minigrid_lava_tasks(split: str) -> list[V2ExternalTask]:
    base = _split_seed_base(split, 9_000)
    configurations = ((9, 1), (9, 2), (9, 3), (11, 5))
    return [
        _task(
            domain="minigrid_lava_crossing",
            split=split,
            index=index,
            environment_id=f"MiniGrid-LavaCrossingS{size}N{crossings}-v0",
            parameters={
                "size": size,
                "num_crossings": crossings,
                "layout_seed_base": base + index * 10_000,
                "max_steps": 4 * size * size,
            },
            features=(size / 11.0, crossings / 5.0),
        )
        for index, (size, crossings) in enumerate(configurations)
    ]


def domain_tasks(domain: str, split: str) -> list[V2ExternalTask]:
    if split not in SPLITS:
        raise KeyError(split)
    generators = {
        "gymnasium_frozenlake": frozenlake_tasks,
        "gymnasium_cliffwalking": cliffwalking_tasks,
        "gymnasium_taxi": taxi_tasks,
        "or_gym_online_knapsack": knapsack_tasks,
        "or_gym_inventory_management": inventory_tasks,
        "safety_gymnasium_point_goal": lambda selected: _safety_tasks(
            selected, button=False
        ),
        "safety_gymnasium_point_button": lambda selected: _safety_tasks(
            selected, button=True
        ),
        "minigrid_dynamic_obstacles": minigrid_dynamic_tasks,
        "minigrid_lava_crossing": minigrid_lava_tasks,
    }
    try:
        return generators[domain](split)
    except KeyError as error:
        raise KeyError(domain) from error


def all_tasks(split: str) -> list[V2ExternalTask]:
    return [task for domain in DOMAINS for task in domain_tasks(domain, split)]


def task_sha256(task: V2ExternalTask) -> str:
    return canonical_sha256(asdict(task))


def task_manifest_sha256(tasks: list[V2ExternalTask]) -> str:
    return canonical_sha256([asdict(task) for task in tasks])


def canonical_episode_seed_base(
    task: V2ExternalTask,
    *,
    stream: str | None = None,
) -> int:
    """Return the frozen nonoverlapping seed block for a task and data stream."""

    if stream is None:
        if task.split == "confirmation":
            raise ValueError("confirmation seeds require an explicit pilot or final stream")
        stream = "evaluation"
    try:
        base = STREAM_SEED_BASES[(task.split, stream)]
    except KeyError as error:
        raise ValueError(f"invalid stream {stream!r} for split {task.split!r}") from error
    try:
        domain_index = DOMAINS.index(task.domain)
        task_index = next(
            index
            for index, expected in enumerate(domain_tasks(task.domain, task.split))
            if expected == task
        )
    except (ValueError, StopIteration) as error:
        raise ValueError("task is not an exact member of the frozen v2 manifest") from error
    return base + domain_index * DOMAIN_SEED_STRIDE + task_index * TASK_SEED_STRIDE


def expected_episode_seeds(
    task: V2ExternalTask,
    *,
    episodes: int,
    seed_base: int,
) -> tuple[int, ...]:
    """Mirror the adapter seed transformation for an auditable CRN schedule."""

    if not 0 < episodes <= MAX_EPISODES_PER_TASK_STREAM:
        raise ValueError(
            "episodes must be positive and fit inside the frozen task seed block"
        )
    parameters = task.parameter_dict()
    task_offset = int(
        parameters.get("layout_seed_base", parameters.get("demand_seed", 0))
    )
    return tuple(task_offset + seed_base + episode for episode in range(episodes))


def validate_design() -> None:
    if len(DOMAINS) < 8:
        raise ValueError("v2 requires at least eight external domains")
    codebases = {spec.codebase for spec in DOMAIN_SPECS.values()}
    if len(codebases) < 4:
        raise ValueError("v2 requires at least four independent codebases")
    high_dimensional = [
        spec
        for spec in DOMAIN_SPECS.values()
        if spec.minimum_observation_coordinates >= 32
    ]
    if len(high_dimensional) < 2:
        raise ValueError("v2 requires at least two high-dimensional domains")
    if codebases != set(CODEBASE_LOCKS):
        raise ValueError("domain codebases and codebase locks differ")

    all_names: set[str] = set()
    signatures_by_domain: dict[str, set[tuple]] = {domain: set() for domain in DOMAINS}
    for split in SPLITS:
        for domain in DOMAINS:
            spec = DOMAIN_SPECS[domain]
            if not spec.score_lower < spec.score_upper:
                raise ValueError(f"score bounds are invalid for {domain}")
            if (spec.score_lower, spec.score_upper) != (0.0, 1.0):
                raise ValueError(f"v2 scores must be normalized to [0, 1]: {domain}")
            if spec.fallback_policy in spec.candidate_policies:
                raise ValueError(f"fallback appears in candidate library for {domain}")
            tasks = domain_tasks(domain, split)
            if len(tasks) < 4:
                raise ValueError(f"too few {split} tasks for {domain}")
            for task in tasks:
                if task.name in all_names:
                    raise ValueError(f"duplicate task name: {task.name}")
                all_names.add(task.name)
                if task.domain != domain or task.split != split:
                    raise ValueError(f"task metadata mismatch: {task.name}")
                if not task.features or any(not isfinite(value) for value in task.features):
                    raise ValueError(f"task features are invalid: {task.name}")
                signature = (task.environment_id, task.parameters)
                if signature in signatures_by_domain[domain]:
                    raise ValueError(f"task reused across splits: {task.name}")
                signatures_by_domain[domain].add(signature)

    seed_blocks = []
    for (split, stream), _base in STREAM_SEED_BASES.items():
        for task in all_tasks(split):
            start = expected_episode_seeds(
                task,
                episodes=1,
                seed_base=canonical_episode_seed_base(task, stream=stream),
            )[0]
            seed_blocks.append(
                (start, start + MAX_EPISODES_PER_TASK_STREAM - 1, task.name, stream)
            )
    seed_blocks.sort()
    for left, right in zip(seed_blocks, seed_blocks[1:], strict=False):
        if left[1] >= right[0]:
            raise ValueError(
                f"canonical episode seed blocks overlap: {left[2:]} and {right[2:]}"
            )


def design_summary() -> dict:
    validate_design()
    return {
        "protocol_id": "riskshiftbench-frontier-v2-development-design",
        "scope": (
            "Outcome-free v2 feasibility design. Importing this module does not reset "
            "any environment; confirmation execution remains prohibited until registration."
        ),
        "confirmation_execution": "prohibited_before_external_registration",
        "outcome_implementation_files": list(OUTCOME_IMPLEMENTATION_FILES),
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "codebase_locks": {
            name: asdict(lock) for name, lock in CODEBASE_LOCKS.items()
        },
        "domains": {name: asdict(spec) for name, spec in DOMAIN_SPECS.items()},
        "domain_count": len(DOMAINS),
        "codebase_count": len({spec.codebase for spec in DOMAIN_SPECS.values()}),
        "high_dimensional_domain_count": sum(
            spec.minimum_observation_coordinates >= 32
            for spec in DOMAIN_SPECS.values()
        ),
        "seed_protocol": {
            "stream_bases": [
                {"split": split, "stream": stream, "base": base}
                for (split, stream), base in STREAM_SEED_BASES.items()
            ],
            "domain_stride": DOMAIN_SEED_STRIDE,
            "task_stride": TASK_SEED_STRIDE,
            "maximum_episodes_per_task_stream": MAX_EPISODES_PER_TASK_STREAM,
            "pairing": "common random numbers within task across policies",
            "all_task_stream_blocks_disjoint": True,
        },
        "splits": {
            split: {
                "task_count": len(all_tasks(split)),
                "manifest_sha256": task_manifest_sha256(all_tasks(split)),
                "domains": {
                    domain: {
                        "task_count": len(domain_tasks(domain, split)),
                        "manifest_sha256": task_manifest_sha256(
                            domain_tasks(domain, split)
                        ),
                    }
                    for domain in DOMAINS
                },
            }
            for split in SPLITS
        },
    }


validate_design()
