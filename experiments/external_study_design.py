"""Outcome-free task and policy design for the external confirmation study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TypeAlias


Scalar: TypeAlias = str | int | float | bool

DOMAINS = (
    "gymnasium_frozenlake",
    "or_gym_online_knapsack",
    "safety_gymnasium_point_goal",
)
SPLITS = ("development", "calibration", "confirmation")


@dataclass(frozen=True)
class ExternalEnvironmentLock:
    domain: str
    distribution: str
    version: str
    repository: str
    commit: str
    license: str
    python: str
    compatibility: str


@dataclass(frozen=True)
class ExternalTask:
    name: str
    domain: str
    split: str
    environment_id: str
    parameters: tuple[tuple[str, Scalar], ...]
    features: tuple[float, ...]

    def parameter_dict(self) -> dict[str, Scalar]:
        return dict(self.parameters)


@dataclass(frozen=True)
class ExternalPolicyLibrary:
    fallback: str
    candidates: tuple[str, ...]


ENVIRONMENT_LOCKS = {
    "gymnasium_frozenlake": ExternalEnvironmentLock(
        domain="gymnasium_frozenlake",
        distribution="gymnasium",
        version="1.3.0",
        repository="https://github.com/Farama-Foundation/Gymnasium",
        commit="53bf3e9a884783eb72ad3fc8b15780914c97c3e1",
        license="MIT",
        python="3.12",
        compatibility="Native Gymnasium 1.3 API.",
    ),
    "or_gym_online_knapsack": ExternalEnvironmentLock(
        domain="or_gym_online_knapsack",
        distribution="or-gym",
        version="0.5.0",
        repository="https://github.com/hubbs5/or-gym",
        commit="0b18d16e569e2db70e83f09e867b53bdb4b87298",
        license="MIT",
        python="3.10",
        compatibility=(
            "Pinned OR-Gym transition source loaded with Gym 0.26.2 and NumPy 1.26.4; "
            "the adapter handles the legacy reset/step API without editing OR-Gym source."
        ),
    ),
    "safety_gymnasium_point_goal": ExternalEnvironmentLock(
        domain="safety_gymnasium_point_goal",
        distribution="safety-gymnasium",
        version="1.2.0",
        repository="https://github.com/PKU-Alignment/safety-gymnasium",
        commit="98231340a4c5b223c8d111fa9597d81836ce09b4",
        license="Apache-2.0",
        python="3.10",
        compatibility="Native six-value Safety-Gymnasium API with MuJoCo 2.3.3.",
    ),
}


# Packages whose numerical or transition behavior is part of each execution
# environment but is not captured by the primary distribution version above.
RUNTIME_DEPENDENCIES = {
    "gymnasium_frozenlake": (),
    "or_gym_online_knapsack": (
        ("gym", "0.26.2"),
        ("numpy", "1.26.4"),
    ),
    "safety_gymnasium_point_goal": (
        ("gymnasium", "0.28.1"),
        ("gymnasium-robotics", "1.2.2"),
        ("mujoco", "2.3.3"),
        ("numpy", "1.23.5"),
    ),
}


POLICY_LIBRARIES = {
    "gymnasium_frozenlake": ExternalPolicyLibrary(
        fallback="nominal_value_iteration",
        candidates=("hazard_averse_value_iteration", "short_path_value_iteration"),
    ),
    "or_gym_online_knapsack": ExternalPolicyLibrary(
        fallback="ratio_threshold_1_25",
        candidates=("dynamic_reserve", "ratio_threshold_2_0"),
    ),
    "safety_gymnasium_point_goal": ExternalPolicyLibrary(
        fallback="goal_greedy",
        candidates=("hazard_aware_moderate", "hazard_aware_strict"),
    ),
}


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_manifest_sha256(tasks: list[ExternalTask]) -> str:
    return canonical_sha256([asdict(task) for task in tasks])


def _task(
    *,
    name: str,
    domain: str,
    split: str,
    environment_id: str,
    parameters: dict[str, Scalar],
    features: tuple[float, ...],
) -> ExternalTask:
    return ExternalTask(
        name=name,
        domain=domain,
        split=split,
        environment_id=environment_id,
        parameters=tuple(sorted(parameters.items())),
        features=features,
    )


def frozenlake_tasks(split: str) -> list[ExternalTask]:
    if split == "development":
        rates = (0.40, 0.60, 0.80)
    elif split == "calibration":
        rates = (0.45, 0.65)
    elif split == "confirmation":
        rates = (0.35, 0.55, 0.75)
    else:
        raise KeyError(split)
    tasks = []
    for map_size in (4, 8):
        for slippery in (False, True):
            for rate in rates:
                name = (
                    f"ExternalFrozenLake-{split}-m{map_size}-"
                    f"{'slip' if slippery else 'det'}-r{int(rate * 100):02d}-v0"
                )
                tasks.append(
                    _task(
                        name=name,
                        domain="gymnasium_frozenlake",
                        split=split,
                        environment_id="FrozenLake-v1",
                        parameters={
                            "map_name": f"{map_size}x{map_size}",
                            "map_size": map_size,
                            "is_slippery": slippery,
                            "success_rate": rate,
                            "max_steps": 100 if map_size == 4 else 200,
                        },
                        features=(map_size / 8.0, float(slippery), rate),
                    )
                )
    return tasks


def _knapsack_regime_features(regime: str) -> tuple[float, float, float]:
    return {
        "balanced": (1.0, 0.0, 0.0),
        "bulky": (0.0, 1.0, 0.0),
        "volatile": (0.0, 0.0, 1.0),
    }[regime]


def knapsack_tasks(split: str) -> list[ExternalTask]:
    grids = {
        "development": ((180, 300), (30, 50), 11_000),
        "calibration": ((240, 360), (40,), 21_000),
        "confirmation": ((210, 330), (35, 55), 31_000),
    }
    if split not in grids:
        raise KeyError(split)
    capacities, horizons, seed_base = grids[split]
    tasks = []
    index = 0
    for capacity in capacities:
        for horizon in horizons:
            for regime in ("balanced", "bulky", "volatile"):
                catalog_seed = seed_base + index
                index += 1
                tasks.append(
                    _task(
                        name=(
                            f"ExternalKnapsack-{split}-c{capacity}-h{horizon}-"
                            f"{regime}-s{catalog_seed}-v0"
                        ),
                        domain="or_gym_online_knapsack",
                        split=split,
                        environment_id="Knapsack-v3",
                        parameters={
                            "capacity": capacity,
                            "horizon": horizon,
                            "catalog_seed": catalog_seed,
                            "item_regime": regime,
                            "catalog_size": 200,
                        },
                        features=(
                            capacity / 400.0,
                            horizon / 60.0,
                            *_knapsack_regime_features(regime),
                        ),
                    )
                )
    return tasks


def _safety_level_features(level: int) -> tuple[float, float, float]:
    hazards = (0, 8, 10)[level]
    vases = (0, 1, 10)[level]
    return level / 2.0, hazards / 10.0, vases / 10.0


def safety_goal_tasks(split: str) -> list[ExternalTask]:
    grids = {
        "development": ((0.5, 2.0, 5.0), 41_000),
        "calibration": ((1.0, 3.0), 51_000),
        "confirmation": ((0.75, 2.5, 6.0), 61_000),
    }
    if split not in grids:
        raise KeyError(split)
    cost_weights, seed_base = grids[split]
    tasks = []
    index = 0
    for level in (0, 1, 2):
        for cost_weight in cost_weights:
            layout_seed_base = seed_base + index * 10_000
            index += 1
            tasks.append(
                _task(
                    name=(
                        f"ExternalPointGoal-{split}-l{level}-"
                        f"cw{str(cost_weight).replace('.', 'p')}-s{layout_seed_base}-v0"
                    ),
                    domain="safety_gymnasium_point_goal",
                    split=split,
                    environment_id=f"SafetyPointGoal{level}-v0",
                    parameters={
                        "level": level,
                        "cost_weight": cost_weight,
                        "layout_seed_base": layout_seed_base,
                        "max_steps": 500,
                    },
                    features=(*_safety_level_features(level), cost_weight / 6.0),
                )
            )
    return tasks


def domain_tasks(domain: str, split: str) -> list[ExternalTask]:
    if domain == "gymnasium_frozenlake":
        return frozenlake_tasks(split)
    if domain == "or_gym_online_knapsack":
        return knapsack_tasks(split)
    if domain == "safety_gymnasium_point_goal":
        return safety_goal_tasks(split)
    raise KeyError(domain)


def all_tasks(split: str) -> list[ExternalTask]:
    return [task for domain in DOMAINS for task in domain_tasks(domain, split)]


def design_summary() -> dict:
    return {
        "scope": "Outcome-free external study design; confirmation environments have not been reset.",
        "environment_locks": {
            domain: asdict(lock) for domain, lock in ENVIRONMENT_LOCKS.items()
        },
        "runtime_dependencies": {
            domain: dict(RUNTIME_DEPENDENCIES[domain]) for domain in DOMAINS
        },
        "policy_libraries": {
            domain: asdict(library) for domain, library in POLICY_LIBRARIES.items()
        },
        "splits": {
            split: {
                domain: {
                    "task_count": len(domain_tasks(domain, split)),
                    "task_manifest_sha256": task_manifest_sha256(domain_tasks(domain, split)),
                }
                for domain in DOMAINS
            }
            for split in SPLITS
        },
    }
