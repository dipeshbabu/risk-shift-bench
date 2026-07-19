"""Prospective competitive-baseline protocol for RiskShiftBench v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.frontier_v2_baseline_hash import (
    BASELINE_IMPLEMENTATION_FILES,
    baseline_implementation_sha256,
)
from experiments.frontier_v2_external_design import (
    DOMAIN_SPECS,
    canonical_sha256,
)


@dataclass(frozen=True)
class BaselineSourceLock:
    name: str
    repository: str
    commit: str
    license: str
    role: str


@dataclass(frozen=True)
class CompetitiveBaselineSpec:
    domain: str
    name: str
    kind: str
    algorithm: str
    implementation_source: str
    observation: str
    training_steps_per_seed: int
    training_seeds: tuple[int, ...]
    checkpoint_interval_steps: int
    checkpoint_selection_rule: str
    safety_cost_limit: float | None = None

    @property
    def identifier(self) -> str:
        return f"{self.domain}:{self.name}"


BASELINE_SOURCE_LOCKS = {
    "omnisafe": BaselineSourceLock(
        name="omnisafe",
        repository="https://github.com/PKU-Alignment/omnisafe",
        commit="15603dd7a654a991d0a4648216b69d60b81a6366",
        license="Apache-2.0",
        role="PPO-Lagrangian and CPO Safety-Gymnasium references",
    ),
    "rl_starter_files": BaselineSourceLock(
        name="rl_starter_files",
        repository="https://github.com/lcswillems/rl-starter-files",
        commit="317da04a9a6fb26506bbd7f6c7c7e10fc0de86e0",
        license="MIT",
        role="MiniGrid PPO reference recommended by the upstream benchmark",
    ),
    "cleanrl": BaselineSourceLock(
        name="cleanrl",
        repository="https://github.com/vwxyzjn/cleanrl",
        commit="fe8d8a03c41a7ef5b523e2e354bd01c363e786bb",
        license="MIT",
        role="Transparent single-file DQN and PPO references",
    ),
}


TRAINING_SEEDS = (50_000_000, 51_000_000, 52_000_000, 53_000_000, 54_000_000)
SCORE_SELECTION = (
    "At each frozen checkpoint, evaluate 100 episodes on every calibration task. "
    "Select the checkpoint with the largest equal-task mean normalized score; "
    "break exact ties by fewer training steps and then smaller training seed."
)
SAFE_SELECTION = (
    "At each frozen checkpoint, evaluate 100 episodes on every calibration task. "
    "Among checkpoints with equal-task mean episode cost at most 25, select the "
    "largest equal-task mean normalized score. If none is feasible, select minimum "
    "cost and then maximum score. Break remaining ties by fewer steps and seed."
)
ANALYTIC_SELECTION = "No learned checkpoint; evaluate the frozen analytic rule directly."
ROUTER_SELECTION = "Use the completed v1 fixed router without refitting or outcome access."


def _learned(
    domain: str,
    name: str,
    *,
    algorithm: str,
    source: str,
    observation: str,
    steps: int,
    safe: bool = False,
) -> CompetitiveBaselineSpec:
    return CompetitiveBaselineSpec(
        domain=domain,
        name=name,
        kind="learned_policy",
        algorithm=algorithm,
        implementation_source=source,
        observation=observation,
        training_steps_per_seed=steps,
        training_seeds=TRAINING_SEEDS,
        checkpoint_interval_steps=50_000,
        checkpoint_selection_rule=SAFE_SELECTION if safe else SCORE_SELECTION,
        safety_cost_limit=25.0 if safe else None,
    )


def _analytic(
    domain: str,
    name: str,
    *,
    algorithm: str,
    observation: str,
    router: bool = False,
) -> CompetitiveBaselineSpec:
    return CompetitiveBaselineSpec(
        domain=domain,
        name=name,
        kind="router_reference" if router else "analytic_reference",
        algorithm=algorithm,
        implementation_source="riskshiftbench_internal",
        observation=observation,
        training_steps_per_seed=0,
        training_seeds=(),
        checkpoint_interval_steps=0,
        checkpoint_selection_rule=ROUTER_SELECTION if router else ANALYTIC_SELECTION,
    )


COMPETITIVE_BASELINES = (
    _analytic(
        "gymnasium_frozenlake",
        "tabular_oracle",
        algorithm="true-task value iteration upper reference",
        observation="discrete state",
    ),
    _analytic(
        "gymnasium_frozenlake",
        "v1_fixed_router",
        algorithm="completed RiskShiftBench v1 fixed routing rule",
        observation="frozen task features",
        router=True,
    ),
    _analytic(
        "gymnasium_cliffwalking",
        "tabular_oracle",
        algorithm="true-task value iteration upper reference",
        observation="discrete state",
    ),
    _learned(
        "gymnasium_cliffwalking",
        "q_learning_reference",
        algorithm="tabular Q-learning",
        source="riskshiftbench_internal",
        observation="discrete state",
        steps=500_000,
    ),
    _analytic(
        "gymnasium_taxi",
        "tabular_oracle",
        algorithm="true-task value iteration upper reference",
        observation="discrete state",
    ),
    _learned(
        "gymnasium_taxi",
        "q_learning_reference",
        algorithm="tabular Q-learning",
        source="riskshiftbench_internal",
        observation="discrete state",
        steps=500_000,
    ),
    _analytic(
        "or_gym_online_knapsack",
        "fractional_oracle",
        algorithm="episode fractional-knapsack upper bound",
        observation="complete episode item sequence; upper reference only",
    ),
    _learned(
        "or_gym_online_knapsack",
        "rl_policy_reference",
        algorithm="double DQN",
        source="cleanrl",
        observation="normalized capacity, item, and time vector",
        steps=1_000_000,
    ),
    _analytic(
        "or_gym_inventory_management",
        "newsvendor_base_stock",
        algorithm="lead-time adjusted newsvendor base-stock rule",
        observation="pipeline inventory vector",
    ),
    _learned(
        "or_gym_inventory_management",
        "ppo_reference",
        algorithm="clipped PPO",
        source="cleanrl",
        observation="normalized pipeline inventory vector",
        steps=1_000_000,
    ),
    _learned(
        "safety_gymnasium_point_goal",
        "ppo_lagrangian_reference",
        algorithm="PPO-Lagrangian",
        source="omnisafe",
        observation="flattened lidar and proprioception",
        steps=1_000_000,
        safe=True,
    ),
    _learned(
        "safety_gymnasium_point_goal",
        "cpo_reference",
        algorithm="constrained policy optimization",
        source="omnisafe",
        observation="flattened lidar and proprioception",
        steps=1_000_000,
        safe=True,
    ),
    _learned(
        "safety_gymnasium_point_button",
        "ppo_lagrangian_reference",
        algorithm="PPO-Lagrangian",
        source="omnisafe",
        observation="flattened lidar and proprioception",
        steps=1_000_000,
        safe=True,
    ),
    _learned(
        "safety_gymnasium_point_button",
        "cpo_reference",
        algorithm="constrained policy optimization",
        source="omnisafe",
        observation="flattened lidar and proprioception",
        steps=1_000_000,
        safe=True,
    ),
    _learned(
        "minigrid_dynamic_obstacles",
        "dqn_reference",
        algorithm="double DQN",
        source="cleanrl",
        observation="fully observable compact image",
        steps=1_000_000,
    ),
    _learned(
        "minigrid_dynamic_obstacles",
        "ppo_reference",
        algorithm="recurrent PPO",
        source="rl_starter_files",
        observation="fully observable compact image",
        steps=1_000_000,
    ),
    _learned(
        "minigrid_lava_crossing",
        "dqn_reference",
        algorithm="double DQN",
        source="cleanrl",
        observation="fully observable compact image",
        steps=1_000_000,
    ),
    _learned(
        "minigrid_lava_crossing",
        "ppo_reference",
        algorithm="recurrent PPO",
        source="rl_starter_files",
        observation="fully observable compact image",
        steps=1_000_000,
    ),
)


def validate_baseline_design() -> None:
    identifiers = [baseline.identifier for baseline in COMPETITIVE_BASELINES]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("competitive baseline identifiers must be unique")
    for domain, domain_spec in DOMAIN_SPECS.items():
        observed = {
            baseline.name
            for baseline in COMPETITIVE_BASELINES
            if baseline.domain == domain
        }
        if observed != set(domain_spec.competitive_baselines):
            raise ValueError(f"competitive baseline coverage changed for {domain}")
    for baseline in COMPETITIVE_BASELINES:
        if baseline.kind == "learned_policy":
            if baseline.training_steps_per_seed < 500_000:
                raise ValueError(f"training budget is too small for {baseline.identifier}")
            if baseline.training_seeds != TRAINING_SEEDS:
                raise ValueError(f"training seeds changed for {baseline.identifier}")
            if baseline.checkpoint_interval_steps <= 0:
                raise ValueError(f"checkpoint interval missing for {baseline.identifier}")
            if baseline.implementation_source not in {
                "riskshiftbench_internal",
                *BASELINE_SOURCE_LOCKS,
            }:
                raise ValueError(f"unknown source for {baseline.identifier}")
        elif baseline.training_steps_per_seed != 0 or baseline.training_seeds:
            raise ValueError(f"nonlearned baseline has a training budget: {baseline.identifier}")
        if baseline.algorithm in {"PPO-Lagrangian", "constrained policy optimization"}:
            if baseline.safety_cost_limit != 25.0:
                raise ValueError(f"safe baseline cost limit changed: {baseline.identifier}")


def baseline_design_summary() -> dict:
    validate_baseline_design()
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-competitive-baselines-v1",
        "scope": (
            "Development and calibration training protocol only. All selected "
            "checkpoints must be frozen before confirmation execution."
        ),
        "source_locks": {
            name: asdict(lock) for name, lock in BASELINE_SOURCE_LOCKS.items()
        },
        "internal_implementation_files": list(BASELINE_IMPLEMENTATION_FILES),
        "internal_implementation_sha256": baseline_implementation_sha256(),
        "training_task_split": "development",
        "checkpoint_selection_split": "calibration",
        "training_seed_count": len(TRAINING_SEEDS),
        "baselines": [asdict(baseline) for baseline in COMPETITIVE_BASELINES],
    }
    payload["design_sha256"] = canonical_sha256(payload)
    return payload


validate_baseline_design()
