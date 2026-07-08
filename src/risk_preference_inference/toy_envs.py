"""Non-Blackjack finite stochastic tasks for risk-objective validation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from risk_preference_inference.adaptive_risk import AdaptiveCVaRObjective
from risk_preference_inference.objectives import (
    CVaRObjective,
    Distribution,
    MeanObjective,
    ObjectiveContext,
    cvar_lower,
    mean,
    normalize,
    probability_at_or_above,
    probability_at_or_below,
)


@dataclass(frozen=True)
class ToyTask:
    name: str
    horizon: int
    initial_wealth: float
    ruin_wealth: float
    target_wealth: float


@dataclass(frozen=True)
class ToyPolicy:
    name: str
    objective: object

    def choose(self, state: "ToyState", task: ToyTask) -> str:
        distributions = toy_action_distributions(state, task)
        context = ObjectiveContext(
            bankroll=state.wealth,
            initial_bankroll=task.initial_wealth,
            ruin_bankroll=task.ruin_wealth,
            target_bankroll=task.target_wealth,
            peak_bankroll=state.peak_wealth,
            rounds_remaining=task.horizon - state.step,
        )
        scores = {action: self.objective.score(distribution, context) for action, distribution in distributions.items()}
        return max(scores, key=scores.get)


@dataclass(frozen=True)
class ToyState:
    wealth: float
    peak_wealth: float
    step: int


@dataclass(frozen=True)
class ToyEpisodeResult:
    task: str
    policy: str
    seed: int
    final_wealth: float
    ruined: bool
    target_hit: bool
    max_drawdown: float


def toy_tasks() -> list[ToyTask]:
    return [
        ToyTask("GamblerRuin-v0", horizon=30, initial_wealth=50.0, ruin_wealth=0.0, target_wealth=100.0),
        ToyTask("InventoryRisk-v0", horizon=20, initial_wealth=80.0, ruin_wealth=0.0, target_wealth=130.0),
    ]


def toy_policies() -> list[ToyPolicy]:
    return [
        ToyPolicy("toy_expected_value", MeanObjective()),
        ToyPolicy("toy_cvar_05", CVaRObjective(alpha=0.05)),
        ToyPolicy("toy_adaptive_cvar", AdaptiveCVaRObjective()),
    ]


def toy_action_distributions(state: ToyState, task: ToyTask) -> dict[str, Distribution]:
    if task.name == "GamblerRuin-v0":
        return {
            "safe": normalize(((state.wealth + 1.0, 0.52), (state.wealth - 1.0, 0.48))),
            "risky": normalize(((state.wealth + 6.0, 0.42), (state.wealth - 5.0, 0.58))),
        }
    if task.name == "InventoryRisk-v0":
        demand_shock = ((12.0, 0.25), (0.0, 0.50), (-18.0, 0.25))
        return {
            "conservative": normalize(tuple((state.wealth + value + 2.0, prob) for value, prob in demand_shock)),
            "aggressive": normalize(tuple((state.wealth + 2.2 * value + 8.0, prob) for value, prob in demand_shock)),
        }
    raise ValueError(f"Unknown toy task: {task.name}")


def simulate_toy_episode(task: ToyTask, policy: ToyPolicy, seed: int) -> ToyEpisodeResult:
    rng = random.Random(seed)
    state = ToyState(task.initial_wealth, task.initial_wealth, 0)
    min_wealth = state.wealth
    max_drawdown = 0.0
    target_hit = state.wealth >= task.target_wealth
    for step in range(task.horizon):
        if state.wealth <= task.ruin_wealth or state.wealth >= task.target_wealth:
            break
        action = policy.choose(state, task)
        distribution = toy_action_distributions(state, task)[action]
        threshold = rng.random()
        cumulative = 0.0
        next_wealth = state.wealth
        for value, prob in distribution:
            cumulative += prob
            if threshold <= cumulative:
                next_wealth = value
                break
        state = ToyState(next_wealth, max(state.peak_wealth, next_wealth), step + 1)
        min_wealth = min(min_wealth, next_wealth)
        max_drawdown = max(max_drawdown, state.peak_wealth - state.wealth)
        target_hit = target_hit or state.wealth >= task.target_wealth
    return ToyEpisodeResult(
        task=task.name,
        policy=policy.name,
        seed=seed,
        final_wealth=state.wealth,
        ruined=state.wealth <= task.ruin_wealth,
        target_hit=target_hit,
        max_drawdown=max_drawdown,
    )


def run_toy_benchmark(episodes: int = 100, seed: int = 0) -> tuple[list[ToyEpisodeResult], list[dict]]:
    results = []
    summaries = []
    for task_idx, task in enumerate(toy_tasks()):
        for policy_idx, policy in enumerate(toy_policies()):
            policy_results = [
                simulate_toy_episode(task, policy, seed + task_idx * 100_000 + policy_idx * 10_000 + idx)
                for idx in range(episodes)
            ]
            results.extend(policy_results)
            distribution = normalize(tuple((result.final_wealth, 1.0) for result in policy_results))
            summaries.append(
                {
                    "task": task.name,
                    "policy": policy.name,
                    "episodes": episodes,
                    "mean_final_wealth": mean(distribution),
                    "cvar_5_final_wealth": cvar_lower(distribution, 0.05),
                    "ruin_probability": sum(1.0 for result in policy_results if result.ruined) / episodes,
                    "target_probability": sum(1.0 for result in policy_results if result.target_hit) / episodes,
                    "mean_max_drawdown": sum(result.max_drawdown for result in policy_results) / episodes,
                }
            )
    return results, summaries


def toy_results_as_dicts(results: list[ToyEpisodeResult]) -> list[dict]:
    return [asdict(result) for result in results]
