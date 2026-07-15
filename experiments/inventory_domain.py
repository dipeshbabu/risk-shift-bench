"""Inventory-control domain for cross-domain risk-routing evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt

from risk_shift_bench.objectives import cvar_lower, mean, normalize


DemandDistribution = tuple[tuple[int, float], ...]
DemandRegimeDistribution = tuple[tuple[DemandDistribution, float], ...]


def normalize_demand(values: DemandDistribution | dict[int, float]) -> DemandDistribution:
    items = tuple(values.items()) if isinstance(values, dict) else tuple(values)
    total = sum(probability for _demand, probability in items)
    if total <= 0.0:
        raise ValueError("demand probability mass must be positive")
    return tuple(sorted((int(demand), float(probability) / total) for demand, probability in items))


LOW_DEMAND = normalize_demand({0: 0.10, 1: 0.30, 2: 0.40, 3: 0.20})
STEADY_DEMAND = normalize_demand({2: 0.10, 3: 0.25, 4: 0.35, 5: 0.20, 6: 0.10})
HIGH_DEMAND = normalize_demand({4: 0.10, 5: 0.20, 6: 0.30, 7: 0.25, 8: 0.15})
VOLATILE_DEMAND = normalize_demand({0: 0.20, 2: 0.20, 5: 0.25, 8: 0.20, 12: 0.15})
SPIKE_DEMAND = normalize_demand({1: 0.35, 3: 0.25, 7: 0.20, 12: 0.15, 16: 0.05})
INTERMITTENT_DEMAND = normalize_demand({0: 0.45, 2: 0.20, 5: 0.20, 9: 0.15})


@dataclass(frozen=True)
class InventoryTask:
    name: str
    periods: int = 24
    initial_cash: float = 900.0
    initial_inventory: int = 4
    max_inventory: int = 18
    unit_cost: float = 8.0
    unit_price: float = 15.0
    holding_cost: float = 0.75
    shortage_penalty: float = 2.0
    salvage_value: float = 4.0
    bankruptcy_cash: float = 0.0
    target_wealth: float = 1250.0
    demand: DemandDistribution = STEADY_DEMAND
    episode_regimes: DemandRegimeDistribution | None = None


@dataclass(frozen=True)
class InventoryState:
    cash: float
    inventory: int
    peak_wealth: float
    periods_remaining: int


@dataclass(frozen=True)
class InventoryEpisodeResult:
    task: str
    policy: str
    seed: int
    final_bankroll: float
    min_bankroll: float
    max_drawdown: float
    ruined: bool
    target_hit: bool
    rounds_played: int


@dataclass(frozen=True)
class InventorySummary:
    task: str
    policy: str
    episodes: int
    mean_final_bankroll: float
    std_final_bankroll: float
    cvar_5_final_bankroll: float
    ruin_probability: float
    target_probability: float
    mean_max_drawdown: float
    mean_rounds_played: float


def demand_mean(distribution: DemandDistribution) -> float:
    return sum(demand * probability for demand, probability in distribution)


def demand_std(distribution: DemandDistribution) -> float:
    expected = demand_mean(distribution)
    return sqrt(sum(probability * (demand - expected) ** 2 for demand, probability in distribution))


def demand_quantile(distribution: DemandDistribution, quantile: float) -> int:
    cumulative = 0.0
    for demand, probability in distribution:
        cumulative += probability
        if cumulative >= quantile:
            return demand
    return distribution[-1][0]


def visible_demand(task: InventoryTask) -> DemandDistribution:
    if task.episode_regimes is None:
        return task.demand
    weights: dict[int, float] = {}
    for distribution, regime_probability in task.episode_regimes:
        for demand, probability in distribution:
            weights[demand] = weights.get(demand, 0.0) + regime_probability * probability
    return normalize_demand(weights)


def hidden_demand_stats(task: InventoryTask) -> tuple[float, float, float]:
    if task.episode_regimes is None:
        return 0.0, 0.0, 0.0
    means = [demand_mean(distribution) for distribution, _probability in task.episode_regimes]
    return min(means), max(means), max(means) - min(means)


class InventoryPolicy:
    name = "inventory_policy"

    def order_quantity(self, state: InventoryState, task: InventoryTask) -> int:
        raise NotImplementedError


def feasible_order(target_inventory: int, state: InventoryState, task: InventoryTask) -> int:
    desired = max(0, min(task.max_inventory, target_inventory) - state.inventory)
    affordable = max(0, int((state.cash - task.bankruptcy_cash) / max(task.unit_cost, 1e-9)))
    return min(desired, affordable)


@dataclass(frozen=True)
class MeanBaseStockPolicy(InventoryPolicy):
    name: str = "mean_base_stock"

    def order_quantity(self, state: InventoryState, task: InventoryTask) -> int:
        target = round(demand_mean(visible_demand(task)))
        if state.periods_remaining == 1:
            target = max(0, target - 1)
        return feasible_order(target, state, task)


@dataclass(frozen=True)
class ServiceLevelPolicy(InventoryPolicy):
    quantile: float = 0.90
    name: str = "service_level_90"

    def order_quantity(self, state: InventoryState, task: InventoryTask) -> int:
        target = demand_quantile(visible_demand(task), self.quantile)
        if state.periods_remaining <= 2:
            target = max(0, target - 1)
        return feasible_order(target, state, task)


@dataclass(frozen=True)
class CashGuardPolicy(InventoryPolicy):
    name: str = "cash_guard"

    def order_quantity(self, state: InventoryState, task: InventoryTask) -> int:
        target = demand_quantile(visible_demand(task), 0.50)
        cash_buffer = state.cash - task.bankruptcy_cash
        if cash_buffer < 8.0 * task.unit_cost:
            target = max(0, target - 2)
        return feasible_order(target, state, task)


@dataclass(frozen=True)
class TargetChasingPolicy(InventoryPolicy):
    name: str = "target_chasing"

    def order_quantity(self, state: InventoryState, task: InventoryTask) -> int:
        distribution = visible_demand(task)
        target = demand_quantile(distribution, 0.80)
        current_wealth = state.cash + task.salvage_value * state.inventory
        remaining_gap = task.target_wealth - current_wealth
        expected_margin = max(task.unit_price - task.unit_cost, 1.0)
        if remaining_gap / max(state.periods_remaining, 1) > expected_margin * demand_mean(distribution) * 0.5:
            target += 1
        return feasible_order(target, state, task)


@dataclass(frozen=True)
class RobustHiddenTailPolicy(InventoryPolicy):
    name: str = "robust_hidden_tail"

    def order_quantity(self, state: InventoryState, task: InventoryTask) -> int:
        distributions = (
            [distribution for distribution, _probability in task.episode_regimes]
            if task.episode_regimes is not None
            else [task.demand]
        )
        target = max(demand_quantile(distribution, 0.75) for distribution in distributions)
        return feasible_order(target, state, task)


@dataclass(frozen=True)
class AdaptiveBaseStockPolicy(InventoryPolicy):
    name: str = "adaptive_base_stock"

    def order_quantity(self, state: InventoryState, task: InventoryTask) -> int:
        distribution = visible_demand(task)
        target = round(demand_mean(distribution) + 0.50 * demand_std(distribution))
        hidden_span = hidden_demand_stats(task)[2]
        if hidden_span >= 3.0:
            target += 1
        cash_buffer = state.cash - task.bankruptcy_cash
        if cash_buffer < 8.0 * task.unit_cost:
            target = max(0, target - 2)
        current_wealth = state.cash + task.salvage_value * state.inventory
        if current_wealth < task.target_wealth and state.periods_remaining <= 6:
            target += 1
        if state.periods_remaining == 1:
            target = max(0, target - 2)
        return feasible_order(target, state, task)


def inventory_policy_grid() -> list[InventoryPolicy]:
    return [
        AdaptiveBaseStockPolicy(),
        MeanBaseStockPolicy(),
        ServiceLevelPolicy(),
        CashGuardPolicy(),
        TargetChasingPolicy(),
        RobustHiddenTailPolicy(),
    ]


def inventory_policy_lookup() -> dict[str, InventoryPolicy]:
    return {policy.name: policy for policy in inventory_policy_grid()}


def sample_distribution(distribution, rng: random.Random):
    threshold = rng.random()
    cumulative = 0.0
    for value, probability in distribution:
        cumulative += probability
        if threshold <= cumulative:
            return value
    return distribution[-1][0]


def sample_episode_demand(task: InventoryTask, rng: random.Random) -> DemandDistribution:
    if task.episode_regimes is None:
        return task.demand
    return sample_distribution(task.episode_regimes, rng)


def simulate_inventory_episode(task: InventoryTask, policy: InventoryPolicy, seed: int) -> InventoryEpisodeResult:
    rng = random.Random(seed)
    realized_demand = sample_episode_demand(task, rng)
    cash = task.initial_cash
    inventory = task.initial_inventory
    wealth = cash + task.salvage_value * inventory
    peak_wealth = wealth
    min_wealth = wealth
    max_drawdown = 0.0
    target_hit = wealth >= task.target_wealth
    rounds_played = 0
    for period in range(task.periods):
        if cash <= task.bankruptcy_cash:
            break
        state = InventoryState(
            cash=cash,
            inventory=inventory,
            peak_wealth=peak_wealth,
            periods_remaining=task.periods - period,
        )
        order = policy.order_quantity(state, task)
        cash -= order * task.unit_cost
        inventory += order
        demand = sample_distribution(realized_demand, rng)
        sales = min(inventory, demand)
        shortage = max(0, demand - inventory)
        inventory -= sales
        cash += sales * task.unit_price
        cash -= inventory * task.holding_cost
        cash -= shortage * task.shortage_penalty
        wealth = cash + task.salvage_value * inventory
        peak_wealth = max(peak_wealth, wealth)
        min_wealth = min(min_wealth, wealth)
        max_drawdown = max(max_drawdown, peak_wealth - wealth)
        target_hit = target_hit or wealth >= task.target_wealth
        rounds_played += 1
    final_wealth = cash + task.salvage_value * inventory
    return InventoryEpisodeResult(
        task=task.name,
        policy=policy.name,
        seed=seed,
        final_bankroll=final_wealth,
        min_bankroll=min_wealth,
        max_drawdown=max_drawdown,
        ruined=cash <= task.bankruptcy_cash,
        target_hit=target_hit,
        rounds_played=rounds_played,
    )


def summarize_inventory_results(
    results: list[InventoryEpisodeResult],
    task: InventoryTask,
    policy: InventoryPolicy,
) -> InventorySummary:
    finals = normalize(tuple((result.final_bankroll, 1.0) for result in results))
    mean_final = mean(finals)
    variance = sum(probability * (value - mean_final) ** 2 for value, probability in finals)
    return InventorySummary(
        task=task.name,
        policy=policy.name,
        episodes=len(results),
        mean_final_bankroll=mean_final,
        std_final_bankroll=variance**0.5,
        cvar_5_final_bankroll=cvar_lower(finals, 0.05),
        ruin_probability=sum(result.ruined for result in results) / max(len(results), 1),
        target_probability=sum(result.target_hit for result in results) / max(len(results), 1),
        mean_max_drawdown=sum(result.max_drawdown for result in results) / max(len(results), 1),
        mean_rounds_played=sum(result.rounds_played for result in results) / max(len(results), 1),
    )


def run_inventory_benchmark(
    tasks: list[InventoryTask],
    policies: list[InventoryPolicy] | None = None,
    episodes: int = 100,
    seed: int = 0,
) -> tuple[list[InventoryEpisodeResult], list[InventorySummary]]:
    policy_list = inventory_policy_grid() if policies is None else policies
    all_results = []
    summaries = []
    for task_index, task in enumerate(tasks):
        for policy in policy_list:
            results = [
                simulate_inventory_episode(
                    task,
                    policy,
                    seed=seed + task_index * 100_000 + episode_index,
                )
                for episode_index in range(episodes)
            ]
            all_results.extend(results)
            summaries.append(summarize_inventory_results(results, task, policy))
    return all_results, summaries


def inventory_task_features(task: InventoryTask) -> tuple[float, ...]:
    distribution = visible_demand(task)
    hidden_min, hidden_max, hidden_span = hidden_demand_stats(task)
    return (
        task.periods / 48.0,
        task.initial_cash / 1200.0,
        task.initial_inventory / max(task.max_inventory, 1),
        task.unit_cost / max(task.unit_price, 1.0),
        task.holding_cost / max(task.unit_price, 1.0),
        task.shortage_penalty / max(task.unit_price, 1.0),
        (task.target_wealth - task.initial_cash) / max(task.initial_cash, 1.0),
        demand_mean(distribution) / 8.0,
        demand_std(distribution) / 5.0,
        1.0 if task.episode_regimes is not None else 0.0,
        hidden_min / 8.0,
        hidden_max / 8.0,
        hidden_span / 8.0,
    )


def inventory_development_tasks() -> list[InventoryTask]:
    balanced = ((LOW_DEMAND, 0.25), (STEADY_DEMAND, 0.35), (HIGH_DEMAND, 0.25), (VOLATILE_DEMAND, 0.15))
    return [
        InventoryTask(name="RiskInventory-Dev-Steady-v0", demand=STEADY_DEMAND),
        InventoryTask(name="RiskInventory-Dev-Low-v0", demand=LOW_DEMAND),
        InventoryTask(name="RiskInventory-Dev-High-v0", demand=HIGH_DEMAND),
        InventoryTask(name="RiskInventory-Dev-Volatile-v0", demand=VOLATILE_DEMAND),
        InventoryTask(name="RiskInventory-Dev-Spike-v0", demand=SPIKE_DEMAND),
        InventoryTask(name="RiskInventory-Dev-Intermittent-v0", demand=INTERMITTENT_DEMAND),
        InventoryTask(name="RiskInventory-Dev-Long-v0", periods=42, target_wealth=1500.0),
        InventoryTask(name="RiskInventory-Dev-LowCash-v0", initial_cash=520.0, target_wealth=900.0),
        InventoryTask(name="RiskInventory-Dev-HighMargin-v0", unit_price=18.0, target_wealth=1450.0),
        InventoryTask(name="RiskInventory-Dev-HoldingCost-v0", holding_cost=2.0, demand=VOLATILE_DEMAND),
        InventoryTask(name="RiskInventory-Dev-ShortagePenalty-v0", shortage_penalty=7.0, demand=HIGH_DEMAND),
        InventoryTask(name="RiskInventory-Dev-HiddenBalanced-v0", episode_regimes=balanced),
    ]


def inventory_calibration_tasks() -> list[InventoryTask]:
    balanced = ((LOW_DEMAND, 0.20), (STEADY_DEMAND, 0.30), (HIGH_DEMAND, 0.30), (VOLATILE_DEMAND, 0.20))
    tail = ((INTERMITTENT_DEMAND, 0.25), (STEADY_DEMAND, 0.25), (SPIKE_DEMAND, 0.30), (HIGH_DEMAND, 0.20))
    return [
        InventoryTask(name="RiskInventory-Cal-SteadyLong-v0", periods=36, demand=STEADY_DEMAND, target_wealth=1450.0),
        InventoryTask(name="RiskInventory-Cal-LowCash-v0", initial_cash=560.0, demand=LOW_DEMAND, target_wealth=900.0),
        InventoryTask(name="RiskInventory-Cal-HighPenalty-v0", demand=HIGH_DEMAND, shortage_penalty=6.0),
        InventoryTask(name="RiskInventory-Cal-VolatileHolding-v0", demand=VOLATILE_DEMAND, holding_cost=1.8),
        InventoryTask(name="RiskInventory-Cal-SpikeTarget-v0", periods=30, demand=SPIKE_DEMAND, target_wealth=1400.0),
        InventoryTask(name="RiskInventory-Cal-IntermittentLowCash-v0", initial_cash=600.0, demand=INTERMITTENT_DEMAND),
        InventoryTask(name="RiskInventory-Cal-HiddenBalanced-v0", periods=34, episode_regimes=balanced),
        InventoryTask(name="RiskInventory-Cal-HiddenTail-v0", periods=34, episode_regimes=tail, shortage_penalty=5.0),
    ]


def inventory_confirmation_tasks() -> list[InventoryTask]:
    hidden_balanced = ((LOW_DEMAND, 0.18), (STEADY_DEMAND, 0.32), (HIGH_DEMAND, 0.30), (VOLATILE_DEMAND, 0.20))
    hidden_tail = ((INTERMITTENT_DEMAND, 0.28), (STEADY_DEMAND, 0.20), (SPIKE_DEMAND, 0.32), (HIGH_DEMAND, 0.20))
    regimes = (
        ("Low", LOW_DEMAND, None),
        ("Steady", STEADY_DEMAND, None),
        ("High", HIGH_DEMAND, None),
        ("Volatile", VOLATILE_DEMAND, None),
        ("Spike", SPIKE_DEMAND, None),
        ("Intermittent", INTERMITTENT_DEMAND, None),
        ("HiddenBalanced", STEADY_DEMAND, hidden_balanced),
        ("HiddenTail", STEADY_DEMAND, hidden_tail),
    )
    profiles = (
        ("Short", dict(periods=18, initial_cash=900.0, target_wealth=1180.0, holding_cost=0.75, shortage_penalty=2.0)),
        ("Long", dict(periods=46, initial_cash=950.0, target_wealth=1600.0, holding_cost=0.75, shortage_penalty=2.0)),
        ("LowCash", dict(periods=28, initial_cash=540.0, target_wealth=940.0, holding_cost=0.90, shortage_penalty=3.0)),
        ("Costly", dict(periods=32, initial_cash=850.0, target_wealth=1300.0, holding_cost=2.0, shortage_penalty=6.0)),
    )
    return [
        InventoryTask(
            name=f"RiskInventory-Confirm-{regime_name}-{profile_name}-v0",
            demand=demand,
            episode_regimes=episode_regimes,
            **profile,
        )
        for regime_name, demand, episode_regimes in regimes
        for profile_name, profile in profiles
    ]
