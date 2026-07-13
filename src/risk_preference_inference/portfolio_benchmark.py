"""Portfolio allocation benchmark and policies."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from risk_preference_inference.adaptive_risk import (
    AdaptiveUtilitySchedule,
    LearnedMixtureObjective,
    LearnedMixtureSchedule,
    StateAdaptiveUtilityObjective,
)
from risk_preference_inference.objectives import (
    DistributionalObjective,
    EntropicObjective,
    MeanObjective,
    OCEObjective,
    ObjectiveContext,
    TargetSeekingObjective,
    cvar_lower,
    mean,
    normalize,
)
from risk_preference_inference.portfolio_envs import CALM_MARKET, PortfolioTask, ReturnDistribution, portfolio_tasks


ALLOCATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class PortfolioState:
    capital: float
    initial_capital: float
    peak_capital: float
    periods_remaining: int


@dataclass(frozen=True)
class PortfolioEpisodeResult:
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
class PortfolioSummary:
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


class PortfolioPolicy:
    name = "portfolio_policy"

    def allocation(self, state: PortfolioState, task: PortfolioTask) -> float:
        raise NotImplementedError


def sample_distribution(distribution: tuple[tuple[float, float], ...], rng: random.Random) -> float:
    threshold = rng.random()
    cumulative = 0.0
    for value, prob in distribution:
        cumulative += prob
        if threshold <= cumulative:
            return value
    return distribution[-1][0]


def sample_episode_returns(task: PortfolioTask, rng: random.Random) -> ReturnDistribution:
    if task.episode_regimes is None:
        return task.risky_returns
    threshold = rng.random()
    cumulative = 0.0
    for returns, prob in task.episode_regimes:
        cumulative += prob
        if threshold <= cumulative:
            return returns
    return task.episode_regimes[-1][0]


def expected_return(returns: ReturnDistribution) -> float:
    return sum(value * prob for value, prob in returns)


def downside_mass(returns: ReturnDistribution) -> float:
    return sum(prob for value, prob in returns if value < -0.03)


def hidden_return_stats(task: PortfolioTask) -> tuple[float, float, float]:
    if task.episode_regimes is None:
        return 0.0, 0.0, 0.0
    means = [expected_return(returns) for returns, _prob in task.episode_regimes]
    return min(means), max(means), max(means) - min(means)


def visible_return_distribution(task: PortfolioTask) -> ReturnDistribution:
    if task.episode_regimes is None:
        return task.risky_returns
    weights: dict[float, float] = {}
    for returns, regime_prob in task.episode_regimes:
        for value, prob in returns:
            weights[value] = weights.get(value, 0.0) + regime_prob * prob
    return normalize(tuple(weights.items()))


def next_capital_distribution(state: PortfolioState, task: PortfolioTask, allocation: float) -> tuple[tuple[float, float], ...]:
    distribution = []
    for risky_return, prob in visible_return_distribution(task):
        blended_return = task.cash_return * (1.0 - allocation) + risky_return * allocation
        distribution.append((state.capital * (1.0 + blended_return), prob))
    return normalize(tuple(distribution))


def objective_context(state: PortfolioState, task: PortfolioTask) -> ObjectiveContext:
    drawdown = max(0.0, state.peak_capital - state.capital)
    return ObjectiveContext(
        bankroll=state.capital,
        initial_bankroll=task.initial_capital,
        ruin_bankroll=task.ruin_capital,
        target_bankroll=task.target_capital,
        peak_bankroll=state.peak_capital,
        rounds_remaining=state.periods_remaining,
        bet=max(task.initial_capital * 0.05, 1.0),
        drawdown_limit=task.drawdown_limit,
        card_mean_shift=expected_return(visible_return_distribution(task)) - expected_return(CALM_MARKET),
        high_card_mass=1.0 - downside_mass(visible_return_distribution(task)),
    )


@dataclass(frozen=True)
class PortfolioObjectivePolicy(PortfolioPolicy):
    objective: DistributionalObjective = MeanObjective()
    name: str = "portfolio_objective"

    def allocation(self, state: PortfolioState, task: PortfolioTask) -> float:
        context = objective_context(state, task)
        scores = {
            allocation: self.objective.score(next_capital_distribution(state, task, allocation), context)
            for allocation in ALLOCATIONS
        }
        return max(scores, key=lambda allocation: (scores[allocation], -allocation))


@dataclass(frozen=True)
class PortfolioSignedRegimePolicy(PortfolioPolicy):
    name: str = "signed_regime_learned_ensemble"

    def allocation(self, state: PortfolioState, task: PortfolioTask) -> float:
        returns = visible_return_distribution(task)
        mean_return = expected_return(returns)
        tail = downside_mass(returns)
        capital_ratio = state.capital / max(task.initial_capital, 1.0)
        target_gap = (task.target_capital - state.capital) / max(task.initial_capital, 1.0)
        drawdown_ratio = (state.peak_capital - state.capital) / max(state.peak_capital, 1.0)
        hidden_span = hidden_return_stats(task)[2]

        if state.capital <= task.ruin_capital * 1.10 or capital_ratio < 0.82:
            return 0.0
        if drawdown_ratio >= task.drawdown_limit * 0.8 or tail > 0.35:
            return 0.25
        if hidden_span > 0.035 and task.periods >= 40:
            return 0.25
        if target_gap > 0.20 and state.periods_remaining <= 10 and mean_return > 0.0:
            return 0.75
        if mean_return > 0.015:
            return 0.75
        if mean_return < -0.015:
            return 0.25
        return 0.5


@dataclass(frozen=True)
class PortfolioFixedAllocationPolicy(PortfolioPolicy):
    allocation_value: float
    name: str

    def allocation(self, state: PortfolioState, task: PortfolioTask) -> float:
        return self.allocation_value


def portfolio_learned_mixture_policy(name: str = "learned_mixture_searched") -> PortfolioPolicy:
    schedule = LearnedMixtureSchedule(
        risk_intercept=0.0,
        bankroll_weight=0.35,
        drawdown_weight=0.75,
        deck_shift_weight=0.75,
        target_intercept=0.0,
        target_gap_weight=0.85,
        terminal_weight=0.35,
        terminal_window=8,
    )
    objective = LearnedMixtureObjective(
        schedule=schedule,
        cvar_alpha=0.15,
        entropic_eta=0.01,
        oce_penalty=3.0,
        entropic_weight=0.35,
        cvar_weight=0.05,
        oce_weight=0.15,
        deck_entropic_weight=1.25,
        ruin_penalty=250.0,
        drawdown_penalty=0.15,
        target_bonus=350.0,
        target_excess_weight=0.15,
    )
    return PortfolioObjectivePolicy(objective=objective, name=name)


def portfolio_adaptive_utility_policy(name: str = "adaptive_utility_default") -> PortfolioPolicy:
    schedule = AdaptiveUtilitySchedule(
        low_bankroll_ratio=0.55,
        safe_bankroll_ratio=1.15,
        drawdown_trigger=0.12,
        target_window=0.35,
        terminal_window=8,
    )
    objective = StateAdaptiveUtilityObjective(
        schedule=schedule,
        cvar_alpha=0.2,
        entropic_eta=0.01,
        risk_weight=0.35,
        ruin_penalty=400.0,
        drawdown_penalty=0.35,
        target_bonus=180.0,
        target_excess_weight=0.15,
    )
    return PortfolioObjectivePolicy(objective=objective, name=name)


def portfolio_policy_grid() -> list[PortfolioPolicy]:
    return [
        PortfolioFixedAllocationPolicy(0.0, name="cash_only"),
        PortfolioFixedAllocationPolicy(0.5, name="balanced_50"),
        PortfolioObjectivePolicy(MeanObjective(), name="expected_value"),
        PortfolioObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="fixed_entropic_0.025"),
        PortfolioObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="fixed_oce_3"),
        PortfolioObjectivePolicy(TargetSeekingObjective(MeanObjective(), target_bonus=150.0), name="target_seeking_mean"),
        portfolio_adaptive_utility_policy(),
        portfolio_learned_mixture_policy(),
        PortfolioSignedRegimePolicy(),
    ]


def portfolio_policy_lookup() -> dict[str, PortfolioPolicy]:
    return {policy.name: policy for policy in portfolio_policy_grid()}


def simulate_portfolio_episode(
    task: PortfolioTask,
    policy: PortfolioPolicy,
    seed: int,
) -> PortfolioEpisodeResult:
    rng = random.Random(seed)
    realized_returns = sample_episode_returns(task, rng)
    capital = task.initial_capital
    peak = capital
    min_capital = capital
    target_hit = capital >= task.target_capital
    max_drawdown = 0.0
    periods_played = 0
    for period in range(task.periods):
        if capital <= task.ruin_capital:
            break
        state = PortfolioState(
            capital=capital,
            initial_capital=task.initial_capital,
            peak_capital=peak,
            periods_remaining=task.periods - period,
        )
        allocation = min(1.0, max(0.0, policy.allocation(state, task)))
        risky_return = sample_distribution(realized_returns, rng)
        blended_return = task.cash_return * (1.0 - allocation) + risky_return * allocation
        capital *= 1.0 + blended_return
        peak = max(peak, capital)
        min_capital = min(min_capital, capital)
        max_drawdown = max(max_drawdown, peak - capital)
        target_hit = target_hit or capital >= task.target_capital
        periods_played += 1
    return PortfolioEpisodeResult(
        task=task.name,
        policy=policy.name,
        seed=seed,
        final_bankroll=capital,
        min_bankroll=min_capital,
        max_drawdown=max_drawdown,
        ruined=capital <= task.ruin_capital,
        target_hit=target_hit,
        rounds_played=periods_played,
    )


def summarize_portfolio_results(
    results: list[PortfolioEpisodeResult],
    task: PortfolioTask,
    policy: PortfolioPolicy,
) -> PortfolioSummary:
    finals = normalize(tuple((result.final_bankroll, 1.0) for result in results))
    mean_final = mean(finals)
    variance = sum(prob * (value - mean_final) ** 2 for value, prob in finals)
    return PortfolioSummary(
        task=task.name,
        policy=policy.name,
        episodes=len(results),
        mean_final_bankroll=mean_final,
        std_final_bankroll=variance**0.5,
        cvar_5_final_bankroll=cvar_lower(finals, 0.05),
        ruin_probability=sum(1.0 for result in results if result.ruined) / max(len(results), 1),
        target_probability=sum(1.0 for result in results if result.target_hit) / max(len(results), 1),
        mean_max_drawdown=sum(result.max_drawdown for result in results) / max(len(results), 1),
        mean_rounds_played=sum(result.rounds_played for result in results) / max(len(results), 1),
    )


def run_portfolio_benchmark(
    tasks: list[PortfolioTask] | None = None,
    policies: list[PortfolioPolicy] | None = None,
    episodes: int = 100,
    seed: int = 0,
) -> tuple[list[PortfolioEpisodeResult], list[PortfolioSummary]]:
    task_list = portfolio_tasks() if tasks is None else tasks
    policy_list = portfolio_policy_grid() if policies is None else policies
    all_results: list[PortfolioEpisodeResult] = []
    summaries: list[PortfolioSummary] = []
    for task_idx, task in enumerate(task_list):
        for policy in policy_list:
            policy_results = [
                simulate_portfolio_episode(
                    task,
                    policy,
                    seed=seed + task_idx * 100_000 + episode_idx,
                )
                for episode_idx in range(episodes)
            ]
            all_results.extend(policy_results)
            summaries.append(summarize_portfolio_results(policy_results, task, policy))
    return all_results, summaries


def summaries_as_dicts(summaries: list[PortfolioSummary]) -> list[dict]:
    return [asdict(summary) for summary in summaries]
