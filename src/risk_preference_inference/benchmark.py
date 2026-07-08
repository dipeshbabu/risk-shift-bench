"""Simulation benchmark for risk-sensitive Blackjack policies."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from risk_preference_inference.blackjack import DecisionState, add_card, is_bust
from risk_preference_inference.envs import CardDistribution, RiskTask, benchmark_tasks
from risk_preference_inference.objectives import cvar_lower, mean, normalize, probability_at_or_above, probability_at_or_below
from risk_preference_inference.policies import BenchmarkPolicy
from risk_preference_inference.return_distributions import stand_payoff_distribution


@dataclass(frozen=True)
class EpisodeResult:
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
class BenchmarkSummary:
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


def sample_distribution(distribution: tuple[tuple[float, float], ...], rng: random.Random) -> float:
    threshold = rng.random()
    cumulative = 0.0
    for value, prob in distribution:
        cumulative += prob
        if threshold <= cumulative:
            return value
    return distribution[-1][0]


def sample_card(card_probs: CardDistribution, rng: random.Random) -> int:
    return int(sample_distribution(tuple((float(card), prob) for card, prob in card_probs), rng))


def sample_action(probs: dict[str, float], rng: random.Random) -> str:
    return "hit" if rng.random() < probs["hit"] else "stand"


def simulate_episode(
    task: RiskTask,
    policy: BenchmarkPolicy,
    seed: int,
    hand_depth: int = 4,
) -> EpisodeResult:
    rng = random.Random(seed)
    bankroll = task.initial_bankroll
    peak = bankroll
    min_bankroll = bankroll
    max_drawdown = 0.0
    target_hit = bankroll >= task.target_bankroll
    rounds_played = 0

    for round_idx in range(task.rounds):
        if bankroll <= task.ruin_bankroll:
            break
        player_cards = tuple(sorted((sample_card(task.card_probs, rng), sample_card(task.card_probs, rng))))
        dealer_card = sample_card(task.card_probs, rng)
        state = DecisionState(
            player_cards=player_cards,
            dealer_card=dealer_card,
            current_bankroll=bankroll,
            initial_bankroll=task.initial_bankroll,
            bet=task.bet,
            target_bankroll=task.target_bankroll,
        )

        while True:
            probs = policy.action_probabilities(
                state,
                task=task,
                rounds_remaining=task.rounds - round_idx,
                hand_depth=hand_depth,
                peak_bankroll=peak,
            )
            action = sample_action(probs, rng)
            if action == "stand":
                payoff_distribution = stand_payoff_distribution(state.player_total, state.dealer_card, state.bet, task.card_probs)
                payoff = sample_distribution(payoff_distribution, rng)
                break
            state = add_card(state, sample_card(task.card_probs, rng))
            if is_bust(state.player_cards):
                payoff = -task.bet
                break

        bankroll += payoff
        peak = max(peak, bankroll)
        min_bankroll = min(min_bankroll, bankroll)
        max_drawdown = max(max_drawdown, peak - bankroll)
        target_hit = target_hit or bankroll >= task.target_bankroll
        rounds_played += 1

    return EpisodeResult(
        task=task.name,
        policy=policy.name,
        seed=seed,
        final_bankroll=bankroll,
        min_bankroll=min_bankroll,
        max_drawdown=max_drawdown,
        ruined=bankroll <= task.ruin_bankroll,
        target_hit=target_hit,
        rounds_played=rounds_played,
    )


def summarize_results(results: list[EpisodeResult], task: RiskTask, policy: BenchmarkPolicy) -> BenchmarkSummary:
    finals = normalize(tuple((result.final_bankroll, 1.0) for result in results))
    mean_final = mean(finals)
    variance = sum(prob * (value - mean_final) ** 2 for value, prob in finals)
    return BenchmarkSummary(
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


def default_policies() -> list[BenchmarkPolicy]:
    from risk_preference_inference.policy_registry import core_policies

    return core_policies()


def run_benchmark(
    tasks: list[RiskTask] | None = None,
    policies: list[BenchmarkPolicy] | None = None,
    episodes: int = 100,
    seed: int = 0,
    hand_depth: int = 4,
) -> tuple[list[EpisodeResult], list[BenchmarkSummary]]:
    task_list = benchmark_tasks() if tasks is None else tasks
    policy_list = default_policies() if policies is None else policies
    all_results: list[EpisodeResult] = []
    summaries: list[BenchmarkSummary] = []
    for task_idx, task in enumerate(task_list):
        for _policy_idx, policy in enumerate(policy_list):
            policy_results = [
                simulate_episode(
                    task,
                    policy,
                    seed=seed + task_idx * 100_000 + episode_idx,
                    hand_depth=hand_depth,
                )
                for episode_idx in range(episodes)
            ]
            all_results.extend(policy_results)
            summaries.append(summarize_results(policy_results, task, policy))
    return all_results, summaries


def summaries_as_dicts(summaries: list[BenchmarkSummary]) -> list[dict]:
    return [asdict(summary) for summary in summaries]
