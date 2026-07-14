"""Exact small-horizon final-bankroll distributions."""

from __future__ import annotations

from risk_shift_bench.blackjack import DecisionState
from risk_shift_bench.envs import RiskTask
from risk_shift_bench.objectives import Distribution, normalize
from risk_shift_bench.policies import BenchmarkPolicy
from risk_shift_bench.return_distributions import action_payoff_distribution


def starting_state_distribution(task: RiskTask, bankroll: float) -> tuple[tuple[DecisionState, float], ...]:
    states: dict[DecisionState, float] = {}
    for card1, p1 in task.card_probs:
        for card2, p2 in task.card_probs:
            for dealer_card, pd in task.card_probs:
                state = DecisionState(
                    player_cards=tuple(sorted((card1, card2))),
                    dealer_card=dealer_card,
                    current_bankroll=bankroll,
                    initial_bankroll=task.initial_bankroll,
                    bet=task.bet,
                    target_bankroll=task.target_bankroll,
                )
                states[state] = states.get(state, 0.0) + p1 * p2 * pd
    return tuple(states.items())


def final_bankroll_distribution(
    task: RiskTask,
    policy: BenchmarkPolicy,
    rounds: int | None = None,
    hand_depth: int = 2,
    grid: float = 1.0,
) -> Distribution:
    horizon = task.rounds if rounds is None else rounds
    current: Distribution = ((task.initial_bankroll, 1.0),)
    for round_idx in range(horizon):
        merged: dict[float, float] = {}
        for bankroll, bankroll_prob in current:
            if bankroll <= task.ruin_bankroll:
                merged[bankroll] = merged.get(bankroll, 0.0) + bankroll_prob
                continue
            for state, state_prob in starting_state_distribution(task, bankroll):
                action_probs = policy.action_probabilities(
                    state,
                    task=task,
                    rounds_remaining=horizon - round_idx,
                    hand_depth=hand_depth,
                    peak_bankroll=max(task.initial_bankroll, bankroll),
                )
                for action, action_prob in action_probs.items():
                    for payoff, payoff_prob in action_payoff_distribution(
                        state,
                        action,
                        policy,
                        task,
                        hand_depth=hand_depth,
                        rounds_remaining=horizon - round_idx,
                        peak_bankroll=max(task.initial_bankroll, bankroll),
                    ):
                        next_bankroll = bankroll + payoff
                        if grid > 0:
                            next_bankroll = round(next_bankroll / grid) * grid
                        merged[next_bankroll] = merged.get(next_bankroll, 0.0) + bankroll_prob * state_prob * action_prob * payoff_prob
        current = normalize(tuple(merged.items()))
    return current
