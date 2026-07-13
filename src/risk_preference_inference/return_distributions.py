"""Return-distribution helpers for benchmark policies."""

from __future__ import annotations

from functools import lru_cache

from risk_preference_inference.blackjack import ACTIONS, DecisionState, add_card, hand_value, is_bust
from risk_preference_inference.envs import CardDistribution, RiskTask, STANDARD_DECK
from risk_preference_inference.objectives import Distribution, normalize


@lru_cache(maxsize=None)
def dealer_distribution(
    dealer_cards: tuple[int, ...],
    card_probs: CardDistribution = STANDARD_DECK,
) -> tuple[tuple[int | str, float], ...]:
    total = hand_value(dealer_cards)
    if total > 21:
        return (("bust", 1.0),)
    if total >= 17:
        return ((total, 1.0),)

    merged: dict[int | str, float] = {}
    for card, prob in card_probs:
        for outcome, next_prob in dealer_distribution(tuple(sorted(dealer_cards + (card,))), card_probs):
            merged[outcome] = merged.get(outcome, 0.0) + prob * next_prob
    return tuple(sorted(merged.items(), key=lambda item: str(item[0])))


@lru_cache(maxsize=None)
def stand_payoff_distribution(
    player_total: int,
    dealer_card: int,
    bet: float,
    card_probs: CardDistribution = STANDARD_DECK,
) -> Distribution:
    merged: dict[float, float] = {}
    for hidden_card, hidden_prob in card_probs:
        dealer_cards = tuple(sorted((dealer_card, hidden_card)))
        for dealer_outcome, dealer_prob in dealer_distribution(dealer_cards, card_probs):
            if dealer_outcome == "bust" or player_total > int(dealer_outcome):
                payoff = bet
            elif player_total < int(dealer_outcome):
                payoff = -bet
            else:
                payoff = 0.0
            merged[payoff] = merged.get(payoff, 0.0) + hidden_prob * dealer_prob
    return normalize(tuple(merged.items()))


@lru_cache(maxsize=250_000)
def action_payoff_distribution(
    state: DecisionState,
    action: str,
    policy: "BenchmarkPolicy",
    task: RiskTask,
    hand_depth: int = 4,
    rounds_remaining: int = 1,
    peak_bankroll: float | None = None,
) -> Distribution:
    from risk_preference_inference.policies import BenchmarkPolicy

    if not isinstance(policy, BenchmarkPolicy):
        raise TypeError("policy must be a BenchmarkPolicy")
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    if action == "stand" or hand_depth <= 0:
        return stand_payoff_distribution(state.player_total, state.dealer_card, state.bet, task.card_probs)

    merged: dict[float, float] = {}
    for card, card_prob in task.card_probs:
        next_state = add_card(state, card)
        if is_bust(next_state.player_cards):
            merged[-state.bet] = merged.get(-state.bet, 0.0) + card_prob
            continue
        action_probs = policy.action_probabilities(
            next_state,
            task=task,
            rounds_remaining=rounds_remaining,
            hand_depth=hand_depth - 1,
            peak_bankroll=peak_bankroll,
        )
        for next_action, action_prob in action_probs.items():
            for payoff, payoff_prob in action_payoff_distribution(
                next_state,
                next_action,
                policy,
                task,
                hand_depth=hand_depth - 1,
                rounds_remaining=rounds_remaining,
                peak_bankroll=peak_bankroll,
            ):
                merged[payoff] = merged.get(payoff, 0.0) + card_prob * action_prob * payoff_prob
    return normalize(tuple(merged.items()))


@lru_cache(maxsize=250_000)
def action_bankroll_distribution(
    state: DecisionState,
    action: str,
    policy: "BenchmarkPolicy",
    task: RiskTask,
    hand_depth: int = 4,
    rounds_remaining: int = 1,
    peak_bankroll: float | None = None,
) -> Distribution:
    payoffs = action_payoff_distribution(
        state,
        action,
        policy,
        task,
        hand_depth=hand_depth,
        rounds_remaining=rounds_remaining,
        peak_bankroll=peak_bankroll,
    )
    return normalize(tuple((state.current_bankroll + payoff, prob) for payoff, prob in payoffs))
