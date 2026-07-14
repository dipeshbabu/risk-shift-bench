"""Small exact Blackjack model for decision-level risk inference.

The model intentionally uses an infinite-deck card distribution. That keeps
likelihood evaluation fast and deterministic while preserving the relevant
sequential decision structure for hit/stand choices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterable

ACTIONS = ("stand", "hit")
CARD_PROBS = (
    (2, 1.0 / 13.0),
    (3, 1.0 / 13.0),
    (4, 1.0 / 13.0),
    (5, 1.0 / 13.0),
    (6, 1.0 / 13.0),
    (7, 1.0 / 13.0),
    (8, 1.0 / 13.0),
    (9, 1.0 / 13.0),
    (10, 4.0 / 13.0),
    (11, 1.0 / 13.0),
)


@dataclass(frozen=True)
class DecisionState:
    """A single human-observable Blackjack decision state."""

    player_cards: tuple[int, ...]
    dealer_card: int
    current_bankroll: float = 500.0
    initial_bankroll: float = 500.0
    bet: float = 20.0
    recent_outcomes: tuple[float, ...] = ()
    target_bankroll: float | None = None

    @property
    def player_total(self) -> int:
        return hand_value(self.player_cards)

    @property
    def usable_ace(self) -> bool:
        return has_usable_ace(self.player_cards)

    @property
    def target(self) -> float:
        return self.initial_bankroll if self.target_bankroll is None else self.target_bankroll


def hand_value(cards: Iterable[int]) -> int:
    total = sum(cards)
    aces = list(cards).count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def has_usable_ace(cards: Iterable[int]) -> bool:
    cards_tuple = tuple(cards)
    return 11 in cards_tuple and sum(cards_tuple) <= 21


def is_bust(cards: Iterable[int]) -> bool:
    return hand_value(cards) > 21


def draw_distribution() -> tuple[tuple[int, float], ...]:
    return CARD_PROBS


def add_card(state: DecisionState, card: int) -> DecisionState:
    return replace(state, player_cards=tuple(sorted(state.player_cards + (card,))))


def bust_probability(player_cards: tuple[int, ...]) -> float:
    total = 0.0
    for card, prob in CARD_PROBS:
        if is_bust(player_cards + (card,)):
            total += prob
    return total


@lru_cache(maxsize=None)
def dealer_total_distribution(dealer_cards: tuple[int, ...]) -> tuple[tuple[int | str, float], ...]:
    """Return final dealer totals after drawing to 17.

    The returned outcome is either an integer total or the string "bust".
    """

    total = hand_value(dealer_cards)
    if total > 21:
        return (("bust", 1.0),)
    if total >= 17:
        return ((total, 1.0),)

    merged: dict[int | str, float] = {}
    for card, prob in CARD_PROBS:
        for outcome, next_prob in dealer_total_distribution(tuple(sorted(dealer_cards + (card,)))):
            merged[outcome] = merged.get(outcome, 0.0) + prob * next_prob
    return tuple(sorted(merged.items(), key=lambda item: str(item[0])))


@lru_cache(maxsize=None)
def stand_payoff_distribution(player_total: int, dealer_card: int, bet: float) -> tuple[tuple[float, float], ...]:
    """Distribution over terminal dollar payoffs when the player stands."""

    merged: dict[float, float] = {}
    for hidden_card, hidden_prob in CARD_PROBS:
        dealer_cards = tuple(sorted((dealer_card, hidden_card)))
        for dealer_outcome, dealer_prob in dealer_total_distribution(dealer_cards):
            if dealer_outcome == "bust" or player_total > int(dealer_outcome):
                payoff = bet
            elif player_total < int(dealer_outcome):
                payoff = -bet
            else:
                payoff = 0.0
            merged[payoff] = merged.get(payoff, 0.0) + hidden_prob * dealer_prob
    return tuple(sorted(merged.items()))


def terminal_payoffs_for_stand(state: DecisionState) -> tuple[tuple[float, float], ...]:
    return stand_payoff_distribution(state.player_total, state.dealer_card, float(state.bet))

