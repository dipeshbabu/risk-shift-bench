"""Configurable risk-sensitive Blackjack benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass

from risk_preference_inference.blackjack import CARD_PROBS

CardDistribution = tuple[tuple[int, float], ...]


def normalize_card_probs(card_probs: dict[int, float] | CardDistribution) -> CardDistribution:
    items = tuple(card_probs.items()) if isinstance(card_probs, dict) else tuple(card_probs)
    total = sum(prob for _, prob in items)
    if total <= 0.0:
        raise ValueError("card probability mass must be positive")
    return tuple(sorted((int(card), float(prob) / total) for card, prob in items))


STANDARD_DECK = normalize_card_probs(CARD_PROBS)
LOW_CARD_SHIFT = normalize_card_probs({2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1, 9: 1, 10: 2, 11: 1})
HIGH_CARD_SHIFT = normalize_card_probs({2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 8, 11: 2})


@dataclass(frozen=True)
class RiskTask:
    name: str
    rounds: int = 25
    initial_bankroll: float = 500.0
    bet: float = 20.0
    ruin_bankroll: float = 0.0
    target_bankroll: float = 650.0
    drawdown_limit: float = 0.25
    card_probs: CardDistribution = STANDARD_DECK


def benchmark_tasks() -> list[RiskTask]:
    return [
        RiskTask(name="RiskBlackjack-Mean-v0"),
        RiskTask(name="RiskBlackjack-RuinConstraint-v0", initial_bankroll=240.0, target_bankroll=360.0),
        RiskTask(name="RiskBlackjack-Target-v0", rounds=30, target_bankroll=640.0),
        RiskTask(name="RiskBlackjack-Drawdown-v0", drawdown_limit=0.12),
        RiskTask(name="RiskBlackjack-LowCardShift-v0", card_probs=LOW_CARD_SHIFT),
        RiskTask(name="RiskBlackjack-HighCardShift-v0", card_probs=HIGH_CARD_SHIFT),
    ]
