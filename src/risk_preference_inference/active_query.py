"""Active state selection for risk-preference experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations_with_replacement

from risk_preference_inference.blackjack import CARD_PROBS, DecisionState, hand_value
from risk_preference_inference.policy import action_probabilities
from risk_preference_inference.risk_models import RiskModel


@dataclass(frozen=True)
class QueryCandidate:
    state: DecisionState
    score: float
    model_probabilities: dict[str, float]


def candidate_states(
    bankrolls: tuple[float, ...] = (400.0, 460.0, 500.0, 540.0, 600.0),
    initial_bankroll: float = 500.0,
    bet: float = 20.0,
) -> list[DecisionState]:
    card_values = tuple(card for card, _ in CARD_PROBS)
    hands = []
    for hand in combinations_with_replacement(card_values, 2):
        total = hand_value(hand)
        if 8 <= total <= 20:
            hands.append(tuple(sorted(hand)))

    states: list[DecisionState] = []
    for bankroll in bankrolls:
        for hand in hands:
            for dealer_card in card_values:
                states.append(
                    DecisionState(
                        player_cards=hand,
                        dealer_card=dealer_card,
                        current_bankroll=bankroll,
                        initial_bankroll=initial_bankroll,
                        bet=bet,
                    )
                )
    return states


def disagreement_score(state: DecisionState, models: list[RiskModel], max_depth: int = 1) -> QueryCandidate:
    hit_probs = {
        model.name: action_probabilities(state, model, max_depth=max_depth)["hit"]
        for model in models
    }
    mean = sum(hit_probs.values()) / max(len(hit_probs), 1)
    variance = sum((prob - mean) ** 2 for prob in hit_probs.values()) / max(len(hit_probs), 1)
    entropy = 0.0
    if 0.0 < mean < 1.0:
        entropy = -(mean * math.log(mean) + (1.0 - mean) * math.log(1.0 - mean))
    return QueryCandidate(state=state, score=variance * entropy, model_probabilities=hit_probs)


def select_informative_states(
    models: list[RiskModel],
    limit: int = 20,
    max_depth: int = 1,
    states: list[DecisionState] | None = None,
) -> list[QueryCandidate]:
    search_states = candidate_states() if states is None else states
    candidates = [disagreement_score(state, models, max_depth=max_depth) for state in search_states]
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:limit]

