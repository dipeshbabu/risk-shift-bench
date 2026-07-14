"""Action scoring and stochastic choice models."""

from __future__ import annotations

import math
from functools import lru_cache

from risk_shift_bench.blackjack import (
    ACTIONS,
    CARD_PROBS,
    DecisionState,
    add_card,
    is_bust,
    terminal_payoffs_for_stand,
)
from risk_shift_bench.risk_models import RiskModel


def action_value(state: DecisionState, action: str, model: RiskModel, max_depth: int = 6) -> float:
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    return _action_value_cached(state, action, model, max_depth)


def action_values(state: DecisionState, model: RiskModel, max_depth: int = 6) -> dict[str, float]:
    return {action: action_value(state, action, model, max_depth=max_depth) for action in ACTIONS}


def action_probabilities(state: DecisionState, model: RiskModel, max_depth: int = 6) -> dict[str, float]:
    values = action_values(state, model, max_depth=max_depth)
    temperature = max(float(getattr(model, "temperature", 1.0)), 1e-6)
    hit_logit = (values["hit"] - values["stand"]) / temperature
    p_hit = 1.0 / (1.0 + math.exp(-max(min(hit_logit, 60.0), -60.0)))
    return {"stand": 1.0 - p_hit, "hit": p_hit}


def choose_best_action(state: DecisionState, model: RiskModel, max_depth: int = 6) -> str:
    values = action_values(state, model, max_depth=max_depth)
    return max(values, key=values.get)


@lru_cache(maxsize=200_000)
def _action_value_cached(state: DecisionState, action: str, model: RiskModel, max_depth: int) -> float:
    if action == "stand" or max_depth <= 0:
        values = [
            (model.terminal_value(state, payoff), prob)
            for payoff, prob in terminal_payoffs_for_stand(state)
        ]
        return model.aggregate(values)

    values = []
    for card, prob in CARD_PROBS:
        next_state = add_card(state, card)
        if is_bust(next_state.player_cards):
            values.append((model.terminal_value(state, -state.bet), prob))
        else:
            next_values = [
                _action_value_cached(next_state, next_action, model, max_depth - 1)
                for next_action in ACTIONS
            ]
            values.append((max(next_values), prob))
    return model.aggregate(values)


def clear_policy_cache() -> None:
    _action_value_cached.cache_clear()

