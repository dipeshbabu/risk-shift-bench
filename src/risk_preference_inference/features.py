"""Feature extraction for state-dependent risk models."""

from __future__ import annotations

from risk_preference_inference.blackjack import DecisionState, bust_probability, terminal_payoffs_for_stand


def state_features(state: DecisionState) -> dict[str, float]:
    stand_payoffs = terminal_payoffs_for_stand(state)
    p_loss = sum(prob for payoff, prob in stand_payoffs if payoff < 0.0)
    expected_stand = sum(payoff * prob for payoff, prob in stand_payoffs)
    recent = state.recent_outcomes[-5:]
    loss_streak = 0
    for outcome in reversed(recent):
        if outcome < 0:
            loss_streak += 1
        else:
            break

    return {
        "bankroll_ratio": state.current_bankroll / max(state.initial_bankroll, 1.0),
        "distance_from_target": (state.target - state.current_bankroll) / max(state.initial_bankroll, 1.0),
        "hand_total": state.player_total / 21.0,
        "dealer_card": state.dealer_card / 11.0,
        "usable_ace": 1.0 if state.usable_ace else 0.0,
        "bust_probability": bust_probability(state.player_cards),
        "stand_loss_probability": p_loss,
        "stand_expected_payoff": expected_stand / max(state.bet, 1.0),
        "recent_loss_streak": float(loss_streak),
        "recent_mean_outcome": (sum(recent) / len(recent) / max(state.bet, 1.0)) if recent else 0.0,
    }

