"""Synthetic data generation for pipeline validation."""

from __future__ import annotations

import random
from typing import Callable

from risk_preference_inference.blackjack import CARD_PROBS, DecisionState, hand_value
from risk_preference_inference.dataset import DecisionRecord
from risk_preference_inference.policy import action_probabilities
from risk_preference_inference.risk_models import ProspectUtilityModel, RiskModel, StateDependentProspectModel


def random_valid_player_hand(rng: random.Random) -> tuple[int, ...]:
    cards = [rng.choice(CARD_PROBS)[0], rng.choice(CARD_PROBS)[0]]
    while hand_value(cards) < 8 or hand_value(cards) > 20:
        cards = [rng.choice(CARD_PROBS)[0], rng.choice(CARD_PROBS)[0]]
    return tuple(sorted(cards))


def sample_action(probs: dict[str, float], rng: random.Random) -> str:
    return "hit" if rng.random() < probs["hit"] else "stand"


def generate_synthetic_records(
    subjects: int = 8,
    decisions_per_subject: int = 120,
    seed: int = 7,
    model_factory: Callable[[int, random.Random], RiskModel] | None = None,
) -> list[DecisionRecord]:
    rng = random.Random(seed)
    records: list[DecisionRecord] = []

    for subject_idx in range(subjects):
        if model_factory is None:
            model: RiskModel = ProspectUtilityModel(
                alpha=rng.choice((0.55, 0.75, 1.0, 1.25)),
                loss_aversion=rng.choice((1.0, 1.5, 2.25, 3.0)),
                temperature=rng.choice((0.5, 0.9, 1.5)),
            )
        else:
            model = model_factory(subject_idx, rng)

        bankroll = 500.0
        recent_outcomes: list[float] = []
        for step in range(decisions_per_subject):
            player_cards = random_valid_player_hand(rng)
            dealer_card = rng.choice(CARD_PROBS)[0]
            state = DecisionState(
                player_cards=player_cards,
                dealer_card=dealer_card,
                current_bankroll=bankroll,
                initial_bankroll=500.0,
                bet=20.0,
                recent_outcomes=tuple(recent_outcomes[-5:]),
            )
            probs = action_probabilities(state, model, max_depth=1)
            action = sample_action(probs, rng)
            simulated_outcome = rng.choice((-20.0, 0.0, 20.0))
            recent_outcomes.append(simulated_outcome)
            bankroll = max(20.0, bankroll + simulated_outcome)

            records.append(
                DecisionRecord(
                    subject_id=f"subject_{subject_idx:03d}",
                    episode_id=f"synthetic_{subject_idx:03d}",
                    step_id=step,
                    player_cards=player_cards,
                    dealer_card=dealer_card,
                    current_bankroll=state.current_bankroll,
                    initial_bankroll=state.initial_bankroll,
                    bet=state.bet,
                    recent_outcomes=state.recent_outcomes,
                    action_taken=action,
                )
            )
    return records


def generate_state_dependent_synthetic_records(
    subjects: int = 8,
    decisions_per_subject: int = 120,
    seed: int = 11,
) -> list[DecisionRecord]:
    def factory(subject_idx: int, rng: random.Random) -> StateDependentProspectModel:
        return StateDependentProspectModel(
            alpha=rng.choice((0.65, 0.85, 1.05)),
            loss_aversion=rng.choice((1.25, 1.75, 2.5)),
            temperature=rng.choice((0.5, 0.9, 1.4)),
            alpha_loss_streak_weight=rng.choice((-0.35, 0.0, 0.35)),
            lambda_near_ruin_weight=rng.choice((0.0, 1.0, 2.0)),
            lambda_loss_streak_weight=rng.choice((-0.25, 0.0, 0.25)),
        )

    return generate_synthetic_records(subjects, decisions_per_subject, seed, model_factory=factory)
