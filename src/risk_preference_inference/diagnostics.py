"""Policy diagnostics for adaptive risk experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from risk_preference_inference.adaptive_risk import AdaptiveCVaRObjective
from risk_preference_inference.blackjack import DecisionState
from risk_preference_inference.envs import RiskTask
from risk_preference_inference.policies import BenchmarkPolicy, StaticObjectivePolicy


@dataclass(frozen=True)
class ActionMapRow:
    policy: str
    bankroll: float
    player_total: int
    dealer_card: int
    action: str
    hit_probability: float


@dataclass(frozen=True)
class AdaptiveAlphaRow:
    bankroll: float
    alpha: float


def action_map(
    policy: BenchmarkPolicy,
    task: RiskTask,
    bankrolls: tuple[float, ...] = (300.0, 500.0, 700.0),
    hand_depth: int = 1,
) -> list[ActionMapRow]:
    rows: list[ActionMapRow] = []
    for bankroll in bankrolls:
        for player_total in range(8, 21):
            for dealer_card in range(2, 12):
                state = DecisionState(
                    player_cards=(player_total,),
                    dealer_card=dealer_card,
                    current_bankroll=bankroll,
                    initial_bankroll=task.initial_bankroll,
                    bet=task.bet,
                    target_bankroll=task.target_bankroll,
                )
                probs = policy.action_probabilities(state, task, task.rounds, hand_depth=hand_depth, peak_bankroll=max(task.initial_bankroll, bankroll))
                action = "hit" if probs["hit"] >= probs["stand"] else "stand"
                rows.append(ActionMapRow(policy.name, bankroll, player_total, dealer_card, action, probs["hit"]))
    return rows


def adaptive_alpha_curve(
    policy: BenchmarkPolicy,
    task: RiskTask,
    bankrolls: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0),
) -> list[AdaptiveAlphaRow]:
    if not isinstance(policy, StaticObjectivePolicy) or not isinstance(policy.objective, AdaptiveCVaRObjective):
        return []
    rows = []
    for bankroll in bankrolls:
        context = policy.context(
            DecisionState((10, 6), 10, bankroll, task.initial_bankroll, task.bet, target_bankroll=task.target_bankroll),
            task,
            task.rounds,
            max(task.initial_bankroll, bankroll),
        )
        rows.append(AdaptiveAlphaRow(bankroll=bankroll, alpha=policy.objective.schedule.alpha(context)))
    return rows


def rows_as_dicts(rows: list) -> list[dict]:
    return [asdict(row) for row in rows]

