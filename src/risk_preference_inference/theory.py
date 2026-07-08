"""Theory diagnostics for adaptive and static risk objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from risk_preference_inference.blackjack import DecisionState
from risk_preference_inference.envs import RiskTask
from risk_preference_inference.policies import BenchmarkPolicy


@dataclass(frozen=True)
class HorizonActionRow:
    task: str
    policy: str
    player_cards: tuple[int, ...]
    dealer_card: int
    bankroll: float
    horizon: int
    action: str
    hit_probability: float


@dataclass(frozen=True)
class ReversalRow:
    task: str
    policy: str
    player_cards: tuple[int, ...]
    dealer_card: int
    bankroll: float
    short_horizon_action: str
    long_horizon_action: str
    short_horizon: int
    long_horizon: int


def horizon_action_table(
    task: RiskTask,
    policy: BenchmarkPolicy,
    states: list[DecisionState],
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    hand_depth: int = 1,
) -> list[HorizonActionRow]:
    rows = []
    for state in states:
        for horizon in horizons:
            probs = policy.action_probabilities(
                state,
                task=task,
                rounds_remaining=horizon,
                hand_depth=hand_depth,
                peak_bankroll=max(task.initial_bankroll, state.current_bankroll),
            )
            action = "hit" if probs["hit"] >= probs["stand"] else "stand"
            rows.append(
                HorizonActionRow(
                    task.name,
                    policy.name,
                    state.player_cards,
                    state.dealer_card,
                    state.current_bankroll,
                    horizon,
                    action,
                    probs["hit"],
                )
            )
    return rows


def horizon_reversals(
    rows: list[HorizonActionRow],
    short_horizon: int,
    long_horizon: int,
) -> list[ReversalRow]:
    by_state: dict[tuple, dict[int, HorizonActionRow]] = {}
    for row in rows:
        key = (row.task, row.policy, row.player_cards, row.dealer_card, row.bankroll)
        by_state.setdefault(key, {})[row.horizon] = row
    reversals = []
    for key, horizon_rows in by_state.items():
        if short_horizon not in horizon_rows or long_horizon not in horizon_rows:
            continue
        short = horizon_rows[short_horizon]
        long = horizon_rows[long_horizon]
        if short.action != long.action:
            reversals.append(
                ReversalRow(
                    short.task,
                    short.policy,
                    short.player_cards,
                    short.dealer_card,
                    short.bankroll,
                    short.action,
                    long.action,
                    short_horizon,
                    long_horizon,
                )
            )
    return reversals


def rows_as_dicts(rows: list) -> list[dict]:
    return [asdict(row) for row in rows]

