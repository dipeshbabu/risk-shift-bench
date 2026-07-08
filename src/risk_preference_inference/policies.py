"""Benchmark policies for adaptive risk-sensitive planning."""

from __future__ import annotations

from dataclasses import dataclass

from risk_preference_inference.blackjack import ACTIONS, DecisionState
from risk_preference_inference.envs import RiskTask
from risk_preference_inference.objectives import DistributionalObjective, MeanObjective, ObjectiveContext
from risk_preference_inference.return_distributions import action_bankroll_distribution


class BenchmarkPolicy:
    name = "policy"

    def action_probabilities(
        self,
        state: DecisionState,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class StaticObjectivePolicy(BenchmarkPolicy):
    objective: DistributionalObjective = MeanObjective()
    hand_depth: int = 4
    name: str = "static_objective"

    def context(self, state: DecisionState, task: RiskTask, rounds_remaining: int, peak_bankroll: float | None) -> ObjectiveContext:
        peak = state.current_bankroll if peak_bankroll is None else max(peak_bankroll, state.current_bankroll)
        return ObjectiveContext(
            bankroll=state.current_bankroll,
            initial_bankroll=task.initial_bankroll,
            ruin_bankroll=task.ruin_bankroll,
            target_bankroll=task.target_bankroll,
            peak_bankroll=peak,
            rounds_remaining=rounds_remaining,
        )

    def action_scores(
        self,
        state: DecisionState,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        context = self.context(state, task, rounds_remaining, peak_bankroll)
        depth = min(hand_depth, self.hand_depth)
        scores = {}
        for action in ACTIONS:
            distribution = action_bankroll_distribution(
                state,
                action,
                self,
                task,
                hand_depth=depth,
                rounds_remaining=rounds_remaining,
                peak_bankroll=peak_bankroll,
            )
            scores[action] = self.objective.score(distribution, context)
        return scores

    def action_probabilities(
        self,
        state: DecisionState,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        scores = self.action_scores(state, task, rounds_remaining, hand_depth, peak_bankroll)
        best = max(scores, key=scores.get)
        return {action: 1.0 if action == best else 0.0 for action in ACTIONS}


@dataclass(frozen=True)
class BasicStrategyPolicy(BenchmarkPolicy):
    """Small deterministic Blackjack heuristic for comparison."""

    name: str = "basic_strategy_heuristic"

    def action_probabilities(
        self,
        state: DecisionState,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        total = state.player_total
        dealer = state.dealer_card
        hit = total <= 11 or (total == 12 and dealer in (2, 3, 7, 8, 9, 10, 11)) or (13 <= total <= 16 and dealer >= 7)
        return {"stand": 0.0 if hit else 1.0, "hit": 1.0 if hit else 0.0}

