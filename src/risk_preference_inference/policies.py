"""Benchmark policies for adaptive risk-sensitive planning."""

from __future__ import annotations

from dataclasses import dataclass, field

from risk_preference_inference.blackjack import ACTIONS, DecisionState
from risk_preference_inference.envs import STANDARD_DECK, RiskTask
from risk_preference_inference.objectives import (
    DistributionalObjective,
    EntropicObjective,
    MeanObjective,
    OCEObjective,
    ObjectiveContext,
    TargetSeekingObjective,
)
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
        standard_card_mean = sum(card * prob for card, prob in STANDARD_DECK)
        task_card_mean = sum(card * prob for card, prob in task.card_probs)
        high_card_mass = sum(prob for card, prob in task.card_probs if card >= 10)
        return ObjectiveContext(
            bankroll=state.current_bankroll,
            initial_bankroll=task.initial_bankroll,
            ruin_bankroll=task.ruin_bankroll,
            target_bankroll=task.target_bankroll,
            peak_bankroll=peak,
            rounds_remaining=rounds_remaining,
            bet=task.bet,
            drawdown_limit=task.drawdown_limit,
            card_mean_shift=task_card_mean - standard_card_mean,
            high_card_mass=high_card_mass,
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


@dataclass(frozen=True)
class RegimeAdaptivePolicy(BenchmarkPolicy):
    """Switch between objective families using observable task regime features."""

    name: str = "regime_adaptive_ensemble"
    enable_deck_shift: bool = True
    enable_ruin: bool = True
    enable_drawdown: bool = True
    enable_target: bool = True
    require_target_regime: bool = True

    def _card_mean(self, task: RiskTask) -> float:
        return sum(card * prob for card, prob in task.card_probs)

    def _standard_card_mean(self) -> float:
        return sum(card * prob for card, prob in STANDARD_DECK)

    def _delegate(
        self,
        state: DecisionState,
        task: RiskTask,
        rounds_remaining: int,
        peak_bankroll: float | None,
    ) -> BenchmarkPolicy:
        target_gap = max(0.0, task.target_bankroll - state.current_bankroll)
        shifted_deck = abs(self._card_mean(task) - self._standard_card_mean()) > 0.35
        target_regime = task.rounds > 25 or not self.require_target_regime

        if self.enable_deck_shift and shifted_deck:
            return StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="regime_entropic_shift")
        if self.enable_ruin and task.initial_bankroll <= 15.0 * task.bet:
            return StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="regime_oce_ruin")
        if self.enable_drawdown and task.drawdown_limit <= 0.15:
            return StaticObjectivePolicy(EntropicObjective(risk_aversion=0.01), name="regime_entropic_drawdown")
        if self.enable_target and target_regime and (target_gap <= 6.0 * task.bet or rounds_remaining <= 10):
            objective = TargetSeekingObjective(MeanObjective(), target_bonus=300.0)
            return StaticObjectivePolicy(objective, name="regime_target_mean")
        return BasicStrategyPolicy()

    def action_probabilities(
        self,
        state: DecisionState,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        delegate = self._delegate(state, task, rounds_remaining, peak_bankroll)
        return delegate.action_probabilities(
            state,
            task=task,
            rounds_remaining=rounds_remaining,
            hand_depth=hand_depth,
            peak_bankroll=peak_bankroll,
        )


@dataclass(frozen=True)
class SignedRegimeAdaptivePolicy(BenchmarkPolicy):
    """Regime controller with signed deck gates and pluggable delegates."""

    name: str = "signed_regime_learned_ensemble"
    positive_shift_threshold: float = 0.35
    negative_shift_threshold: float = -0.35
    require_target_regime: bool = True
    mean_delegate: BenchmarkPolicy = field(default_factory=BasicStrategyPolicy)
    ruin_delegate: BenchmarkPolicy = field(
        default_factory=lambda: StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="signed_ruin_oce")
    )
    target_delegate: BenchmarkPolicy = field(
        default_factory=lambda: StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="signed_target_entropic")
    )
    drawdown_delegate: BenchmarkPolicy = field(
        default_factory=lambda: StaticObjectivePolicy(EntropicObjective(risk_aversion=0.01), name="signed_drawdown_entropic")
    )
    high_shift_delegate: BenchmarkPolicy = field(
        default_factory=lambda: StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="signed_high_entropic")
    )
    low_shift_delegate: BenchmarkPolicy = field(
        default_factory=lambda: StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="signed_low_oce")
    )

    def _card_mean(self, task: RiskTask) -> float:
        return sum(card * prob for card, prob in task.card_probs)

    def _standard_card_mean(self) -> float:
        return sum(card * prob for card, prob in STANDARD_DECK)

    def _delegate(
        self,
        state: DecisionState,
        task: RiskTask,
        rounds_remaining: int,
        peak_bankroll: float | None,
    ) -> BenchmarkPolicy:
        mean_shift = self._card_mean(task) - self._standard_card_mean()
        target_regime = task.rounds > 25 or not self.require_target_regime

        if mean_shift >= self.positive_shift_threshold:
            return self.high_shift_delegate
        if mean_shift <= self.negative_shift_threshold:
            return self.low_shift_delegate
        if task.initial_bankroll <= 15.0 * task.bet:
            return self.ruin_delegate
        if task.drawdown_limit <= 0.15:
            return self.drawdown_delegate
        if target_regime:
            return self.target_delegate
        return self.mean_delegate

    def action_probabilities(
        self,
        state: DecisionState,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        delegate = self._delegate(state, task, rounds_remaining, peak_bankroll)
        return delegate.action_probabilities(
            state,
            task=task,
            rounds_remaining=rounds_remaining,
            hand_depth=hand_depth,
            peak_bankroll=peak_bankroll,
        )
