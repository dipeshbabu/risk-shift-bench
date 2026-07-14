"""Risk models used to score Blackjack actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from risk_shift_bench.blackjack import DecisionState
from risk_shift_bench.features import state_features


class RiskModel:
    name = "risk_model"
    temperature = 1.0

    def terminal_value(self, state: DecisionState, payoff: float) -> float:
        raise NotImplementedError

    def aggregate(self, values: list[tuple[float, float]]) -> float:
        return sum(prob * value for value, prob in values)


@dataclass(frozen=True)
class ExpectedValueModel(RiskModel):
    name: str = "expected_value"
    temperature: float = 1.0

    def terminal_value(self, state: DecisionState, payoff: float) -> float:
        return payoff / max(state.bet, 1.0)


@dataclass(frozen=True)
class FixedUtilityModel(RiskModel):
    """Expected utility model with a fixed curvature over wealth."""

    alpha: float = 1.0
    loss_aversion: float = 1.0
    reference: float | None = None
    temperature: float = 1.0
    name: str = "fixed_utility"

    def utility(self, wealth: float, reference: float) -> float:
        delta = wealth - reference
        magnitude = abs(delta) ** max(self.alpha, 1e-6)
        if delta >= 0:
            return magnitude
        return -self.loss_aversion * magnitude

    def terminal_value(self, state: DecisionState, payoff: float) -> float:
        reference = state.target if self.reference is None else self.reference
        before = self.utility(state.current_bankroll, reference)
        after = self.utility(state.current_bankroll + payoff, reference)
        scale = max(state.bet ** max(self.alpha, 1e-6), 1.0)
        return (after - before) / scale


@dataclass(frozen=True)
class ProspectUtilityModel(FixedUtilityModel):
    """Prospect-style utility with loss aversion and decision noise."""

    name: str = "static_prospect"


@dataclass(frozen=True)
class CumulativeProspectModel(FixedUtilityModel):
    """Rank-dependent prospect model with probability distortion."""

    probability_weight: float = 0.7
    name: str = "cumulative_prospect"

    def terminal_value(self, state: DecisionState, payoff: float) -> float:
        scaled = payoff / max(state.bet, 1.0)
        magnitude = abs(scaled) ** max(self.alpha, 1e-6)
        if scaled >= 0:
            return magnitude
        return -self.loss_aversion * magnitude

    def weight_probability(self, probability: float) -> float:
        probability = max(0.0, min(1.0, probability))
        gamma = max(self.probability_weight, 1e-6)
        if probability in (0.0, 1.0):
            return probability
        numerator = probability**gamma
        denominator = (probability**gamma + (1.0 - probability) ** gamma) ** (1.0 / gamma)
        return numerator / denominator

    def aggregate(self, values: list[tuple[float, float]]) -> float:
        gains = sorted([(value, prob) for value, prob in values if value >= 0.0], key=lambda item: item[0])
        losses = sorted([(value, prob) for value, prob in values if value < 0.0], key=lambda item: item[0])
        total = 0.0

        tail = sum(prob for _, prob in gains)
        for value, prob in gains:
            next_tail = tail - prob
            decision_weight = self.weight_probability(tail) - self.weight_probability(next_tail)
            total += decision_weight * value
            tail = next_tail

        cumulative = 0.0
        for value, prob in losses:
            next_cumulative = cumulative + prob
            decision_weight = self.weight_probability(next_cumulative) - self.weight_probability(cumulative)
            total += decision_weight * value
            cumulative = next_cumulative

        return total


@dataclass(frozen=True)
class StateDependentProspectModel(FixedUtilityModel):
    """Prospect model whose risk parameters shift with observable state."""

    alpha_bankroll_weight: float = 0.0
    alpha_loss_streak_weight: float = 0.0
    lambda_near_ruin_weight: float = 0.0
    lambda_loss_streak_weight: float = 0.0
    name: str = "state_dependent_prospect"

    def local_params(self, state: DecisionState) -> tuple[float, float]:
        features = state_features(state)
        alpha_logit = math.log(max(self.alpha, 1e-3) / max(2.0 - self.alpha, 1e-3))
        alpha_logit += self.alpha_bankroll_weight * (features["bankroll_ratio"] - 1.0)
        alpha_logit += self.alpha_loss_streak_weight * features["recent_loss_streak"]
        alpha = 2.0 / (1.0 + math.exp(-alpha_logit))
        near_ruin = max(0.0, 0.8 - features["bankroll_ratio"])
        loss_aversion = self.loss_aversion
        loss_aversion += self.lambda_near_ruin_weight * near_ruin
        loss_aversion += self.lambda_loss_streak_weight * features["recent_loss_streak"]
        return max(0.05, min(alpha, 2.0)), max(0.05, loss_aversion)

    def terminal_value(self, state: DecisionState, payoff: float) -> float:
        alpha, loss_aversion = self.local_params(state)
        local_model = FixedUtilityModel(
            alpha=alpha,
            loss_aversion=loss_aversion,
            reference=self.reference,
            temperature=self.temperature,
        )
        return local_model.terminal_value(state, payoff)


@dataclass(frozen=True)
class CVaRModel(ExpectedValueModel):
    """Lower-tail CVaR model over terminal action values."""

    alpha: float = 0.1
    name: str = "cvar"

    def aggregate(self, values: list[tuple[float, float]]) -> float:
        if not values:
            return 0.0
        remaining = max(min(self.alpha, 1.0), 1e-6)
        total = 0.0
        for value, prob in sorted(values, key=lambda item: item[0]):
            take = min(prob, remaining)
            total += take * value
            remaining -= take
            if remaining <= 1e-12:
                break
        return total / max(min(self.alpha, 1.0), 1e-6)


@dataclass(frozen=True)
class EntropicRiskModel(ExpectedValueModel):
    """Entropic certainty equivalent over terminal action values."""

    risk_aversion: float = 1.0
    name: str = "entropic_risk"

    def aggregate(self, values: list[tuple[float, float]]) -> float:
        if not values:
            return 0.0
        eta = float(self.risk_aversion)
        if abs(eta) < 1e-9:
            return super().aggregate(values)
        scaled = [-eta * value for value, _ in values]
        offset = max(scaled)
        expected_exp = sum(prob * math.exp((-eta * value) - offset) for value, prob in values)
        return -(math.log(max(expected_exp, 1e-300)) + offset) / eta


@dataclass(frozen=True)
class OptimizedCertaintyEquivalentModel(ExpectedValueModel):
    """Optimized certainty equivalent with quadratic shortfall loss."""

    shortfall_penalty: float = 1.0
    grid_points: int = 51
    name: str = "oce_quadratic_shortfall"

    def loss(self, shortfall: float) -> float:
        if shortfall <= 0.0:
            return 0.0
        return shortfall + self.shortfall_penalty * shortfall * shortfall

    def aggregate(self, values: list[tuple[float, float]]) -> float:
        if not values:
            return 0.0
        raw_values = [value for value, _ in values]
        low = min(raw_values)
        high = max(raw_values)
        if abs(high - low) < 1e-12:
            return low
        points = max(self.grid_points, 3)
        best = float("-inf")
        for idx in range(points):
            threshold = low + (high - low) * idx / (points - 1)
            penalty = sum(prob * self.loss(threshold - value) for value, prob in values)
            best = max(best, threshold - penalty)
        return best
