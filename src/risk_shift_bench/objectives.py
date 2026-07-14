"""Distributional objectives for risk-sensitive planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

Distribution = tuple[tuple[float, float], ...]


def normalize(distribution: Distribution | list[tuple[float, float]]) -> Distribution:
    merged: dict[float, float] = {}
    for value, prob in distribution:
        if prob <= 0.0:
            continue
        merged[float(value)] = merged.get(float(value), 0.0) + float(prob)
    total = sum(merged.values())
    if total <= 0.0:
        raise ValueError("distribution has no positive probability mass")
    return tuple(sorted((value, prob / total) for value, prob in merged.items()))


def mean(distribution: Distribution) -> float:
    return sum(value * prob for value, prob in distribution)


def variance(distribution: Distribution) -> float:
    mu = mean(distribution)
    return sum(prob * (value - mu) ** 2 for value, prob in distribution)


def cvar_lower(distribution: Distribution, alpha: float) -> float:
    alpha = max(min(alpha, 1.0), 1e-9)
    remaining = alpha
    total = 0.0
    for value, prob in sorted(distribution):
        take = min(prob, remaining)
        total += take * value
        remaining -= take
        if remaining <= 1e-12:
            break
    return total / alpha


def probability_at_or_below(distribution: Distribution, threshold: float) -> float:
    return sum(prob for value, prob in distribution if value <= threshold)


def probability_at_or_above(distribution: Distribution, threshold: float) -> float:
    return sum(prob for value, prob in distribution if value >= threshold)


def expected_shortfall_below(distribution: Distribution, threshold: float) -> float:
    return sum(prob * max(0.0, threshold - value) for value, prob in distribution)


def expected_excess_above(distribution: Distribution, threshold: float) -> float:
    return sum(prob * max(0.0, value - threshold) for value, prob in distribution)


def entropic_ce(distribution: Distribution, risk_aversion: float) -> float:
    eta = float(risk_aversion)
    if abs(eta) < 1e-9:
        return mean(distribution)
    scaled = [-eta * value for value, _ in distribution]
    offset = max(scaled)
    expected_exp = sum(prob * math.exp((-eta * value) - offset) for value, prob in distribution)
    return -(math.log(max(expected_exp, 1e-300)) + offset) / eta


@dataclass(frozen=True)
class ObjectiveContext:
    bankroll: float
    initial_bankroll: float
    ruin_bankroll: float
    target_bankroll: float
    peak_bankroll: float
    rounds_remaining: int
    bet: float = 20.0
    drawdown_limit: float = 0.25
    card_mean_shift: float = 0.0
    high_card_mass: float = 0.0

    @property
    def bankroll_ratio(self) -> float:
        return self.bankroll / max(self.initial_bankroll, 1.0)

    @property
    def drawdown_fraction(self) -> float:
        return max(0.0, self.peak_bankroll - self.bankroll) / max(self.peak_bankroll, 1.0)

    @property
    def target_gap_fraction(self) -> float:
        return max(0.0, self.target_bankroll - self.bankroll) / max(self.initial_bankroll, 1.0)

    @property
    def bet_pressure(self) -> float:
        return self.bet / max(self.bankroll, 1.0)

    @property
    def deck_shift_magnitude(self) -> float:
        return abs(self.card_mean_shift)


class DistributionalObjective:
    name = "objective"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class MeanObjective(DistributionalObjective):
    name: str = "mean"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        return mean(distribution)


@dataclass(frozen=True)
class CVaRObjective(DistributionalObjective):
    alpha: float = 0.1
    name: str = "cvar"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        return cvar_lower(distribution, self.alpha)


@dataclass(frozen=True)
class EntropicObjective(DistributionalObjective):
    risk_aversion: float = 0.01
    name: str = "entropic"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        return entropic_ce(distribution, self.risk_aversion)


@dataclass(frozen=True)
class OCEObjective(DistributionalObjective):
    shortfall_penalty: float = 1.0
    grid_points: int = 51
    name: str = "oce"

    def loss(self, shortfall: float) -> float:
        if shortfall <= 0.0:
            return 0.0
        return shortfall + self.shortfall_penalty * shortfall * shortfall

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        values = [value for value, _ in distribution]
        low = min(values)
        high = max(values)
        if abs(high - low) < 1e-12:
            return low
        best = float("-inf")
        for idx in range(max(self.grid_points, 3)):
            threshold = low + (high - low) * idx / (max(self.grid_points, 3) - 1)
            penalty = sum(prob * self.loss((threshold - value) / max(context.initial_bankroll, 1.0)) for value, prob in distribution)
            best = max(best, threshold - context.initial_bankroll * penalty)
        return best


@dataclass(frozen=True)
class RuinConstrainedObjective(DistributionalObjective):
    base: DistributionalObjective
    ruin_penalty: float = 500.0
    name: str = "ruin_constrained"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        ruin_prob = probability_at_or_below(distribution, context.ruin_bankroll)
        return self.base.score(distribution, context) - self.ruin_penalty * ruin_prob


@dataclass(frozen=True)
class TargetSeekingObjective(DistributionalObjective):
    base: DistributionalObjective
    target_bonus: float = 200.0
    name: str = "target_seeking"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        target_prob = probability_at_or_above(distribution, context.target_bankroll)
        return self.base.score(distribution, context) + self.target_bonus * target_prob
