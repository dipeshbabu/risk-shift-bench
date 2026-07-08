"""State-adaptive risk schedules."""

from __future__ import annotations

from dataclasses import dataclass

from risk_preference_inference.objectives import (
    CVaRObjective,
    Distribution,
    DistributionalObjective,
    ObjectiveContext,
    probability_at_or_above,
    probability_at_or_below,
)


@dataclass(frozen=True)
class AdaptiveCVaRSchedule:
    """Choose CVaR tail mass from bankroll state and objective context."""

    min_alpha: float = 0.05
    max_alpha: float = 0.75
    ruin_zone_ratio: float = 0.6
    safe_zone_ratio: float = 1.25

    def alpha(self, context: ObjectiveContext) -> float:
        ratio = context.bankroll_ratio
        if ratio <= self.ruin_zone_ratio:
            return self.min_alpha
        if ratio >= self.safe_zone_ratio:
            return self.max_alpha
        span = self.safe_zone_ratio - self.ruin_zone_ratio
        weight = (ratio - self.ruin_zone_ratio) / max(span, 1e-9)
        drawdown_discount = min(context.drawdown_fraction, 0.5)
        target_pressure = min(context.target_gap_fraction, 0.5)
        alpha = self.min_alpha + weight * (self.max_alpha - self.min_alpha)
        alpha += 0.15 * target_pressure
        alpha -= 0.20 * drawdown_discount
        return max(self.min_alpha, min(self.max_alpha, alpha))


@dataclass(frozen=True)
class AdaptiveCVaRObjective(DistributionalObjective):
    schedule: AdaptiveCVaRSchedule = AdaptiveCVaRSchedule()
    ruin_penalty: float = 250.0
    target_bonus: float = 100.0
    name: str = "adaptive_cvar"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        alpha = self.schedule.alpha(context)
        base = CVaRObjective(alpha=alpha).score(distribution, context)
        ruin_prob = probability_at_or_below(distribution, context.ruin_bankroll)
        target_prob = probability_at_or_above(distribution, context.target_bankroll)
        return base - self.ruin_penalty * ruin_prob + self.target_bonus * target_prob


@dataclass(frozen=True)
class LinearAdaptiveCVaRSchedule:
    """Linear state-feature schedule for CVaR tail mass."""

    intercept: float = 0.25
    bankroll_weight: float = 0.25
    drawdown_weight: float = -0.25
    target_gap_weight: float = 0.10
    min_alpha: float = 0.01
    max_alpha: float = 0.9

    def alpha(self, context: ObjectiveContext) -> float:
        raw = (
            self.intercept
            + self.bankroll_weight * (context.bankroll_ratio - 1.0)
            + self.drawdown_weight * context.drawdown_fraction
            + self.target_gap_weight * context.target_gap_fraction
        )
        return max(self.min_alpha, min(self.max_alpha, raw))


@dataclass(frozen=True)
class LearnedAdaptiveCVaRObjective(DistributionalObjective):
    schedule: LinearAdaptiveCVaRSchedule
    ruin_penalty: float = 250.0
    target_bonus: float = 100.0
    name: str = "learned_adaptive_cvar"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        alpha = self.schedule.alpha(context)
        base = CVaRObjective(alpha=alpha).score(distribution, context)
        ruin_prob = probability_at_or_below(distribution, context.ruin_bankroll)
        target_prob = probability_at_or_above(distribution, context.target_bankroll)
        return base - self.ruin_penalty * ruin_prob + self.target_bonus * target_prob
