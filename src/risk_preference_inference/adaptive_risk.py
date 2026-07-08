"""State-adaptive risk schedules."""

from __future__ import annotations

from dataclasses import dataclass

from risk_preference_inference.objectives import (
    CVaRObjective,
    Distribution,
    DistributionalObjective,
    EntropicObjective,
    ObjectiveContext,
    expected_excess_above,
    expected_shortfall_below,
    mean,
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


@dataclass(frozen=True)
class AdaptiveUtilitySchedule:
    """State-pressure gates for an adaptive utility objective."""

    low_bankroll_ratio: float = 0.55
    safe_bankroll_ratio: float = 1.15
    drawdown_trigger: float = 0.12
    target_window: float = 0.35
    terminal_window: int = 8

    def risk_pressure(self, context: ObjectiveContext) -> float:
        bankroll_span = max(self.safe_bankroll_ratio - self.low_bankroll_ratio, 1e-9)
        bankroll_pressure = (self.safe_bankroll_ratio - context.bankroll_ratio) / bankroll_span
        drawdown_pressure = context.drawdown_fraction / max(self.drawdown_trigger, 1e-9)
        return max(0.0, min(1.0, 0.65 * bankroll_pressure + 0.35 * drawdown_pressure))

    def target_pressure(self, context: ObjectiveContext) -> float:
        target_gap = context.target_gap_fraction / max(self.target_window, 1e-9)
        proximity = 1.0 - max(0.0, min(1.0, target_gap))
        terminal = 1.0 - max(0.0, min(1.0, context.rounds_remaining / max(self.terminal_window, 1)))
        return max(0.0, min(1.0, 0.7 * proximity + 0.3 * terminal))


@dataclass(frozen=True)
class StateAdaptiveUtilityObjective(DistributionalObjective):
    """Blend mean, CE, CVaR, and constraint probabilities using state pressure.

    The objective is intentionally mean-seeking in safe states, then increases
    risk penalties only near bankroll, ruin, or drawdown pressure. This avoids
    the failure mode where a globally conservative CVaR schedule sacrifices
    upside even when constraints are inactive.
    """

    schedule: AdaptiveUtilitySchedule = AdaptiveUtilitySchedule()
    cvar_alpha: float = 0.2
    entropic_eta: float = 0.01
    risk_weight: float = 0.35
    ruin_penalty: float = 400.0
    drawdown_penalty: float = 0.35
    target_bonus: float = 180.0
    target_excess_weight: float = 0.15
    name: str = "state_adaptive_utility"

    def score(self, distribution: Distribution, context: ObjectiveContext) -> float:
        mean_score = mean(distribution)
        cvar_score = CVaRObjective(alpha=self.cvar_alpha).score(distribution, context)
        entropic_score = EntropicObjective(risk_aversion=self.entropic_eta).score(distribution, context)
        risk_pressure = self.schedule.risk_pressure(context)
        target_pressure = self.schedule.target_pressure(context)

        risk_adjusted = (
            (1.0 - self.risk_weight * risk_pressure) * mean_score
            + (self.risk_weight * risk_pressure * 0.55) * entropic_score
            + (self.risk_weight * risk_pressure * 0.45) * cvar_score
        )

        ruin_prob = probability_at_or_below(distribution, context.ruin_bankroll)
        target_prob = probability_at_or_above(distribution, context.target_bankroll)
        drawdown_limit_bankroll = context.peak_bankroll - context.initial_bankroll * self.schedule.drawdown_trigger
        drawdown_shortfall = expected_shortfall_below(distribution, drawdown_limit_bankroll)
        target_excess = expected_excess_above(distribution, context.target_bankroll)

        return (
            risk_adjusted
            - risk_pressure * self.ruin_penalty * ruin_prob
            - risk_pressure * self.drawdown_penalty * drawdown_shortfall
            + target_pressure * self.target_bonus * target_prob
            + target_pressure * self.target_excess_weight * target_excess
        )
