"""Policy factories for benchmark experiments."""

from __future__ import annotations

from risk_shift_bench.adaptive_risk import (
    AdaptiveCVaRObjective,
    AdaptiveCVaRSchedule,
    AdaptiveUtilitySchedule,
    LearnedMixtureObjective,
    LearnedMixtureSchedule,
    LearnedAdaptiveCVaRObjective,
    LinearAdaptiveCVaRSchedule,
    StateAdaptiveUtilityObjective,
)
from risk_shift_bench.objectives import (
    CVaRObjective,
    EntropicObjective,
    MeanObjective,
    OCEObjective,
    RuinConstrainedObjective,
    TargetSeekingObjective,
)
from risk_shift_bench.policies import (
    BasicStrategyPolicy,
    BenchmarkPolicy,
    RegimeAdaptivePolicy,
    SignedRegimeAdaptivePolicy,
    StateActionBlendPolicy,
    StaticObjectivePolicy,
)


def adaptive_cvar_policy(
    min_alpha: float = 0.05,
    max_alpha: float = 0.75,
    ruin_zone_ratio: float = 0.6,
    safe_zone_ratio: float = 1.25,
    ruin_penalty: float = 250.0,
    target_bonus: float = 100.0,
    name: str | None = None,
) -> BenchmarkPolicy:
    schedule = AdaptiveCVaRSchedule(
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        ruin_zone_ratio=ruin_zone_ratio,
        safe_zone_ratio=safe_zone_ratio,
    )
    objective = AdaptiveCVaRObjective(
        schedule=schedule,
        ruin_penalty=ruin_penalty,
        target_bonus=target_bonus,
    )
    policy_name = name or (
        f"adaptive_cvar_a{min_alpha:g}_{max_alpha:g}_"
        f"z{ruin_zone_ratio:g}_{safe_zone_ratio:g}_"
        f"rp{ruin_penalty:g}_tb{target_bonus:g}"
    )
    return StaticObjectivePolicy(objective, name=policy_name)


def learned_adaptive_cvar_policy(
    intercept: float = 0.25,
    bankroll_weight: float = 0.25,
    drawdown_weight: float = -0.25,
    target_gap_weight: float = 0.10,
    min_alpha: float = 0.01,
    max_alpha: float = 0.9,
    ruin_penalty: float = 250.0,
    target_bonus: float = 100.0,
    name: str | None = None,
) -> BenchmarkPolicy:
    schedule = LinearAdaptiveCVaRSchedule(
        intercept=intercept,
        bankroll_weight=bankroll_weight,
        drawdown_weight=drawdown_weight,
        target_gap_weight=target_gap_weight,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
    )
    objective = LearnedAdaptiveCVaRObjective(
        schedule=schedule,
        ruin_penalty=ruin_penalty,
        target_bonus=target_bonus,
    )
    return StaticObjectivePolicy(objective, name=name or "learned_adaptive_cvar")


def state_adaptive_utility_policy(
    low_bankroll_ratio: float = 0.55,
    safe_bankroll_ratio: float = 1.15,
    drawdown_trigger: float = 0.12,
    target_window: float = 0.35,
    terminal_window: int = 8,
    cvar_alpha: float = 0.2,
    entropic_eta: float = 0.01,
    risk_weight: float = 0.35,
    ruin_penalty: float = 400.0,
    drawdown_penalty: float = 0.35,
    target_bonus: float = 180.0,
    target_excess_weight: float = 0.15,
    name: str | None = None,
) -> BenchmarkPolicy:
    schedule = AdaptiveUtilitySchedule(
        low_bankroll_ratio=low_bankroll_ratio,
        safe_bankroll_ratio=safe_bankroll_ratio,
        drawdown_trigger=drawdown_trigger,
        target_window=target_window,
        terminal_window=terminal_window,
    )
    objective = StateAdaptiveUtilityObjective(
        schedule=schedule,
        cvar_alpha=cvar_alpha,
        entropic_eta=entropic_eta,
        risk_weight=risk_weight,
        ruin_penalty=ruin_penalty,
        drawdown_penalty=drawdown_penalty,
        target_bonus=target_bonus,
        target_excess_weight=target_excess_weight,
    )
    policy_name = name or (
        f"adaptive_utility_rw{risk_weight:g}_"
        f"lb{low_bankroll_ratio:g}_sb{safe_bankroll_ratio:g}_"
        f"tb{target_bonus:g}"
    )
    return StaticObjectivePolicy(objective, name=policy_name)


def learned_mixture_policy(
    risk_intercept: float = 0.0,
    bankroll_weight: float = 0.5,
    drawdown_weight: float = 0.5,
    deck_shift_weight: float = 0.5,
    target_intercept: float = 0.0,
    target_gap_weight: float = 0.75,
    terminal_weight: float = 0.25,
    terminal_window: int = 10,
    cvar_alpha: float = 0.25,
    entropic_eta: float = 0.025,
    oce_penalty: float = 3.0,
    entropic_weight: float = 0.4,
    cvar_weight: float = 0.1,
    oce_weight: float = 0.25,
    deck_entropic_weight: float = 0.75,
    ruin_penalty: float = 250.0,
    drawdown_penalty: float = 0.15,
    target_bonus: float = 250.0,
    target_excess_weight: float = 0.15,
    name: str | None = None,
) -> BenchmarkPolicy:
    schedule = LearnedMixtureSchedule(
        risk_intercept=risk_intercept,
        bankroll_weight=bankroll_weight,
        drawdown_weight=drawdown_weight,
        deck_shift_weight=deck_shift_weight,
        target_intercept=target_intercept,
        target_gap_weight=target_gap_weight,
        terminal_weight=terminal_weight,
        terminal_window=terminal_window,
    )
    objective = LearnedMixtureObjective(
        schedule=schedule,
        cvar_alpha=cvar_alpha,
        entropic_eta=entropic_eta,
        oce_penalty=oce_penalty,
        entropic_weight=entropic_weight,
        cvar_weight=cvar_weight,
        oce_weight=oce_weight,
        deck_entropic_weight=deck_entropic_weight,
        ruin_penalty=ruin_penalty,
        drawdown_penalty=drawdown_penalty,
        target_bonus=target_bonus,
        target_excess_weight=target_excess_weight,
    )
    policy_name = name or (
        f"learned_mixture_ew{entropic_weight:g}_"
        f"dw{deck_entropic_weight:g}_tb{target_bonus:g}"
    )
    return StaticObjectivePolicy(objective, name=policy_name)


def searched_learned_mixture_policy(name: str = "learned_mixture_searched") -> BenchmarkPolicy:
    return learned_mixture_policy(
        risk_intercept=0.0,
        bankroll_weight=0.25,
        drawdown_weight=0.5,
        deck_shift_weight=0.5,
        target_intercept=0.0,
        target_gap_weight=0.75,
        terminal_weight=0.25,
        terminal_window=8,
        cvar_alpha=0.15,
        entropic_eta=0.01,
        oce_penalty=3.0,
        entropic_weight=0.35,
        cvar_weight=0.05,
        oce_weight=0.15,
        deck_entropic_weight=1.25,
        ruin_penalty=250.0,
        drawdown_penalty=0.1,
        target_bonus=350.0,
        target_excess_weight=0.15,
        name=name,
    )


def target_branch_searched_policy(name: str = "target_branch_searched") -> BenchmarkPolicy:
    return learned_mixture_policy(
        risk_intercept=0.0,
        bankroll_weight=0.25,
        drawdown_weight=0.5,
        deck_shift_weight=0.5,
        target_intercept=0.0,
        target_gap_weight=1.25,
        terminal_weight=0.0,
        terminal_window=16,
        cvar_alpha=0.25,
        entropic_eta=0.01,
        oce_penalty=3.0,
        entropic_weight=0.35,
        cvar_weight=0.0,
        oce_weight=0.1,
        deck_entropic_weight=1.25,
        ruin_penalty=250.0,
        drawdown_penalty=0.1,
        target_bonus=500.0,
        target_excess_weight=0.5,
        name=name,
    )


def signed_regime_learned_policy(name: str = "signed_regime_learned_ensemble") -> BenchmarkPolicy:
    target_delegate = target_branch_searched_policy(name=f"{name}_target_delegate")
    return SignedRegimeAdaptivePolicy(
        name=name,
        mean_delegate=learned_mixture_policy(name=f"{name}_mean_mixture"),
        severe_ruin_delegate=BasicStrategyPolicy(name=f"{name}_severe_ruin_basic"),
        hidden_drawdown_delegate=StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=f"{name}_hidden_drawdown_oce"),
        hidden_long_delegate=StaticObjectivePolicy(MeanObjective(), name=f"{name}_hidden_long_mean"),
        near_ruin_delegate=StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=f"{name}_near_ruin_oce"),
        short_target_delegate=BasicStrategyPolicy(name=f"{name}_short_target_basic"),
        long_drawdown_delegate=BasicStrategyPolicy(name=f"{name}_long_drawdown_basic"),
        long_shift_drawdown_delegate=StaticObjectivePolicy(MeanObjective(), name=f"{name}_long_shift_drawdown_mean"),
        hidden_tail_delegate=BasicStrategyPolicy(name=f"{name}_hidden_tail_basic"),
        ruin_delegate=StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=f"{name}_ruin_oce"),
        low_shift_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name=f"{name}_low_entropic"),
        extreme_low_shift_delegate=searched_learned_mixture_policy(name=f"{name}_extreme_low_mixture"),
        target_delegate=target_delegate,
        drawdown_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.01), name=f"{name}_drawdown_entropic"),
        high_shift_delegate=learned_mixture_policy(name=f"{name}_high_mixture"),
    )


def state_action_blend_policy(name: str = "state_action_blend") -> BenchmarkPolicy:
    return StateActionBlendPolicy(
        name=name,
        mean_delegate=learned_mixture_policy(name=f"{name}_mean_mixture"),
        risk_delegate=StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=f"{name}_risk_oce"),
        drawdown_delegate=StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name=f"{name}_drawdown_entropic"),
        target_delegate=target_branch_searched_policy(name=f"{name}_target_delegate"),
        basic_delegate=BasicStrategyPolicy(name=f"{name}_basic"),
        signed_delegate=signed_regime_learned_policy(name=f"{name}_signed_delegate"),
    )


def core_policies() -> list[BenchmarkPolicy]:
    return [
        BasicStrategyPolicy(),
        StaticObjectivePolicy(MeanObjective(), name="expected_value"),
        StaticObjectivePolicy(CVaRObjective(alpha=0.05), name="fixed_cvar_05"),
        StaticObjectivePolicy(EntropicObjective(risk_aversion=0.01), name="fixed_entropic_001"),
        StaticObjectivePolicy(OCEObjective(shortfall_penalty=1.0), name="fixed_oce_1"),
        StaticObjectivePolicy(RuinConstrainedObjective(MeanObjective(), ruin_penalty=400.0), name="ruin_constrained_mean"),
        StaticObjectivePolicy(TargetSeekingObjective(MeanObjective(), target_bonus=150.0), name="target_seeking_mean"),
        adaptive_cvar_policy(name="adaptive_cvar"),
        state_adaptive_utility_policy(name="state_adaptive_utility"),
        learned_mixture_policy(name="learned_mixture"),
        searched_learned_mixture_policy(),
        RegimeAdaptivePolicy(),
        signed_regime_learned_policy(),
        state_action_blend_policy(),
    ]


def strong_baseline_grid() -> list[BenchmarkPolicy]:
    policies: list[BenchmarkPolicy] = [BasicStrategyPolicy(), StaticObjectivePolicy(MeanObjective(), name="expected_value")]
    for alpha in (0.01, 0.05, 0.1, 0.25, 0.5):
        policies.append(StaticObjectivePolicy(CVaRObjective(alpha=alpha), name=f"fixed_cvar_{alpha:g}"))
    for eta in (0.001, 0.005, 0.01, 0.025):
        policies.append(StaticObjectivePolicy(EntropicObjective(risk_aversion=eta), name=f"fixed_entropic_{eta:g}"))
    for penalty in (0.25, 0.75, 1.5, 3.0):
        policies.append(StaticObjectivePolicy(OCEObjective(shortfall_penalty=penalty), name=f"fixed_oce_{penalty:g}"))
    for penalty in (100.0, 250.0, 500.0, 1000.0):
        policies.append(StaticObjectivePolicy(RuinConstrainedObjective(MeanObjective(), ruin_penalty=penalty), name=f"ruin_mean_{penalty:g}"))
    for bonus in (50.0, 150.0, 300.0, 600.0):
        policies.append(StaticObjectivePolicy(TargetSeekingObjective(MeanObjective(), target_bonus=bonus), name=f"target_mean_{bonus:g}"))
    policies.append(adaptive_cvar_policy(name="adaptive_cvar_default"))
    policies.append(learned_adaptive_cvar_policy(name="learned_adaptive_cvar_default"))
    policies.append(state_adaptive_utility_policy(name="state_adaptive_utility_default"))
    policies.append(
        state_adaptive_utility_policy(
            low_bankroll_ratio=0.45,
            safe_bankroll_ratio=1.05,
            risk_weight=0.2,
            target_bonus=250.0,
            target_excess_weight=0.25,
            name="state_adaptive_utility_aggressive",
        )
    )
    policies.append(learned_mixture_policy(name="learned_mixture_default"))
    policies.append(
        learned_mixture_policy(
            risk_intercept=0.0,
            bankroll_weight=0.25,
            drawdown_weight=0.35,
            deck_shift_weight=0.75,
            entropic_weight=0.25,
            cvar_weight=0.05,
            oce_weight=0.15,
            deck_entropic_weight=1.0,
            target_bonus=350.0,
            target_excess_weight=0.25,
            name="learned_mixture_shift_target",
        )
    )
    policies.append(searched_learned_mixture_policy())
    policies.append(RegimeAdaptivePolicy())
    policies.append(signed_regime_learned_policy())
    policies.append(state_action_blend_policy())
    return policies
