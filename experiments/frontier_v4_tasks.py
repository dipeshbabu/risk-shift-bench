"""Untouched factorial tasks for the pilot-verified routing evaluation."""

from __future__ import annotations

from risk_shift_bench.envs import (
    ACE_RICH_SHIFT,
    EXTREME_HIGH_CARD_SHIFT,
    EXTREME_LOW_CARD_SHIFT,
    HIGH_CARD_SHIFT,
    LOW_CARD_SHIFT,
    STANDARD_DECK,
    TEN_DEPLETED_SHIFT,
    CardRegimeDistribution,
    RiskTask,
)
from risk_shift_bench.portfolio_envs import (
    BEAR_MARKET,
    BULL_MARKET,
    CALM_MARKET,
    CRASH_TAIL,
    INFLATION_SHOCK,
    VOLATILE_MARKET,
    PortfolioTask,
    RegimeDistribution,
)


def blackjack_confirmation_v4_tasks() -> list[RiskTask]:
    """Return a 10-by-4 Cartesian suite defined before v4 simulation."""

    hidden_balanced: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.18),
        (STANDARD_DECK, 0.24),
        (HIGH_CARD_SHIFT, 0.18),
        (TEN_DEPLETED_SHIFT, 0.20),
        (ACE_RICH_SHIFT, 0.20),
    )
    hidden_left_tail: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.32),
        (TEN_DEPLETED_SHIFT, 0.28),
        (STANDARD_DECK, 0.18),
        (HIGH_CARD_SHIFT, 0.12),
        (ACE_RICH_SHIFT, 0.10),
    )
    hidden_switching: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.35),
        (LOW_CARD_SHIFT, 0.15),
        (HIGH_CARD_SHIFT, 0.15),
        (EXTREME_HIGH_CARD_SHIFT, 0.35),
    )
    regimes = (
        ("Standard", STANDARD_DECK, None),
        ("Low", LOW_CARD_SHIFT, None),
        ("High", HIGH_CARD_SHIFT, None),
        ("ExtremeLow", EXTREME_LOW_CARD_SHIFT, None),
        ("ExtremeHigh", EXTREME_HIGH_CARD_SHIFT, None),
        ("TenDepleted", TEN_DEPLETED_SHIFT, None),
        ("AceRich", ACE_RICH_SHIFT, None),
        ("HiddenBalanced", STANDARD_DECK, hidden_balanced),
        ("HiddenLeftTail", STANDARD_DECK, hidden_left_tail),
        ("HiddenSwitching", STANDARD_DECK, hidden_switching),
    )
    profiles = (
        (
            "RapidTarget",
            dict(
                rounds=22,
                initial_bankroll=520.0,
                bet=35.0,
                ruin_bankroll=0.0,
                target_bankroll=720.0,
                drawdown_limit=0.18,
            ),
        ),
        (
            "ExtendedTarget",
            dict(
                rounds=62,
                initial_bankroll=580.0,
                bet=25.0,
                ruin_bankroll=0.0,
                target_bankroll=870.0,
                drawdown_limit=0.13,
            ),
        ),
        (
            "CapitalPreservation",
            dict(
                rounds=48,
                initial_bankroll=550.0,
                bet=20.0,
                ruin_bankroll=0.0,
                target_bankroll=750.0,
                drawdown_limit=0.07,
            ),
        ),
        (
            "LeveragedRecovery",
            dict(
                rounds=32,
                initial_bankroll=340.0,
                bet=50.0,
                ruin_bankroll=50.0,
                target_bankroll=680.0,
                drawdown_limit=0.15,
            ),
        ),
    )
    return [
        RiskTask(
            name=f"RiskBlackjack-ConfirmV4-{regime_name}-{profile_name}-v0",
            card_probs=card_probs,
            episode_card_regimes=episode_regimes,
            **profile,
        )
        for regime_name, card_probs, episode_regimes in regimes
        for profile_name, profile in profiles
    ]

def portfolio_confirmation_v2_tasks() -> list[PortfolioTask]:
    """Return an 8-by-4 Cartesian portfolio suite defined before simulation."""

    hidden_balanced: RegimeDistribution = (
        (BEAR_MARKET, 0.20),
        (CALM_MARKET, 0.25),
        (BULL_MARKET, 0.25),
        (VOLATILE_MARKET, 0.20),
        (INFLATION_SHOCK, 0.10),
    )
    hidden_tail: RegimeDistribution = (
        (CRASH_TAIL, 0.32),
        (BEAR_MARKET, 0.24),
        (INFLATION_SHOCK, 0.20),
        (CALM_MARKET, 0.14),
        (BULL_MARKET, 0.10),
    )
    regimes = (
        ("Calm", CALM_MARKET, None),
        ("Bull", BULL_MARKET, None),
        ("Bear", BEAR_MARKET, None),
        ("Crash", CRASH_TAIL, None),
        ("Volatile", VOLATILE_MARKET, None),
        ("Inflation", INFLATION_SHOCK, None),
        ("HiddenBalanced", CALM_MARKET, hidden_balanced),
        ("HiddenTail", CALM_MARKET, hidden_tail),
    )
    profiles = (
        (
            "ShortGrowth",
            dict(
                periods=20,
                initial_capital=1050.0,
                target_capital=1250.0,
                ruin_capital=650.0,
                drawdown_limit=0.20,
            ),
        ),
        (
            "LongGrowth",
            dict(
                periods=60,
                initial_capital=1150.0,
                target_capital=1550.0,
                ruin_capital=700.0,
                drawdown_limit=0.16,
            ),
        ),
        (
            "TightRisk",
            dict(
                periods=42,
                initial_capital=1000.0,
                target_capital=1300.0,
                ruin_capital=700.0,
                drawdown_limit=0.08,
            ),
        ),
        (
            "LowCapital",
            dict(
                periods=30,
                initial_capital=760.0,
                target_capital=1040.0,
                ruin_capital=620.0,
                drawdown_limit=0.14,
            ),
        ),
    )
    return [
        PortfolioTask(
            name=f"RiskPortfolio-ConfirmV2-{regime_name}-{profile_name}-v0",
            risky_returns=returns,
            episode_regimes=episode_regimes,
            **profile,
        )
        for regime_name, returns, episode_regimes in regimes
        for profile_name, profile in profiles
    ]
