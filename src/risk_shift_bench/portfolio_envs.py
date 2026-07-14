"""Portfolio allocation stress-test environments."""

from __future__ import annotations

from dataclasses import dataclass


ReturnDistribution = tuple[tuple[float, float], ...]
RegimeDistribution = tuple[tuple[ReturnDistribution, float], ...]


def normalize_returns(returns: ReturnDistribution | dict[float, float]) -> ReturnDistribution:
    items = tuple(returns.items()) if isinstance(returns, dict) else tuple(returns)
    total = sum(prob for _value, prob in items)
    if total <= 0.0:
        raise ValueError("return probability mass must be positive")
    return tuple(sorted((float(value), float(prob) / total) for value, prob in items))


CALM_MARKET = normalize_returns({-0.015: 0.10, 0.0: 0.20, 0.006: 0.35, 0.014: 0.25, 0.025: 0.10})
BULL_MARKET = normalize_returns({-0.02: 0.08, 0.004: 0.12, 0.018: 0.32, 0.035: 0.32, 0.06: 0.16})
BEAR_MARKET = normalize_returns({-0.08: 0.18, -0.04: 0.28, -0.015: 0.24, 0.005: 0.20, 0.025: 0.10})
CRASH_TAIL = normalize_returns({-0.18: 0.08, -0.10: 0.16, -0.045: 0.30, 0.0: 0.28, 0.035: 0.18})
VOLATILE_MARKET = normalize_returns({-0.12: 0.12, -0.045: 0.18, 0.0: 0.26, 0.045: 0.26, 0.11: 0.18})
INFLATION_SHOCK = normalize_returns({-0.065: 0.20, -0.025: 0.28, 0.004: 0.24, 0.025: 0.18, 0.055: 0.10})


@dataclass(frozen=True)
class PortfolioTask:
    name: str
    periods: int = 24
    initial_capital: float = 1000.0
    target_capital: float = 1180.0
    ruin_capital: float = 650.0
    drawdown_limit: float = 0.22
    cash_return: float = 0.001
    risky_returns: ReturnDistribution = CALM_MARKET
    episode_regimes: RegimeDistribution | None = None


def portfolio_development_tasks() -> list[PortfolioTask]:
    mixed_regimes: RegimeDistribution = ((BEAR_MARKET, 0.30), (CALM_MARKET, 0.40), (BULL_MARKET, 0.30))
    tail_regimes: RegimeDistribution = ((CRASH_TAIL, 0.30), (BEAR_MARKET, 0.25), (CALM_MARKET, 0.25), (BULL_MARKET, 0.20))
    return [
        PortfolioTask(name="RiskPortfolio-Mean-v0"),
        PortfolioTask(name="RiskPortfolio-Target-v0", periods=30, target_capital=1240.0),
        PortfolioTask(name="RiskPortfolio-Drawdown-v0", drawdown_limit=0.12),
        PortfolioTask(name="RiskPortfolio-NearRuin-v0", initial_capital=760.0, ruin_capital=620.0, target_capital=980.0),
        PortfolioTask(name="RiskPortfolio-BullShift-v0", risky_returns=BULL_MARKET),
        PortfolioTask(name="RiskPortfolio-BearShift-v0", risky_returns=BEAR_MARKET),
        PortfolioTask(name="RiskPortfolio-CrashTail-v0", risky_returns=CRASH_TAIL, drawdown_limit=0.16),
        PortfolioTask(name="RiskPortfolio-VolatileTarget-v0", risky_returns=VOLATILE_MARKET, target_capital=1260.0),
        PortfolioTask(name="RiskPortfolio-HiddenMarket-v0", episode_regimes=mixed_regimes),
        PortfolioTask(name="RiskPortfolio-HiddenTail-v0", episode_regimes=tail_regimes, ruin_capital=700.0),
    ]


def portfolio_holdout_tasks() -> list[PortfolioTask]:
    hidden_target: RegimeDistribution = ((BEAR_MARKET, 0.20), (CALM_MARKET, 0.25), (BULL_MARKET, 0.35), (VOLATILE_MARKET, 0.20))
    adverse_tail: RegimeDistribution = ((CRASH_TAIL, 0.35), (INFLATION_SHOCK, 0.25), (CALM_MARKET, 0.25), (BULL_MARKET, 0.15))
    return [
        PortfolioTask(name="RiskPortfolio-HoldoutHiddenTarget-v0", periods=32, target_capital=1280.0, episode_regimes=hidden_target),
        PortfolioTask(
            name="RiskPortfolio-HoldoutLowCapitalTail-v0",
            periods=26,
            initial_capital=780.0,
            ruin_capital=620.0,
            target_capital=1040.0,
            episode_regimes=adverse_tail,
        ),
        PortfolioTask(name="RiskPortfolio-HoldoutInflationDrawdown-v0", periods=34, risky_returns=INFLATION_SHOCK, drawdown_limit=0.11),
        PortfolioTask(name="RiskPortfolio-HoldoutVolatileLong-v0", periods=48, risky_returns=VOLATILE_MARKET, target_capital=1360.0),
    ]


def portfolio_audit_tasks() -> list[PortfolioTask]:
    hidden_drawdown: RegimeDistribution = ((CRASH_TAIL, 0.20), (BEAR_MARKET, 0.25), (CALM_MARKET, 0.30), (BULL_MARKET, 0.25))
    return [
        PortfolioTask(name="RiskPortfolio-AuditHiddenDrawdown-v0", periods=36, drawdown_limit=0.10, episode_regimes=hidden_drawdown),
        PortfolioTask(name="RiskPortfolio-AuditCrashTarget-v0", periods=28, risky_returns=CRASH_TAIL, target_capital=1220.0),
        PortfolioTask(name="RiskPortfolio-AuditNearRuinBull-v0", initial_capital=760.0, ruin_capital=620.0, risky_returns=BULL_MARKET),
        PortfolioTask(name="RiskPortfolio-AuditInflationLong-v0", periods=44, risky_returns=INFLATION_SHOCK, target_capital=1320.0),
    ]


def portfolio_confirmation_tasks() -> list[PortfolioTask]:
    hidden_long: RegimeDistribution = ((BEAR_MARKET, 0.22), (CALM_MARKET, 0.28), (BULL_MARKET, 0.30), (VOLATILE_MARKET, 0.20))
    adverse_tail: RegimeDistribution = ((CRASH_TAIL, 0.35), (BEAR_MARKET, 0.25), (INFLATION_SHOCK, 0.20), (BULL_MARKET, 0.20))
    return [
        PortfolioTask(name="RiskPortfolio-ConfirmHiddenLongTight-v0", periods=50, target_capital=1380.0, drawdown_limit=0.10, episode_regimes=hidden_long),
        PortfolioTask(name="RiskPortfolio-ConfirmHiddenLongLoose-v0", periods=50, target_capital=1380.0, drawdown_limit=0.18, episode_regimes=hidden_long),
        PortfolioTask(
            name="RiskPortfolio-ConfirmAdverseTailLowCapital-v0",
            periods=30,
            initial_capital=790.0,
            ruin_capital=625.0,
            target_capital=1060.0,
            episode_regimes=adverse_tail,
        ),
        PortfolioTask(name="RiskPortfolio-ConfirmInflationDrawdown-v0", periods=38, risky_returns=INFLATION_SHOCK, drawdown_limit=0.10),
        PortfolioTask(name="RiskPortfolio-ConfirmVolatileShortTarget-v0", periods=18, risky_returns=VOLATILE_MARKET, target_capital=1180.0),
        PortfolioTask(name="RiskPortfolio-ConfirmCrashLongTarget-v0", periods=42, risky_returns=CRASH_TAIL, target_capital=1260.0),
    ]


def portfolio_suite_names() -> tuple[str, ...]:
    return ("portfolio_dev", "portfolio_holdout", "portfolio_audit", "portfolio_confirmation", "portfolio")


def portfolio_tasks(suite: str = "portfolio_dev") -> list[PortfolioTask]:
    suites = {
        "portfolio_dev": portfolio_development_tasks,
        "portfolio_holdout": portfolio_holdout_tasks,
        "portfolio_audit": portfolio_audit_tasks,
        "portfolio_confirmation": portfolio_confirmation_tasks,
    }
    if suite == "portfolio":
        return [
            *portfolio_development_tasks(),
            *portfolio_holdout_tasks(),
            *portfolio_audit_tasks(),
            *portfolio_confirmation_tasks(),
        ]
    if suite not in suites:
        raise ValueError(f"unknown portfolio suite: {suite}")
    return suites[suite]()
