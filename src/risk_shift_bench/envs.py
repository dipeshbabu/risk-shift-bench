"""Configurable risk-sensitive Blackjack benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass

from risk_shift_bench.blackjack import CARD_PROBS

CardDistribution = tuple[tuple[int, float], ...]


def normalize_card_probs(card_probs: dict[int, float] | CardDistribution) -> CardDistribution:
    items = tuple(card_probs.items()) if isinstance(card_probs, dict) else tuple(card_probs)
    total = sum(prob for _, prob in items)
    if total <= 0.0:
        raise ValueError("card probability mass must be positive")
    return tuple(sorted((int(card), float(prob) / total) for card, prob in items))


STANDARD_DECK = normalize_card_probs(CARD_PROBS)
LOW_CARD_SHIFT = normalize_card_probs({2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1, 9: 1, 10: 2, 11: 1})
HIGH_CARD_SHIFT = normalize_card_probs({2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 8, 11: 2})
EXTREME_LOW_CARD_SHIFT = normalize_card_probs({2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 2, 8: 1, 9: 1, 10: 1, 11: 1})
EXTREME_HIGH_CARD_SHIFT = normalize_card_probs({2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 12, 11: 4})
TEN_DEPLETED_SHIFT = normalize_card_probs({2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 2, 8: 2, 9: 2, 10: 1, 11: 1})
ACE_RICH_SHIFT = normalize_card_probs({2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 4, 11: 4})

CardRegimeDistribution = tuple[tuple[CardDistribution, float], ...]


@dataclass(frozen=True)
class RiskTask:
    name: str
    rounds: int = 25
    initial_bankroll: float = 500.0
    bet: float = 20.0
    ruin_bankroll: float = 0.0
    target_bankroll: float = 650.0
    drawdown_limit: float = 0.25
    card_probs: CardDistribution = STANDARD_DECK
    episode_card_regimes: CardRegimeDistribution | None = None


def standard_benchmark_tasks() -> list[RiskTask]:
    return [
        RiskTask(name="RiskBlackjack-Mean-v0"),
        RiskTask(name="RiskBlackjack-RuinConstraint-v0", initial_bankroll=240.0, target_bankroll=360.0),
        RiskTask(name="RiskBlackjack-Target-v0", rounds=30, target_bankroll=640.0),
        RiskTask(name="RiskBlackjack-Drawdown-v0", drawdown_limit=0.12),
        RiskTask(name="RiskBlackjack-LowCardShift-v0", card_probs=LOW_CARD_SHIFT),
        RiskTask(name="RiskBlackjack-HighCardShift-v0", card_probs=HIGH_CARD_SHIFT),
    ]


def frontier_development_tasks() -> list[RiskTask]:
    hidden_shift_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.35),
        (STANDARD_DECK, 0.30),
        (HIGH_CARD_SHIFT, 0.35),
    )
    tail_regimes: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.30),
        (TEN_DEPLETED_SHIFT, 0.20),
        (STANDARD_DECK, 0.20),
        (EXTREME_HIGH_CARD_SHIFT, 0.20),
        (ACE_RICH_SHIFT, 0.10),
    )
    return [
        *standard_benchmark_tasks(),
        RiskTask(name="RiskBlackjack-ExtremeLowCardShift-v0", card_probs=EXTREME_LOW_CARD_SHIFT),
        RiskTask(name="RiskBlackjack-ExtremeHighCardShift-v0", card_probs=EXTREME_HIGH_CARD_SHIFT),
        RiskTask(name="RiskBlackjack-TenDepletedShift-v0", card_probs=TEN_DEPLETED_SHIFT),
        RiskTask(name="RiskBlackjack-AceRichShift-v0", card_probs=ACE_RICH_SHIFT),
        RiskTask(
            name="RiskBlackjack-HiddenDeckShift-v0",
            card_probs=STANDARD_DECK,
            episode_card_regimes=hidden_shift_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-TailRegimeMixture-v0",
            card_probs=STANDARD_DECK,
            episode_card_regimes=tail_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-NearRuinHighBet-v0",
            rounds=20,
            initial_bankroll=180.0,
            bet=40.0,
            ruin_bankroll=40.0,
            target_bankroll=340.0,
            drawdown_limit=0.20,
        ),
        RiskTask(
            name="RiskBlackjack-TightTargetShortHorizon-v0",
            rounds=12,
            initial_bankroll=500.0,
            bet=30.0,
            target_bankroll=650.0,
        ),
        RiskTask(
            name="RiskBlackjack-LongHorizonTightDrawdown-v0",
            rounds=60,
            initial_bankroll=500.0,
            target_bankroll=760.0,
            drawdown_limit=0.08,
        ),
        RiskTask(
            name="RiskBlackjack-ShiftedTargetHighVariance-v0",
            rounds=35,
            initial_bankroll=420.0,
            bet=35.0,
            target_bankroll=700.0,
            card_probs=HIGH_CARD_SHIFT,
        ),
    ]


def frontier_holdout_tasks() -> list[RiskTask]:
    volatile_target_regimes: CardRegimeDistribution = (
        (TEN_DEPLETED_SHIFT, 0.25),
        (STANDARD_DECK, 0.20),
        (HIGH_CARD_SHIFT, 0.30),
        (ACE_RICH_SHIFT, 0.25),
    )
    adverse_tail_regimes: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.35),
        (TEN_DEPLETED_SHIFT, 0.30),
        (STANDARD_DECK, 0.20),
        (EXTREME_HIGH_CARD_SHIFT, 0.15),
    )
    balanced_hidden_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.20),
        (STANDARD_DECK, 0.35),
        (HIGH_CARD_SHIFT, 0.20),
        (ACE_RICH_SHIFT, 0.25),
    )
    return [
        RiskTask(
            name="RiskBlackjack-HoldoutVolatileHiddenTarget-v0",
            rounds=28,
            initial_bankroll=460.0,
            bet=30.0,
            target_bankroll=680.0,
            drawdown_limit=0.10,
            episode_card_regimes=volatile_target_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-HoldoutLowBankrollTail-v0",
            rounds=24,
            initial_bankroll=220.0,
            bet=35.0,
            ruin_bankroll=35.0,
            target_bankroll=430.0,
            drawdown_limit=0.18,
            episode_card_regimes=adverse_tail_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-HoldoutAceRichShortTarget-v0",
            rounds=14,
            initial_bankroll=480.0,
            bet=35.0,
            target_bankroll=690.0,
            card_probs=ACE_RICH_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-HoldoutTenDepletedDrawdown-v0",
            rounds=45,
            initial_bankroll=500.0,
            bet=25.0,
            target_bankroll=720.0,
            drawdown_limit=0.10,
            card_probs=TEN_DEPLETED_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-HoldoutExtremeHighRuin-v0",
            rounds=30,
            initial_bankroll=260.0,
            bet=40.0,
            ruin_bankroll=40.0,
            target_bankroll=620.0,
            card_probs=EXTREME_HIGH_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-HoldoutBalancedHiddenLong-v0",
            rounds=55,
            initial_bankroll=520.0,
            bet=25.0,
            target_bankroll=790.0,
            drawdown_limit=0.12,
            episode_card_regimes=balanced_hidden_regimes,
        ),
    ]


def frontier_audit_tasks() -> list[RiskTask]:
    mixed_hidden_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.15),
        (STANDARD_DECK, 0.25),
        (HIGH_CARD_SHIFT, 0.25),
        (ACE_RICH_SHIFT, 0.20),
        (TEN_DEPLETED_SHIFT, 0.15),
    )
    audit_tail_regimes: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.25),
        (STANDARD_DECK, 0.25),
        (EXTREME_HIGH_CARD_SHIFT, 0.25),
        (ACE_RICH_SHIFT, 0.25),
    )
    return [
        RiskTask(
            name="RiskBlackjack-AuditHiddenDrawdownTarget-v0",
            rounds=32,
            initial_bankroll=480.0,
            bet=25.0,
            target_bankroll=700.0,
            drawdown_limit=0.11,
            episode_card_regimes=mixed_hidden_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-AuditLongHiddenShift-v0",
            rounds=52,
            initial_bankroll=540.0,
            bet=25.0,
            target_bankroll=780.0,
            drawdown_limit=0.14,
            episode_card_regimes=audit_tail_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-AuditExtremeLowTarget-v0",
            rounds=30,
            initial_bankroll=500.0,
            bet=25.0,
            target_bankroll=720.0,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-AuditNearRuinHighShift-v0",
            rounds=26,
            initial_bankroll=300.0,
            bet=45.0,
            ruin_bankroll=45.0,
            target_bankroll=620.0,
            card_probs=EXTREME_HIGH_CARD_SHIFT,
        ),
    ]


def frontier_final_audit_tasks() -> list[RiskTask]:
    hidden_long_regimes: CardRegimeDistribution = (
        (TEN_DEPLETED_SHIFT, 0.20),
        (LOW_CARD_SHIFT, 0.20),
        (STANDARD_DECK, 0.25),
        (HIGH_CARD_SHIFT, 0.20),
        (ACE_RICH_SHIFT, 0.15),
    )
    adverse_bankroll_regimes: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.30),
        (TEN_DEPLETED_SHIFT, 0.25),
        (STANDARD_DECK, 0.25),
        (EXTREME_HIGH_CARD_SHIFT, 0.20),
    )
    balanced_target_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.20),
        (STANDARD_DECK, 0.30),
        (HIGH_CARD_SHIFT, 0.25),
        (ACE_RICH_SHIFT, 0.25),
    )
    return [
        RiskTask(
            name="RiskBlackjack-FinalAuditHiddenLongDrawdown-v0",
            rounds=58,
            initial_bankroll=520.0,
            bet=25.0,
            target_bankroll=810.0,
            drawdown_limit=0.09,
            episode_card_regimes=hidden_long_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-FinalAuditAdverseLowBankroll-v0",
            rounds=26,
            initial_bankroll=240.0,
            bet=40.0,
            ruin_bankroll=40.0,
            target_bankroll=500.0,
            drawdown_limit=0.16,
            episode_card_regimes=adverse_bankroll_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-FinalAuditExtremeLowLongTarget-v0",
            rounds=42,
            initial_bankroll=500.0,
            bet=25.0,
            target_bankroll=780.0,
            drawdown_limit=0.18,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-FinalAuditAceRichShortTarget-v0",
            rounds=16,
            initial_bankroll=500.0,
            bet=35.0,
            target_bankroll=700.0,
            drawdown_limit=0.22,
            card_probs=ACE_RICH_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-FinalAuditTenDepletedDrawdown-v0",
            rounds=48,
            initial_bankroll=520.0,
            bet=25.0,
            target_bankroll=760.0,
            drawdown_limit=0.10,
            card_probs=TEN_DEPLETED_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-FinalAuditBalancedHiddenTarget-v0",
            rounds=34,
            initial_bankroll=460.0,
            bet=30.0,
            target_bankroll=700.0,
            drawdown_limit=0.13,
            episode_card_regimes=balanced_target_regimes,
        ),
    ]


def frontier_blind_audit_tasks() -> list[RiskTask]:
    hidden_drawdown_regimes: CardRegimeDistribution = (
        (TEN_DEPLETED_SHIFT, 0.20),
        (LOW_CARD_SHIFT, 0.20),
        (STANDARD_DECK, 0.25),
        (HIGH_CARD_SHIFT, 0.20),
        (ACE_RICH_SHIFT, 0.15),
    )
    hidden_short_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.25),
        (STANDARD_DECK, 0.25),
        (HIGH_CARD_SHIFT, 0.25),
        (EXTREME_HIGH_CARD_SHIFT, 0.25),
    )
    adverse_tail_regimes: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.25),
        (TEN_DEPLETED_SHIFT, 0.25),
        (STANDARD_DECK, 0.25),
        (ACE_RICH_SHIFT, 0.25),
    )
    return [
        RiskTask(
            name="RiskBlackjack-BlindAuditHiddenLongModerateDrawdown-v0",
            rounds=54,
            initial_bankroll=520.0,
            bet=25.0,
            target_bankroll=800.0,
            drawdown_limit=0.13,
            episode_card_regimes=hidden_drawdown_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-BlindAuditExtremeLowShortTarget-v0",
            rounds=28,
            initial_bankroll=500.0,
            bet=25.0,
            target_bankroll=700.0,
            drawdown_limit=0.20,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-BlindAuditExtremeLowLongSafe-v0",
            rounds=50,
            initial_bankroll=560.0,
            bet=20.0,
            target_bankroll=760.0,
            drawdown_limit=0.20,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-BlindAuditTenDepletedTightDrawdown-v0",
            rounds=50,
            initial_bankroll=520.0,
            bet=25.0,
            target_bankroll=760.0,
            drawdown_limit=0.09,
            card_probs=TEN_DEPLETED_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-BlindAuditHighShiftNearRuin-v0",
            rounds=28,
            initial_bankroll=300.0,
            bet=45.0,
            ruin_bankroll=45.0,
            target_bankroll=620.0,
            drawdown_limit=0.18,
            card_probs=EXTREME_HIGH_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-BlindAuditHiddenShortTail-v0",
            rounds=18,
            initial_bankroll=460.0,
            bet=35.0,
            target_bankroll=700.0,
            drawdown_limit=0.16,
            episode_card_regimes=hidden_short_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-BlindAuditAdverseTailBankroll-v0",
            rounds=30,
            initial_bankroll=260.0,
            bet=40.0,
            ruin_bankroll=40.0,
            target_bankroll=540.0,
            drawdown_limit=0.16,
            episode_card_regimes=adverse_tail_regimes,
        ),
    ]


def frontier_confirmation_audit_tasks() -> list[RiskTask]:
    hidden_long_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.20),
        (TEN_DEPLETED_SHIFT, 0.20),
        (STANDARD_DECK, 0.25),
        (HIGH_CARD_SHIFT, 0.20),
        (ACE_RICH_SHIFT, 0.15),
    )
    hidden_short_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.20),
        (STANDARD_DECK, 0.20),
        (HIGH_CARD_SHIFT, 0.30),
        (ACE_RICH_SHIFT, 0.30),
    )
    adverse_tail_regimes: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.30),
        (TEN_DEPLETED_SHIFT, 0.20),
        (STANDARD_DECK, 0.25),
        (EXTREME_HIGH_CARD_SHIFT, 0.15),
        (ACE_RICH_SHIFT, 0.10),
    )
    return [
        RiskTask(
            name="RiskBlackjack-ConfirmHiddenLongLooseDrawdown-v0",
            rounds=56,
            initial_bankroll=520.0,
            bet=25.0,
            target_bankroll=805.0,
            drawdown_limit=0.14,
            episode_card_regimes=hidden_long_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmHiddenLongTightDrawdown-v0",
            rounds=56,
            initial_bankroll=520.0,
            bet=25.0,
            target_bankroll=805.0,
            drawdown_limit=0.10,
            episode_card_regimes=hidden_long_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmExtremeLowShortTarget-v0",
            rounds=30,
            initial_bankroll=500.0,
            bet=25.0,
            target_bankroll=710.0,
            drawdown_limit=0.22,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmExtremeLowShortTight-v0",
            rounds=26,
            initial_bankroll=500.0,
            bet=25.0,
            target_bankroll=700.0,
            drawdown_limit=0.18,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmExtremeLowLongSafe-v0",
            rounds=48,
            initial_bankroll=560.0,
            bet=20.0,
            target_bankroll=760.0,
            drawdown_limit=0.20,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmTenDepletedTightDrawdown-v0",
            rounds=50,
            initial_bankroll=520.0,
            bet=25.0,
            target_bankroll=760.0,
            drawdown_limit=0.09,
            card_probs=TEN_DEPLETED_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmHighShiftNearRuin-v0",
            rounds=28,
            initial_bankroll=300.0,
            bet=45.0,
            ruin_bankroll=45.0,
            target_bankroll=620.0,
            drawdown_limit=0.18,
            card_probs=EXTREME_HIGH_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmHiddenShortTail-v0",
            rounds=20,
            initial_bankroll=460.0,
            bet=35.0,
            target_bankroll=700.0,
            drawdown_limit=0.16,
            episode_card_regimes=hidden_short_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmAdverseTailBankroll-v0",
            rounds=30,
            initial_bankroll=260.0,
            bet=40.0,
            ruin_bankroll=40.0,
            target_bankroll=540.0,
            drawdown_limit=0.16,
            episode_card_regimes=adverse_tail_regimes,
        ),
    ]


def frontier_confirmation_audit_v2_tasks() -> list[RiskTask]:
    hidden_long_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.15),
        (TEN_DEPLETED_SHIFT, 0.25),
        (STANDARD_DECK, 0.20),
        (HIGH_CARD_SHIFT, 0.25),
        (ACE_RICH_SHIFT, 0.15),
    )
    hidden_tail_regimes: CardRegimeDistribution = (
        (EXTREME_LOW_CARD_SHIFT, 0.25),
        (TEN_DEPLETED_SHIFT, 0.25),
        (STANDARD_DECK, 0.20),
        (HIGH_CARD_SHIFT, 0.15),
        (ACE_RICH_SHIFT, 0.15),
    )
    short_hidden_regimes: CardRegimeDistribution = (
        (LOW_CARD_SHIFT, 0.20),
        (STANDARD_DECK, 0.20),
        (HIGH_CARD_SHIFT, 0.20),
        (EXTREME_HIGH_CARD_SHIFT, 0.20),
        (ACE_RICH_SHIFT, 0.20),
    )
    return [
        RiskTask(
            name="RiskBlackjack-ConfirmV2HiddenLongTightTarget-v0",
            rounds=58,
            initial_bankroll=540.0,
            bet=25.0,
            target_bankroll=830.0,
            drawdown_limit=0.10,
            episode_card_regimes=hidden_long_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmV2HiddenLongLooseTarget-v0",
            rounds=58,
            initial_bankroll=540.0,
            bet=25.0,
            target_bankroll=830.0,
            drawdown_limit=0.15,
            episode_card_regimes=hidden_long_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmV2AdverseTailBankroll-v0",
            rounds=32,
            initial_bankroll=280.0,
            bet=40.0,
            ruin_bankroll=40.0,
            target_bankroll=570.0,
            drawdown_limit=0.16,
            episode_card_regimes=hidden_tail_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmV2ShortHiddenTail-v0",
            rounds=22,
            initial_bankroll=470.0,
            bet=35.0,
            target_bankroll=720.0,
            drawdown_limit=0.15,
            episode_card_regimes=short_hidden_regimes,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmV2ExtremeLowShortTarget-v0",
            rounds=28,
            initial_bankroll=500.0,
            bet=25.0,
            target_bankroll=715.0,
            drawdown_limit=0.20,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmV2ExtremeLowLongTarget-v0",
            rounds=50,
            initial_bankroll=560.0,
            bet=20.0,
            target_bankroll=790.0,
            drawdown_limit=0.19,
            card_probs=EXTREME_LOW_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmV2TenDepletedLongDrawdown-v0",
            rounds=52,
            initial_bankroll=530.0,
            bet=25.0,
            target_bankroll=780.0,
            drawdown_limit=0.09,
            card_probs=TEN_DEPLETED_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmV2HighShiftNearRuin-v0",
            rounds=30,
            initial_bankroll=310.0,
            bet=45.0,
            ruin_bankroll=45.0,
            target_bankroll=650.0,
            drawdown_limit=0.18,
            card_probs=EXTREME_HIGH_CARD_SHIFT,
        ),
        RiskTask(
            name="RiskBlackjack-ConfirmV2AceRichShortTarget-v0",
            rounds=18,
            initial_bankroll=500.0,
            bet=35.0,
            target_bankroll=715.0,
            drawdown_limit=0.22,
            card_probs=ACE_RICH_SHIFT,
        ),
    ]


def frontier_benchmark_tasks() -> list[RiskTask]:
    return [
        *frontier_development_tasks(),
        *frontier_holdout_tasks(),
        *frontier_audit_tasks(),
        *frontier_final_audit_tasks(),
        *frontier_blind_audit_tasks(),
        *frontier_confirmation_audit_tasks(),
        *frontier_confirmation_audit_v2_tasks(),
    ]


def benchmark_suite_names() -> tuple[str, ...]:
    return (
        "standard",
        "frontier_dev",
        "frontier_holdout",
        "frontier_audit",
        "frontier_final_audit",
        "frontier_blind_audit",
        "frontier_confirmation_audit",
        "frontier_confirmation_audit_v2",
        "frontier",
    )


def benchmark_tasks(suite: str = "standard") -> list[RiskTask]:
    if suite == "standard":
        return standard_benchmark_tasks()
    if suite == "frontier_dev":
        return frontier_development_tasks()
    if suite == "frontier_holdout":
        return frontier_holdout_tasks()
    if suite == "frontier_audit":
        return frontier_audit_tasks()
    if suite == "frontier_final_audit":
        return frontier_final_audit_tasks()
    if suite == "frontier_blind_audit":
        return frontier_blind_audit_tasks()
    if suite == "frontier_confirmation_audit":
        return frontier_confirmation_audit_tasks()
    if suite == "frontier_confirmation_audit_v2":
        return frontier_confirmation_audit_v2_tasks()
    if suite == "frontier":
        return frontier_benchmark_tasks()
    raise ValueError(f"Unknown benchmark suite: {suite}")


def available_benchmark_tasks() -> list[RiskTask]:
    tasks_by_name = {task.name: task for task in frontier_benchmark_tasks()}
    return [tasks_by_name[name] for name in sorted(tasks_by_name)]


def target_family_tasks() -> list[RiskTask]:
    return [
        RiskTask(name="TargetFamily-Near-v0", rounds=20, initial_bankroll=500.0, target_bankroll=580.0),
        RiskTask(name="TargetFamily-Far-v0", rounds=35, initial_bankroll=500.0, target_bankroll=720.0),
        RiskTask(name="TargetFamily-ShortHorizon-v0", rounds=12, initial_bankroll=500.0, target_bankroll=620.0),
        RiskTask(name="TargetFamily-LongHorizon-v0", rounds=45, initial_bankroll=500.0, target_bankroll=700.0),
        RiskTask(name="TargetFamily-LowBankroll-v0", rounds=30, initial_bankroll=320.0, target_bankroll=480.0),
        RiskTask(name="TargetFamily-LargeBet-v0", rounds=25, initial_bankroll=500.0, bet=40.0, target_bankroll=700.0),
        RiskTask(name="TargetFamily-Drawdown-v0", rounds=30, initial_bankroll=500.0, target_bankroll=650.0, drawdown_limit=0.12),
        RiskTask(name="TargetFamily-LowCardShift-v0", rounds=30, initial_bankroll=500.0, target_bankroll=640.0, card_probs=LOW_CARD_SHIFT),
        RiskTask(name="TargetFamily-HighCardShift-v0", rounds=30, initial_bankroll=500.0, target_bankroll=680.0, card_probs=HIGH_CARD_SHIFT),
    ]


def target_family_split() -> tuple[list[RiskTask], list[RiskTask]]:
    tasks = target_family_tasks()
    train_names = {
        "TargetFamily-Near-v0",
        "TargetFamily-Far-v0",
        "TargetFamily-ShortHorizon-v0",
        "TargetFamily-LongHorizon-v0",
        "TargetFamily-LowBankroll-v0",
        "TargetFamily-Drawdown-v0",
    }
    train = [task for task in tasks if task.name in train_names]
    test = [task for task in tasks if task.name not in train_names]
    return train, test
