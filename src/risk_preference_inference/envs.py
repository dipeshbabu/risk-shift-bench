"""Configurable risk-sensitive Blackjack benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass

from risk_preference_inference.blackjack import CARD_PROBS

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


def frontier_benchmark_tasks() -> list[RiskTask]:
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


def benchmark_suite_names() -> tuple[str, ...]:
    return ("standard", "frontier")


def benchmark_tasks(suite: str = "standard") -> list[RiskTask]:
    if suite == "standard":
        return standard_benchmark_tasks()
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
