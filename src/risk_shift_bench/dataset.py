"""Decision-level dataset schema and JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from risk_shift_bench.blackjack import ACTIONS, DecisionState


@dataclass(frozen=True)
class DecisionRecord:
    subject_id: str
    episode_id: str
    step_id: int
    player_cards: tuple[int, ...]
    dealer_card: int
    current_bankroll: float
    initial_bankroll: float
    bet: float
    recent_outcomes: tuple[float, ...]
    action_taken: str
    target_bankroll: float | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.action_taken not in ACTIONS:
            raise ValueError(f"action_taken must be one of {ACTIONS}, got {self.action_taken!r}")

    def to_state(self) -> DecisionState:
        return DecisionState(
            player_cards=tuple(self.player_cards),
            dealer_card=self.dealer_card,
            current_bankroll=self.current_bankroll,
            initial_bankroll=self.initial_bankroll,
            bet=self.bet,
            recent_outcomes=tuple(self.recent_outcomes),
            target_bankroll=self.target_bankroll,
        )


def save_jsonl(records: Iterable[DecisionRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            row = asdict(record)
            row["player_cards"] = list(record.player_cards)
            row["recent_outcomes"] = list(record.recent_outcomes)
            file.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            row["player_cards"] = tuple(row["player_cards"])
            row["recent_outcomes"] = tuple(row.get("recent_outcomes", ()))
            records.append(DecisionRecord(**row))
    return records


def split_by_subject(
    records: Iterable[DecisionRecord],
    train_fraction: float = 0.7,
) -> tuple[list[DecisionRecord], list[DecisionRecord]]:
    by_subject: dict[str, list[DecisionRecord]] = {}
    for record in records:
        by_subject.setdefault(record.subject_id, []).append(record)

    train: list[DecisionRecord] = []
    test: list[DecisionRecord] = []
    for subject_records in by_subject.values():
        ordered = sorted(subject_records, key=lambda item: (item.episode_id, item.step_id))
        cutoff = max(1, int(len(ordered) * train_fraction))
        train.extend(ordered[:cutoff])
        test.extend(ordered[cutoff:])
    return train, test

