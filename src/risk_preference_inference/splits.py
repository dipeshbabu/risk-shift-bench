"""Train/test split protocols for decision records."""

from __future__ import annotations

import random
from typing import Iterable

from risk_preference_inference.dataset import DecisionRecord, split_by_subject


def split_cross_subject(
    records: Iterable[DecisionRecord],
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[list[DecisionRecord], list[DecisionRecord]]:
    by_subject: dict[str, list[DecisionRecord]] = {}
    for record in records:
        by_subject.setdefault(record.subject_id, []).append(record)

    subjects = sorted(by_subject)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    cutoff = max(1, int(len(subjects) * train_fraction))
    train_subjects = set(subjects[:cutoff])

    train: list[DecisionRecord] = []
    test: list[DecisionRecord] = []
    for subject, subject_records in by_subject.items():
        if subject in train_subjects:
            train.extend(subject_records)
        else:
            test.extend(subject_records)
    return train, test


def split_cross_bankroll(
    records: Iterable[DecisionRecord],
    holdout_low: float = 0.9,
    holdout_high: float = 1.1,
) -> tuple[list[DecisionRecord], list[DecisionRecord]]:
    train: list[DecisionRecord] = []
    test: list[DecisionRecord] = []
    for record in records:
        ratio = record.current_bankroll / max(record.initial_bankroll, 1.0)
        if ratio < holdout_low or ratio > holdout_high:
            test.append(record)
        else:
            train.append(record)
    return train, test


def make_split(
    records: Iterable[DecisionRecord],
    protocol: str = "within_subject",
    train_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[list[DecisionRecord], list[DecisionRecord]]:
    records_list = list(records)
    if protocol == "within_subject":
        return split_by_subject(records_list, train_fraction=train_fraction)
    if protocol == "cross_subject":
        return split_cross_subject(records_list, train_fraction=train_fraction, seed=seed)
    if protocol == "cross_bankroll":
        return split_cross_bankroll(records_list)
    raise ValueError(f"Unknown split protocol: {protocol}")

