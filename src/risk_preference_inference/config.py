"""Config loading for benchmark experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    episodes: int = 100
    seed: int = 0
    hand_depth: int = 4
    suite: str = "standard"
    tasks: tuple[str, ...] | None = None
    policy_set: str = "core"
    out_dir: str = "artifacts/risk_benchmark"


@dataclass(frozen=True)
class AdaptiveSearchConfig:
    train_tasks: tuple[str, ...]
    test_tasks: tuple[str, ...]
    episodes: int = 100
    seed: int = 0
    hand_depth: int = 3
    max_candidates: int | None = None
    out_dir: str = "artifacts/adaptive_search"


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("tasks") is not None:
        data["tasks"] = tuple(data["tasks"])
    return BenchmarkConfig(**data)


def load_adaptive_search_config(path: str | Path) -> AdaptiveSearchConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    data["train_tasks"] = tuple(data["train_tasks"])
    data["test_tasks"] = tuple(data["test_tasks"])
    return AdaptiveSearchConfig(**data)
