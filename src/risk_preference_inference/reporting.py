"""Reporting helpers for benchmark outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from risk_preference_inference.benchmark import BenchmarkSummary, EpisodeResult


def write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def write_episode_jsonl(path: str | Path, results: list[EpisodeResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def write_summary_csv(path: str | Path, summaries: list[BenchmarkSummary]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(summary) for summary in summaries]
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

