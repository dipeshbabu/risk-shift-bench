"""Paired bootstrap and sign-flip tests for seed-task score rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from risk_shift_bench.reporting import write_json
from risk_shift_bench.statistics import paired_score_report


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-task-scores", required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--baseline-policy", required=True)
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--unit", choices=("task_seed", "task"), default="task_seed")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--randomization-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = load_csv(Path(args.seed_task_scores))
    report = paired_score_report(
        rows=rows,
        reference_policy=args.reference_policy,
        baseline_policy=args.baseline_policy,
        score_field=args.score_field,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        seed=args.seed,
        unit=args.unit,
    )
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "paired_score_report.csv", [report])
    write_json(out_dir / "paired_score_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
