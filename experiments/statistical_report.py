"""Build bootstrap confidence intervals and paired policy comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from risk_shift_bench.statistics import confidence_table, paired_policy_differences


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    parser.add_argument("--episodes", default="artifacts/risk_benchmark/episodes.jsonl")
    parser.add_argument("--baseline-policy", default="expected_value")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--out-dir", default="artifacts/statistics")
    args = parser.parse_args()

    rows = load_jsonl(args.episodes)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "final_bankroll_ci.csv", confidence_table(rows, samples=args.samples))
    write_csv(
        out_dir / "paired_final_bankroll_vs_baseline.csv",
        paired_policy_differences(rows, args.baseline_policy, samples=args.samples),
    )
    write_csv(
        out_dir / "paired_drawdown_vs_baseline.csv",
        paired_policy_differences(rows, args.baseline_policy, metric="max_drawdown", samples=args.samples),
    )
    print(f"wrote statistical reports to {out_dir}")


if __name__ == "__main__":
    main()
