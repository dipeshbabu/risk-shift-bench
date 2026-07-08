"""Build compact tables from benchmark summary JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from risk_preference_inference.tables import best_policy_by_task, normalized_policy_ranks
from risk_preference_inference.tables import to_latex_table


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
    parser.add_argument("--summary", default="artifacts/risk_benchmark/summary.json")
    parser.add_argument("--out-dir", default="artifacts/tables")
    args = parser.parse_args()

    with Path(args.summary).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    summaries = payload["summaries"] if isinstance(payload, dict) and "summaries" in payload else payload
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "best_policy_by_task.csv", best_policy_by_task(summaries))
    best = best_policy_by_task(summaries)
    ranks = normalized_policy_ranks(summaries)
    write_csv(out_dir / "best_policy_by_task.csv", best)
    write_csv(out_dir / "normalized_policy_ranks.csv", ranks)
    (out_dir / "best_policy_by_task.tex").write_text(
        to_latex_table(best, caption="Best policy by task.", label="tab:best-policy"),
        encoding="utf-8",
    )
    (out_dir / "normalized_policy_ranks.tex").write_text(
        to_latex_table(ranks, caption="Normalized policy ranks.", label="tab:policy-ranks"),
        encoding="utf-8",
    )
    print(f"wrote tables to {out_dir}")


if __name__ == "__main__":
    main()
