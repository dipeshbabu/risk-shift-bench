"""Run the locked frontier development/holdout evaluation protocol."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from risk_preference_inference.envs import benchmark_tasks
from risk_preference_inference.multiseed import run_multiseed_evaluation
from risk_preference_inference.reporting import write_json


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_summary_rows(split: str, aggregate_rows: list[dict]) -> list[dict]:
    rows = []
    for row in aggregate_rows:
        if row["scope"] != "all_tasks":
            continue
        rows.append(
            {
                "split": split,
                "policy": row["policy"],
                "n": row["n"],
                "mean_score": row["mean_score"],
                "std_score": row["std_score"],
            }
        )
    return sorted(rows, key=lambda row: (row["split"], -row["mean_score"]))


def write_split_artifacts(
    out_dir: Path,
    split: str,
    rows: list[dict],
    aggregate: list[dict],
    paired_deltas: list[dict],
    metadata: dict,
) -> None:
    split_dir = out_dir / split
    write_csv(split_dir / "seed_task_scores.csv", rows)
    write_csv(split_dir / "aggregate_scores.csv", aggregate)
    write_csv(split_dir / "paired_deltas.csv", paired_deltas)
    write_json(
        split_dir / "summary.json",
        {
            **metadata,
            "split": split,
            "seed_task_scores": rows,
            "aggregate_scores": aggregate,
            "paired_deltas": paired_deltas,
        },
    )


def run_protocol(
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    out_dir: Path,
    reference_policy: str,
) -> dict:
    dev_tasks = benchmark_tasks("frontier_dev")
    holdout_tasks = benchmark_tasks("frontier_holdout")
    dev_rows, dev_aggregate, dev_deltas = run_multiseed_evaluation(
        tasks=dev_tasks,
        seeds=seeds,
        episodes=episodes,
        hand_depth=hand_depth,
        reference_policy=reference_policy,
    )
    holdout_rows, holdout_aggregate, holdout_deltas = run_multiseed_evaluation(
        tasks=holdout_tasks,
        seeds=seeds,
        episodes=episodes,
        hand_depth=hand_depth,
        reference_policy=reference_policy,
    )

    metadata = {
        "seeds": seeds,
        "episodes": episodes,
        "hand_depth": hand_depth,
        "reference_policy": reference_policy,
    }
    write_split_artifacts(out_dir, "frontier_dev", dev_rows, dev_aggregate, dev_deltas, metadata)
    write_split_artifacts(out_dir, "frontier_holdout", holdout_rows, holdout_aggregate, holdout_deltas, metadata)

    summary_rows = split_summary_rows("frontier_dev", dev_aggregate) + split_summary_rows("frontier_holdout", holdout_aggregate)
    write_csv(out_dir / "protocol_summary.csv", summary_rows)
    payload = {
        **metadata,
        "dev_tasks": [task.name for task in dev_tasks],
        "holdout_tasks": [task.name for task in holdout_tasks],
        "protocol_summary": summary_rows,
    }
    write_json(out_dir / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--hand-depth", type=int, default=1)
    parser.add_argument("--reference-policy", default="signed_regime_learned_ensemble")
    parser.add_argument("--out-dir", default="artifacts/frontier_protocol")
    args = parser.parse_args()

    payload = run_protocol(
        seeds=parse_seeds(args.seeds),
        episodes=args.episodes,
        hand_depth=args.hand_depth,
        out_dir=Path(args.out_dir),
        reference_policy=args.reference_policy,
    )
    for split in ("frontier_dev", "frontier_holdout"):
        rows = [row for row in payload["protocol_summary"] if row["split"] == split]
        print(split)
        for row in rows[:5]:
            print(f"  {row['policy']} | mean_score={row['mean_score']:.3f} std={row['std_score']:.3f}")


if __name__ == "__main__":
    main()
