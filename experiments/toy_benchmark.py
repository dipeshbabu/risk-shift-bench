"""Run non-Blackjack toy risk benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from risk_shift_bench.reporting import write_json
from risk_shift_bench.toy_envs import run_toy_benchmark, toy_results_as_dicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="artifacts/toy_benchmark")
    args = parser.parse_args()

    results, summaries = run_toy_benchmark(args.episodes, args.seed)
    out_dir = Path(args.out_dir)
    write_json(out_dir / "episodes.json", toy_results_as_dicts(results))
    write_json(out_dir / "summary.json", summaries)
    for row in summaries:
        print(
            f"{row['task']} | {row['policy']} | "
            f"mean={row['mean_final_wealth']:.2f} "
            f"cvar5={row['cvar_5_final_wealth']:.2f} "
            f"ruin={row['ruin_probability']:.3f} "
            f"target={row['target_probability']:.3f}"
        )


if __name__ == "__main__":
    main()

