"""Run the full paper artifact pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from risk_preference_inference.config import load_adaptive_search_config, load_benchmark_config
from risk_preference_inference.run_management import (
    check_artifacts,
    copy_config_snapshot,
    paper_run_paths,
    timestamped_run_root,
    write_manifest,
)


def run(cmd: list[str], outputs: list[str] | None = None, skip_existing: bool = False) -> None:
    if skip_existing and outputs and all(Path(output).exists() for output in outputs):
        print("skip existing: " + " ".join(outputs), flush=True)
        return
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default="configs/benchmark_full.json")
    parser.add_argument("--adaptive-config", default="configs/adaptive_search_full.json")
    parser.add_argument("--toy-episodes", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-root", default=None)
    args = parser.parse_args()

    py = sys.executable
    benchmark_config = load_benchmark_config(args.benchmark_config)
    adaptive_config = load_adaptive_search_config(args.adaptive_config)
    run_root = args.run_root or timestamped_run_root()
    paths = paper_run_paths(run_root)
    copy_config_snapshot(paths, args.benchmark_config, args.adaptive_config)
    write_manifest(
        paths,
        command=sys.argv,
        benchmark_config=args.benchmark_config,
        adaptive_config=args.adaptive_config,
        extra={
            "toy_episodes": args.toy_episodes,
            "bootstrap_samples": args.bootstrap_samples,
            "skip_exact": args.skip_exact,
            "benchmark_config_values": benchmark_config.__dict__,
            "adaptive_config_values": adaptive_config.__dict__,
        },
    )

    run([py, "-m", "experiments.validate_pipeline", "--benchmark-config", args.benchmark_config, "--adaptive-config", args.adaptive_config])
    run(
        [py, "-m", "experiments.risk_benchmark", "--config", args.benchmark_config, "--out-dir", paths.benchmark],
        outputs=[f"{paths.benchmark}/episodes.jsonl", f"{paths.benchmark}/summary.json"],
        skip_existing=args.skip_existing,
    )
    run(
        [py, "-m", "experiments.adaptive_search", "--config", args.adaptive_config, "--out-dir", paths.adaptive_search],
        outputs=[f"{paths.adaptive_search}/summary.json"],
        skip_existing=args.skip_existing,
    )
    run(
        [py, "-m", "experiments.toy_benchmark", "--episodes", str(args.toy_episodes), "--out-dir", paths.toy_benchmark],
        outputs=[f"{paths.toy_benchmark}/summary.json", f"{paths.toy_benchmark}/episodes.json"],
        skip_existing=args.skip_existing,
    )
    run(
        [py, "-m", "experiments.statistical_report", "--episodes", f"{paths.benchmark}/episodes.jsonl", "--samples", str(args.bootstrap_samples), "--out-dir", paths.statistics],
        outputs=[f"{paths.statistics}/final_bankroll_ci.csv"],
        skip_existing=args.skip_existing,
    )
    run(
        [py, "-m", "experiments.build_tables", "--summary", f"{paths.benchmark}/summary.json", "--out-dir", paths.tables],
        outputs=[f"{paths.tables}/best_policy_by_task.csv", f"{paths.tables}/normalized_policy_ranks.csv"],
        skip_existing=args.skip_existing,
    )
    run(
        [py, "-m", "experiments.make_figures", "--summary", f"{paths.benchmark}/summary.json", "--out-dir", paths.figures],
        outputs=[f"{paths.figures}/mean_final_bankroll.svg", f"{paths.figures}/cvar_5_final_bankroll.svg"],
        skip_existing=args.skip_existing,
    )
    run(
        [py, "-m", "experiments.policy_diagnostics", "--task", "RiskBlackjack-RuinConstraint-v0", "--out-dir", paths.policy_diagnostics, "--hand-depth", "1"],
        outputs=[f"{paths.policy_diagnostics}/diagnostics.json"],
        skip_existing=args.skip_existing,
    )
    run(
        [py, "-m", "experiments.theory_diagnostics", "--task", "RiskBlackjack-RuinConstraint-v0", "--out-dir", paths.theory_diagnostics, "--hand-depth", "1"],
        outputs=[f"{paths.theory_diagnostics}/diagnostics.json"],
        skip_existing=args.skip_existing,
    )
    if not args.skip_exact:
        run(
            [py, "-m", "experiments.multiround_exact", "--task", "RiskBlackjack-Mean-v0", "--rounds", "2", "--hand-depth", "1", "--out-dir", paths.multiround_exact],
            outputs=[f"{paths.multiround_exact}/summary.json"],
            skip_existing=args.skip_existing,
        )

    _present, missing = check_artifacts(paths, include_exact=not args.skip_exact)
    if missing:
        print("missing artifacts:", flush=True)
        for artifact in missing:
            print(f"  {artifact}", flush=True)
        raise SystemExit(1)
    print(f"paper run complete: {paths.root}", flush=True)


if __name__ == "__main__":
    main()
