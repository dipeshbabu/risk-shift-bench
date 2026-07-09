# Experiments

Smoke benchmark:

```bash
uv run python -m experiments.risk_benchmark --config configs/benchmark_smoke.json
```

Full benchmark:

```bash
uv run python -m experiments.risk_benchmark --config configs/benchmark_full.json
```

Validate full-run configuration without running the expensive experiment:

```bash
uv run python -m experiments.validate_pipeline
```

Run the full artifact pipeline:

```bash
uv run python -m experiments.run_paper_pipeline
```

Resume an existing run directory:

```bash
uv run python -m experiments.run_paper_pipeline \
  --run-root artifacts/paper_run_YYYYMMDD_HHMMSS \
  --skip-existing
```

Check that required artifacts exist:

```bash
uv run python -m experiments.check_artifacts \
  --run-root artifacts/paper_run_YYYYMMDD_HHMMSS
```

Adaptive schedule search:

```bash
uv run python -m experiments.adaptive_search --config configs/adaptive_search_full.json
```

The search artifact includes tuned CVaR schedules, tuned adaptive utility
schedules, strong static baseline summaries, and a held-out score report.

Regime-policy ablations:

```bash
uv run python -m experiments.ablation_study --config configs/benchmark_full.json
```

The ablation artifact reports aggregate scores, per-task scores, and raw
summary metrics for branch-disabled variants of the regime-adaptive ensemble.

Target-branch search:

```bash
uv run python -m experiments.target_branch_search \
  --episodes 80 \
  --max-candidates 64 \
  --selection-seeds 3
```

The search selects candidates with paired comparisons against the incumbent on
target-family train tasks and the original `RiskBlackjack-Target-v0` task. It
then reports whether the winner is worth promoting into
`signed_regime_learned_ensemble`. The promotion gate requires the candidate to
beat the incumbent target delegate on held-out target-family tasks without
regressing the original benchmark target task or the full signed-ensemble
benchmark.

Multi-seed evaluation:

```bash
uv run python -m experiments.multiseed_evaluation \
  --config configs/benchmark_full.json \
  --seeds 0,1,2,3,4 \
  --episodes 300
```

The command writes `seed_task_scores.csv`, `aggregate_scores.csv`,
`paired_deltas.csv`, and `summary.json`. Use `--input-scores` with an existing
`seed_task_scores.csv` file to regenerate the aggregate and paired tables
without rerunning simulation. The paired-delta reference defaults to
`signed_regime_learned_ensemble` and can be changed with `--reference-policy`.

Statistical report:

```bash
uv run python -m experiments.statistical_report \
  --episodes artifacts/benchmark_full/episodes.jsonl
```

Tables and figures:

```bash
uv run python -m experiments.build_tables --summary artifacts/benchmark_full/summary.json
uv run python -m experiments.make_figures --summary artifacts/benchmark_full/summary.json
```
