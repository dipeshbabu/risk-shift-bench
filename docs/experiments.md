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
