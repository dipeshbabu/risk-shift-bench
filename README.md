# Risk Preference Inference

Utilities for adaptive risk-sensitive planning under bankroll constraints. This
repository contains a compact Blackjack benchmark, distributional objectives,
state-adaptive risk policies, simulation tooling, and the earlier
decision-inference utilities used for synthetic and human hit/stand data.

The central research question is whether risk sensitivity should be static or
state-adaptive in sequential decision problems with ruin, target, drawdown, and
distribution-shift constraints.

## Repository Contents

- `risk_preference_inference.envs`: benchmark task definitions such as mean-return, ruin-constrained, target-reaching, drawdown, and shifted-deck regimes.
- `risk_preference_inference.objectives`: mean, CVaR, entropic risk, OCE, ruin-constrained, and target-seeking distributional objectives.
- `risk_preference_inference.adaptive_risk`: state-adaptive CVaR schedules, adaptive utility objectives, learned objective mixtures, and constraint-aware risk gates.
- `risk_preference_inference.policies`: benchmark policies, including objective policies and a regime-adaptive ensemble.
- `risk_preference_inference.return_distributions`: exact hand-level payoff and bankroll distributions under an infinite-deck model.
- `risk_preference_inference.benchmark`: policy x task simulation and aggregate risk metrics.
- `risk_preference_inference.reporting`: JSONL, JSON, and CSV benchmark writers.
- `risk_preference_inference.blackjack`: core Blackjack state utilities.
- `risk_preference_inference.risk_models`: choice-model baselines for human/synthetic hit-stand prediction.
- `risk_preference_inference.dataset`: decision records and JSONL IO.
- `risk_preference_inference.fitting`: likelihood-based fitting for prospect-style choice models.
- `risk_preference_inference.evaluation`: action-prediction metrics.
- `risk_preference_inference.active_query`: disagreement-based state selection for data collection.
- `risk_preference_inference.synthetic`: synthetic decision data generation.
- `risk_preference_inference.statistics`: bootstrap confidence intervals and paired policy comparisons.
- `risk_preference_inference.toy_envs`: non-Blackjack toy risk tasks.

## Basic Usage

Install dependencies with `uv`:

```bash
uv sync
```

Run the adaptive risk benchmark:

```bash
uv run python -m experiments.risk_benchmark \
  --episodes 100 \
  --hand-depth 4
```

The command writes:

- `artifacts/risk_benchmark/episodes.jsonl`: one row per simulated episode.
- `artifacts/risk_benchmark/summary.csv`: policy x task aggregate metrics.
- `artifacts/risk_benchmark/summary.json`: run metadata and summary records.

Run a small focused task:

```bash
uv run python -m experiments.risk_benchmark \
  --tasks RiskBlackjack-RuinConstraint-v0 \
  --episodes 50 \
  --hand-depth 2
```

Run from a checked-in config:

```bash
uv run python -m experiments.risk_benchmark --config configs/benchmark_smoke.json
```

Validate the full paper pipeline without launching the expensive runs:

```bash
uv run python -m experiments.validate_pipeline
```

Run the full paper artifact pipeline:

```bash
uv run python -m experiments.run_paper_pipeline
```

The pipeline writes a single run directory:

```text
artifacts/paper_run_YYYYMMDD_HHMMSS/
|-- benchmark/
|-- adaptive_search/
|-- ablations/
|-- toy_benchmark/
|-- statistics/
|-- tables/
|-- figures/
|-- policy_diagnostics/
|-- theory_diagnostics/
|-- multiround_exact/
|-- configs/
`-- manifest.json
```

Resume a partially completed run:

```bash
uv run python -m experiments.run_paper_pipeline \
  --run-root artifacts/paper_run_YYYYMMDD_HHMMSS \
  --skip-existing
```

Check artifact completeness:

```bash
uv run python -m experiments.check_artifacts \
  --run-root artifacts/paper_run_YYYYMMDD_HHMMSS
```

Search adaptive CVaR and adaptive utility schedules on train tasks and evaluate held-out tasks:

```bash
uv run python -m experiments.adaptive_search --config configs/adaptive_search_smoke.json
```

Run branch-level ablations for the regime-adaptive ensemble:

```bash
uv run python -m experiments.ablation_study --config configs/benchmark_full.json
```

Export exact small-horizon final-bankroll distributions:

```bash
uv run python -m experiments.multiround_exact \
  --task RiskBlackjack-Mean-v0 \
  --rounds 2 \
  --hand-depth 1
```

Export policy diagnostics:

```bash
uv run python -m experiments.policy_diagnostics \
  --task RiskBlackjack-RuinConstraint-v0
```

Build compact tables from a benchmark summary:

```bash
uv run python -m experiments.build_tables \
  --summary artifacts/risk_benchmark/summary.json
```

Generate statistical reports and SVG figures:

```bash
uv run python -m experiments.statistical_report \
  --episodes artifacts/risk_benchmark/episodes.jsonl

uv run python -m experiments.make_figures \
  --summary artifacts/risk_benchmark/summary.json
```

Run non-Blackjack toy tasks:

```bash
uv run python -m experiments.toy_benchmark --episodes 100
```

## Benchmark Policies

The default benchmark compares:

- `basic_strategy_heuristic`
- `expected_value`
- `fixed_cvar_05`
- `fixed_entropic_001`
- `fixed_oce_1`
- `ruin_constrained_mean`
- `target_seeking_mean`
- `adaptive_cvar`
- `state_adaptive_utility`
- `learned_mixture`
- `regime_adaptive_ensemble`

The key comparison is static risk objectives versus state-adaptive risk
objectives under changing bankroll and task constraints.

## Benchmark Metrics

Each policy/task pair reports:

- mean final bankroll,
- standard deviation of final bankroll,
- 5% CVaR of final bankroll,
- ruin probability,
- target-hit probability,
- mean maximum drawdown,
- mean rounds played.

## Decision-Inference Workflow

The repository also includes the earlier decision-modeling workflow for fitting
choice models to synthetic or human hit/stand records.

Run the synthetic model-comparison workflow:

```bash
uv run python -m experiments.synthetic_experiment \
  --subjects 4 \
  --decisions 60 \
  --split within_subject
```

Run a parameter-recovery check:

```bash
uv run python -m experiments.parameter_recovery \
  --subjects 6 \
  --decisions 120
```

Collect a small terminal-based human decision file:

```bash
uv run python -m experiments.collect_decisions \
  --subject-id subject_001 \
  --decisions 30
```

## Decision Data Format

Each JSONL decision row has this shape:

```json
{
  "subject_id": "subject_000",
  "episode_id": "synthetic_000",
  "step_id": 0,
  "player_cards": [10, 6],
  "dealer_card": 10,
  "current_bankroll": 500.0,
  "initial_bankroll": 500.0,
  "bet": 20.0,
  "recent_outcomes": [],
  "action_taken": "stand",
  "target_bankroll": null,
  "timestamp": null
}
```

`action_taken` must be `hit` or `stand`.

## Notes

The benchmark currently uses an infinite-deck approximation so hand-level return
distributions are fast and deterministic. Generated artifacts, local datasets,
reports, plots, notebooks, virtual environments, and caches are intentionally
kept outside git.

## Verification

Run the lightweight regression tests with:

```bash
uv run python -m unittest discover -s tests -v
```
