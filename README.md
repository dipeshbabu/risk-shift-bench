# Risk Preference Inference

Utilities for adaptive risk-sensitive planning under bankroll constraints. This
repository contains a compact Blackjack benchmark, distributional objectives,
state-adaptive risk policies, simulation tooling, and the earlier
decision-inference utilities used for synthetic and human hit/stand data.

The central research question is whether risk sensitivity should be static or
state-adaptive in sequential decision problems with ruin, target, drawdown, and
distribution-shift constraints. Reproduction commands for the locked benchmark
protocol are in [`docs/reproduction.md`](docs/reproduction.md).

## Repository Contents

- `risk_preference_inference.envs`: benchmark task suites, including standard mean-return, ruin-constrained, target-reaching, drawdown, shifted-deck, hidden-regime, and tail-risk regimes.
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
- `risk_preference_inference.multiseed`: seed-level policy comparisons for higher-confidence evaluations.
- `risk_preference_inference.robust_gate_search`: development-only search for signed-regime gate variants.
- `risk_preference_inference.portfolio_selector`: task-feature policy portfolio selection across frontier tasks.
- `risk_preference_inference.state_action_blend_search`: validation-selected per-decision blend-weight search.
- `risk_preference_inference.incumbent_switch`: validation-selected task-regime switching between strong incumbents.
- `risk_preference_inference.meta_selector`: learned task-feature KNN policy selection.
- `risk_preference_inference.family_selector`: conservative family-level delegate promotion.
- `risk_preference_inference.lcb_selector`: uncertainty-penalized lower-confidence delegate selection.
- `risk_preference_inference.portfolio_envs`: portfolio allocation task suites with hidden market regimes, drawdown, ruin, and target constraints.
- `risk_preference_inference.portfolio_benchmark`: portfolio simulator, policy grid, and benchmark summaries.
- `risk_preference_inference.portfolio_lcb_selector`: lower-confidence delegate selection for the portfolio domain.
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

Run the expanded frontier benchmark suite:

```bash
uv run python -m experiments.risk_benchmark --config configs/benchmark_frontier.json
```

The frontier suite extends the standard benchmark with extreme deck shifts,
hidden per-episode regime mixtures, near-ruin high-bet episodes, tight target
horizons, and long-horizon drawdown stress tests. For locked evaluations,
develop on `frontier_dev`, use `frontier_holdout` for diagnostic generalization
checks, and reserve audit/confirmation splits for post-freeze evaluation.

```bash
uv run python -m experiments.frontier_protocol \
  --seeds 0,1,2 \
  --episodes 100 \
  --out-dir artifacts/frontier_protocol
```

Run the final audit split after freezing a method:

```bash
uv run python -m experiments.multiseed_evaluation \
  --config configs/benchmark_frontier_final_audit.json \
  --seeds 0,1,2 \
  --episodes 100
```

Run the blind audit split after freezing the refined switch:

```bash
uv run python -m experiments.incumbent_switch \
  --selected-search-summary configs/incumbent_switch_multidelegate.json \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --eval-splits frontier_blind_audit
```

Run the post-freeze confirmation split without further method changes:

```bash
uv run python -m experiments.incumbent_switch \
  --selected-search-summary configs/incumbent_switch_multidelegate.json \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --eval-splits frontier_confirmation_audit
```

Train and evaluate the learned task-feature meta-selector:

```bash
uv run python -m experiments.meta_selector \
  --cv-selection \
  --selection-seeds 0 \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --train-suite frontier_dev \
  --validation-suite frontier_holdout \
  --eval-splits frontier_confirmation_audit
```

Evaluate the frozen lower-confidence selector on the first confirmation split:

```bash
uv run python -m experiments.cached_lcb_selector \
  --score-cache artifacts/meta_selector_confirmation_5seed_v1/selection_train_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_final_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_multidelegate_blind_5seed_v1/frontier_blind_audit/aggregate_scores.csv \
  --eval-split frontier_confirmation_audit \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --hand-depth 1 \
  --out-dir artifacts/lcb_selector_fresh_confirmation
```

Run the same frozen selector on the second confirmation split:

```bash
uv run python -m experiments.cached_lcb_selector \
  --score-cache artifacts/meta_selector_confirmation_5seed_v1/selection_train_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_final_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_multidelegate_blind_5seed_v1/frontier_blind_audit/aggregate_scores.csv \
  --eval-split frontier_confirmation_audit_v2 \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --hand-depth 1 \
  --out-dir artifacts/lcb_selector_fresh_confirmation_v2
```

Run the second environment family:

```bash
uv run python -m experiments.portfolio_benchmark \
  --suite portfolio_dev \
  --seeds 0,1,2 \
  --episodes 100 \
  --out-dir artifacts/portfolio_benchmark
```

Run the portfolio confirmation split:

```bash
uv run python -m experiments.portfolio_benchmark \
  --suite portfolio_confirmation \
  --seeds 0,1,2,3,4 \
  --episodes 100 \
  --out-dir artifacts/portfolio_benchmark
```

Train and evaluate the portfolio robust fallback selector:

```bash
uv run python -m experiments.portfolio_lcb_selector \
  --score-cache artifacts/portfolio_benchmark/portfolio_dev/aggregate_scores.csv \
  --score-cache artifacts/portfolio_benchmark/portfolio_holdout/aggregate_scores.csv \
  --score-cache artifacts/portfolio_benchmark/portfolio_audit/aggregate_scores.csv \
  --eval-split portfolio_confirmation \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --out-dir artifacts/portfolio_lcb_selector
```

Validate the full artifact pipeline without launching the expensive runs:

```bash
uv run python -m experiments.validate_pipeline
```

Run the full artifact pipeline:

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

Use `configs/benchmark_frontier.json` with the same command to run ablations on
the harder suite.

Run seed-level policy comparisons:

```bash
uv run python -m experiments.multiseed_evaluation --config configs/benchmark_full.json
```

This writes per seed/task scores, aggregate scores, and paired deltas against
`signed_regime_learned_ensemble` by default.

Search a target-specific branch for the signed ensemble:

```bash
uv run python -m experiments.target_branch_search
```

The search reports a promotion gate before any candidate should replace the
current signed-ensemble target delegate.

Search a validation-selected state/action blend:

```bash
uv run python -m experiments.state_action_blend_search \
  --selection-seeds 0 \
  --eval-seeds 0,1 \
  --episodes 100 \
  --dev-validation-count 4 \
  --eval-splits frontier_audit
```

This evaluates blend-weight candidates only on an internal `frontier_dev`
validation split, freezes the selected policy, and then compares it with the
standard multiseed baselines on the requested split.

Search a validation-selected incumbent switch:

```bash
uv run python -m experiments.incumbent_switch \
  --selected-search-summary configs/incumbent_switch_refined.json \
  --eval-seeds 0,1 \
  --episodes 100 \
  --eval-splits frontier_audit
```

This compares small task-regime switches between the learned-mixture and
signed-regime incumbents. It is meant to test whether frontier performance is
limited by objective quality or by regime selection. The checked-in refined
summary freezes the current best switch for confirmation runs.

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
- `learned_mixture_searched`
- `regime_adaptive_ensemble`
- `signed_regime_learned_ensemble`
- `state_action_blend`

The key comparison is static risk objectives versus state-adaptive risk
objectives under changing bankroll and task constraints.

`configs/benchmark_full.json` uses the standard six-task suite.
`configs/benchmark_frontier_dev.json` is the development split, and
`configs/benchmark_frontier_holdout.json` is the locked diagnostic split.
`configs/benchmark_frontier_audit.json` is a fresh audit split for post-change
generalization checks. `configs/benchmark_frontier_final_audit.json` is an
additional held-back suite for frozen-method checks.
`configs/benchmark_frontier_blind_audit.json` is reserved for post-freeze
confirmation. `configs/incumbent_switch_multidelegate.json` is the current
frozen multi-incumbent selector. `configs/benchmark_frontier_confirmation_audit.json`
is the first post-freeze confirmation suite, and
`configs/benchmark_frontier_confirmation_audit_v2.json` is the second
post-method confirmation suite. `configs/benchmark_frontier.json` runs all
frontier splits together for stress testing, but it should not be used for
tuning and final reporting at the same time.

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
