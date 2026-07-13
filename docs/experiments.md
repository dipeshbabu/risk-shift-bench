# Experiments

For the exact commands used by the current paper figures and confirmation
tables, see [`paper_reproduction.md`](paper_reproduction.md). This page keeps
the broader experiment command reference.

Smoke benchmark:

```bash
uv run python -m experiments.risk_benchmark --config configs/benchmark_smoke.json
```

Full benchmark:

```bash
uv run python -m experiments.risk_benchmark --config configs/benchmark_full.json
```

Frontier benchmark:

```bash
uv run python -m experiments.risk_benchmark --config configs/benchmark_frontier.json
```

The frontier config uses the expanded task suite: the original benchmark plus
extreme card-distribution shifts, hidden episode-level deck regimes, tight
target horizons, near-ruin high-bet settings, and long-horizon drawdown stress
tests.

Locked frontier protocol:

```bash
uv run python -m experiments.frontier_protocol \
  --seeds 0,1,2 \
  --episodes 100 \
  --out-dir artifacts/frontier_protocol
```

Use the `frontier_dev` split for method development, `frontier_holdout` for
diagnostic generalization checks, and `frontier_audit` for fresh post-change
audit runs. The protocol runner writes separate dev/holdout artifacts and a
top-level `protocol_summary.csv`.

Fresh audit split:

```bash
uv run python -m experiments.multiseed_evaluation \
  --config configs/benchmark_frontier_audit.json \
  --seeds 0,1,2 \
  --episodes 100
```

Final audit split:

```bash
uv run python -m experiments.multiseed_evaluation \
  --config configs/benchmark_frontier_final_audit.json \
  --seeds 0,1,2 \
  --episodes 100
```

Blind audit split:

```bash
uv run python -m experiments.multiseed_evaluation \
  --config configs/benchmark_frontier_blind_audit.json \
  --seeds 0,1,2 \
  --episodes 100
```

Confirmation audit split:

```bash
uv run python -m experiments.multiseed_evaluation \
  --config configs/benchmark_frontier_confirmation_audit.json \
  --seeds 0,1,2 \
  --episodes 100
```

Second confirmation audit split:

```bash
uv run python -m experiments.multiseed_evaluation \
  --config configs/benchmark_frontier_confirmation_audit_v2.json \
  --seeds 0,1,2 \
  --episodes 100
```

Dev-only robust gate search:

```bash
uv run python -m experiments.robust_gate_search \
  --selection-seeds 0 \
  --eval-seeds 0,1,2 \
  --episodes 100 \
  --out-dir artifacts/robust_gate_search
```

This searches gate variants only on `frontier_dev`, freezes the selected
variant, then evaluates it on `frontier_dev`, `frontier_holdout`, and
`frontier_audit`.

Task-feature portfolio selector:

```bash
uv run python -m experiments.portfolio_selector \
  --selection-seeds 0 \
  --eval-seeds 0,1 \
  --episodes 100 \
  --dev-validation-count 4 \
  --eval-splits frontier_audit \
  --out-dir artifacts/portfolio_selector
```

This learns a nearest-neighbor policy portfolio from `frontier_dev` task
features, chooses selector hyperparameters on an internal dev-validation split,
then evaluates the frozen selector on the requested splits.

State/action blend audit:

```bash
uv run python -m experiments.multiseed_evaluation \
  --config configs/benchmark_frontier_audit.json \
  --seeds 0,1 \
  --episodes 100 \
  --out-dir artifacts/state_action_blend_audit
```

The multiseed policy set includes `state_action_blend`, a per-decision mixture
of mean, risk, drawdown, target, basic, and signed delegates. Treat this as an
experimental baseline unless it clears the locked audit split.

Validation-selected state/action blend:

```bash
uv run python -m experiments.state_action_blend_search \
  --selection-seeds 0 \
  --eval-seeds 0,1 \
  --episodes 100 \
  --dev-validation-count 4 \
  --eval-splits frontier_audit \
  --out-dir artifacts/state_action_blend_search
```

This evaluates blend-weight candidates on an internal `frontier_dev`
validation split, freezes the selected weights, then compares that frozen
policy against the standard multiseed baselines on the requested split.

Validation-selected incumbent switch:

```bash
uv run python -m experiments.incumbent_switch \
  --selection-seeds 0 \
  --eval-seeds 0,1 \
  --episodes 100 \
  --selection-suite frontier_holdout \
  --dev-validation-count 4 \
  --eval-splits frontier_audit \
  --out-dir artifacts/incumbent_switch
```

This searches task-regime switches between `learned_mixture_default` and
`signed_regime_learned_ensemble` on a named validation suite. Use
`frontier_dev` for method development, or consume `frontier_holdout` for a
stronger pre-audit selector. The frozen switch is then compared with the
standard multiseed policy set on the requested split.

Frozen incumbent switch on final audit:

```bash
uv run python -m experiments.incumbent_switch \
  --selected-search-summary configs/incumbent_switch_refined.json \
  --eval-seeds 0,1 \
  --episodes 100 \
  --eval-splits frontier_final_audit \
  --out-dir artifacts/incumbent_switch_final_audit
```

Frozen refined switch on blind audit:

```bash
uv run python -m experiments.incumbent_switch \
  --selected-search-summary configs/incumbent_switch_multidelegate.json \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --eval-splits frontier_blind_audit \
  --out-dir artifacts/incumbent_switch_blind_audit
```

Frozen multi-incumbent selector on confirmation audit:

```bash
uv run python -m experiments.incumbent_switch \
  --selected-search-summary configs/incumbent_switch_multidelegate.json \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --eval-splits frontier_confirmation_audit \
  --out-dir artifacts/incumbent_switch_confirmation_audit
```

Learned task-feature meta-selector:

```bash
uv run python -m experiments.meta_selector \
  --cv-selection \
  --selection-seeds 0 \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --train-suite frontier_dev \
  --validation-suite frontier_holdout \
  --eval-splits frontier_confirmation_audit \
  --out-dir artifacts/meta_selector_confirmation
```

This builds empirical task profiles from development tasks, selects KNN
hyperparameters on `frontier_holdout`, refits profiles on development plus
holdout tasks, and evaluates the frozen selector on the requested split.

Frozen lower-confidence selector on the first confirmation audit:

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

Frozen lower-confidence selector on the second confirmation audit:

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

Focused second-confirmation competitor batch:

```bash
uv run python -m experiments.cached_lcb_selector \
  --extra-policy-names expected_value,adaptive_utility_default,learned_mixture_searched,regime_adaptive_ensemble \
  --score-cache artifacts/meta_selector_confirmation_5seed_v1/selection_train_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_final_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_multidelegate_blind_5seed_v1/frontier_blind_audit/aggregate_scores.csv \
  --eval-split frontier_confirmation_audit_v2 \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --hand-depth 1 \
  --out-dir artifacts/lcb_selector_fresh_confirmation_v2_competitor_batch
```

The selector search also supports robust candidate selection and alternate
fallback policies:

```bash
uv run python -m experiments.cached_lcb_selector \
  --robust-selection \
  --fallback-policy learned_mixture_searched \
  --promotion-loss-weight 4.0 \
  --worst-loss-weight 2.0 \
  --score-cache artifacts/meta_selector_confirmation_5seed_v1/selection_train_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_final_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_multidelegate_blind_5seed_v1/frontier_blind_audit/aggregate_scores.csv \
  --eval-split frontier_confirmation_audit_v2 \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --hand-depth 1 \
  --out-dir artifacts/robust_searched_fallback_lcb_confirmation_v2
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

Portfolio allocation benchmark:

```bash
uv run python -m experiments.portfolio_benchmark \
  --suite portfolio_dev \
  --seeds 0,1,2 \
  --episodes 100 \
  --out-dir artifacts/portfolio_benchmark
```

Portfolio locked split protocol:

```bash
uv run python -m experiments.portfolio_benchmark \
  --suite portfolio_holdout \
  --seeds 0,1,2 \
  --episodes 100 \
  --out-dir artifacts/portfolio_benchmark

uv run python -m experiments.portfolio_benchmark \
  --suite portfolio_audit \
  --seeds 0,1,2 \
  --episodes 100 \
  --out-dir artifacts/portfolio_benchmark

uv run python -m experiments.portfolio_benchmark \
  --suite portfolio_confirmation \
  --seeds 0,1,2,3,4 \
  --episodes 100 \
  --out-dir artifacts/portfolio_benchmark
```

Portfolio robust fallback selector:

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
