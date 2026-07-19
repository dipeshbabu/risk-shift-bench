# Reproduction commands

This file contains the run commands for the locked benchmark artifacts. The
repository documentation keeps execution details separate from the manuscript.

All commands assume dependencies were installed with:

```bash
uv sync
```

## Pilot-verified three-domain study

The canonical final protocol is
`configs/frontier_pilot_verified_protocol.json`. It assumes a clean output path;
the runner refuses to overwrite checkpoints. First validate the task, cache,
proposal, source, and protocol hashes:

```bash
uv run python -m experiments.pilot_verified_evaluation --dry-run
```

Run seven pilot checkpoints per domain, then freeze each domain's gate file:

```bash
for domain in blackjack_v4 portfolio_v2 inventory_v1; do
  for batch in 0 1 2 3 4 5 6; do
    uv run python -m experiments.pilot_verified_evaluation --pilot "$domain" --pilot-batch "$batch"
  done
  uv run python -m experiments.pilot_verified_evaluation --lock-gates "$domain"
done
```

Run the disjoint final seeds and combine each domain:

```bash
for domain in blackjack_v4 portfolio_v2 inventory_v1; do
  for seed in 0 1 2 3 4; do
    uv run python -m experiments.pilot_verified_evaluation --eval "$domain" --eval-seed "$seed"
  done
  uv run python -m experiments.pilot_verified_evaluation --combine-domain "$domain"
done
uv run python -m experiments.pilot_verified_evaluation --combine-all
```

The canonical artifact is
`artifacts/frontier_pilot_verified_3domain_v1`. Its confirmatory result is
+1.2403% equal-domain relative improvement, with within-domain task-bootstrap 95% CI
[0.7004%, 1.8780%] and sign-flip p<1e-5. The domain raw-score effects are 0.00,
+3.65, and +70.70 for Blackjack, RiskPortfolio, and RiskInventory. These
commands reproduce the local lock; it was not externally preregistered.

## Post-confirmation robustness checks

After the canonical artifact exists, recompute the descriptive checks with:

```bash
uv run python -m experiments.pilot_verified_robustness
```

The command writes `artifacts/frontier_pilot_verified_robustness_v1`. It does
not simulate new outcomes. The main outputs are `strategy_comparison.csv`,
`random_matched_summary.json`, `pilot_budget_curve.csv`,
`score_weight_sensitivity.csv`, and `summary.json`. Expected headline values
are -1.87% for candidate everywhere, +0.53% for all fit-only proposals, and
+1.24% for pilot verification. The 81 fixed-route score variants range from
+0.96% to +1.56%, all with zero harmful accepted routes.

These values are post-confirmation because they use the opened final task
effects. The count-matched random percentile is not a primary p-value, and the
score grid was not part of the local confirmation lock.

## Registered external extension

[`preregistration_external_domains_v1.md`](preregistration_external_domains_v1.md)
documents the independent external study, publicly registered at
[doi:10.17605/OSF.IO/C576U](https://doi.org/10.17605/OSF.IO/C576U). Its exact locked-design file is
[`../configs/external_confirmation_locked_design_v1.json`](../configs/external_confirmation_locked_design_v1.json),
and the earlier planning JSON is retained only as a superseded record. Validate
the registered lock with:

```powershell
uv run python -m experiments.external_confirmation_evaluation `
  --protocol configs/external_confirmation_protocol_v1.registered.json `
  dry-run
```

The command must report valid hashes and
`confirmation_execution_allowed=true`. Every pilot, gate-locking, final, and
combine command must use the registered wrapper and follow the order listed in
the protocol document.

## Pre-confirmation Blackjack score caches

The robust Blackjack selectors consume cached scores from development, holdout,
audit, final-audit, and blind-audit work. The locked runs expect these cache
paths:

```text
artifacts/meta_selector_confirmation_5seed_v1/selection_train_scores.csv
artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_audit/aggregate_scores.csv
artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_final_audit/aggregate_scores.csv
artifacts/incumbent_switch_multidelegate_blind_5seed_v1/frontier_blind_audit/aggregate_scores.csv
```

## Family-promotion selector

This reproduces the intermediate signed-fallback family selector described in
the appendix. The learned promotions should be:

```text
hidden_long_tight -> fixed_oce_3
hidden_low_bankroll_tail -> learned_mixture_default
hidden_short_tail -> learned_mixture_default
```

```bash
uv run python -m experiments.cached_family_selector \
  --score-cache artifacts/meta_selector_confirmation_5seed_v1/selection_train_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_refined_confirm_5seed_v1/frontier_final_audit/aggregate_scores.csv \
  --score-cache artifacts/incumbent_switch_multidelegate_blind_5seed_v1/frontier_blind_audit/aggregate_scores.csv \
  --eval-split frontier_confirmation_audit \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --hand-depth 1 \
  --min-delta 2.0 \
  --out-dir artifacts/family_selector_fresh_confirmation_5seed_100ep_v1
```

## Signed-fallback lower-confidence selector

First confirmation split:

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
  --out-dir artifacts/lcb_selector_fresh_confirmation_5seed_100ep_v2
```

Second confirmation split:

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
  --out-dir artifacts/lcb_selector_fresh_confirmation_v2_5seed_100ep_v1
```

## Robust searched-fallback lower-confidence selector

These are the main Blackjack robust fallback commands. The fallback is
`learned_mixture_searched`; harmful promotions are penalized during
pre-confirmation selector search.

First confirmation split:

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
  --robust-selection \
  --fallback-policy learned_mixture_searched \
  --promotion-loss-weight 4.0 \
  --worst-loss-weight 2.0 \
  --out-dir artifacts/robust_searched_fallback_lcb_fresh_confirmation_v1_5seed_100ep_v1
```

Second confirmation split:

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
  --robust-selection \
  --fallback-policy learned_mixture_searched \
  --promotion-loss-weight 4.0 \
  --worst-loss-weight 2.0 \
  --out-dir artifacts/robust_searched_fallback_lcb_fresh_confirmation_v2_5seed_100ep_v1
```

## Locked factorial Blackjack confirmation

The third confirmation suite is evaluated from the local pre-outcome protocol,
which verifies the task, source, frozen-selector, and score-cache hashes before
running. Validate the lock first:

```bash
uv run python -m experiments.frozen_confirmation_v3 --dry-run
```

Run each locked seed into its own checkpoint. The seeds may be launched in
parallel because their output paths are disjoint:

```bash
uv run python -m experiments.frozen_confirmation_v3 --seed 0
uv run python -m experiments.frozen_confirmation_v3 --seed 1
uv run python -m experiments.frozen_confirmation_v3 --seed 2
uv run python -m experiments.frozen_confirmation_v3 --seed 3
uv run python -m experiments.frozen_confirmation_v3 --seed 4
```

After all five checkpoints exist, validate their task, policy, and seed
coverage and compute the locked task-level analysis:

```bash
uv run python -m experiments.frozen_confirmation_v3 --combine
```

The result directory is
`artifacts/frontier_confirmation_v3_frozen_5seed_100ep_v1`. The primary
selector-minus-searched-mixture result is -0.20 with task-bootstrap 95% CI
[-2.98, 1.88] and task-level sign-flip p=0.936.

## Portfolio benchmark caches

Run the portfolio development, holdout, audit, and confirmation splits:

```bash
uv run python -m experiments.portfolio_benchmark \
  --suite portfolio_dev \
  --seeds 0,1,2 \
  --episodes 100 \
  --out-dir artifacts/portfolio_benchmark

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

## Portfolio robust fallback selector

```bash
uv run python -m experiments.portfolio_lcb_selector \
  --score-cache artifacts/portfolio_benchmark/portfolio_dev/aggregate_scores.csv \
  --score-cache artifacts/portfolio_benchmark/portfolio_holdout/aggregate_scores.csv \
  --score-cache artifacts/portfolio_benchmark/portfolio_audit/aggregate_scores.csv \
  --eval-split portfolio_confirmation \
  --eval-seeds 0,1,2,3,4 \
  --episodes 100 \
  --out-dir artifacts/portfolio_lcb_selector_confirmation_5seed_100ep_v1
```
