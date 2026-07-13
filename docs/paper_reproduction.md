# Paper reproduction commands

This file contains the run commands used for the paper artifacts. The paper
itself describes the protocol and results; execution details live here.

All commands assume dependencies were installed with:

```bash
uv sync
```

## Pre-confirmation Blackjack score caches

The robust Blackjack selectors consume cached scores from development, holdout,
audit, final-audit, and blind-audit work. The paper artifacts expect these
cache paths:

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

These are the main Blackjack paper commands. The fallback is
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

## Paper figures

Regenerate the paper figures after the artifacts above exist:

```bash
uv run python paper/scripts/make_figures.py
```

The generated files are written to `paper/figures/` as both PDF and SVG.
