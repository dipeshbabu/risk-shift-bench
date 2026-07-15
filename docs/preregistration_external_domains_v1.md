# Draft preregistration: external-domain extension

Status: **draft, not registered, and not locked**.

This document prepares the next confirmatory study. It does not change the
status of the completed three-domain result. No episode from an external
confirmation suite should be run until the final protocol has an immutable
external registration URL and all source, task, policy, and analysis hashes.

The machine-readable draft is
[`configs/external_domain_extension_preregistration_draft_v1.json`](../configs/external_domain_extension_preregistration_draft_v1.json).

## External environments

The extension targets environment implementations maintained outside this
repository:

1. [Gymnasium FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/),
   using the official 4x4 and 8x8 maps and the documented `is_slippery` and
   `success_rate` parameters.
2. [OR-Gym Knapsack-v3](https://github.com/hubbs5/or-gym), a stochastic online
   knapsack environment. Its source commit and compatibility adapter must be
   pinned before the protocol is locked.
3. [Safety-Gymnasium PointGoal](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_navigation.html),
   using the official levels 0, 1, and 2. Policy checkpoint provenance and the
   MuJoCo-based execution environment must be frozen before task generation.

These choices broaden the study to hazard navigation, online resource
allocation, and continuous safe control. The wrappers may expose task features
and convert native outcomes to auditable risk metrics, but they must not replace
the external transition code.

## Required sequence

1. Pin the package version or source commit and license for every environment.
2. Implement adapters and deterministic smoke tests without running a
   confirmation task.
3. Define development and calibration tasks, train or import policy libraries,
   and freeze checkpoint hashes.
4. Define complete factorial confirmation suites and hash every task.
5. Run the development-only candidate screen and freeze all proposals.
6. Compute the total proposal count $m$. Use
   $B=\lceil\log_2(m/0.05)\rceil$ pilot batches so a unanimous sign result can
   cross the Bonferroni familywise threshold.
7. Freeze pilot seeds, final seeds, score weights, sensitivity grid, and both
   fixed-domain and domain-resampling analyses.
8. Register the protocol externally. Add the immutable URL and timestamp to the
   final JSON protocol.
9. Only then run confirmation pilots and final evaluation.

## Cost-matched comparisons

The external study will report fallback only, candidate everywhere, fit-only
proposals, pilot-verified routing, uniform pilot allocation with the same total
episode budget, and random promotion matched to the verified promotion count.
The uniform allocation must use genuinely simulated pilot outcomes; it cannot
be reconstructed from final outcomes.

## Interpretation rule

The new external-domain result will stand on its own. The opened Blackjack,
RiskPortfolio, and RiskInventory suites may appear in a combined descriptive
analysis, but they cannot contribute new confirmatory evidence. A null or
negative external result will be reported without method revision on the same
suites.
