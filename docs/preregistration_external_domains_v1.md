# External-domain confirmation protocol

Status: **publicly registered; confirmation execution not started**.

This protocol defines an independent confirmation study. It does not change the
status of the completed three-domain result. No external confirmation episode
was run before the locked design received an immutable public registration URL
and timestamp. All confirmation commands must use the registered wrapper.

The earlier planning record is
[`configs/external_domain_extension_preregistration_draft_v1.json`](../configs/external_domain_extension_preregistration_draft_v1.json).
The registration-ready payload is generated as
`configs/external_confirmation_locked_design_v1.json`, with a separate
registration wrapper. Before execution, the evaluator checks the registered
design byte hash and canonical hash, source hashes, task manifests, development
artifacts, and frozen proposal table.

## External environments

The study uses transition code maintained outside this repository:

1. [Gymnasium FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/),
   pinned to commit `53bf3e9a884783eb72ad3fc8b15780914c97c3e1`, with the official 4x4 and
   8x8 maps, explicit 100-step and 200-step horizons respectively, and the
   documented `is_slippery` and `success_rate` parameters.
2. [OR-Gym Knapsack-v3](https://github.com/hubbs5/or-gym), pinned to commit
   `0b18d16e569e2db70e83f09e867b53bdb4b87298`. The unchanged transition source
   runs through a compatibility adapter using Gym 0.26.2 and NumPy 1.26.4.
3. [Safety-Gymnasium PointGoal](https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_navigation.html),
   using official levels 0, 1, and 2 at commit
   `98231340a4c5b223c8d111fa9597d81836ce09b4`, with Gymnasium 0.28.1,
   Gymnasium-Robotics 1.2.2, MuJoCo 2.3.3, and NumPy 1.23.5.

These choices cover hazard navigation, online resource allocation, and
continuous safe control. Each execution command must supply an external Git
checkout at the locked commit; the adapter rejects a dirty worktree and verifies
`HEAD`, Python, and the locked runtime dependency versions before importing the
package.

## Frozen study sequence

1. Run only the disjoint development and calibration tasks.
2. Retain at most one positive-mean candidate per domain and freeze every
   confirmation proposal.
3. Let $m$ be the complete proposal-family size and set
   $B=\lceil\log_2(m/0.05)\rceil$. A proposal passes only if all $B$ paired
   batch advantages are positive and its mean advantage is positive.
4. Freeze pilot seeds, final seeds, task manifests, source hashes, score
   weights, the sensitivity grid, and the analysis implementation.
5. Register the resulting JSON externally.
6. Only then run the confirmation pilots and final evaluation.

The Bonferroni threshold controls the familywise error rate over the complete
proposal family. Pilot and final seed ranges are disjoint. Zero batch advantage
counts as a rejection.

The frozen router produced 23 proposals: 4 in FrozenLake, 12 in online
knapsack, and 7 in PointGoal. The locked gate therefore uses nine batches of 20
episodes per policy. Both the proposal-focused allocation and the outcome-blind
random-task allocation use 8,280 candidate-plus-fallback pilot episodes. The
exact locked-design file SHA-256—the digest to verify against the external
registry—is
`5c102349a41306537cd15cfda3b843db92efd26f40bf3a6d99f1a8000b1da095`.
Its format-independent canonical JSON SHA-256 is
`17321e553c75f7991b899e22b254fb38949549e1235d7462c3aea726a5e65694`.

## Cost-matched comparisons

The study reports fallback only, candidate everywhere, fit-only proposals,
pilot-verified routing, and an outcome-blind random-task allocation. The random
allocation samples the same number of tasks as the proposal family before any
outcome is observed, then gives every sampled task the complete familywise gate
budget. Its candidate-plus-fallback episode count matches the proposal-focused
method exactly. A thin uniform-all-task allocation remains in the JSON as a
budget-accounting diagnostic, not as a viable gate.

## Reproducible execution boundary

Clone each upstream repository outside this repository and check out the exact
commit recorded in `experiments/external_study_design.py`. Use an isolated
Python environment for each domain because the pinned packages have
incompatible dependencies. Development and calibration may be rerun with
`experiments.external_development`; that command cannot select the confirmation
split.

The frozen payload was registered on the Open Science Framework at
[doi:10.17605/OSF.IO/C576U](https://doi.org/10.17605/OSF.IO/C576U) on
2026-07-15 at `23:36:07.848189Z`. The archived 36,097-byte file and the local
locked design both have SHA-256
`5c102349a41306537cd15cfda3b843db92efd26f40bf3a6d99f1a8000b1da095`.

Validate the registered wrapper without regenerating or overwriting the lock:

```powershell
uv run python -m experiments.external_confirmation_evaluation `
  --protocol configs/external_confirmation_protocol_v1.registered.json `
  dry-run
```

The dry run must print `protocol_status=externally_registered_locked`,
`protocol_hashes_valid=true`, and `confirmation_execution_allowed=true`.

Pilot and final commands require
`configs/external_confirmation_protocol_v1.registered.json`. An unregistered
draft is rejected before an environment can be reset.

Pass that wrapper explicitly. For example, one locked pilot batch and one locked
final seed are run as follows (repeat over the indices, modes, and domains
recorded in the design):

```powershell
uv run python -m experiments.external_confirmation_evaluation `
  --protocol configs/external_confirmation_protocol_v1.registered.json `
  pilot --mode proposal --domain gymnasium_frozenlake --batch-index 0 `
  --environment-source "<clean Gymnasium checkout>"

uv run python -m experiments.external_confirmation_evaluation `
  --protocol configs/external_confirmation_protocol_v1.registered.json `
  final --domain gymnasium_frozenlake --seed-index 0 `
  --environment-source "<clean Gymnasium checkout>"
```

## Interpretation rule

The external-domain result stands on its own. The opened Blackjack,
RiskPortfolio, and RiskInventory suites may appear in a combined descriptive
analysis, but they cannot contribute new confirmatory evidence. A null or
negative external result must be reported without revising the method on these
suites.
