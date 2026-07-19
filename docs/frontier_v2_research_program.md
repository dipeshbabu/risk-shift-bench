# RiskShiftBench v2 research program

Working title: **Anytime-Valid Familywise Policy Routing Under Distribution
Shift**

Status: development-only design. This document does not amend, reinterpret, or
extend the completed v1 registration. No v2 confirmation task may be reset
until a separate machine-readable design has been publicly registered.

## Why v2 needs a new method

The v1 external study made the deployment rule prospective and familywise, but
its mathematical claim is about nine independent batch signs. It does not
directly test the mean score effect, cannot stop early, spends the same pilot
budget on easy and hard proposals, and does not bound the expected improvement
of the deployed router. The external evidence also spans only three domains.

Adding tasks without changing the method would improve breadth but would not
make the contribution substantially deeper. V2 therefore has two independent
requirements:

1. a stronger deployment theorem; and
2. a broader, budget-matched external evaluation.

## Proposed statistical object

For proposal \(i\), let \(X_{i,t}\in[a_i,b_i]\) be the paired
candidate-minus-fallback score difference from the next pilot unit. Scores and
bounds are fixed before confirmation. The deployment margin is
\(\delta_i\geq 0\), and the task-level null is

\[
H_i:\quad
\mathbb{E}[X_{i,t}\mid\mathcal{F}_{t-1}]\leq\delta_i.
\]

For a fixed betting fraction \(\lambda\geq0\), Hoeffding's lemma gives the
test supermartingale

\[
E_{i,n}(\lambda)=
\exp\left\{
\lambda\sum_{t=1}^{n}(X_{i,t}-\delta_i)
-\frac{n\lambda^2(b_i-a_i)^2}{8}
\right\}.
\]

V2 mixes a prespecified finite grid of betting fractions. A convex mixture of
nonnegative test supermartingales remains a test supermartingale. Proposal
\(i\) is deployed when its mixture e-process reaches \(1/\alpha_i\), where the
task weights are frozen from development evidence and satisfy
\(\sum_i\alpha_i\leq\alpha\).

## Primary theorem target

Assume that every next paired observation remains within its registered bounds
and satisfies the task-level conditional-mean null when that null is true.
Tasks may be interleaved using any predictable, outcome-adaptive allocation
rule, and sampling may stop at arbitrary data-dependent times. Then

\[
\Pr\{\text{at least one proposal with }\mu_i\leq\delta_i
\text{ is deployed}\}
\leq\sum_i\alpha_i\leq\alpha.
\]

Proof sketch: each task mixture is an e-process under its null. Optional
skipping preserves the supermartingale property under predictable adaptive
allocation. Ville's inequality bounds the probability that task \(i\) ever
crosses \(1/\alpha_i\) by \(\alpha_i\). A union bound completes the result. No
independence assumption across tasks is needed; the conditional validity of
the next within-task observation is the substantive assumption.

On the same event, if fallback deployment is defined to have difference zero,
the expected score improvement of the deployed router on the fixed task family
obeys

\[
\sum_i w_i R_i\mu_i
>\sum_i w_i R_i\delta_i\geq0,
\]

where \(R_i\) records candidate deployment. This is a familywise lower bound
on expected improvement over fallback, not an oracle-regret guarantee and not
a guarantee about every finite final sample.

The reference implementation is
`experiments/anytime_familywise_router.py`. It contains two e-processes. A
finite Hoeffding mixture is the conservative validity baseline. The primary
development method is a mixture of bounded betting processes. After mapping
the observation to \(Y_{i,t}\in[0,1]\) with null mean \(q_i\), each component
updates as

\[
B_{i,n}(\lambda)=\prod_{t=1}^{n}
\{1+\lambda(Y_{i,t}-q_i)\},
\qquad 0\leq\lambda<1/q_i.
\]

The betting fractions are fixed before sampling, every factor is nonnegative,
and the conditional expected factor is at most one under the null. Mixing the
components therefore preserves anytime validity while adapting evidence growth
to effect size. Before v2 is locked, this finite mixture should also be compared
with tighter Bentkus and predictable plug-in betting constructions:

- [Near-Optimal Confidence Sequences for Bounded Random Variables](https://proceedings.mlr.press/v139/kuchibhotla21a.html)
- [Estimating Means of Bounded Random Variables by Betting](https://arxiv.org/abs/2010.09686)
- [Familywise Error Rate Control by Interactive Unmasking](https://proceedings.mlr.press/v119/duan20d.html)

## Adaptive pilot allocation

Validity must not depend on the allocation heuristic. The first implementation
therefore separates the e-process from sampling. It includes:

- uniform round-robin allocation; and
- a resolution heuristic that forces initial coverage, estimates the remaining
  evidence needed for either decision, and samples the proposal predicted to
  resolve next.

The primary development scheduler now adds a certified betting allocation.
Before confirmation, each task receives a score-scale planning gap
\(\Delta_i>0\) derived only
from development evidence and a task-level resolution-failure budget
\(\beta_i\), with \(\sum_i\beta_i\leq\beta\). For a standardized paired
difference \(Z_t\in[-1,1]\), let \(d_i=\Delta_i/r_i\), where \(r_i\) is the
radius of the symmetric registered score-difference interval. For a fixed
betting fraction \(f\in(0,1)\) and a conditional standardized mean gap of at
least \(d\) in either direction, concavity gives the per-observation expected
log-growth bound

\[
g_f(d)=\frac{1}{2}\log(1-f^2)
+\frac{d}{2}\log\left(\frac{1+f}{1-f}\right).
\]

The range of the log increment is
\(h_f=\log((1+f)/(1-f))\). Conditional Hoeffding--Azuma concentration
therefore implies, with probability at least \(1-\beta_i\),

\[
\sum_{t=1}^n\log(1+fZ_t)
\geq ng_f(d_i)-h_f\sqrt{n\log(1/\beta_i)/2}.
\]

If the component has frozen mixture weight \(v_f\), its task e-process has
crossed \(1/\alpha_i\) once the right-hand side reaches
\(\log(1/(\alpha_i v_f))\). The implementation solves this quadratic in
\(\sqrt n\) for every frozen betting fraction and uses the smallest integer
bound \(n_i^*\). The scheduler forces initial coverage and then completes the
smallest remaining registered quota first. If no per-task cap truncates a
quota and the global budget is at least \(\sum_i n_i^*\), every task whose
conditional mean is separated from the margin by at least its planning gap is
resolved with probability at least \(1-\sum_i\beta_i\). No independence across
tasks is used.

Planning gaps affect only this cost guarantee. If they are optimistic, the
familywise false-deployment theorem still holds because acceptance continues
to use the original e-process and alpha thresholds; the affected task may
simply remain unresolved. `CertifiedSampleTarget` exposes both the theoretical
quota and the scheduled quota and marks every cap-truncated target. Efficiency
is evaluated at identical candidate-plus-fallback episode budgets.

The first transparent reference is now implemented in
`experiments/familywise_policy_baselines.py`. For task \(i\), it spends the
one-sided error budget over sample sizes as

\[
\gamma_{i,n}=\frac{6\alpha_i}{\pi^2n^2},\qquad
r_{i,n}=(b_i-a_i)
\sqrt{\frac{\log(1/\gamma_{i,n})}{2n}}.
\]

Hoeffding's inequality and

\[
\sum_{n=1}^{\infty}\gamma_{i,n}=\alpha_i
\]

give a time-uniform lower confidence sequence
\(\bar X_{i,n}-r_{i,n}\) with task-level error at most \(\alpha_i\).
A separate upper sequence supports safe futility decisions. The racing rule
samples an unresolved task with the widest interval and removes a task after
either boundary crosses the deployment margin. On the simultaneous coverage
event, a task with gap \(\Delta_i=|\mu_i-\delta_i|>0\) is resolved no later
than the first \(n\) satisfying \(2r_{i,n}<\Delta_i\), subject to its frozen
per-task cap. Thus, when the global cap is nonbinding, the total sample count
is at most the sum of these task-specific resolution bounds. This baseline is
deliberately conservative, but it supplies an implementation-checkable
familywise proof and a concrete gap-dependent cost statement against which
the betting allocation can be judged.

## Required baselines

Every comparison must receive the same pilot episode budget and the same
candidate/fallback pairs.

1. fallback only;
2. candidate everywhere;
3. fit-only routing;
4. v1 fixed unanimity with Bonferroni;
5. fixed-sample Bonferroni mean test;
6. fixed-sample Holm step-down test;
7. anytime mixture e-process with uniform allocation;
8. anytime mixture e-process with active allocation;
9. an outcome-blind random-task allocation; and
10. a robust test-selection baseline inspired by
   [RPOSST](https://proceedings.mlr.press/v216/morrill23a.html).

The primary comparisons are familywise-valid methods. Candidate-everywhere and
fit-only routing remain descriptive upper-recall references.

`experiments/familywise_policy_comparison.py` implements paired synthetic
comparisons for fixed-sample Hoeffding tests with Bonferroni or Holm correction,
fixed-sample sign tests with Bonferroni or Holm correction, the alpha-spending
racing rule, the Hoeffding-mixture router, and the betting-mixture router. The
sign methods test an independent sign null rather than the conditional-mean
null and are labeled separately; they are not interchangeable guarantees.

`experiments/robust_test_subset_baseline.py` implements a separate
RPOSST-inspired comparison in the task-composition layer. It greedily selects a
task subset and uses projected subgradient optimization to fit simplex weights
that minimize the worst absolute full-test score error across tuning policies
and frozen target task distributions. This is not the original k-of-N RPOSST
algorithm and does not inherit its theorem. In an initial Taxi development
matrix with four tasks, three policies, and 50 episodes per task-policy pair,
the two-task subset reproduced the full equal-task policy scores with worst
absolute error 0.00216. The test is useful enough to retain, but broader-domain
calibration and comparisons with uniform and random subsets are still required.

## External benchmark implementation

The development manifest in `experiments/frontier_v2_external_design.py`
contains nine genuinely different domain families from four independently
maintained codebases:

- Gymnasium 1.3.0: FrozenLake, CliffWalking, and Taxi;
- OR-Gym 0.5.0: online knapsack and inventory management;
- Safety-Gymnasium 1.2.0: PointGoal and PointButton; and
- MiniGrid 3.1.0: DynamicObstacles and LavaCrossing.

Each domain has four disjoint development, four calibration, and four declared
confirmation tasks, giving 36 tasks per split. The current manifest hashes are
`6de94c6456eccff522e9f9f359d589d10280f551a9616920f17746652a1c235e`
for development,
`da9faca59d0e1a59e8d98e03d99cdd86b698c5ac618b574492e110d71a2475c2`
for calibration, and
`6538bfd9910eae99be7e692e6d00ba67afbcb1dae68b1eb7ace146fc7aa885b2`
for confirmation. These are development hashes and are not yet registration
locks.

The source audit verifies clean checkouts at Gymnasium commit
`53bf3e9a884783eb72ad3fc8b15780914c97c3e1`, OR-Gym commit
`0b18d16e569e2db70e83f09e867b53bdb4b87298`, Safety-Gymnasium commit
`98231340a4c5b223c8d111fa9597d81836ce09b4`, and MiniGrid commit
`90928729376741a41222a257911343b97103b548`. The complete import-time
dependency stack is explicitly versioned, including dependencies that the
upstream OR-Gym and Safety-Gymnasium packages import eagerly.

Every episode is transformed by a frozen domain rule to a score in [0, 1], so
the paired candidate-minus-fallback difference is always in [-1, 1]. The two
MiniGrid suites operate on fully observable compact images, while both Safety
suites use flattened lidar and proprioceptive observations; four domains
therefore exceed the prespecified 32-coordinate high-dimensional threshold.

A provenance-bound end-to-end rehearsal has now completed for all 36
development and all 36 calibration tasks. Each task ran the complete
three-policy library for one episode and then repeated the run exactly. Each
split therefore contains 108 episode rows. Both passed task-hash,
whole-manifest-hash, clean source-commit, dependency-lock, canonical seed-block,
common-random-number, complete outcome-schema, derived-summary, score-bound,
deterministic-replay, and runtime-ledger checks. The development and calibration
runtime sums were 518.04 and 513.55 seconds, respectively. Every artifact is
also bound to outcome-implementation digest
`5ea81a98337337f57ae77a96ea3d4cb47b603c53748b268d7dc6428e47d08cd7`,
so a policy or adapter code change makes the artifact fail the current audit.
The split manifest hashes remained
`6de94c6456eccff522e9f9f359d589d10280f551a9616920f17746652a1c235e`
and `da9faca59d0e1a59e8d98e03d99cdd86b698c5ac618b574492e110d71a2475c2`.

This is full adapter coverage, not a statistically informative policy
comparison and not evidence that the scripted policy libraries are competitive.
`experiments/frontier_v2_full_rehearsal.py` will declare a split complete only
after all 36 exact task artifacts pass the strict audit and unconditionally
refuses confirmation. Statistically sized development and calibration runs,
score-bound stress tests beyond observed trajectories, and trained
DQN/PPO/safe-RL references remain gates before registration.

The rehearsal also triggered a policy-library repair before any confirmation
lock. The original CliffWalking fallback used shifted dynamics and all three
policies were identical on every state. The repaired fallback is planned under
nominal nonslippery dynamics, while candidates use the observed task dynamics;
the risk-averse candidate additionally pays a frozen near-cliff state penalty.
In 20 paired development/calibration episodes per policy, its mean score effect
was mildly negative on deterministic tasks (-0.0059 to -0.0125) and strongly
positive on slippery tasks (0.5725 to 0.6979). The fast shifted-model candidate
gained 0.6315 to 0.7360 on slippery tasks and tied fallback on deterministic
tasks. FrozenLake now uses the same nominal-versus-shifted distinction: the
hazard-averse candidate tied fallback on deterministic tasks and gained 0.2391
to 0.5563 on slippery calibration tasks and 0.3305 to 0.3960 on slippery
development tasks. These are development diagnostics, not confirmation
effects.

Taxi candidates disagree with fallback on 8--9% of tabular states, but a
20-episode paired diagnostic found only rare trajectory changes and effects
near zero. Taxi is retained as a difficult/null-routing domain; its weakness is
not hidden by tuning an artificial score effect. Proposal freezing may exclude
zero-contrast task-policy pairs using development and calibration data only.

The current machine has six CPU cores, 15.8 GB RAM, and a 4 GB GTX 1650, so
scripted-policy development can run locally, but training or evaluating the
high-dimensional reference policies may require a separately recorded compute
environment.

## Empirical questions

The v2 confirmation should distinguish four claims:

1. **Validity:** Does the empirical null-family false-deployment rate agree
   with the theorem under synthetic and simulator nulls?
2. **Efficiency:** How many paired episodes are saved relative to the fixed
   nine-batch and fixed-sample gates at matched familywise level?
3. **Utility:** What equal-domain improvement remains after the stronger
   deployment margin is enforced?
4. **Breadth:** Does the result remain positive when domains, rather than only
   tasks within domains, are resampled?

The primary estimand remains an equal-domain router-minus-fallback effect, but
the study must also report total pilot cost, acceptance precision and recall
against separate final outcomes, harmful accepted routes, unresolved routes,
and per-domain effects. A route-held-fixed score grid and leave-one-domain-out
analysis are prespecified sensitivity checks.

Synthetic comparisons use deterministic task-specific random streams. The
same trial seed therefore gives every allocation method the same latent stream
for each task even when the methods visit tasks in different orders. Efficiency
contrasts can be analyzed as paired trial-level differences.

## Development gates before preregistration

These gates determine whether the design is mature enough to register. They do
not determine whether final results are published.

- A machine-checked proof appendix matches the implemented e-process formula.
- Unit tests cover thresholds, bounds, alpha weights, terminal decisions, and
  allocation invariance.
- At least 10,000 synthetic null families show no detectable inflation beyond
  Monte Carlo uncertainty.
- Development-only simulations compare power and episode cost across all
  familywise-valid baselines.
- Every external adapter passes seed determinism, common-random-number pairing,
  score-bound, and clean-up tests.
- A compute rehearsal completes without opening any confirmation outcome.
- The complete task family, candidate policies, score bounds, margins, alpha
  weights, allocation rule, maximum budget, seeds, inference, and reporting
  commitment are hashed and publicly registered.

## Initial development diagnostics

These numbers are implementation diagnostics, not confirmation results. With
23 bounded synthetic tasks, familywise alpha 0.05, a maximum of 200
observations per task, and a shared budget of 1,150 observations, the completed
10,000-family global-null gate produced 37 false-deployment families under
uniform allocation: rate 0.0037, Wilson 95% interval [0.00269, 0.00510]. The
resolution allocation produced 54: rate 0.0054, interval [0.00414, 0.00704].
Both interval upper bounds are below 0.01 and far below the 0.05 familywise
target. This checks the implementation under adaptive interleaving; the theorem,
not simulation, supplies the validity claim.

In 300 paired-stream mixed-effect families at the same budget, uniform betting
accepted 11.1% of truly positive tasks and produced mean equal-task expected
improvement 0.0187. Resolution allocation accepted 27.8% and produced 0.0382.
The paired improvement difference was 0.0194 with a normal 95% interval
[0.0173, 0.0215], and the paired positive-task acceptance-rate difference was
0.1667 [0.1518, 0.1816]. Neither betting strategy falsely accepted a null task
in that run.

At the same maximum budget, fixed-sample Hoeffding with either Bonferroni or
Holm accepted 15.2% of positive tasks, the uniform Hoeffding mixture accepted
8.3%, and alpha-spending racing accepted 1.5%. The fixed sign methods accepted
23.6%, but they use a distinct independent-sign null and had a 0.0067 empirical
false-deployment rate in these 300 families. The results establish that the
active betting heuristic can materially improve utility under a binding budget,
while the conservative racing baseline supplies a transparent reference.

The certified betting schedule produced 18 false-deployment families in a
separate 10,000-family global-null run: rate 0.0018, Wilson 95% interval
[0.00114, 0.00284]. The frozen planning gaps were deliberately false in this
null stress test, illustrating that allocation misspecification does not alter
the alpha guarantee. In the 300 mixed-effect families, certified allocation
accepted 38.8% of positive tasks and produced mean equal-task improvement
0.0522. Relative to uniform allocation, the paired acceptance-rate difference
was 0.2762 [0.2638, 0.2886] and the paired improvement difference was 0.0334
[0.0317, 0.0352]. It made no false acceptance in that run.

Those power numbers are promising but do not yet demonstrate the untruncated
sample-complexity theorem. At the 200-observation per-task cap, 22 of the 23
mixed-scenario certified quotas are truncated; only the task with planning gap
0.6 has an untruncated 170-observation quota. The next calibration must compare
the realized cost with the registered bounds in regimes where the global and
per-task caps are nonbinding, as well as under deliberately optimistic planning
gaps.

## Evidential firewall

V1 is complete and remains unchanged. Its pilot and final outcomes may motivate
v2 but cannot be used as v2 confirmation data. V2 development tasks may reuse
environment implementations, but confirmation task parameters and seeds must
be disjoint. All positive, null, and negative v2 outcomes will be reported.
