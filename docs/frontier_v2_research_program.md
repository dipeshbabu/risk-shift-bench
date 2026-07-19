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

For proposal (i\), let (X_{i,t}\in[a_i,b_i]\) be the paired
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
\(i\) is deployed when its mixture e-process reaches (1/\alpha_i\), where the
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
allocation. Ville's inequality bounds the probability that task (i\) ever
crosses (1/\alpha_i\) by \(\alpha_i\). A union bound completes the result. No
independence assumption across tasks is needed; the conditional validity of
the next within-task observation is the substantive assumption.

On the same event, if fallback deployment is defined to have difference zero,
the expected score improvement of the deployed router on the fixed task family
obeys

\[
\sum_i w_i R_i\mu_i
>\sum_i w_i R_i\delta_i\geq0,
\]

where (R_i\) records candidate deployment. This is a familywise lower bound
on expected improvement over fallback, not an oracle-regret guarantee and not
a guarantee about every finite final sample.

The reference implementation is
`experiments/anytime_familywise_router.py`. It contains two e-processes. A
finite Hoeffding mixture is the conservative validity baseline. The primary
development method is a mixture of bounded betting processes. After mapping
the observation to (Y_{i,t}\in[0,1]\) with null mean (q_i\), each component
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

The frontier method should add a principled allocation rule with a sample
complexity statement. The target is a successive-elimination or track-and-stop
style rule whose total paired-sample requirement scales with task difficulty,
while retaining the same anytime familywise guarantee. Efficiency is evaluated
at identical candidate-plus-fallback episode budgets.

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

## External benchmark target

The final manifest should contain at least eight genuinely different domain
families from at least four independently maintained codebases, with at least
two domains using high-dimensional observations. Candidate pools under
feasibility review include:

- Gymnasium: FrozenLake, CliffWalking, Taxi, and one classic-control domain;
- OR-Gym: online knapsack, inventory management, vehicle routing, and bin
  packing;
- Safety-Gymnasium: PointGoal, PointButton, PointPush, and CarGoal; and
- MiniGrid or an equivalent visual-control suite for high-dimensional transfer.

This is a feasibility pool, not a confirmation manifest. A domain enters the
registered study only if its upstream commit, deterministic adapter contract,
score bounds, candidate library, runtime, and task disjointness are validated
before confirmation. The current machine has six CPU cores, 15.8 GB RAM, and a
4 GB GTX 1650, so development can run locally but high-dimensional final
evaluation may require a separately recorded compute environment.

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
observations per task, and a shared budget of 1,150 observations, 1,000
global-null families produced a false-acceptance rate of 0.001 under both
uniform and resolution allocation. The Wilson 95% interval was
[0.0002, 0.0056]. This is consistent with the
registered target being conservative, but it is not a substitute for the
proof or the planned 10,000-family calibration.

In 300 paired-stream mixed-effect families at the same budget, the betting
process with uniform allocation accepted 12.8% of truly positive tasks and
produced mean equal-task expected improvement 0.0211. Resolution allocation
accepted 28.3% and produced 0.0386. Neither strategy falsely accepted a null task in that
run. The improvement shows that adaptive allocation can matter under a binding
global budget. The next development stage must compare this heuristic with a
principled allocation rule and report uncertainty over efficiency differences.

## Evidential firewall

V1 is complete and remains unchanged. Its pilot and final outcomes may motivate
v2 but cannot be used as v2 confirmation data. V2 development tasks may reuse
environment implementations, but confirmation task parameters and seeds must
be disjoint. All positive, null, and negative v2 outcomes will be reported.
