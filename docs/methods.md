# Methods

This project studies adaptive risk-sensitive planning under bankroll constraints.

The benchmark evaluates policies that map a Blackjack decision state to `hit` or
`stand`. Policies are scored by the distribution of final bankroll outcomes,
not only by expected value.

The main policy family uses a state-adaptive CVaR objective:

```text
alpha = f(bankroll_ratio, drawdown, target_gap)
action = argmax CVaR_alpha(P(final_bankroll | state, action))
```

Static baselines include expected value, fixed CVaR, entropic risk, OCE,
ruin-constrained mean objectives, and target-seeking mean objectives.

## Metrics

- Mean final bankroll.
- 5% CVaR of final bankroll.
- Ruin probability.
- Target-hit probability.
- Mean maximum drawdown.
- Mean rounds played.

