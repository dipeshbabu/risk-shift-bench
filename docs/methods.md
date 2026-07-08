# Methods

This project studies adaptive risk-sensitive planning under bankroll constraints.

The benchmark evaluates policies that map a Blackjack decision state to `hit` or
`stand`. Policies are scored by the distribution of final bankroll outcomes,
not only by expected value.

The first adaptive policy family uses a state-adaptive CVaR objective:

```text
alpha = f(bankroll_ratio, drawdown, target_gap)
action = argmax CVaR_alpha(P(final_bankroll | state, action))
```

The second adaptive policy family uses a state-adaptive utility objective. It is
mean-seeking in safe states, then increases CVaR, entropic, ruin, drawdown, and
target terms when bankroll pressure or target pressure becomes active.

```text
risk_pressure = f(bankroll_ratio, drawdown)
target_pressure = g(target_gap, rounds_remaining)
action = argmax adaptive_utility(P(final_bankroll | state, action), context)
```

The benchmark also includes a regime-adaptive ensemble. It switches between
objective families using observable task features: bankroll pressure, target
gap, drawdown limit, and card-distribution shift.

The ablation study disables one branch at a time: deck-shift detection, ruin
handling, drawdown handling, target handling, and target-regime gating. It also
compares against single-objective adaptive utility and naive adaptive CVaR.

Static baselines include expected value, fixed CVaR, entropic risk, OCE,
ruin-constrained mean objectives, and target-seeking mean objectives.

## Metrics

- Mean final bankroll.
- 5% CVaR of final bankroll.
- Ruin probability.
- Target-hit probability.
- Mean maximum drawdown.
- Mean rounds played.
