"""Domain adapters for conformal regret-controlled routing."""

from __future__ import annotations

from risk_shift_bench.family_selector import family_candidate_lookup
from risk_shift_bench.policies import BenchmarkPolicy
from risk_shift_bench.portfolio_benchmark import PortfolioPolicy, portfolio_policy_lookup

from experiments.conformal_router import ConformalAdvantageRouter


class BlackjackConformalRouterPolicy(BenchmarkPolicy):
    name = "conformal_regret_router"

    def __init__(
        self,
        router: ConformalAdvantageRouter,
        policies: dict[str, BenchmarkPolicy] | None = None,
        name: str = "conformal_regret_router",
    ) -> None:
        self.router = router
        self.policies = policies or family_candidate_lookup()
        self.name = name

    def selected_policy_name(self, task) -> str:
        return self.router.decision(task).selected_policy

    def action_probabilities(
        self,
        state,
        task,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        delegate = self.policies[self.selected_policy_name(task)]
        return delegate.action_probabilities(
            state,
            task=task,
            rounds_remaining=rounds_remaining,
            hand_depth=hand_depth,
            peak_bankroll=peak_bankroll,
        )


class PortfolioConformalRouterPolicy(PortfolioPolicy):
    name = "portfolio_conformal_regret_router"

    def __init__(
        self,
        router: ConformalAdvantageRouter,
        policies: dict[str, PortfolioPolicy] | None = None,
        name: str = "portfolio_conformal_regret_router",
    ) -> None:
        self.router = router
        self.policies = policies or portfolio_policy_lookup()
        self.name = name

    def selected_policy_name(self, task) -> str:
        return self.router.decision(task).selected_policy

    def allocation(self, state, task) -> float:
        return self.policies[self.selected_policy_name(task)].allocation(state, task)
