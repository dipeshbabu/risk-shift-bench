"""Domain runtime helpers for pilot-verified policy routing."""

from __future__ import annotations

from dataclasses import asdict

from experiments.frontier_router_builders import (
    build_blackjack_router,
    build_inventory_router,
    build_portfolio_router,
)
from experiments.frontier_v4_tasks import (
    blackjack_confirmation_v4_tasks,
    portfolio_confirmation_v2_tasks,
)
from experiments.inventory_domain import (
    InventoryPolicy,
    inventory_confirmation_tasks,
    inventory_policy_lookup,
    run_inventory_benchmark,
)
from risk_shift_bench.adaptive_search import summary_score
from risk_shift_bench.benchmark import run_benchmark
from risk_shift_bench.family_selector import family_candidate_lookup
from risk_shift_bench.policies import BenchmarkPolicy
from risk_shift_bench.portfolio_benchmark import (
    PortfolioPolicy,
    portfolio_policy_lookup,
    run_portfolio_benchmark,
)


DOMAINS = ("blackjack_v4", "portfolio_v2", "inventory_v1")


class MappedBlackjackPolicy(BenchmarkPolicy):
    name = "pilot_verified_router"

    def __init__(self, delegates_by_task: dict[str, str], name: str = "pilot_verified_router") -> None:
        self.delegates_by_task = dict(delegates_by_task)
        self.policies = family_candidate_lookup()
        self.name = name

    def selected_policy_name(self, task) -> str:
        return self.delegates_by_task[task.name]

    def action_probabilities(
        self,
        state,
        task,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        return self.policies[self.selected_policy_name(task)].action_probabilities(
            state,
            task=task,
            rounds_remaining=rounds_remaining,
            hand_depth=hand_depth,
            peak_bankroll=peak_bankroll,
        )


class MappedPortfolioPolicy(PortfolioPolicy):
    name = "pilot_verified_router"

    def __init__(self, delegates_by_task: dict[str, str], name: str = "pilot_verified_router") -> None:
        self.delegates_by_task = dict(delegates_by_task)
        self.policies = portfolio_policy_lookup()
        self.name = name

    def selected_policy_name(self, task) -> str:
        return self.delegates_by_task[task.name]

    def allocation(self, state, task) -> float:
        return self.policies[self.selected_policy_name(task)].allocation(state, task)


class MappedInventoryPolicy(InventoryPolicy):
    name = "pilot_verified_router"

    def __init__(self, delegates_by_task: dict[str, str], name: str = "pilot_verified_router") -> None:
        self.delegates_by_task = dict(delegates_by_task)
        self.policies = inventory_policy_lookup()
        self.name = name

    def selected_policy_name(self, task) -> str:
        return self.delegates_by_task[task.name]

    def order_quantity(self, state, task) -> int:
        return self.policies[self.selected_policy_name(task)].order_quantity(state, task)


def domain_tasks(domain: str):
    if domain == "blackjack_v4":
        return blackjack_confirmation_v4_tasks()
    if domain == "portfolio_v2":
        return portfolio_confirmation_v2_tasks()
    if domain == "inventory_v1":
        return inventory_confirmation_tasks()
    raise ValueError(f"unknown domain: {domain}")


def domain_router(domain: str):
    if domain == "blackjack_v4":
        return build_blackjack_router()
    if domain == "portfolio_v2":
        return build_portfolio_router()
    if domain == "inventory_v1":
        return build_inventory_router()
    raise ValueError(f"unknown domain: {domain}")


def domain_policy_lookup(domain: str):
    if domain == "blackjack_v4":
        return family_candidate_lookup()
    if domain == "portfolio_v2":
        return portfolio_policy_lookup()
    if domain == "inventory_v1":
        return inventory_policy_lookup()
    raise ValueError(f"unknown domain: {domain}")


def mapped_policy(domain: str, delegates_by_task: dict[str, str]):
    if domain == "blackjack_v4":
        return MappedBlackjackPolicy(delegates_by_task)
    if domain == "portfolio_v2":
        return MappedPortfolioPolicy(delegates_by_task)
    if domain == "inventory_v1":
        return MappedInventoryPolicy(delegates_by_task)
    raise ValueError(f"unknown domain: {domain}")


def score_summaries(domain: str, summaries, seed: int) -> list[dict]:
    return [
        {
            "domain": domain,
            "task": summary.task,
            "seed": seed,
            "policy": summary.policy,
            "score": summary_score(summary),
            **{
                key: value
                for key, value in asdict(summary).items()
                if key not in {"task", "policy"}
            },
        }
        for summary in summaries
    ]


def run_domain(
    domain: str,
    tasks,
    policies,
    episodes: int,
    seed: int,
    hand_depth: int,
) -> list[dict]:
    if domain == "blackjack_v4":
        _episodes, summaries = run_benchmark(
            tasks=tasks,
            policies=policies,
            episodes=episodes,
            seed=seed,
            hand_depth=hand_depth,
        )
    elif domain == "portfolio_v2":
        _episodes, summaries = run_portfolio_benchmark(
            tasks=tasks,
            policies=policies,
            episodes=episodes,
            seed=seed,
        )
    elif domain == "inventory_v1":
        _episodes, summaries = run_inventory_benchmark(
            tasks=tasks,
            policies=policies,
            episodes=episodes,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown domain: {domain}")
    return score_summaries(domain, summaries, seed)


def proposal_rows(domain: str) -> list[dict]:
    router = domain_router(domain)
    rows = []
    for task in domain_tasks(domain):
        proposal = router.proposal(task)
        prediction = proposal.prediction
        rows.append(
            {
                "domain": domain,
                "task": task.name,
                "fallback_policy": router.params.fallback_policy,
                "candidate_policy": router.candidate_policies[0],
                "proposed_policy": proposal.selected_policy,
                "proposal_active": proposal.promoted,
                "proposal_reason": proposal.reason,
                "predicted_advantage": prediction.predicted_advantage if prediction else "",
                "support_radius": prediction.support_radius if prediction else "",
                "support_radius_limit": router.calibration.support_radius_limit,
            }
        )
    return rows
