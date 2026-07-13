"""Task-family promotions over a conservative signed-regime fallback."""

from __future__ import annotations

from dataclasses import dataclass

from risk_preference_inference.envs import STANDARD_DECK, RiskTask
from risk_preference_inference.policies import BenchmarkPolicy, RegimeAdaptivePolicy, StaticObjectivePolicy
from risk_preference_inference.objectives import EntropicObjective, MeanObjective, OCEObjective
from risk_preference_inference.policy_registry import (
    learned_mixture_policy,
    searched_learned_mixture_policy,
    signed_regime_learned_policy,
    state_action_blend_policy,
    state_adaptive_utility_policy,
)


@dataclass(frozen=True)
class FamilyPromotionParams:
    family_delegates: dict[str, str]
    fallback_policy: str = "signed_regime_learned_ensemble"
    min_delta: float = 2.0
    min_sparse_evidence: int = 2
    require_nonnegative_sparse_evidence: bool = True


def card_mean(card_probs: tuple[tuple[int, float], ...]) -> float:
    return sum(card * prob for card, prob in card_probs)


def regime_span(task: RiskTask) -> float:
    if task.episode_card_regimes is None:
        return 0.0
    standard_mean = card_mean(STANDARD_DECK)
    shifts = [card_mean(card_probs) - standard_mean for card_probs, _prob in task.episode_card_regimes]
    return max(shifts) - min(shifts)


def task_family(task: RiskTask) -> str:
    mean_shift = card_mean(task.card_probs) - card_mean(STANDARD_DECK)
    bankroll_bet_ratio = task.initial_bankroll / max(task.bet, 1.0)
    hidden = task.episode_card_regimes is not None

    if hidden and bankroll_bet_ratio <= 8.0:
        return "hidden_low_bankroll_tail"
    if hidden and task.rounds >= 45 and task.drawdown_limit <= 0.11:
        return "hidden_long_tight"
    if hidden and task.rounds >= 45:
        return "hidden_long_loose"
    if hidden and task.rounds < 45 and regime_span(task) > 2.0:
        return "hidden_short_tail"
    if hidden:
        return "hidden_general"
    if mean_shift <= -2.1 and task.rounds <= 32:
        return "extreme_low_short"
    if mean_shift <= -2.1:
        return "extreme_low_long"
    if mean_shift < -1.0 and task.rounds >= 45 and task.drawdown_limit <= 0.11:
        return "ten_depleted_tight"
    if mean_shift >= 0.35 and bankroll_bet_ratio <= 8.0:
        return "high_shift_near_ruin"
    if mean_shift >= 0.35:
        return "high_shift"
    return "default"


def family_candidate_policies() -> list[BenchmarkPolicy]:
    return [
        StaticObjectivePolicy(MeanObjective(), name="expected_value"),
        StaticObjectivePolicy(EntropicObjective(risk_aversion=0.025), name="fixed_entropic_0.025"),
        StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name="fixed_oce_3"),
        state_adaptive_utility_policy(name="adaptive_utility_default"),
        learned_mixture_policy(name="learned_mixture_default"),
        searched_learned_mixture_policy(),
        RegimeAdaptivePolicy(),
        signed_regime_learned_policy(),
        state_action_blend_policy(),
    ]


def family_candidate_lookup() -> dict[str, BenchmarkPolicy]:
    return {policy.name: policy for policy in family_candidate_policies()}


class FamilyPromotionPolicy(BenchmarkPolicy):
    name = "family_promotion_selector"

    def __init__(
        self,
        params: FamilyPromotionParams,
        policies: dict[str, BenchmarkPolicy] | None = None,
        name: str = "family_promotion_selector",
    ) -> None:
        self.params = params
        self.policies = policies or family_candidate_lookup()
        self.name = name

    def selected_policy_name(self, task: RiskTask) -> str:
        family = task_family(task)
        return self.params.family_delegates.get(family, self.params.fallback_policy)

    def action_probabilities(
        self,
        state,
        task: RiskTask,
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


def learn_family_promotions(
    tasks: list[RiskTask],
    scores_by_task: dict[str, dict[str, float]],
    candidate_policies: list[str] | None = None,
    fallback_policy: str = "signed_regime_learned_ensemble",
    min_delta: float = 2.0,
    min_sparse_evidence: int = 2,
    require_nonnegative_sparse_evidence: bool = True,
) -> FamilyPromotionParams:
    candidates = candidate_policies or [policy.name for policy in family_candidate_policies()]
    tasks_by_family: dict[str, list[RiskTask]] = {}
    for task in tasks:
        if task.name in scores_by_task and fallback_policy in scores_by_task[task.name]:
            tasks_by_family.setdefault(task_family(task), []).append(task)

    family_delegates: dict[str, str] = {}
    for family, family_tasks in sorted(tasks_by_family.items()):
        best_policy = fallback_policy
        best_delta = 0.0
        for policy_name in candidates:
            evidence_tasks = [task for task in family_tasks if policy_name in scores_by_task[task.name]]
            if not evidence_tasks:
                continue
            has_complete_evidence = len(evidence_tasks) == len(family_tasks)
            if not has_complete_evidence and len(evidence_tasks) < min_sparse_evidence:
                continue
            deltas = [
                scores_by_task[task.name][policy_name] - scores_by_task[task.name][fallback_policy]
                for task in evidence_tasks
            ]
            if min(deltas) < 0.0:
                if not has_complete_evidence and require_nonnegative_sparse_evidence:
                    continue
                if has_complete_evidence and len(evidence_tasks) < 3:
                    continue
            mean_delta = sum(deltas) / len(deltas)
            if mean_delta > best_delta:
                best_policy = policy_name
                best_delta = mean_delta
        if best_policy != fallback_policy and best_delta >= min_delta:
            family_delegates[family] = best_policy
    return FamilyPromotionParams(
        family_delegates=family_delegates,
        fallback_policy=fallback_policy,
        min_delta=min_delta,
        min_sparse_evidence=min_sparse_evidence,
        require_nonnegative_sparse_evidence=require_nonnegative_sparse_evidence,
    )
