"""Task-regime switches between strong incumbent policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from risk_preference_inference.adaptive_search import summary_score
from risk_preference_inference.benchmark import run_benchmark
from risk_preference_inference.envs import RiskTask, STANDARD_DECK
from risk_preference_inference.objectives import MeanObjective, OCEObjective
from risk_preference_inference.policies import BenchmarkPolicy
from risk_preference_inference.policies import StaticObjectivePolicy
from risk_preference_inference.policy_registry import learned_mixture_policy, signed_regime_learned_policy


@dataclass(frozen=True)
class IncumbentSwitchParams:
    hidden_long_rounds: int | None = 50
    hidden_long_min_drawdown: float | None = None
    extreme_low_shift: float | None = -2.1
    extreme_low_max_rounds: int | None = None
    low_shift_drawdown: float | None = None
    hidden_tight_drawdown: float | None = None
    low_bankroll_bet_ratio: float | None = None
    high_shift: float | None = None
    high_shift_min_bet_ratio: float | None = None
    oce_hidden_low_bankroll_bet_ratio: float | None = None
    oce_hidden_long_min_drawdown: float | None = None
    oce_hidden_long_max_drawdown: float | None = None
    oce_high_shift_near_ruin_max_drawdown: float | None = None
    expected_extreme_low_max_rounds: int | None = None
    expected_extreme_low_max_drawdown: float | None = None


@dataclass(frozen=True)
class IncumbentSwitchSearchResult:
    params: IncumbentSwitchParams
    validation_score: float
    validation_summaries: list[dict]
    candidate_scores: list[dict]


def card_mean(task: RiskTask) -> float:
    return sum(card * prob for card, prob in task.card_probs)


def standard_card_mean() -> float:
    return sum(card * prob for card, prob in STANDARD_DECK)


class IncumbentSwitchPolicy(BenchmarkPolicy):
    """Select learned mixture or signed-regime by task-level regime features."""

    def __init__(
        self,
        params: IncumbentSwitchParams,
        learned_policy: BenchmarkPolicy | None = None,
        signed_policy: BenchmarkPolicy | None = None,
        oce_policy: BenchmarkPolicy | None = None,
        expected_policy: BenchmarkPolicy | None = None,
        name: str = "validated_incumbent_switch",
    ) -> None:
        self.params = params
        self.learned_policy = learned_policy or learned_mixture_policy(name=f"{name}_learned_mixture")
        self.signed_policy = signed_policy or signed_regime_learned_policy(name=f"{name}_signed_regime")
        self.oce_policy = oce_policy or StaticObjectivePolicy(OCEObjective(shortfall_penalty=3.0), name=f"{name}_fixed_oce_3")
        self.expected_policy = expected_policy or StaticObjectivePolicy(MeanObjective(), name=f"{name}_expected_value")
        self.name = name

    def delegate(self, task: RiskTask) -> BenchmarkPolicy:
        mean_shift = card_mean(task) - standard_card_mean()
        bankroll_bet_ratio = task.initial_bankroll / max(task.bet, 1.0)
        hidden = task.episode_card_regimes is not None
        if (
            self.params.oce_hidden_low_bankroll_bet_ratio is not None
            and hidden
            and bankroll_bet_ratio <= self.params.oce_hidden_low_bankroll_bet_ratio
        ):
            return self.oce_policy
        if (
            self.params.oce_hidden_long_min_drawdown is not None
            and self.params.oce_hidden_long_max_drawdown is not None
            and hidden
            and task.rounds >= 45
            and self.params.oce_hidden_long_min_drawdown <= task.drawdown_limit <= self.params.oce_hidden_long_max_drawdown
        ):
            return self.oce_policy
        if (
            self.params.oce_high_shift_near_ruin_max_drawdown is not None
            and mean_shift >= 1.0
            and bankroll_bet_ratio <= 7.0
            and task.drawdown_limit <= self.params.oce_high_shift_near_ruin_max_drawdown
        ):
            return self.oce_policy
        if (
            self.params.expected_extreme_low_max_rounds is not None
            and self.params.expected_extreme_low_max_drawdown is not None
            and mean_shift <= -2.1
            and task.rounds <= self.params.expected_extreme_low_max_rounds
            and task.drawdown_limit <= self.params.expected_extreme_low_max_drawdown
        ):
            return self.expected_policy
        hidden_long_allowed = (
            self.params.hidden_long_min_drawdown is None or task.drawdown_limit >= self.params.hidden_long_min_drawdown
        )
        if (
            self.params.hidden_long_rounds is not None
            and hidden
            and task.rounds >= self.params.hidden_long_rounds
            and hidden_long_allowed
        ):
            return self.signed_policy
        extreme_low_allowed = self.params.extreme_low_max_rounds is None or task.rounds <= self.params.extreme_low_max_rounds
        if self.params.extreme_low_shift is not None and mean_shift <= self.params.extreme_low_shift and extreme_low_allowed:
            return self.signed_policy
        if self.params.low_shift_drawdown is not None and mean_shift <= -1.6 and task.drawdown_limit <= self.params.low_shift_drawdown:
            return self.signed_policy
        if self.params.hidden_tight_drawdown is not None and hidden and task.drawdown_limit <= self.params.hidden_tight_drawdown:
            return self.signed_policy
        if self.params.low_bankroll_bet_ratio is not None and bankroll_bet_ratio <= self.params.low_bankroll_bet_ratio:
            return self.signed_policy
        high_shift_allowed = (
            self.params.high_shift_min_bet_ratio is None or bankroll_bet_ratio >= self.params.high_shift_min_bet_ratio
        )
        if self.params.high_shift is not None and mean_shift >= self.params.high_shift and high_shift_allowed:
            return self.signed_policy
        return self.learned_policy

    def use_signed_policy(self, task: RiskTask) -> bool:
        return self.delegate(task) is self.signed_policy

    def selected_policy_name(self, task: RiskTask) -> str:
        return self.delegate(task).name

    def action_probabilities(
        self,
        state,
        task: RiskTask,
        rounds_remaining: int,
        hand_depth: int = 4,
        peak_bankroll: float | None = None,
    ) -> dict[str, float]:
        delegate = self.delegate(task)
        return delegate.action_probabilities(
            state,
            task=task,
            rounds_remaining=rounds_remaining,
            hand_depth=hand_depth,
            peak_bankroll=peak_bankroll,
        )


def incumbent_switch_policy(params: IncumbentSwitchParams, name: str = "validated_incumbent_switch") -> IncumbentSwitchPolicy:
    return IncumbentSwitchPolicy(params=params, name=name)


def incumbent_switch_candidates(smoke: bool = False) -> list[IncumbentSwitchParams]:
    candidates = [
        IncumbentSwitchParams(hidden_long_rounds=None, extreme_low_shift=None),
        IncumbentSwitchParams(hidden_long_rounds=50, extreme_low_shift=-2.1),
        IncumbentSwitchParams(hidden_long_rounds=45, extreme_low_shift=-2.1),
        IncumbentSwitchParams(hidden_long_rounds=50, extreme_low_shift=None),
        IncumbentSwitchParams(hidden_long_rounds=None, extreme_low_shift=-2.1),
        IncumbentSwitchParams(hidden_long_rounds=45, extreme_low_shift=-1.6),
        IncumbentSwitchParams(
            hidden_long_rounds=45,
            hidden_long_min_drawdown=0.12,
            extreme_low_shift=-2.1,
            extreme_low_max_rounds=35,
            low_shift_drawdown=0.12,
        ),
        IncumbentSwitchParams(
            hidden_long_rounds=45,
            hidden_long_min_drawdown=0.12,
            extreme_low_shift=-2.1,
            extreme_low_max_rounds=32,
            low_shift_drawdown=0.12,
        ),
        IncumbentSwitchParams(
            hidden_long_rounds=50,
            hidden_long_min_drawdown=0.12,
            extreme_low_shift=-2.1,
            extreme_low_max_rounds=35,
            low_shift_drawdown=0.12,
        ),
        IncumbentSwitchParams(hidden_long_rounds=50, extreme_low_shift=-2.1, hidden_tight_drawdown=0.12),
        IncumbentSwitchParams(hidden_long_rounds=45, extreme_low_shift=-2.1, high_shift=1.0, high_shift_min_bet_ratio=10.0),
    ]
    return candidates[:3] if smoke else candidates


def evaluate_switch_policy(
    tasks: list[RiskTask],
    policy: IncumbentSwitchPolicy,
    seeds: list[int],
    episodes: int,
    hand_depth: int,
) -> tuple[float, list[dict]]:
    rows = []
    scores = []
    for seed in seeds:
        _episodes, summaries = run_benchmark(tasks=tasks, policies=[policy], episodes=episodes, seed=seed, hand_depth=hand_depth)
        for summary in summaries:
            row = asdict(summary)
            row["seed"] = seed
            row["score"] = summary_score(summary)
            row["selected_policy"] = policy.selected_policy_name(next(task for task in tasks if task.name == summary.task))
            rows.append(row)
            scores.append(row["score"])
    return sum(scores) / len(scores) if scores else float("-inf"), rows


def search_incumbent_switch(
    validation_tasks: list[RiskTask],
    seeds: list[int],
    episodes: int,
    hand_depth: int,
    smoke: bool = False,
) -> IncumbentSwitchSearchResult:
    best: IncumbentSwitchSearchResult | None = None
    candidate_scores = []
    for index, params in enumerate(incumbent_switch_candidates(smoke=smoke)):
        policy = incumbent_switch_policy(params, name=f"incumbent_switch_candidate_{index}")
        score, rows = evaluate_switch_policy(
            tasks=validation_tasks,
            policy=policy,
            seeds=seeds,
            episodes=episodes,
            hand_depth=hand_depth,
        )
        candidate_row = {"candidate": index, "validation_score": score, **asdict(params)}
        candidate_scores.append(candidate_row)
        result = IncumbentSwitchSearchResult(
            params=params,
            validation_score=score,
            validation_summaries=rows,
            candidate_scores=list(candidate_scores),
        )
        if best is None or result.validation_score > best.validation_score:
            best = result
    if best is None:
        raise RuntimeError("no incumbent-switch candidates were evaluated")
    return IncumbentSwitchSearchResult(
        params=best.params,
        validation_score=best.validation_score,
        validation_summaries=best.validation_summaries,
        candidate_scores=candidate_scores,
    )
