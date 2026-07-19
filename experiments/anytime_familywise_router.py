"""Anytime-valid familywise policy routing for RiskShiftBench v2.

This module is development-only.  It does not read or modify any v1 pilot,
gate, final, or combined artifact.

The acceptance process tests a task-level null of the form

    H_i: E[X_{i,t} | F_{t-1}] <= effect_margin,

where the paired candidate-minus-fallback observations are known to lie in a
fixed interval.  A finite mixture of Hoeffding test supermartingales gives an
e-process for each task.  Prespecified task-level alpha weights and a union
bound then control the probability of accepting any null proposal, even when
tasks are sampled and stopped adaptively.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite, log, log1p


DEFAULT_ETA_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_BETTING_FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)


class RouteDecision(str, Enum):
    UNDECIDED = "undecided"
    ACCEPT_CANDIDATE = "accept_candidate"
    REJECT_TO_FALLBACK = "reject_to_fallback"
    BUDGET_EXHAUSTED = "budget_exhausted"


def _logsumexp(values: list[float]) -> float:
    if not values:
        raise ValueError("log-sum-exp requires at least one value")
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


def _mixture_log_weights(component_count: int) -> tuple[float, ...]:
    """Put more prior mass on conservative betting fractions."""

    if component_count <= 0:
        raise ValueError("component_count must be positive")
    raw = [1.0 / (index * (index + 1.0)) for index in range(1, component_count + 1)]
    total = sum(raw)
    return tuple(log(weight / total) for weight in raw)


@dataclass(frozen=True)
class AnytimeFamilywisePlan:
    """Frozen statistical and budget choices for a proposal family."""

    task_names: tuple[str, ...]
    familywise_alpha: float = 0.05
    futility_familywise_alpha: float = 0.05
    effect_margin: float = 0.0
    observation_lower: float = -1.0
    observation_upper: float = 1.0
    minimum_observations: int = 1
    maximum_observations_per_task: int = 1_000
    task_weights: tuple[tuple[str, float], ...] = ()
    e_process_method: str = "betting_mixture"
    eta_grid: tuple[float, ...] = DEFAULT_ETA_GRID
    betting_fraction_grid: tuple[float, ...] = DEFAULT_BETTING_FRACTIONS

    def __post_init__(self) -> None:
        if not self.task_names:
            raise ValueError("at least one proposal task is required")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("proposal task names must be unique")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must lie strictly between zero and one")
        if not 0.0 < self.futility_familywise_alpha < 1.0:
            raise ValueError(
                "futility_familywise_alpha must lie strictly between zero and one"
            )
        if not self.observation_lower < self.observation_upper:
            raise ValueError("observation bounds must be ordered")
        if not self.observation_lower < self.effect_margin < self.observation_upper:
            raise ValueError("effect_margin must lie strictly inside the observation bounds")
        if self.minimum_observations <= 0:
            raise ValueError("minimum_observations must be positive")
        if self.maximum_observations_per_task < self.minimum_observations:
            raise ValueError(
                "maximum_observations_per_task cannot be below minimum_observations"
            )
        if not self.eta_grid or any(eta <= 0.0 for eta in self.eta_grid):
            raise ValueError("eta_grid entries must all be positive")
        if len(set(self.eta_grid)) != len(self.eta_grid):
            raise ValueError("eta_grid entries must be unique")
        if self.e_process_method not in {"hoeffding_mixture", "betting_mixture"}:
            raise ValueError(
                "e_process_method must be 'hoeffding_mixture' or 'betting_mixture'"
            )
        if not self.betting_fraction_grid or any(
            not 0.0 < fraction < 1.0 for fraction in self.betting_fraction_grid
        ):
            raise ValueError("betting fractions must lie strictly between zero and one")
        if len(set(self.betting_fraction_grid)) != len(self.betting_fraction_grid):
            raise ValueError("betting fraction entries must be unique")

        if self.task_weights:
            weight_names = [name for name, _weight in self.task_weights]
            if len(set(weight_names)) != len(weight_names):
                raise ValueError("task weight names must be unique")
            if set(weight_names) != set(self.task_names):
                raise ValueError("task weights must cover exactly the proposal family")
            if any(not isfinite(weight) or weight <= 0.0 for _name, weight in self.task_weights):
                raise ValueError("task weights must be finite and positive")

    @property
    def observation_width(self) -> float:
        return self.observation_upper - self.observation_lower

    def _normalized_weights(self) -> dict[str, float]:
        if not self.task_weights:
            uniform = 1.0 / len(self.task_names)
            return {task: uniform for task in self.task_names}
        total = sum(weight for _task, weight in self.task_weights)
        return {task: weight / total for task, weight in self.task_weights}

    def acceptance_alpha(self, task: str) -> float:
        try:
            weight = self._normalized_weights()[task]
        except KeyError as error:
            raise KeyError(f"unknown proposal task: {task}") from error
        return self.familywise_alpha * weight

    def futility_alpha(self, task: str) -> float:
        try:
            weight = self._normalized_weights()[task]
        except KeyError as error:
            raise KeyError(f"unknown proposal task: {task}") from error
        return self.futility_familywise_alpha * weight


@dataclass(frozen=True)
class TaskEvidence:
    task: str
    decision: RouteDecision
    observations: int
    mean_difference: float
    acceptance_log_e: float
    acceptance_log_threshold: float
    futility_log_e: float
    futility_log_threshold: float


class AnytimeFamilywiseRouter:
    """Maintain task-level e-processes and route decisions."""

    def __init__(self, plan: AnytimeFamilywisePlan):
        self.plan = plan
        self._observations = {task: [] for task in plan.task_names}
        self._observation_sums = {task: 0.0 for task in plan.task_names}
        self._acceptance_log_e = {task: 0.0 for task in plan.task_names}
        self._futility_log_e = {task: 0.0 for task in plan.task_names}
        self._acceptance_log_threshold = {
            task: log(1.0 / plan.acceptance_alpha(task)) for task in plan.task_names
        }
        self._futility_log_threshold = {
            task: log(1.0 / plan.futility_alpha(task)) for task in plan.task_names
        }
        self._decisions = {
            task: RouteDecision.UNDECIDED for task in plan.task_names
        }
        component_count = (
            len(plan.eta_grid)
            if plan.e_process_method == "hoeffding_mixture"
            else len(plan.betting_fraction_grid)
        )
        self._log_weights = _mixture_log_weights(component_count)
        self._acceptance_component_logs = {
            task: list(self._log_weights) for task in plan.task_names
        }
        self._futility_component_logs = {
            task: list(self._log_weights) for task in plan.task_names
        }

    def _validate_task(self, task: str) -> None:
        if task not in self._observations:
            raise KeyError(f"unknown proposal task: {task}")

    def _validate_observation(self, value: float) -> float:
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError("paired score differences must be finite")
        if not self.plan.observation_lower <= numeric <= self.plan.observation_upper:
            raise ValueError(
                "paired score difference lies outside the frozen observation bounds"
            )
        return numeric

    def _update_e_value(self, task: str, value: float, direction: float) -> float:
        components = (
            self._acceptance_component_logs[task]
            if direction > 0.0
            else self._futility_component_logs[task]
        )
        if self.plan.e_process_method == "hoeffding_mixture":
            centered = direction * (value - self.plan.effect_margin)
            width = self.plan.observation_width
            for index, eta in enumerate(self.plan.eta_grid):
                components[index] += eta * centered / width - eta * eta / 8.0
        else:
            normalized = (
                value - self.plan.observation_lower
            ) / self.plan.observation_width
            null_mean = (
                self.plan.effect_margin - self.plan.observation_lower
            ) / self.plan.observation_width
            if direction < 0.0:
                normalized = 1.0 - normalized
                null_mean = 1.0 - null_mean
            centered = normalized - null_mean
            for index, fraction in enumerate(self.plan.betting_fraction_grid):
                betting_fraction = fraction / null_mean
                components[index] += log1p(betting_fraction * centered)
        return _logsumexp(components)

    def evidence(self, task: str) -> TaskEvidence:
        self._validate_task(task)
        observations = self._observations[task]
        return TaskEvidence(
            task=task,
            decision=self._decisions[task],
            observations=len(observations),
            mean_difference=(
                self._observation_sums[task] / len(observations)
                if observations
                else 0.0
            ),
            acceptance_log_e=self._acceptance_log_e[task],
            acceptance_log_threshold=self._acceptance_log_threshold[task],
            futility_log_e=self._futility_log_e[task],
            futility_log_threshold=self._futility_log_threshold[task],
        )

    def update(self, task: str, paired_score_difference: float) -> TaskEvidence:
        self._validate_task(task)
        if self._decisions[task] is not RouteDecision.UNDECIDED:
            raise RuntimeError(f"task {task} already has a terminal route decision")
        value = self._validate_observation(paired_score_difference)
        self._observations[task].append(value)
        self._observation_sums[task] += value
        self._acceptance_log_e[task] = self._update_e_value(
            task, value, direction=1.0
        )
        self._futility_log_e[task] = self._update_e_value(
            task, value, direction=-1.0
        )
        observation_count = len(self._observations[task])

        if observation_count >= self.plan.minimum_observations:
            accept = (
                self._acceptance_log_e[task]
                >= self._acceptance_log_threshold[task]
            )
            reject = (
                self._futility_log_e[task] >= self._futility_log_threshold[task]
            )
            if accept and reject:
                raise RuntimeError("acceptance and futility processes crossed together")
            if accept:
                self._decisions[task] = RouteDecision.ACCEPT_CANDIDATE
            elif reject:
                self._decisions[task] = RouteDecision.REJECT_TO_FALLBACK

        if (
            self._decisions[task] is RouteDecision.UNDECIDED
            and observation_count >= self.plan.maximum_observations_per_task
        ):
            self._decisions[task] = RouteDecision.BUDGET_EXHAUSTED
        return self.evidence(task)

    def next_task(
        self,
        strategy: str = "resolution",
        *,
        forced_initial_observations: int = 2,
        information_floor: float = 1e-6,
    ) -> str | None:
        """Choose the next unresolved task without affecting test validity.

        ``uniform`` balances sample counts.  ``resolution`` first forces a
        small number of samples per task, then chooses the task with the
        smallest plug-in estimate of additional samples needed for either
        terminal decision.  The allocation rule may affect efficiency, but
        the e-process thresholds remain valid under predictable interleaving.
        """

        if strategy not in {"uniform", "resolution"}:
            raise ValueError("strategy must be 'uniform' or 'resolution'")
        if forced_initial_observations <= 0:
            raise ValueError("forced_initial_observations must be positive")
        if information_floor <= 0.0:
            raise ValueError("information_floor must be positive")

        unresolved = [
            task
            for task in self.plan.task_names
            if self._decisions[task] is RouteDecision.UNDECIDED
        ]
        if not unresolved:
            return None

        if strategy == "uniform":
            return min(unresolved, key=lambda task: (len(self._observations[task]), task))

        forced = [
            task
            for task in unresolved
            if len(self._observations[task]) < forced_initial_observations
        ]
        if forced:
            return min(forced, key=lambda task: (len(self._observations[task]), task))

        width_squared = self.plan.observation_width**2

        def predicted_remaining(task: str) -> tuple[float, int, str]:
            observations = len(self._observations[task])
            mean_difference = self._observation_sums[task] / observations
            gap = abs(mean_difference - self.plan.effect_margin)
            information_rate = max(2.0 * gap * gap / width_squared, information_floor)
            if mean_difference >= self.plan.effect_margin:
                shortfall = max(
                    self._acceptance_log_threshold[task]
                    - self._acceptance_log_e[task],
                    0.0,
                )
            else:
                shortfall = max(
                    self._futility_log_threshold[task] - self._futility_log_e[task],
                    0.0,
                )
            return (shortfall / information_rate, observations, task)

        return min(unresolved, key=predicted_remaining)

    def decisions(self) -> dict[str, TaskEvidence]:
        return {task: self.evidence(task) for task in self.plan.task_names}

    def accepted_tasks(self) -> tuple[str, ...]:
        return tuple(
            task
            for task in self.plan.task_names
            if self._decisions[task] is RouteDecision.ACCEPT_CANDIDATE
        )

    def total_observations(self) -> int:
        return sum(len(values) for values in self._observations.values())
