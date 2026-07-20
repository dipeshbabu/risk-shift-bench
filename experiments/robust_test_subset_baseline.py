"""RPOSST-inspired minimax task-subset baseline for policy evaluation.

The implementation selects a small task set and simplex weights to minimize
the worst absolute policy-score approximation error over frozen target task
distributions. It is a deterministic comparison baseline, not an implementation
of RPOSST's k-of-N algorithm and does not claim its robustness theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class RobustSubsetResult:
    selected_tasks: tuple[str, ...]
    task_weights: tuple[tuple[str, float], ...]
    worst_absolute_error: float
    errors: tuple[tuple[str, str, float], ...]
    optimization_iterations: int


def _project_simplex(values: list[float]) -> list[float]:
    """Euclidean projection onto the probability simplex."""

    if not values:
        raise ValueError("simplex projection requires at least one value")
    ordered = sorted(values, reverse=True)
    cumulative = 0.0
    threshold_index = 0
    for index, value in enumerate(ordered, start=1):
        cumulative += value
        threshold = (cumulative - 1.0) / index
        if value - threshold > 0.0:
            threshold_index = index
    threshold = (sum(ordered[:threshold_index]) - 1.0) / threshold_index
    projected = [max(value - threshold, 0.0) for value in values]
    total = sum(projected)
    return [value / total for value in projected]


def _validate_result_matrix(
    result_matrix: dict[str, dict[str, float]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not result_matrix:
        raise ValueError("result_matrix must contain tasks")
    tasks = tuple(sorted(result_matrix))
    first_policies = set(result_matrix[tasks[0]])
    if not first_policies:
        raise ValueError("result_matrix must contain policies")
    for task in tasks:
        if set(result_matrix[task]) != first_policies:
            raise ValueError("result_matrix must be rectangular")
        if any(not isfinite(float(value)) for value in result_matrix[task].values()):
            raise ValueError("result_matrix values must be finite")
    return tasks, tuple(sorted(first_policies))


def _normalized_target_distributions(
    tasks: tuple[str, ...],
    target_distributions: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    if target_distributions is None:
        uniform = 1.0 / len(tasks)
        return {"equal_task": {task: uniform for task in tasks}}
    if not target_distributions:
        raise ValueError("target_distributions cannot be empty")
    normalized = {}
    for name, weights in sorted(target_distributions.items()):
        if set(weights) != set(tasks):
            raise ValueError("every target distribution must cover every task")
        if any(not isfinite(float(weight)) or weight < 0.0 for weight in weights.values()):
            raise ValueError("target weights must be finite and nonnegative")
        total = sum(float(weight) for weight in weights.values())
        if total <= 0.0:
            raise ValueError("target distribution must have positive mass")
        normalized[name] = {task: float(weights[task]) / total for task in tasks}
    return normalized


def _targets(
    result_matrix: dict[str, dict[str, float]],
    policies: tuple[str, ...],
    distributions: dict[str, dict[str, float]],
) -> dict[tuple[str, str], float]:
    return {
        (distribution, policy): sum(
            weights[task] * float(result_matrix[task][policy])
            for task in result_matrix
        )
        for distribution, weights in distributions.items()
        for policy in policies
    }


def _errors(
    result_matrix: dict[str, dict[str, float]],
    selected: tuple[str, ...],
    weights: list[float],
    targets: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    return {
        key: sum(
            weight * float(result_matrix[task][key[1]])
            for task, weight in zip(selected, weights, strict=True)
        )
        - target
        for key, target in targets.items()
    }


def _optimize_subset_weights(
    result_matrix: dict[str, dict[str, float]],
    selected: tuple[str, ...],
    targets: dict[tuple[str, str], float],
    *,
    iterations: int,
) -> tuple[list[float], dict[tuple[str, str], float]]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    weights = [1.0 / len(selected)] * len(selected)
    best_weights = list(weights)
    best_errors = _errors(result_matrix, selected, weights, targets)
    best_objective = max(abs(error) for error in best_errors.values())
    for iteration in range(1, iterations + 1):
        errors = _errors(result_matrix, selected, weights, targets)
        worst_key, worst_error = max(
            errors.items(),
            key=lambda item: (abs(item[1]), item[0]),
        )
        policy = worst_key[1]
        sign = 1.0 if worst_error >= 0.0 else -1.0
        step_size = 0.5 / sqrt(iteration)
        candidate = [
            weight - step_size * sign * float(result_matrix[task][policy])
            for task, weight in zip(selected, weights, strict=True)
        ]
        weights = _project_simplex(candidate)
        candidate_errors = _errors(result_matrix, selected, weights, targets)
        objective = max(abs(error) for error in candidate_errors.values())
        if objective < best_objective:
            best_objective = objective
            best_weights = list(weights)
            best_errors = candidate_errors
    return best_weights, best_errors


def select_robust_test_subset(
    result_matrix: dict[str, dict[str, float]],
    *,
    subset_size: int,
    target_distributions: dict[str, dict[str, float]] | None = None,
    optimization_iterations: int = 2_000,
) -> RobustSubsetResult:
    tasks, policies = _validate_result_matrix(result_matrix)
    if not 1 <= subset_size <= len(tasks):
        raise ValueError("subset_size must lie between one and the task count")
    distributions = _normalized_target_distributions(tasks, target_distributions)
    targets = _targets(result_matrix, policies, distributions)

    selected: tuple[str, ...] = ()
    remaining = set(tasks)
    for _selection_index in range(subset_size):
        candidates = []
        for task in sorted(remaining):
            proposed = tuple(sorted((*selected, task)))
            weights, errors = _optimize_subset_weights(
                result_matrix,
                proposed,
                targets,
                iterations=max(100, optimization_iterations // 4),
            )
            objective = max(abs(error) for error in errors.values())
            candidates.append((objective, task, proposed, weights))
        _objective, chosen, selected, _weights = min(candidates)
        remaining.remove(chosen)

    weights, errors = _optimize_subset_weights(
        result_matrix,
        selected,
        targets,
        iterations=optimization_iterations,
    )
    return RobustSubsetResult(
        selected_tasks=selected,
        task_weights=tuple(zip(selected, weights, strict=True)),
        worst_absolute_error=max(abs(error) for error in errors.values()),
        errors=tuple(
            (distribution, policy, error)
            for (distribution, policy), error in sorted(errors.items())
        ),
        optimization_iterations=optimization_iterations,
    )
