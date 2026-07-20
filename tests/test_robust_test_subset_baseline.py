from __future__ import annotations

import pytest

from experiments.robust_test_subset_baseline import (
    _project_simplex,
    select_robust_test_subset,
)


RESULT_MATRIX = {
    "task_a": {"policy_x": 0.0, "policy_y": 1.0},
    "task_b": {"policy_x": 0.0, "policy_y": 1.0},
    "task_c": {"policy_x": 1.0, "policy_y": 0.0},
}


def test_simplex_projection_is_nonnegative_and_normalized() -> None:
    projected = _project_simplex([2.0, -1.0, 0.5])
    assert sum(projected) == pytest.approx(1.0)
    assert min(projected) >= 0.0


def test_robust_subset_is_deterministic_and_nearly_recovers_full_scores() -> None:
    first = select_robust_test_subset(
        RESULT_MATRIX, subset_size=2, optimization_iterations=4_000
    )
    second = select_robust_test_subset(
        RESULT_MATRIX, subset_size=2, optimization_iterations=4_000
    )
    assert first == second
    assert set(first.selected_tasks) in (
        {"task_a", "task_c"},
        {"task_b", "task_c"},
    )
    assert first.worst_absolute_error < 0.02
    assert sum(weight for _task, weight in first.task_weights) == pytest.approx(1.0)


def test_larger_subset_does_not_increase_optimized_error() -> None:
    one = select_robust_test_subset(
        RESULT_MATRIX, subset_size=1, optimization_iterations=2_000
    )
    two = select_robust_test_subset(
        RESULT_MATRIX, subset_size=2, optimization_iterations=2_000
    )
    assert two.worst_absolute_error <= one.worst_absolute_error


def test_multiple_target_distributions_are_supported() -> None:
    targets = {
        "uniform": {"task_a": 1.0, "task_b": 1.0, "task_c": 1.0},
        "shifted": {"task_a": 0.1, "task_b": 0.1, "task_c": 0.8},
    }
    result = select_robust_test_subset(
        RESULT_MATRIX,
        subset_size=2,
        target_distributions=targets,
        optimization_iterations=2_000,
    )
    assert {distribution for distribution, _policy, _error in result.errors} == {
        "uniform",
        "shifted",
    }


def test_invalid_matrix_or_subset_is_rejected() -> None:
    with pytest.raises(ValueError, match="rectangular"):
        select_robust_test_subset(
            {"a": {"p": 1.0}, "b": {"q": 1.0}}, subset_size=1
        )
    with pytest.raises(ValueError, match="subset_size"):
        select_robust_test_subset(RESULT_MATRIX, subset_size=0)
