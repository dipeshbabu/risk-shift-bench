from __future__ import annotations

import pytest

from experiments.frontier_v2_tabular_policy_audit import compare_action_tables


def test_compare_action_tables_counts_fallback_disagreements() -> None:
    result = compare_action_tables(
        {
            "fallback": {0: 0, 1: 1, 2: 0},
            "candidate_a": {0: 0, 1: 0, 2: 0},
            "candidate_b": {0: 1, 1: 0, 2: 1},
        },
        fallback_policy="fallback",
    )
    assert result["candidate_contrasts"]["candidate_a"][
        "action_disagreement_count"
    ] == 1
    assert result["candidate_contrasts"]["candidate_b"][
        "action_disagreement_fraction"
    ] == 1.0
    assert result["any_candidate_differs_from_fallback"] is True


def test_compare_action_tables_requires_rectangular_state_space() -> None:
    with pytest.raises(ValueError, match="same state"):
        compare_action_tables(
            {"fallback": {0: 0}, "candidate": {0: 0, 1: 1}},
            fallback_policy="fallback",
        )
