from __future__ import annotations

import pytest

from experiments.frontier_v2_external_design import domain_tasks
from experiments.frontier_v2_tabular_q_learning import (
    SUPPORTED_BASELINES,
    _episode_score,
    _greedy_action,
)


class Row(list):
    def max(self):
        return max(self)


def test_tabular_q_references_cover_cliffwalking_and_taxi() -> None:
    assert set(SUPPORTED_BASELINES) == {
        "gymnasium_cliffwalking",
        "gymnasium_taxi",
    }


def test_greedy_action_uses_reproducible_low_action_tie_break() -> None:
    assert _greedy_action([Row([0.2, 0.5, 0.5])], 0) == 1


def test_episode_score_matches_bounded_external_score_rule() -> None:
    task = domain_tasks("gymnasium_cliffwalking", "calibration")[0]
    score = _episode_score(task, success=True, steps=40, cost=1.0)
    assert score == pytest.approx(1.0 - 0.5 * 40 / 90 - 0.1)
    assert _episode_score(task, success=False, steps=40, cost=2.0) == 0.0
