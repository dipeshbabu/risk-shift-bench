from __future__ import annotations

from experiments.frontier_v2_baseline_design import COMPETITIVE_BASELINES
from experiments.frontier_v2_readiness import LEARNED_BASELINES


def test_readiness_gate_covers_every_learned_baseline() -> None:
    expected = tuple(
        baseline
        for baseline in COMPETITIVE_BASELINES
        if baseline.kind == "learned_policy"
    )
    assert LEARNED_BASELINES == expected
    assert len(LEARNED_BASELINES) == 12
    assert len({baseline.identifier for baseline in LEARNED_BASELINES}) == 12
