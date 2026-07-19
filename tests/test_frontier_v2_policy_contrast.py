from __future__ import annotations

import pytest

from experiments.frontier_v2_policy_contrast import paired_policy_contrast


def row(episode: int, seed: int, score: float, *, steps: int = 10) -> dict:
    return {
        "episode": episode,
        "seed": seed,
        "score": score,
        "raw_utility": score,
        "raw_return": score,
        "cost": 0.0,
        "failure": False,
        "steps": steps,
        "successes": 1,
    }


def test_paired_policy_contrast_reports_score_and_trajectory_changes() -> None:
    fallback = [row(0, 10, 0.4), row(1, 11, 0.6)]
    candidate = [row(0, 10, 0.5), row(1, 11, 0.6, steps=9)]
    contrast = paired_policy_contrast(candidate, fallback)
    assert contrast["mean_paired_score_difference"] == pytest.approx(0.05)
    assert contrast["mean_absolute_paired_score_difference"] == pytest.approx(0.05)
    assert contrast["nonzero_score_difference_fraction"] == 0.5
    assert contrast["trajectory_difference_fraction"] == 1.0


def test_paired_policy_contrast_requires_common_episode_seed_keys() -> None:
    with pytest.raises(ValueError, match="paired"):
        paired_policy_contrast([row(0, 99, 0.5)], [row(0, 10, 0.5)])
