from __future__ import annotations

from pathlib import Path

import pytest

from experiments.frontier_v2_baseline_runner_hash import (
    runner_implementation_files,
    runner_implementation_sha256,
)


def test_double_dqn_runner_file_is_frozen() -> None:
    assert runner_implementation_files("double DQN") == (
        "experiments/frontier_v2_double_dqn.py",
    )
    assert runner_implementation_files("clipped PPO")[-1] == (
        "experiments/frontier_v2_ppo.py"
    )
    assert runner_implementation_files("recurrent PPO") == runner_implementation_files(
        "clipped PPO"
    )
    assert runner_implementation_files("PPO-Lagrangian") == (
        "experiments/frontier_v2_omnisafe.py",
    )
    assert runner_implementation_files("constrained policy optimization") == (
        "experiments/frontier_v2_omnisafe.py",
    )


def test_runner_hash_is_newline_canonical(tmp_path: Path) -> None:
    path = tmp_path / "experiments" / "frontier_v2_double_dqn.py"
    path.parent.mkdir()
    path.write_bytes(b"line1\r\nline2\r\n")
    windows_hash = runner_implementation_sha256("double DQN", tmp_path)
    path.write_bytes(b"line1\nline2\n")
    assert runner_implementation_sha256("double DQN", tmp_path) == windows_hash


def test_unknown_runner_is_rejected() -> None:
    with pytest.raises(KeyError, match="no frozen runner"):
        runner_implementation_files("missing")
