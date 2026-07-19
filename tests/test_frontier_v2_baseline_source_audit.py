from __future__ import annotations

from pathlib import Path

import pytest

from experiments.frontier_v2_baseline_design import BASELINE_SOURCE_LOCKS
from experiments.frontier_v2_baseline_source_audit import (
    BASELINE_SOURCE_DIRECTORIES,
    BASELINE_SOURCE_REQUIREMENTS,
    source_requirements,
)
from experiments.frontier_v2_source_audit import verify_source_requirements


def test_every_external_baseline_source_has_audited_files() -> None:
    assert set(BASELINE_SOURCE_DIRECTORIES) == set(BASELINE_SOURCE_LOCKS)
    assert set(BASELINE_SOURCE_REQUIREMENTS) == set(BASELINE_SOURCE_LOCKS)
    assert all(len(source_requirements(name)) >= 3 for name in BASELINE_SOURCE_LOCKS)


def test_required_baseline_source_files_are_hashed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    path = source / "algorithm.py"
    path.parent.mkdir()
    path.write_text("class FrozenAlgorithm:\n    pass\n", encoding="utf-8")
    records = verify_source_requirements(
        source, (("algorithm.py", "class FrozenAlgorithm"),)
    )
    assert records[0].path == "algorithm.py"
    assert len(records[0].sha256) == 64


def test_unknown_baseline_source_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown"):
        source_requirements("missing")
