from __future__ import annotations

from pathlib import Path

import pytest

from experiments.frontier_v2_source_audit import (
    DOMAIN_SOURCE_MARKERS,
    VERSION_MARKERS,
    codebase_requirements,
    verify_source_requirements,
)


def test_every_domain_and_codebase_has_a_source_marker() -> None:
    assert len(DOMAIN_SOURCE_MARKERS) == 9
    assert len(VERSION_MARKERS) == 4
    assert len(codebase_requirements("gymnasium")) == 4
    assert len(codebase_requirements("minigrid")) == 3


def test_verify_source_requirements_hashes_matching_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    path = source / "package" / "module.py"
    path.parent.mkdir(parents=True)
    path.write_text("class Expected:\n    pass\n", encoding="utf-8")
    records = verify_source_requirements(
        source, (("package/module.py", "class Expected"),)
    )
    assert records[0].path == "package/module.py"
    assert len(records[0].sha256) == 64


def test_verify_source_requirements_rejects_missing_marker(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("class Different:\n    pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="required marker"):
        verify_source_requirements(tmp_path, (("module.py", "class Expected"),))


def test_unknown_codebase_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown"):
        codebase_requirements("missing")
