"""Canonical implementation hash for internal v2 competitive baselines."""

from __future__ import annotations

import hashlib
from pathlib import Path


BASELINE_IMPLEMENTATION_FILES = (
    "experiments/frontier_v2_baseline_hash.py",
    "experiments/frontier_v2_baseline_design.py",
    "experiments/frontier_v2_tabular_q_learning.py",
)


def baseline_implementation_sha256(repository_root: Path | None = None) -> str:
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root
    )
    digest = hashlib.sha256()
    for relative in BASELINE_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"baseline implementation file is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()
