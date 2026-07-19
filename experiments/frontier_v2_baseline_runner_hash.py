"""Hashes for repository-local adapters around pinned baseline implementations."""

from __future__ import annotations

import hashlib
from pathlib import Path


RUNNER_IMPLEMENTATION_FILES = {
    "double DQN": ("experiments/frontier_v2_double_dqn.py",),
    "clipped PPO": (
        "experiments/frontier_v2_double_dqn.py",
        "experiments/frontier_v2_ppo.py",
    ),
    "recurrent PPO": (
        "experiments/frontier_v2_double_dqn.py",
        "experiments/frontier_v2_ppo.py",
    ),
}


def runner_implementation_files(algorithm: str) -> tuple[str, ...]:
    try:
        return RUNNER_IMPLEMENTATION_FILES[algorithm]
    except KeyError as error:
        raise KeyError(f"no frozen runner implementation for {algorithm}") from error


def runner_implementation_sha256(
    algorithm: str,
    repository_root: Path | None = None,
) -> str:
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root
    )
    digest = hashlib.sha256()
    for relative in runner_implementation_files(algorithm):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"baseline runner file is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()
