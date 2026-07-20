from __future__ import annotations

from pathlib import Path

import pytest

from experiments.frontier_v2_external_design import DOMAIN_SPECS
from experiments.frontier_v2_full_rehearsal import (
    build_rehearsal_jobs,
    execute_full_rehearsal,
    rehearsal_command,
)
from experiments.frontier_v2_source_audit import SOURCE_DIRECTORIES


def interpreter_map(root: Path) -> dict[str, Path]:
    return {codebase: root / f"{codebase}-python" for codebase in SOURCE_DIRECTORIES}


def test_full_rehearsal_job_manifest_has_exact_split_coverage(tmp_path: Path) -> None:
    jobs = build_rehearsal_jobs(
        split="development",
        output_directory=tmp_path / "out",
        interpreters=interpreter_map(tmp_path),
    )
    assert len(jobs) == 36
    assert len({job.task.name for job in jobs}) == 36
    assert len({job.output for job in jobs}) == 36
    for domain in DOMAIN_SPECS:
        domain_jobs = [job for job in jobs if job.task.domain == domain]
        assert [job.task_index for job in domain_jobs] == [0, 1, 2, 3]


def test_rehearsal_command_forces_determinism_and_canonical_seed_default(
    tmp_path: Path,
) -> None:
    job = build_rehearsal_jobs(
        split="calibration",
        output_directory=tmp_path / "out",
        interpreters=interpreter_map(tmp_path),
    )[0]
    command = rehearsal_command(
        job,
        episodes=7,
        source_root=tmp_path / "sources",
    )
    assert "--verify-determinism" in command
    assert "--seed-base" not in command
    assert command[command.index("--episodes") + 1] == "7"


def test_full_rehearsal_refuses_confirmation_and_missing_environments(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outcome-eligible"):
        build_rehearsal_jobs(
            split="confirmation",
            output_directory=tmp_path,
            interpreters=interpreter_map(tmp_path),
        )
    with pytest.raises(RuntimeError, match="interpreters are missing"):
        execute_full_rehearsal(
            split="development",
            episodes=1,
            source_root=tmp_path / "sources",
            output_directory=tmp_path / "out",
            interpreters=interpreter_map(tmp_path),
            resume=False,
            timeout_per_task_seconds=10.0,
        )
