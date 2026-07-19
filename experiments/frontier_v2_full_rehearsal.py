"""Execute and audit complete outcome-eligible v2 benchmark splits.

The runner uses one isolated Python environment per upstream codebase, always
performs an exact deterministic rerun, and refuses confirmation tasks. Existing
task artifacts are resumed only after the same strict provenance audit used for
the completed split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from experiments.frontier_v2_external_design import (
    DOMAIN_SPECS,
    V2ExternalTask,
    all_tasks,
    domain_tasks,
)
from experiments.frontier_v2_rehearsal_audit import (
    audit_rehearsal_payload,
    audit_split_coverage_payloads,
)
from experiments.frontier_v2_source_audit import SOURCE_DIRECTORIES


@dataclass(frozen=True)
class RehearsalJob:
    task: V2ExternalTask
    task_index: int
    codebase: str
    interpreter: Path
    output: Path


def default_interpreters(environment_root: Path) -> dict[str, Path]:
    executable = "python.exe" if sys.platform == "win32" else "python"
    binary_directory = "Scripts" if sys.platform == "win32" else "bin"
    return {
        codebase: environment_root
        / SOURCE_DIRECTORIES[codebase]
        / binary_directory
        / executable
        for codebase in SOURCE_DIRECTORIES
    }


def build_rehearsal_jobs(
    *,
    split: str,
    output_directory: Path,
    interpreters: dict[str, Path],
) -> tuple[RehearsalJob, ...]:
    if split not in {"development", "calibration"}:
        raise ValueError("full rehearsal is restricted to outcome-eligible splits")
    if set(interpreters) != set(SOURCE_DIRECTORIES):
        raise ValueError("interpreter map must cover exactly the four v2 codebases")
    task_indices = {
        task.name: index
        for domain in DOMAIN_SPECS
        for index, task in enumerate(domain_tasks(domain, split))
    }
    return tuple(
        RehearsalJob(
            task=task,
            task_index=task_indices[task.name],
            codebase=DOMAIN_SPECS[task.domain].codebase,
            interpreter=interpreters[DOMAIN_SPECS[task.domain].codebase],
            output=output_directory / f"{task.name}.json",
        )
        for task in all_tasks(split)
    )


def rehearsal_command(
    job: RehearsalJob,
    *,
    episodes: int,
    source_root: Path,
) -> tuple[str, ...]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    return (
        str(job.interpreter),
        "-m",
        "experiments.frontier_v2_development",
        "--domain",
        job.task.domain,
        "--split",
        job.task.split,
        "--task-index",
        str(job.task_index),
        "--episodes",
        str(episodes),
        "--verify-determinism",
        "--source-root",
        str(source_root),
        "--output",
        str(job.output),
        "--quiet",
    )


def _load_resumable_payload(job: RehearsalJob, *, episodes: int) -> dict | None:
    if not job.output.is_file():
        return None
    payload = json.loads(job.output.read_text(encoding="utf-8"))
    record = audit_rehearsal_payload(payload)
    if record["task"] != job.task.name:
        raise RuntimeError(f"resume artifact task mismatch: {job.output}")
    if record["episodes_per_policy"] != episodes:
        raise RuntimeError(f"resume artifact episode-count mismatch: {job.output}")
    if record["canonical_seed_schedule"] is not True:
        raise RuntimeError(f"resume artifact uses a noncanonical seed schedule: {job.output}")
    return payload


def execute_full_rehearsal(
    *,
    split: str,
    episodes: int,
    source_root: Path,
    output_directory: Path,
    interpreters: dict[str, Path],
    resume: bool,
    timeout_per_task_seconds: float,
) -> dict:
    if timeout_per_task_seconds <= 0.0:
        raise ValueError("timeout_per_task_seconds must be positive")
    jobs = build_rehearsal_jobs(
        split=split,
        output_directory=output_directory,
        interpreters=interpreters,
    )
    missing = sorted(
        str(path.resolve())
        for path in {job.interpreter for job in jobs}
        if not path.is_file()
    )
    if missing:
        raise RuntimeError(f"isolated Python interpreters are missing: {missing}")
    output_directory.mkdir(parents=True, exist_ok=True)

    payloads = []
    for position, job in enumerate(jobs, start=1):
        payload = _load_resumable_payload(job, episodes=episodes) if resume else None
        if payload is None:
            print(f"[{position:02d}/{len(jobs)}] {job.task.name}", flush=True)
            completed = subprocess.run(
                rehearsal_command(job, episodes=episodes, source_root=source_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_per_task_seconds,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"rehearsal task failed: {job.task.name}\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            payload = json.loads(job.output.read_text(encoding="utf-8"))
            audit_rehearsal_payload(payload)
        payloads.append(payload)

    audit = audit_split_coverage_payloads(
        payloads,
        split=split,
        expected_episodes_per_policy=episodes,
    )
    audit_path = output_directory / "full_split_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("development", "calibration"),
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument(
        "--environment-root",
        type=Path,
        default=Path("artifacts/frontier_v2_envs"),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-per-task-seconds", type=float, default=1_800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = execute_full_rehearsal(
        split=args.split,
        episodes=args.episodes,
        source_root=args.source_root,
        output_directory=args.output_directory,
        interpreters=default_interpreters(args.environment_root),
        resume=args.resume,
        timeout_per_task_seconds=args.timeout_per_task_seconds,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
