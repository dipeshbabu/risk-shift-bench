"""Audit pinned upstream source trees for the v2 external feasibility suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.frontier_v2_external_design import CODEBASE_LOCKS, DOMAIN_SPECS


SOURCE_DIRECTORIES = {
    "gymnasium": "gymnasium",
    "or_gym": "or-gym",
    "safety_gymnasium": "safety-gymnasium",
    "minigrid": "minigrid",
}

VERSION_MARKERS = {
    "gymnasium": ("gymnasium/__init__.py", '__version__ = "1.3.0"'),
    "or_gym": ("or_gym/version.py", "VERSION='0.5.0'"),
    "safety_gymnasium": (
        "safety_gymnasium/version.py",
        "__version__ = '1.2.0'",
    ),
    "minigrid": ("minigrid/__init__.py", '__version__ = "3.1.0"'),
}

DOMAIN_SOURCE_MARKERS = {
    "gymnasium_frozenlake": (
        "gymnasium/envs/toy_text/frozen_lake.py",
        "class FrozenLakeEnv",
    ),
    "gymnasium_cliffwalking": (
        "gymnasium/envs/toy_text/cliffwalking.py",
        "class CliffWalkingEnv",
    ),
    "gymnasium_taxi": (
        "gymnasium/envs/toy_text/taxi.py",
        "class TaxiEnv",
    ),
    "or_gym_online_knapsack": (
        "or_gym/envs/classic_or/knapsack.py",
        "class OnlineKnapsackEnv",
    ),
    "or_gym_inventory_management": (
        "or_gym/envs/supply_chain/inventory_management.py",
        "class InvManagementMasterEnv",
    ),
    "safety_gymnasium_point_goal": (
        "safety_gymnasium/tasks/safe_navigation/goal/goal_level0.py",
        "class GoalLevel0",
    ),
    "safety_gymnasium_point_button": (
        "safety_gymnasium/tasks/safe_navigation/button/button_level0.py",
        "class ButtonLevel0",
    ),
    "minigrid_dynamic_obstacles": (
        "minigrid/envs/dynamicobstacles.py",
        "class DynamicObstaclesEnv",
    ),
    "minigrid_lava_crossing": (
        "minigrid/envs/crossing.py",
        "class CrossingEnv",
    ),
}


@dataclass(frozen=True)
class SourceFileAudit:
    path: str
    sha256: str
    required_marker: str


@dataclass(frozen=True)
class CodebaseSourceAudit:
    codebase: str
    source: str
    expected_commit: str
    observed_commit: str
    expected_version: str
    clean: bool
    files: tuple[SourceFileAudit, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_requirements(
    source: Path,
    requirements: tuple[tuple[str, str], ...],
) -> tuple[SourceFileAudit, ...]:
    records = []
    for relative_path, marker in requirements:
        path = source / relative_path
        if not path.is_file():
            raise RuntimeError(f"required upstream source file is missing: {path}")
        content = path.read_text(encoding="utf-8")
        if marker not in content:
            raise RuntimeError(
                f"required marker {marker!r} is missing from upstream file {path}"
            )
        records.append(
            SourceFileAudit(
                path=relative_path,
                sha256=file_sha256(path),
                required_marker=marker,
            )
        )
    return tuple(records)


def codebase_requirements(codebase: str) -> tuple[tuple[str, str], ...]:
    try:
        version_requirement = VERSION_MARKERS[codebase]
    except KeyError as error:
        raise KeyError(f"unknown v2 codebase: {codebase}") from error
    domain_requirements = tuple(
        DOMAIN_SOURCE_MARKERS[domain]
        for domain, spec in DOMAIN_SPECS.items()
        if spec.codebase == codebase
    )
    return (version_requirement, *domain_requirements)


def audit_codebase_source(source: Path, codebase: str) -> CodebaseSourceAudit:
    try:
        lock = CODEBASE_LOCKS[codebase]
    except KeyError as error:
        raise KeyError(f"unknown v2 codebase: {codebase}") from error
    if not source.is_dir():
        raise RuntimeError(f"upstream source directory is missing: {source}")
    resolved_source = source.resolve()
    git_prefix = ["git", "-c", f"safe.directory={resolved_source}", "-C", str(source)]
    observed_commit = subprocess.check_output(
        [*git_prefix, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed_commit != lock.commit:
        raise RuntimeError(
            f"upstream commit changed for {codebase}: expected {lock.commit}, "
            f"found {observed_commit}"
        )
    dirty = subprocess.check_output(
        [*git_prefix, "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"upstream source checkout is dirty for {codebase}")
    return CodebaseSourceAudit(
        codebase=codebase,
        source=str(source.resolve()),
        expected_commit=lock.commit,
        observed_commit=observed_commit,
        expected_version=lock.version,
        clean=True,
        files=verify_source_requirements(source, codebase_requirements(codebase)),
    )


def audit_source_suite(source_root: Path) -> dict:
    audits = {
        codebase: audit_codebase_source(
            source_root / SOURCE_DIRECTORIES[codebase], codebase
        )
        for codebase in CODEBASE_LOCKS
    }
    return {
        "design": "riskshiftbench-frontier-v2-upstream-source-audit",
        "scope": (
            "Read-only source audit; no environment is imported, instantiated, or reset."
        ),
        "codebase_count": len(audits),
        "domain_count": len(DOMAIN_SPECS),
        "audits": {name: asdict(audit) for name, audit in audits.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_source_suite(args.source_root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
