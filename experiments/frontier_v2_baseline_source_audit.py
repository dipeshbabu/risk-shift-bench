"""Audit pinned upstream implementations used by v2 competitive baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.frontier_v2_baseline_design import BASELINE_SOURCE_LOCKS
from experiments.frontier_v2_source_audit import SourceFileAudit, verify_source_requirements


BASELINE_SOURCE_DIRECTORIES = {
    "omnisafe": "omnisafe-pinned",
    "rl_starter_files": "rl-starter-files",
    "cleanrl": "cleanrl-pinned",
}

BASELINE_SOURCE_REQUIREMENTS = {
    "omnisafe": (
        ("LICENSE", "Apache License"),
        (
            "omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.py",
            "class PPOLag(PPO)",
        ),
        (
            "omnisafe/algorithms/on_policy/second_order/cpo.py",
            "class CPO(TRPO)",
        ),
    ),
    "rl_starter_files": (
        ("LICENSE", "MIT License"),
        ("scripts/train.py", 'help="algorithm to use: a2c | ppo (REQUIRED)"'),
        ("model.py", "class ACModel(nn.Module, torch_ac.RecurrentACModel)"),
        ("requirements.txt", "torch-ac"),
    ),
    "cleanrl": (
        ("LICENSE", "MIT License"),
        ("cleanrl/dqn.py", "class QNetwork(nn.Module)"),
        ("cleanrl/ppo.py", "class Agent(nn.Module)"),
    ),
}


@dataclass(frozen=True)
class BaselineSourceAudit:
    source_name: str
    source: str
    repository: str
    expected_commit: str
    observed_commit: str
    expected_license: str
    clean: bool
    files: tuple[SourceFileAudit, ...]


def source_requirements(source_name: str) -> tuple[tuple[str, str], ...]:
    try:
        return BASELINE_SOURCE_REQUIREMENTS[source_name]
    except KeyError as error:
        raise KeyError(f"unknown v2 baseline source: {source_name}") from error


def audit_baseline_source(source: Path, source_name: str) -> BaselineSourceAudit:
    try:
        lock = BASELINE_SOURCE_LOCKS[source_name]
    except KeyError as error:
        raise KeyError(f"unknown v2 baseline source: {source_name}") from error
    if not source.is_dir():
        raise RuntimeError(f"baseline source directory is missing: {source}")

    resolved_source = source.resolve()
    git_prefix = ["git", "-c", f"safe.directory={resolved_source}", "-C", str(source)]
    observed_commit = subprocess.check_output(
        [*git_prefix, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed_commit != lock.commit:
        raise RuntimeError(
            f"upstream commit changed for {source_name}: expected {lock.commit}, "
            f"found {observed_commit}"
        )
    dirty = subprocess.check_output(
        [*git_prefix, "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"upstream baseline checkout is dirty for {source_name}")

    return BaselineSourceAudit(
        source_name=source_name,
        source=str(source.resolve()),
        repository=lock.repository,
        expected_commit=lock.commit,
        observed_commit=observed_commit,
        expected_license=lock.license,
        clean=True,
        files=verify_source_requirements(source, source_requirements(source_name)),
    )


def audit_baseline_source_suite(source_root: Path) -> dict:
    if set(BASELINE_SOURCE_DIRECTORIES) != set(BASELINE_SOURCE_LOCKS):
        raise RuntimeError("baseline source-directory coverage changed")
    audits = {
        source_name: audit_baseline_source(
            source_root / BASELINE_SOURCE_DIRECTORIES[source_name], source_name
        )
        for source_name in BASELINE_SOURCE_LOCKS
    }
    return {
        "design": "riskshiftbench-frontier-v2-baseline-source-audit-v1",
        "scope": "Read-only source and license audit; no environment is reset.",
        "source_count": len(audits),
        "audits": {name: asdict(audit) for name, audit in audits.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baseline_sources"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_baseline_source_suite(args.source_root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
