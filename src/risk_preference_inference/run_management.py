"""Run manifests, artifact paths, and integrity checks."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PaperRunPaths:
    root: str
    benchmark: str
    adaptive_search: str
    toy_benchmark: str
    statistics: str
    tables: str
    figures: str
    policy_diagnostics: str
    theory_diagnostics: str
    multiround_exact: str
    configs: str
    manifest: str


def timestamped_run_root(prefix: str = "artifacts/paper_run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def paper_run_paths(root: str) -> PaperRunPaths:
    return PaperRunPaths(
        root=root,
        benchmark=f"{root}/benchmark",
        adaptive_search=f"{root}/adaptive_search",
        toy_benchmark=f"{root}/toy_benchmark",
        statistics=f"{root}/statistics",
        tables=f"{root}/tables",
        figures=f"{root}/figures",
        policy_diagnostics=f"{root}/policy_diagnostics",
        theory_diagnostics=f"{root}/theory_diagnostics",
        multiround_exact=f"{root}/multiround_exact",
        configs=f"{root}/configs",
        manifest=f"{root}/manifest.json",
    )


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_manifest(
    paths: PaperRunPaths,
    command: list[str],
    benchmark_config: str,
    adaptive_config: str,
    extra: dict | None = None,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "benchmark_config": benchmark_config,
        "adaptive_config": adaptive_config,
        "paths": asdict(paths),
    }
    if extra:
        payload.update(extra)
    output = Path(paths.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def copy_config_snapshot(paths: PaperRunPaths, *config_paths: str) -> None:
    output_dir = Path(paths.configs)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_names = ("benchmark_config.json", "adaptive_search_config.json")
    for idx, config_path in enumerate(config_paths):
        source = Path(config_path)
        shutil.copy2(source, output_dir / source.name)
        if idx < len(canonical_names):
            shutil.copy2(source, output_dir / canonical_names[idx])


def required_artifacts(paths: PaperRunPaths, include_exact: bool = True) -> list[str]:
    artifacts = [
        f"{paths.benchmark}/episodes.jsonl",
        f"{paths.benchmark}/summary.csv",
        f"{paths.benchmark}/summary.json",
        f"{paths.adaptive_search}/summary.json",
        f"{paths.toy_benchmark}/summary.json",
        f"{paths.toy_benchmark}/episodes.json",
        f"{paths.statistics}/final_bankroll_ci.csv",
        f"{paths.statistics}/paired_final_bankroll_vs_baseline.csv",
        f"{paths.statistics}/paired_drawdown_vs_baseline.csv",
        f"{paths.tables}/best_policy_by_task.csv",
        f"{paths.tables}/normalized_policy_ranks.csv",
        f"{paths.tables}/best_policy_by_task.tex",
        f"{paths.tables}/normalized_policy_ranks.tex",
        f"{paths.figures}/mean_final_bankroll.svg",
        f"{paths.figures}/cvar_5_final_bankroll.svg",
        f"{paths.figures}/ruin_probability.svg",
        f"{paths.policy_diagnostics}/diagnostics.json",
        f"{paths.theory_diagnostics}/diagnostics.json",
        f"{paths.configs}/benchmark_config.json",
        f"{paths.configs}/adaptive_search_config.json",
        paths.manifest,
    ]
    if include_exact:
        artifacts.append(f"{paths.multiround_exact}/summary.json")
    return artifacts


def check_artifacts(paths: PaperRunPaths, include_exact: bool = True) -> tuple[list[str], list[str]]:
    required = required_artifacts(paths, include_exact=include_exact)
    present = []
    missing = []
    for artifact in required:
        if Path(artifact).exists():
            present.append(artifact)
        else:
            missing.append(artifact)
    return present, missing
