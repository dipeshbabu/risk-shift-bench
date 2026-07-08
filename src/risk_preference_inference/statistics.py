"""Statistical summaries for benchmark result artifacts."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(max(int(round(q * (len(ordered) - 1))), 0), len(ordered) - 1)
    return ordered[idx]


def bootstrap_ci(
    values: list[float],
    samples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(draw))
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(mean(values), percentile(estimates, alpha), percentile(estimates, 1.0 - alpha))


def grouped_values(rows: list[dict], key_fields: tuple[str, ...], value_field: str) -> dict[tuple, list[float]]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(float(row[value_field]))
    return grouped


def confidence_table(
    rows: list[dict],
    key_fields: tuple[str, ...] = ("task", "policy"),
    value_field: str = "final_bankroll",
    samples: int = 1000,
    seed: int = 0,
) -> list[dict]:
    output = []
    for key, values in sorted(grouped_values(rows, key_fields, value_field).items()):
        ci = bootstrap_ci(values, samples=samples, seed=seed)
        row = {field: value for field, value in zip(key_fields, key)}
        row.update(
            {
                "metric": value_field,
                "estimate": ci.estimate,
                "ci_low": ci.low,
                "ci_high": ci.high,
                "n": len(values),
            }
        )
        output.append(row)
    return output


def paired_policy_differences(
    rows: list[dict],
    baseline_policy: str,
    metric: str = "final_bankroll",
    samples: int = 1000,
    seed: int = 0,
) -> list[dict]:
    by_task_seed: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_task_seed[(row["task"], int(row["seed"]))][row["policy"]] = float(row[metric])

    policies = sorted({row["policy"] for row in rows if row["policy"] != baseline_policy})
    output = []
    for policy in policies:
        diffs_by_task: dict[str, list[float]] = defaultdict(list)
        for (task, _seed), values in by_task_seed.items():
            if baseline_policy in values and policy in values:
                diffs_by_task[task].append(values[policy] - values[baseline_policy])
        for task, diffs in sorted(diffs_by_task.items()):
            ci = bootstrap_ci(diffs, samples=samples, seed=seed)
            output.append(
                {
                    "task": task,
                    "policy": policy,
                    "baseline_policy": baseline_policy,
                    "metric": metric,
                    "paired_difference": ci.estimate,
                    "ci_low": ci.low,
                    "ci_high": ci.high,
                    "n": len(diffs),
                }
            )
    return output

