"""Post-confirmation robustness analyses for pilot-verified routing.

These analyses use already-open final outcomes and are therefore descriptive.
They do not alter or replace the confirmatory result in the locked artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import combinations, product
from math import ceil, log2
from pathlib import Path
from random import Random
from statistics import mean, median


DOMAINS = ("blackjack_v4", "portfolio_v2", "inventory_v1")
DEFAULT_ARTIFACT = Path("artifacts/frontier_pilot_verified_3domain_v1")
DEFAULT_OUT_DIR = Path("artifacts/frontier_pilot_verified_robustness_v1")


@dataclass(frozen=True)
class ScoreWeights:
    cvar: float = 0.5
    target: float = 150.0
    ruin: float = 500.0
    drawdown: float = 0.25


DEFAULT_WEIGHTS = ScoreWeights()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() == "true"


def score_from_row(row: dict[str, str], weights: ScoreWeights) -> float:
    return (
        float(row["mean_final_bankroll"])
        + weights.cvar * float(row["cvar_5_final_bankroll"])
        + weights.target * float(row["target_probability"])
        - weights.ruin * float(row["ruin_probability"])
        - weights.drawdown * float(row["mean_max_drawdown"])
    )


def score_weight_grid() -> list[ScoreWeights]:
    return [
        ScoreWeights(cvar=cvar, target=target, ruin=ruin, drawdown=drawdown)
        for cvar, target, ruin, drawdown in product(
            (0.25, 0.5, 1.0),
            (75.0, 150.0, 300.0),
            (250.0, 500.0, 1000.0),
            (0.125, 0.25, 0.5),
        )
    ]


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_protocol(artifact_root: Path) -> dict:
    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
    protocol_path = Path(summary["protocol"])
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def load_gate_rows(artifact_root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        domain: read_csv(artifact_root / "gates" / f"{domain}.csv")
        for domain in DOMAINS
    }


def load_final_task_effects(artifact_root: Path) -> dict[str, list[dict]]:
    gate_rows = load_gate_rows(artifact_root)
    output: dict[str, list[dict]] = {}
    for domain in DOMAINS:
        gates = {row["task"]: row for row in gate_rows[domain]}
        rows = read_csv(artifact_root / "evaluation" / domain / "task_effects.csv")
        output[domain] = []
        for row in rows:
            gate = gates[row["task"]]
            fallback_score = float(row["fallback_score"])
            candidate_score = float(row["candidate_score"])
            candidate_delta = candidate_score - fallback_score
            output[domain].append(
                {
                    "domain": domain,
                    "task": row["task"],
                    "proposed": truthy(gate["proposal_active"]),
                    "gate_accepted": truthy(gate["gate_accepted"]),
                    "fallback_score": fallback_score,
                    "candidate_score": candidate_score,
                    "candidate_delta": candidate_delta,
                    "candidate_relative_delta": candidate_delta / max(abs(fallback_score), 1.0),
                }
            )
    return output


def summarize_selection(
    effects: dict[str, list[dict]],
    accepted: dict[str, set[str]],
    name: str,
) -> dict:
    domain_raw: dict[str, float] = {}
    domain_relative: dict[str, float] = {}
    accepted_count = 0
    harmful_count = 0
    for domain in DOMAINS:
        rows = effects[domain]
        selected = accepted.get(domain, set())
        deltas = [row["candidate_delta"] if row["task"] in selected else 0.0 for row in rows]
        relative = [row["candidate_relative_delta"] if row["task"] in selected else 0.0 for row in rows]
        domain_raw[domain] = mean(deltas)
        domain_relative[domain] = mean(relative)
        accepted_count += len(selected)
        harmful_count += sum(
            row["task"] in selected and row["candidate_delta"] < 0.0
            for row in rows
        )
    return {
        "strategy": name,
        "accepted": accepted_count,
        "harmful": harmful_count,
        "equal_domain_relative": mean(domain_relative.values()),
        **{f"{domain}_raw": domain_raw[domain] for domain in DOMAINS},
        **{f"{domain}_relative": domain_relative[domain] for domain in DOMAINS},
    }


def strategy_rows(effects: dict[str, list[dict]], gates: dict[str, list[dict[str, str]]]) -> list[dict]:
    fallback = {domain: set() for domain in DOMAINS}
    candidate_everywhere = {
        domain: {row["task"] for row in effects[domain]}
        for domain in DOMAINS
    }
    fit_only = {
        domain: {row["task"] for row in effects[domain] if row["proposed"]}
        for domain in DOMAINS
    }
    proposal_oracle = {
        domain: {
            row["task"]
            for row in effects[domain]
            if row["proposed"] and row["candidate_delta"] > 0.0
        }
        for domain in DOMAINS
    }
    p_values = [
        (domain, row["task"], float(row["sign_test_p"]))
        for domain in DOMAINS
        for row in gates[domain]
        if truthy(row["proposal_active"])
    ]
    m = len(p_values)
    per_proposal = {
        domain: {task for d, task, p in p_values if d == domain and p <= 0.01}
        for domain in DOMAINS
    }
    bonferroni = {
        domain: {task for d, task, p in p_values if d == domain and p <= 0.05 / m}
        for domain in DOMAINS
    }
    ordered = sorted(p_values, key=lambda item: (item[2], item[0], item[1]))
    holm_flat: set[tuple[str, str]] = set()
    for index, (domain, task, p_value) in enumerate(ordered):
        if p_value > 0.05 / (m - index):
            break
        holm_flat.add((domain, task))
    holm = {
        domain: {task for d, task in holm_flat if d == domain}
        for domain in DOMAINS
    }
    bh_index = 0
    for index, (_domain, _task, p_value) in enumerate(ordered, start=1):
        if p_value <= 0.05 * index / m:
            bh_index = index
    bh_flat = {(domain, task) for domain, task, _p in ordered[:bh_index]}
    benjamini_hochberg = {
        domain: {task for d, task in bh_flat if d == domain}
        for domain in DOMAINS
    }
    strategies = (
        ("fallback_only", fallback),
        ("candidate_everywhere", candidate_everywhere),
        ("fit_only_proposals", fit_only),
        ("pilot_verified_per_proposal_0.01", per_proposal),
        ("bonferroni_fwer_0.05", bonferroni),
        ("holm_fwer_0.05", holm),
        ("benjamini_hochberg_fdr_0.05", benjamini_hochberg),
        ("final_outcome_oracle_within_proposals", proposal_oracle),
    )
    rows = [summarize_selection(effects, accepted, name) for name, accepted in strategies]
    required = ceil(log2(m / 0.05))
    for row in rows:
        row["proposal_family_size"] = m
        row["bonferroni_threshold"] = 0.05 / m
        row["unanimous_batches_needed_for_bonferroni"] = required
    return rows


def random_matched_summary(
    effects: dict[str, list[dict]],
    samples: int,
    seed: int,
) -> dict:
    rng = Random(seed)
    proposals = {
        domain: [row["task"] for row in effects[domain] if row["proposed"]]
        for domain in DOMAINS
    }
    accepted_counts = {
        domain: sum(row["gate_accepted"] for row in effects[domain])
        for domain in DOMAINS
    }
    actual = {
        domain: {row["task"] for row in effects[domain] if row["gate_accepted"]}
        for domain in DOMAINS
    }
    actual_summary = summarize_selection(effects, actual, "pilot_verified")
    relative_values = []
    harmful_values = []
    for _ in range(samples):
        chosen = {
            domain: set(rng.sample(proposals[domain], accepted_counts[domain]))
            for domain in DOMAINS
        }
        summary = summarize_selection(effects, chosen, "random_matched")
        relative_values.append(summary["equal_domain_relative"])
        harmful_values.append(summary["harmful"])
    return {
        "scope": "Post-confirmation matched-random diagnostic; final outcomes are already open.",
        "samples": samples,
        "seed": seed,
        "matched_accepted_counts": accepted_counts,
        "actual_equal_domain_relative": actual_summary["equal_domain_relative"],
        "actual_harmful": actual_summary["harmful"],
        "random_equal_domain_relative_mean": mean(relative_values),
        "random_equal_domain_relative_median": median(relative_values),
        "random_equal_domain_relative_ci_low": quantile(relative_values, 0.025),
        "random_equal_domain_relative_ci_high": quantile(relative_values, 0.975),
        "fraction_random_at_or_below_actual": sum(
            value <= actual_summary["equal_domain_relative"] for value in relative_values
        )
        / samples,
        "random_harmful_mean": mean(harmful_values),
        "random_harmful_probability_any": sum(value > 0 for value in harmful_values) / samples,
        "random_harmful_max": max(harmful_values),
    }


def load_pilot_advantages(artifact_root: Path) -> dict[str, dict[str, dict[int, float]]]:
    output: dict[str, dict[str, dict[int, float]]] = {domain: {} for domain in DOMAINS}
    for domain in DOMAINS:
        for batch in range(7):
            rows = read_csv(artifact_root / "pilot" / domain / f"batch_{batch}.csv")
            for row in rows:
                if not truthy(row["proposal_active"]):
                    continue
                output[domain].setdefault(row["task"], {})[batch] = float(row["pilot_advantage"])
        for task, values in output[domain].items():
            if set(values) != set(range(7)):
                raise ValueError(f"pilot batch coverage changed for {domain}/{task}")
    return output


def pilot_budget_rows(artifact_root: Path, effects: dict[str, list[dict]]) -> list[dict]:
    advantages = load_pilot_advantages(artifact_root)
    rows = []
    for budget in range(1, 8):
        subset_summaries = []
        for subset in combinations(range(7), budget):
            accepted = {
                domain: {
                    task
                    for task, batch_values in advantages[domain].items()
                    if all(batch_values[index] > 0.0 for index in subset)
                }
                for domain in DOMAINS
            }
            subset_summaries.append(summarize_selection(effects, accepted, "unanimity"))
        prefix = subset_summaries[0]
        relative = [row["equal_domain_relative"] for row in subset_summaries]
        accepted_counts = [row["accepted"] for row in subset_summaries]
        harmful = [row["harmful"] for row in subset_summaries]
        rows.append(
            {
                "pilot_batches": budget,
                "episodes_per_policy_per_proposal": 50 * budget,
                "batch_subsets": len(subset_summaries),
                "exact_unanimity_p": 2.0 ** (-budget),
                "meets_per_proposal_alpha_0.01": 2.0 ** (-budget) <= 0.01,
                "prefix_accepted": prefix["accepted"],
                "prefix_harmful": prefix["harmful"],
                "prefix_equal_domain_relative": prefix["equal_domain_relative"],
                "accepted_min": min(accepted_counts),
                "accepted_median": median(accepted_counts),
                "accepted_max": max(accepted_counts),
                "harmful_min": min(harmful),
                "harmful_median": median(harmful),
                "harmful_max": max(harmful),
                "equal_domain_relative_min": min(relative),
                "equal_domain_relative_median": median(relative),
                "equal_domain_relative_max": max(relative),
            }
        )
    return rows


def final_metric_rows(
    artifact_root: Path,
) -> tuple[
    dict,
    dict[str, set[str]],
    dict[str, dict[str, dict[str, list[dict[str, str]]]]],
]:
    protocol = load_protocol(artifact_root)
    selected = {
        domain: {
            row["task"]
            for row in read_csv(artifact_root / "gates" / f"{domain}.csv")
            if truthy(row["gate_accepted"])
        }
        for domain in DOMAINS
    }
    by_domain: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {}
    for domain in DOMAINS:
        rows = read_csv(artifact_root / "evaluation" / domain / "combined.csv")
        by_task: dict[str, dict[str, list[dict[str, str]]]] = {}
        for row in rows:
            by_task.setdefault(row["task"], {}).setdefault(row["policy"], []).append(row)
        by_domain[domain] = by_task
    return (protocol, selected, by_domain)


def score_sensitivity_rows(artifact_root: Path) -> list[dict]:
    protocol, selected, by_domain = final_metric_rows(artifact_root)
    rows = []
    for weights in score_weight_grid():
        domain_raw: dict[str, float] = {}
        domain_relative: dict[str, float] = {}
        harmful = 0
        for domain in DOMAINS:
            fallback = protocol["routing"][domain]["fallback_policy"]
            task_deltas = []
            task_relative = []
            for task, policies in by_domain[domain].items():
                router_score = mean(score_from_row(row, weights) for row in policies["pilot_verified_router"])
                fallback_score = mean(score_from_row(row, weights) for row in policies[fallback])
                delta = router_score - fallback_score
                task_deltas.append(delta)
                task_relative.append(delta / max(abs(fallback_score), 1.0))
                harmful += task in selected[domain] and delta < 0.0
            domain_raw[domain] = mean(task_deltas)
            domain_relative[domain] = mean(task_relative)
        rows.append(
            {
                **asdict(weights),
                "equal_domain_relative": mean(domain_relative.values()),
                "harmful_accepted": harmful,
                **{f"{domain}_raw": domain_raw[domain] for domain in DOMAINS},
                **{f"{domain}_relative": domain_relative[domain] for domain in DOMAINS},
            }
        )
    return rows


def score_sensitivity_summary(rows: list[dict]) -> dict:
    values = [row["equal_domain_relative"] for row in rows]
    default = next(
        row
        for row in rows
        if all(row[key] == value for key, value in asdict(DEFAULT_WEIGHTS).items())
    )
    return {
        "scope": "Post-confirmation score sensitivity with the deployed routes held fixed.",
        "weight_grid": {
            "cvar": [0.25, 0.5, 1.0],
            "target": [75.0, 150.0, 300.0],
            "ruin": [250.0, 500.0, 1000.0],
            "drawdown": [0.125, 0.25, 0.5],
        },
        "variants": len(rows),
        "default_equal_domain_relative": default["equal_domain_relative"],
        "positive_variants": sum(value > 0.0 for value in values),
        "nonnegative_variants": sum(value >= 0.0 for value in values),
        "zero_harmful_variants": sum(row["harmful_accepted"] == 0 for row in rows),
        "equal_domain_relative_min": min(values),
        "equal_domain_relative_median": median(values),
        "equal_domain_relative_max": max(values),
        "worst_variant": min(rows, key=lambda row: row["equal_domain_relative"]),
        "best_variant": max(rows, key=lambda row: row["equal_domain_relative"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    effects = load_final_task_effects(args.artifact_root)
    gates = load_gate_rows(args.artifact_root)
    strategies = strategy_rows(effects, gates)
    random_summary = random_matched_summary(effects, args.random_samples, args.seed)
    budgets = pilot_budget_rows(args.artifact_root, effects)
    sensitivity = score_sensitivity_rows(args.artifact_root)
    sensitivity_summary = score_sensitivity_summary(sensitivity)

    write_csv(args.out_dir / "strategy_comparison.csv", strategies)
    write_json(args.out_dir / "random_matched_summary.json", random_summary)
    write_csv(args.out_dir / "pilot_budget_curve.csv", budgets)
    write_csv(args.out_dir / "score_weight_sensitivity.csv", sensitivity)
    write_json(args.out_dir / "score_weight_sensitivity_summary.json", sensitivity_summary)
    summary = {
        "scope": (
            "All analyses in this directory are post-confirmation and descriptive; "
            "they do not replace the locked primary analysis."
        ),
        "artifact_root": str(args.artifact_root),
        "strategy_comparison": strategies,
        "random_matched": random_summary,
        "pilot_budget": budgets,
        "score_weight_sensitivity": sensitivity_summary,
    }
    write_json(args.out_dir / "summary.json", summary)
    print(f"robustness_summary={args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
