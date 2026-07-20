"""Machine-check core algebra and finite null paths of the v2 e-process code."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
    DEFAULT_BETTING_FRACTIONS,
    _mixture_log_weights,
)
from experiments.frontier_v2_statistical_hash import (
    statistical_implementation_sha256,
)


PROOF_CERTIFICATE_FILE = "anytime_familywise_proof_certificate_current.json"


def implementation_sha256() -> str:
    return hashlib.sha256(
        Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


def _stopped_acceptance_e(method: str, sequence: tuple[float, ...]) -> float:
    plan = AnytimeFamilywisePlan(
        task_names=("task",),
        familywise_alpha=0.05,
        futility_familywise_alpha=0.05,
        effect_margin=0.0,
        observation_lower=-1.0,
        observation_upper=1.0,
        minimum_observations=1,
        maximum_observations_per_task=len(sequence),
        e_process_method=method,
    )
    router = AnytimeFamilywiseRouter(plan)
    for observation in sequence:
        if router.evidence("task").decision.value != "undecided":
            break
        router.update("task", observation)
    return math.exp(router.evidence("task").acceptance_log_e)


def exhaustive_null_expectations(method: str, maximum_horizon: int) -> list[dict]:
    records = []
    for horizon in range(1, maximum_horizon + 1):
        values = [
            _stopped_acceptance_e(method, sequence)
            for sequence in itertools.product((-1.0, 1.0), repeat=horizon)
        ]
        expectation = sum(values) / (2**horizon)
        records.append(
            {
                "horizon": horizon,
                "path_count": 2**horizon,
                "expected_stopped_acceptance_e": expectation,
                "maximum_path_e": max(values),
            }
        )
    return records


def build_proof_certificate(maximum_horizon: int = 8) -> dict:
    if maximum_horizon < 1:
        raise ValueError("proof-certificate horizon must be positive")
    weights = [math.exp(value) for value in _mixture_log_weights(
        len(DEFAULT_BETTING_FRACTIONS)
    )]
    endpoint_records = []
    for fraction in DEFAULT_BETTING_FRACTIONS:
        lower_factor = 1.0 - fraction
        upper_factor = 1.0 + fraction
        endpoint_records.append(
            {
                "betting_fraction": fraction,
                "lower_endpoint_factor": lower_factor,
                "upper_endpoint_factor": upper_factor,
                "null_endpoint_expectation": 0.5 * (
                    lower_factor + upper_factor
                ),
                "both_factors_positive": lower_factor > 0.0 and upper_factor > 0.0,
            }
        )
    task_names = tuple(f"task_{index:02d}" for index in range(36))
    family_plan = AnytimeFamilywisePlan(
        task_names=task_names,
        familywise_alpha=0.05,
        futility_familywise_alpha=0.05,
    )
    acceptance_alphas = [
        family_plan.acceptance_alpha(task) for task in task_names
    ]
    methods = {
        method: exhaustive_null_expectations(method, maximum_horizon)
        for method in ("betting_mixture", "predictable_betting")
    }
    return {
        "design": "riskshiftbench-frontier-v2-proof-certificate-v1",
        "scope": (
            "Implementation certificate for algebraic endpoint factors, mixture "
            "weights, familywise alpha allocation, and exhaustive stopped null paths."
        ),
        "implementation_sha256": implementation_sha256(),
        "statistical_implementation_sha256": statistical_implementation_sha256(),
        "maximum_exhaustive_horizon": maximum_horizon,
        "betting_endpoint_algebra": endpoint_records,
        "mixture_weights": weights,
        "mixture_weight_sum": sum(weights),
        "family_task_count": len(task_names),
        "familywise_alpha": family_plan.familywise_alpha,
        "task_acceptance_alphas": acceptance_alphas,
        "task_acceptance_alpha_sum": sum(acceptance_alphas),
        "null_path_checks": methods,
    }


def audit_proof_certificate(payload: dict) -> dict:
    if payload.get("design") != "riskshiftbench-frontier-v2-proof-certificate-v1":
        raise RuntimeError("unexpected proof-certificate design")
    if payload.get("implementation_sha256") != implementation_sha256():
        raise RuntimeError("proof-certificate implementation changed")
    if payload.get("statistical_implementation_sha256") != statistical_implementation_sha256():
        raise RuntimeError("statistical implementation changed after proof certification")
    if int(payload.get("maximum_exhaustive_horizon", -1)) < 8:
        raise RuntimeError("proof-certificate path horizon is too short")
    endpoints = payload.get("betting_endpoint_algebra")
    if not isinstance(endpoints, list) or len(endpoints) != len(
        DEFAULT_BETTING_FRACTIONS
    ):
        raise RuntimeError("proof-certificate betting grid changed")
    if any(
        record.get("both_factors_positive") is not True
        or not math.isclose(
            float(record["null_endpoint_expectation"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for record in endpoints
    ):
        raise RuntimeError("betting endpoint supermartingale algebra failed")
    if not math.isclose(
        float(payload.get("mixture_weight_sum", math.nan)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("betting mixture weights do not sum to one")
    if not math.isclose(
        float(payload.get("task_acceptance_alpha_sum", math.nan)),
        float(payload.get("familywise_alpha", math.nan)),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("task-level acceptance alphas exceed the familywise budget")
    checks = payload.get("null_path_checks")
    if not isinstance(checks, dict) or set(checks) != {
        "betting_mixture",
        "predictable_betting",
    }:
        raise RuntimeError("proof-certificate method coverage changed")
    for method, records in checks.items():
        if len(records) < 8 or any(
            float(record["expected_stopped_acceptance_e"]) > 1.0 + 1e-12
            for record in records
        ):
            raise RuntimeError(f"stopped null expectation exceeds one: {method}")
    return {
        "maximum_exhaustive_horizon": int(payload["maximum_exhaustive_horizon"]),
        "betting_component_count": len(endpoints),
        "mixture_weight_sum": float(payload["mixture_weight_sum"]),
        "family_task_count": int(payload["family_task_count"]),
        "task_acceptance_alpha_sum": float(
            payload["task_acceptance_alpha_sum"]
        ),
        "methods": {
            method: {
                "maximum_expected_stopped_acceptance_e": max(
                    float(record["expected_stopped_acceptance_e"])
                    for record in records
                ),
                "total_enumerated_paths": sum(
                    int(record["path_count"]) for record in records
                ),
            }
            for method, records in checks.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-horizon", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/frontier_v2_development") / PROOF_CERTIFICATE_FILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_proof_certificate(args.maximum_horizon)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
