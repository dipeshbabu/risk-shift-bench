"""Check whether fitted models recover known synthetic subject parameters."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from risk_shift_bench.fitting import fit_static_prospect
from risk_shift_bench.risk_models import ProspectUtilityModel, RiskModel
from risk_shift_bench.synthetic import generate_synthetic_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--decisions", type=int, default=120)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="artifacts/parameter_recovery/summary.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    truth: dict[str, dict[str, float]] = {}

    def factory(subject_idx: int, subject_rng: random.Random) -> RiskModel:
        model = ProspectUtilityModel(
            alpha=subject_rng.choice((0.55, 0.75, 1.0, 1.25)),
            loss_aversion=subject_rng.choice((1.0, 1.5, 2.25, 3.5)),
            temperature=subject_rng.choice((0.3, 0.7, 1.2, 2.0)),
        )
        truth[f"subject_{subject_idx:03d}"] = {
            "alpha": model.alpha,
            "loss_aversion": model.loss_aversion,
            "temperature": model.temperature,
        }
        return model

    records = generate_synthetic_records(args.subjects, args.decisions, rng.randrange(1_000_000), factory)
    by_subject: dict[str, list] = {}
    for record in records:
        by_subject.setdefault(record.subject_id, []).append(record)

    results = {}
    for subject_id, subject_records in by_subject.items():
        fit = fit_static_prospect(subject_records)
        results[subject_id] = {
            "truth": truth[subject_id],
            "fit": asdict(fit),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

