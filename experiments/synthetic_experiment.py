"""Run a full synthetic-data model comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from risk_shift_bench.dataset import save_jsonl
from risk_shift_bench.evaluation import evaluate
from risk_shift_bench.fitting import fit_state_dependent_prospect, fit_static_prospect
from risk_shift_bench.risk_models import (
    CVaRModel,
    CumulativeProspectModel,
    EntropicRiskModel,
    ExpectedValueModel,
    OptimizedCertaintyEquivalentModel,
    ProspectUtilityModel,
    StateDependentProspectModel,
)
from risk_shift_bench.splits import make_split
from risk_shift_bench.synthetic import generate_state_dependent_synthetic_records, generate_synthetic_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--decisions", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--state-dependent", action="store_true")
    parser.add_argument("--fit-depth", type=int, default=1)
    parser.add_argument(
        "--split",
        choices=("within_subject", "cross_subject", "cross_bankroll"),
        default="within_subject",
    )
    parser.add_argument("--out-dir", default="artifacts/synthetic")
    args = parser.parse_args()

    if args.state_dependent:
        records = generate_state_dependent_synthetic_records(args.subjects, args.decisions, args.seed)
    else:
        records = generate_synthetic_records(args.subjects, args.decisions, args.seed)

    train, test = make_split(records, protocol=args.split, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(records, out_dir / "decisions.jsonl")

    static_fit = fit_static_prospect(train, max_depth=args.fit_depth)
    dynamic_fit = fit_state_dependent_prospect(train, max_depth=args.fit_depth)

    models = [
        ExpectedValueModel(),
        CVaRModel(alpha=0.1),
        EntropicRiskModel(risk_aversion=1.0),
        OptimizedCertaintyEquivalentModel(shortfall_penalty=1.0),
        CumulativeProspectModel(alpha=0.75, loss_aversion=2.25, probability_weight=0.7),
        ProspectUtilityModel(
            alpha=static_fit.params["alpha"],
            loss_aversion=static_fit.params["loss_aversion"],
            temperature=static_fit.params["temperature"],
        ),
        StateDependentProspectModel(
            **{key: value for key, value in dynamic_fit.params.items() if key != "name"}
        ),
    ]

    summary = {
        "records": len(records),
        "train_records": len(train),
        "test_records": len(test),
        "split": args.split,
        "fits": {
            "static_prospect": asdict(static_fit),
            "state_dependent_prospect": asdict(dynamic_fit),
        },
        "test_evaluation": [asdict(evaluate(test, model, max_depth=args.fit_depth)) for model in models],
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    print(json.dumps(summary["test_evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
