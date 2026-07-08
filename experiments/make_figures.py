"""Generate SVG figures from benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk_preference_inference.figures import bar_chart_svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="artifacts/risk_benchmark/summary.json")
    parser.add_argument("--out-dir", default="artifacts/figures")
    args = parser.parse_args()

    with Path(args.summary).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    rows = payload["summaries"] if isinstance(payload, dict) and "summaries" in payload else payload
    out_dir = Path(args.out_dir)
    bar_chart_svg(rows, "mean_final_bankroll", out_dir / "mean_final_bankroll.svg", "Mean Final Bankroll")
    bar_chart_svg(rows, "cvar_5_final_bankroll", out_dir / "cvar_5_final_bankroll.svg", "5% CVaR Final Bankroll")
    bar_chart_svg(rows, "ruin_probability", out_dir / "ruin_probability.svg", "Ruin Probability")
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()

