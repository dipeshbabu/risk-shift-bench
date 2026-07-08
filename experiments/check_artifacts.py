"""Check paper-run artifact completeness."""

from __future__ import annotations

import argparse

from risk_preference_inference.run_management import check_artifacts, paper_run_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--skip-exact", action="store_true")
    args = parser.parse_args()

    present, missing = check_artifacts(paper_run_paths(args.run_root), include_exact=not args.skip_exact)
    print(f"present={len(present)}")
    print(f"missing={len(missing)}")
    for path in missing:
        print(f"MISSING {path}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

