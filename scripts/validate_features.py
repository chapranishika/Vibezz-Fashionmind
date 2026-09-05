#!/usr/bin/env python3
"""
Run the feature-parquet data contracts (src/data/contracts.py).

    python scripts/validate_features.py [features_dir]

Exit 1 if any `error`-severity violation is found. Run this after the ETL and
as a gate before deploying a retrained model — the "unused 169 MB
customers_clean.parquet" only sat there because nobody audited the outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.contracts import check_all, summarise  # noqa: E402


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/features"
    print(f"validating {d}/ ...")
    violations = check_all(d)
    errors, warns = summarise(violations)

    for v in violations:
        print("  " + str(v))
    print(f"\n{len(errors)} errors, {len(warns)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
