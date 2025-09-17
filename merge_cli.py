"""
merge_cli.py — Command-line wrapper around datamerge.py

Usage example:
  python merge_cli.py --left customers.csv --right orders.csv --on customer_id --how left --output merged.csv --report report.txt --dedupe-left customer_id --dedupe-right customer_id --coalesce city email

The example above will:
- Left-join customers to orders on customer_id
- De-duplicate both sides on customer_id, keeping the last occurrence
- Coalesce overlapping 'city' and 'email' columns (taking left where available)
- Save merged.csv and a merge audit report
"""

import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datamerge import quick_merge_with_audit, audit_counts, resolve_conflicts

def main():
    p = argparse.ArgumentParser(description="Merge two CSVs with safe defaults and an audit report.")
    p.add_argument("--left", required=True, help="Path to left CSV")
    p.add_argument("--right", required=True, help="Path to right CSV")
    p.add_argument("--on", required=True, nargs="+", help="Join key(s). Example: --on customer_id OR --on col1 col2")
    p.add_argument("--how", default="left", choices=["inner", "left", "right", "outer"], help="Join type (default: left)")
    p.add_argument("--output", required=True, help="Where to write merged CSV")
    p.add_argument("--report", help="Optional path for a text audit report")
    p.add_argument("--dedupe-left", nargs="+", help="Key(s) to de-duplicate left by (keep last)")
    p.add_argument("--dedupe-right", nargs="+", help="Key(s) to de-duplicate right by (keep last)")
    p.add_argument("--validate", help="pandas merge validate string, e.g., one_to_one, one_to_many, many_to_one")
    p.add_argument("--coalesce", nargs="*", default=[], help="Overlapping columns to coalesce (prefer left, fallback to right)")
    args = p.parse_args()

    merged, counts = quick_merge_with_audit(
        left_path=args.left,
        right_path=args.right,
        on=args.on,
        how=args.how,
        dedupe_left_keys=args.dedupe_left,
        dedupe_right_keys=args.dedupe_right,
        validate=args.validate,
        suffixes=("_left", "_right"),
        conflicts={c: "coalesce" for c in args.coalesce} if args.coalesce else None
    )

    # Save merged output
    out_path = Path(args.output)
    merged.to_csv(out_path, index=False)
    print(f"Wrote merged output to {out_path.resolve()}")

    # Optional report
    if args.report:
        # Prepare small samples of left-only/right-only
        left_only = merged[merged["_merge"] == "left_only"]
        right_only = merged[merged["_merge"] == "right_only"]
        # Reuse datamerge.save_report to write a friendly audit
        from datamerge import save_report
        save_report(args.report, counts, left_only, right_only, sample_size=5)
        print(f"Wrote audit report to {Path(args.report).resolve()}")

    # Print a tiny summary to stdout too
    print("=== Merge summary ===")
    for k, v in counts.items():
        print(f"{k:>12}: {v}")

if __name__ == "__main__":
    main()
