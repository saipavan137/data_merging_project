"""
datamerge.py — A tiny, reusable helper for merging tabular data (CSV/Parquet) with safeguards.

Beginner-friendly features:
- Consistent column cleanup (lowercase, trim, spaces→underscores)
- Safe dtype casting (won't crash on bad values; logs problems)
- De-duplication on key columns
- Merge with an audit trail (how many matched, left-only, right-only)
- Simple conflict resolution for overlapping columns (prefer left/right, or coalesce)

Requires: pandas
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple, Literal
import pandas as pd
import sys

# Type hint for join type
JoinHow = Literal["inner", "left", "right", "outer"]

def normalize_columns(df: pd.DataFrame,
                      lower: bool = True,
                      strip: bool = True,
                      spaces_to_underscores: bool = True) -> pd.DataFrame:
    """
    Standardize column names so joins don't fail because of casing or stray spaces.
    """
    def _clean(col: str) -> str:
        new = col
        if strip:
            new = new.strip()
        if lower:
            new = new.lower()
        if spaces_to_underscores:
            new = new.replace(" ", "_")
        return new
    df = df.rename(columns={c: _clean(c) for c in df.columns})
    return df


def read_csv(path: str,
             dtype_map: Optional[Dict[str, str]] = None,
             parse_dates: Optional[Iterable[str]] = None,
             rename_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Load a CSV safely and optionally parse dtypes/dates and rename columns.

    - dtype_map: e.g., {"customer_id": "Int64"} (note capital I for pandas nullable int)
    - parse_dates: e.g., ["signup_date"]
    - rename_map: e.g., {"Customer Id": "customer_id"} BEFORE normalization
    """
    df = pd.read_csv(path)
    if rename_map:
        df = df.rename(columns=rename_map)
    df = normalize_columns(df)  # standardize after optional rename

    # Safe casting: try each dtype; if it fails, keep original and warn
    if dtype_map:
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"[WARN] Could not cast column '{col}' to {dtype}: {e}", file=sys.stderr)

    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception as e:
                    print(f"[WARN] Could not parse dates for '{col}': {e}", file=sys.stderr)
    return df


def drop_dupes_on(df: pd.DataFrame, keys: List[str], keep: str = "last") -> pd.DataFrame:
    """
    Drop duplicate key rows, keeping 'first' or 'last' occurrence.
    """
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"Keys not found in dataframe: {missing}")
    before = len(df)
    out = df.drop_duplicates(subset=keys, keep=keep).copy()
    after = len(out)
    if after != before:
        print(f"[INFO] drop_dupes_on: removed {before - after} duplicate rows based on {keys}.", file=sys.stderr)
    return out


def merge_frames(left: pd.DataFrame,
                 right: pd.DataFrame,
                 on: List[str],
                 how: JoinHow = "inner",
                 validate: Optional[str] = None,
                 suffixes: Tuple[str, str] = ("_left", "_right"),
                 indicator: bool = True) -> pd.DataFrame:
    """
    Merge with pandas.merge and (optionally) get a _merge indicator column.
    - on: join keys. They must exist in both frames.
    - how: inner/left/right/outer
    - validate: pandas join validation string, e.g., "one_to_one", "one_to_many", etc.
    - suffixes: appended to overlapping column names from left/right
    """
    missing_left = [k for k in on if k not in left.columns]
    missing_right = [k for k in on if k not in right.columns]
    if missing_left or missing_right:
        raise KeyError(f"Join keys missing — left:{missing_left} right:{missing_right}")

    merged = pd.merge(
        left, right,
        how=how,
        on=on,
        suffixes=suffixes,
        validate=validate,
        indicator=indicator
    )
    return merged


def resolve_conflicts(df: pd.DataFrame,
                      base_to_strategy: Dict[str, Literal["left", "right", "coalesce"]],
                      suffixes: Tuple[str, str] = ("_left", "_right")) -> pd.DataFrame:
    """
    For overlapping columns (e.g., 'city_left' and 'city_right'), produce a single 'city' column.
    Strategies:
      - 'left'     -> always take left value
      - 'right'    -> always take right value
      - 'coalesce' -> take left if not null else right
    Leaves original suffixed columns in place for audit unless you drop them later.
    """
    l_suf, r_suf = suffixes
    for base, strategy in base_to_strategy.items():
        left_col = base + l_suf
        right_col = base + r_suf
        if left_col not in df.columns or right_col not in df.columns:
            # Not an overlapping column; skip quietly.
            continue

        out_col = base  # final, de-suffixed column
        if strategy == "left":
            df[out_col] = df[left_col]
        elif strategy == "right":
            df[out_col] = df[right_col]
        elif strategy == "coalesce":
            df[out_col] = df[left_col].where(df[left_col].notna(), df[right_col])
        else:
            raise ValueError(f"Unknown strategy '{strategy}' for column '{base}'. Use left/right/coalesce.")
    return df


def audit_counts(merged_with_indicator: pd.DataFrame) -> Dict[str, int]:
    """
    Summarize merge results when indicator=True.
    Returns counts for left_only, right_only, both.
    """
    if "_merge" not in merged_with_indicator.columns:
        raise KeyError("No _merge column found. Call merge_frames(..., indicator=True).")
    value_counts = merged_with_indicator["_merge"].value_counts(dropna=False)
    return {
        "left_only": int(value_counts.get("left_only", 0)),
        "right_only": int(value_counts.get("right_only", 0)),
        "both": int(value_counts.get("both", 0)),
        "total_rows": int(len(merged_with_indicator)),
    }


def save_report(path: str,
                counts: Dict[str, int],
                sample_left_only: Optional[pd.DataFrame] = None,
                sample_right_only: Optional[pd.DataFrame] = None,
                sample_size: int = 5) -> None:
    """
    Write a plain-text audit report for your merge.
    """
    lines = []
    lines.append("=== Merge Audit Report ===")
    lines.append(f"Total rows in merged output: {counts.get('total_rows', 0)}")
    lines.append(f"Matched on both sides      : {counts.get('both', 0)}")
    lines.append(f"Left-only rows             : {counts.get('left_only', 0)}")
    lines.append(f"Right-only rows            : {counts.get('right_only', 0)}")
    lines.append("")

    def _df_to_text(df: Optional[pd.DataFrame], title: str) -> List[str]:
        if df is None or df.empty:
            return [f"{title}: (none)"]
        head = df.head(sample_size)
        return [title + " (showing up to " + str(sample_size) + "):", head.to_string(index=False)]

    lines += _df_to_text(sample_left_only, "Examples of LEFT-ONLY rows")
    lines.append("")
    lines += _df_to_text(sample_right_only, "Examples of RIGHT-ONLY rows")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def quick_merge_with_audit(
    left_path: str,
    right_path: str,
    on: List[str],
    how: JoinHow = "left",
    left_dtypes: Optional[Dict[str, str]] = None,
    right_dtypes: Optional[Dict[str, str]] = None,
    left_parse_dates: Optional[Iterable[str]] = None,
    right_parse_dates: Optional[Iterable[str]] = None,
    dedupe_left_keys: Optional[List[str]] = None,
    dedupe_right_keys: Optional[List[str]] = None,
    validate: Optional[str] = None,
    suffixes: Tuple[str, str] = ("_left", "_right"),
    conflicts: Optional[Dict[str, Literal["left", "right", "coalesce"]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    One-call helper for the common case:
      - reads two CSVs
      - normalizes columns
      - optional de-duplication
      - merges with indicator
      - optional conflict resolution
      - returns merged df + counts
    """
    left = read_csv(left_path, dtype_map=left_dtypes, parse_dates=left_parse_dates)
    right = read_csv(right_path, dtype_map=right_dtypes, parse_dates=right_parse_dates)

    if dedupe_left_keys:
        left = drop_dupes_on(left, dedupe_left_keys, keep="last")
    if dedupe_right_keys:
        right = drop_dupes_on(right, dedupe_right_keys, keep="last")

    merged = merge_frames(left, right, on=on, how=how, validate=validate, suffixes=suffixes, indicator=True)

    if conflicts:
        merged = resolve_conflicts(merged, conflicts, suffixes=suffixes)

    counts = audit_counts(merged)
    return merged, counts
