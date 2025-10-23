# src/data_clean.py
import pandas as pd
import re

"""
data_clean.py
-------------
Load and clean raw vehicle data for analysis and modeling.

This module provides functions to import the Craigslist dataset,
clean missing or inconsistent fields, remove outliers, and convert
data types for modeling. It also standardizes categorical fields,
simplifies model names, and parses posting dates.

Author
------
Fawaz Imtiaz
Date: October 2025
"""

def load_data(filepath_or_buffer):
    """Load the Craigslist dataset (CSV)."""
    return pd.read_csv(filepath_or_buffer, low_memory=False)


def simplify_model_name(model):
    """
    Extract the main car model name.
    Example: 'corolla le 4dr sedan' → 'corolla'
    """
    if pd.isna(model):
        return None
    m = re.match(r'^[A-Za-z0-9]+', str(model))
    return m.group(0).lower() if m else None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Craigslist car data for EDA and modeling.
    - Keep relevant columns
    - Remove invalid rows
    - Simplify model names
    - Parse posting_date
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    keep_cols = [
        "price", "year", "manufacturer", "model", "condition",
        "cylinders", "fuel", "odometer", "title_status",
        "transmission", "drive", "type", "state", "posting_date"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # numeric
    for c in ["price", "year", "odometer"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ranges
    if "price" in df.columns:
        df = df[df["price"].between(1000, 100000, inclusive="both")]
    if "year" in df.columns:
        df = df[df["year"].between(1980, 2024, inclusive="both")]
    if "odometer" in df.columns:
        df = df[df["odometer"].between(0, 400000, inclusive="both")]

    # text cleanup
    for c in ["manufacturer", "model", "condition", "fuel", "title_status",
              "transmission", "drive", "type", "state"]:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": None})
            )

    # simplify model
    if "model" in df.columns:
        df["model"] = df["model"].apply(simplify_model_name)

    # date parsing
    if "posting_date" in df.columns:
        df["posting_date"] = pd.to_datetime(
            df["posting_date"], errors="coerce", utc=True
        ).dt.tz_localize(None)

    # required fields
    req = [c for c in ["price", "year", "odometer", "manufacturer", "model"]
           if c in df.columns]
    df = df.dropna(subset=req).drop_duplicates()

    return df.reset_index(drop=True)


def coerce_posting_date(df: pd.DataFrame, col: str = "posting_date") -> pd.DataFrame:
    """
    Parse df[col] to datetime with robust fallback parsing.
    Tries pandas' general parser, then explicit common formats.
    Bad values -> NaT.
    """
    out = df.copy()
    if col not in out.columns:
        return out

    # First pass
    s = pd.to_datetime(out[col], errors="coerce", utc=False)

    # Fallbacks
    mask = s.isna() & out[col].notna()
    if mask.any():
        fmts = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
        vals = out.loc[mask, col]
        for fmt in fmts:
            parsed = pd.to_datetime(vals, format=fmt, errors="coerce")
            still_nat_idx = s.loc[mask][s.loc[mask].isna()].index
            s.loc[still_nat_idx] = parsed.reindex(still_nat_idx)
            mask = s.isna() & out[col].notna()
            if not mask.any():
                break

    out[col] = s
    return out
