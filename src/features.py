"""
features.py

Define feature lists and simple feature engineering so model
uses the same inputs.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

CANON_NUM_COLS = ["year", "odometer"]
CANON_CAT_COLS = [
    "manufacturer",
    "model",
    "condition",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "state",
]
TARGET_COL = "price"


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (num_cols, cat_cols) that actually exist in df.
    """
    num_cols = [c for c in CANON_NUM_COLS if c in df.columns]
    cat_cols = [c for c in CANON_CAT_COLS if c in df.columns]
    return num_cols, cat_cols

def engineer_features(
    df: pd.DataFrame,
    use_log_price: bool = False,
    current_year: Optional[int] = None,
    clip_age_min: int = 0,
    clip_age_max: int = 60,
) -> pd.DataFrame:
    print("Starting feature engineering")
    out = df.copy()
    if current_year is None:
        yr = 2021
    else:
        yr = current_year
    if "year" in out.columns:
        year_num = pd.to_numeric(out["year"], errors="coerce")
        age = (yr - year_num).clip(clip_age_min, clip_age_max)
        out["age"] = age
    else:
        out["age"] = np.nan

    if "odometer" in out.columns:
        odo = pd.to_numeric(out["odometer"], errors="coerce")
        age_for_rate = out["age"].where(out["age"].notna(), 1).clip(lower=1)
        out["odo_per_year"] = odo / age_for_rate
    else:
        out["odo_per_year"] = np.nan

    out["age2"] = out["age"] ** 2
    out["log_odometer"] = np.log1p(pd.to_numeric(out["odometer"], errors="coerce"))
    out["age_odo_per_year"] = out["age"] * out["odo_per_year"]

    if use_log_price and TARGET_COL in out.columns:
        out["price_log"] = np.log1p(pd.to_numeric(out[TARGET_COL], errors="coerce"))
    return out

def get_target(df: pd.DataFrame, use_log_price: bool = False) -> pd.Series:
    """
    Return the target vector according to flag.
    """
    if use_log_price:
        if "price_log" in df.columns:
            return pd.to_numeric(df["price_log"], errors="coerce")
        if TARGET_COL in df.columns:
            return np.log1p(pd.to_numeric(df[TARGET_COL], errors="coerce"))
        raise KeyError("Neither 'price_log' nor 'price' found for log target.")
    else:
        if TARGET_COL in df.columns:
            return pd.to_numeric(df[TARGET_COL], errors="coerce")
        raise KeyError("'price' target column not found.")


def assemble_feature_frame(
    df: pd.DataFrame,
    include_engineered: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:

    work = engineer_features(df) if include_engineered else df.copy()
    num_cols, cat_cols = get_feature_lists(work)

    engineered_numeric = [
        "age",
        "odo_per_year",
        "age2",
        "log_odometer",
        "age_odo_per_year",
        "manufacturer_price_mean",
        "state_price_mean",
    ]

    for extra in engineered_numeric:
        if extra in work.columns and extra not in num_cols:
            num_cols.append(extra)

    X = work[num_cols + cat_cols].copy()

    # ensure numeric columns are numeric
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].apply(
                lambda v: " ".join(v)
                if isinstance(v, list)
                else ("" if pd.isna(v) else str(v))
            )

    print(f"Final feature count = {X.shape[1]}")
    print(f"Numeric: {num_cols}")
    print(f"Categorical: {cat_cols}")
    return X, num_cols, cat_cols
#Parts of this file was completed with the help of ChatGPT 5.1