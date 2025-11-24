"""
features.py
-----------
Central place to define feature lists and lightweight feature engineering
so every model uses the exact same inputs.

Usage:
    from src.features import get_feature_lists, engineer_features

Design:
- Keep this file stateless and fast.
- No I/O, no model code. Only column logic and simple transforms.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime


# ---- canonical raw columns (as used elsewhere) ----
CANON_NUM_COLS = ["year", "odometer"]
CANON_CAT_COLS = [
    "manufacturer", "model", "condition", "fuel", "title_status",
    "transmission", "drive", "type", "state"
]
TARGET_COL = "price"


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (num_cols, cat_cols) filtered to columns that actually exist in df.
    Keeps ordering stable for reproducibility.
    """
    print("[features.get_feature_lists] Determining available numeric/categorical columns...")
    num_cols = [c for c in CANON_NUM_COLS if c in df.columns]
    cat_cols = [c for c in CANON_CAT_COLS if c in df.columns]

    print(f"[features.get_feature_lists] num_cols = {num_cols}")
    print(f"[features.get_feature_lists] cat_cols = {cat_cols}")
    return num_cols, cat_cols


def _infer_current_year(df: pd.DataFrame) -> int:
    """
    Try to infer a reference year:
    - If a 'posting_date' column exists and is datetime-like, use its max year.
    - Otherwise, use the current calendar year.
    """
    print("[features._infer_current_year] Inferring reference year...")
    if "posting_date" in df.columns:
        s = df["posting_date"]
        if np.issubdtype(s.dtype, np.datetime64):
            try:
                yr = int(pd.to_datetime(s).dt.year.max())
                print(f"[features._infer_current_year] Inferred year from posting_date: {yr}")
                if 1900 <= yr <= 2100:
                    return yr
            except Exception:
                pass

    yr = datetime.now().year
    print(f"[features._infer_current_year] Defaulting to system year: {yr}")
    return yr


def engineer_features(
    df: pd.DataFrame,
    use_log_price: bool = False,
    current_year: Optional[int] = None,
    clip_age_min: int = 0,
    clip_age_max: int = 60
) -> pd.DataFrame:
    """
    Add simple, robust engineered features commonly helpful for price prediction.

    New columns:
        - age = current_year - year (clipped)
        - odo_per_year = odometer / max(age, 1)
        - age2
        - log_odometer
        - age_odo_per_year

    Optionally:
        - price_log = log1p(price)
    """
    print("[features.engineer_features] Starting feature engineering...")
    out = df.copy()

    # Determine reference year
    yr = current_year if current_year is not None else _infer_current_year(out)
    print(f"[features.engineer_features] Using reference year = {yr}")

    # age
    if "year" in out.columns:
        print("[features.engineer_features] Computing 'age' feature...")
        age = (yr - pd.to_numeric(out["year"], errors="coerce")).clip(
            clip_age_min, clip_age_max
        )
        out["age"] = age
    else:
        print("[features.engineer_features] 'year' column missing → setting age to NaN.")
        out["age"] = np.nan

    # odo_per_year
    if "odometer" in out.columns:
        print("[features.engineer_features] Computing 'odo_per_year' feature...")
        odo = pd.to_numeric(out["odometer"], errors="coerce")
        age_for_rate = out["age"].where(out["age"].notna(), 1).clip(lower=1)
        out["odo_per_year"] = odo / age_for_rate
    else:
        print("[features.engineer_features] 'odometer' missing → odo_per_year = NaN.")
        out["odo_per_year"] = np.nan

    # ------------------------------------------------------------
    # 🔥 NEW FEATURES: nonlinear + log + interaction
    # ------------------------------------------------------------
    print("[features.engineer_features] Adding extra engineered features: age2, log_odometer, age_odo_per_year...")

    # age squared (nonlinear effect)
    out["age2"] = out["age"] ** 2

    # log transform odometer (helps with long tail)
    out["log_odometer"] = np.log1p(pd.to_numeric(out["odometer"], errors="coerce"))

    # interaction of age and usage rate
    out["age_odo_per_year"] = out["age"] * out["odo_per_year"]
    # ------------------------------------------------------------

    # Optional: log-transformed price
    if use_log_price and TARGET_COL in out.columns:
        print("[features.engineer_features] Adding log-transformed target 'price_log'...")
        out["price_log"] = np.log1p(pd.to_numeric(out[TARGET_COL], errors="coerce"))

    print("[features.engineer_features] Done.")
    return out


def get_target(df: pd.DataFrame, use_log_price: bool = False) -> pd.Series:
    """
    Return the target vector according to flag.
    """
    print(f"[features.get_target] Getting target (log={use_log_price})...")

    if use_log_price:
        if "price_log" in df.columns:
            print("[features.get_target] Using existing price_log column.")
            return pd.to_numeric(df["price_log"], errors="coerce")
        if TARGET_COL in df.columns:
            print("[features.get_target] Computing price_log on the fly.")
            return np.log1p(pd.to_numeric(df[TARGET_COL], errors="coerce"))
        raise KeyError("Neither 'price_log' nor 'price' found for log target.")

    else:
        if TARGET_COL in df.columns:
            print("[features.get_target] Returning raw 'price'.")
            return pd.to_numeric(df[TARGET_COL], errors="coerce")
        raise KeyError("'price' target column not found.")


def assemble_feature_frame(
    df: pd.DataFrame,
    include_engineered: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Convenience helper:
      - Optionally engineer features,
      - Return X with numeric+categorical+engineered columns, and the lists.
    """
    print("[features.assemble_feature_frame] Assembling feature frame...")

    work = engineer_features(df) if include_engineered else df.copy()
    num_cols, cat_cols = get_feature_lists(work)

    # Add engineered numeric columns if present
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
            print(f"[features.assemble_feature_frame] Adding engineered numeric feature: {extra}")
            num_cols = num_cols + [extra]

    X = work[num_cols + cat_cols]
    print(f"[features.assemble_feature_frame] Final feature count = {X.shape[1]}")
    print(f"[features.assemble_feature_frame] Numeric: {num_cols}")
    print(f"[features.assemble_feature_frame] Categorical: {cat_cols}")
    print("[features.assemble_feature_frame] Done.")

    return X, num_cols, cat_cols