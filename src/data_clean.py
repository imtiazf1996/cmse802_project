# src/data_clean.py
import pandas as pd
import re

from .make_model_sorted.toyota import TOYOTA_MODEL_MAP
from .make_model_sorted.ford import FORD_MODEL_MAP
from .make_model_sorted.Chevrolet import CHEVROLET_MODEL_MAP
from .make_model_sorted.Honda import HONDA_MODEL_MAP
from .make_model_sorted.Mazda import MAZDA_MODEL_MAP
from .make_model_sorted.Nissan import NISSAN_MODEL_MAP
from .make_model_sorted.Subaru import SUBARU_MODEL_MAP
from .make_model_sorted.Kia import KIA_MODEL_MAP
from .make_model_sorted.Hyundai import HYUNDAI_MODEL_MAP
from .make_model_sorted.BMW import BMW_MODEL_MAP
from .make_model_sorted.Buick import Buick_MODEL_MAP
from .make_model_sorted.Cadillac import CADILLAC_MODEL_MAP
from .make_model_sorted.GMC import GMC_MODEL_MAP
from .make_model_sorted.Mercedes import MERCEDES_MODEL_MAP

"""
data_clean.py

Load and clean raw vehicle data for analysis and modeling.
"""


def load_data(filepath_or_buffer):
    """Load dataset (CSV)."""
    print(f"Loading data from: {filepath_or_buffer}")
    df = pd.read_csv(filepath_or_buffer, low_memory=False)
    return df

def simplify_model_name(model):
    """
    Simplify vehicle model names to a canonical form.
    """
    if pd.isna(model):
        return None

    s = str(model).lower()
    for manu in ["toyota", "ford", "chevrolet", "honda", "nissan", "bmw", "mercedes"]:
        s = s.replace(manu, " ")

    s = re.sub(r"[^a-z0-9 ]", " ", s)

    tokens = s.strip().split()
    return tokens[0] if tokens else None
## Written by ChatGPT 5.1

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data for EDA and modeling.
    """
    print("Starting cleaning process")
    df = df.copy()
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    keep_cols = [
        "price", "year", "manufacturer", "model", "condition",
        "cylinders", "fuel", "odometer", "title_status",
        "transmission", "drive", "type", "state", "posting_date",
    ]
    existing_keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep_cols]

    # Convert numeric columns
    for c in ["price", "year", "odometer"]:
        if c in df.columns:
            print(f"Converting '{c}' to numeric")
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter numeric ranges (basic sanity filters)
    before = len(df)
    if "price" in df.columns:
        df = df[df["price"].between(1000, 100000, inclusive="both")]
    if "year" in df.columns:
        df = df[df["year"].between(1980, 2024, inclusive="both")]
    if "odometer" in df.columns:
        df = df[df["odometer"].between(0, 400000, inclusive="both")]
    print(f"[data_clean.clean] Filtered numeric ranges: {before:,} → {len(df):,} rows.")

    # Clip extreme tails (stabilizes regression training)
    price_lo = df["price"].quantile(0.01)
    price_hi = df["price"].quantile(0.99)
    odo_lo   = df["odometer"].quantile(0.01)
    odo_hi   = df["odometer"].quantile(0.99)
##Clipping was done by ChatGPT 5.1  
    df["price"] = df["price"].clip(lower=price_lo, upper=price_hi)
    df["odometer"] = df["odometer"].clip(lower=odo_lo, upper=odo_hi)

    # Clean text fields
    for c in [
        "manufacturer", "model", "condition", "fuel", "title_status",
        "transmission", "drive", "type", "state",
    ]:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({"nan": None})
            )

    # Simplify model names
    if "model" in df.columns:
        df["model"] = df["model"].apply(simplify_model_name)

    if "manufacturer" in df.columns and "model" in df.columns:

        # Toyota
        mask_toyota = df["manufacturer"].str.lower() == "toyota"
        df.loc[mask_toyota, "model"] = (df.loc[mask_toyota, "model"].map(TOYOTA_MODEL_MAP).fillna(df.loc[mask_toyota, "model"]))

        # Ford
        mask_ford = df["manufacturer"].str.lower() == "ford"
        df.loc[mask_ford, "model"] = (df.loc[mask_ford, "model"].map(FORD_MODEL_MAP).fillna(df.loc[mask_ford, "model"]))

        # Chevrolet
        mask_chevy = df["manufacturer"].str.lower().isin(["chevrolet", "chevy"])
        df.loc[mask_chevy, "model"] = (df.loc[mask_chevy, "model"].map(CHEVROLET_MODEL_MAP).fillna(df.loc[mask_chevy, "model"]))

        # Honda
        mask_honda = df["manufacturer"].str.lower().isin(["honda"])
        df.loc[mask_honda, "model"] = (df.loc[mask_honda, "model"].map(HONDA_MODEL_MAP).fillna(df.loc[mask_honda, "model"]))
        # Mazda
        mask_mazda = df["manufacturer"].str.lower() == "mazda"
        df.loc[mask_mazda, "model"] = (df.loc[mask_mazda, "model"].map(MAZDA_MODEL_MAP).fillna(df.loc[mask_mazda, "model"]))

        # Nissan
        mask_nissan = df["manufacturer"].str.lower() == "nissan"
        df.loc[mask_nissan, "model"] = (df.loc[mask_nissan, "model"].map(NISSAN_MODEL_MAP).fillna(df.loc[mask_nissan, "model"]))
        # Subaru
        mask_subaru = df["manufacturer"].str.lower() == "subaru"
        df.loc[mask_subaru, "model"] = (df.loc[mask_subaru, "model"].map(SUBARU_MODEL_MAP).fillna(df.loc[mask_subaru, "model"]))

        # Kia
        mask_kia = df["manufacturer"].str.lower() == "kia"
        df.loc[mask_kia, "model"] = (df.loc[mask_kia, "model"].map(KIA_MODEL_MAP).fillna(df.loc[mask_kia, "model"]))

        # Hyundai
        mask_hyundai = df["manufacturer"].str.lower() == "hyundai"
        df.loc[mask_hyundai, "model"] = (df.loc[mask_hyundai, "model"].map(HYUNDAI_MODEL_MAP).fillna(df.loc[mask_hyundai, "model"]))

        # BMW
        mask_bmw = df["manufacturer"].str.lower() == "bmw"
        df.loc[mask_bmw, "model"] = (df.loc[mask_bmw, "model"].map(BMW_MODEL_MAP).fillna(df.loc[mask_bmw, "model"]))

        # Buick
        mask_buick = df["manufacturer"].str.lower() == "buick"
        df.loc[mask_buick, "model"] = (df.loc[mask_buick, "model"].map(Buick_MODEL_MAP).fillna(df.loc[mask_buick, "model"]))

        # Cadillac
        mask_cadillac = df["manufacturer"].str.lower() == "cadillac"
        df.loc[mask_cadillac, "model"] = (df.loc[mask_cadillac, "model"].map(CADILLAC_MODEL_MAP).fillna(df.loc[mask_cadillac, "model"]))

        # GMC
        mask_gmc = df["manufacturer"].str.lower() == "gmc"
        df.loc[mask_gmc, "model"] = (df.loc[mask_gmc, "model"].map(GMC_MODEL_MAP).fillna(df.loc[mask_gmc, "model"]))

        # Mercedes
        mask_mercedes = df["manufacturer"].str.lower().isin(["mercedes", "mercedes-benz"])
        df.loc[mask_mercedes, "model"] = (df.loc[mask_mercedes, "model"].map(MERCEDES_MODEL_MAP).fillna(df.loc[mask_mercedes, "model"]))

    # Drop rows with missing required fields
    req = [c for c in ["price", "year", "odometer", "manufacturer", "model"]
           if c in df.columns]
    before = len(df)
    df = df.dropna(subset=req)
    safe_for_dedup = []
    for col in req:
        if df[col].dropna().map(lambda v: not isinstance(v, list)).all():
            safe_for_dedup.append(col)
    if safe_for_dedup:
        df = df.drop_duplicates(subset=safe_for_dedup)

    print(f"Dropped NaNs/duplicates: {before:,} → {len(df):,} rows.")
    df=df.head(1000)
    print("Cleaning complete.")
    return df.reset_index(drop=True)