# src/data_clean.py
import pandas as pd
import re

"""
data_clean.py
-------------
Load and clean raw vehicle data for analysis and modeling.
"""

def load_data(filepath_or_buffer):
    """Load the Craigslist dataset (CSV)."""
    print(f"[data_clean.load] Loading data from: {filepath_or_buffer}")
    df = pd.read_csv(filepath_or_buffer, low_memory=False)
    #df = df.sample(n=50000, random_state=42)
    print(f"[data_clean.load] Loaded {len(df):,} rows.")
    return df


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
    """
    print("[data_clean.clean] Starting cleaning process...")
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    print(f"[data_clean.clean] Columns standardized → {list(df.columns)}")

    keep_cols = [
        "price", "year", "manufacturer", "model", "condition",
        "cylinders", "fuel", "odometer", "title_status",
        "transmission", "drive", "type", "state", "posting_date"
    ]
    existing_keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep_cols]
    print(f"[data_clean.clean] Keeping {len(existing_keep_cols)} essential columns.")

    # Convert numeric columns
    for c in ["price", "year", "odometer"]:
        if c in df.columns:
            print(f"[data_clean.clean] Converting '{c}' to numeric...")
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter numeric ranges
    before = len(df)
    if "price" in df.columns:
        df = df[df["price"].between(1000, 100000, inclusive="both")]
    if "year" in df.columns:
        df = df[df["year"].between(1980, 2024, inclusive="both")]
    if "odometer" in df.columns:
        df = df[df["odometer"].between(0, 400000, inclusive="both")]
    print(f"[data_clean.clean] Filtered numeric ranges: {before:,} → {len(df):,} rows.")
    ###FAWAZ###
    price_lo = df["price"].quantile(0.01)
    price_hi = df["price"].quantile(0.99)
    odo_lo   = df["odometer"].quantile(0.01)
    odo_hi   = df["odometer"].quantile(0.99)

    df["price"] = df["price"].clip(lower=price_lo, upper=price_hi)
    df["odometer"] = df["odometer"].clip(lower=odo_lo, upper=odo_hi)
    # Clean text fields
    for c in [
        "manufacturer", "model", "condition", "fuel", "title_status",
        "transmission", "drive", "type", "state"
    ]:
        if c in df.columns:
            print(f"[data_clean.clean] Cleaning text column '{c}'...")
            df[c] = (
                df[c].astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({"nan": None})
            )

    # Simplify model names
    if "model" in df.columns:
        print("[data_clean.clean] Simplifying 'model' names...")
        df["model"] = df["model"].apply(simplify_model_name)

    # Parse posting_date
    if "posting_date" in df.columns:
        print("[data_clean.clean] Parsing posting_date...")
        df["posting_date"] = pd.to_datetime(
            df["posting_date"], errors="coerce", utc=True
        ).dt.tz_localize(None)

    # Drop rows with missing required fields
    req = [c for c in ["price", "year", "odometer", "manufacturer", "model"]
           if c in df.columns]
    before = len(df)
    df = df.dropna(subset=req).drop_duplicates()
    print(f"[data_clean.clean] Dropped NaNs/duplicates: {before:,} → {len(df):,} rows.")

    print("[data_clean.clean] Cleaning complete.")
    return df.reset_index(drop=True)


def coerce_posting_date(df: pd.DataFrame, col: str = "posting_date") -> pd.DataFrame:
    """
    OPTIONAL: More robust posting_date parsing.
    """
    print("[data_clean.date] Coercing posting_date column...")
    out = df.copy()

    if col not in out.columns:
        print(f"[data_clean.date] '{col}' column not found.")
        return out

    # First pass
    s = pd.to_datetime(out[col], errors="coerce", utc=False)

    # Fallback parsing
    mask = s.isna() & out[col].notna()
    if mask.any():
        print(f"[data_clean.date] {mask.sum()} values failed initial parsing.")
        fmts = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
        for fmt in fmts:
            print(f"[data_clean.date] Trying format: {fmt}")
            parsed = pd.to_datetime(out.loc[mask, col], format=fmt, errors="coerce")
            still_nat_idx = s.loc[mask][s.loc[mask].isna()].index
            s.loc[still_nat_idx] = parsed.reindex(still_nat_idx)
            mask = s.isna() & out[col].notna()
            if not mask.any():
                print("[data_clean.date] All remaining dates parsed successfully.")
                break

    out[col] = s
    print("[data_clean.date] Finished coercing posting_date.")
    return out
