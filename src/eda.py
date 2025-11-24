"""
eda.py
------
Exploratory Data Analysis (EDA) utilities for the vehicle dataset.
"""

from __future__ import annotations
from typing import Dict, Optional

import pandas as pd
import plotly.express as px


# ---------------------------- Helper utilities ---------------------------- #

def _numeric_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    print(f"[EDA] _numeric_columns: Found {len(cols)} numeric columns.")
    return cols


# ------------------------- Distribution-level EDA ------------------------- #

def plot_distributions(df: pd.DataFrame) -> Dict[str, "px.Figure"]:
    """
    Create basic distribution plots for numeric features.
    """
    print("[EDA] plot_distributions: Starting distribution plots...")
    figs: Dict[str, "px.Figure"] = {}

    if df is None or df.empty:
        print("[EDA] plot_distributions: No data available.")
        return figs

    num_cols = _numeric_columns(df)
    if not num_cols:
        print("[EDA] plot_distributions: No numeric columns detected.")
        return figs

    # Price
    if "price" in df.columns:
        print("[EDA] plot_distributions: Plotting price distribution...")
        figs["price_dist"] = px.histogram(
            df, x="price", nbins=60, marginal="box",
            title="Price distribution (with boxplot)"
        )

    # Odometer
    if "odometer" in df.columns:
        print("[EDA] plot_distributions: Plotting odometer distribution...")
        figs["odometer_dist"] = px.histogram(
            df, x="odometer", nbins=60, marginal="box",
            title="Odometer distribution (with boxplot)"
        )

    # Year
    if "year" in df.columns:
        print("[EDA] plot_distributions: Plotting year histogram...")
        figs["year_hist"] = px.histogram(
            df, x="year", nbins=40,
            title="Vehicle year distribution"
        )

    # Other numeric columns
    for col in num_cols:
        if col in {"price", "odometer", "year"}:
            continue
        print(f"[EDA] plot_distributions: Plotting histogram for {col}...")
        figs[f"{col}_hist"] = px.histogram(
            df, x=col, nbins=40,
            title=f"Distribution of {col}"
        )

    print(f"[EDA] plot_distributions: Created {len(figs)} figures.")
    return figs


# --------------------------- Correlation heatmap -------------------------- #

def plot_correlation(df: pd.DataFrame) -> Optional["px.Figure"]:
    """
    Plot a correlation heatmap for numeric columns.
    """
    print("[EDA] plot_correlation: Creating correlation heatmap...")

    if df is None or df.empty:
        print("[EDA] plot_correlation: No data available.")
        return None

    num_cols = _numeric_columns(df)
    if len(num_cols) < 2:
        print("[EDA] plot_correlation: Need at least 2 numeric columns.")
        return None

    print(f"[EDA] plot_correlation: Using {len(num_cols)} numeric columns.")
    corr = df[num_cols].corr()

    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Correlation heatmap (numeric features)"
    )
    print("[EDA] plot_correlation: Heatmap created.")
    return fig


# ------------------------------- Trend plots ------------------------------- #

def plot_trends(df: pd.DataFrame) -> Dict[str, "px.Figure"]:
    """
    Create simple trend plots based on time and mileage.
    """
    print("[EDA] plot_trends: Creating trend plots...")
    figs: Dict[str, "px.Figure"] = {}

    if df is None or df.empty:
        print("[EDA] plot_trends: No data available.")
        return figs

    # Posting date → Price over time
    if "posting_date" in df.columns:
        print("[EDA] plot_trends: Handling posting_date → price over time...")
        s = pd.to_datetime(df["posting_date"], errors="coerce")
        mask = s.notna()

        if mask.any():
            df_time = df.loc[mask].copy()
            df_time["month"] = pd.to_datetime(df_time["posting_date"]).dt.to_period("M").dt.to_timestamp()
            grp = df_time.groupby("month", as_index=False)["price"].mean()

            figs["price_over_time"] = px.line(
                grp, x="month", y="price", markers=True,
                title="Average price over time (by month)"
            )
        else:
            print("[EDA] plot_trends: No valid posting_date rows.")

    # Price vs odometer
    if {"price", "odometer"}.issubset(df.columns):
        print("[EDA] plot_trends: Plotting price vs odometer...")
        figs["price_vs_odometer"] = px.scatter(
            df, x="odometer", y="price", opacity=0.4,
            title="Price vs odometer"
        )

    # Price vs year
    if {"price", "year"}.issubset(df.columns):
        print("[EDA] plot_trends: Plotting price vs year...")
        figs["price_vs_year"] = px.scatter(
            df, x="year", y="price", opacity=0.4,
            title="Price vs year"
        )

    print(f"[EDA] plot_trends: Created {len(figs)} figures.")
    return figs
