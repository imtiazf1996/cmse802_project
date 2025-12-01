"""
eda.py
------
Exploratory Data Analysis (EDA) utilities for the vehicle dataset.
"""

from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
import plotly.express as px


def _numeric_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return cols
#This function was written by ChatGPT 5.1

def plot_distributions(df: pd.DataFrame) -> Dict[str, "px.Figure"]:
    """
    Create basic distribution plots for numeric features.
    Used in the EDA tab / section, independent of the model (GBR).
    """
    figs: Dict[str, "px.Figure"] = {}

    if df is None or df.empty:
        return figs

    num_cols = _numeric_columns(df)
    if not num_cols:
        return figs

    # Price
    if "price" in df.columns:
        figs["price_dist"] = px.histogram(
            df, x="price", nbins=60, marginal="box",
            title="Price distribution (with boxplot)"
        )

    # Odometer
    if "odometer" in df.columns:
        figs["odometer_dist"] = px.histogram(
            df, x="odometer", nbins=60, marginal="box",
            title="Odometer distribution (with boxplot)"
        )

    # Year
    if "year" in df.columns:
        figs["year_hist"] = px.histogram(
            df, x="year", nbins=40,
            title="Vehicle year distribution"
        )

    # Other numeric columns
    for col in num_cols:
        if col in {"price", "odometer", "year"}:
            continue
        figs[f"{col}_hist"] = px.histogram(
            df, x=col, nbins=40,
            title=f"Distribution of {col}"
        )

    print(f"plot_distributions: Created figures.")
    return figs

def plot_correlation(df: pd.DataFrame) -> Optional["px.Figure"]:
    """
    Plot a correlation heatmap for numeric columns.
    """

    if df is None or df.empty:
        return None

    num_cols = _numeric_columns(df)
    if len(num_cols) < 2:
        return None

    corr = df[num_cols].corr()

    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Correlation heatmap (numeric features)"
    )
    print("plot_correlation: Heatmap created.")
    return fig

def plot_trends(df: pd.DataFrame) -> Dict[str, "px.Figure"]:
    """
    Create simple trend plots based on time and mileage..
    """
    print("Creating trend plots")
    figs: Dict[str, "px.Figure"] = {}

    if df is None or df.empty:
        return figs

    # Price vs odometer
    if {"price", "odometer"}.issubset(df.columns):
        figs["price_vs_odometer"] = px.scatter(
            df, x="odometer", y="price", opacity=0.4,
            title="Price vs odometer"
        )

    # Price vs year
    if {"price", "year"}.issubset(df.columns):
        figs["price_vs_year"] = px.scatter(
            df, x="year", y="price", opacity=0.4,
            title="Price vs year"
        )
    #3D price–odo–year plot
    if {"price", "odometer", "year", "manufacturer"}.issubset(df.columns):
        target_makes = ["toyota", "honda", "ford", "chevrolet", "chevy", "bmw"]
        df_3d = df.copy()
        df_3d["manufacturer"] = df_3d["manufacturer"].str.lower()
        df_3d = df_3d[df_3d["manufacturer"].isin(target_makes)]

        if not df_3d.empty:
            figs["price_3d_odo_year"] = px.scatter_3d(
                df_3d,
                x="odometer",
                y="year",
                z="price",
                color="manufacturer",
                opacity=0.6,
                title="Price vs Odometer vs Year (selected manufacturers)",
            )

    return figs

#This function was written by ChatGPT 5.1