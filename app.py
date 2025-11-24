# app.py
import os
import json

import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

from src.data_clean import load_data, clean_data
from src.features import assemble_feature_frame

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price — EDA & Prediction",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Used Car Price — EDA & Gradient Boosting Prediction")


# -----------------------------------------------------------------------------
# Data + model loading helpers
# -----------------------------------------------------------------------------
DATA_PATH = "vehicles.csv"
MODEL_PATH = "results/gbr/best_model.joblib"
TEST_METRICS_PATH = "results/gbr/test_metrics.json"


@st.cache_data(show_spinner=True)
def get_clean_data(path: str) -> pd.DataFrame:
    raw = load_data(path)
    clean = clean_data(raw)
    return clean


@st.cache_resource(show_spinner=True)
def get_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=True)
def get_test_metrics(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Could not find `{DATA_PATH}` in the repo. Please add it.")
    st.stop()

df = get_clean_data(DATA_PATH)

if df.empty:
    st.error("Cleaned dataset is empty after processing. Check cleaning rules.")
    st.stop()

# Quick info
st.write(f"Cleaned dataset loaded with **{len(df):,} rows** and **{df.shape[1]} columns**.")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_data, tab_eda, tab_metrics, tab_predict = st.tabs(
    ["Dataset", "EDA plots", "Model performance", "Price prediction"]
)

# -----------------------------------------------------------------------------
# 1. Dataset tab
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("Cleaned dataset preview")

    if st.button("View cleaned data"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Showing first 100 rows of the cleaned dataset.")
    else:
        st.info("Click **View cleaned data** to show a preview.")

    st.markdown("**Basic stats (price, year, mileage):**")
    cols = [c for c in ["price", "year", "odometer"] if c in df.columns]
    if cols:
        st.write(df[cols].describe())
    else:
        st.info("No `price`, `year`, or `odometer` columns found.")


# -----------------------------------------------------------------------------
# 2. EDA plots tab
# -----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory plots")

    plot_type = st.selectbox(
        "Choose a plot",
        [
            "Price distribution",
            "Mileage distribution",
            "Price vs year",
            "Price vs mileage",
            "Price by manufacturer (top 10)",
        ],
    )

    if plot_type == "Price distribution":
        if "price" in df.columns:
            fig = px.histogram(df, x="price", nbins=60, title="Price distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No `price` column found in data.")

    elif plot_type == "Mileage distribution":
        if "odometer" in df.columns:
            fig = px.histogram(df, x="odometer", nbins=60, title="Mileage distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No `odometer` column found in data.")

    elif plot_type == "Price vs year":
        if {"price", "year"}.issubset(df.columns):
            fig = px.scatter(
                df,
                x="year",
                y="price",
                opacity=0.4,
                title="Price vs year",
                hover_data=[c for c in ["manufacturer", "model", "odometer"] if c in df.columns],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need both `price` and `year` columns for this plot.")

    elif plot_type == "Price vs mileage":
        if {"price", "odometer"}.issubset(df.columns):
            fig = px.scatter(
                df,
                x="odometer",
                y="price",
                opacity=0.4,
                title="Price vs mileage",
                hover_data=[c for c in ["manufacturer", "model", "year"] if c in df.columns],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need both `price` and `odometer` columns for this plot.")

    elif plot_type == "Price by manufacturer (top 10)":
        if {"manufacturer", "price"}.issubset(df.columns):
            top_manu = df["manufacturer"].value_counts().head(10).index
            manu_df = df[df["manufacturer"].isin(top_manu)]
            fig = px.box(
                manu_df,
                x="manufacturer",
                y="price",
                title="Price by manufacturer (top 10 by count)",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need `manufacturer` and `price` columns for this plot.")


# -----------------------------------------------------------------------------
# 3. Model performance tab
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Model performance (GBR)")

    # ---- Metrics from JSON ----
    metrics = get_test_metrics(TEST_METRICS_PATH)
    if metrics is None:
        st.warning(
            f"Could not find test metrics file at `{TEST_METRICS_PATH}`.\n"
            "Save your GBR evaluation metrics there as JSON to see them here."
        )
    else:
        st.markdown("### 📈 Test metrics")
        st.markdown("**Raw metrics (from JSON):**")
        st.json(metrics)

        # Plot numeric metrics as a simple bar chart
        num_items = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if num_items:
            mdf = pd.DataFrame(
                {"metric": list(num_items.keys()), "value": list(num_items.values())}
            )
            fig = px.bar(mdf, x="metric", y="value", title="Numeric metrics")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric metrics found to plot.")

    st.markdown("---")

    # ---- Hyperparameters from the saved model ----
    st.markdown("### ⚙️ Hyperparameters used")

    model_obj = get_model(MODEL_PATH)
    if model_obj is None:
        st.warning(
            f"Could not load model from `{MODEL_PATH}`.\n"
            "Hyperparameters are only shown when the model file is available."
        )
    else:
        # Get all parameters
        all_params = model_obj.get_params()

        # Try to extract only the estimator (GBR) params if using a pipeline
        # Common pattern: final step named 'model' -> keys like 'model__n_estimators'
        gbr_params = {
            k.replace("model__", ""): v
            for k, v in all_params.items()
            if k.startswith("model__")
        }

        # Fallback: if that dict is empty, just show all params
        if gbr_params:
            display_params = gbr_params
            st.caption("Showing hyperparameters for the GBR step (`model__*` in the pipeline).")
        else:
            display_params = all_params
            st.caption("Could not detect a `model__` step; showing all model parameters.")

        # Show nicely
        st.json(display_params)

# -----------------------------------------------------------------------------
# 4. Price prediction tab
# -----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict price with Gradient Boosting Regressor")

    model = get_model(MODEL_PATH)
    if model is None:
        st.warning(
            f"Could not load model from `{MODEL_PATH}`.\n"
            "Train your GBR model and save it there as a joblib file."
        )
    else:
        # Prepare value options from dataset
        manu_list = sorted(df["manufacturer"].dropna().unique()) if "manufacturer" in df.columns else []
        model_list = sorted(df["model"].dropna().unique()) if "model" in df.columns else []

        col1, col2 = st.columns(2)

        with col1:
            year_in = st.number_input(
                "Model year",
                min_value=1970,
                max_value=2025,
                value=int(df["year"].median()) if "year" in df.columns else 2015,
            )
            mileage_in = st.number_input(
                "Mileage (mi)",
                min_value=0,
                max_value=500_000,
                value=int(df["odometer"].median()) if "odometer" in df.columns else 80_000,
                step=1_000,
            )

        with col2:
            if manu_list:
                manu_in = st.selectbox("Manufacturer", manu_list)
            else:
                manu_in = st.text_input("Manufacturer", "toyota")

            if model_list:
                model_in = st.selectbox("Model", model_list)
            else:
                model_in = st.text_input("Model", "corolla")

        if st.button("Predict price"):
            # Use a template row from the cleaned dataset so we have *all* columns
            base = df.iloc[[0]].copy()

            # Overwrite only the fields the user controls (if they exist)
            overrides = {
                "year": year_in,
                "odometer": mileage_in,
                "manufacturer": manu_in,
                "model": model_in,
            }

            for col, val in overrides.items():
                if col in base.columns:
                    base[col] = val

            # Recreate the engineered feature frame exactly as in training
            X, _, _ = assemble_feature_frame(base, include_engineered=True)

            # Predict
            y_pred = model.predict(X)[0]
            st.success(f"Estimated price: **${y_pred:,.0f}**")

            st.caption(
                "This estimate is based on the training data distribution. "
                "Very unusual combinations of features (e.g. extremely low mileage "
                "for a very old car) may be less reliable."
            )
