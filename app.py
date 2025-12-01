# app.py
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import streamlit.components.v1 as components

from src.data_clean import load_data, clean_data
from src.features import assemble_feature_frame

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price — EDA & Prediction",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Used Car Price — EDA & Gradient Boosting Prediction")

st.markdown(
    """
This app is part of the **CMSE 802** final project and demonstrates a complete
machine-learning pipeline for predicting used car prices from a Craigslist dataset.

**Features:**
- Explore the cleaned dataset  
- Interactive EDA visualizations  
- Gradient Boosting Regressor performance overview  
- Live price prediction with a ±$500 range (KBB-style)

**GitHub repository:**  
[https://github.com/your-username/cmse802_project](https://github.com/your-username/cmse802_project)

**Contact:** _your.email@domain.com_ (update this in `app.py`)
"""
)

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
DATA_PATH = "vehicles.csv"
MODEL_PATH = "results/gbr/best_model.joblib"
TRAIN_TEST_METRICS_CSV = "results/gbr/train_test_metrics.csv"
FI_CSV = "results/gbr/feature_importances.csv"

PLOT_FILES = {
    "Parity (train)": "results/gbr/parity_train.png",
    "Parity (test)": "results/gbr/parity_test.png",
    "Residuals histogram (test)": "results/gbr/residuals_hist_test.png",
    "Residuals vs predicted (test)": "results/gbr/residuals_vs_pred_test.png",
}

EDA_HTML_FILES = {
    "Price distribution": "results/eda/price_dist.html",
    "Year histogram": "results/eda/year_hist.html",
    "Mileage (odometer) distribution": "results/eda/odometer_dist.html",
    "Price vs year": "results/eda/price_vs_year.html",
    "Price vs odometer": "results/eda/price_vs_odometer.html",
    "3D price vs odometer vs year": "results/eda/price_3d_odo_year.html",
    "Price over time": "results/eda/price_over_time.html",
    "Correlation heatmap": "results/eda/correlation_heatmap.html",
    "Mean price by manufacturer": "results/eda/manufacturer_price_mean_hist.html",
    "Mean price by state": "results/eda/state_price_mean_hist.html",
}


# -------------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------------
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
def get_feature_template(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full engineered feature matrix once, for use as a template."""
    X, _, _ = assemble_feature_frame(df, include_engineered=True)
    return X


@st.cache_data(show_spinner=True)
def load_train_test_metrics(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=True)
def load_feature_importances(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Could not find `{DATA_PATH}` in the repo. Please add it.")
    st.stop()

df = get_clean_data(DATA_PATH)

if df.empty:
    st.error("Cleaned dataset is empty after processing. Check cleaning rules.")
    st.stop()

# Not strictly needed for prediction anymore, but useful for debugging
X_template = get_feature_template(df)

st.write(
    f"Cleaned dataset loaded with **{len(df):,} rows** and **{df.shape[1]} columns**."
)

# -------------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------------
tab_data, tab_eda, tab_metrics, tab_predict = st.tabs(
    ["Dataset / Intro", "EDA plots", "GBR performance", "Price prediction"]
)

# -------------------------------------------------------------------------
# 1. Dataset / Intro tab
# -------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset overview")

    if st.checkbox("Show cleaned data preview"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Showing first 100 rows of the cleaned dataset.")

    st.markdown("### Basic statistics (price, year, mileage)")
    cols = [c for c in ["price", "year", "odometer"] if c in df.columns]
    if cols:
        st.write(df[cols].describe())
    else:
        st.info("No `price`, `year`, or `odometer` columns found.")

    st.markdown("### Column summary")
    st.write(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)}))


# -------------------------------------------------------------------------
# 2. EDA plots tab  (uses precomputed HTML in results/eda/)
# -------------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory data analysis")

    choice = st.selectbox(
        "Choose an EDA visualization",
        list(EDA_HTML_FILES.keys()),
    )

    html_path = EDA_HTML_FILES[choice]
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_str = f.read()
        components.html(html_str, height=700, scrolling=True)
    else:
        st.warning(f"EDA HTML file not found at `{html_path}`.")


# -------------------------------------------------------------------------
# 3. GBR performance tab
# -------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Gradient Boosting Regressor performance")

    model = get_model(MODEL_PATH)
    if model is None:
        st.warning(
            f"Could not load model from `{MODEL_PATH}`.\n"
            "Run `python -m src.run_experiment --input vehicles.csv` first."
        )
    else:
        # --- Train/Test metrics table ---
        st.markdown("### Train/Test metrics")
        tt_df = load_train_test_metrics(TRAIN_TEST_METRICS_CSV)
        if tt_df is None:
            st.info(f"No `train_test_metrics.csv` found at `{TRAIN_TEST_METRICS_CSV}`.")
        else:
            st.dataframe(tt_df, use_container_width=True)

        # --- Feature importances ---
        st.markdown("### Feature importances (from GBR)")
        fi_df = load_feature_importances(FI_CSV)
        if fi_df is None:
            st.info(f"No `feature_importances.csv` found at `{FI_CSV}`.")
        else:
            st.dataframe(fi_df.head(30), use_container_width=True)
            fig = px.bar(
                fi_df.head(30),
                x="feature",
                y="importance",
                title="Top 30 features by importance",
            )
            fig.update_layout(xaxis_tickangle=-60)
            st.plotly_chart(fig, use_container_width=True)

        # --- Diagnostic plots ---
        st.markdown("### Diagnostic plots")
        for label, rel_path in PLOT_FILES.items():
            if os.path.exists(rel_path):
                st.markdown(f"**{label}**")
                st.image(rel_path, use_column_width=True)
            else:
                st.caption(f"Plot not found: `{rel_path}`")

        # --- Hyperparameters (table) ---
        st.markdown("### GBR hyperparameters")
        try:
            est = model.named_steps.get("est", model)
        except AttributeError:
            est = model
        params = est.get_params()
        params_df = pd.DataFrame(
            {"parameter": list(params.keys()), "value": [str(v) for v in params.values()]}
        )
        st.dataframe(params_df, use_container_width=True)


# -------------------------------------------------------------------------
# 4. Price prediction tab
# -------------------------------------------------------------------------
with tab_predict:
    st.subheader("Live price prediction with GBR")

    model = get_model(MODEL_PATH)
    if model is None:
        st.warning(
            f"Could not load model from `{MODEL_PATH}`.\n"
            "Run `python -m src.run_experiment --input vehicles.csv` to train & save the model."
        )
    else:
        manu_list = (
            sorted(df["manufacturer"].dropna().unique())
            if "manufacturer" in df.columns
            else []
        )

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
                value=int(df["odometer"].median())
                if "odometer" in df.columns
                else 80_000,
                step=1_000,
            )

        with col2:
            # Manufacturer
            if manu_list:
                manu_in = st.selectbox("Manufacturer", manu_list)
            else:
                manu_in = st.text_input("Manufacturer", "toyota")

            # Model list conditioned on manufacturer
            if (
                "manufacturer" in df.columns
                and "model" in df.columns
                and manu_list
            ):
                mask = df["manufacturer"].eq(manu_in)
                models_for_manu = sorted(df.loc[mask, "model"].dropna().unique())
            else:
                models_for_manu = (
                    sorted(df["model"].dropna().unique())
                    if "model" in df.columns
                    else []
                )

            if models_for_manu:
                model_in = st.selectbox("Model", models_for_manu)
            else:
                model_in = st.text_input("Model", "corolla")

        if st.button("Predict price"):
            # 1) What columns does the model expect?
            try:
                required_cols = list(model.feature_names_in_)
            except AttributeError:
                st.error(
                    "Model does not expose `feature_names_in_`. "
                    "Was it trained on a pandas DataFrame?"
                )
                st.stop()

            # 2) Build one-row DataFrame with those columns
            row = pd.DataFrame(index=[0], columns=required_cols)

            # 3) Fill defaults from dataset
            for col in required_cols:
                if col in df.columns:
                    series = df[col]
                    if pd.api.types.is_numeric_dtype(series):
                        row.loc[0, col] = series.median()
                    else:
                        mode_vals = series.mode()
                        if not mode_vals.empty:
                            row.loc[0, col] = mode_vals.iloc[0]
                        else:
                            non_null = series.dropna()
                            row.loc[0, col] = (
                                non_null.iloc[0] if not non_null.empty else None
                            )
                else:
                    row.loc[0, col] = 0

            # 4) Override with user inputs
            if "year" in required_cols:
                row.loc[0, "year"] = year_in
            if "odometer" in required_cols:
                row.loc[0, "odometer"] = mileage_in
            if "manufacturer" in required_cols:
                row.loc[0, "manufacturer"] = manu_in
            if "model" in required_cols:
                row.loc[0, "model"] = model_in

            # 5) Predict
            # 5) Predict
            y_pred = float(model.predict(row)[0])
            low = max(0.0, y_pred - 500.0)
            high = max(0.0, y_pred + 500.0)
            
            # Show only the range as the main result
            st.success(
                f"Estimated price range: **${low:,.0f} – ${high:,.0f}** "
            )
    
            # Keep the reliability note below
            st.caption(
                "This estimate assumes patterns similar to the training data. "
                "Unusual combinations (e.g., very old car with ultra-low mileage) "
                "may be less reliable."
            )
