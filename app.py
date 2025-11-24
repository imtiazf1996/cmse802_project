# app.py
import os
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

from src.data_clean import load_data, clean_data

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price — EDA & Prediction",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Used Car Price — EDA & Gradient Boosting Prediction")
st.caption(
    "Interactive dashboard for exploring the vehicles dataset and predicting "
    "prices with a trained Gradient Boosting Regressor (GBR)."
)

# -----------------------------------------------------------------------------
# Sidebar: data source + global filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data Source")
    default_path = "vehicles.csv"
    csv_path = st.text_input("CSV path in repo", value=default_path)
    uploaded = st.file_uploader("…or upload CSV", type=["csv"])

    st.divider()
    st.header("Filters")

    price_min, price_max = st.slider(
        "Price range",
        0,
        150_000,
        (1_000, 80_000),
        step=500,
    )

    year_min, year_max = st.slider(
        "Model year",
        1970,
        2025,
        (2000, 2024),
        step=1,
    )

    odo_min, odo_max = st.slider(
        "Mileage (mi)",
        0,
        500_000,
        (0, 250_000),
        step=5_000,
    )

    st.divider()
    sample_n = st.number_input(
        "Sample rows for plotting (speed)",
        min_value=1_000,
        max_value=200_000,
        value=25_000,
        step=1_000,
        help="Plots use a random sample when the filtered dataset is large.",
    )


# -----------------------------------------------------------------------------
# Cached helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def _load_and_clean(buffer_or_path):
    raw = load_data(buffer_or_path)
    clean = clean_data(raw)
    return raw, clean


@st.cache_resource(show_spinner=True)
def _load_model(model_path: str = "results/gbr/best_model.joblib"):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
buffer_or_path: io.BytesIO | str
if uploaded is not None:
    buffer_or_path = uploaded
elif os.path.exists(csv_path):
    buffer_or_path = csv_path
else:
    st.error(
        "CSV not found. Provide a valid path (e.g. `vehicles.csv`) or upload the file."
    )
    st.stop()

with st.spinner("Loading and cleaning data…"):
    df_raw, df = _load_and_clean(buffer_or_path)

if df.empty:
    st.error("Cleaned dataframe is empty after processing. Check cleaning rules.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: category filters that depend on data
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Categorical filters")

    manu_opts = ["(All)"] + sorted(
        df["manufacturer"].dropna().unique().tolist()
    ) if "manufacturer" in df.columns else ["(All)"]
    manu_sel = st.selectbox("Manufacturer", manu_opts, index=0)

    state_opts = ["(All)"] + sorted(
        df["state"].dropna().unique().tolist()
    ) if "state" in df.columns else ["(All)"]
    state_sel = st.selectbox("State", state_opts, index=0)

    type_opts = ["(All)"] + sorted(
        df["type"].dropna().unique().tolist()
    ) if "type" in df.columns else ["(All)"]
    type_sel = st.selectbox("Body type", type_opts, index=0)


# -----------------------------------------------------------------------------
# Apply global filters
# -----------------------------------------------------------------------------
mask = pd.Series(True, index=df.index)

if "price" in df.columns:
    mask &= df["price"].between(price_min, price_max)

if "year" in df.columns:
    mask &= df["year"].between(year_min, year_max)

if "odometer" in df.columns:
    mask &= df["odometer"].between(odo_min, odo_max)

if "manufacturer" in df.columns and manu_sel != "(All)":
    mask &= df["manufacturer"].eq(manu_sel)

if "state" in df.columns and state_sel != "(All)":
    mask &= df["state"].eq(state_sel)

if "type" in df.columns and type_sel != "(All)":
    mask &= df["type"].eq(type_sel)

df_f = df.loc[mask].copy()

if df_f.empty:
    st.warning("No rows match the filters. Try relaxing them.")
    st.stop()

# Optional downsample for plots
if len(df_f) > sample_n:
    df_plot = df_f.sample(sample_n, random_state=42)
else:
    df_plot = df_f


# -----------------------------------------------------------------------------
# Top-level metrics
# -----------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows (clean, filtered)", f"{len(df_f):,}")
if "price" in df_f.columns:
    k2.metric("Median price", f"${df_f['price'].median():,.0f}")
if "year" in df_f.columns:
    k3.metric("Median year", int(df_f["year"].median()))
if "odometer" in df_f.columns:
    k4.metric("Median mileage", f"{int(df_f['odometer'].median()):,} mi")
if "manufacturer" in df_f.columns:
    k5.metric("Manufacturers", df_f["manufacturer"].nunique())


# -----------------------------------------------------------------------------
# Tabs layout
# -----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "Dataset & Summary",
        "Distributions",
        "Price Relationships",
        "Manufacturers & States",
        "Time Trend",
        "GBR Price Prediction",
    ]
)

# -----------------------------------------------------------------------------
# Tab 0: Dataset & summary
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Preview of cleaned data")
    st.dataframe(df_f.head(20), use_container_width=True)

    st.subheader("Summary statistics")
    cols_for_desc = [c for c in ["price", "year", "odometer"] if c in df_f.columns]
    if cols_for_desc:
        st.write(df_f[cols_for_desc].describe())
    else:
        st.info("No numeric columns `price`, `year`, or `odometer` found for summary.")

    st.subheader("NaN counts (after cleaning & filtering)")
    na = df_f.isna().sum().sort_values(ascending=False)
    st.write(na[na > 0])

    st.divider()
    st.download_button(
        "⬇️ Download filtered dataset (CSV)",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="vehicles_filtered.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Tab 1: Distributions
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Distributions")

    c1, c2 = st.columns(2)

    if "price" in df_f.columns:
        with c1:
            fig = px.histogram(
                df_f,
                x="price",
                nbins=60,
                title="Price distribution",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    if "odometer" in df_f.columns:
        with c2:
            fig = px.histogram(
                df_f,
                x="odometer",
                nbins=60,
                title="Mileage distribution",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 2: Price relationships
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Price relationships")

    c1, c2 = st.columns(2)

    if {"year", "price"}.issubset(df_plot.columns):
        with c1:
            color_col = "manufacturer" if "manufacturer" in df_plot.columns else None
            fig = px.scatter(
                df_plot,
                x="year",
                y="price",
                opacity=0.35,
                color=color_col,
                title="Price vs. year",
                hover_data=[c for c in ["manufacturer", "model", "odometer", "state"] if c in df_plot.columns],
            )
            st.plotly_chart(fig, use_container_width=True)

    if {"odometer", "price"}.issubset(df_plot.columns):
        with c2:
            color_col = "manufacturer" if "manufacturer" in df_plot.columns else None
            fig = px.scatter(
                df_plot,
                x="odometer",
                y="price",
                opacity=0.35,
                color=color_col,
                title="Price vs. mileage",
                hover_data=[c for c in ["manufacturer", "model", "year", "state"] if c in df_plot.columns],
            )
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 3: Manufacturers & States
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Manufacturers & geographic patterns")
    c1, c2 = st.columns(2)

    # Box plot by manufacturer
    if "manufacturer" in df_f.columns and "price" in df_f.columns:
        with c1:
            top_manu = df_f["manufacturer"].value_counts().head(15).index
            manu_df = df_f[df_f["manufacturer"].isin(top_manu)]
            if not manu_df.empty:
                fig = px.box(
                    manu_df,
                    x="manufacturer",
                    y="price",
                    points="outliers",
                    title="Price by manufacturer (top 15 by count)",
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough variety for manufacturer plot with current filters.")

    # Median price by state
    if "state" in df_f.columns and "price" in df_f.columns:
        with c2:
            med_state = (
                df_f.groupby("state", as_index=False)["price"]
                .median()
                .sort_values("price", ascending=False)
                .head(20)
            )
            if not med_state.empty:
                fig = px.bar(
                    med_state,
                    x="state",
                    y="price",
                    title="Median price by state (top 20)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough state variation for this plot.")


# -----------------------------------------------------------------------------
# Tab 4: Time trend
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Price trend over time")

    if "posting_date" in df_f.columns:
        pd_col = df_f["posting_date"]
        if pd_col.notna().any():
            month = pd_col.dt.to_period("M")
            trend = (
                df_f.assign(month=month)
                .groupby("month", as_index=False)["price"]
                .median()
            )
            if not trend.empty:
                trend["month_ts"] = trend["month"].dt.to_timestamp()
                trend = trend.sort_values("month_ts")

                fig = px.line(
                    trend,
                    x="month_ts",
                    y="price",
                    markers=True,
                    title="Median price over posting month",
                )
                fig.update_xaxes(title="Month", tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Time trend uses cleaned `posting_date` parsed in `src/data_clean.py`."
                )
            else:
                st.info("No valid price data for time trend after grouping.")
        else:
            st.info("`posting_date` exists but has no valid dates after cleaning.")
    else:
        st.info("No `posting_date` column found; skipping time trend.")


# -----------------------------------------------------------------------------
# Tab 5: GBR Price Prediction
# -----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("🔮 Gradient Boosting Price Prediction")

    model = _load_model()
    if model is None:
        st.warning(
            "GBR model file not found at `results/gbr/best_model.joblib`.\n\n"
            "Train your model and save it to that path to enable prediction."
        )
    else:
        st.write(
            "Use the controls below to describe a single vehicle, then click "
            "**Predict price** to get the model’s estimate."
        )

        # Build category options from filtered data (for realistic values)
        manu_list = sorted(df_f["manufacturer"].dropna().unique()) if "manufacturer" in df_f.columns else []
        model_list = sorted(df_f["model"].dropna().unique()) if "model" in df_f.columns else []
        state_list = sorted(df_f["state"].dropna().unique()) if "state" in df_f.columns else []
        fuel_list = sorted(df_f["fuel"].dropna().unique()) if "fuel" in df_f.columns else []
        trans_list = sorted(df_f["transmission"].dropna().unique()) if "transmission" in df_f.columns else []
        type_list = sorted(df_f["type"].dropna().unique()) if "type" in df_f.columns else []

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)

            with c1:
                year_in = st.number_input(
                    "Model year",
                    min_value=1970,
                    max_value=2025,
                    value=int(df_f["year"].median()) if "year" in df_f.columns else 2015,
                )
                odo_in = st.number_input(
                    "Mileage (mi)",
                    min_value=0,
                    max_value=500_000,
                    value=int(df_f["odometer"].median())
                    if "odometer" in df_f.columns
                    else 80_000,
                    step=1_000,
                )
                price_hint = st.checkbox(
                    "Show reliability note", value=True
                )

            with c2:
                manu_in = (
                    st.selectbox("Manufacturer", manu_list)
                    if manu_list
                    else st.text_input("Manufacturer", "toyota")
                )
                model_in = (
                    st.selectbox("Model", model_list)
                    if model_list
                    else st.text_input("Model", "corolla")
                )
                state_in = (
                    st.selectbox("State", state_list)
                    if state_list
                    else st.text_input("State (e.g. tx, ca, mi)", "mi")
                )

            # optional extra categorical features if present in training
            extra_cats = st.expander("Optional details (fuel, transmission, body type)", expanded=False)
            with extra_cats:
                fuel_in = (
                    st.selectbox("Fuel", fuel_list)
                    if fuel_list
                    else st.text_input("Fuel", "gas")
                )
                trans_in = (
                    st.selectbox("Transmission", trans_list)
                    if trans_list
                    else st.text_input("Transmission", "automatic")
                )
                type_in = (
                    st.selectbox("Body type", type_list)
                    if type_list
                    else st.text_input("Body type", "sedan")
                )

            submitted = st.form_submit_button("Predict price")

             if submitted:
            # Start from a template row so all columns expected by the pipeline exist
            # Use a random row from the FULL cleaned dataframe (not filtered), to be safe
                base = df.iloc[[0]].copy()

            # Overwrite the fields the user controls
                overrides = {
                    "year": year_in,
                    "odometer": odo_in,
                    "manufacturer": manu_in,
                    "model": model_in,
                    "state": state_in,
                    "fuel": fuel_in,
                    "transmission": trans_in,
                    "type": type_in,
                }
    
                for col, val in overrides.items():
                    if col in base.columns:
                        base[col] = val
    
                # Restrict to the exact columns the pipeline was trained on, if available
                feature_cols = getattr(model, "feature_names_in_", None)
                if feature_cols is not None:
                    X = base[list(feature_cols)]
                else:
                    # Fallback: pass all columns
                    X = base
    
                # Predict
                y_pred = model.predict(X)[0]
    
                st.success(f"Estimated price: **${y_pred:,.0f}**")
    
                if price_hint:
                    st.caption(
                        "This estimate is based on the training data distribution. "
                        "Very unusual feature combinations may be less reliable."
                    )
                if price_hint:
                    st.caption(
                        "This estimate is based on the training data distribution. "
                        "Very unusual combinations of features (e.g., extremely low "
                        "mileage for a very old car) may be less reliable."
                    )
