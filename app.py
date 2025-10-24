# app.py
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

from src.data_clean import load_data, clean_data

st.set_page_config(page_title="Used Car Price EDA", page_icon="🚗", layout="wide")
st.title("🚗 Used Car Price — EDA")
#st.caption("Dataset: Kaggle Craigslist Cars & Trucks (Austin Reese). Cleaned per project rules.")

# ---- Sidebar: data source & filters ----
with st.sidebar:
    st.header("Data")
    default_path = "vehicles.csv"   # your file in repo root
    path = st.text_input("CSV path", value=default_path)
    uploaded = st.file_uploader("…or upload CSV", type=["csv"])

    st.divider()
    st.header("Filters")
    price_min, price_max = st.slider("Price (USD)", 0, 150_000, (1_000, 100_000), step=500)
    year_min, year_max   = st.slider("Year", 1970, 2025, (2000, 2024), step=1)
    odo_min, odo_max     = st.slider("Mileage (mi)", 0, 500_000, (0, 250_000), step=5_000)

    st.divider()
    sample_n = st.number_input("Sample rows for plotting (speed)", min_value=1000, max_value=200000,
                               value=25000, step=1000)

@st.cache_data(show_spinner=True)
def _load_clean(buffer_or_path):
    raw = load_data(buffer_or_path)
    clean = clean_data(raw)
    return raw, clean

# ---- Load data ----
buffer_or_path: io.BytesIO | str
if uploaded is not None:
    buffer_or_path = uploaded
elif os.path.exists(path):
    buffer_or_path = path
else:
    st.error("CSV not found. Provide a valid path or upload vehicles.csv.")
    st.stop()

with st.spinner("Loading & cleaning data…"):
    df_raw, df = _load_clean(buffer_or_path)

# ---- Quick health panel ----
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows (clean)", f"{len(df):,}")
k2.metric("Median price", f"${df['price'].median():,.0f}")
k3.metric("Median year", int(df['year'].median()))
k4.metric("Median mileage", f"{int(df['odometer'].median()):,} mi")
k5.metric("Manufacturers", df["manufacturer"].nunique())

with st.expander("Preview & Summary", expanded=False):
    st.subheader("Head (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Describe (price / year / odometer)")
    st.write(df[["price", "year", "odometer"]].describe())
    st.subheader("NaN counts (after cleaning)")
    na = df.isna().sum().sort_values(ascending=False)
    st.write(na[na > 0])

# ---- Extra filter widgets (manufacturer/state/type) ----
with st.sidebar:
    manu_opts = ["(All)"] + sorted(df["manufacturer"].dropna().unique().tolist())
    manu_sel  = st.selectbox("Manufacturer", manu_opts, index=0)

    state_opts = ["(All)"] + sorted(df["state"].dropna().unique().tolist())
    state_sel  = st.selectbox("State", state_opts, index=0)

    type_opts  = ["(All)"] + sorted(df["type"].dropna().unique().tolist())
    type_sel   = st.selectbox("Body type", type_opts, index=0)

# ---- Apply filters ----
mask = (
    df["price"].between(price_min, price_max)
    & df["year"].between(year_min, year_max)
    & df["odometer"].between(odo_min, odo_max)
)
if manu_sel != "(All)":
    mask &= df["manufacturer"].eq(manu_sel)
if state_sel != "(All)":
    mask &= df["state"].eq(state_sel)
if type_sel != "(All)":
    mask &= df["type"].eq(type_sel)

df_f = df.loc[mask].copy()
if df_f.empty:
    st.warning("No rows match the filters. Relax them to see data.")
    st.stop()

# optional downsample for plotting speed
df_plot = df_f.sample(sample_n, random_state=42) if len(df_f) > sample_n else df_f

st.markdown("## 📊 Exploratory Plots")
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Distributions", "Correlation", "Price vs Year", "Price vs Mileage",
    "By Manufacturer / State", "Time Trend"
])

# ---- T1: distributions ----
with t1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df_f, x="price", nbins=60, title="Price Distribution")
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_f, x="odometer", nbins=60, title="Mileage Distribution")
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---- T2: correlation heatmap ----
with t2:
    num_cols = ["price", "year", "odometer"]
    corr = df_f[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", origin="lower",
                    title="Correlation (numeric features)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Negative correlation expected between price and odometer; positive with year.")

# ---- T3: price vs year ----
with t3:
    color_col = "manufacturer" if manu_sel == "(All)" else None
    fig = px.scatter(df_plot, x="year", y="price", opacity=0.35, color=color_col,
                     title="Price vs Year",
                     hover_data=["manufacturer","model","odometer","state"])
    st.plotly_chart(fig, use_container_width=True)

# ---- T4: price vs mileage ----
with t4:
    color_col = "manufacturer" if manu_sel == "(All)" else None
    fig = px.scatter(df_plot, x="odometer", y="price", opacity=0.35, color=color_col,
                     title="Price vs Mileage",
                     hover_data=["manufacturer","model","year","state"])
    st.plotly_chart(fig, use_container_width=True)

# ---- T5: box by manufacturer & median by state ----
with t5:
    c1, c2 = st.columns(2)
    with c1:
        top_manu = df_f["manufacturer"].value_counts().head(15).index
        manu_df = df_f[df_f["manufacturer"].isin(top_manu)]
        if not manu_df.empty:
            fig = px.box(manu_df, x="manufacturer", y="price", points="outliers",
                         title="Price by Manufacturer (Top 15)")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough variety for manufacturer plot with current filters.")

    with c2:
        if "state" in df_f.columns:
            med_state = (df_f.groupby("state", as_index=False)["price"]
                           .median().sort_values("price", ascending=False).head(20))
            fig = px.bar(med_state, x="state", y="price",
                         title="Median Price by State (Top 20)")
            st.plotly_chart(fig, use_container_width=True)

# ---- T6: time trend (monthly) ----
# ---- T6: time trend (monthly) ----
with t6:
    if "posting_date" in df_f.columns:
        pd_col = df_f["posting_date"]

        if pd_col.notna().any():
            month = pd_col.dt.to_period("M")
            trend = (
                df_f.assign(month=month)
                    .groupby("month", as_index=False)["price"].median()
            )
            trend["month_ts"] = trend["month"].dt.to_timestamp()
            trend = trend.sort_values("month_ts")

            fig = px.line(trend, x="month_ts", y="price", markers=True,
                          title="Median Price Trend (by posting month)")
            fig.update_xaxes(tickangle=45, title="Month")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Time trend uses cleaned `posting_date` parsed in src/data_clean.py.")
        else:
            st.info("`posting_date` exists but contains no valid dates after cleaning.")
    else:
        st.info("No `posting_date` column found; skipping time trend.")

st.divider()
st.download_button(
    "⬇️ Download filtered dataset (CSV)",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="vehicles_filtered.csv",
    mime="text/csv",
)
