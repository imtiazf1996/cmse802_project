import streamlit as st
import pandas as pd

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
st.title("Car Price Predictor")

st.write("Upload a CSV to preview it. This is Fawaz")

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.dataframe(df.head(20))
    st.success(f"Rows: {len(df)} | Columns: {df.shape[1]}")
else:
    st.info("No file uploaded yet.")
