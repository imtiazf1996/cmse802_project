# src/train_ml.py
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_clean import load_data, clean_data
"""
train_ml.py
-----------
Train and evaluate an advanced Gradient Boosting model.

This script implements a Gradient Boosting Regressor (GBR) to improve
over the baseline linear regression. It loads the cleaned data,
encodes categorical features, splits into training and testing sets,
fits the GBR model, and saves the trained model and performance metrics.

Author
------
Fawaz Imtiaz

Date
----
October 2025

Command Line Usage
------------------
Example:
    py -m src.train_ml --input vehicles.csv --model_out results/model_gb.pkl --metrics_out results/metrics_gb.json

Parameters (CLI)
----------------
--input : str
    Path to the input CSV file.
--model_out : str
    Output path for the trained Gradient Boosting model (.pkl).
--metrics_out : str
    Output path for the metrics file (.json).

Outputs
-------
- Trained Gradient Boosting model (Pickle file)
- Model metrics (JSON file) including MAE, RMSE, R²
"""

def main(input_csv="vehicles.csv", model_out="results/model_gb.pkl", metrics_out="results/metrics_gb.json", test_size=0.2, random_state=42):
    df = clean_data(load_data(input_csv))

    target = "price"
    num_cols = [c for c in ["year", "odometer"] if c in df.columns]
    cat_cols = [c for c in ["manufacturer","model","condition","fuel","title_status","transmission","drive","type","state"] if c in df.columns]

    X = df[num_cols + cat_cols]
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("passthrough", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    model = Pipeline(steps=[
        ("pre", pre),
        ("gb", GradientBoostingRegressor(random_state=int(random_state)))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    joblib.dump(model, model_out)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved GB model → {model_out}")
    print(f"Saved metrics → {metrics_out}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="vehicles.csv")
    p.add_argument("--model_out", default="results/model_gb.pkl")
    p.add_argument("--metrics_out", default="results/metrics_gb.json")
    p.add_argument("--test_size", default=0.2)
    p.add_argument("--random_state", default=42)
    args = p.parse_args()
    main(args.input, args.model_out, args.metrics_out, args.test_size, args.random_state)
