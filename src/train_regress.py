
# src/train_regress.py
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_clean import load_data, clean_data
"""
train_regress.py
----------------
Train and evaluate a baseline Linear Regression model.

This script loads cleaned vehicle data, performs feature encoding
and scaling, splits the data into train/test sets, fits a Linear
Regression model, and saves both the trained model and evaluation
metrics.

Author
------
Fawaz Imtiaz

Date
----
October 2025

Command Line Usage
------------------
Example:
    py -m src.train_regress --input vehicles.csv --model_out results/model_baseline.pkl --metrics_out results/metrics_baseline.json

Parameters (CLI)
----------------
--input : str
    Path to the input CSV file.
--model_out : str
    Output path for the trained model (.pkl).
--metrics_out : str
    Output path for the metrics file (.json).

Outputs
-------
- Trained Linear Regression model (Pickle file)
- Model metrics (JSON file) including MAE, RMSE, R²
"""

def main(input_csv="vehicles.csv", model_out="results/model_baseline.pkl", metrics_out="results/metrics_baseline.json", test_size=0.2, random_state=42):
    # 1) load & clean
    df = clean_data(load_data(input_csv))

    # 2) features/target
    target = "price"
    num_cols = [c for c in ["year", "odometer"] if c in df.columns]
    cat_cols = [c for c in ["manufacturer","model","condition","fuel","title_status","transmission","drive","type","state"] if c in df.columns]

    X = df[num_cols + cat_cols]
    y = df[target].values

    # 3) split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))

    # 4) pipeline
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
        ("reg", LinearRegression())
    ])

    # 5) train
    model.fit(X_train, y_train)

    # 6) eval
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    # 7) save
    joblib.dump(model, model_out)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved baseline model → {model_out}")
    print(f"Saved metrics → {metrics_out}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="vehicles.csv")
    p.add_argument("--model_out", default="results/model_baseline.pkl")
    p.add_argument("--metrics_out", default="results/metrics_baseline.json")
    p.add_argument("--test_size", default=0.2)
    p.add_argument("--random_state", default=42)
    args = p.parse_args()
    main(args.input, args.model_out, args.metrics_out, args.test_size, args.random_state)
