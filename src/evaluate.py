# src/evaluate.py
import json
from pathlib import Path
import pandas as pd
"""
evaluate.py
------------
Aggregate and compare model performance metrics.

This script reads multiple JSON metric files from the results folder
(e.g., baseline and Gradient Boosting models), combines them into a
single DataFrame, and exports a consolidated comparison CSV file.

Author
------
Fawaz Imtiaz

Date
----
October 2025

Command Line Usage
------------------
Example:
    py -m src.evaluate --out_csv results/metrics_compare.csv

Parameters (CLI)
----------------
--out_csv : str
    Output path for the combined comparison CSV file.

Outputs
-------
- A CSV summary comparing models by MAE, RMSE, and R².
"""

def main(out_csv="results/metrics_compare.csv"):
    paths = [
        ("baseline", "results/metrics_baseline.json"),
        ("gb", "results/metrics_gb.json"),
    ]
    rows = []
    for name, p in paths:
        if Path(p).exists():
            with open(p) as f:
                m = json.load(f)
            m["model"] = name
            rows.append(m)

    if not rows:
        print("No metrics found. Train models first.")
        return

    df = pd.DataFrame(rows)[["model","MAE","RMSE","R2","n_train","n_test"]]
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default="results/metrics_compare.csv")
    args = p.parse_args()
    main(args.out_csv)
