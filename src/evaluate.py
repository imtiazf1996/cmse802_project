# src/evaluate.py
import json
from pathlib import Path
import pandas as pd

"""
evaluate.py

Aggregate and compare model performance metrics (new pipeline version).

For the final project, if you only keep GradientBoostingRegressor,
this will still work and just produce a single-row summary for 'gbr'.
"""

def collect_model_metrics(results_dir: Path) -> pd.DataFrame:
    """
    Walk through results_dir, find per-model JSONs, and assemble a summary DataFrame.
    Expects per-model folders like: results/gbr/test_metrics.json, cv_summary.json
    """
    rows = []

    # Iterate through model subdirectories (e.g. gbr/, rf/, etc.)
    for sub in results_dir.iterdir():
        if not sub.is_dir():
            continue

        model_name = sub.name

        test_metrics_path = sub / "test_metrics.json"
        cv_summary_path = sub / "cv_summary.json"

        if not test_metrics_path.exists():
            continue

        # Load test metrics
        with open(test_metrics_path, "r") as f:
            test = json.load(f)

        cv_mean = cv_std = scoring = None

        if cv_summary_path.exists():
            with open(cv_summary_path, "r") as f:
                cv = json.load(f)
            cv_mean = cv.get("cv_mean_score")
            cv_std = cv.get("cv_std_score")
            scoring = cv.get("scoring")

        rows.append({
            "model": model_name,
            "cv_score_mean": cv_mean,
            "cv_score_std": cv_std,
            "cv_scoring": scoring,
            "test_MAE": test.get("MAE"),
            "test_RMSE": test.get("RMSE"),
            "test_R2": test.get("R2"),
            "n_test": test.get("n_test"),
            "timestamp": test.get("timestamp"),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Sort by RMSE
    if "test_RMSE" in df.columns and df["test_RMSE"].notna().any():
        print("[evaluate.collect] Sorting models by test_RMSE...")
        df = df.sort_values(by="test_RMSE", ascending=True)
    return df


def main(results_dir="results", out_csv="results/metrics_compare.csv"):
    results_dir = Path(results_dir)

    if not results_dir.exists():
        return
    df = collect_model_metrics(results_dir)

    if df.empty:
        return

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_csv", default="results/metrics_compare.csv")
    args = p.parse_args()
    main(args.results_dir, args.out_csv)
