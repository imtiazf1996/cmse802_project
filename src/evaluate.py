# src/evaluate.py
import json
from pathlib import Path
import pandas as pd

"""
evaluate.py
-----------
Aggregate and compare model performance metrics (new pipeline version).

For the final project, if you only keep GradientBoostingRegressor,
this will still work and just produce a single-row summary for 'gbr'.
"""

def collect_model_metrics(results_dir: Path) -> pd.DataFrame:
    """
    Walk through results_dir, find per-model JSONs, and assemble a summary DataFrame.
    Expects per-model folders like: results/gbr/test_metrics.json, cv_summary.json
    """
    print(f"[evaluate.collect] Scanning results directory: {results_dir}")
    rows = []

    # Iterate through model subdirectories (e.g. gbr/, rf/, etc.)
    for sub in results_dir.iterdir():
        if not sub.is_dir():
            continue

        model_name = sub.name
        print(f"[evaluate.collect] Found directory: {model_name}")

        test_metrics_path = sub / "test_metrics.json"
        cv_summary_path = sub / "cv_summary.json"

        if not test_metrics_path.exists():
            print(f"[evaluate.collect]   No test_metrics.json found → skipping {model_name}")
            continue

        # Load test metrics
        print(f"[evaluate.collect]   Loading test_metrics.json for {model_name}")
        with open(test_metrics_path, "r") as f:
            test = json.load(f)

        # Try loading CV summary (optional)
        cv_mean = cv_std = scoring = None

        if cv_summary_path.exists():
            print(f"[evaluate.collect]   Loading cv_summary.json for {model_name}")
            with open(cv_summary_path, "r") as f:
                cv = json.load(f)
            cv_mean = cv.get("cv_mean_score")
            cv_std = cv.get("cv_std_score")
            scoring = cv.get("scoring")
        else:
            print(f"[evaluate.collect]   No CV summary for {model_name}")

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
        print("[evaluate.collect] No models found with test_metrics.json")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by RMSE if available (for regression)
    if "test_RMSE" in df.columns and df["test_RMSE"].notna().any():
        print("[evaluate.collect] Sorting models by test_RMSE...")
        df = df.sort_values(by="test_RMSE", ascending=True)

    print(f"[evaluate.collect] Collected metrics for {len(df)} models.")
    return df


def main(results_dir="results", out_csv="results/metrics_compare.csv"):
    print(f"[evaluate.main] Starting evaluation...")
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"[evaluate.main] ERROR: No results directory at {results_dir}")
        print("                 Run experiments first with run_experiment.py")
        return

    print(f"[evaluate.main] Reading model metrics from: {results_dir}")
    df = collect_model_metrics(results_dir)

    if df.empty:
        print("[evaluate.main] No metrics found. Did you run run_experiment.py?")
        return

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate.main] Writing combined metrics CSV → {out_csv}")
    df.to_csv(out_csv, index=False)

    print("[evaluate.main] Final metrics summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_csv", default="results/metrics_compare.csv")
    args = p.parse_args()
    main(args.results_dir, args.out_csv)
