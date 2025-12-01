"""
run_experiment.py
-----------------
This is modified for GBR
- EDA
- Train/Test split
- Fit GBR
- Save metrics
- Save feature importances
- Save diagnostic plots
- Save CV logs
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src import eda as eda_mod
from src import plots as plots_mod
from src.data_clean import load_data, clean_data
from src.features import assemble_feature_frame, get_target
from src.registry import make_model, get_search_space, DEFAULT_SCORING

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _append_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _compute_regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def _export_feature_importances(model, model_dir: Path) -> None:
    """Export GBR feature importances to CSV."""
    if not hasattr(model, "named_steps"):
        return

    if "est" not in model.named_steps:
        return

    est = model.named_steps["est"]
    if not hasattr(est, "feature_importances_"):
        return

    pre = model.named_steps.get("pre", None)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(est.feature_importances_))]

    importances = np.asarray(est.feature_importances_)

    if len(importances) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    fi_df["rank"] = np.arange(1, len(fi_df) + 1)
    out_path = model_dir / "feature_importances.csv"
    fi_df.to_csv(out_path, index=False)

def run_one_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    num_cols: List[str],
    cat_cols: List[str],
    use_pca: bool,
    pca_components: Optional[int],
    random_state: int,
    results_dir: Path,
    scoring: str = DEFAULT_SCORING,
):
    pipeline = make_model(
        name,
        num_cols,
        cat_cols,
        use_pca=use_pca,
        pca_components=pca_components,
    )

    space = get_search_space(name)
    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

    if space:

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=space,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=kf, scoring=scoring)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

    else:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring=scoring)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        pipeline.fit(X_train, y_train)
        pipeline.fit(X_train, y_train)
        best_estimator = pipeline

    y_train_pred = best_estimator.predict(X_train)
    y_test_pred = best_estimator.predict(X_test)
    train_metrics = _compute_regression_metrics(y_train, y_train_pred)
    test_metrics_vals = _compute_regression_metrics(y_test, y_test_pred)

    print("[TRAIN]", train_metrics)
    print("[TEST]", test_metrics_vals)

    model_dir = results_dir / name
    _ensure_dir(model_dir)

    joblib.dump(best_estimator, model_dir / "best_model.joblib")

    train_test_table = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "test",  **test_metrics_vals},
    ])
    train_test_table.to_csv(model_dir / "train_test_metrics.csv", index=False)
    _export_feature_importances(best_estimator, model_dir)

    plots_mod.plot_parity(
        y_train,
        y_train_pred,
        out_png=str(model_dir / "parity_train.png"),
        title=f"{name.upper()} – Predicted vs Actual (train)",
    )
    plots_mod.plot_parity(
        y_test,
        y_test_pred,
        out_png=str(model_dir / "parity_test.png"),
        title=f"{name.upper()} – Predicted vs Actual (test)",
    )
    plots_mod.plot_residuals(
        y_test,
        y_test_pred,
        out_png=str(model_dir / "residuals_hist_test.png"),
        title=f"{name.upper()} – Residual distribution (test)",
    )
    plots_mod.plot_residuals_vs_pred(
        y_test,
        y_test_pred,
        out_png=str(model_dir / "residuals_vs_pred_test.png"),
        title=f"{name.upper()} – Residuals vs predicted (test)",
    )

    log_rows = [
        {
            "model": name,
            "fold": i,
            "cv_score": float(cv_scores[i]),
            "timestamp": _timestamp(),
        }
        for i in range(len(cv_scores))
    ]
    _append_csv(pd.DataFrame(log_rows), results_dir / "all_cv_runs.csv")
    return {
        "name": name,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test_mae": test_metrics_vals["MAE"],
        "test_rmse": test_metrics_vals["RMSE"],
        "test_r2":  test_metrics_vals["R2"],
    }
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="vehicles.csv")
    ap.add_argument("--models", default="gbr")
    ap.add_argument("--use_pca", action="store_true")
    ap.add_argument("--pca_components", type=int, default=None)
    ap.add_argument("--use_log_price", action="store_true")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    _ensure_dir(results_dir)

    raw = load_data(args.input)
    df = clean_data(raw)
    df["manufacturer_price_mean"] = df.groupby("manufacturer")["price"].transform("mean")
    df["state_price_mean"] = df.groupby("state")["price"].transform("mean")
    _ensure_dir(Path("results/eda"))
    out_eda = Path("results/eda")

    dist_figs = eda_mod.plot_distributions(df)
    for name, fig in dist_figs.items():
        fig.write_html(out_eda / f"{name}.html")

    corr_fig = eda_mod.plot_correlation(df)
    if corr_fig:
        corr_fig.write_html(out_eda / "correlation_heatmap.html")

    trend_figs = eda_mod.plot_trends(df)
    for name, fig in trend_figs.items():
        fig.write_html(out_eda / f"{name}.html")

    X_all, num_cols, cat_cols = assemble_feature_frame(df, include_engineered=True)
    y_all = get_target(df, use_log_price=args.use_log_price)
    work = pd.concat([X_all, y_all.rename("__target__")], axis=1).dropna()
    X_all = work.drop(columns=["__target__"])
    y_all = work["__target__"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    run_one_model(
        name="gbr",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_cols=num_cols,
        cat_cols=cat_cols,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        random_state=args.random_state,
        results_dir=results_dir,
    )

    print("Completed successfully.")
if __name__ == "__main__":
    main()
# This was completed with the help of ChatGPT 5.1