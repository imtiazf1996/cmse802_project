"""
run_experiment.py
-----------------
Single entry-point to train/evaluate one or many models with shared CV,
hyperparameter search, consistent preprocessing, and unified logging.

Outputs
-------
results/
  all_cv_runs.csv        # long-form per-fold CV log for every model
  leaderboard_cv.csv     # mean±std across folds per model (CV metric)
  <model>/
    best_model.joblib    # refit on full train (after CV selection)
    cv_summary.json      # CV mean/std and best params
    test_metrics.json    # MAE, RMSE, R2 on held-out test set

Usage
-----
py -m src.run_experiment --input vehicles.csv --models linear,rf,gbr,svr,mlp
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted

from src.evaluate import main as eval_main

from src import eda as eda_mod
from src import plots as plots_mod
from src.data_clean import load_data, clean_data
from src.features import assemble_feature_frame, get_target
from src.registry import (
    get_model_names, make_model, get_search_space,
    DEFAULT_SCORING, DEFAULT_CV_FOLDS
)


# ------------------------------- IO helpers -------------------------------- #

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_json(obj: Dict[str, Any], path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _append_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


# --------------------------- Experiment procedure -------------------------- #

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    cv_splits: int,
    random_state: int,
    n_iter: int,
    results_dir: Path,
    scoring: str = DEFAULT_SCORING,
) -> Dict[str, Any]:
    """Train/tune a single model, log CV results, save artifacts, and test metrics."""
    print(f"\n[RUN] ===== Starting model: {name} =====")
    print(f"[RUN] Building pipeline and search space for: {name}")

    # Build pipeline and search space
    pipeline = make_model(name, num_cols, cat_cols, use_pca=use_pca, pca_components=pca_components)
    space = get_search_space(name)

    print(f"[CV] Using KFold with {cv_splits} splits (shuffle=True, random_state={random_state})")
    kf = KFold(n_splits=3, shuffle=True, random_state=random_state) #n_splits=cv_splits Fawaz

    # If there is nothing to tune, just cross-validate once and fit on train
    if not space:
        print(f"[INFO] Model '{name}' has no hyperparameter search space. Running simple CV...")
        print(f"[CV] cross_val_score for model '{name}' started...")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring=scoring, n_jobs=None)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        print(f"[CV] cross_val_score for model '{name}' completed. Mean={cv_mean:.4f}, Std={cv_std:.4f}")

        # Fit best (here just pipeline) on full train
        print(f"[FIT] Fitting final model '{name}' on full training set...")
        pipeline.fit(X_train, y_train)

        best_estimator = pipeline
        best_params = {}
        split_scores = {f"split{i}_score": float(s) for i, s in enumerate(cv_scores)}
    else:
    # If you want: use GridSearch for gbr, RandomizedSearch for others
        if name == "gbr":
            print(f"[CV] Starting GridSearchCV for model '{name}' ...")
            print(f"[CV] Grid size for '{name}' = {len(space['est__n_estimators'])} * "
              f"{len(space['est__learning_rate'])} * "
              f"{len(space['est__max_depth'])} * "
              f"{len(space['est__subsample'])} combinations.")
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=space,
                scoring=scoring,
                cv=kf,
                refit=True,
                n_jobs=None,
                verbose=1,
                return_train_score=False,
        )
        else:
            print(f"[CV] Starting RandomizedSearchCV for model '{name}' with n_iter={n_iter}...")
            print(f"[CV] Search space for '{name}' has {len(space)} hyperparameters.")
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=space,
                n_iter=3,  # or int(n_iter)
                scoring=scoring,
                cv=kf,
                refit=True,
                random_state=random_state,
                verbose=1,
                n_jobs=None,
                return_train_score=False,
                )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        best_params = search.best_params_
        cv_mean = float(search.best_score_)
        # Pull per-split test scores of the best params from cv_results_
        split_scores = {}
        res = search.cv_results_
        best_idx = search.best_index_
        for i in range(cv_splits):
            key = f"split{i}_test_score"
            if key in res:
                split_scores[f"split{i}_score"] = float(res[key][best_idx])
        cv_std = float(np.std(list(split_scores.values()))) if split_scores else float(np.nan)
        print(f"[CV] RandomizedSearchCV complete for '{name}'. Best score={cv_mean:.4f}")
        print(f"[CV] Best params for '{name}': {best_params}")

    # Evaluate on held-out test set
    print(f"[TEST] Predicting on test set for model '{name}'...")
    y_pred = best_estimator.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2 = float(r2_score(y_test, y_pred))
    print(f"[TEST] Test MAE={test_mae:.4f} RMSE={test_rmse:.4f} R2={test_r2:.4f} for '{name}'")

    # Persist artifacts
    model_dir = results_dir / name
    _ensure_dir(model_dir)
    print(f"[SAVE] Saving best estimator for '{name}' to {model_dir} ...")
    joblib.dump(best_estimator, model_dir / "best_model.joblib")

    cv_summary = {
        "model": name,
        "scoring": scoring,
        "cv_splits": int(cv_splits),
        "cv_mean_score": float(cv_mean),
        "cv_std_score": float(cv_std),
        "best_params": best_params,
        "timestamp": _timestamp(),
        "use_pca": bool(use_pca),
        "pca_components": int(pca_components) if pca_components is not None else None,
    }
    _write_json(cv_summary, model_dir / "cv_summary.json")

    test_metrics = {
        "model": name,
        "MAE": test_mae,
        "RMSE": test_rmse,
        "R2": test_r2,
        "n_test": int(len(y_test)),
        "timestamp": _timestamp(),
    }
    _write_json(test_metrics, model_dir / "test_metrics.json")

    print(f"[SAVE] Saved model + CV summary + test metrics for '{name}'.")

    # Long-form CV log (per split) for leaderboard & plots
    print(f"[LOG] Writing detailed CV logs for '{name}'...")
    cv_log_rows = []
    if split_scores:
        for k, v in split_scores.items():
            cv_log_rows.append({
                "timestamp": _timestamp(),
                "model": name,
                "fold": int(k.replace("split", "").replace("_score", "")),
                "cv_score": float(v),
                "scoring": scoring,
                "use_pca": bool(use_pca),
                "pca_components": int(pca_components) if pca_components is not None else None,
            })
    else:
        # If we couldn't recover per-split, at least log the mean
        cv_log_rows.append({
            "timestamp": _timestamp(),
            "model": name,
            "fold": -1,
            "cv_score": float(cv_mean),
            "scoring": scoring,
            "use_pca": bool(use_pca),
            "pca_components": int(pca_components) if pca_components is not None else None,
        })
    _append_csv(pd.DataFrame(cv_log_rows), results_dir / "all_cv_runs.csv")
    print(f"[LOG] CV logs appended to all_cv_runs.csv for '{name}'.")

    print(f"[END] ===== Completed model: {name} =====\n")

    return {
        "name": name,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_params": best_params,
    }


def build_leaderboard(results: List[Dict[str, Any]], out_csv: Path) -> pd.DataFrame:
    """Write a compact leaderboard CSV and return it."""
    rows = []
    for r in results:
        rows.append({
            "model": r["name"],
            "cv_score_mean": r["cv_mean"],
            "cv_score_std": r["cv_std"],
            "test_RMSE": r["test_rmse"],
            "test_MAE": r["test_mae"],
            "test_R2": r["test_r2"],
        })
    df = pd.DataFrame(rows)
    # Sort by primary test metric (RMSE ascending)
    df = df.sort_values(by="test_RMSE", ascending=True)
    _ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    return df

##EDA Section

def _run_and_save_eda(df, out_dir: str = "results/eda") -> None:
    """
    Run EDA using src.eda and save all figures as HTML files.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned vehicle dataset.
    out_dir : str
        Directory to store EDA outputs (HTML).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[EDA] Running EDA and saving outputs to {out_path} ...")

    # 1) Distributions
    dist_figs = eda_mod.plot_distributions(df)
    for name, fig in dist_figs.items():
        fig.write_html(out_path / f"{name}.html")

    # 2) Correlation heatmap
    corr_fig = eda_mod.plot_correlation(df)
    if corr_fig is not None:
        corr_fig.write_html(out_path / "correlation_heatmap.html")

    # 3) Trends
    trend_figs = eda_mod.plot_trends(df)
    for name, fig in trend_figs.items():
        fig.write_html(out_path / f"{name}.html")

    print(f"[EDA] Saved EDA plots under {out_path}")

# ----------------------------------- CLI ----------------------------------- #

##performance Evaluation plots section


def _run_and_save_performance_plots(results_dir: str = "results") -> None:
    res_dir = Path(results_dir)
    cv_csv = res_dir / "all_cv_runs.csv"
    leader_csv = res_dir / "leaderboard_cv.csv"

    if not cv_csv.exists() or not leader_csv.exists():
        print("[PLOTS] Skipping performance plots; required CSVs not found yet.")
        return

    plots_out = res_dir / "plots"
    plots_out.mkdir(parents=True, exist_ok=True)

    print(f"[PLOTS] Generating performance plots into {plots_out} ...")

    plots_mod.plot_leaderboard_rmse(
        leaderboard_csv=str(leader_csv),
        out_png=str(plots_out / "leaderboard_rmse.png"),  # already fixed from .html → .png
    )

    plots_mod.plot_cv_boxplot(
        all_cv_runs_csv=str(cv_csv),                      # 🔧 fixed name
        out_png=str(plots_out / "cv_boxplot.png"),
    )

    print(f"[PLOTS] Saved performance plots under {plots_out}")

def main():
    print("\n[MAIN] Starting run_experiment.py ...")
    print(f"[MAIN] Timestamp: {_timestamp()}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="vehicles.csv", help="Path to raw CSV")
    ap.add_argument("--models", default="all", help="Comma-separated list or 'all'")
    ap.add_argument("--use_pca", action="store_true", help="Attach a PCA stage")
    ap.add_argument("--pca_components", type=int, default=None, help="PCA components when --use_pca")
    ap.add_argument("--use_log_price", action="store_true", help="Train on log1p(price); report test metrics in original space")
    ap.add_argument("--cv", type=int, default=DEFAULT_CV_FOLDS, help="CV folds")
    ap.add_argument("--n_iter", type=int, default=20, help="RandomizedSearch iterations (per model)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Hold-out test size fraction")
    ap.add_argument("--random_state", type=int, default=42, help="RNG seed")
    ap.add_argument("--results_dir", default="results", help="Output directory for artifacts")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    _ensure_dir(results_dir)

    # ------------------ Load & prepare data ------------------ #
    print("[DATA] Loading raw data...")
    raw = load_data(args.input)
    print("[DATA] Cleaning data...")
    df = clean_data(raw)
    ###FAWAZ###
    print("[DATA] Adding manufacturer/state mean price features...")
    df["manufacturer_price_mean"] = (
    df.groupby("manufacturer")["price"].transform("mean")
    )
    df["state_price_mean"] = (
    df.groupby("state")["price"].transform("mean")
    )

    _run_and_save_eda(df, out_dir="results/eda")
    print("[EDA] Completed EDA generation.")


    # Assemble X with engineered features and get y
    print("[DATA] Assembling feature frame and target...")
    X_all, num_cols, cat_cols = assemble_feature_frame(df, include_engineered=True)
    y_all = get_target(df.assign(price_log=np.log1p(df["price"])), use_log_price=args.use_log_price) \
            if args.use_log_price else get_target(df, use_log_price=False)

    # Align X and y (drop rows with NA in features/target)
    print("[DATA] Aligning X and y, dropping NA rows...")
    work = pd.concat([X_all, y_all.rename("__target__")], axis=1).dropna()
    X_all = work.drop(columns=["__target__"])
    y_all = work["__target__"].to_numpy()

    # Split once; models share identical hold-out test set
    print("[DATA] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.random_state
    )

    # ------------------ Select models to run ------------------ #
    if args.models.strip().lower() == "all":
        model_names = get_model_names()
    else:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        available = set(get_model_names())
        unknown = [m for m in requested if m not in available]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Available: {sorted(available)}")
        model_names = requested

    print(f"[INFO] Running models: {model_names}")
    print(f"[INFO] CV folds={args.cv}, scoring={DEFAULT_SCORING}, n_iter={args.n_iter}")
    if args.use_pca:
        print(f"[INFO] PCA enabled with components={args.pca_components}")

    # ------------------ Train/evaluate each ------------------ #
    print("\n[TRAIN] Beginning model loop...\n")
    results = []
    for name in model_names:
        out = run_one_model(
            name=name,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            num_cols=num_cols, cat_cols=cat_cols,
            use_pca=bool(args.use_pca),
            pca_components=args.pca_components,
            cv_splits=args.cv,
            random_state=args.random_state,
            n_iter=args.n_iter,
            results_dir=results_dir,
            scoring=DEFAULT_SCORING,
        )
        results.append(out)

    print("\n[TRAIN] All models finished training.")

    # ------------------ Leaderboard ------------------ #
    print("[LEADERBOARD] Building leaderboard...")
    leaderboard = build_leaderboard(results, results_dir / "leaderboard_cv.csv")
    print("[LEADERBOARD] Leaderboard saved to leaderboard_cv.csv")

    print("\n=== Leaderboard (sorted by test RMSE) ===")
    print(leaderboard.to_string(index=False))

    # ------------------ Plots & evaluation ------------------ #
    print("[PLOTS] Generating performance plots...")
    _run_and_save_performance_plots(results_dir=args.results_dir)

    print("[EVAL] Running evaluation comparison across models...")
    eval_out_csv = results_dir / "metrics_compare.csv"
    eval_main(results_dir=str(results_dir), out_csv=str(eval_out_csv))
    print(f"[EVAL] Metrics comparison written to {eval_out_csv}")

    print("\n[MAIN] run_experiment.py completed successfully.")
    print("[MAIN] All artifacts saved under:", results_dir)

if __name__ == "__main__":
    main()
