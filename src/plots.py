"""
plots.py
--------
Reusable plotting utilities for model performance visualization.

This module focuses on:
- Comparing models on a common metric (leaderboard plots).
- Visualizing CV behavior from `results/all_cv_runs.csv`.
- Generic helpers for parity (y_true vs y_pred) and residual plots.

These functions are designed to be called from:
- Notebooks
- Small driver scripts
- Streamlit app (`app.py`) if desired

Author
------
Fawaz Imtiaz
Date
----
October 2025
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Styling helpers ----------------------------- #

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _default_figsize():
    return (8, 5)


# -------------------------- Leaderboard-level plots ------------------------ #

def plot_leaderboard_rmse(
    leaderboard_csv: str | Path = "results/leaderboard_cv.csv",
    out_png: str | Path = "results/plots/leaderboard_rmse.png",
    title: str = "Model comparison (test RMSE)",
) -> Optional[Path]:
    """
    Bar plot of test RMSE per model from leaderboard CSV (sorted ascending).

    Parameters
    ----------
    leaderboard_csv : str or Path
        Path to CSV created by run_experiment.py (build_leaderboard).
    out_png : str or Path
        Output path for the PNG file.
    title : str
        Plot title.

    Returns
    -------
    Path of saved PNG, or None if input is missing/empty.
    """
    leaderboard_csv = Path(leaderboard_csv)
    print(f"[plot_leaderboard_rmse] Using leaderboard CSV: {leaderboard_csv}")

    if not leaderboard_csv.exists():
        print(f"[plot_leaderboard_rmse] No leaderboard file at {leaderboard_csv}")
        return None

    df = pd.read_csv(leaderboard_csv)
    if df.empty:
        print("[plot_leaderboard_rmse] Leaderboard is empty.")
        return None

    print(f"[plot_leaderboard_rmse] Loaded {len(df)} rows. Sorting by test_RMSE...")
    df = df.sort_values(by="test_RMSE", ascending=True)

    fig, ax = plt.subplots(figsize=_default_figsize())
    ax.bar(df["model"], df["test_RMSE"])
    ax.set_ylabel("Test RMSE")
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(df["test_RMSE"]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, rotation=0)

    out_png = Path(out_png)
    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot_leaderboard_rmse] Saved → {out_png}")
    return out_png


def plot_cv_boxplot(
    all_cv_runs_csv: str | Path = "results/all_cv_runs.csv",
    out_png: str | Path = "results/plots/cv_boxplot.png",
    title: str = "Cross-validation scores by model",
) -> Optional[Path]:
    """
    Boxplot of CV scores by model from `results/all_cv_runs.csv`.

    This uses the `cv_score` column logged in run_experiment.py,
    where scores are in the metric space of DEFAULT_SCORING
    (neg_root_mean_squared_error by default).

    Parameters
    ----------
    all_cv_runs_csv : str or Path
        Long-form CSV log from run_experiment.py.
    out_png : str or Path
        Output path.
    title : str
        Plot title.

    Returns
    -------
    Path or None.
    """
    all_cv_runs_csv = Path(all_cv_runs_csv)
    print(f"[plot_cv_boxplot] Using CV log CSV: {all_cv_runs_csv}")

    if not all_cv_runs_csv.exists():
        print(f"[plot_cv_boxplot] No CV log at {all_cv_runs_csv}")
        return None

    df = pd.read_csv(all_cv_runs_csv)
    if df.empty:
        print("[plot_cv_boxplot] CV log is empty.")
        return None

    print(f"[plot_cv_boxplot] Loaded {len(df)} rows of CV results.")

    # Convert negative scores back to positive RMSE if using neg_root_mean_squared_error
    scores = df["cv_score"].copy()
    if (scores < 0).all():
        scores = -scores
        ylabel = "CV score (approx. RMSE)"
        print("[plot_cv_boxplot] Detected all-negative scores; flipping sign to show RMSE.")
    else:
        ylabel = "CV score"

    fig, ax = plt.subplots(figsize=_default_figsize())
    df_plot = pd.DataFrame({"model": df["model"], "score": scores})
    df_plot.boxplot(column="score", by="model", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    plt.suptitle("")  # remove default pandas boxplot title
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    out_png = Path(out_png)
    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot_cv_boxplot] Saved → {out_png}")
    return out_png


# ---------------------------- Per-model diagnostics ------------------------ #

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str | Path,
    title: str = "Predicted vs Actual",
    xlabel: str = "Actual price",
    ylabel: str = "Predicted price",
) -> Path:
    """
    Scatter plot of predictions vs ground truth (parity plot).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted targets.
    out_png : str or Path
        Output PNG path.
    title, xlabel, ylabel : str
        Labels for the plot.

    Returns
    -------
    Path of saved PNG.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    print(f"[plot_parity] Creating parity plot for {len(y_true)} samples.")

    fig, ax = plt.subplots(figsize=_default_figsize())
    ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    lims = [
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    ]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)

    out_png = Path(out_png)
    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot_parity] Saved → {out_png}")
    return out_png


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str | Path,
    title: str = "Residual distribution",
    bins: int = 40,
) -> Path:
    """
    Plot histogram of residuals (y_true - y_pred).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted targets.
    out_png : str or Path
        Output PNG path.
    title : str
        Plot title.
    bins : int
        Number of histogram bins.

    Returns
    -------
    Path of saved PNG.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    print(f"[plot_residuals] Plotting residual histogram for {len(residuals)} samples with {bins} bins.")

    fig, ax = plt.subplots(figsize=_default_figsize())
    ax.hist(residuals, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    out_png = Path(out_png)
    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot_residuals] Saved → {out_png}")
    return out_png


def plot_residuals_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str | Path,
    title: str = "Residuals vs Predicted",
) -> Path:
    """
    Scatter plot of residuals vs predicted values (diagnostics for heteroscedasticity).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted targets.
    out_png : str or Path
        Output PNG path.
    title : str
        Plot title.

    Returns
    -------
    Path of saved PNG.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    print(f"[plot_residuals_vs_pred] Plotting residuals vs predicted for {len(residuals)} samples.")

    fig, ax = plt.subplots(figsize=_default_figsize())
    ax.scatter(y_pred, residuals, alpha=0.4, s=10)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.grid(True, linestyle="--", alpha=0.5)

    out_png = Path(out_png)
    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot_residuals_vs_pred] Saved → {out_png}")
    return out_png
