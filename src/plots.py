"""
plots.py

Diagnostic and results plots for model evaluation.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(path: Path) -> None:
    """Ensure parent directory of a file path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _default_figsize():
    return (8, 5)
def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str | Path,
    title: str = "Predicted vs Actual",
    xlabel: str = "Actual price",
    ylabel: str = "Predicted price",
) -> Path:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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
    return out_png


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str | Path,
    title: str = "Residual distribution",
    bins: int = 40,
) -> Path:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
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
    return out_png


def plot_residuals_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str | Path,
    title: str = "Residuals vs Predicted",
) -> Path:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
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
    return out_png


def plot_train_test_parity(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    out_png: str | Path = "results/plots/gbr_train_test_parity.png",
    title: str = "GBR: Predicted vs Actual (Train vs Test)",
) -> Path:

    y_train = np.asarray(y_train)
    y_train_pred = np.asarray(y_train_pred)
    y_test = np.asarray(y_test)
    y_test_pred = np.asarray(y_test_pred)
    fig, ax = plt.subplots(figsize=_default_figsize())
    ax.scatter(y_train, y_train_pred, alpha=0.3, s=10, label="Train")
    ax.scatter(y_test, y_test_pred, alpha=0.5, s=12, label="Test")

    all_true = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    lims = [
        min(np.min(all_true), np.min(all_pred)),
        max(np.max(all_true), np.max(all_pred)),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="Ideal")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(title)
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    out_png = Path(out_png)
    _ensure_dir(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png

def save_gbr_metrics_table(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    out_xlsx: str | Path = "results/plots/gbr_metrics.xlsx",
) -> Path:
    """
    Compute MAE, RMSE, R² and n for train and test, and save to an Excel file.

    """
    y_train = np.asarray(y_train)
    y_train_pred = np.asarray(y_train_pred)
    y_test = np.asarray(y_test)
    y_test_pred = np.asarray(y_test_pred)

    def _metrics(y_true, y_hat):
        residuals = y_true - y_hat
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        # R² (coefficient of determination)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return mae, rmse, r2, len(y_true)

    mae_tr, rmse_tr, r2_tr, n_tr = _metrics(y_train, y_train_pred)
    mae_te, rmse_te, r2_te, n_te = _metrics(y_test, y_test_pred)

    df = pd.DataFrame(
        data={
            "MAE": [mae_tr, mae_te],
            "RMSE": [rmse_tr, rmse_te],
            "R2": [r2_tr, r2_te],
            "n": [n_tr, n_te],
        },
        index=["train", "test"],
    )

    out_xlsx = Path(out_xlsx)
    _ensure_dir(out_xlsx)
    df.to_excel(out_xlsx, index=True)
    print(f"Saved metrics table")
    print(df)

    return out_xlsx

#Part of this was completed with the help of ChatGPT 5.1