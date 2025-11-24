"""
models/svr.py
-------------
Support Vector Regression (SVR) model for used-car price prediction.

Defines:
- build_svr()   -> full Pipeline: [preprocess (+ optional PCA)] -> SVR
- space_svr()   -> hyperparameter search grid for RandomizedSearchCV

Important Notes
---------------
- SVR is **very sensitive to feature scaling**, so we always enable
  scale_numeric=True in the preprocessing pipeline.
- SVR can be slow on very high-dimensional OHE data, but with PCA enabled
  it becomes significantly faster — this lets you show meaningful
  performance comparisons (SVR vs PCA-SVR).

Author
------
Fawaz Imtiaz
Date
----
October 2025
"""

from __future__ import annotations
from typing import List, Optional, Dict

from sklearn.svm import SVR

from src.preprocess import build_model_pipeline


# ------------------------- Pipeline builder function ----------------------- #

def build_svr(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    """
    Build a Support Vector Regression (SVR) pipeline.

    Notes
    -----
    - We MUST scale numeric features before SVR.
    - PCA is recommended for SVR when you have many one-hot categorical
      features; it drastically reduces training time.
    """
    est = SVR(kernel="rbf")   # default but strong kernel choice

    return build_model_pipeline(
        estimator=est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=True,       # <-- REQUIRED for SVR
        use_pca=use_pca,
        pca_components=pca_components,
    )


# --------------------- Hyperparameter search space ------------------------ #

def space_svr() -> Dict:
    """
    Hyperparameter search space for SVR.

    'est__' prefix matches Pipeline step naming inside build_model_pipeline().
    """
    return {
        "est__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "est__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        "est__gamma": [
            "scale",           # adaptive gamma = 1 / (n_features * X.var())
            1e-3, 3e-3, 1e-2, 3e-2, 1e-1
        ],
    }
