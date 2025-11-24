"""
models/gb.py
------------
Gradient Boosting model family for used-car price prediction.

Defines:
- build_gb()   -> Pipeline: [preprocess (+ optional PCA)] -> GradientBoostingRegressor
- space_gb()   -> hyperparameter search space for RandomizedSearchCV
"""

from __future__ import annotations
from typing import List, Optional, Dict

from sklearn.ensemble import GradientBoostingRegressor

from src.preprocess import build_model_pipeline


# ------------------------- Pipeline builder function ----------------------- #

def build_gb(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    """
    Build a GradientBoostingRegressor pipeline.

    Notes
    -----
    - Tree-based boosting does not require standardized numeric features.
    - PCA is optional and typically not necessary, but you can experiment
      with it for dimensionality reduction.
    """
    print(
        "[models.gb.build_gb] Building GradientBoosting pipeline "
        f"(use_pca={use_pca}, pca_components={pca_components})"
    )
    print(f"[models.gb.build_gb]   num_cols = {num_cols}")
    print(f"[models.gb.build_gb]   cat_cols = {cat_cols}")

    est = GradientBoostingRegressor(
        n_estimators=1200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42,
    )

    pipeline = build_model_pipeline(
        estimator=est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=False,
        use_pca=use_pca,
        pca_components=pca_components,
    )

    print("[models.gb.build_gb] GradientBoosting pipeline constructed.")
    return pipeline


# --------------------- Hyperparameter search space ------------------------ #

def space_gb() -> Dict:
    """
    Hyperparameter *grid* for GradientBoostingRegressor.
    Sized for GridSearchCV (≈ 4*4*3*3 = 144 combos).
    """
    #print("[models.gb.space_gb] Returning GB hyperparameter grid.")
    #return {
    #    "est__n_estimators": [300, 600, 900, 1200],      # 4 values
    #   "est__learning_rate": [0.03, 0.05, 0.1, 0.2],    # 4 values (drop 0.01)
    #    "est__max_depth": [2, 3, 4],                     # 3 values (drop 5,6)
    #    "est__subsample": [0.6, 0.8, 1.0],               # 3 values
    #}
    print("[models.gb.space_gb] Returning EMPTY GB hyperparameter space (using fixed best params).")
    return {}