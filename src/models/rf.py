"""
models/rf.py
------------
Random Forest model family for used-car price prediction.
"""

from __future__ import annotations
from typing import List, Optional, Dict

from sklearn.ensemble import RandomForestRegressor

from src.preprocess import build_model_pipeline


# ------------------------- Pipeline builder function ----------------------- #

def build_rf(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    """
    Build a RandomForestRegressor pipeline.
    """
    print(
        "[models.rf.build_rf] Building RandomForest pipeline "
        f"(use_pca={use_pca}, pca_components={pca_components})"
    )
    print(f"[models.rf.build_rf]   num_cols = {num_cols}")
    print(f"[models.rf.build_rf]   cat_cols = {cat_cols}")

    est = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = build_model_pipeline(
        estimator=est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=False,     # trees do not need scaling
        use_pca=use_pca,
        pca_components=pca_components,
    )

    print("[models.rf.build_rf] RandomForest pipeline constructed.")
    return pipeline


# --------------------- Hyperparameter search space ------------------------ #

def space_rf() -> Dict:
    """
    Hyperparameter search space for RandomForestRegressor.
    """
    print("[models.rf.space_rf] Returning RandomForest hyperparameter search space.")
    return {
        "est__n_estimators": [300, 500, 800, 1000],
        "est__max_depth": [None, 6, 10, 14, 18, 24],
        "est__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "est__min_samples_leaf": [1, 2, 5],
    }
