"""
models/gb.py
Gradient Boosting model family for used-car price prediction.

Defines:
- build_gb()   -> GradientBoostingRegressor
- space_gb()   -> hyperparameter search space 

- We use a single, fixed-parameter GradientBoostingRegressor that was tuned
  in earlier experiments.
"""

from __future__ import annotations
from typing import List, Optional, Dict
from sklearn.ensemble import GradientBoostingRegressor
from src.preprocess import build_model_pipeline

def build_gb(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    """
    Build a GradientBoostingRegressor pipeline.
    """
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
    return pipeline

def space_gb() -> Dict:
    """
    Hyperparameter search space for GradientBoostingRegressor.
    """
    # print("[models.gb.space_gb] Returning GB hyperparameter grid.")
    # return {
    #     "est__n_estimators": [1200, 1600],
    #     "est__learning_rate": [0.03, 0.05, 0.07],
    #     "est__max_depth": [4, 5, 6],
    #     "est__subsample": [0.8, 0.9, 1.0],
    #     "est__min_samples_leaf": [2, 5, 10],
    #     "est__min_samples_split": [10, 20, 40],
    # }
    return {}
