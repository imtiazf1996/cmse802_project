# src/models/lgbm.py
from __future__ import annotations
from typing import List, Optional, Dict, Any

from lightgbm import LGBMRegressor
from src.preprocess import build_model_pipeline


def build_lgbm(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    """
    Build a LightGBM regression pipeline:
        preprocessor -> [optional PCA] -> LGBMRegressor
    """
    est = LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    # IMPORTANT: estimator is the FIRST positional argument
    pipe = build_model_pipeline(
        est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        use_pca=use_pca,
        pca_components=pca_components,
    )
    return pipe


def space_lgbm() -> Dict[str, Any]:
    """
    Grid for LGBM hyperparameters on the 'est' step.
    """
    return {
        "est__n_estimators": [300, 500],
        "est__learning_rate": [0.05, 0.1],
        "est__num_leaves": [31, 63, 127],
        "est__max_depth": [-1, 7],
        "est__subsample": [0.8, 1.0],
        "est__colsample_bytree": [0.8, 1.0],
    }
