# src/models/xgb.py
from __future__ import annotations
from typing import List, Optional, Dict, Any

from xgboost import XGBRegressor
from src.preprocess import build_model_pipeline


def build_xgb(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    """
    Build an XGBoost regression pipeline:
        preprocessor -> [optional PCA] -> XGBRegressor
    """
    est = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    # Again: estimator is the FIRST positional argument
    pipe = build_model_pipeline(
        est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        use_pca=use_pca,
        pca_components=pca_components,
    )
    return pipe


def space_xgb() -> Dict[str, Any]:
    """
    Grid for XGB hyperparameters on the 'est' step.
    """
    return {
        "est__n_estimators": [300, 500],
        "est__learning_rate": [0.05, 0.1],
        "est__max_depth": [4, 6, 8],
        "est__subsample": [0.8, 1.0],
        "est__colsample_bytree": [0.8, 1.0],
        "est__reg_lambda": [1.0, 3.0],
    }
