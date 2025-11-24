"""
models/mlp.py
-------------
Multi-Layer Perceptron (Neural Network) model for used-car price prediction.

Defines:
- build_mlp()  -> Pipeline: [preprocess (+ optional PCA)] -> MLPRegressor
- space_mlp()  -> hyperparameter search space for RandomizedSearchCV

Author
------
Fawaz Imtiaz
Date
----
October 2025
"""

from __future__ import annotations
from typing import List, Optional, Dict

from sklearn.neural_network import MLPRegressor

from src.preprocess import build_model_pipeline


# ------------------------- Pipeline builder function ----------------------- #

def build_mlp(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    """
    Build an MLPRegressor pipeline.

    Notes
    -----
    - MLP requires scaled numeric features.
    - PCA can sometimes help stabilize optimization.
    """
    print(
        "[models.mlp.build_mlp] Building MLPRegressor pipeline "
        f"(use_pca={use_pca}, pca_components={pca_components})"
    )
    print(f"[models.mlp.build_mlp]   num_cols = {num_cols}")
    print(f"[models.mlp.build_mlp]   cat_cols = {cat_cols}")

    est = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=600,
        random_state=42,
    )

    pipeline = build_model_pipeline(
        estimator=est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=True,          # IMPORTANT for MLP
        use_pca=use_pca,
        pca_components=pca_components,
    )

    print("[models.mlp.build_mlp] MLPRegressor pipeline constructed.")
    return pipeline


# --------------------- Hyperparameter search space ------------------------ #

def space_mlp() -> Dict:
    """
    Hyperparameter search space for MLPRegressor.

    Uses the 'est__' prefix since the estimator step in the pipeline is named 'est'.
    """
    print("[models.mlp.space_mlp] Returning MLP hyperparameter search space.")
    return {
        "est__hidden_layer_sizes": [
            (64, 32),
            (128, 64),
            (256, 128),
            (128, 128, 64),
        ],
        "est__alpha": [0.0001, 0.0005, 0.001, 0.005],
        "est__learning_rate_init": [0.0005, 0.001, 0.003],
        "est__max_iter": [400, 600, 900],
    }
