"""
models/linear.py
----------------
Linear-model family for used-car price prediction.
"""

from __future__ import annotations
from typing import List, Optional, Dict

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from src.preprocess import build_model_pipeline


# ------------------------- Pipeline builder functions ---------------------- #

def build_linear(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    print(
        "[models.linear.build_linear] Building Linear Regression pipeline "
        f"(use_pca={use_pca}, pca_components={pca_components})"
    )
    print(f"[models.linear.build_linear]   num_cols = {num_cols}")
    print(f"[models.linear.build_linear]   cat_cols = {cat_cols}")

    est = LinearRegression()

    pipeline = build_model_pipeline(
        estimator=est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=False,
        use_pca=use_pca,
        pca_components=pca_components,
    )
    print("[models.linear.build_linear] Linear Regression pipeline constructed.")
    return pipeline


def build_ridge(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    print(
        "[models.linear.build_ridge] Building Ridge Regression pipeline "
        f"(use_pca={use_pca}, pca_components={pca_components})"
    )
    print(f"[models.linear.build_ridge]   num_cols = {num_cols}")
    print(f"[models.linear.build_ridge]   cat_cols = {cat_cols}")

    est = Ridge(random_state=42)

    pipeline = build_model_pipeline(
        estimator=est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=True,
        use_pca=use_pca,
        pca_components=pca_components,
    )
    print("[models.linear.build_ridge] Ridge pipeline constructed.")
    return pipeline


def build_lasso(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    print(
        "[models.linear.build_lasso] Building Lasso Regression pipeline "
        f"(use_pca={use_pca}, pca_components={pca_components})"
    )
    print(f"[models.linear.build_lasso]   num_cols = {num_cols}")
    print(f"[models.linear.build_lasso]   cat_cols = {cat_cols}")

    est = Lasso(random_state=42, max_iter=2000)

    pipeline = build_model_pipeline(
        estimator=est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=True,
        use_pca=use_pca,
        pca_components=pca_components,
    )
    print("[models.linear.build_lasso] Lasso pipeline constructed.")
    return pipeline


def build_elastic(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    print(
        "[models.linear.build_elastic] Building ElasticNet pipeline "
        f"(use_pca={use_pca}, pca_components={pca_components})"
    )
    print(f"[models.linear.build_elastic]   num_cols = {num_cols}")
    print(f"[models.linear.build_elastic]   cat_cols = {cat_cols}")

    est = ElasticNet(random_state=42, max_iter=3000)

    pipeline = build_model_pipeline(
        estimator=est,
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=True,
        use_pca=use_pca,
        pca_components=pca_components,
    )
    print("[models.linear.build_elastic] ElasticNet pipeline constructed.")
    return pipeline


# --------------------- Hyperparameter search spaces ------------------------ #

def space_linear() -> Dict:
    print("[models.linear.space_linear] Linear Regression has no hyperparameters.")
    return {}


def space_ridge() -> Dict:
    print("[models.linear.space_ridge] Returning Ridge hyperparameter space.")
    return {
        "est__alpha": [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01],
    }


def space_lasso() -> Dict:
    print("[models.linear.space_lasso] Returning Lasso hyperparameter space.")
    return {
        "est__alpha": [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    }


def space_elastic() -> Dict:
    print("[models.linear.space_elastic] Returning ElasticNet hyperparameter space.")
    return {
        "est__alpha": [1.0, 0.3, 0.1, 0.03, 0.01, 0.003],
        "est__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
