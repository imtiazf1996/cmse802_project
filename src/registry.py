"""
registry.py
-----------
Model registry: map short names to pipeline builders and tuning spaces.

Purpose
-------
- Give the orchestrator (run_experiment.py) a simple way to ask for models
  by name and receive:
    * a pipeline builder (preprocess -> [optional PCA] -> estimator)
    * a param distribution/grid for CV search

- Keeps all model definitions in one discoverable place.

Usage
-----
    from src.registry import get_model_names, make_model, get_search_space

Design choices
--------------
- Uses build_model_pipeline() from src.preprocess to ensure uniform preprocessing.
- Param distributions are sized for RandomizedSearchCV (small-to-medium budgets).
- You can later refactor each model into src/models/*.py and import here
  without changing the external API.
"""

from __future__ import annotations
from typing import Dict, Callable, List, Optional

# --- sklearn imports ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from src.models.gb import build_gb, space_gb
# --- local ---
from src.preprocess import build_model_pipeline



# --------------- Builders (preprocess -> [PCA] -> estimator) --------------- #

def _build_linear(num_cols: List[str], cat_cols: List[str], *, use_pca=False, pca_components: Optional[int]=None):
    est = LinearRegression()
    return build_model_pipeline(
        estimator=est, num_cols=num_cols, cat_cols=cat_cols,
        scale_numeric=False, use_pca=use_pca, pca_components=pca_components
    )

def _build_ridge(num_cols, cat_cols, *, use_pca=False, pca_components=None):
    est = Ridge(random_state=42)
    return build_model_pipeline(est, num_cols, cat_cols, scale_numeric=True, use_pca=use_pca, pca_components=pca_components)

def _build_lasso(num_cols, cat_cols, *, use_pca=False, pca_components=None):
    est = Lasso(random_state=42, max_iter=2000)
    return build_model_pipeline(est, num_cols, cat_cols, scale_numeric=True, use_pca=use_pca, pca_components=pca_components)

def _build_elastic(num_cols, cat_cols, *, use_pca=False, pca_components=None):
    est = ElasticNet(random_state=42, max_iter=3000)
    return build_model_pipeline(est, num_cols, cat_cols, scale_numeric=True, use_pca=use_pca, pca_components=pca_components)

def _build_rf(num_cols, cat_cols, *, use_pca=False, pca_components=None):
    est = RandomForestRegressor(random_state=42, n_estimators=400, n_jobs=-1)
    return build_model_pipeline(est, num_cols, cat_cols, scale_numeric=False, use_pca=use_pca, pca_components=pca_components)

def _build_gbr(num_cols, cat_cols, *, use_pca=False, pca_components=None):
    est = GradientBoostingRegressor(random_state=42, n_estimators=600, learning_rate=0.05, max_depth=3)
    return build_model_pipeline(est, num_cols, cat_cols, scale_numeric=False, use_pca=use_pca, pca_components=pca_components)

def _build_svr(num_cols, cat_cols, *, use_pca=False, pca_components=None):
    # SVR needs scaling for numerics; OHE cats are fine as-is.
    est = SVR(kernel="rbf")
    return build_model_pipeline(est, num_cols, cat_cols, scale_numeric=True, use_pca=use_pca, pca_components=pca_components)

def _build_mlp(num_cols, cat_cols, *, use_pca=False, pca_components=None):
    est = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,            # L2 weight decay
        batch_size=256,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
    )
    # Scale numerics helps MLP; OHE cats are fine.
    return build_model_pipeline(est, num_cols, cat_cols, scale_numeric=True, use_pca=use_pca, pca_components=pca_components)


# --------------- Search spaces (RandomizedSearchCV-friendly) --------------- #

def _space_linear():
    # nothing to tune for plain OLS
    return {}

def _space_ridge():
    return {
        "est__alpha": [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]
    }

def _space_lasso():
    return {
        "est__alpha": [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    }

def _space_elastic():
    return {
        "est__alpha": [1.0, 0.3, 0.1, 0.03, 0.01, 0.003],
        "est__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

def _space_rf():
    return {
        "est__n_estimators": [300, 500, 800, 1000],
        "est__max_depth": [None, 6, 10, 14, 18, 24],
        "est__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "est__min_samples_leaf": [1, 2, 5],
    }

def _space_gbr():
    return {
        "est__n_estimators": [300, 600, 900, 1200],
        "est__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "est__max_depth": [2, 3, 4, 5, 6],
        "est__subsample": [0.6, 0.8, 1.0],
    }

def _space_svr():
    return {
        "est__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "est__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        "est__gamma": ["scale", 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    }

def _space_mlp():
    return {
        "est__hidden_layer_sizes": [(64,64), (128,64), (256,128)],
        "est__activation": ["relu", "tanh"],
        "est__alpha": [1e-6, 1e-5, 1e-4, 1e-3],
        "est__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
        "est__batch_size": [128, 256, 512],
        # 'solver' can be tuned too: ['adam','sgd'], but keep small at first
    }


# ----------------------------- Public API ---------------------------------- #

# Registry entries: you can comment out models you don't want to run yet
_MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    #"linear":   {"family": "linear",   "builder": _build_linear,  "space": _space_linear},
    #"ridge":    {"family": "linear",   "builder": _build_ridge,   "space": _space_ridge},
    #"lasso":    {"family": "linear",   "builder": _build_lasso,   "space": _space_lasso},
    #"elastic":  {"family": "linear",   "builder": _build_elastic, "space": _space_elastic},
    #"rf":       {"family": "tree",     "builder": _build_rf,      "space": _space_rf},
    "gbr":      {"family": "boosting", "builder": build_gb,     "space": space_gb},
    #"svr":      {"family": "kernel",   "builder": _build_svr,     "space": _space_svr},
    #"mlp":      {"family": "nn",       "builder": _build_mlp,     "space": _space_mlp},
    # Optional later:
    # "xgb":    {"family": "boosting", "builder": _build_xgb,     "space": _space_xgb},
    # "lgbm":   {"family": "boosting", "builder": _build_lgbm,    "space": _space_lgbm},
}

DEFAULT_SCORING = "neg_root_mean_squared_error"   # for RandomizedSearchCV
DEFAULT_CV_FOLDS = 5


def get_model_names() -> List[str]:
    """List available model keys."""
    return list(_MODEL_REGISTRY.keys())


def make_model(
    name: str,
    num_cols: List[str],
    cat_cols: List[str],
    *,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    """
    Build a pipeline for the given model key.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model name '{name}'. Available: {get_model_names()}")
    builder: Callable = _MODEL_REGISTRY[name]["builder"]
    print(f"[REGISTRY] Building model '{name}' (use_pca={use_pca}, pca_components={pca_components})")

    return builder(num_cols, cat_cols, use_pca=use_pca, pca_components=pca_components)


def get_search_space(name: str) -> Dict:
    """Return the param distribution/grid for the given model key."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model name '{name}'. Available: {get_model_names()}")
    print(f"[REGISTRY] Fetching search space for model '{name}'")

    return _MODEL_REGISTRY[name]["space"]()
