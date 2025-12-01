"""
registry.py
-----------
Model registry: map short names to pipeline builders and tuning spaces.
- a simple way to ask for models
  by name and receive:
    * a pipeline builder (preprocess -> estimator)
    * a param distribution/grid for CV search
Usage
-----
    from src.registry import get_model_names, make_model, get_search_space
"""

from __future__ import annotations
from typing import Dict, Callable, List, Optional
from src.models.gb import build_gb, space_gb
# from src.models.lgbm import build_lgbm, space_lgbm
# from src.models.xgb import build_xgb, space_xgb

_MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "gbr": {
        "family": "boosting",
        "builder": build_gb, 
        "space":   space_gb,   
    },
    # "lgbm": {
    #     "family": "boosting",
    #     "builder": build_lgbm,
    #     "space":   space_lgbm,
    # },
    # "xgb": {
    #     "family": "boosting",
    #     "builder": build_xgb,
    #     "space":   space_xgb,
    # },
}
# Global defaults used by run_experiment.py
DEFAULT_SCORING = "neg_root_mean_squared_error"
DEFAULT_CV_FOLDS = 5


def get_model_names() -> List[str]:
    """List available model keys (currently ['gbr'] for this project)."""
    return list(_MODEL_REGISTRY.keys())

def make_model(
    name: str,
    num_cols: List[str],
    cat_cols: List[str],
    * ,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
):
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model name '{name}'. Available: {get_model_names()}")
    builder: Callable = _MODEL_REGISTRY[name]["builder"]
    return builder(
        num_cols=num_cols,
        cat_cols=cat_cols,
        use_pca=use_pca,
        pca_components=pca_components,
    )

def get_search_space(name: str) -> Dict:
    """
    Return the param distribution/grid for the given model key.

    For GBR in the current setup, this typically returns an empty dict `{}`,
    which tells run_experiment.py to skip hyperparameter search 
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model name '{name}'. Available: {get_model_names()}")
    space_fn: Callable = _MODEL_REGISTRY[name]["space"]
    return space_fn()

#This file was completed with Chatgpt 5.1, however there were old models such as rf, svc etc, used but were removed to focus on GBR only.