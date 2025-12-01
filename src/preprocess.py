"""
preprocess.py
-------------
GBR-only preprocessing for tabular data.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    scale_numeric: bool = False,
    sparse_ohe: bool = True,
    sparse_threshold: float = 1.0,
) -> ColumnTransformer:
    """      - one-hot encodes categorical columns
      - passes numeric columns through unchanged
    """
    num_tf = "passthrough"
    cat_tf = "drop"
    if cat_cols:
        cat_tf = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=sparse_ohe,
        )
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
        n_jobs=None,
    )
    return pre

def build_model_pipeline(
    estimator,
    num_cols: List[str],
    cat_cols: List[str],
    *,
    scale_numeric: bool = False,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    random_state: int = 42,
    sparse_ohe: bool = True,
    sparse_threshold: float = 1.0,
) -> Pipeline:
    """
    Assemble a full sklearn Pipeline for GBR:

    - Preprocessing = OneHotEncoder for categoricals + passthrough numerics.
    - No PCA stage is attached.
    - 'scale_numeric', 'use_pca', 'pca_components' are kept in the signature
      for compatibility with earlier multi-model experiments but are not used.
    Parameters
    estimator : sklearn estimator
        In this project, GradientBoostingRegressor.
    num_cols, cat_cols : list of str
        Feature column names.
    """

    pre = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=False, 
        sparse_ohe=sparse_ohe,
        sparse_threshold=sparse_threshold,
    )

    if isinstance(pre, Pipeline):
        steps = list(pre.steps) + [("est", estimator)]
        pipe = Pipeline(steps)
    else:
        pipe = Pipeline([
            ("pre", pre),
            ("est", estimator),
        ])
    return pipe

def build_pipeline_from_df(
    df: pd.DataFrame,
    estimator,
    *,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    scale_numeric: bool = False,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    random_state: int = 42,
    sparse_ohe: bool = True,
    sparse_threshold: float = 1.0,
) -> Pipeline:
    """
    Build a GBR model pipeline directly from a DataFrame by inferring feature lists.
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

    return build_model_pipeline(
        estimator=estimator,
        num_cols=numeric_cols,
        cat_cols=categorical_cols,
        scale_numeric=False,
        use_pca=False,
        pca_components=None,
        random_state=random_state,
        sparse_ohe=sparse_ohe,
        sparse_threshold=sparse_threshold,
    )
